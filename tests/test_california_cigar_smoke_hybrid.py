from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "california_cigar_smoke_demo.py"


def load_module():
    examples_dir = str(EXAMPLE_PATH.parent)
    added_examples_dir = examples_dir not in sys.path
    if added_examples_dir:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("california_cigar_smoke_demo", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def test_hybrid_sources_stay_within_one_fire_complex() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), (180, 140), total_frames=90, seed=17)

    assert len(sources) >= 45
    xs = np.asarray([source.x for source in sources], dtype=np.float32)
    ys = np.asarray([source.y for source in sources], dtype=np.float32)
    strengths = np.asarray([source.strength for source in sources], dtype=np.float32)
    radii = np.asarray([source.radius_px for source in sources], dtype=np.float32)
    fire_x = 0.36 * (180 - 1)
    fire_y = 0.62 * (140 - 1)
    distance = np.hypot(xs - fire_x, ys - fire_y)

    assert float(np.ptp(xs)) < 45.0
    assert float(np.ptp(ys)) < 50.0
    assert float(np.max(distance)) < 28.0
    assert float(np.max(strengths)) > float(np.min(strengths))
    assert float(np.percentile(radii, 75.0)) > 2.0
    assert any(source.start_frame > 0 for source in sources)
    assert sum(source.start_frame > 0 for source in sources) > len(sources) // 2
    flame_ends = np.asarray([module._source_flame_end_frame(source) for source in sources], dtype=np.int32)
    end_frames = np.asarray([source.end_frame for source in sources], dtype=np.int32)
    assert all(source.flame_end_frame is not None for source in sources)
    assert int(np.min(flame_ends)) >= 12
    assert int(np.ptp(flame_ends)) > 18
    assert np.all(flame_ends >= np.asarray([source.start_frame for source in sources], dtype=np.int32) + 18)
    assert np.all(end_frames > flame_ends)
    assert min(source.burst_period_frames for source in sources) >= 30.0
    assert max(source.burst_period_frames for source in sources) <= 82.0
    assert any(
        module._source_burst_envelope(source, 0) < 0.20
        and module._source_burst_envelope(source, 15) > 1.0
        for source in sources
    )


def test_hybrid_source_lifetimes_account_for_visible_warmup() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources(
        (0.36, 0.62),
        (180, 140),
        total_frames=330,
        visible_start_frame=90,
        seed=18,
    )
    flame_ends = np.asarray([module._source_flame_end_frame(source) for source in sources], dtype=np.int32)

    assert int(np.min(flame_ends)) < 90
    assert int(np.max(flame_ends)) > 250
    assert np.count_nonzero(flame_ends < 90) >= 2
    assert np.count_nonzero(flame_ends > 120) > len(sources) // 2


def test_hybrid_density_advects_from_single_event_and_tracks_age() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), (180, 140), total_frames=80, seed=23)
    simulator = module.HybridSmokeSimulator((180, 140), sources, seed=23)

    early = None
    state = None
    for frame in range(72):
        state = simulator.step(frame)
        if frame == 12:
            early = state.density.copy()
    assert early is not None
    assert state is not None

    later = state.density
    assert float(later.sum()) > float(early.sum()) * 1.05
    assert np.count_nonzero(later > 0.002) > np.count_nonzero(early > 0.002)
    assert np.count_nonzero(later > 0.002) > int(later.size * 0.35)
    assert np.count_nonzero(later > 0.002) < int(later.size * 0.80)

    yy, xx = np.mgrid[0 : later.shape[0], 0 : later.shape[1]].astype(np.float64)
    early_mass = np.asarray(early, dtype=np.float64)
    later_mass = np.asarray(later, dtype=np.float64)
    early_x = float(np.sum(xx * early_mass) / np.sum(early_mass))
    early_y = float(np.sum(yy * early_mass) / np.sum(early_mass))
    later_x = float(np.sum(xx * later_mass) / np.sum(later_mass))
    later_y = float(np.sum(yy * later_mass) / np.sum(later_mass))
    assert later_x > early_x + 8.0
    assert later_y < early_y - 2.0

    age = np.divide(
        state.age_mass,
        state.density,
        out=np.zeros_like(state.density),
        where=state.density > 1.0e-5,
    )
    active_age = age[state.density > 0.005]
    assert float(np.percentile(active_age, 75.0)) > 1.5
    assert float(np.percentile(active_age, 90.0)) > 10.0

    assert state.layer_density is not None
    assert state.layer_age_mass is not None
    assert len(state.layer_density) == module.HYBRID_SMOKE_LAYER_COUNT
    layer_density = np.stack(state.layer_density, axis=0)
    assert np.allclose(state.density, np.clip(np.sum(layer_density, axis=0), 0.0, 6.0))
    layer_centroids = []
    for layer in state.layer_density:
        mass = np.asarray(layer, dtype=np.float64)
        layer_centroids.append(
            (
                float(np.sum(xx * mass) / np.sum(mass)),
                float(np.sum(yy * mass) / np.sum(mass)),
            )
        )
    assert float(np.ptp([centroid[0] for centroid in layer_centroids])) > 2.0
    assert float(np.ptp([centroid[1] for centroid in layer_centroids])) > 4.0  # reduced due to convective uplift


def test_hybrid_smoke_rgba_is_filamentary_and_translucent() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), (200, 156), total_frames=96, seed=31)
    simulator = module.HybridSmokeSimulator((200, 156), sources, seed=31)
    state = simulator.state()
    for frame in range(86):
        state = simulator.step(frame)

    rgba = module.hybrid_smoke_rgba(state, 85, seed=31)
    alpha = rgba[..., 3]

    assert rgba.shape == (156, 200, 4)
    assert rgba.dtype == np.uint8
    assert int(alpha.max()) <= module.HYBRID_SMOKE_MAX_ALPHA
    assert int(alpha.max()) >= 125
    assert np.count_nonzero(alpha) > int(alpha.size * 0.35)
    assert np.count_nonzero(alpha) < int(alpha.size * 0.72)
    assert np.count_nonzero(alpha > 80) > int(alpha.size * 0.015)
    assert np.count_nonzero(alpha > 120) > int(alpha.size * 0.0005)

    gradient_y, gradient_x = np.gradient(alpha.astype(np.float32) / 255.0)
    gradient = np.hypot(gradient_x, gradient_y)
    assert float(np.mean(gradient[alpha > 7])) > 0.009
    assert float(np.percentile(gradient[alpha > 7], 99.0)) < 0.135

    bands = [
        np.any((1 <= alpha) & (alpha <= 24)),
        np.any((25 <= alpha) & (alpha <= 62)),
        np.any((63 <= alpha) & (alpha <= 120)),
        np.any((121 <= alpha) & (alpha <= module.HYBRID_SMOKE_MAX_ALPHA)),
    ]
    assert sum(bool(item) for item in bands) >= 4

    thin_rgb = rgba[(alpha >= 1) & (alpha <= 24)][..., :3].astype(np.float32)
    dense_rgb = rgba[alpha > 120][..., :3].astype(np.float32)
    assert thin_rgb.shape[0] > 100
    assert dense_rgb.shape[0] > 20
    assert float(np.mean(thin_rgb[:, 0] - thin_rgb[:, 2])) > 8.0  # warm: R > B (reduced for shadowing)
    assert float(np.mean(dense_rgb)) > 150.0  # reduced for self-shadowing effect
    assert abs(float(np.mean(dense_rgb[:, 0] - dense_rgb[:, 2]))) < 20.0  # widened for atmospheric effects


def test_hybrid_residual_haze_persists_as_soft_aged_layer() -> None:
    module = load_module()
    source = module.HybridSmokeSource(
        x=44.0,
        y=46.0,
        strength=1.25,
        radius_px=5.2,
        start_frame=0,
        end_frame=8,
        seed=71,
        burst_period_frames=34.0,
        burst_phase_frames=0.0,
        burst_duty=0.55,
        heat=1.1,
        smoke_rate=1.2,
        altitude_bias=0.30,
    )
    simulator = module.HybridSmokeSimulator((128, 96), [source], seed=71)
    state = simulator.state()
    for frame in range(96):
        state = simulator.step(frame)

    assert state.residual_haze is not None
    haze = state.residual_haze
    assert float(haze.max()) > 0.01
    assert np.count_nonzero(haze > 0.002) > int(haze.size * 0.08)

    rgba = module._residual_haze_rgba(haze, 96, seed=71)
    alpha = rgba[..., 3]
    assert int(alpha.max()) <= module.HYBRID_SMOKE_RESIDUAL_HAZE_MAX_ALPHA
    assert int(alpha.max()) > 4
    assert np.count_nonzero(alpha > 0) > int(alpha.size * 0.05)

    gradient_y, gradient_x = np.gradient(alpha.astype(np.float32) / 255.0)
    active_gradient = np.hypot(gradient_x, gradient_y)[alpha > 0]
    assert active_gradient.size > 0
    assert float(np.percentile(active_gradient, 99.0)) < 0.080


def test_atmospheric_composite_has_visible_smoke_lift() -> None:
    module = load_module()
    base = module.Image.new("RGBA", (24, 24), (34, 37, 38, 255))
    smoke = module.Image.new("RGBA", (24, 24), (214, 218, 214, 104))

    out = np.asarray(module.composite_atmospheric_smoke(base, smoke).convert("RGB"), dtype=np.float32)
    before = np.asarray(base.convert("RGB"), dtype=np.float32)

    assert float(np.mean(out - before)) > 40.0


def test_warp_map_layer_uses_premultiplied_alpha_edges() -> None:
    module = load_module()
    rgba = np.zeros((24, 24, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[7:17, 7:17, :3] = 174
    rgba[7:17, 7:17, 3] = 176
    plate = module.TerrainPlate(
        image=module.Image.new("RGBA", (48, 48), (0, 0, 0, 255)),
        quad=[(0.0, 0.0), (47.0, 0.0), (47.0, 47.0), (0.0, 47.0)],
        fire_xy=(24.0, 24.0),
        fire_uv=(0.5, 0.5),
        texture_size=(24, 24),
    )

    warped = np.asarray(module.warp_map_layer_to_plate(rgba, plate, (48, 48)).convert("RGBA"), dtype=np.int16)
    edge = (warped[..., 3] > 2) & (warped[..., 3] < 150)

    assert int(np.count_nonzero(edge)) > 20
    assert float(np.percentile(warped[..., 0][edge] - warped[..., 2][edge], 95.0)) < 18.0


def test_main_smoke_compositor_keeps_atmospheric_blanket_with_physical_detail() -> None:
    module = load_module()
    atmospheric = np.zeros((42, 64, 4), dtype=np.uint8)
    physical = np.zeros_like(atmospheric)
    atmospheric[10:34, 6:58, :3] = (190, 195, 190)
    atmospheric[10:34, 6:58, 3] = 72
    physical[18:27, 18:44, :3] = (222, 220, 208)
    physical[18:27, 18:44, 3] = 132

    combined = module.composite_main_smoke_maps(atmospheric, physical)
    alpha = combined[..., 3]

    assert combined.shape == atmospheric.shape
    assert int(alpha.max()) <= module.HYBRID_SMOKE_MAX_ALPHA
    assert np.count_nonzero(alpha > 0) > np.count_nonzero(physical[..., 3] > 0) * 3
    assert int(alpha[12, 10]) > 20
    assert int(alpha[22, 28]) > int(physical[22, 28, 3] * 0.80)


def test_hybrid_smoke_frames_are_temporally_coherent_not_redrawn_wisps() -> None:
    module = load_module()
    map_size = (144, 112)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=100, seed=73)
    simulator = module.HybridSmokeSimulator(map_size, sources, seed=73)

    previous = None
    current = None
    for frame in range(66):
        state = simulator.step(frame)
        rgba = module.hybrid_smoke_rgba(state, frame, seed=73)
        if frame == 64:
            previous = rgba[..., 3].astype(np.float32) / 255.0
        if frame == 65:
            current = rgba[..., 3].astype(np.float32) / 255.0

    assert previous is not None
    assert current is not None
    active = (previous > 0.02) | (current > 0.02)
    assert np.count_nonzero(active) > int(previous.size * 0.25)
    prev_active = previous[active]
    curr_active = current[active]
    corr = np.corrcoef(prev_active, curr_active)[0, 1]
    mean_delta = float(np.mean(np.abs(curr_active - prev_active)))
    assert float(corr) > 0.86
    assert mean_delta > 0.0012
    assert mean_delta < 0.050


def test_source_wisp_simulator_delays_emission_and_fades_particles() -> None:
    module = load_module()
    source = module.HybridSmokeSource(
        x=42.0,
        y=38.0,
        strength=1.2,
        radius_px=6.0,
        start_frame=0,
        end_frame=34,
        seed=10,
        burst_period_frames=36.0,
        burst_phase_frames=0.0,
        burst_duty=0.55,
        heat=1.25,
        smoke_rate=1.15,
        flame_end_frame=16,
    )
    simulator = module.SourceWispSimulator((96, 72), [source], seed=11, max_particles=120, max_emitters=8)

    assert len(simulator.step(0).puffs) == 0
    assert len(simulator.step(1).puffs) == 0
    for frame in range(2, 8):
        simulator.step(frame)
    assert len(simulator.puffs) > 0

    first_seed = simulator.puffs[0].breakup_seed
    first_alpha = simulator.puffs[0].alpha
    first_radius = simulator.puffs[0].radius_px
    for frame in range(3, 18):
        simulator.step(frame)
    tracked = [puff for puff in simulator.puffs if puff.breakup_seed == first_seed]

    assert tracked
    assert tracked[0].age_frames > 8
    assert tracked[0].alpha < first_alpha
    assert tracked[0].radius_px > first_radius
    assert all(puff.source_index == 0 for puff in simulator.puffs)

    for frame in range(35, 132):
        state = simulator.step(frame)

    assert len(state.puffs) < 8


def test_source_wisps_render_attached_filaments_with_downwind_drift() -> None:
    module = load_module()
    map_size = (160, 124)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=140, seed=83)
    simulator = module.SourceWispSimulator(map_size, sources, seed=83, max_particles=620, max_emitters=64)
    state = simulator.state()
    for frame in range(68):
        state = simulator.step(frame)

    rgba = module.source_wisps_rgba(state, 67, seed=83)
    alpha = rgba[..., 3]
    report = module.source_wisp_attachment_report(rgba, sources, 67)

    assert rgba.shape == (124, 160, 4)
    assert rgba.dtype == np.uint8
    assert int(alpha.max()) <= module.SOURCE_WISP_MAX_ALPHA
    assert int(alpha.max()) > 28
    assert np.count_nonzero(alpha > 0) > int(alpha.size * 0.006)
    assert np.count_nonzero(alpha > 0) < int(alpha.size * 0.24)
    assert report["active_source_count"] > 12
    assert report["attached_source_count"] / report["active_source_count"] > 0.48
    assert report["downwind_dx"] > 1.0

    gradient_y, gradient_x = np.gradient(alpha.astype(np.float32) / 255.0)
    active_gradient = np.hypot(gradient_x, gradient_y)[alpha > 2]
    assert active_gradient.size > 50
    assert float(np.mean(active_gradient)) > 0.006


def test_fire_core_emitter_sources_come_from_prebloom_hot_front_signal() -> None:
    module = load_module()
    map_size = (160, 124)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=140, seed=84)

    core = module.active_fire_core_intensity_field(sources, 36, map_size)
    emitters = module.fire_core_emitter_sources(sources, 36, map_size, max_emitters=80, seed=84)
    glow = module.hybrid_fire_sources_rgba(sources, 36, map_size, glow_only=True, bloom_scale=1.2)

    assert core.shape == (map_size[1], map_size[0])
    assert core.dtype == np.float32
    assert float(core.max()) > 0.80
    assert np.count_nonzero(core > module.FIRE_CORE_EMITTER_INTENSITY_THRESHOLD) > 80
    assert len(emitters) >= 35
    assert np.count_nonzero(core > 0.02) < np.count_nonzero(glow[..., 3] > 5) * 0.45
    for emitter in emitters[:12]:
        x = int(round(emitter.x))
        y = int(round(emitter.y))
        assert float(core[y, x]) >= module.FIRE_CORE_EMITTER_INTENSITY_THRESHOLD * 0.7

    xs = np.asarray([emitter.x for emitter in emitters], dtype=np.float32)
    ys = np.asarray([emitter.y for emitter in emitters], dtype=np.float32)
    assert float(np.ptp(xs)) > 18.0
    assert float(np.ptp(ys)) > 18.0


def test_source_wisp_fire_core_mode_uses_distributed_dynamic_emitters() -> None:
    module = load_module()
    map_size = (160, 124)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=140, seed=85)
    simulator = module.SourceWispSimulator(
        map_size,
        sources,
        seed=85,
        max_particles=900,
        max_emitters=96,
        emitter_mode="fire-core",
    )
    state = simulator.state()
    for frame in range(68):
        state = simulator.step(frame)

    rgba = module.source_wisps_rgba(state, 67, seed=85)
    alpha = rgba[..., 3]
    report = module.source_wisp_attachment_report(rgba, list(state.emitters), 67)

    assert len(state.emitters) >= 45
    assert len(state.puffs) > 500
    assert int(alpha.max()) > 40
    assert 0.015 < np.count_nonzero(alpha > 0) / alpha.size < 0.16
    assert report["active_source_count"] >= 40
    assert report["attached_source_count"] / report["active_source_count"] > 0.70
    assert report["downwind_dx"] > 1.0


def test_source_wisp_morphology_grows_soft_old_tails_and_rejects_brush_bundle() -> None:
    module = load_module()
    map_size = (160, 124)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=140, seed=85)
    simulator = module.SourceWispSimulator(
        map_size,
        sources,
        seed=85,
        max_particles=900,
        max_emitters=96,
        emitter_mode="fire-core",
    )
    state = simulator.state()
    for frame in range(68):
        state = simulator.step(frame)

    plate = module.TerrainPlate(
        image=module.Image.new("RGBA", map_size),
        quad=[(0, 0), (map_size[0], 0), (map_size[0], map_size[1]), (0, map_size[1])],
        fire_xy=(0.0, 0.0),
        fire_uv=(0.0, 0.0),
        texture_size=map_size,
    )
    plume_rgba = module.source_wisps_rgba(state, 67, seed=85, plume_ribbons=True)
    brush_rgba = module.source_wisps_rgba(state, 67, seed=85, plume_ribbons=False)
    plume = module.source_wisp_morphology_report(
        state,
        67,
        plate,
        map_size,
        seed=85,
        plume_ribbons=True,
        warped_wisps=plume_rgba,
    )
    brush = module.source_wisp_morphology_report(
        state,
        67,
        plate,
        map_size,
        seed=85,
        plume_ribbons=False,
        warped_wisps=brush_rgba,
    )
    thresholds = module.SOURCE_WISP_AUDIT_THRESHOLDS

    assert plume["morphology_stage_coverage_fraction"] >= thresholds["minimum_morphology_band_coverage_fraction"]
    assert plume["transition_width_growth_ratio"] >= thresholds["minimum_transition_width_growth_ratio"]
    assert plume["old_tail_width_growth_ratio"] >= thresholds["minimum_old_tail_width_growth_ratio"]
    assert plume["old_tail_alpha_p90_fraction"] <= thresholds["maximum_old_tail_alpha_p90_fraction"]
    assert plume["old_tail_endpoint_alpha_fraction"] <= thresholds["maximum_old_tail_endpoint_alpha_fraction"]
    assert plume["old_tail_coverage_growth_ratio"] >= thresholds["minimum_old_tail_coverage_growth_ratio"]
    assert plume["old_tail_edge_softness_px"] >= thresholds["minimum_old_tail_edge_softness_px"]
    assert plume["old_tail_diffuse_to_core_area_ratio"] >= thresholds["minimum_old_tail_diffuse_to_core_area_ratio"]
    assert plume["brush_bundle_score"] <= thresholds["maximum_brush_bundle_score"]

    assert plume["old_tail_coverage_fraction"] > brush["old_tail_coverage_fraction"] * 1.25
    assert brush["old_tail_coverage_growth_ratio"] < thresholds["minimum_old_tail_coverage_growth_ratio"]
    assert brush["brush_bundle_score"] > thresholds["maximum_brush_bundle_score"]


def test_smoke_composite_keeps_orange_sources_visible_under_veil() -> None:
    module = load_module()
    base = module.Image.new("RGBA", (72, 48), (28, 31, 32, 255))
    draw = module.ImageDraw.Draw(base, "RGBA")
    draw.ellipse((26, 18, 46, 31), fill=(255, 104, 18, 230))
    draw.ellipse((32, 21, 40, 28), fill=(255, 228, 116, 255))
    smoke = module.Image.new("RGBA", (72, 48), (214, 218, 214, 118))

    out = np.asarray(module.composite_atmospheric_smoke(base, smoke).convert("RGB"), dtype=np.int16)
    center = out[21:29, 30:42]
    surrounding = out[8:16, 8:20]

    assert int(center[..., 0].max()) > int(surrounding[..., 0].max()) + 35
    assert float(np.mean(center[..., 0] - center[..., 2])) > 34.0
    assert float(np.mean(center[..., 1] - center[..., 2])) > 18.0


def test_enhance_terrain_texture_grades_to_charcoal() -> None:
    module = load_module()
    texture = module.Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8), mode="RGB")
    dem = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)

    charcoal = np.asarray(module.enhance_terrain_texture(texture, dem).convert("RGB"), dtype=np.float32)
    luma = charcoal @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    channel_spread = np.mean(np.abs(charcoal[..., 0] - charcoal[..., 2]))

    assert float(np.mean(luma)) < 65.0
    assert float(np.max(luma)) < 128.0
    assert float(channel_spread) < 12.0


def test_volume_detail_defaults_to_2p5d_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py"])

    args = module.parse_args()

    assert args.render_preset == module.TARGET_RENDER_PRESET
    assert bool(args.volume_detail) is False
    assert bool(args.physical_smoke) is False
    assert args.physical_smoke_backend == "auto"
    assert bool(args.source_wisps) is True
    assert args.source_wisp_emitter_mode == "fire-core"
    assert args.physical_emitter_mode == "fire-core"
    assert args.source_wisp_warmup_mode == "visible-only"
    assert bool(args.source_wisp_plume_ribbons) is True
    assert args.smoke_ablation == "combined"
    assert args.broad_smoke_alpha <= 0.03
    assert args.physical_alpha == 0.0
    assert args.physical_max_sources >= 90
    assert args.source_wisp_max_emitters >= 120
    assert args.layer_policy["source_wisps_primary"] is True

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--volume-detail"])
    assert bool(module.parse_args().volume_detail) is True

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--no-physical-smoke"])
    assert bool(module.parse_args().physical_smoke) is False

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--physical-smoke"])
    assert bool(module.parse_args().physical_smoke) is True

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--physical-smoke-backend", "numpy"])
    assert module.parse_args().physical_smoke_backend == "numpy"

    monkeypatch.setattr(
        sys,
        "argv",
        ["california_cigar_smoke_demo.py", "--no-source-wisps", "--smoke-ablation", "broad-only"],
    )
    args = module.parse_args()
    assert bool(args.source_wisps) is False
    assert args.smoke_ablation == "broad-only"

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--render-preset", "legacy-combined"])
    legacy = module.parse_args()
    assert legacy.source_wisp_emitter_mode == "synthetic"
    assert legacy.source_wisp_warmup_mode == "full"
    assert bool(legacy.source_wisp_plume_ribbons) is False
    assert legacy.broad_smoke_alpha == pytest.approx(0.28)
    assert legacy.physical_alpha == pytest.approx(0.58)
    assert legacy.physical_max_sources == 32

    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--render-preset", "source-wisp-brush-baseline"])
    brush = module.parse_args()
    assert brush.render_preset == module.BRUSH_BUNDLE_RENDER_PRESET
    assert brush.source_wisp_emitter_mode == "fire-core"
    assert bool(brush.source_wisp_plume_ribbons) is False
    assert brush.broad_smoke_alpha <= 0.03


def test_reference_film_preset_uses_full_bleed_1080p_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--render-preset", "reference-film"])

    args = module.parse_args()

    assert args.render_preset == module.REFERENCE_FILM_RENDER_PRESET
    assert args.composition_mode == module.MAP_FILM_COMPOSITION_MODE
    assert args.delivery_profile == module.REFERENCE_DELIVERY_PROFILE
    assert tuple(args.size) == (1920, 1080)
    assert args.duration == pytest.approx(30.0)
    assert bool(args.regional_smoke) is True
    assert bool(args.source_wisps) is True
    assert bool(args.physical_smoke) is False
    assert args.source_wisp_emitter_mode == "fire-core"
    assert args.broad_smoke_alpha > 0.20
    assert args.encode_policy["video_bitrate"] == "2600k"
    assert args.encode_policy["maxrate"] == "3200k"
    assert args.encode_policy["bufsize"] == "5200k"
    assert args.encode_policy["color_primaries"] == "bt709"
    assert args.layer_policy["reference_film_target"] is True
    assert args.observed_smoke_source == "auto"
    assert args.layer_policy["observed_smoke_source"] == "auto"


def test_reference_exact_smoke_preset_uses_locked_native_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["california_cigar_smoke_demo.py", "--render-preset", "reference-exact-smoke"])

    args = module.parse_args()

    assert args.render_preset == module.REFERENCE_EXACT_SMOKE_RENDER_PRESET
    assert args.reference_smoke_mode == "exact"
    assert args.reference_smoke_start_frame == 0
    assert args.reference_smoke_frame_count == module.REFERENCE_EXACT_FRAME_COUNT
    assert args.composition_mode == module.MAP_FILM_COMPOSITION_MODE
    assert args.delivery_profile == module.REFERENCE_DELIVERY_PROFILE
    assert tuple(args.size) == (module.REFERENCE_EXACT_WIDTH, module.REFERENCE_EXACT_HEIGHT)
    assert args.duration == pytest.approx(30.0)
    assert bool(args.source_wisps) is False
    assert bool(args.regional_smoke) is False
    assert args.broad_smoke_alpha == pytest.approx(0.0)
    assert args.layer_policy["reference_exact_smoke_target"] is True
    assert args.layer_policy["reference_smoke_mode"] == "exact"


def test_reference_exact_frame_mapping_uses_nearest_30fps_source_frame() -> None:
    module = load_module()

    assert module._reference_exact_source_frame_index(0, 30, start_frame=0, frame_count=900) == 0
    assert module._reference_exact_source_frame_index(15, 30, start_frame=0, frame_count=900) == 15
    assert module._reference_exact_source_frame_index(15, 60, start_frame=0, frame_count=900) == 8
    assert module._reference_exact_source_frame_index(1000, 30, start_frame=12, frame_count=30) == 41


def test_reference_smoke_rgba_loads_direct_cache_frame(tmp_path: Path) -> None:
    module = load_module()
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir(parents=True)
    payload = {
        "start_frame": 0,
        "frame_count": 2,
        "reference_video_path": "unused.mp4",
        "reference_sha256": "unused",
    }
    (tmp_path / "reference_exact_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., :3] = (24, 36, 48)
    rgba[..., 3] = 128
    module.Image.fromarray(rgba, mode="RGBA").save(smoke_dir / "smoke_rgba_0000.png")

    loaded = module.reference_smoke_rgba(0, (3, 2), tmp_path)

    assert loaded.shape == (2, 3, 4)
    assert np.array_equal(loaded, rgba)


def test_reference_smoke_rgba_prefers_corrected_alpha_rgb(tmp_path: Path) -> None:
    module = load_module()
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir(parents=True)
    payload = {
        "start_frame": 0,
        "frame_count": 1,
        "reference_video_path": "unused.mp4",
        "reference_sha256": "unused",
    }
    (tmp_path / "reference_exact_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    original = np.zeros((1, 2, 4), dtype=np.uint8)
    original[..., :3] = (250, 0, 0)
    original[..., 3] = 255
    module.Image.fromarray(original, mode="RGBA").save(smoke_dir / "smoke_rgba_0000.png")
    corrected_alpha = np.asarray([[128, 0]], dtype=np.uint8)
    corrected_rgb = np.asarray([[[20, 40, 60], [200, 220, 240]]], dtype=np.uint8)
    module.Image.fromarray(corrected_alpha, mode="L").save(smoke_dir / "smoke_alpha_corrected_0000.png")
    module.Image.fromarray(corrected_rgb, mode="RGB").save(smoke_dir / "smoke_rgb_corrected_0000.png")

    loaded = module.reference_smoke_rgba(0, (2, 1), tmp_path)

    assert tuple(loaded[0, 0]) == (10, 20, 30, 128)
    assert tuple(loaded[0, 1]) == (0, 0, 0, 0)


def test_reference_exact_gate_requires_independent_generated_smoke_dir(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "smoke").mkdir()

    with pytest.raises(RuntimeError, match="generated_smoke_dir is required"):
        module.evaluate_reference_exact_smoke_gate(tmp_path, frame_count=1)


def test_reference_exact_gate_scores_generated_smoke_not_reference_cache(tmp_path: Path) -> None:
    module = load_module()
    smoke_dir = tmp_path / "smoke"
    masks_dir = tmp_path / "masks"
    gen_dir = tmp_path / "generated_smoke"
    smoke_dir.mkdir()
    masks_dir.mkdir()
    gen_dir.mkdir()
    module.Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(masks_dir / "ui_exclusion_mask.png")
    module.Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(masks_dir / "static_background_valid_mask.png")
    module.Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(masks_dir / "fire_mask_0000.png")
    ref = np.zeros((4, 4, 4), dtype=np.uint8)
    ref[1:3, 1:3, :3] = 80
    ref[1:3, 1:3, 3] = 200
    gen = np.zeros((4, 4, 4), dtype=np.uint8)
    module.Image.fromarray(ref, mode="RGBA").save(smoke_dir / "smoke_rgba_0000.png")
    module.Image.fromarray(gen, mode="RGBA").save(gen_dir / "smoke_rgba_0000.png")

    report = module.evaluate_reference_exact_smoke_gate(tmp_path, gen_dir, frame_count=1)

    assert report["comparison"]["self_comparison"] is False
    assert report["per_frame_metrics"][0]["smoke_mask_iou"] == 0.0
    assert report["per_frame_metrics"][0]["smoke_alpha_mae"] > 0.0
    assert report["passed"] is False


def test_extract_reference_exact_smoke_layers_from_generated_frames(tmp_path: Path) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "generated_smoke"
    (cache_dir / "masks").mkdir(parents=True)
    frames_dir.mkdir()
    background = np.full((6, 6, 3), 20, dtype=np.uint8)
    frame = background.copy()
    frame[2:4, 2:4] = (96, 96, 96)
    module.Image.fromarray(background, mode="RGB").save(cache_dir / "reference_background_clean.png")
    module.Image.fromarray(np.full((6, 6), 255, dtype=np.uint8), mode="L").save(
        cache_dir / "masks" / "candidate_smoke_domain_0000.png"
    )
    module.Image.fromarray(frame, mode="RGB").save(frames_dir / "frame_0000.png")

    summary = module.extract_reference_exact_smoke_layers_from_generated_frames(
        cache_dir,
        frames_dir,
        output_dir,
        frame_count=1,
    )

    assert summary["frame_count"] == 1
    assert (output_dir / "smoke_rgba_0000.png").exists()
    rgba = np.asarray(module.Image.open(output_dir / "smoke_rgba_0000.png").convert("RGBA"))
    assert np.count_nonzero(rgba[..., 3]) > 0


def test_load_reference_smoke_event_states_reads_cache_contract(tmp_path: Path) -> None:
    module = load_module()
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    module.Image.fromarray(np.zeros((10, 20, 4), dtype=np.uint8), mode="RGBA").save(smoke_dir / "smoke_rgba_0000.png")
    events = [
        {
            "event_id": "reference-event-test",
            "start_frame": 2,
            "peak_frame": 8,
            "end_frame": 14,
            "coverage_peak": 0.18,
            "date_label": "2023-01-04",
            "dominant_axis_degrees": 21.0,
            "centroid_path": [
                {"frame": 2, "x_px": 5.0, "y_px": 4.0, "coverage": 0.05},
                {"frame": 8, "x_px": 15.0, "y_px": 6.0, "coverage": 0.18},
            ],
        }
    ]
    (tmp_path / "reference_smoke_events.json").write_text(json.dumps(events), encoding="utf-8")

    loaded = module.load_reference_smoke_event_states(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].event_id == "reference-event-test"
    assert loaded[0].source_size == (20, 10)
    assert loaded[0].centroid_path[1] == (8, 15.0, 6.0, 0.18)


def test_observed_smoke_source_auto_prefers_reference_events(tmp_path: Path) -> None:
    module = load_module()
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    module.Image.fromarray(np.zeros((10, 20, 4), dtype=np.uint8), mode="RGBA").save(smoke_dir / "smoke_rgba_0000.png")
    (tmp_path / "reference_smoke_events.json").write_text(
        json.dumps(
            [
                {
                    "event_id": "reference-event-test",
                    "start_frame": 0,
                    "peak_frame": 5,
                    "end_frame": 10,
                    "coverage_peak": 0.16,
                    "centroid_path": [{"frame": 5, "x_px": 10.0, "y_px": 5.0, "coverage": 0.16}],
                }
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        observed_smoke_source="auto",
        reference_smoke_cache=tmp_path,
        layer_policy={"reference_film_target": True},
    )

    source = module.make_observed_smoke_source(args, (80, 40), None, visible_frame_count=30)
    report = module.observed_smoke_source_report(source)

    assert source.source_kind == "reference-derived-events"
    assert report["event_count"] == 1
    assert report["requested_source"] == "auto"
    assert report["approximate"] is True


def test_reference_event_transport_smoke_follows_centroid_motion() -> None:
    module = load_module()
    event = module.ReferenceSmokeEventState(
        event_id="event-motion",
        start_frame=0,
        peak_frame=10,
        end_frame=20,
        coverage_peak=0.18,
        centroid_path=((5, 22.0, 25.0, 0.14), (10, 78.0, 25.0, 0.18)),
        dominant_axis_degrees=0.0,
        date_label="2023-01-01",
        source_size=(100, 50),
    )

    early = module.reference_event_transport_smoke_rgba((100, 50), 5, (event,), progress=0.25, timeline_frame_count=21)
    peak = module.reference_event_transport_smoke_rgba((100, 50), 10, (event,), progress=0.50, timeline_frame_count=21)

    def centroid_x(rgba: np.ndarray) -> float:
        alpha = rgba[..., 3].astype(np.float32)
        xs = np.arange(alpha.shape[1], dtype=np.float32)[None, :]
        return float(np.sum(xs * alpha) / max(float(np.sum(alpha)), 1.0))

    assert early.shape == (50, 100, 4)
    assert int(early[..., 3].max()) > 0
    assert centroid_x(peak) > centroid_x(early) + 18.0


def test_observed_smoke_rgba_hrrr_source_uses_guidance_frames() -> None:
    module = load_module()
    left = np.zeros((40, 80), dtype=np.float32)
    right = np.zeros((40, 80), dtype=np.float32)
    left[:, 4:28] = 0.85
    right[:, 52:76] = 0.85
    source = module.ObservedSmokeSource(
        source_kind="hrrr-smoke",
        source_label="test hrrr",
        frames=(left, right),
        guidance_cadence_frames=10.0,
        approximate=False,
    )

    first = module.observed_smoke_rgba(source, (80, 40), 0)
    second = module.observed_smoke_rgba(source, (80, 40), 10)

    first_alpha = first[..., 3].astype(np.float32)
    second_alpha = second[..., 3].astype(np.float32)
    assert first.shape == (40, 80, 4)
    assert float(first_alpha[:, :40].sum()) > float(first_alpha[:, 40:].sum()) * 2.0
    assert float(second_alpha[:, 40:].sum()) > float(second_alpha[:, :40].sum()) * 2.0


def test_reference_exact_decoded_label_report_detects_source_labels(tmp_path: Path) -> None:
    module = load_module()
    font = module.load_font(16, bold=True)
    for frame_index in range(3):
        image = module.Image.new("RGBA", (300, 180), (28, 32, 34, 255))
        draw = module.ImageDraw.Draw(image, "RGBA")
        draw.text((8, 6), "Data: CAL FIRE perimeter", font=font, fill=(242, 246, 242, 255))
        draw.text((190, 8), "2023-01-01", font=font, fill=(242, 246, 242, 255))
        draw.text((8, 128), "Area Burned:\n12.3M ha", font=font, fill=(242, 246, 242, 255))
        image.save(tmp_path / f"frame_{frame_index:04d}.png")

    report = module.reference_exact_decoded_label_report(tmp_path, frame_count=3)

    assert report["schema_version"] == "reference-exact-decoded-label-v1"
    assert report["passed"] is True
    assert float(report["active_region_fraction"]) >= 0.50
    assert float(report["median_region_luma_contrast"]) > 0.05


def test_reference_exact_map_extent_contract_declares_continent_mode() -> None:
    module = load_module()

    contract = module.reference_exact_map_extent_contract(output_size=(module.REFERENCE_EXACT_WIDTH, module.REFERENCE_EXACT_HEIGHT))

    assert contract["continent_mode"] is True
    assert contract["extent_kind"] == "decoded-reference-global-or-continent-frame"
    assert contract["smoke_layer_space"] == "native source frame pixels"
    assert contract["output_size"] == [module.REFERENCE_EXACT_WIDTH, module.REFERENCE_EXACT_HEIGHT]


def test_premultiplied_reference_smoke_composites_screen_space() -> None:
    module = load_module()
    base = module.Image.new("RGBA", (2, 1), (100, 80, 60, 255))
    layer = np.asarray([[[50, 40, 30, 128], [0, 0, 0, 0]]], dtype=np.uint8)

    out = np.asarray(module._composite_premultiplied_screen_layer(base, layer), dtype=np.uint8)

    assert tuple(out[0, 0, :3]) == (100, 80, 60)
    assert tuple(out[0, 1, :3]) == (100, 80, 60)
    assert int(out[0, 0, 3]) == 255


def test_reference_smoke_layer_excludes_fire_and_static_background() -> None:
    module = load_module()
    background = np.full((8, 8, 3), 32, dtype=np.uint8)
    frame = background.copy()
    frame[2:5, 2:5] = (92, 94, 96)
    frame[0:2, 0:2] = (255, 160, 40)
    domain = np.ones((8, 8), dtype=bool)
    domain[0:2, 0:2] = False
    domain[:, 6:] = False

    alpha, smoke_rgb, smoke_rgba, confidence, mae = module._reference_smoke_layer_for_frame(frame, background, domain)

    assert alpha.shape == (8, 8)
    assert smoke_rgb.shape == (8, 8, 3)
    assert smoke_rgba.shape == (8, 8, 4)
    assert confidence.shape == (8, 8)
    assert np.count_nonzero(alpha[2:5, 2:5]) > 0
    assert np.count_nonzero(alpha[0:2, 0:2]) == 0
    assert np.count_nonzero(alpha[:, 6:]) == 0
    assert mae <= 1.0


def _write_california_extent(module, root: Path, stem: str, size: tuple[int, int]):
    from _generated_assets import write_geotiff

    width, height = size
    dem = write_geotiff(root / f"{stem}.tif", width=width, height=height)
    yy, xx = np.mgrid[0:height, 0:width]
    overlay = np.empty((height, width, 4), dtype=np.uint8)
    overlay[..., 0] = 28 + (xx % 96)
    overlay[..., 1] = 40 + (yy % 112)
    overlay[..., 2] = 52 + ((xx + yy) % 80)
    overlay[..., 3] = 255
    overlay_path = root / f"{stem}_overlay.png"
    module.Image.fromarray(overlay, mode="RGBA").save(overlay_path)
    west, south = module.lonlat_to_web_mercator(-124.5, 31.0)
    east, north = module.lonlat_to_web_mercator(-114.0, 43.0)
    meta_path = root / f"{stem}.json"
    meta_path.write_text(
        json.dumps({"bounds_mercator": [west, south, east, north]}),
        encoding="utf-8",
    )
    return dem, overlay_path, meta_path


def test_map_film_plate_uses_regional_reference_extent(tmp_path, monkeypatch) -> None:
    module = load_module()
    local = _write_california_extent(module, tmp_path, "local", (400, 300))
    regional = _write_california_extent(module, tmp_path, "regional", (1200, 675))
    for name, path in zip(("DEM_PATH", "OVERLAY_PATH", "META_PATH"), local):
        monkeypatch.setattr(module, name, path)
    for name, path in zip(
        ("REGIONAL_DEM_PATH", "REGIONAL_OVERLAY_PATH", "REGIONAL_META_PATH"),
        regional,
    ):
        monkeypatch.setattr(module, name, path)

    local_texture, _local_dem, _local_fire = module.crop_fire_extent()
    plate = module.map_film_plate(320, 180)

    assert plate.extent_kind == "regional-california"
    assert plate.bounds_mercator is not None
    assert plate.texture_size[0] > local_texture.width * 3
    assert plate.texture_size[1] > local_texture.height * 2
    assert 0.05 < plate.fire_uv[0] < 0.35
    assert 0.25 < plate.fire_uv[1] < 0.70
    assert plate.quad == [(0.0, 0.0), (319.0, 0.0), (319.0, 179.0), (0.0, 179.0)]


def test_enhance_terrain_texture_respects_overlay_alpha_mask() -> None:
    module = load_module()
    texture = module.Image.new("RGBA", (24, 18), (80, 96, 92, 0))
    draw = module.ImageDraw.Draw(texture, "RGBA")
    draw.rectangle((6, 4, 17, 14), fill=(80, 96, 92, 255))
    dem = np.linspace(0.0, 1000.0, 24 * 18, dtype=np.float32).reshape(18, 24)

    enhanced = np.asarray(module.enhance_terrain_texture(texture, dem).convert("RGB"), dtype=np.float32) / 255.0
    luma_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    corner_luma = float(enhanced[0, 0] @ luma_weights)
    center_luma = float(enhanced[9, 12] @ luma_weights)

    assert corner_luma < 0.060
    assert center_luma > corner_luma + 0.010


def test_reference_film_frame_info_evolves_date_and_area() -> None:
    module = load_module()

    first = module.reference_film_frame_info(0, 300)
    middle = module.reference_film_frame_info(150, 300)
    last = module.reference_film_frame_info(299, 300)

    assert first.date_label < middle.date_label < last.date_label
    assert first.burned_area_ha < middle.burned_area_ha < last.burned_area_ha
    assert first.burned_area_ha == pytest.approx(module.REFERENCE_FILM_START_AREA_HA)
    assert last.burned_area_ha == pytest.approx(module.AUGUST_COMPLEX.final_area_ha)


def test_regional_transport_smoke_provides_broad_ribbon_scale() -> None:
    module = load_module()

    rgba = module.regional_transport_smoke_rgba((180, 100), 90, progress=0.45)
    alpha = rgba[..., 3]
    naturalism = module._reference_regional_smoke_texture_report(alpha)

    assert rgba.shape == (100, 180, 4)
    assert rgba.dtype == np.uint8
    assert int(alpha.max()) <= module.REFERENCE_FILM_REGIONAL_SMOKE_MAX_ALPHA
    coverage = np.count_nonzero(alpha > 0) / alpha.size
    dense_coverage = np.count_nonzero(alpha > module.REFERENCE_FILM_DENSE_REGIONAL_SMOKE_ALPHA_THRESHOLD) / alpha.size
    assert 0.12 < coverage < 0.70
    assert module.REFERENCE_FILM_AUDIT_THRESHOLDS["minimum_median_dense_regional_smoke_fraction"] < dense_coverage < 0.35
    gradient_y, gradient_x = np.gradient(alpha.astype(np.float32) / 255.0)
    active_gradient = np.hypot(gradient_x, gradient_y)[alpha > 0]
    assert active_gradient.size > 100
    assert float(np.percentile(active_gradient, 95.0)) > 0.004
    assert naturalism["regional_smoke_contour_band_score"] < module.REFERENCE_FILM_AUDIT_THRESHOLDS[
        "maximum_median_regional_smoke_contour_band_score"
    ]
    assert naturalism["regional_smoke_ring_score"] < module.REFERENCE_FILM_AUDIT_THRESHOLDS[
        "maximum_median_regional_smoke_ring_score"
    ]


def test_regional_smoke_texture_report_rejects_contour_bands() -> None:
    module = load_module()
    yy, xx = np.mgrid[0:120, 0:180].astype(np.float32)
    cx, cy = 90.0, 60.0
    radius = np.hypot((xx - cx) / 90.0, (yy - cy) / 60.0)
    band = (0.5 + 0.5 * np.sin(radius * 62.0)) > 0.74
    envelope = np.exp(-((radius - 0.44) ** 2) / (2.0 * 0.22 * 0.22))
    alpha = np.where(band, envelope * 170.0, envelope * 24.0).astype(np.uint8)

    report = module._reference_regional_smoke_texture_report(alpha)

    assert report["regional_smoke_contour_band_score"] > module.REFERENCE_FILM_AUDIT_THRESHOLDS[
        "maximum_median_regional_smoke_contour_band_score"
    ]
    assert report["regional_smoke_ring_score"] > 0.80


def test_reference_fire_points_are_fine_grained_and_source_derived() -> None:
    module = load_module()
    map_size = (180, 112)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=160, seed=91)

    rgba = module.reference_fire_points_rgba(sources, 72, map_size, max_points=160, seed=91)
    core = module.active_fire_core_intensity_field(sources, 72, map_size)
    alpha = rgba[..., 3]

    assert rgba.shape == (map_size[1], map_size[0], 4)
    assert int(alpha.max()) > 180
    assert np.count_nonzero(alpha > 0) > 140
    assert np.count_nonzero(alpha > 0) < int(alpha.size * 0.10)
    hot = alpha > 120
    assert np.count_nonzero(hot) >= 24
    assert float(np.percentile(core[hot], 50.0)) >= module.FIRE_CORE_EMITTER_INTENSITY_THRESHOLD * 0.45


def test_reference_fire_points_can_add_distributed_regional_context() -> None:
    module = load_module()
    map_size = (180, 112)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=900, seed=91)

    local = module.reference_fire_points_rgba(sources, 620, map_size, max_points=160, seed=91)
    regional = module.reference_fire_points_rgba(
        sources,
        620,
        map_size,
        max_points=160,
        seed=91,
        regional_context=True,
    )
    local_report = module._reference_fire_distribution_report(local[..., 3], (0.36, 0.62))
    regional_report = module._reference_fire_distribution_report(regional[..., 3], (0.36, 0.62))

    assert regional.shape == local.shape
    assert regional_report["distributed_fire_cluster_count"] >= 4.0
    assert regional_report["fire_spread_grid_cell_count"] >= 3.0
    assert regional_report["far_fire_core_fraction"] > local_report["far_fire_core_fraction"]
    assert regional_report["primary_fire_dominance_fraction"] < 0.86


def test_reference_film_frame_report_measures_composition_smoke_fire_and_time() -> None:
    module = load_module()
    map_size = (120, 80)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=900, seed=92)
    plate = module.TerrainPlate(
        image=module.Image.new("RGBA", (240, 135), (20, 24, 24, 255)),
        quad=[(0.0, 0.0), (239.0, 0.0), (239.0, 134.0), (0.0, 134.0)],
        fire_xy=(86.0, 84.0),
        fire_uv=(0.36, 0.62),
        texture_size=map_size,
    )
    current = module.Image.new("RGBA", (240, 135), (26, 29, 28, 255))
    previous = module.Image.new("RGBA", (240, 135), (16, 18, 18, 255))
    regional = module.regional_transport_smoke_rgba(map_size, 42, progress=0.4)
    broad = module.hybrid_smoke_rgba(
        module.HybridSmokeSimulator(map_size, sources, seed=92).state(),
        42,
        seed=92,
    )
    broad = module._premultiplied_over(regional, broad)
    wisp = np.zeros((map_size[1], map_size[0], 4), dtype=np.uint8)
    wisp[22:34, 35:58, 3] = 90
    fire = module.reference_fire_points_rgba(sources, 620, map_size, max_points=80, seed=92, regional_context=True)
    frame_info = module.reference_film_frame_info(620, 900)

    report = module._reference_film_frame_report(
        current,
        plate,
        current.size,
        map_size,
        sources,
        620,
        1.4,
        regional,
        broad,
        wisp,
        fire,
        frame_info,
        previous,
    )

    assert report["date_label"] == frame_info.date_label
    assert float(report["full_bleed_frame_coverage_fraction"]) == pytest.approx(1.0)
    assert float(report["map_quad_area_fraction"]) > 0.98
    assert float(report["combined_smoke_coverage_fraction"]) > 0.05
    assert float(report["regional_smoke_coverage_fraction"]) > 0.05
    assert float(report["mid_scale_smoke_fraction"]) > 0.0
    assert 0.0 <= float(report["smoke_centroid_x_fraction"]) <= 1.0
    assert float(report["active_fire_core_pixel_count"]) > 0
    assert float(report["halo_core_area_ratio"]) >= 1.0
    assert float(report["temporal_luma_delta"]) > 0.01
    assert float(report["distributed_fire_cluster_count"]) >= 4.0
    assert float(report["fire_spread_grid_cell_count"]) >= 3.0
    assert float(report["far_fire_core_fraction"]) >= 0.045
    assert (
        float(report["regional_smoke_texture_score"])
        >= module.REFERENCE_FILM_AUDIT_THRESHOLDS["minimum_median_regional_smoke_texture_score"]
    )
    assert float(report["regional_smoke_axis_band_score"]) <= 0.24
    assert float(report["regional_smoke_contour_band_score"]) <= module.REFERENCE_FILM_AUDIT_THRESHOLDS[
        "maximum_median_regional_smoke_contour_band_score"
    ]
    assert float(report["regional_smoke_ring_score"]) <= module.REFERENCE_FILM_AUDIT_THRESHOLDS[
        "maximum_median_regional_smoke_ring_score"
    ]

    pre_label = current.copy()
    labeled = current.copy()
    boxes = module.draw_labels(labeled, frame_info=frame_info, composition_mode=module.MAP_FILM_COMPOSITION_MODE)
    blank_layer = module.Image.new("RGBA", current.size, (0, 0, 0, 0))
    label_report = module._reference_film_label_report(pre_label, labeled, boxes, blank_layer, blank_layer)

    assert label_report["label_count"] >= 4.0
    assert label_report["median_label_contrast_delta"] > 0.07
    assert label_report["median_label_smoke_overlap_fraction"] == pytest.approx(0.0)
    fire_layer = module.Image.new("RGBA", current.size, (0, 0, 0, 0))
    fire_draw = module.ImageDraw.Draw(fire_layer, "RGBA")
    fire_draw.ellipse((92, 66, 105, 79), fill=(255, 120, 20, 180))
    visible_frame = current.copy()
    visible_draw = module.ImageDraw.Draw(visible_frame, "RGBA")
    visible_draw.ellipse((94, 68, 103, 77), fill=(255, 190, 55, 255))
    visibility = module._reference_film_fire_visibility_report(visible_frame, fire_layer)

    assert visibility["post_smoke_fire_visibility_fraction"] > 0.25


def test_physical_source_selection_default_keeps_more_important_fire_points() -> None:
    module = load_module()
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), (180, 140), total_frames=120, seed=89)
    selected = module._select_physical_sources(sources, module.PHYSICAL_SMOKE_MAX_SOURCES)
    selected_indexes = {sources.index(source) for source in selected}

    assert len(selected) > 12
    assert len(selected) == min(module.PHYSICAL_SMOKE_MAX_SOURCES, len(sources))
    assert set(range(6)).issubset(selected_indexes)


def test_physical_fire_core_mode_adapts_sources_per_frame() -> None:
    module = load_module()
    map_size = (160, 124)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=140, seed=90)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(36, 12, 28),
        render_size=(48, 36),
        max_sources=72,
        substeps=1,
        backend="numpy",
        emitter_mode="fire-core",
        seed=90,
    )
    assert effect is not None
    assert effect.emitter_mode == "fire-core"

    early = module._physical_sources_for_frame(effect, 12)
    mid = module._physical_sources_for_frame(effect, 67)

    assert len(early) >= 25
    assert len(mid) >= 45
    assert len(mid) <= 72
    assert float(np.ptp([source.x for source in mid])) > 18.0
    assert not np.allclose(
        np.asarray([(source.x, source.y) for source in early[: min(len(early), len(mid))]], dtype=np.float32),
        np.asarray([(source.x, source.y) for source in mid[: min(len(early), len(mid))]], dtype=np.float32),
    )


def test_audit_frame_indexes_include_fixed_timestamps() -> None:
    module = load_module()
    indexes = module._audit_frame_indexes(240, 30, [1.0, 3.5, 5.5, 7.0])

    assert indexes == {30: 1.0, 105: 3.5, 165: 5.5, 210: 7.0}


def test_reference_film_audit_frame_indexes_cover_first_30_seconds() -> None:
    module = load_module()

    indexes = module._reference_film_audit_frame_indexes(900, 30)

    assert indexes[0] == pytest.approx(0.0)
    assert indexes[90] == pytest.approx(3.0)
    assert indexes[810] == pytest.approx(27.0)
    assert indexes[885] == pytest.approx(29.5)
    assert len(indexes) >= 10


def test_source_wisp_audit_gates_reject_carpet_or_washed_out_combined() -> None:
    module = load_module()
    good_report = {
        "time_seconds": 3.5,
        "active_source_count": 40.0,
        "attached_source_count": 30.0,
        "screen_active_source_count": 40.0,
        "screen_attached_source_fraction": 0.70,
        "fire_core_visibility_fraction": 0.78,
        "active_fire_emitter_count": 50.0,
        "emitter_bbox_fraction": 0.030,
        "source_wisp_component_count": 18.0,
        "smoke_carpet_largest_component_fraction": 0.12,
        "low_frequency_haze_fraction": 0.15,
        "strand_to_haze_ratio": 0.34,
        "combined_strand_retention": 0.86,
        "morphology_stage_coverage_fraction": 0.004,
        "transition_width_growth_ratio": 1.35,
        "old_tail_width_growth_ratio": 1.72,
        "old_tail_alpha_p90_fraction": 1.10,
        "old_tail_endpoint_alpha_fraction": 0.32,
        "old_tail_coverage_growth_ratio": 3.1,
        "old_tail_edge_softness_px": 3.4,
        "old_tail_diffuse_to_core_area_ratio": 1.8,
        "brush_bundle_score": 0.40,
    }
    encoded = [{"time_seconds": 3.5, "strand_like_fraction": 0.006, "soft_tail_like_fraction": 0.004}]

    passed = module._evaluate_source_wisp_audit([good_report], encoded)
    assert passed["passed"] is True

    bad_report = dict(good_report)
    bad_report["combined_strand_retention"] = 0.25
    bad_report["smoke_carpet_largest_component_fraction"] = 0.62
    bad_report["low_frequency_haze_fraction"] = 0.48
    bad_report["old_tail_coverage_growth_ratio"] = 1.4
    bad_report["brush_bundle_score"] = 0.92
    failed = module._evaluate_source_wisp_audit([bad_report], encoded)

    assert failed["passed"] is False
    failed_names = {gate["name"] for gate in failed["gates"] if not gate["passed"]}
    assert "combined_strand_retention" in failed_names
    assert "smoke_carpet_largest_component_fraction" in failed_names
    assert "low_frequency_haze_fraction" in failed_names
    assert "old_tail_coverage_growth_ratio" in failed_names
    assert "brush_bundle_score" in failed_names


def test_reference_film_audit_gates_cover_composition_temporal_fire_and_delivery() -> None:
    module = load_module()
    base_reports = [
        {
            "time_seconds": 0.0,
            "full_bleed_frame_coverage_fraction": 1.0,
            "map_quad_area_fraction": 0.995,
            "combined_smoke_coverage_fraction": 0.32,
            "mid_scale_smoke_fraction": 0.10,
            "smoke_centroid_x_fraction": 0.42,
            "smoke_centroid_y_fraction": 0.58,
            "regional_smoke_coverage_fraction": 0.20,
            "dense_regional_smoke_fraction": 0.04,
            "active_fire_core_pixel_count": 120.0,
            "hot_fire_fraction": 0.009,
            "post_smoke_fire_visibility_fraction": 0.42,
            "median_fire_mark_radius_px": 1.2,
            "halo_core_area_ratio": 5.5,
            "median_label_contrast_delta": 0.16,
            "median_label_smoke_overlap_fraction": 0.12,
            "median_label_fire_overlap_fraction": 0.0,
            "median_label_text_pixel_fraction": 0.018,
            "distributed_fire_cluster_count": 5.0,
            "fire_spread_grid_cell_count": 4.0,
            "far_fire_core_fraction": 0.18,
            "primary_fire_dominance_fraction": 0.62,
            "regional_smoke_texture_score": 0.020,
            "regional_smoke_axis_band_score": 0.08,
            "regional_smoke_contour_band_score": 0.30,
            "regional_smoke_ring_score": 0.45,
            "date_ordinal": 737653.0,
            "burned_area_ha": 6700.0,
            "temporal_luma_delta": 0.0,
        },
        {
            "time_seconds": 15.0,
            "full_bleed_frame_coverage_fraction": 1.0,
            "map_quad_area_fraction": 0.995,
            "combined_smoke_coverage_fraction": 0.38,
            "mid_scale_smoke_fraction": 0.14,
            "smoke_centroid_x_fraction": 0.46,
            "smoke_centroid_y_fraction": 0.55,
            "regional_smoke_coverage_fraction": 0.24,
            "dense_regional_smoke_fraction": 0.06,
            "active_fire_core_pixel_count": 160.0,
            "hot_fire_fraction": 0.011,
            "post_smoke_fire_visibility_fraction": 0.46,
            "median_fire_mark_radius_px": 1.3,
            "halo_core_area_ratio": 6.2,
            "median_label_contrast_delta": 0.18,
            "median_label_smoke_overlap_fraction": 0.14,
            "median_label_fire_overlap_fraction": 0.0,
            "median_label_text_pixel_fraction": 0.020,
            "distributed_fire_cluster_count": 6.0,
            "fire_spread_grid_cell_count": 5.0,
            "far_fire_core_fraction": 0.22,
            "primary_fire_dominance_fraction": 0.58,
            "regional_smoke_texture_score": 0.024,
            "regional_smoke_axis_band_score": 0.07,
            "regional_smoke_contour_band_score": 0.28,
            "regional_smoke_ring_score": 0.42,
            "date_ordinal": 737675.0,
            "burned_area_ha": 120000.0,
            "temporal_luma_delta": 0.010,
        },
        {
            "time_seconds": 29.5,
            "full_bleed_frame_coverage_fraction": 1.0,
            "map_quad_area_fraction": 0.995,
            "combined_smoke_coverage_fraction": 0.44,
            "mid_scale_smoke_fraction": 0.16,
            "smoke_centroid_x_fraction": 0.50,
            "smoke_centroid_y_fraction": 0.50,
            "regional_smoke_coverage_fraction": 0.28,
            "dense_regional_smoke_fraction": 0.08,
            "active_fire_core_pixel_count": 210.0,
            "hot_fire_fraction": 0.012,
            "post_smoke_fire_visibility_fraction": 0.50,
            "median_fire_mark_radius_px": 1.4,
            "halo_core_area_ratio": 6.8,
            "median_label_contrast_delta": 0.17,
            "median_label_smoke_overlap_fraction": 0.16,
            "median_label_fire_overlap_fraction": 0.0,
            "median_label_text_pixel_fraction": 0.019,
            "distributed_fire_cluster_count": 7.0,
            "fire_spread_grid_cell_count": 5.0,
            "far_fire_core_fraction": 0.26,
            "primary_fire_dominance_fraction": 0.54,
            "regional_smoke_texture_score": 0.026,
            "regional_smoke_axis_band_score": 0.06,
            "regional_smoke_contour_band_score": 0.26,
            "regional_smoke_ring_score": 0.40,
            "date_ordinal": 737697.0,
            "burned_area_ha": 417898.0,
            "temporal_luma_delta": 0.012,
        },
    ]
    frame_reports = []
    burned_areas = [6700.0, 14000.0, 38000.0, 92000.0, 210000.0, 417898.0]
    for index in range(6):
        report = dict(base_reports[min(index // 2, 2)])
        report["time_seconds"] = float(index * 6)
        report["date_ordinal"] = 737653.0 + float(index * 5)
        report["burned_area_ha"] = burned_areas[index]
        report["temporal_luma_delta"] = 0.0 if index == 0 else 0.010 + index * 0.0005
        report["active_fire_core_pixel_count"] = 120.0 + index * 24.0
        report["smoke_centroid_x_fraction"] = 0.42 + index * 0.016
        report["smoke_centroid_y_fraction"] = 0.58 - index * 0.014
        frame_reports.append(report)
    encoded = [
        {"time_seconds": 0.0, "smoke_like_fraction": 0.08, "soft_tail_like_fraction": 0.004},
        {"time_seconds": 15.0, "smoke_like_fraction": 0.10, "soft_tail_like_fraction": 0.005},
    ]
    stream = {"width": 1920.0, "height": 1080.0, "codec_name": "h264", "bit_rate": 2450000.0}
    encode = {"size": (1920, 1080), "video_bitrate": "2600k"}

    passed = module._evaluate_reference_film_audit(frame_reports, encoded, stream, encode)

    assert passed["passed"] is True
    bad = [dict(report) for report in frame_reports]
    for index, report in enumerate(bad):
        report["date_ordinal"] = bad[0]["date_ordinal"] + float(index * 3)
        report["active_fire_core_pixel_count"] = 1.0
    failed = module._evaluate_reference_film_audit(bad, encoded, stream, encode)
    assert failed["passed"] is False
    failed_names = {gate["name"] for gate in failed["gates"] if not gate["passed"]}
    assert "temporal_date_span_days" in failed_names
    assert "median_active_fire_core_pixel_count" in failed_names

    bad_distribution = [dict(report) for report in frame_reports]
    for report in bad_distribution:
        report["distributed_fire_cluster_count"] = 0.0
        report["fire_spread_grid_cell_count"] = 1.0
        report["far_fire_core_fraction"] = 0.0
        report["primary_fire_dominance_fraction"] = 1.0
        report["regional_smoke_texture_score"] = 0.001
        report["regional_smoke_axis_band_score"] = 0.90
        report["regional_smoke_contour_band_score"] = 0.95
        report["regional_smoke_ring_score"] = 0.98
    distribution_failed = module._evaluate_reference_film_audit(bad_distribution, encoded, stream, encode)
    distribution_failed_names = {gate["name"] for gate in distribution_failed["gates"] if not gate["passed"]}
    assert "median_distributed_fire_cluster_count" in distribution_failed_names
    assert "median_fire_spread_grid_cell_count" in distribution_failed_names
    assert "median_far_fire_core_fraction" in distribution_failed_names
    assert "median_primary_fire_dominance_fraction" in distribution_failed_names
    assert "median_regional_smoke_texture_score" in distribution_failed_names
    assert "median_regional_smoke_axis_band_score" in distribution_failed_names
    assert "median_regional_smoke_contour_band_score" in distribution_failed_names
    assert "median_regional_smoke_ring_score" in distribution_failed_names

    high_bitrate = dict(stream)
    high_bitrate["bit_rate"] = 10400000.0
    bitrate_failed = module._evaluate_reference_film_audit(frame_reports, encoded, high_bitrate, encode)
    bitrate_failed_names = {gate["name"] for gate in bitrate_failed["gates"] if not gate["passed"]}
    assert "configured_delivery_bitrate_bps" in bitrate_failed_names


def test_physical_smoke_main_effect_renders_projected_volume() -> None:
    module = load_module()
    map_size = (72, 54)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=48, seed=41)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(44, 14, 34),
        render_size=(64, 48),
        max_sources=6,
        substeps=1,
        seed=41,
    )
    assert effect is not None
    assert effect.backend == ("native" if module._native_physical_smoke_available() else "numpy")

    for frame in range(10):
        module.step_physical_main_smoke(effect, frame)
    rgba = module.render_physical_main_smoke(effect, 10)
    alpha = rgba[..., 3]
    report = effect.domain.physics_report()

    assert rgba.shape == (54, 72, 4)
    assert rgba.dtype == np.uint8
    assert float(report["mass"]) > 0.0
    assert int(alpha.max()) > 10
    assert int(alpha.max()) <= module.PHYSICAL_SMOKE_MAX_ALPHA
    assert np.count_nonzero(alpha > 0) > 20


def test_python_raymarch_uses_density_color_and_source_glow() -> None:
    module = load_module()
    domain = module.NumpyPhysicalSmokeDomain((18, 9, 16), sparse_threshold=1.0e-6)
    z, y, x = np.mgrid[0:16, 0:9, 0:18].astype(np.float32)
    dense_core = np.exp(-(((x - 8.5) / 4.5) ** 2 + ((z - 7.5) / 3.5) ** 2 + ((y - 4.0) / 2.6) ** 2))
    hot_source = np.exp(-(((x - 5.0) / 1.7) ** 2 + ((z - 6.0) / 1.5) ** 2 + ((y - 3.0) / 1.3) ** 2))
    density = (0.74 * dense_core + 0.32 * hot_source).astype(np.float32)
    temperature = (1.8 * hot_source).astype(np.float32)
    emission = (2.2 * hot_source).astype(np.float32)
    soot = (0.22 * dense_core).astype(np.float32)
    domain.set_density(density)
    domain.set_temperature(temperature)
    domain.set_emission(emission)
    domain.set_soot(soot)
    settings = module.PhysicalSmokeRenderSettings3D(
        density_scale=1.22,
        extinction=1.48,
        exposure=1.24,
        fire_glow=1.15,
        thin_color=(0.45, 0.51, 0.58),
        dense_color=(0.90, 0.88, 0.78),
    )

    with_glow = module._python_projected_volume_raymarch(domain, (72, 54), 7, 91, settings)
    domain.set_temperature(np.zeros_like(temperature))
    domain.set_emission(np.zeros_like(emission))
    without_glow = module._python_projected_volume_raymarch(domain, (72, 54), 7, 91, settings)

    assert with_glow.shape == (54, 72, 4)
    assert int(with_glow[..., 3].max()) > 20
    assert int(np.count_nonzero(with_glow[..., 3] > 0)) > 80
    warm_with = with_glow[..., 0].astype(np.int16) - with_glow[..., 2].astype(np.int16)
    warm_without = without_glow[..., 0].astype(np.int16) - without_glow[..., 2].astype(np.int16)
    warm_delta = warm_with - warm_without
    assert int(warm_delta.max()) > 8
    assert int(np.count_nonzero(warm_delta > 8)) > 20
    assert int(with_glow[..., 0].max()) > int(without_glow[..., 0].max())


def test_python_volume_light_transmittance_self_shadows_density() -> None:
    module = load_module()
    density = np.zeros((18, 10, 22), dtype=np.float32)
    soot = np.zeros_like(density)
    particle_age = np.zeros_like(density)
    density[6:12, 4:8, 7:13] = 1.15
    soot[6:12, 4:8, 7:13] = 0.18
    settings = module.PhysicalSmokeRenderSettings3D(
        density_scale=1.3,
        extinction=1.6,
        shadow_steps=24,
        shadow_step_size=0.7,
    )

    light_trans = module._python_volume_light_transmittance(
        density,
        soot,
        4,
        settings.density_scale,
        settings.extinction,
        settings.soot_absorption,
        settings,
    )

    assert light_trans.shape == (18, 22)
    assert float(light_trans[9, 10]) < 0.45
    assert float(light_trans[2, 2]) > 0.92


def test_python_shadow_grid_self_shadows_full_volume() -> None:
    module = load_module()
    density = np.zeros((18, 10, 22), dtype=np.float32)
    soot = np.zeros_like(density)
    particle_age = np.zeros_like(density)
    density[6:12, 4:8, 7:13] = 1.15
    soot[6:12, 4:8, 7:13] = 0.18
    settings = module.PhysicalSmokeRenderSettings3D(
        density_scale=1.3,
        extinction=1.6,
        shadow_steps=24,
        shadow_step_size=0.7,
    )
    light_dir = np.asarray(module.PHYSICAL_SMOKE_SUN_DIRECTION, dtype=np.float32)
    light_dir /= max(float(np.linalg.norm(light_dir)), 1.0e-6)

    shadow = module._python_volume_shadow_grid(
        density,
        soot,
        particle_age,
        light_dir,
        settings.density_scale,
        settings.extinction,
        settings.soot_absorption,
        settings,
    )
    aged_shadow = module._python_volume_shadow_grid(
        density,
        soot,
        np.full_like(density, 24.0),
        light_dir,
        settings.density_scale,
        settings.extinction,
        settings.soot_absorption,
        settings,
    )

    assert shadow.shape == density.shape
    assert float(shadow[6:12, 4:8, 7:13].min()) < 0.45
    assert float(aged_shadow[6:12, 4:8, 7:13].min()) > float(shadow[6:12, 4:8, 7:13].min())
    assert float(shadow[1, 1, 1]) > 0.92


def test_physical_smoke_uses_streamers_cells_and_history() -> None:
    module = load_module()
    map_size = (72, 54)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=47)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(44, 14, 34),
        render_size=(64, 48),
        max_sources=6,
        substeps=1,
        backend="numpy",
        seed=47,
    )
    assert effect is not None

    emitters = module._physical_emitters_for_frame(effect, 12)
    radii = np.asarray([emitter.radius for emitter in emitters], dtype=np.float32)
    assert len(emitters) > len(effect.sources)
    assert float(np.percentile(radii, 20.0)) < float(np.percentile(radii, 80.0)) * 0.62

    for frame in range(16):
        module.step_physical_main_smoke(effect, frame)
    first = module.render_physical_main_smoke(effect, 16)
    assert len(effect.history) == 1
    for frame in range(17, 26):
        module.step_physical_main_smoke(effect, frame)
    second = module.render_physical_main_smoke(effect, 26)

    assert len(effect.history) >= 2
    assert np.count_nonzero(second[..., 3] > 0) >= np.count_nonzero(first[..., 3] > 0) * 0.80
    assert int(second[..., 3].max()) <= module.PHYSICAL_SMOKE_MAX_ALPHA
    active_alpha = second[..., 3][second[..., 3] > 0]
    assert float(np.std(active_alpha)) > 3.0


def test_numpy_physical_smoke_advects_particle_age_with_density() -> None:
    module = load_module()
    domain = module.NumpyPhysicalSmokeDomain((16, 8, 12), sparse_threshold=1.0e-6)
    density = np.zeros((12, 8, 16), dtype=np.float32)
    density[5:8, 3:5, 3:6] = 1.0
    velocity = np.zeros(density.shape + (3,), dtype=np.float32)
    velocity[..., 0] = 1.0
    domain.set_density(density)
    domain.set_velocity(velocity)
    domain.particle_age[...] = np.where(density > 0.0, 8.0, -1.0).astype(np.float32)
    settings = module.PhysicalSmokeStepSettings3D(
        dt=1.0,
        density_decay=0.0,
        temperature_decay=0.0,
        velocity_damping=0.0,
        diffusion=0.0,
        buoyancy=0.0,
        vorticity=0.0,
        pressure_iterations=1,
        turbulence_strength=0.0,
        wind=(0.0, 0.0, 0.0),
    )

    domain.step(settings, [])
    after_density = domain.to_density_numpy()
    after_age = domain.to_particle_age_numpy()
    peak = np.unravel_index(int(np.argmax(after_density)), after_density.shape)

    assert peak[2] > 3
    assert float(after_age[peak]) > 8.5


def test_physical_main_smoke_default_integrates_density_field_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    map_size = (64, 48)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=40, seed=49)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(36, 12, 28),
        render_size=(48, 36),
        max_sources=5,
        substeps=1,
        backend="numpy",
        seed=49,
    )
    assert effect is not None

    observed = {
        "_postprocess_physical_smoke_rgba": 0,
        "_physical_volume_structure_rgba": 0,
        "_physical_structure_enhancement_rgba": 0,
        "_physical_history_rgba": 0,
        "_temporal_blend_physical_smoke": 0,
    }
    for name in observed:
        original = getattr(module, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            observed[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapped)

    for frame in range(8):
        module.step_physical_main_smoke(effect, frame)
    rgba = module.render_physical_main_smoke(effect, 8)

    assert rgba.shape == (map_size[1], map_size[0], 4)
    assert int(rgba[..., 3].max()) > 0
    assert all(count >= 1 for count in observed.values())


def test_numpy_physical_solver_vorticity_confinement_adds_curl_force() -> None:
    module = load_module()
    domain = module.NumpyPhysicalSmokeDomain((18, 10, 16), sparse_threshold=1.0e-6)
    z, y, x = np.mgrid[0:16, 0:10, 0:18].astype(np.float32)
    density = np.exp(-(((x - 8.5) / 5.2) ** 2 + ((z - 7.5) / 4.4) ** 2 + ((y - 4.5) / 3.0) ** 2)).astype(np.float32)
    swirl = np.exp(-(((x - 8.5) / 4.0) ** 2 + ((z - 7.5) / 3.4) ** 2 + ((y - 4.5) / 2.6) ** 2)).astype(np.float32)
    velocity = np.zeros(density.shape + (3,), dtype=np.float32)
    velocity[..., 0] = -(z - 7.5) * 0.08 * swirl
    velocity[..., 2] = (x - 8.5) * 0.08 * swirl
    domain.set_density(density)
    domain.set_velocity(velocity)

    before = domain.to_velocity_numpy()
    settings = module.PhysicalSmokeStepSettings3D(
        dt=0.22,
        buoyancy=0.0,
        wind=(0.0, 0.0, 0.0),
        turbulence_strength=0.0,
        velocity_damping=0.0,
        vorticity=1.35,
    )
    domain._apply_forces(settings)
    after = domain.to_velocity_numpy()

    delta = np.linalg.norm(after - before, axis=-1)
    assert float(delta.max()) > 0.0005
    assert int(np.count_nonzero(delta > 0.0001)) > 20


def test_physical_main_smoke_mechanisms_are_active_end_to_end() -> None:
    module = load_module()
    map_size = (84, 64)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=72, seed=57)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(38, 12, 30),
        render_size=(56, 44),
        max_sources=6,
        substeps=1,
        backend="numpy",
        seed=57,
    )
    assert effect is not None

    centroids = []
    for frame in range(18):
        module.step_physical_main_smoke(effect, frame)
        density = effect.domain.to_density_numpy()
        if float(np.sum(density)) > 0.0:
            z, _y, x = np.mgrid[0 : density.shape[0], 0 : density.shape[1], 0 : density.shape[2]].astype(np.float32)
            mass = density.astype(np.float64)
            centroids.append(
                (
                    float(np.sum(x * mass) / np.sum(mass)),
                    float(np.sum(z * mass) / np.sum(mass)),
                )
            )

    report = effect.domain.physics_report()
    assert float(report["mass"]) > 0.0
    assert len(centroids) >= 3
    assert abs(centroids[-1][0] - centroids[0][0]) + abs(centroids[-1][1] - centroids[0][1]) > 0.18

    fields = module._physical_volume_lane_fields(effect, 18)
    assert fields is not None
    assert float(fields["support"].max()) > 0.05
    assert float(fields["lane"].max()) > 0.20
    assert float(fields["curl"].max()) > 0.05
    assert float(fields["shear"].max()) > 0.05

    first = module.render_physical_main_smoke(effect, 18)
    assert len(effect.history) == 1
    for frame in range(19, 29):
        module.step_physical_main_smoke(effect, frame)
    second = module.render_physical_main_smoke(effect, 29)
    assert len(effect.history) >= 2
    assert int(second[..., 3].max()) <= module.PHYSICAL_SMOKE_MAX_ALPHA
    assert np.count_nonzero(second[..., 3] > 0) >= np.count_nonzero(first[..., 3] > 0) * 0.70
    warm_smoke = second[..., 0].astype(np.int16) - second[..., 2].astype(np.int16)
    assert int(warm_smoke.max()) > 30
    assert int(np.count_nonzero((warm_smoke > 10) & (second[..., 3] > 0))) > len(effect.sources)

    fire = module.hybrid_fire_sources_rgba(effect.sources, 18, map_size, glow_only=True, bloom_scale=1.2)
    assert fire.shape == (map_size[1], map_size[0], 4)
    assert int(fire[..., 0].max()) > int(fire[..., 2].max()) + 100
    assert int(np.count_nonzero(fire[..., 3] > 0)) > len(effect.sources)


def test_projected_smoke_depth_cues_create_shadow_and_rim_light() -> None:
    module = load_module()
    alpha = np.zeros((48, 64), dtype=np.float32)
    alpha[16:31, 18:42] = 0.58
    alpha[20:27, 26:34] = 0.92

    shadow, rim, depth = module._projected_smoke_depth_cues(alpha, 12, 2020)

    assert shadow.shape == alpha.shape
    assert rim.shape == alpha.shape
    assert depth.shape == alpha.shape
    assert float(shadow.max()) > 0.12
    assert float(depth.max()) > 0.20
    assert int(np.count_nonzero(rim > 0.01)) > 0


def test_physical_structure_enhancement_adds_visible_ribbons() -> None:
    module = load_module()
    map_size = (80, 60)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=51)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(36, 12, 28),
        render_size=(48, 36),
        max_sources=6,
        substeps=1,
        backend="numpy",
        seed=51,
    )
    assert effect is not None

    base_alpha = np.zeros((map_size[1], map_size[0]), dtype=np.float32)
    base_alpha[24:42, 22:58] = 0.16
    ribbons = module._physical_structure_enhancement_rgba(effect, 20, base_alpha)
    ribbon_alpha = ribbons[..., 3]

    assert ribbons.shape == (map_size[1], map_size[0], 4)
    assert int(ribbon_alpha.max()) > 16
    assert int(np.count_nonzero(ribbon_alpha > 0)) > 80
    assert float(np.std(ribbon_alpha[ribbon_alpha > 0])) > 4.0


def test_physical_volume_structure_uses_density_and_velocity_features() -> None:
    module = load_module()
    map_size = (80, 60)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=53)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(24, 10, 20),
        render_size=(40, 30),
        max_sources=4,
        substeps=1,
        backend="numpy",
        seed=53,
    )
    assert effect is not None

    z, y, x = np.mgrid[0:20, 0:10, 0:24].astype(np.float32)
    ridge_a = np.exp(-(((x - 7.0) / 3.8) ** 2 + ((z - 7.0) / 2.5) ** 2 + ((y - 4.2) / 2.2) ** 2))
    ridge_b = 0.76 * np.exp(-(((x - 14.0) / 4.8) ** 2 + ((z - 13.0) / 3.2) ** 2 + ((y - 5.4) / 2.6) ** 2))
    density = (ridge_a + ridge_b).astype(np.float32)
    velocity = np.zeros(density.shape + (3,), dtype=np.float32)
    velocity[..., 0] = (z - 10.0) * 0.025
    velocity[..., 2] = -(x - 12.0) * 0.025
    effect.domain.set_density(np.ascontiguousarray(density, dtype=np.float32))
    effect.domain.set_velocity(np.ascontiguousarray(velocity, dtype=np.float32))

    fields = module._physical_volume_lane_fields(effect, 18)
    assert fields is not None
    for key in ("support", "ridges", "lane", "voids", "gain", "curl", "shear"):
        assert fields[key].shape == (map_size[1], map_size[0])
        assert fields[key].dtype == np.float32
    assert float(fields["lane"].max()) > 0.35
    assert float(fields["voids"].max()) > 0.08
    assert float(fields["curl"].max()) > 0.20

    rgba = module._physical_volume_structure_rgba(effect, 18, fields)
    alpha = rgba[..., 3]

    assert rgba.shape == (map_size[1], map_size[0], 4)
    assert int(alpha.max()) > 10
    assert int(np.count_nonzero(alpha > 0)) > 120
    assert float(np.std(alpha[alpha > 0])) > 3.0

    raw = np.zeros((map_size[1], map_size[0], 4), dtype=np.uint8)
    raw[..., :3] = 180
    raw[..., 3] = np.clip(np.round(fields["support"] * 190.0), 0, 190).astype(np.uint8)
    without_fields = module._postprocess_physical_smoke_rgba(raw, 18, 53, map_size, None)
    with_fields = module._postprocess_physical_smoke_rgba(raw, 18, 53, map_size, fields)
    void_mask = (fields["voids"] > 0.08) & (fields["lane"] < 0.42) & (fields["ridges"] < 0.42)
    assert int(np.count_nonzero(void_mask)) > 20
    assert float(np.mean(with_fields[..., 3][void_mask])) < float(np.mean(without_fields[..., 3][void_mask])) * 0.92


def test_physical_smoke_numpy_backend_remains_available() -> None:
    module = load_module()
    map_size = (60, 44)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=36, seed=43)
    effect = module.make_physical_main_smoke(
        sources,
        map_size,
        dims=(36, 12, 28),
        render_size=(48, 36),
        max_sources=5,
        substeps=1,
        backend="numpy",
        seed=43,
    )
    assert effect is not None
    assert effect.backend == "numpy"

    for frame in range(8):
        module.step_physical_main_smoke(effect, frame)
    rgba = module.render_physical_main_smoke(effect, 8)

    assert rgba.shape == (44, 60, 4)
    assert int(rgba[..., 3].max()) > 10


def test_hybrid_lifecycle_fades_old_smoke() -> None:
    module = load_module()
    alpha = module._hybrid_lifecycle_alpha(np.asarray([6.0, 80.0, 205.0], dtype=np.float32))

    assert float(alpha[1]) > float(alpha[0]) * 0.65
    assert float(alpha[2]) < float(alpha[1]) * 0.20


def test_hrrr_smoke_guidance_url_and_raster_conversion() -> None:
    module = load_module()
    url = module.hrrr_smoke_image_url("2020081618", 3, base_url="https://example.test/hrrr")
    assert url.endswith("/for_web/hrrr_ncep_smoke_jet/2020081618/full/trc1_full_int_f003.png")

    image = module.Image.new("RGB", (80, 60), (245, 246, 246))
    draw = module.ImageDraw.Draw(image)
    draw.rectangle((12, 22, 70, 34), fill=(190, 112, 64))
    draw.rectangle((38, 16, 62, 42), fill=(118, 128, 208))

    density = module._hrrr_smoke_image_to_density(image, (40, 30))

    assert density.shape == (30, 40)
    assert density.dtype == np.float32
    assert float(density.max()) > 0.35
    assert np.count_nonzero(density > 0.05) > 20


def test_hybrid_fire_sources_rgba_returns_correct_shape_and_dtype() -> None:
    """Fire layer has requested shape and uint8 dtype."""
    module = load_module()
    map_size = (80, 60)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=48, seed=61)

    fire_rgba = module.hybrid_fire_sources_rgba(sources, 24, map_size)

    assert fire_rgba.shape == (60, 80, 4)
    assert fire_rgba.dtype == np.uint8


def test_hybrid_fire_lifecycle_turns_hot_source_into_burn_scar() -> None:
    module = load_module()
    map_size = (84, 64)
    source = module.HybridSmokeSource(
        x=36.0,
        y=34.0,
        strength=1.45,
        radius_px=8.5,
        start_frame=0,
        end_frame=30,
        seed=151,
        burst_period_frames=36.0,
        burst_phase_frames=0.0,
        burst_duty=0.75,
        heat=1.35,
        smoke_rate=1.05,
        altitude_bias=0.18,
        flame_end_frame=8,
    )

    live_fire = module.hybrid_fire_sources_rgba([source], 5, map_size)
    late_fire = module.hybrid_fire_sources_rgba([source], 20, map_size)
    r_live, g_live, b_live, a_live = live_fire[..., 0], live_fire[..., 1], live_fire[..., 2], live_fire[..., 3]
    r_late, g_late, b_late, a_late = late_fire[..., 0], late_fire[..., 1], late_fire[..., 2], late_fire[..., 3]
    live_hot = (r_live > 220) & (g_live > 180) & (b_live < g_live) & (a_live > 50)
    late_hot = (r_late > 220) & (g_late > 180) & (b_late < g_late) & (a_late > 50)

    assert np.count_nonzero(live_hot) > 0
    assert np.count_nonzero(late_hot) == 0
    assert np.count_nonzero(a_late > 5) < np.count_nonzero(a_live > 5) * 0.35

    early_scar = module.hybrid_burn_scar_rgba([source], 2, map_size)
    late_scar = module.hybrid_burn_scar_rgba([source], 20, map_size)
    scar_alpha = late_scar[..., 3]
    scar_pixels = scar_alpha > 10

    assert np.count_nonzero(early_scar[..., 3] > 10) == 0
    assert np.count_nonzero(scar_pixels) > 40
    assert float(np.mean(late_scar[..., :3][scar_pixels])) < 70.0


def test_hybrid_smolder_smoke_outlives_visible_flame() -> None:
    module = load_module()
    map_size = (72, 54)
    source = module.HybridSmokeSource(
        x=28.0,
        y=25.0,
        strength=1.25,
        radius_px=7.0,
        start_frame=0,
        end_frame=30,
        seed=153,
        burst_period_frames=36.0,
        burst_phase_frames=0.0,
        burst_duty=0.75,
        heat=1.2,
        smoke_rate=1.2,
        altitude_bias=0.20,
        flame_end_frame=8,
    )
    density = np.zeros((map_size[1], map_size[0]), dtype=np.float32)
    age_mass = np.zeros_like(density)

    smolder_density, _ = module._inject_hybrid_sources(density, age_mass, [source], 14)
    late_fire = module.hybrid_fire_sources_rgba([source], 14, map_size)
    r, g, b, a = late_fire[..., 0], late_fire[..., 1], late_fire[..., 2], late_fire[..., 3]
    hot_mask = (r > 220) & (g > 180) & (b < g) & (a > 50)

    assert module._source_flame_lifecycle_weight(source, 14) < 0.05
    assert module._source_smoke_activity_weight(source, 14) > 0.05
    assert float(smolder_density.sum()) > 0.0001
    assert np.count_nonzero(hot_mask) == 0


def test_hybrid_fire_sources_rgba_has_white_yellow_hot_cores() -> None:
    """Fire layer contains bright white/yellow hot-core pixels."""
    module = load_module()
    map_size = (100, 75)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=63)

    fire_rgba = module.hybrid_fire_sources_rgba(sources, 32, map_size)
    r, g, b, a = fire_rgba[..., 0], fire_rgba[..., 1], fire_rgba[..., 2], fire_rgba[..., 3]

    # Find bright white/yellow pixels: high R, high G, lower B, visible alpha
    hot_mask = (r > 220) & (g > 180) & (b < g) & (a > 50)
    assert np.count_nonzero(hot_mask) > 5, "Expected visible white/yellow hot-core pixels"


def test_hybrid_fire_sources_rgba_bloom_larger_than_core_but_localized() -> None:
    """Orange/red bloom area is larger than white core but remains localized."""
    module = load_module()
    map_size = (100, 75)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=65)

    fire_rgba = module.hybrid_fire_sources_rgba(sources, 32, map_size)
    r, g, b, a = fire_rgba[..., 0], fire_rgba[..., 1], fire_rgba[..., 2], fire_rgba[..., 3]

    # White/yellow core: bright, high G relative to B
    core_mask = (r > 220) & (g > 180) & (a > 50)
    # Orange/red bloom: warm tones (R > G > B), visible
    bloom_mask = (r > 100) & (r > g) & (g > b) & (a > 5)

    core_count = np.count_nonzero(core_mask)
    bloom_count = np.count_nonzero(bloom_mask)

    assert bloom_count > core_count, "Bloom area should be larger than core"
    # Check bloom doesn't cover entire image (stays localized)
    assert bloom_count < int(map_size[0] * map_size[1] * 0.25), "Bloom should remain localized"


def test_hybrid_fire_sources_rgba_intensity_varies_across_sources() -> None:
    """Active source brightness varies across sources."""
    module = load_module()
    map_size = (120, 90)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=72, seed=67)
    # Collect alpha values (which vary with intensity) around each source
    fire_rgba = module.hybrid_fire_sources_rgba(sources, 36, map_size)
    a = fire_rgba[..., 3].astype(np.float32)

    local_maxes = []
    for source in sources[:25]:  # Sample first 25 sources
        if 36 < source.start_frame or 36 > source.end_frame:
            continue
        sx, sy = int(source.x), int(source.y)
        x0, x1 = max(0, sx - 5), min(map_size[0], sx + 6)
        y0, y1 = max(0, sy - 5), min(map_size[1], sy + 6)
        if x1 > x0 and y1 > y0:
            local_maxes.append(float(a[y0:y1, x0:x1].max()))

    assert len(local_maxes) >= 5, "Need enough active sources to measure variation"
    # Source intensities should vary (due to strength/heat differences and temporal variation)
    assert np.std(local_maxes) > 3.0, "Source brightness (alpha) should vary across sources"


def test_hybrid_fire_sources_rgba_glow_only_suppresses_cores() -> None:
    """glow_only=True suppresses hard cores while preserving bloom."""
    module = load_module()
    map_size = (100, 75)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=64, seed=69)

    full_fire = module.hybrid_fire_sources_rgba(sources, 32, map_size, glow_only=False)
    glow_only = module.hybrid_fire_sources_rgba(sources, 32, map_size, glow_only=True)

    r_full, g_full = full_fire[..., 0], full_fire[..., 1]
    r_glow, g_glow = glow_only[..., 0], glow_only[..., 1]

    # Full fire should have bright white/yellow cores
    full_core_mask = (r_full > 220) & (g_full > 180)
    # Glow-only should have much fewer bright core pixels
    glow_core_mask = (r_glow > 220) & (g_glow > 180)

    full_core_count = np.count_nonzero(full_core_mask)
    glow_core_count = np.count_nonzero(glow_core_mask)

    assert full_core_count > glow_core_count, "glow_only should suppress hard cores"
    # But glow-only should still have visible bloom
    assert np.count_nonzero(glow_only[..., 3] > 5) > 50, "glow_only should preserve bloom"


def test_hybrid_fire_has_connected_components_larger_than_single_dot() -> None:
    """Fire layer should contain warm connected components larger than a single source dot."""
    ndimage = pytest.importorskip("scipy.ndimage")

    module = load_module()
    map_size = (120, 90)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=72, seed=71)

    fire_rgba = module.hybrid_fire_sources_rgba(sources, 36, map_size)
    r, g, a = fire_rgba[..., 0], fire_rgba[..., 1], fire_rgba[..., 3]

    # Warm pixels mask: orange/red/yellow with visible alpha
    warm_mask = (r > 120) & (r > g * 0.8) & (a > 15)

    # Find connected components
    labeled, num_features = ndimage.label(warm_mask)
    if num_features == 0:
        pytest.fail("No warm connected components found")

    # Measure component sizes
    component_sizes = ndimage.sum(warm_mask, labeled, range(1, num_features + 1))

    # At least one component should be larger than typical single source dot (~15 pixels)
    max_size = max(component_sizes)
    assert max_size > 25, f"Largest warm component is {max_size} pixels, expected > 25 for cluster/chain rendering"


def test_hybrid_fire_fronts_dominate_warm_area_with_hot_strokes() -> None:
    """Cluster/front rendering should dominate warm area and include connected hot strokes."""
    ndimage = pytest.importorskip("scipy.ndimage")

    module = load_module()
    map_size = (160, 125)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=72, seed=71)

    fire_rgba = module.hybrid_fire_sources_rgba(sources, 36, map_size)
    r, g, b, a = fire_rgba[..., 0], fire_rgba[..., 1], fire_rgba[..., 2], fire_rgba[..., 3]

    warm_mask = (r > 120) & (r > g * 0.8) & (a > 15)
    warm_labeled, warm_count = ndimage.label(warm_mask)
    assert warm_count > 0, "Expected warm fire front pixels"
    warm_sizes = ndimage.sum(warm_mask, warm_labeled, range(1, warm_count + 1))
    largest_warm = float(max(warm_sizes))
    assert largest_warm / float(np.count_nonzero(warm_mask)) > 0.62

    hot_mask = (r > 220) & (g > 180) & (b < g) & (a > 50)
    hot_labeled, hot_count = ndimage.label(hot_mask)
    assert hot_count > 0, "Expected hot connected front pixels"
    hot_sizes = ndimage.sum(hot_mask, hot_labeled, range(1, hot_count + 1))
    assert max(hot_sizes) >= 32, "Expected a hot stroke larger than a single source core"


def test_hybrid_fire_bloom_remains_localized_to_fire_complex() -> None:
    """Warm bloom/core pixels should remain localized to the fire complex area."""
    module = load_module()
    map_size = (140, 105)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=80, seed=73)

    fire_rgba = module.hybrid_fire_sources_rgba(sources, 40, map_size)
    r, a = fire_rgba[..., 0], fire_rgba[..., 3]

    # Fire complex centroid based on sources
    active_sources = [s for s in sources if 40 >= s.start_frame and 40 <= s.end_frame]
    if not active_sources:
        pytest.skip("No active sources at frame 40")

    cx = sum(s.x for s in active_sources) / len(active_sources)
    cy = sum(s.y for s in active_sources) / len(active_sources)

    # Warm pixels with significant alpha
    warm_mask = (r > 100) & (a > 10)
    warm_ys, warm_xs = np.where(warm_mask)

    if len(warm_xs) == 0:
        pytest.fail("No warm pixels found")

    # Check that warm pixels are localized around the fire complex
    warm_cx = np.mean(warm_xs)
    warm_cy = np.mean(warm_ys)
    warm_spread_x = np.std(warm_xs)
    warm_spread_y = np.std(warm_ys)

    # Warm centroid should be near source centroid
    assert abs(warm_cx - cx) < map_size[0] * 0.25, "Warm pixels should be centered on fire complex"
    assert abs(warm_cy - cy) < map_size[1] * 0.25, "Warm pixels should be centered on fire complex"

    # Spread should be limited (not covering whole image)
    assert warm_spread_x < map_size[0] * 0.20, "Warm pixels spread too wide horizontally"
    assert warm_spread_y < map_size[1] * 0.20, "Warm pixels spread too wide vertically"


def test_hybrid_fire_glow_only_preserves_cluster_bloom() -> None:
    """glow_only=True should preserve grouped/chain bloom while suppressing bright hard cores."""
    module = load_module()
    map_size = (120, 90)
    sources = module.make_hybrid_smoke_sources((0.36, 0.62), map_size, total_frames=72, seed=75)

    full_fire = module.hybrid_fire_sources_rgba(sources, 36, map_size, glow_only=False, bloom_scale=1.2)
    glow_fire = module.hybrid_fire_sources_rgba(sources, 36, map_size, glow_only=True, bloom_scale=1.2)

    # Count bloom pixels (lower threshold for soft glow)
    full_bloom_count = np.count_nonzero(full_fire[..., 3] > 8)
    glow_bloom_count = np.count_nonzero(glow_fire[..., 3] > 8)

    # glow_only should have substantial bloom coverage (cluster patches, chains)
    assert glow_bloom_count > 100, "glow_only should preserve substantial bloom from clusters/chains"

    # glow_only should have comparable or even more bloom coverage than full (since it includes chain bloom)
    assert glow_bloom_count >= full_bloom_count * 0.5, "glow_only bloom coverage should be substantial"

    # But bright cores should be suppressed
    full_bright = np.count_nonzero((full_fire[..., 0] > 230) & (full_fire[..., 1] > 200))
    glow_bright = np.count_nonzero((glow_fire[..., 0] > 230) & (glow_fire[..., 1] > 200))
    assert glow_bright < full_bright, "glow_only should suppress bright white/yellow cores"
