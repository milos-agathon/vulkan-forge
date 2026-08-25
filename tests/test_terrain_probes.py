from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import (
    PomSettings,
    ProbeSettings,
    ReflectionProbeSettings,
    make_terrain_params_config,
)


ROOT = Path(__file__).resolve().parents[1]
TERRAIN_SHADER = ROOT / "src" / "shaders" / "terrain_pbr_pom.wgsl"
PROBE_SHADER = ROOT / "src" / "shaders" / "terrain_probes.wgsl"
GPU_AVAILABLE = terrain_rendering_available()


def _write_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 180, 128]))


def _build_test_ibl(*, rotation_deg: float = 0.0, intensity: float = 1.0):
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = tmp.name
    _write_test_hdr(hdr_path)
    ibl = f3d.IBL.from_hdr(hdr_path, intensity=float(intensity), rotate_deg=float(rotation_deg))
    Path(hdr_path).unlink(missing_ok=True)
    return ibl


def _build_bowl_heightmap(size: int = 192) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    rim = 0.85 * np.exp(-((radius - 0.42) ** 2) * 90.0)
    basin = -0.55 * np.exp(-(radius**2) * 28.0)
    spur = 0.20 * np.exp(-((xx - 0.55) ** 2 * 22.0 + (yy + 0.10) ** 2 * 18.0))
    heightmap = rim + basin + spur
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_water_mask(size: int = 192) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    lake = (xx / 0.44) ** 2 + (yy / 0.34) ** 2 <= 1.0
    return np.where(lake, 1.0, 0.0).astype(np.float32)


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#16341a"),
            (0.35, "#40692f"),
            (0.65, "#8b7b53"),
            (1.0, "#f3f6fb"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _mean_luminance(image: np.ndarray) -> float:
    rgb = image[..., :3].astype(np.float32)
    return float(np.mean(rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722))


def _mean_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left[..., :3].astype(np.float32) - right[..., :3].astype(np.float32))))


def _masked_mean_abs_diff(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(left[..., :3].astype(np.float32) - right[..., :3].astype(np.float32))
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    selected = diff[mask]
    if selected.size == 0:
        return 0.0
    return float(np.mean(selected))


def _reflection_probe_texture_bytes(probe_count: int, resolution: int) -> int:
    total_texels = 0
    size = max(int(resolution), 1)
    while True:
        total_texels += size * size
        if size == 1:
            break
        size = max(size // 2, 1)
    return total_texels * int(probe_count) * 6 * 8


def _render_probe_scene(
    renderer,
    material_set,
    ibl,
    heightmap: np.ndarray,
    overlay,
    *,
    probes: ProbeSettings | None,
    reflection_probes: ReflectionProbeSettings | None = None,
    water_mask: np.ndarray | None = None,
    debug_mode: int = 0,
    cam_phi_deg: float = 138.0,
    cam_theta_deg: float = 58.0,
) -> np.ndarray:
    params = make_terrain_params_config(
        size_px=(224, 224),
        render_scale=1.0,
        terrain_span=4.0,
        msaa_samples=1,
        z_scale=2.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=True,
        ibl_intensity=3.0,
        light_azimuth_deg=138.0,
        light_elevation_deg=16.0,
        sun_intensity=0.8,
        cam_radius=6.2,
        cam_phi_deg=cam_phi_deg,
        cam_theta_deg=cam_theta_deg,
        fov_y_deg=48.0,
        camera_mode="screen",
        debug_mode=debug_mode,
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        probes=probes,
        reflection_probes=reflection_probes,
    )
    native_params = f3d.TerrainRenderParams(params)
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=native_params,
        heightmap=heightmap,
        target=None,
        water_mask=water_mask,
    )
    return frame.to_numpy()


def test_probe_shader_module_exists() -> None:
    assert PROBE_SHADER.exists(), f"Missing probe shader include: {PROBE_SHADER}"
    source = PROBE_SHADER.read_text(encoding="utf-8")
    assert "struct ProbeGridUniforms" in source
    assert "struct ReflectionProbeGridUniforms" in source
    assert "fn sample_probe_irradiance" in source
    assert "fn sample_reflection_probe" in source


def test_terrain_shader_includes_probe_module() -> None:
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    assert '#include "terrain_probes.wgsl"' in source
    assert "DBG_PROBE_IRRADIANCE" in source
    assert "DBG_PROBE_WEIGHT" in source
    assert "DBG_REFLECTION_PROBE_COLOR" in source
    assert "DBG_REFLECTION_PROBE_WEIGHT" in source


def test_reflection_probe_water_path_uses_shared_specular_mix() -> None:
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    assert (
        "let blended_specular = det_mix3(ibl_split.specular, local_specular, reflection_probe_result.weight);"
        in source
    )
    assert "var combined_reflection = ibl_contrib;" in source
    assert "combined_reflection = blend_water_reflection(" in source


def test_reflection_probe_water_path_avoids_double_weight_and_probe_override_block() -> None:
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    water_start = source.index("if (is_water) {")
    water_end = source.index("        } else {", water_start)
    water_block = source[water_start:water_end]

    # Guard against the regressed water-only probe block that squared the probe weight
    # and blended baked probes over planar reflections after the planar composite.
    forbidden_tokens = (
        "local_water_reflection",
        "local_probe_blend",
        "local_probe_highlight",
        "reflection_strength",
        "reflection_probe_weight",
    )
    for token in forbidden_tokens:
        assert token not in water_block, f"Unexpected water-only reflection probe token: {token}"


def test_probe_settings_defaults_disabled() -> None:
    settings = ProbeSettings()
    assert settings.enabled is False
    assert settings.grid_dims == (8, 8)


def test_probe_settings_validation_zero_dims() -> None:
    with pytest.raises(ValueError, match="grid_dims must be >= \\(1, 1\\)"):
        ProbeSettings(enabled=True, grid_dims=(0, 0))


def test_probe_settings_validation_probe_limit() -> None:
    with pytest.raises(ValueError, match="probe count limit"):
        ProbeSettings(enabled=True, grid_dims=(65, 65))


def test_probe_settings_single_probe_grid() -> None:
    settings = ProbeSettings(enabled=True, grid_dims=(1, 1))
    assert settings.grid_dims == (1, 1)


def test_reflection_probe_settings_defaults_disabled() -> None:
    settings = ReflectionProbeSettings()
    assert settings.enabled is False
    assert settings.grid_dims == (4, 4)


def test_reflection_probe_settings_validation_probe_limit() -> None:
    with pytest.raises(ValueError, match="reflection probe count limit"):
        ReflectionProbeSettings(enabled=True, grid_dims=(17, 17))


def test_reflection_probe_settings_validation_resolution_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        ReflectionProbeSettings(enabled=True, resolution=12)


@pytest.fixture(scope="module")
def probe_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("Probe lighting tests require a terrain-capable hardware-backed forge3d runtime")

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    overlay = _build_overlay()
    heightmap = _build_bowl_heightmap()
    water_mask = _build_water_mask(heightmap.shape[0])
    ibl = _build_test_ibl()
    return renderer, material_set, ibl, heightmap, overlay, water_mask


@pytest.mark.skipif(not GPU_AVAILABLE, reason="Probe lighting tests require GPU-backed forge3d module")
class TestTerrainProbeLighting:
    def test_probe_fallback_pixel_identical(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        baseline = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=None,
        )
        disabled = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
        )
        max_diff = int(np.max(np.abs(baseline.astype(np.int16) - disabled.astype(np.int16))))
        assert max_diff <= 1, f"Disabled probes regressed baseline output by {max_diff} LSB"

    def test_probe_memory_tracked(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(4, 4), ray_count=32),
        )
        report = renderer.get_probe_memory_report()
        assert report["probe_count"] == 16
        assert report["grid_uniform_bytes"] == 48
        assert report["probe_ssbo_bytes"] == 16 * 144
        assert report["total_bytes"] == 48 + 16 * 144

    def test_probe_valley_darker(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        probe_debug = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=48),
            debug_mode=50,
        )
        valley = _mean_luminance(probe_debug[96:128, 96:128])
        ridge = _mean_luminance(probe_debug[150:182, 96:128])
        assert ridge > valley * 1.02, (
            f"Expected ridge probe irradiance > valley, got ridge={ridge:.3f}, valley={valley:.3f}"
        )

    def test_probe_ridge_brighter(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        probe_debug = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=48),
            debug_mode=50,
        )
        ridge = _mean_luminance(probe_debug[96:128, 40:72])
        shoulder = _mean_luminance(probe_debug[96:128, 96:128])
        assert ridge > shoulder, (
            f"Expected exposed ridge > shoulder luminance, got ridge={ridge:.3f}, shoulder={shoulder:.3f}"
        )

    def test_probe_out_of_bounds_weight_zero(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        weight = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(
                enabled=True,
                grid_dims=(2, 2),
                origin=(-0.15, -0.15),
                spacing=(0.3, 0.3),
                fallback_blend_distance=0.18,
                ray_count=24,
            ),
            debug_mode=51,
        )
        center = float(np.mean(weight[92:132, 92:132, 0]))
        corner = float(np.mean(weight[0:24, 0:24, 0]))
        assert center > 20.0, f"Expected non-zero probe coverage near center, got {center:.2f}"
        assert corner < 5.0, f"Expected out-of-bounds weight to fall back to zero, got {corner:.2f}"

    def test_probe_edge_blend_smooth(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        weight = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(
                enabled=True,
                grid_dims=(2, 2),
                origin=(-0.15, -0.15),
                spacing=(0.3, 0.3),
                fallback_blend_distance=0.24,
                ray_count=24,
            ),
            debug_mode=51,
        )
        weight_map = weight[..., 0].astype(np.float32)
        midtones = weight_map[(weight_map > 10.0) & (weight_map < 245.0)]
        assert midtones.size > 0, "Expected smooth edge blend to produce intermediate probe weights"

    def test_probe_single_probe_grid(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        weight = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(1, 1), ray_count=24),
            debug_mode=51,
        )
        report = renderer.get_probe_memory_report()
        weight_map = weight[..., 0].astype(np.float32)
        assert report["probe_count"] == 1
        assert float(np.mean(weight_map)) > 240.0
        assert float(np.std(weight_map)) < 2.0

    def test_probe_invalidation_triggers(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        base = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(5, 5), ray_count=32),
            debug_mode=50,
        )
        sky_changed = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(
                enabled=True,
                grid_dims=(5, 5),
                ray_count=32,
                sky_color=(1.0, 0.65, 0.45),
            ),
            debug_mode=50,
        )
        dims_changed = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=True, grid_dims=(3, 3), ray_count=32),
            debug_mode=50,
        )
        assert _mean_abs_diff(base, sky_changed) > 0.20
        assert _mean_abs_diff(base, dims_changed) > 0.20

    def test_reflection_probe_fallback_pixel_identical(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        baseline = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=None,
        )
        disabled = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=False),
        )
        max_diff = int(np.max(np.abs(baseline.astype(np.int16) - disabled.astype(np.int16))))
        assert max_diff <= 1, f"Disabled reflection probes regressed baseline output by {max_diff} LSB"

    def test_reflection_probe_memory_tracked(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        settings = ReflectionProbeSettings(enabled=True, grid_dims=(4, 4), resolution=16, ray_count=9)
        _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=settings,
        )
        report = renderer.get_reflection_probe_memory_report()
        expected_texture_bytes = _reflection_probe_texture_bytes(16, 16)
        assert report["probe_count"] == 16
        assert report["resolution"] == 16
        assert report["mip_levels"] == 5
        assert report["grid_uniform_bytes"] == 80
        assert report["cubemap_texture_bytes"] == expected_texture_bytes
        assert report["total_bytes"] == 80 + expected_texture_bytes

    def test_reflection_probe_spatially_varies(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        reflection_debug = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(enabled=True, grid_dims=(6, 6), ray_count=16),
            debug_mode=8,
        )
        center = _mean_luminance(reflection_debug[96:128, 96:128])
        lower = _mean_luminance(reflection_debug[150:182, 96:128])
        upper = _mean_luminance(reflection_debug[40:72, 96:128])
        spread = max(center, lower, upper) - min(center, lower, upper)
        assert spread > 2.0, (
            "Expected reflection probe debug view to vary across the terrain, "
            f"got center={center:.3f}, lower={lower:.3f}, upper={upper:.3f}"
        )

    def test_reflection_probe_out_of_bounds_weight_zero(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        baseline = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=None,
            debug_mode=53,
        )
        centered = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(
                enabled=True,
                grid_dims=(2, 2),
                origin=(-0.15, -0.15),
                spacing=(0.3, 0.3),
                fallback_blend_distance=(0.18, 0.12),
                ray_count=9,
            ),
            debug_mode=53,
        )
        out_of_bounds = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(
                enabled=True,
                grid_dims=(2, 2),
                origin=(10.0, 10.0),
                spacing=(0.3, 0.3),
                fallback_blend_distance=(0.18, 0.12),
                ray_count=9,
            ),
            debug_mode=53,
        )
        assert _mean_abs_diff(baseline, centered) > 3.0
        assert _mean_abs_diff(baseline, out_of_bounds) <= 1.0

    def test_reflection_probe_env_rotation_changes_bake(self, probe_render_env) -> None:
        renderer, material_set, _, heightmap, overlay, _ = probe_render_env
        base_ibl = _build_test_ibl(rotation_deg=0.0)
        rotated_ibl = _build_test_ibl(rotation_deg=90.0)
        settings = ReflectionProbeSettings(enabled=True, grid_dims=(5, 5), resolution=16, ray_count=16)
        base = _render_probe_scene(
            renderer,
            material_set,
            base_ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=settings,
            debug_mode=8,
        )
        rotated = _render_probe_scene(
            renderer,
            material_set,
            rotated_ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=settings,
            debug_mode=8,
        )
        assert _mean_abs_diff(base, rotated) > 5.0

    def test_reflection_probe_stress_64_limit(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, _ = probe_render_env
        settings = ReflectionProbeSettings(
            enabled=True,
            grid_dims=(8, 8),
            resolution=8,
            ray_count=8,
            trace_steps=96,
            trace_refine_steps=3,
        )
        image = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=settings,
            debug_mode=8,
        )
        report = renderer.get_reflection_probe_memory_report()
        expected_texture_bytes = _reflection_probe_texture_bytes(64, 8)
        assert report["probe_count"] == 64
        assert report["resolution"] == 8
        assert report["mip_levels"] == 4
        assert report["cubemap_texture_bytes"] == expected_texture_bytes
        assert report["total_bytes"] == 80 + expected_texture_bytes
        assert np.isfinite(image).all()
        assert float(np.mean(image[..., :3])) > 1.0

    def test_reflection_probe_water_beauty_scales_with_strength(self, probe_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay, water_mask = probe_render_env
        water_debug = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=None,
            water_mask=water_mask,
            debug_mode=4,
        )
        water_pixels = (water_debug[..., 1] > 200) & (water_debug[..., 2] > 200) & (water_debug[..., 0] < 40)
        ys, xs = np.where(water_pixels)
        grid_y, grid_x = np.indices(water_pixels.shape)
        x0, x1 = np.quantile(xs, [0.2, 0.8])
        y0, y1 = np.quantile(ys, [0.2, 0.8])
        roi_mask = water_pixels & (grid_x >= x0) & (grid_x <= x1) & (grid_y >= y0) & (grid_y <= y1)
        assert int(np.count_nonzero(roi_mask)) > 1000, "Expected substantial interior water ROI from debug mode 4"

        baseline = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=None,
            water_mask=water_mask,
        )
        half_strength = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(
                enabled=True,
                grid_dims=(1, 1),
                resolution=16,
                ray_count=16,
                strength=0.5,
            ),
            water_mask=water_mask,
        )
        full_strength = _render_probe_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            probes=ProbeSettings(enabled=False),
            reflection_probes=ReflectionProbeSettings(
                enabled=True,
                grid_dims=(1, 1),
                resolution=16,
                ray_count=16,
                strength=1.0,
            ),
            water_mask=water_mask,
        )

        half_delta = _masked_mean_abs_diff(baseline, half_strength, roi_mask)
        full_delta = _masked_mean_abs_diff(baseline, full_strength, roi_mask)
        assert full_delta > 0.8, f"Expected measurable full-strength water probe contribution, got {full_delta:.3f}"
        ratio = half_delta / full_delta
        assert 0.35 <= ratio <= 0.85, (
            "Expected half-strength reflection probes to produce an intermediate water beauty delta; "
            f"got half={half_delta:.3f}, full={full_delta:.3f}, ratio={ratio:.3f}"
        )
