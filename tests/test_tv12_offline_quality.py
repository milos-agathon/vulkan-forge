"""TV12: Offline terrain accumulation, adaptive sampling, and session semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _terrain_runtime import terrain_rendering_available

try:
    import forge3d as f3d
    from forge3d.terrain_params import PomSettings, make_terrain_params_config
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.apple_metal_physical


def _write_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 160, 128]))


def _overlay() -> object:
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#0f172a"),
            (0.33, "#2b6cb0"),
            (0.66, "#e8b04b"),
            (1.0, "#fff7ed"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _complex_heightmap(size: int = 96) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    waves = (
        0.42 * np.sin(xx * 18.0)
        + 0.31 * np.cos(yy * 23.0)
        + 0.19 * np.sin((xx + yy) * 31.0)
        + 0.08 * np.sign(np.sin(xx * 9.0) * np.cos(yy * 11.0))
    )
    ramp = 0.22 * (0.7 * xx - yy)
    heightmap = waves + ramp
    heightmap -= heightmap.min()
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _flat_heightmap(size: int = 96) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)


def _make_params(
    *,
    aa_samples: int,
    aa_seed: int | None,
    size_px: tuple[int, int] = (96, 64),
    denoise: DenoiseSettings | None = None,
    debug_mode: int = 0,
) -> object:
    return f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=size_px,
            render_scale=1.0,
            terrain_span=2.2,
            msaa_samples=1,
            z_scale=1.55,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="colormap",
            colormap_strength=1.0,
            ibl_enabled=True,
            light_azimuth_deg=132.0,
            light_elevation_deg=22.0,
            sun_intensity=2.8,
            cam_radius=3.7,
            cam_phi_deg=133.0,
            cam_theta_deg=60.0,
            fov_y_deg=50.0,
            camera_mode="mesh",
            overlays=[_overlay()],
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            aa_samples=aa_samples,
            aa_seed=aa_seed,
            denoise=denoise,
            debug_mode=debug_mode,
        )
    )


def _psnr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    mse = float(np.mean((lhs - rhs) ** 2))
    if mse <= 1e-10:
        return 99.0
    peak = max(float(lhs.max()), float(rhs.max()), 1e-6)
    return float(20.0 * np.log10(peak / np.sqrt(mse)))


@pytest.fixture()
def runtime_assets(tmp_path: Path) -> tuple[object, object, object, object]:
    hdr_path = tmp_path / "tv12_env.hdr"
    _write_test_hdr(hdr_path)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    return renderer, material_set, ibl, hdr_path


def test_offline_is_deterministic_for_same_seed(runtime_assets: tuple[object, object, object, object]) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    settings = f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=4)

    params_a = _make_params(aa_samples=16, aa_seed=12345)
    result_a = f3d.render_offline(renderer, material_set, ibl, params_a, heightmap, settings=settings)

    params_b = _make_params(aa_samples=16, aa_seed=12345)
    result_b = f3d.render_offline(renderer, material_set, ibl, params_b, heightmap, settings=settings)

    np.testing.assert_array_equal(result_a.frame.to_numpy(), result_b.frame.to_numpy())
    assert result_a.metadata["samples_used"] == result_b.metadata["samples_used"] == 16


def test_multisample_offline_render_improves_psnr(runtime_assets: tuple[object, object, object, object]) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    settings = f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=4)

    baseline = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        _make_params(aa_samples=1, aa_seed=7),
        heightmap,
        settings=settings,
    )
    improved = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        _make_params(aa_samples=12, aa_seed=7),
        heightmap,
        settings=settings,
    )
    reference = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        _make_params(aa_samples=32, aa_seed=7),
        heightmap,
        settings=settings,
    )

    baseline_rgb = np.asarray(baseline.hdr_frame.to_numpy_f32(), dtype=np.float32)[..., :3]
    improved_rgb = np.asarray(improved.hdr_frame.to_numpy_f32(), dtype=np.float32)[..., :3]
    reference_rgb = np.asarray(reference.hdr_frame.to_numpy_f32(), dtype=np.float32)[..., :3]

    baseline_psnr = _psnr(baseline_rgb, reference_rgb)
    improved_psnr = _psnr(improved_rgb, reference_rgb)
    assert improved_psnr > baseline_psnr + 0.5, (
        f"Expected offline accumulation PSNR to improve by >0.5 dB, got "
        f"{baseline_psnr:.3f} -> {improved_psnr:.3f}"
    )


def test_adaptive_sampling_uses_fewer_samples_on_flat_scene(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    flat_heightmap = _flat_heightmap()
    params = _make_params(aa_samples=32, aa_seed=99)

    adaptive = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        params,
        flat_heightmap,
        settings=f3d.OfflineQualitySettings(
            enabled=True,
            adaptive=True,
            target_variance=0.002,
            max_samples=32,
            min_samples=4,
            batch_size=4,
            tile_size=8,
            convergence_ratio=0.90,
        ),
    )
    uniform = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        params,
        flat_heightmap,
        settings=f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=4),
    )

    assert adaptive.metadata["samples_used"] < uniform.metadata["samples_used"]
    assert uniform.metadata["samples_used"] == 32


def test_session_guard_blocks_one_shot_render_and_metrics_are_queryable(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    params = _make_params(aa_samples=8, aa_seed=5)

    renderer.begin_offline_accumulation(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        water_mask=None,
    )
    try:
        batch = renderer.accumulate_batch(4)
        assert isinstance(batch, f3d.OfflineBatchResult)
        assert int(batch["total_samples"]) == 4

        metrics = renderer.read_accumulation_metrics(0.002, 8)
        assert isinstance(metrics, f3d.OfflineMetrics)
        assert 0.0 <= float(metrics["converged_tile_ratio"]) <= 1.0
        assert float(metrics["mean_delta"]) >= 0.0
        assert float(metrics["p95_delta"]) >= 0.0

        with pytest.raises(RuntimeError, match="offline accumulation session"):
            renderer.render_terrain_pbr_pom(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
                target=None,
                water_mask=None,
            )
    finally:
        renderer.end_offline_accumulation()


def test_hdr_frame_save_requires_explicit_exr_extension(
    runtime_assets: tuple[object, object, object, object],
    tmp_path: Path,
) -> None:
    renderer, _, _, _ = runtime_assets
    hdr_frame = renderer.upload_hdr_frame(
        np.ones((4, 4, 3), dtype=np.float32),
        (4, 4),
    )

    with pytest.raises(ValueError, match=r"expected \.exr extension"):
        hdr_frame.save(str(tmp_path / "terrain_offline_hdr"))


def test_begin_offline_session_raises_when_already_active(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    params = _make_params(aa_samples=8, aa_seed=17)

    renderer.begin_offline_accumulation(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        water_mask=None,
    )
    try:
        with pytest.raises(RuntimeError, match="already active"):
            renderer.begin_offline_accumulation(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
                water_mask=None,
            )
    finally:
        renderer.end_offline_accumulation()


def test_single_sample_offline_matches_manual_session_baseline(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    params = _make_params(aa_samples=1, aa_seed=77)

    renderer.begin_offline_accumulation(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        water_mask=None,
    )
    try:
        batch = renderer.accumulate_batch(1)
        assert int(batch["total_samples"]) == 1
        manual_hdr, manual_aov = renderer.resolve_offline_hdr()
        manual_frame = renderer.tonemap_offline_hdr(manual_hdr)
    finally:
        renderer.end_offline_accumulation()

    one_shot = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        params,
        heightmap,
        settings=f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=1),
    )

    np.testing.assert_array_equal(one_shot.frame.to_numpy(), manual_frame.to_numpy())
    np.testing.assert_allclose(one_shot.hdr_frame.to_numpy_f32(), manual_hdr.to_numpy_f32())
    np.testing.assert_allclose(one_shot.aov_frame.albedo(), manual_aov.albedo())
    np.testing.assert_allclose(one_shot.aov_frame.normal(), manual_aov.normal())
    np.testing.assert_allclose(one_shot.aov_frame.depth(), manual_aov.depth())


def test_adaptive_sampling_complex_scene_respects_min_and_max_samples(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    params = _make_params(aa_samples=32, aa_seed=29)

    adaptive = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        params,
        heightmap,
        settings=f3d.OfflineQualitySettings(
            enabled=True,
            adaptive=True,
            target_variance=0.001,
            max_samples=12,
            min_samples=4,
            batch_size=4,
            tile_size=8,
            convergence_ratio=0.95,
        ),
    )

    assert 4 < adaptive.metadata["samples_used"] <= 12


def test_progress_callback_reports_metrics_and_metadata_matches_last_update(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    params = _make_params(aa_samples=16, aa_seed=41)
    updates: list[f3d.OfflineProgress] = []

    result = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        params,
        heightmap,
        settings=f3d.OfflineQualitySettings(
            enabled=True,
            adaptive=True,
            target_variance=0.001,
            max_samples=12,
            min_samples=4,
            batch_size=4,
            tile_size=8,
            convergence_ratio=0.95,
        ),
        progress_callback=updates.append,
    )

    assert updates
    assert all(update.max_samples == 12 for update in updates)
    assert all(update.samples_so_far > 0 for update in updates)
    assert all(update.elapsed_ms >= 0.0 for update in updates)
    assert updates[-1].samples_so_far == result.metadata["samples_used"]
    assert result.metadata["adaptive"] is True
    assert result.metadata["target_samples"] == 12
    assert result.metadata["denoiser_used"] == "none"
    assert result.metadata["final_p95_delta"] == pytest.approx(updates[-1].p95_delta, rel=1e-5)
    assert result.metadata["converged_ratio"] == pytest.approx(
        updates[-1].converged_ratio,
        rel=1e-5,
    )


def test_render_offline_cleans_up_after_progress_callback_exception(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _complex_heightmap()
    params = _make_params(aa_samples=8, aa_seed=57)

    def _boom(_progress: f3d.OfflineProgress) -> None:
        raise RuntimeError("stop after first progress update")

    with pytest.raises(RuntimeError, match="stop after first progress update"):
        f3d.render_offline(
            renderer,
            material_set,
            ibl,
            params,
            heightmap,
            settings=f3d.OfflineQualitySettings(
                enabled=True,
                adaptive=True,
                target_variance=0.002,
                max_samples=8,
                min_samples=4,
                batch_size=4,
                tile_size=8,
                convergence_ratio=0.95,
            ),
            progress_callback=_boom,
        )

    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
        water_mask=None,
    )
    assert isinstance(frame, f3d.Frame)


def test_converged_tile_ratio_trends_upward_on_flat_scene(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _flat_heightmap()
    params = _make_params(aa_samples=16, aa_seed=123)

    renderer.begin_offline_accumulation(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        water_mask=None,
    )
    try:
        ratios: list[float] = []
        for _ in range(4):
            renderer.accumulate_batch(4)
            metrics = renderer.read_accumulation_metrics(0.002, 8)
            ratios.append(float(metrics["converged_tile_ratio"]))
    finally:
        renderer.end_offline_accumulation()

    assert ratios[0] == 0.0
    assert all(0.0 <= ratio <= 1.0 for ratio in ratios)
    assert ratios[2] >= ratios[0]
    assert ratios[3] >= ratios[1]


def test_offline_water_mask_path_changes_output(
    runtime_assets: tuple[object, object, object, object],
) -> None:
    renderer, material_set, ibl, _ = runtime_assets
    heightmap = _flat_heightmap()
    water_mask = np.zeros_like(heightmap, dtype=np.float32)
    water_mask[:, water_mask.shape[1] // 2 :] = 1.0
    params = _make_params(aa_samples=4, aa_seed=211, debug_mode=5)
    settings = f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=2)

    without_mask = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        params,
        heightmap,
        settings=settings,
        water_mask=None,
    )
    with_mask = f3d.render_offline(
        renderer,
        material_set,
        ibl,
        params,
        heightmap,
        settings=settings,
        water_mask=water_mask,
    )

    assert not np.array_equal(without_mask.frame.to_numpy(), with_mask.frame.to_numpy())
