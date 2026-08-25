# tests/test_terrain_tv4_material_variation.py
# TV4: Terrain material variation regression and acceptance coverage.
# Verifies the shared noise module, public material-noise controls, image output
# differences for snow/rock/wetness, and a bounded render-time budget.

from __future__ import annotations

import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import (
    MaterialLayerSettings,
    MaterialNoiseSettings,
    PomSettings,
    make_terrain_params_config,
)


TERRAIN_SHADER = Path(__file__).resolve().parents[1] / "src" / "shaders" / "terrain_pbr_pom.wgsl"
TERRAIN_NOISE_SHADER = Path(__file__).resolve().parents[1] / "src" / "shaders" / "terrain_noise.wgsl"


def test_shared_terrain_noise_module_exists() -> None:
    assert TERRAIN_NOISE_SHADER.exists(), f"Missing shared terrain noise module: {TERRAIN_NOISE_SHADER}"
    source = TERRAIN_NOISE_SHADER.read_text(encoding="utf-8")
    for fn_name in (
        "fn terrain_value_noise",
        "fn terrain_fbm",
        "fn terrain_ridged_fbm",
        "fn terrain_cellular_distance",
    ):
        assert fn_name in source, f"{fn_name} missing from shared terrain noise module"


def test_terrain_shader_uses_shared_noise_module() -> None:
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    assert '#include "terrain_noise.wgsl"' in source
    assert "fn hash31(" not in source
    assert "fn value_noise(" not in source
    assert "terrain_value_noise(" in source


def test_material_noise_settings_round_trip_defaults() -> None:
    variation = MaterialNoiseSettings()
    materials = MaterialLayerSettings(variation=variation)
    assert materials.variation.macro_scale == pytest.approx(3.5)
    assert materials.variation.detail_scale == pytest.approx(18.0)
    assert materials.variation.octaves == 4
    assert materials.variation.snow_macro_amplitude == pytest.approx(0.0)
    assert materials.variation.rock_detail_amplitude == pytest.approx(0.0)
    assert materials.variation.wetness_macro_amplitude == pytest.approx(0.0)


GPU_AVAILABLE = terrain_rendering_available()


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 164, 128]))


def _build_heightmap(size: int = 128) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ridge = 0.58 * np.exp(-((xx + 0.20) ** 2 * 7.5 + (yy - 0.10) ** 2 * 11.0))
    basin = -0.20 * np.exp(-((xx - 0.04) ** 2 * 18.0 + (yy + 0.10) ** 2 * 25.0))
    spur = 0.26 * np.exp(-((xx - 0.42) ** 2 * 24.0 + (yy + 0.28) ** 2 * 16.0))
    shelf = 0.17 * np.exp(-((xx + 0.48) ** 2 * 28.0 + (yy + 0.38) ** 2 * 20.0))
    slope = 0.24 * (1.0 - yy) + 0.11 * xx
    heightmap = ridge + basin + spur + shelf + slope
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#17351b"),
            (0.20, "#3a632d"),
            (0.48, "#69733d"),
            (0.68, "#8f7b53"),
            (0.86, "#c2b29a"),
            (1.0, "#f4f7fb"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _render_scene(
    renderer,
    material_set,
    ibl,
    heightmap: np.ndarray,
    overlay,
    materials: MaterialLayerSettings,
    *,
    size_px: tuple[int, int] = (224, 160),
) -> np.ndarray:
    params = make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=2.8,
        msaa_samples=1,
        z_scale=1.35,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=True,
        light_azimuth_deg=136.0,
        light_elevation_deg=20.0,
        sun_intensity=2.3,
        cam_radius=5.2,
        cam_phi_deg=140.0,
        cam_theta_deg=58.0,
        fov_y_deg=52.0,
        camera_mode="screen",
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        materials=materials,
    )
    native_params = f3d.TerrainRenderParams(params)
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=native_params,
        heightmap=heightmap,
        target=None,
    )
    return frame.to_numpy()


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a[..., :3].astype(np.float32) - b[..., :3].astype(np.float32))))


@pytest.fixture(scope="module")
def tv4_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("TV4 rendering tests require GPU-backed forge3d module")

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    overlay = _build_overlay()
    heightmap = _build_heightmap()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)

    try:
        yield renderer, material_set, ibl, heightmap, overlay
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not GPU_AVAILABLE, reason="TV4 rendering tests require GPU-backed forge3d module")
class TestTerrainMaterialVariationRendering:
    def test_zero_amplitude_preserves_baseline(self, tv4_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = tv4_render_env
        baseline_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.72,
            snow_altitude_blend=0.14,
            rock_enabled=True,
            rock_slope_min=36.0,
            rock_slope_blend=11.0,
            wetness_enabled=True,
            wetness_strength=0.40,
            wetness_slope_influence=0.85,
        )
        zero_amp_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.72,
            snow_altitude_blend=0.14,
            rock_enabled=True,
            rock_slope_min=36.0,
            rock_slope_blend=11.0,
            wetness_enabled=True,
            wetness_strength=0.40,
            wetness_slope_influence=0.85,
            variation=MaterialNoiseSettings(
                macro_scale=6.0,
                detail_scale=26.0,
                octaves=6,
            ),
        )

        baseline = _render_scene(renderer, material_set, ibl, heightmap, overlay, baseline_materials)
        zero_amp = _render_scene(renderer, material_set, ibl, heightmap, overlay, zero_amp_materials)

        max_diff = int(np.max(np.abs(baseline.astype(np.int16) - zero_amp.astype(np.int16))))
        assert max_diff <= 1, f"Zero-amplitude variation regressed baseline output by {max_diff} LSB"

    def test_snow_variation_changes_output(self, tv4_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = tv4_render_env
        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            MaterialLayerSettings(
                snow_enabled=True,
                snow_altitude_min=0.72,
                snow_altitude_blend=0.14,
            ),
        )
        varied = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            MaterialLayerSettings(
                snow_enabled=True,
                snow_altitude_min=0.72,
                snow_altitude_blend=0.14,
                variation=MaterialNoiseSettings(
                    macro_scale=4.5,
                    detail_scale=22.0,
                    octaves=5,
                    snow_macro_amplitude=0.26,
                    snow_detail_amplitude=0.12,
                ),
            ),
        )
        assert _mean_abs_diff(baseline, varied) > 0.35

    def test_rock_variation_changes_output(self, tv4_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = tv4_render_env
        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            MaterialLayerSettings(
                rock_enabled=True,
                rock_slope_min=36.0,
                rock_slope_blend=11.0,
            ),
        )
        varied = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            MaterialLayerSettings(
                rock_enabled=True,
                rock_slope_min=36.0,
                rock_slope_blend=11.0,
                variation=MaterialNoiseSettings(
                    macro_scale=3.2,
                    detail_scale=18.0,
                    octaves=4,
                    rock_macro_amplitude=0.24,
                    rock_detail_amplitude=0.18,
                ),
            ),
        )
        assert _mean_abs_diff(baseline, varied) > 0.25

    def test_wetness_variation_changes_output(self, tv4_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = tv4_render_env
        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            MaterialLayerSettings(
                wetness_enabled=True,
                wetness_strength=0.45,
                wetness_slope_influence=0.90,
            ),
        )
        varied = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            MaterialLayerSettings(
                wetness_enabled=True,
                wetness_strength=0.45,
                wetness_slope_influence=0.90,
                variation=MaterialNoiseSettings(
                    macro_scale=2.8,
                    detail_scale=16.0,
                    octaves=5,
                    wetness_macro_amplitude=0.30,
                    wetness_detail_amplitude=0.14,
                ),
            ),
        )
        assert _mean_abs_diff(baseline, varied) > 0.20

    def test_material_variation_perf_budget(self, tv4_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = tv4_render_env
        baseline_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.72,
            snow_altitude_blend=0.14,
            rock_enabled=True,
            rock_slope_min=36.0,
            rock_slope_blend=11.0,
            wetness_enabled=True,
            wetness_strength=0.40,
            wetness_slope_influence=0.85,
        )
        varied_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.72,
            snow_altitude_blend=0.14,
            rock_enabled=True,
            rock_slope_min=36.0,
            rock_slope_blend=11.0,
            wetness_enabled=True,
            wetness_strength=0.40,
            wetness_slope_influence=0.85,
            variation=MaterialNoiseSettings(
                macro_scale=4.2,
                detail_scale=20.0,
                octaves=5,
                snow_macro_amplitude=0.22,
                snow_detail_amplitude=0.10,
                rock_macro_amplitude=0.20,
                rock_detail_amplitude=0.15,
                wetness_macro_amplitude=0.18,
                wetness_detail_amplitude=0.10,
            ),
        )

        # Warm both paths so pipeline creation and IBL setup do not dominate the budget.
        _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            baseline_materials,
            size_px=(192, 128),
        )
        _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            varied_materials,
            size_px=(192, 128),
        )

        start = time.perf_counter()
        _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            baseline_materials,
            size_px=(192, 128),
        )
        baseline_s = time.perf_counter() - start

        start = time.perf_counter()
        _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            varied_materials,
            size_px=(192, 128),
        )
        varied_s = time.perf_counter() - start

        # TV4 budget: bounded overhead for the richer material-variation path after warm-up.
        assert varied_s <= max(baseline_s * 2.25, baseline_s + 0.75), (
            f"TV4 material variation exceeded budget: baseline={baseline_s:.3f}s varied={varied_s:.3f}s"
        )
