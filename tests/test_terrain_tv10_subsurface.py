# tests/test_terrain_tv10_subsurface.py
# TV10: Terrain subsurface regression and acceptance coverage.
# Verifies shader wiring, per-layer controls, zero-strength neutrality, and
# visible output differences across two terrain lighting setups.

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import (
    MaterialLayerSettings,
    PomSettings,
    make_terrain_params_config,
)


TERRAIN_SHADER = Path(__file__).resolve().parents[1] / "src" / "shaders" / "terrain_pbr_pom.wgsl"


def test_terrain_shader_declares_tv10_subsurface_support() -> None:
    source = TERRAIN_SHADER.read_text(encoding="utf-8")
    for token in (
        "snow_sss_tint:",
        "rock_sss_tint:",
        "wetness_sss_tint:",
        "resolve_terrain_subsurface(",
        "evaluate_terrain_subsurface(",
    ):
        assert token in source, f"{token} missing from terrain TV10 shader path"


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
                handle.write(bytes([r, g, 180, 128]))


def _build_heightmap(size: int = 144) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    massif = 0.64 * np.exp(-((xx + 0.18) ** 2 * 7.5 + (yy - 0.06) ** 2 * 11.5))
    cirque = 0.30 * np.exp(-((xx - 0.24) ** 2 * 20.0 + (yy + 0.18) ** 2 * 18.0))
    ridge = 0.22 * np.exp(-((xx - 0.48) ** 2 * 42.0 + (yy + 0.28) ** 2 * 22.0))
    basin = -0.18 * np.exp(-((xx + 0.06) ** 2 * 24.0 + (yy + 0.02) ** 2 * 24.0))
    slope = 0.26 * (1.0 - yy) + 0.10 * xx
    heightmap = massif + cirque + ridge + basin + slope
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#1b381d"),
            (0.22, "#416a30"),
            (0.50, "#7d7a4b"),
            (0.72, "#b6a98d"),
            (1.0, "#f4f7fb"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _base_materials() -> MaterialLayerSettings:
    return MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=0.78,
        snow_altitude_blend=0.24,
        snow_slope_max=58.0,
        snow_slope_blend=18.0,
        rock_enabled=True,
        rock_slope_min=38.0,
        rock_slope_blend=10.0,
        wetness_enabled=True,
        wetness_strength=0.18,
        wetness_slope_influence=0.45,
        snow_subsurface_strength=0.0,
        rock_subsurface_strength=0.0,
        wetness_subsurface_strength=0.0,
    )


def _tv10_materials() -> MaterialLayerSettings:
    return MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=0.78,
        snow_altitude_blend=0.24,
        snow_slope_max=58.0,
        snow_slope_blend=18.0,
        rock_enabled=True,
        rock_slope_min=38.0,
        rock_slope_blend=10.0,
        wetness_enabled=True,
        wetness_strength=0.18,
        wetness_slope_influence=0.45,
        snow_subsurface_strength=0.58,
        snow_subsurface_tint=(0.72, 0.85, 0.98),
        rock_subsurface_strength=0.04,
        rock_subsurface_tint=(0.45, 0.38, 0.30),
        wetness_subsurface_strength=0.16,
        wetness_subsurface_tint=(0.38, 0.27, 0.18),
    )


SCENE_A = dict(
    light_azimuth_deg=132.0,
    light_elevation_deg=11.0,
    sun_intensity=2.6,
    cam_radius=4.2,
    cam_phi_deg=138.0,
    cam_theta_deg=42.0,
    fov_y_deg=42.0,
    size_px=(240, 160),
)

SCENE_B = dict(
    light_azimuth_deg=214.0,
    light_elevation_deg=9.0,
    sun_intensity=2.8,
    cam_radius=4.5,
    cam_phi_deg=218.0,
    cam_theta_deg=38.0,
    fov_y_deg=40.0,
    size_px=(240, 160),
)


def _render_scene(
    renderer,
    material_set,
    ibl,
    heightmap: np.ndarray,
    overlay,
    materials: MaterialLayerSettings,
    *,
    scene: dict,
) -> np.ndarray:
    params = make_terrain_params_config(
        size_px=scene["size_px"],
        render_scale=1.0,
        terrain_span=2.9,
        msaa_samples=1,
        z_scale=1.45,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="mix",
        colormap_strength=0.25,
        ibl_enabled=True,
        light_azimuth_deg=scene["light_azimuth_deg"],
        light_elevation_deg=scene["light_elevation_deg"],
        sun_intensity=scene["sun_intensity"],
        cam_radius=scene["cam_radius"],
        cam_phi_deg=scene["cam_phi_deg"],
        cam_theta_deg=scene["cam_theta_deg"],
        fov_y_deg=scene["fov_y_deg"],
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


def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a[..., :3].astype(np.float32) - b[..., :3].astype(np.float32))
    return float(np.mean(diff)), float(np.percentile(diff, 99.0))


@pytest.fixture(scope="module")
def tv10_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("TV10 rendering tests require GPU-backed forge3d module")

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
@pytest.mark.skipif(not GPU_AVAILABLE, reason="TV10 rendering tests require GPU-backed forge3d module")
class TestTerrainSubsurfaceRendering:
    def test_zero_strength_ignores_subsurface_colors(self, tv10_render_env) -> None:
        renderer, material_set, ibl, heightmap, overlay = tv10_render_env
        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            _base_materials(),
            scene=SCENE_A,
        )
        zero_strength_custom = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            MaterialLayerSettings(
                snow_enabled=True,
                snow_altitude_min=0.78,
                snow_altitude_blend=0.24,
                snow_slope_max=58.0,
                snow_slope_blend=18.0,
                rock_enabled=True,
                rock_slope_min=38.0,
                rock_slope_blend=10.0,
                wetness_enabled=True,
                wetness_strength=0.18,
                wetness_slope_influence=0.45,
                snow_subsurface_strength=0.0,
                snow_subsurface_tint=(0.55, 0.12, 0.10),
                rock_subsurface_strength=0.0,
                rock_subsurface_tint=(0.15, 0.55, 0.22),
                wetness_subsurface_strength=0.0,
                wetness_subsurface_tint=(0.18, 0.20, 0.82),
            ),
            scene=SCENE_A,
        )
        max_diff = int(np.max(np.abs(baseline.astype(np.int16) - zero_strength_custom.astype(np.int16))))
        assert max_diff <= 1, f"Zero-strength TV10 path regressed baseline output by {max_diff} LSB"

    @pytest.mark.parametrize("scene", [SCENE_A, SCENE_B])
    def test_snow_subsurface_changes_output(self, tv10_render_env, scene: dict) -> None:
        renderer, material_set, ibl, heightmap, overlay = tv10_render_env
        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            _base_materials(),
            scene=scene,
        )
        subsurface = _render_scene(
            renderer,
            material_set,
            ibl,
            heightmap,
            overlay,
            _tv10_materials(),
            scene=scene,
        )
        mean_abs, peak_p99 = _diff_stats(baseline, subsurface)
        assert mean_abs > 0.10, f"TV10 mean RGB delta too small: {mean_abs:.4f}"
        assert peak_p99 >= 3.0, f"TV10 highlight delta too small: {peak_p99:.4f}"
