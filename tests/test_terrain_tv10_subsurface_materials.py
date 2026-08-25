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


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#23371f"),
            (0.22, "#476438"),
            (0.46, "#77805a"),
            (0.70, "#b1a07c"),
            (0.86, "#d8d4ce"),
            (1.0, "#f6f9ff"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _build_snow_heightmap(size: int = 160) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ridge = 0.52 * np.exp(-((xx + 0.18) ** 2 * 7.0 + (yy - 0.06) ** 2 * 9.5))
    crest = 0.30 * np.exp(-((xx - 0.26) ** 2 * 16.0 + (yy + 0.20) ** 2 * 22.0))
    bowl = -0.18 * np.exp(-((xx - 0.02) ** 2 * 20.0 + (yy + 0.05) ** 2 * 28.0))
    shelf = 0.14 * np.exp(-((xx + 0.46) ** 2 * 24.0 + (yy + 0.34) ** 2 * 18.0))
    slope = 0.28 * (1.0 - yy) + 0.10 * xx
    heightmap = ridge + crest + bowl + shelf + slope
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_glacier_heightmap(size: int = 160) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    glacier_body = 0.42 * np.exp(-((xx + 0.08) ** 2 * 4.0 + (yy + 0.18) ** 2 * 8.0))
    moraine = 0.24 * np.exp(-((xx - 0.36) ** 2 * 18.0 + (yy - 0.04) ** 2 * 26.0))
    basin = -0.16 * np.exp(-((xx + 0.10) ** 2 * 22.0 + (yy - 0.18) ** 2 * 16.0))
    tongue = 0.20 * np.exp(-((xx + 0.34) ** 2 * 11.0 + (yy - 0.42) ** 2 * 6.5))
    slope = 0.20 * (1.0 - yy) - 0.05 * xx
    heightmap = glacier_body + moraine + basin + tongue + slope
    heightmap -= float(heightmap.min())
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _render_scene(
    renderer,
    material_set,
    ibl,
    overlay,
    heightmap: np.ndarray,
    materials: MaterialLayerSettings,
    *,
    light_azimuth_deg: float,
    light_elevation_deg: float,
    sun_intensity: float,
    cam_phi_deg: float,
    cam_theta_deg: float,
    size_px: tuple[int, int] = (224, 160),
) -> np.ndarray:
    params = make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=3.2,
        msaa_samples=1,
        z_scale=1.45,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=True,
        light_azimuth_deg=light_azimuth_deg,
        light_elevation_deg=light_elevation_deg,
        sun_intensity=sun_intensity,
        cam_radius=5.6,
        cam_phi_deg=cam_phi_deg,
        cam_theta_deg=cam_theta_deg,
        fov_y_deg=50.0,
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


def _mean_luma(image: np.ndarray) -> float:
    rgb = image[..., :3].astype(np.float32) / 255.0
    return float(np.mean(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]))


@pytest.fixture(scope="module")
def tv10_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("TV10 rendering tests require GPU-backed forge3d module")

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    overlay = _build_overlay()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)

    try:
        yield renderer, material_set, ibl, overlay
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not GPU_AVAILABLE, reason="TV10 rendering tests require GPU-backed forge3d module")
class TestTerrainSubsurfaceMaterials:
    def test_zero_strength_preserves_baseline(self, tv10_render_env) -> None:
        renderer, material_set, ibl, overlay = tv10_render_env
        heightmap = _build_snow_heightmap()
        baseline_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.60,
            snow_altitude_blend=0.18,
            snow_slope_max=58.0,
            snow_slope_blend=18.0,
        )
        zero_strength_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.60,
            snow_altitude_blend=0.18,
            snow_slope_max=58.0,
            snow_slope_blend=18.0,
            snow_subsurface_strength=0.0,
            snow_subsurface_tint=(0.70, 0.84, 1.0),
        )

        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            baseline_materials,
            light_azimuth_deg=132.0,
            light_elevation_deg=11.0,
            sun_intensity=2.8,
            cam_phi_deg=306.0,
            cam_theta_deg=58.0,
        )
        zero_strength = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            zero_strength_materials,
            light_azimuth_deg=132.0,
            light_elevation_deg=11.0,
            sun_intensity=2.8,
            cam_phi_deg=306.0,
            cam_theta_deg=58.0,
        )

        max_diff = int(np.max(np.abs(baseline.astype(np.int16) - zero_strength.astype(np.int16))))
        assert max_diff <= 1, f"Zero-strength terrain SSS regressed baseline output by {max_diff} LSB"

    def test_snow_subsurface_changes_output(self, tv10_render_env) -> None:
        renderer, material_set, ibl, overlay = tv10_render_env
        heightmap = _build_snow_heightmap()
        baseline_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.60,
            snow_altitude_blend=0.18,
            snow_slope_max=58.0,
            snow_slope_blend=18.0,
        )
        sss_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.60,
            snow_altitude_blend=0.18,
            snow_slope_max=58.0,
            snow_slope_blend=18.0,
            snow_subsurface_strength=0.82,
            snow_subsurface_tint=(0.78, 0.88, 1.0),
        )

        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            baseline_materials,
            light_azimuth_deg=132.0,
            light_elevation_deg=11.0,
            sun_intensity=2.8,
            cam_phi_deg=306.0,
            cam_theta_deg=58.0,
        )
        subsurface = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            sss_materials,
            light_azimuth_deg=132.0,
            light_elevation_deg=11.0,
            sun_intensity=2.8,
            cam_phi_deg=306.0,
            cam_theta_deg=58.0,
        )

        assert _mean_abs_diff(baseline, subsurface) > 0.65
        assert _mean_luma(subsurface) > _mean_luma(baseline) + 0.005

    def test_glacial_ice_subsurface_cools_highlights(self, tv10_render_env) -> None:
        renderer, material_set, ibl, overlay = tv10_render_env
        heightmap = _build_glacier_heightmap()
        baseline_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.34,
            snow_altitude_blend=0.26,
            snow_slope_max=64.0,
            snow_slope_blend=22.0,
            snow_color=(0.88, 0.92, 0.96),
        )
        icy_materials = MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=0.34,
            snow_altitude_blend=0.26,
            snow_slope_max=64.0,
            snow_slope_blend=22.0,
            snow_color=(0.88, 0.92, 0.96),
            snow_subsurface_strength=0.92,
            snow_subsurface_tint=(0.58, 0.80, 1.0),
        )

        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            baseline_materials,
            light_azimuth_deg=42.0,
            light_elevation_deg=8.0,
            sun_intensity=3.1,
            cam_phi_deg=222.0,
            cam_theta_deg=54.0,
        )
        icy = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            icy_materials,
            light_azimuth_deg=42.0,
            light_elevation_deg=8.0,
            sun_intensity=3.1,
            cam_phi_deg=222.0,
            cam_theta_deg=54.0,
        )

        baseline_rgb = baseline[..., :3].astype(np.float32)
        icy_rgb = icy[..., :3].astype(np.float32)
        snow_mask = baseline_rgb.mean(axis=2) >= np.percentile(baseline_rgb.mean(axis=2), 65.0)
        baseline_cool_bias = float(np.mean((baseline_rgb[..., 2] - baseline_rgb[..., 0])[snow_mask]))
        icy_cool_bias = float(np.mean((icy_rgb[..., 2] - icy_rgb[..., 0])[snow_mask]))

        assert _mean_abs_diff(baseline, icy) > 0.80
        assert icy_cool_bias > baseline_cool_bias + 0.75

    def test_wetness_subsurface_changes_output_independently(self, tv10_render_env) -> None:
        renderer, material_set, ibl, overlay = tv10_render_env
        heightmap = _build_snow_heightmap()
        baseline_materials = MaterialLayerSettings(
            wetness_enabled=True,
            wetness_strength=0.60,
            wetness_slope_influence=1.0,
        )
        sss_materials = MaterialLayerSettings(
            wetness_enabled=True,
            wetness_strength=0.60,
            wetness_slope_influence=1.0,
            wetness_subsurface_strength=0.60,
            wetness_subsurface_tint=(0.72, 0.80, 0.88),
        )

        baseline = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            baseline_materials,
            light_azimuth_deg=118.0,
            light_elevation_deg=10.0,
            sun_intensity=2.6,
            cam_phi_deg=300.0,
            cam_theta_deg=56.0,
        )
        subsurface = _render_scene(
            renderer,
            material_set,
            ibl,
            overlay,
            heightmap,
            sss_materials,
            light_azimuth_deg=118.0,
            light_elevation_deg=10.0,
            sun_intensity=2.6,
            cam_phi_deg=300.0,
            cam_theta_deg=56.0,
        )

        assert _mean_abs_diff(baseline, subsurface) > 0.80
