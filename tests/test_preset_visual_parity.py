from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import forge3d as f3d
from forge3d import terrain_demo

from _terrain_runtime import terrain_rendering_available
from tests._ssim import ssim


pytestmark = pytest.mark.nvidia_vulkan


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "presets"
SSIM_MIN = 0.995
MEAN_ABS_MAX = 2.0


def _heightmap(size: int = 128) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    peak = 700.0 * np.exp(-4.0 * (xx * xx + yy * yy))
    ridges = 90.0 * np.sin(10.0 * xx) * np.cos(8.0 * yy)
    return (1200.0 + peak + ridges).astype(np.float32)


def _demo_args(dem_path: Path, output_path: Path, size: int) -> SimpleNamespace:
    return SimpleNamespace(
        preset="rainier_showcase",
        dem=dem_path,
        output=output_path,
        overwrite=True,
        window=False,
        viewer=False,
        size=(size, size),
        render_scale=1.0,
        msaa=1,
        hdr=terrain_demo.DEFAULT_HDR,
        ibl_res=64,
        ibl_cache=None,
        colormap="terrain",
        colormap_domain=None,
        colormap_interpolate=False,
        colormap_size=256,
        albedo_mode="mix",
        colormap_strength=0.5,
        normal_strength=1.0,
        pom_disabled=False,
        camera_mode=terrain_demo.DEFAULT_CAMERA_MODE,
        cam_radius=terrain_demo.DEFAULT_CAM_RADIUS,
        cam_phi=terrain_demo.DEFAULT_CAM_PHI,
        cam_theta=terrain_demo.DEFAULT_CAM_THETA,
        cam_fov=terrain_demo.DEFAULT_CAM_FOV,
        sun_azimuth=None,
        sun_elevation=None,
        sun_intensity=None,
        sun_color=None,
        ibl_intensity=1.0,
        z_scale=2.0,
        exposure=1.0,
        brdf=None,
        light=None,
        shadow_technique=None,
        shadows=None,
        shadow_map_res=None,
        cascades=None,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi=None,
        sky=None,
        volumetric=None,
        oit=None,
        taa=False,
        taa_history_weight=None,
        height_curve_mode="linear",
        height_curve_strength=0.0,
        height_curve_power=1.0,
        height_curve_lut=None,
        fog_density=0.0,
        fog_height_falloff=0.0,
        fog_inscatter="1.0,1.0,1.0",
        water_reflections=False,
        reflection_intensity=0.8,
        reflection_fresnel_power=5.0,
        reflection_wave_strength=0.02,
        reflection_shore_atten=0.3,
        reflection_plane_height=0.0,
        render=None,
        detail_normals=None,
        detail_strength=0.0,
        detail_sigma_px=3.0,
        lambert_contrast=0.0,
        colormap_srgb=False,
        output_srgb_eotf=False,
        debug_mode=0,
        debug_lights=False,
        unsharp_strength=0.0,
        _explicit_cli_args=set(),
    )


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)[..., :3]


def _assert_close(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    score = ssim(ref, test, data_range=255.0)
    mean_abs = float(np.mean(np.abs(test.astype(np.int16) - ref.astype(np.int16))))
    assert score >= SSIM_MIN
    assert mean_abs <= MEAN_ABS_MAX
    return score, mean_abs


def test_rainier_showcase_mapscene_matches_terrain_demo_and_golden(tmp_path: Path) -> None:
    size = 128
    dem_path = tmp_path / "rainier-parity-dem.npy"
    demo_path = tmp_path / "terrain_demo_rainier_showcase.png"
    mapscene_path = tmp_path / "mapscene_rainier_showcase.png"
    np.save(dem_path, _heightmap(size))

    terrain_demo.run(_demo_args(dem_path, demo_path, size))
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=dem_path,
            crs="EPSG:32610",
            metadata={"width": size, "height": size, "resolution": [1.0, 1.0], "source_id": "preset-parity-dem"},
            elevation_sampling_available=True,
        ),
        lighting=f3d.LightingPreset(name="rainier_showcase"),
        output=f3d.OutputSpec(width=size, height=size, format="png", path=str(mapscene_path)),
    )
    scene.render()

    demo = _load_rgb(demo_path)
    mapscene = _load_rgb(mapscene_path)
    golden = _load_rgb(GOLDEN_DIR / "rainier_showcase_mapscene.png")

    _assert_close(golden, mapscene)
    _assert_close(demo, mapscene)
    assert scene.last_render_backend == "gpu_terrain"
