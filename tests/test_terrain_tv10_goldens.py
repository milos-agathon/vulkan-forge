from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.terrain_params import (
    MaterialLayerSettings,
    PomSettings,
    make_terrain_params_config,
)

from tests._golden_variants import (
    assert_nvidia_vulkan_golden_adapter,
    nvidia_vulkan_golden_selected,
    selected_golden_path,
)
from tests._ssim import ssim


if os.environ.get("FORGE3D_RUN_TERRAIN_GOLDENS") != "1":
    pytest.skip("TV10 golden tests run only in the dedicated GPU lane", allow_module_level=True)

if not f3d.has_gpu() or not all(
    hasattr(f3d, name)
    for name in ("TerrainRenderer", "TerrainRenderParams", "OverlayLayer", "MaterialSet", "IBL", "Session")
):
    pytest.skip("TV10 golden tests require GPU-backed native module", allow_module_level=True)


GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "terrain"
UPDATE_GOLDENS = os.environ.get("FORGE3D_UPDATE_TERRAIN_GOLDENS") == "1"
ARTIFACT_DIR = (
    Path(os.environ["FORGE3D_TERRAIN_GOLDEN_ARTIFACT_DIR"])
    if os.environ.get("FORGE3D_TERRAIN_GOLDEN_ARTIFACT_DIR")
    else None
)


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


def _zero_subsurface_materials() -> MaterialLayerSettings:
    common = dict(
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
    )
    defaults = MaterialLayerSettings()
    if hasattr(defaults, "snow_subsurface_strength"):
        return MaterialLayerSettings(
            **common,
            snow_subsurface_strength=0.0,
            rock_subsurface_strength=0.0,
            wetness_subsurface_strength=0.0,
        )
    return MaterialLayerSettings(**common)


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


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(path, image)


def _golden_path(scene_name: str) -> Path:
    """Use an explicitly selected backend baseline without cross-fallback."""
    return selected_golden_path(
        GOLDEN_DIR,
        scene_name,
        "FORGE3D_TERRAIN_GOLDEN_VARIANT",
        implicit_metal=True,
    )


def _write_failure_artifacts(scene_name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    diff = np.abs(actual[..., :3].astype(np.int16) - expected[..., :3].astype(np.int16)).astype(
        np.uint8
    )
    diff_rgba = np.concatenate(
        [diff, np.full(diff.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1
    )
    _save_png(ARTIFACT_DIR / f"{scene_name}_actual.png", actual)
    _save_png(ARTIFACT_DIR / f"{scene_name}_expected.png", expected)
    _save_png(ARTIFACT_DIR / f"{scene_name}_diff.png", diff_rgba)


def _assert_matches_golden(scene_name: str, actual: np.ndarray) -> None:
    golden_path = _golden_path(scene_name)
    if UPDATE_GOLDENS:
        _save_png(golden_path, actual)
        return

    assert golden_path.exists(), (
        f"Missing TV10 golden {golden_path}. "
        "Regenerate with FORGE3D_UPDATE_TERRAIN_GOLDENS=1."
    )

    expected = f3d.png_to_numpy(golden_path)
    assert actual.shape == expected.shape

    mean_abs = float(
        np.mean(np.abs(actual[..., :3].astype(np.float32) - expected[..., :3].astype(np.float32)))
    )
    score = ssim(actual[..., :3], expected[..., :3], data_range=255.0)

    if score < 0.995 or mean_abs > 2.0:
        _write_failure_artifacts(scene_name, actual, expected)

    assert score >= 0.995, f"{scene_name} SSIM too low: {score:.6f}"
    assert mean_abs <= 2.0, f"{scene_name} mean absolute difference too high: {mean_abs:.4f}"


@pytest.fixture(scope="module")
def tv10_golden_env():
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    if nvidia_vulkan_golden_selected("FORGE3D_TERRAIN_GOLDEN_VARIANT"):
        adapter_probe = f3d.device_probe("vulkan")
        assert_nvidia_vulkan_golden_adapter(
            "FORGE3D_TERRAIN_GOLDEN_VARIANT", adapter_probe
        )
        if ARTIFACT_DIR is not None:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            (ARTIFACT_DIR / "tv10-render-adapter.json").write_text(
                json.dumps(adapter_probe, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
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


@pytest.mark.parametrize(
    ("scene_name", "scene", "mode"),
    [
        ("terrain_tv10_zero_sss", SCENE_A, "zero"),
        ("terrain_tv10_scene_a_sss", SCENE_A, "sss"),
        ("terrain_tv10_scene_b_sss", SCENE_B, "sss"),
    ],
)
def test_tv10_goldens(tv10_golden_env, scene_name: str, scene: dict, mode: str) -> None:
    renderer, material_set, ibl, heightmap, overlay = tv10_golden_env
    if mode == "zero":
        materials = _zero_subsurface_materials()
    elif mode == "sss":
        materials = _tv10_materials()
    else:
        raise AssertionError(f"unexpected mode {mode}")

    actual = _render_scene(renderer, material_set, ibl, heightmap, overlay, materials, scene=scene)
    _assert_matches_golden(scene_name, actual)
