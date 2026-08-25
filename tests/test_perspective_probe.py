# tests/test_perspective_probe.py
# Perspective probe tests using the native renderer in mesh mode.
from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import (
    ClampSettings,
    IblSettings,
    LightSettings,
    LodSettings,
    PomSettings,
    SamplingSettings,
    ShadowSettings,
    TerrainRenderParams as TerrainRenderParamsConfig,
    TriplanarSettings,
)


pytestmark = pytest.mark.apple_metal_physical
def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    """Create a minimal valid HDR file for testing."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for _y in range(height):
            for _x in range(width):
                f.write(bytes([128, 128, 128, 128]))


def create_peak_heightmap(size: int = 128) -> np.ndarray:
    """Create a synthetic heightmap with a tall sharp peak for perspective testing."""
    x = np.linspace(-1, 1, size, dtype=np.float32)
    y = np.linspace(-1, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    heightmap = np.full((size, size), 0.1, dtype=np.float32)
    sigma = 0.15
    peak = 0.9 * np.exp(-(xx**2 + yy**2) / (sigma**2))
    heightmap += peak
    ridge = 0.3 * np.exp(-((xx - 0.5) ** 2 / 0.1 + yy**2 / 0.3))
    heightmap += ridge
    return heightmap.astype(np.float32)


def _build_probe_config(fov_deg: float, theta_deg: float, phi_deg: float = 135.0):
    """Build config for perspective probe with specific FOV, theta, and phi."""
    return TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        terrain_span=2.0,
        msaa_samples=1,
        z_scale=2.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=3.0,
        cam_phi_deg=float(phi_deg),
        cam_theta_deg=float(theta_deg),
        cam_gamma_deg=0.0,
        fov_y_deg=float(fov_deg),
        clip=(0.1, 100.0),
        camera_mode="mesh",
        debug_mode=41,  # NDC depth probe to expose FOV differences
        light=LightSettings("Directional", 45.0, 45.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 0.3, 0.0),
        shadows=ShadowSettings(
            False, "NONE", 512, 1, 100.0, 0.01, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        lod=LodSettings(0, 0.0, 0.0),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="colormap",
        colormap_strength=1.0,
    )


def _render_and_get_pixels(renderer, material_set, ibl, params, heightmap):
    """Render and return both hash and pixel array."""
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
    )
    arr = frame.to_numpy()
    h = hashlib.md5(arr.tobytes()).hexdigest()
    return h, arr


def pixel_diff_count(arr1: np.ndarray, arr2: np.ndarray, threshold: int = 1) -> int:
    """Count pixels that differ by more than threshold in any channel."""
    diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16))
    max_diff = np.max(diff, axis=2)
    return int(np.sum(max_diff > threshold))


def _prepare_renderer():
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)

    return renderer, material_set, ibl


def test_fov_affects_output():
    """FOV sweep must change NDC depth probe output."""
    renderer, material_set, ibl = _prepare_renderer()
    heightmap = create_peak_heightmap(128)

    theta = 45.0
    results = {}
    baseline_arr = None

    for fov in [30.0, 60.0, 90.0]:
        config = _build_probe_config(fov, theta)
        params = f3d.TerrainRenderParams(config)
        h, arr = _render_and_get_pixels(renderer, material_set, ibl, params, heightmap)

        if baseline_arr is None:
            baseline_arr = arr
            pixel_diff = 0
        else:
            pixel_diff = pixel_diff_count(arr, baseline_arr)

        results[fov] = {"hash": h, "pixel_diff": pixel_diff}

    hashes = [r["hash"] for r in results.values()]
    assert len(set(hashes)) == 3, "FOV sweep should produce unique hashes"
    for fov, data in results.items():
        if fov != 30.0:
            assert data["pixel_diff"] > 0, f"FOV={fov} expected non-zero pixel diff"


def test_theta_affects_output():
    """Camera elevation should change projection probe output."""
    renderer, material_set, ibl = _prepare_renderer()
    heightmap = create_peak_heightmap(128)

    fov = 55.0
    results = {}
    baseline_arr = None

    for theta in [15.0, 45.0, 75.0]:
        config = _build_probe_config(fov, theta)
        params = f3d.TerrainRenderParams(config)
        h, arr = _render_and_get_pixels(renderer, material_set, ibl, params, heightmap)

        if baseline_arr is None:
            baseline_arr = arr
            pixel_diff = 0
        else:
            pixel_diff = pixel_diff_count(arr, baseline_arr)

        results[theta] = {"hash": h, "pixel_diff": pixel_diff}

    hashes = [r["hash"] for r in results.values()]
    assert len(set(hashes)) == 3, "Theta sweep should produce unique hashes"
    assert results[75.0]["pixel_diff"] > 0
    assert results[45.0]["pixel_diff"] > 0


def test_phi_affects_output():
    """Camera azimuth should rotate the projection probe."""
    renderer, material_set, ibl = _prepare_renderer()
    heightmap = create_peak_heightmap(128)

    fov = 55.0
    theta = 45.0
    results = {}
    baseline_arr = None

    for phi in [0.0, 90.0]:
        config = _build_probe_config(fov, theta, phi_deg=phi)
        params = f3d.TerrainRenderParams(config)
        h, arr = _render_and_get_pixels(renderer, material_set, ibl, params, heightmap)

        if baseline_arr is None:
            baseline_arr = arr
            pixel_diff = 0
        else:
            pixel_diff = pixel_diff_count(arr, baseline_arr)

        results[phi] = {"hash": h, "pixel_diff": pixel_diff}

    assert len(set(r["hash"] for r in results.values())) == 2, "Phi change should affect hash"
    assert results[90.0]["pixel_diff"] > 0, "Phi change should alter projection probe"
