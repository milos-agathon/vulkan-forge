# tests/test_cam_phi_wiring.py
# Deterministic test that fails if cam_phi is ignored
# Exists to guard against camera azimuth parameter wiring regressions
# RELEVANT FILES: src/terrain/renderer.rs, src/terrain/render_params.rs, python/forge3d/terrain_params.py
from __future__ import annotations

import hashlib
import tempfile
import os

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


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    """Create a minimal valid HDR file for testing."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                b = 128
                e = 128
                f.write(bytes([r, g, b, e]))


def _build_config_with_phi(phi_deg: float, overlay):
    """Build a TerrainRenderParamsConfig with a specific cam_phi_deg."""
    return TerrainRenderParamsConfig(
        size_px=(128, 128),  # Small for fast test
        render_scale=1.0,
        terrain_span=2.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=4.0,
        cam_phi_deg=phi_deg,
        cam_theta_deg=45.0,  # Fixed elevation
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 250.0),
        light=LightSettings("Directional", 135.0, 35.0, 2.5, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            True, "HARD", 512, 1, 250.0, 0.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),  # Disable POM for speed
        lod=LodSettings(0, 0.0, 0.0),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="colormap",
        colormap_strength=1.0,
    )


def _render_and_hash(renderer, material_set, ibl, params, heightmap) -> str:
    """Render and return md5 hash of the frame."""
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
    )
    arr = frame.to_numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()


@pytest.mark.apple_metal_contract
@pytest.mark.apple_metal_physical
def test_cam_phi_changes_output():
    """Test that cam_phi parameter actually changes the rendered output.
    
    This test renders the same scene with phi=0 and phi=180 (opposite sides)
    and asserts the md5 hashes differ. If phi is ignored/overwritten somewhere
    in the pipeline, this test will fail.
    
    Uses an asymmetric heightmap (gradient) so rotation is visually obvious.
    """
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    
    # Create asymmetric colormap to make rotation visible
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#0000ff"), (0.5, "#00ff00"), (1.0, "#ff0000")],
        domain=(0.0, 1.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)
    
    # Create asymmetric heightmap (gradient from corner to corner)
    # This ensures rotation changes the view significantly
    size = 64
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heightmap = (xx + yy) / 2.0  # Diagonal gradient
    
    # Create test HDR
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)
    
    # Render with phi=0
    config_phi0 = _build_config_with_phi(0.0, overlay)
    params_phi0 = f3d.TerrainRenderParams(config_phi0)
    hash_phi0 = _render_and_hash(renderer, material_set, ibl, params_phi0, heightmap)
    
    # Render with phi=180 (opposite side)
    config_phi180 = _build_config_with_phi(180.0, overlay)
    params_phi180 = f3d.TerrainRenderParams(config_phi180)
    hash_phi180 = _render_and_hash(renderer, material_set, ibl, params_phi180, heightmap)
    
    # Assert hashes differ - if they're equal, phi is being ignored
    assert hash_phi0 != hash_phi180, (
        f"cam_phi has no effect! phi=0 and phi=180 produced identical output.\n"
        f"  phi=0:   {hash_phi0}\n"
        f"  phi=180: {hash_phi180}\n"
        f"This indicates a wiring bug in the cam_phi parameter path."
    )


@pytest.mark.apple_metal_contract
@pytest.mark.apple_metal_physical
def test_cam_phi_four_quadrants():
    """Test that cam_phi works for all four quadrants (0, 90, 180, 270).
    
    All four renders should produce different outputs when viewing an
    asymmetric terrain from different angles.
    """
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#000000"), (1.0, "#ffffff")],
        domain=(0.0, 1.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)
    
    # Asymmetric heightmap
    size = 64
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heightmap = xx * 0.7 + yy * 0.3  # Asymmetric gradient
    
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)
    
    hashes = {}
    for phi in [0.0, 90.0, 180.0, 270.0]:
        config = _build_config_with_phi(phi, overlay)
        params = f3d.TerrainRenderParams(config)
        hashes[phi] = _render_and_hash(renderer, material_set, ibl, params, heightmap)
    
    # All four hashes should be unique
    unique_hashes = set(hashes.values())
    assert len(unique_hashes) == 4, (
        f"cam_phi does not produce 4 unique outputs for 4 quadrants!\n"
        f"  phi=0:   {hashes[0.0]}\n"
        f"  phi=90:  {hashes[90.0]}\n"
        f"  phi=180: {hashes[180.0]}\n"
        f"  phi=270: {hashes[270.0]}\n"
        f"Unique hashes: {len(unique_hashes)}"
    )
