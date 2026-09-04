# tests/test_terrain_demo.py
# Terrain demo integration test validating synthetic DEM renders
# Exists to guard PBR terrain output and artifact-free PNG saves
# RELEVANT FILES: src/terrain_renderer.rs, src/material_set.rs, src/ibl_wrapper.rs, tools/validate_rows.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path

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


# ============================================================================
# P0-09: CLI Integration Smoke Test (CPU-only, no rendering)
# ============================================================================

def test_terrain_demo_build_renderer_config() -> None:
    """Smoke test for _build_renderer_config() parsing CLI flags.
    
    Tests the terrain_demo CLI flag parsing without requiring GPU or rendering.
    Validates that _build_renderer_config() correctly translates argparse flags
    into a normalized RendererConfig.
    """
    # Import terrain_demo functions
    import sys
    from pathlib import Path
    
    # Add examples to path to import terrain_demo
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    
    try:
        from terrain_demo import _build_renderer_config
        import argparse
    finally:
        sys.path.pop(0)
    
    # Create mock argparse.Namespace with various CLI flags
    args = argparse.Namespace(
        light=["type=directional,dir=0.2,0.8,-0.55,intensity=8"],
        exposure=1.5,
        brdf="cooktorrance-ggx",
        shadows="pcf",
        shadow_map_res=2048,
        cascades=3,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi="ibl,ssao",
        sky="hosek-wilkie",
        hdr=str(Path("assets/snow_field_4k.hdr")),
        volumetric=None,
        preset=None,
    )
    
    # Build config (no rendering, CPU-only)
    config = _build_renderer_config(args)
    
    # Assert config is a RendererConfig instance
    from forge3d.config import RendererConfig
    assert isinstance(config, RendererConfig)
    
    # Validate config structure
    config.validate()
    
    # Get dict representation for assertions
    config_dict = config.to_dict()
    
    # Assert lighting configuration
    assert len(config_dict["lighting"]["lights"]) == 1
    light = config_dict["lighting"]["lights"][0]
    assert light["type"] == "directional"
    assert light["intensity"] == pytest.approx(8.0)
    assert config_dict["lighting"]["exposure"] == pytest.approx(1.5)
    
    # Assert shading configuration
    assert config_dict["shading"]["brdf"] == "cooktorrance-ggx"
    
    # Assert shadow configuration
    assert config_dict["shadows"]["technique"] == "pcf"
    assert config_dict["shadows"]["map_size"] == 2048
    assert config_dict["shadows"]["cascades"] == 3
    
    # Assert GI configuration
    assert "ibl" in config_dict["gi"]["modes"]
    assert "ssao" in config_dict["gi"]["modes"]
    
    # Assert atmosphere configuration
    assert config_dict["atmosphere"]["sky"] == "hosek-wilkie"


def test_terrain_demo_build_renderer_config_with_preset() -> None:
    """Test _build_renderer_config() with preset override.
    
    Validates that CLI flags correctly override preset values.
    """
    import sys
    from pathlib import Path
    
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    
    try:
        from terrain_demo import _build_renderer_config
        import argparse
    finally:
        sys.path.pop(0)
    
    # Test with preset + overrides
    args = argparse.Namespace(
        light=[],
        exposure=1.0,
        brdf="toon",
        shadows="hard",
        shadow_map_res=None,
        cascades=2,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi=None,
        sky=None,
        hdr=None,
        volumetric=None,
        preset="outdoor_sun",  # Apply preset
    )
    
    # Build config
    config = _build_renderer_config(args)
    config.validate()
    config_dict = config.to_dict()
    
    # Overrides should take precedence over preset
    assert config_dict["shading"]["brdf"] == "toon"
    assert config_dict["shadows"]["technique"] == "hard"
    assert config_dict["shadows"]["cascades"] == 2


def test_terrain_demo_build_renderer_config_minimal() -> None:
    """Test _build_renderer_config() with minimal flags (all defaults).
    
    Validates that default config is created when no flags are provided.
    """
    import sys
    from pathlib import Path
    
    examples_path = Path(__file__).parent.parent / "examples"
    sys.path.insert(0, str(examples_path))
    
    try:
        from terrain_demo import _build_renderer_config
        import argparse
    finally:
        sys.path.pop(0)
    
    # Minimal args (all None/empty/default)
    args = argparse.Namespace(
        light=[],
        exposure=1.0,
        brdf=None,
        shadows=None,
        shadow_map_res=None,
        cascades=None,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi=None,
        sky=None,
        hdr=None,
        volumetric=None,
        preset=None,
    )
    
    # Build config
    config = _build_renderer_config(args)
    config.validate()
    config_dict = config.to_dict()
    
    # Should have defaults
    assert config_dict["shading"]["brdf"] == "cooktorrance-ggx"  # Default BRDF
    assert config_dict["shadows"]["technique"] == "pcf"  # Default shadow technique
    assert config_dict["lighting"]["exposure"] == pytest.approx(1.0)


# ============================================================================
# GPU-dependent tests below (will be skipped in CPU-only CI)
# ============================================================================

if not terrain_rendering_available():
    pytest.skip("Terrain demo requires a terrain-capable hardware-backed forge3d runtime", allow_module_level=True)


def test_terrain_demo_synthetic_render(tmp_path: Path) -> None:
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    hdr_path = _create_hdr_fixture(tmp_path)
    try:
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)

    heightmap = _synthetic_dem(256, 256)
    params_config = _build_params()
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=f3d.TerrainRenderParams(params_config),
        heightmap=heightmap,
        target=None,
    )
    shadow_off_config = _build_params()
    shadow_off_config.shadows.enabled = False
    shadow_off_config.shadows.technique = "NONE"
    shadow_off_frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=f3d.TerrainRenderParams(shadow_off_config),
        heightmap=heightmap,
        target=None,
    )

    output_path = tmp_path / "terrain_demo_synthetic.png"
    frame.save(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    pixels = frame.to_numpy()
    assert pixels.shape == (256, 256, 4)
    assert pixels.dtype == np.uint8

    unique_rgb = _unique_color_count(pixels)
    assert unique_rgb >= 256, f"Expected at least 256 unique colors, found {unique_rgb}"

    shadow_off_pixels = shadow_off_frame.to_numpy()
    shadowed_luminance = _linear_luminance_map(pixels)
    shadow_off_luminance = _linear_luminance_map(shadow_off_pixels)
    shadow_delta = shadow_off_luminance - shadowed_luminance

    shadow_off_mean = float(shadow_off_luminance.mean())
    assert 0.15 <= shadow_off_mean <= 0.85, (
        f"Shadow-off mean luminance {shadow_off_mean:.6f} outside [0.15, 0.85]"
    )

    mean_darkening = float(shadow_delta.mean())
    assert 0.0002 <= mean_darkening <= 0.01, (
        f"PCSS mean darkening {mean_darkening:.6f} outside [0.0002, 0.01]"
    )

    localized_dark_fraction = float(np.mean(shadow_delta > (1.0 / 255.0)))
    assert 0.02 <= localized_dark_fraction <= 0.25, (
        "PCSS did not produce a localized cast-shadow region: "
        f"fraction={localized_dark_fraction:.6f}"
    )

    brightening = np.maximum(-shadow_delta, 0.0)
    brightened_fraction = float(np.mean(brightening > (1.0 / 255.0)))
    mean_brightening = float(brightening.mean())
    assert brightened_fraction <= 0.001 and mean_brightening <= 1e-5, (
        "Enabling PCSS brightened pixels unexpectedly: "
        f"fraction={brightened_fraction:.6f}, mean={mean_brightening:.8f}"
    )

    # With this fixed camera/light setup, the upper quarter is the far-lit
    # control region while the cast shadow falls lower and to the right.
    far_lit_delta = np.abs(shadow_delta[: pixels.shape[0] // 4, :])
    far_lit_mean_drift = float(far_lit_delta.mean())
    far_lit_changed_fraction = float(np.mean(far_lit_delta > 0.001))
    assert far_lit_mean_drift <= 0.0005 and far_lit_changed_fraction <= 0.03, (
        "PCSS changed the far-lit control region too broadly: "
        f"mean={far_lit_mean_drift:.8f}, fraction={far_lit_changed_fraction:.6f}"
    )


def _build_params() -> TerrainRenderParamsConfig:
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#1e3a5f"), (0.5, "#6ca365"), (1.0, "#f5f1d0")],
        domain=(0.0, 1.0),
    )
    overlay = f3d.OverlayLayer.from_colormap1d(cmap, strength=0.4)

    return TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        terrain_span=2.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=6.0,
        cam_phi_deg=135.0,
        cam_theta_deg=42.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 500.0),
        light=LightSettings("Directional", 135.0, 40.0, 3.0, [1.0, 0.97, 0.92]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            True,
            "PCSS",
            1024,
            2,
            500.0,
            1.0,
            0.8,
            0.002,
            0.001,
            0.3,
            1e-4,
            0.5,
            2.0,
            0.9,
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.08,
        gamma=2.2,
        albedo_mode="mix",
        colormap_strength=0.5,
        output_srgb_eotf=True,
    )


def _synthetic_dem(width: int, height: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    peak = 400.0 * np.exp(-(xx ** 2 + yy ** 2) / 0.18)
    ridges = 120.0 * np.sin(9.0 * np.arctan2(yy, xx)) * np.exp(-(xx ** 2 + yy ** 2) / 0.5)

    np.random.seed(7)
    noise = 35.0 * np.random.randn(height, width).astype(np.float32)

    heightmap = peak + ridges + noise
    heightmap = np.clip(heightmap, 0.0, 1000.0)
    return heightmap.astype(np.float32)


def _unique_color_count(pixels: np.ndarray) -> int:
    rgb = pixels[:, :, :3]
    flat = rgb.reshape(-1, 3)
    return int(np.unique(flat, axis=0).shape[0])


def _linear_luminance_map(pixels: np.ndarray) -> np.ndarray:
    rgb = pixels[:, :, :3].astype(np.float32) / 255.0
    rgb_linear = np.power(rgb, 2.2)
    return (
        0.2126 * rgb_linear[:, :, 0]
        + 0.7152 * rgb_linear[:, :, 1]
        + 0.0722 * rgb_linear[:, :, 2]
    )


def _create_hdr_fixture(tmp_path: Path) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=".hdr", dir=tmp_path)
    os.close(fd)
    path = Path(path_str)

    width, height = 16, 8
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode("ascii"))
        for y in range(height):
            for x in range(width):
                r = int(255.0 * (x / max(width - 1, 1)))
                g = int(255.0 * (y / max(height - 1, 1)))
                b = 180
                e = 128
                handle.write(bytes((r, g, b, e)))
    return path
