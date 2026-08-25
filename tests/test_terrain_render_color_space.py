# tests/test_terrain_render_color_space.py
# Test that terrain rendering produces correct color-space output without horizontal banding

import numpy as np
import pytest
from _terrain_runtime import terrain_rendering_available

import forge3d

pytestmark = pytest.mark.apple_metal_physical


def _test_ibl(tmp_path):
    from _generated_assets import write_hdr

    return forge3d.IBL.from_hdr(
        str(write_hdr(tmp_path / "test.hdr")), intensity=1.0
    )


def _test_overlay():
    colormap = forge3d.Colormap1D.from_stops(
        [(0.0, "#7a8a9a"), (1.0, "#7a8a9a")],
        domain=(0.0, 3000.0),
    )
    return forge3d.OverlayLayer.from_colormap1d(colormap, strength=1.0)


def test_terrain_render_no_horizontal_banding(tmp_path):
    """Test that terrain rendering doesn't produce horizontal banding artifacts."""
    # Create session and renderer
    sess = forge3d.Session(window=False)
    renderer = forge3d.TerrainRenderer(sess)

    # Create a flat heightmap at constant elevation
    heights = np.ones((256, 256), dtype=np.float32) * 1500.0

    # Create materials and params
    materials = forge3d.MaterialSet.terrain_default()

    ibl = _test_ibl(tmp_path)

    # Create rendering params with full configuration
    from forge3d import (
        TerrainRenderParamsConfig,
        LightSettings,
        IblSettings,
        ShadowSettings,
        TriplanarSettings,
        PomSettings,
        LodSettings,
        SamplingSettings,
        ClampSettings,
    )
    config = TerrainRenderParamsConfig(
        size_px=(512, 512),
        render_scale=1.0,
        terrain_span=2.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(False, "PCSS", 2048, 3, 2000.0, 1.0, 0.8, 0.5, 0.002, 0.5, 1e-4, 0.5, 40.0, 1.0),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.04, 8, 24, 2, False, False),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 4, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 3000.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[_test_overlay()],
        exposure=1.0,
        gamma=2.2,
        colormap_srgb=True,
        output_srgb_eotf=True,
        albedo_mode="colormap",
        colormap_strength=1.0,
    )
    params = forge3d.TerrainRenderParams(config)

    # Render
    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        target=None,
        heightmap=heights,
    )

    pixels = frame.to_numpy()

    # Verify output shape and type
    assert pixels.shape == (512, 512, 4), f"Expected (512, 512, 4), got {pixels.shape}"
    assert pixels.dtype == np.uint8, f"Expected uint8, got {pixels.dtype}"

    # Check that the image is not too dark (mean should be > 100)
    mean_brightness = pixels[:, :, :3].mean()
    assert mean_brightness > 100, f"Image too dark: mean={mean_brightness:.1f}"

    # Check for horizontal banding by analyzing row consistency
    # Calculate mean brightness for each row
    row_means = pixels[:, :, :3].mean(axis=(1, 2))

    # The middle 80% of rows should have similar values (ignore edges)
    middle_start = int(512 * 0.1)
    middle_end = int(512 * 0.9)
    middle_rows = row_means[middle_start:middle_end]

    # Standard deviation of row means should be small (< 10) for uniform rendering
    row_std = middle_rows.std()
    assert row_std < 10, f"Horizontal banding detected: row std={row_std:.2f}"

    # Check that we have reasonable color variation (not all one color)
    color_std = pixels[:, :, :3].std()
    assert color_std > 1.0, f"No color variation: std={color_std:.2f}"


def test_terrain_render_color_space_correct(tmp_path):
    """Test that color-space conversion is working correctly."""
    sess = forge3d.Session(window=False)
    renderer = forge3d.TerrainRenderer(sess)

    # Create a flat heightmap
    heights = np.ones((128, 128), dtype=np.float32) * 1500.0

    materials = forge3d.MaterialSet.terrain_default()

    ibl = _test_ibl(tmp_path)

    from forge3d import (
        TerrainRenderParamsConfig,
        LightSettings,
        IblSettings,
        ShadowSettings,
        TriplanarSettings,
        PomSettings,
        LodSettings,
        SamplingSettings,
        ClampSettings,
    )
    config = TerrainRenderParamsConfig(
        size_px=(256, 256),
        render_scale=1.0,
        terrain_span=2.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(False, "PCSS", 2048, 3, 2000.0, 1.0, 0.8, 0.5, 0.002, 0.5, 1e-4, 0.5, 40.0, 1.0),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.04, 8, 24, 2, False, False),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 4, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 3000.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[_test_overlay()],
        exposure=2.0,
        gamma=2.2,
        colormap_srgb=True,
        output_srgb_eotf=True,
        albedo_mode="colormap",
        colormap_strength=1.0,
    )
    params = forge3d.TerrainRenderParams(config)

    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        target=None,
        heightmap=heights,
    )

    pixels = frame.to_numpy()

    # With correct color-space handling, we should get reasonably bright mid-tones
    # Linear 0.35 (rock gray) → sRGB ~0.62 → u8 ~158
    # With lighting and exposure, should be in 100-200 range
    mean = pixels[:, :, :3].mean()
    assert 100 < mean < 230, f"Color-space issue: mean={mean:.1f} (expected 100-230)"

    # Check that we're not clipped to white or black
    assert pixels[:, :, :3].min() > 30, "Too much black clipping"
    assert pixels[:, :, :3].max() < 250, "Too much white clipping"


def test_terrain_render_non_aligned_dimensions(tmp_path):
    """Test that non-256-aligned dimensions work correctly (padding test)."""
    sess = forge3d.Session(window=False)
    renderer = forge3d.TerrainRenderer(sess)

    # Use odd dimensions that aren't 256-aligned
    heights = np.ones((127, 127), dtype=np.float32) * 1500.0

    materials = forge3d.MaterialSet.terrain_default()

    ibl = _test_ibl(tmp_path)

    from forge3d import (
        TerrainRenderParamsConfig,
        LightSettings,
        IblSettings,
        ShadowSettings,
        TriplanarSettings,
        PomSettings,
        LodSettings,
        SamplingSettings,
        ClampSettings,
    )
    config = TerrainRenderParamsConfig(
        size_px=(253, 251),  # Odd primes to stress padding
        render_scale=1.0,
        terrain_span=2.0,
        msaa_samples=1,
        z_scale=1.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=1200.0,
        cam_phi_deg=135.0,
        cam_theta_deg=45.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 6000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(False, "PCSS", 2048, 3, 2000.0, 1.0, 0.8, 0.5, 0.002, 0.5, 1e-4, 0.5, 40.0, 1.0),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.04, 8, 24, 2, False, False),
        lod=LodSettings(0, 0.0, -0.5),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 4, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 3000.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[_test_overlay()],
        exposure=1.0,
        gamma=2.2,
        colormap_srgb=True,
        output_srgb_eotf=True,
        albedo_mode="colormap",
        colormap_strength=1.0,
    )
    params = forge3d.TerrainRenderParams(config)

    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        target=None,
        heightmap=heights,
    )

    pixels = frame.to_numpy()

    # Verify correct output shape
    assert pixels.shape == (251, 253, 4), f"Expected (251, 253, 4), got {pixels.shape}"

    # Check for horizontal artifacts from padding issues
    row_means = pixels[:, :, :3].mean(axis=(1, 2))
    middle_rows = row_means[50:200]
    row_std = middle_rows.std()
    assert row_std < 10, f"Padding artifacts detected: row std={row_std:.2f}"
