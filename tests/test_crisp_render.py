#!/usr/bin/env python3
"""
Regression test for terrain crisp rendering.

Ensures the terrain renderer produces non-black output with proper dimensions.
"""

import sys
import os
from pathlib import Path
import tempfile
import numpy as np

# Add python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vulkan_forge.terrain import (
    create_terrain_data, build_verified_mesh, compute_vertex_normals,
    apply_fragment_colors, create_terrain_lut_png, linear_to_srgb,
    render_high_quality, apply_atmospheric_perspective
)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import scipy.ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_crisp_render_non_black():
    """Test that terrain crisp rendering produces non-black output."""
    print("Testing terrain crisp rendering...")
    
    # Generate terrain data
    X, Y, Z = create_terrain_data(size=33, seed=42)  # Smaller for faster testing
    
    # Build mesh with verified geometry
    mesh_cache = build_verified_mesh(Y, world_scale=4.0, height_scale=0.6)
    
    # Compute vertex normals for lighting
    mesh_cache = compute_vertex_normals(mesh_cache)
    
    # Create terrain LUT and apply colors
    lut_colors = create_terrain_lut_png()
    mesh_cache = apply_fragment_colors(mesh_cache, Y, lut_colors)
    
    # Render at smaller resolution for faster testing
    width, height = 640, 480
    
    if HAS_SCIPY:
        image_linear = render_high_quality(
            mesh_cache, 
            width=width, 
            height=height, 
            supersample=2,  # Smaller supersample for faster testing
            enable_lighting=True
        )
    else:
        # Fallback without supersampling
        from vulkan_forge.terrain.plot3d import render_with_gl_camera, create_optimal_terrain_camera, create_lighting_system
        camera = create_optimal_terrain_camera(mesh_cache)
        camera.aspect = width / height
        lighting_system = create_lighting_system()
        image_linear = render_with_gl_camera(
            mesh_cache, camera, width, height, 
            is_perspective=True, lighting_system=lighting_system
        )
    
    # Verify image is not black
    assert image_linear.mean() > 0.05, f"Image is too dark: mean={image_linear.mean():.6f}"
    assert image_linear.max() > 0.1, f"Image lacks bright pixels: max={image_linear.max():.6f}"
    assert image_linear.shape == (height, width, 3), f"Wrong image dimensions: {image_linear.shape}"
    
    # Convert to sRGB and save
    image_srgb = linear_to_srgb(image_linear)
    image_uint8 = np.clip(image_srgb * 255, 0, 255).astype(np.uint8)
    
    # Test PNG saving if PIL available
    if HAS_PIL:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            Image.fromarray(image_uint8).save(tmp_path)
            
            # Check file size and dimensions
            file_size = tmp_path.stat().st_size / 1024  # KB
            assert file_size > 50, f"PNG file too small: {file_size:.1f} KB"
            
            # Verify saved image can be loaded
            saved_image = Image.open(tmp_path)
            assert saved_image.size == (width, height), f"Saved image wrong size: {saved_image.size}"
            saved_image.close()
            
        finally:
            # Clean up
            try:
                tmp_path.unlink()
            except:
                pass
    
    print(f"[PASS] Terrain rendering test passed:")
    print(f"  Image mean: {image_linear.mean():.6f}")
    print(f"  Image max: {image_linear.max():.6f}")
    print(f"  Dimensions: {image_linear.shape}")
    if HAS_PIL:
        print(f"  PNG file size: {file_size:.1f} KB")


def test_fallback_render():
    """Test fallback rendering path without scipy."""
    print("Testing fallback rendering path...")
    
    # Generate minimal terrain data
    X, Y, Z = create_terrain_data(size=17, seed=123)  # Very small for quick test
    
    # Build mesh
    mesh_cache = build_verified_mesh(Y, world_scale=2.0, height_scale=0.3)
    mesh_cache = compute_vertex_normals(mesh_cache)
    
    # Apply colors
    lut_colors = create_terrain_lut_png()
    mesh_cache = apply_fragment_colors(mesh_cache, Y, lut_colors)
    
    # Use fallback rendering path directly
    from vulkan_forge.terrain.plot3d import render_with_gl_camera, create_optimal_terrain_camera, create_lighting_system
    
    camera = create_optimal_terrain_camera(mesh_cache)
    camera.aspect = 320 / 240
    lighting_system = create_lighting_system()
    
    image_linear = render_with_gl_camera(
        mesh_cache, camera, 320, 240, 
        is_perspective=True, lighting_system=lighting_system
    )
    
    # Verify non-black output
    assert image_linear.mean() > 0.01, f"Fallback render too dark: mean={image_linear.mean():.6f}"
    assert image_linear.shape == (240, 320, 3), f"Wrong fallback dimensions: {image_linear.shape}"
    
    print(f"[PASS] Fallback rendering test passed:")
    print(f"  Image mean: {image_linear.mean():.6f}")
    print(f"  Dimensions: {image_linear.shape}")


if __name__ == "__main__":
    try:
        test_crisp_render_non_black()
        test_fallback_render()
        print("\n[PASS] All terrain rendering tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
