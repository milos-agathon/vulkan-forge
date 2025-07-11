#!/usr/bin/env python3
"""
Slim terrain GPU rendering demo using refactored VulkanForge library modules.

This example demonstrates the complete terrain rendering pipeline using
the extracted and refactored library functions.
"""

import sys
import os
import argparse
from pathlib import Path

# Add python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vulkan_forge.terrain import (
    create_terrain_data, build_verified_mesh, apply_fragment_colors,
    create_terrain_lut_png, save_heightmap_16bit, load_heightmap_16bit_strict,
    render_with_gl_camera, create_verified_preview, linear_to_srgb, Camera
)
from vulkan_forge.testing import run_automated_validation

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def main():
    parser = argparse.ArgumentParser(description="Terrain GPU rendering demo")
    parser.add_argument('--size', type=int, default=65, help='Terrain size')
    parser.add_argument('--output', type=str, default='terrain_FINAL_VERIFIED.png', help='Output file')
    parser.add_argument('--preview', action='store_true', help='Show preview')
    parser.add_argument('--generate', action='store_true', help='Generate new terrain')
    args = parser.parse_args()
    
    print("="*60)
    print("TERRAIN GPU RENDERING DEMO")
    print("="*60)
    
    # Generate or load terrain using existing create_terrain_data
    if args.generate:
        print("+ Generating terrain data...")
        X, Y, Z = create_terrain_data(size=args.size, seed=0)
        terrain = Y  # Use Y as height data
        save_heightmap_16bit(terrain, "heightmap16_verified.png")
    else:
        print("+ Loading existing terrain...")
        if Path("heightmap16_verified.png").exists():
            terrain = load_heightmap_16bit_strict("heightmap16_verified.png")
        else:
            print("  No existing heightmap found, generating new one...")
            X, Y, Z = create_terrain_data(size=args.size, seed=0)
            terrain = Y
            save_heightmap_16bit(terrain, "heightmap16_verified.png")
    
    # Create terrain LUT
    print("+ Creating terrain color LUT...")
    lut_colors = create_terrain_lut_png()
    
    # Build verified mesh
    print("+ Building verified mesh...")
    mesh_cache = build_verified_mesh(terrain)
    
    # Apply fragment colors
    print("+ Applying fragment colors...")
    mesh_cache = apply_fragment_colors(mesh_cache, terrain, lut_colors)
    
    # Setup camera
    print("+ Setting up camera...")
    camera = Camera()
    camera.set_orbit_position(
        center=np.array([0.0, 0.0, 0.0]),
        angle_degrees=45,
        elevation_degrees=30,
        distance=6
    )
    
    # Render with GL camera
    print("+ Rendering with GL camera...")
    image_linear = render_with_gl_camera(mesh_cache, camera, 1024, 768)
    
    # Convert to sRGB and save
    image_srgb = linear_to_srgb(image_linear)
    image_uint8 = (image_srgb * 255).astype('uint8')
    
    if HAS_PIL:
        Image.fromarray(image_uint8).save(args.output)
        print(f"+ Saved render: {args.output}")
    
    # Show preview if requested
    if args.preview:
        print("+ Creating preview...")
        ortho_render, _ = create_verified_preview(mesh_cache, terrain, camera, lut_colors)
    else:
        # Generate orthographic for validation
        ortho_image = render_with_gl_camera(mesh_cache, camera, 512, 512, is_perspective=False)
        ortho_render = (linear_to_srgb(ortho_image) * 255).astype('uint8')
    
    # Run validation
    print("+ Running validation...")
    validation_passed = run_automated_validation(mesh_cache, terrain, camera, ortho_render, lut_colors)
    
    if validation_passed:
        print("\n" + "="*60)
        print("SUCCESS: All validations passed!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("FAIL: Validation failed!")
        print("="*60)
        return 1


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
