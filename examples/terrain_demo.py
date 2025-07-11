#!/usr/bin/env python3
"""
Terrain visualization demo using extracted VulkanForge library modules.

This slim example demonstrates the terrain plotting capabilities after
extracting the core functionality into proper library modules.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from vulkan_forge.terrain import (
    create_terrain_data,
    create_terrain_colormap,
    create_unified_terrain_plot
)
from vulkan_forge.testing import validate_color_consistency


def main():
    """Main execution with terrain visualization using extracted modules."""
    print("="*60)
    print("TERRAIN DEMO - EXTRACTED MODULES")
    print("="*60)
    print("+ Creating deterministic terrain data...")
    
    # Create terrain data using extracted function
    X, Y, Z = create_terrain_data(size=33, seed=0)
    print(f"  Terrain: {X.shape[0]}x{X.shape[1]} grid")
    print(f"  GL bounds: X[{X.min():.1f}, {X.max():.1f}], "
          f"Y[{Y.min():.2f}, {Y.max():.2f}], Z[{Z.min():.1f}, {Z.max():.1f}]")
    
    print("+ Creating unified terrain colormap...")
    terrain_lut = create_terrain_colormap()
    print("  Single terrain LUT with 256 colors created")
    
    print("+ Generating unified 3-panel visualization...")
    fig, axes, surface_colors, norm = create_unified_terrain_plot(X, Y, Z, terrain_lut)
    print("  3 panels created with shared colormap and GL convention")
    
    print("+ Running validation...")
    validate_color_consistency(surface_colors, terrain_lut, norm, Y)
    
    # Save output
    output_file = 'terrain_verified_aligned.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"+ Saved verified output: {output_file}")
    
    # Show plot
    import matplotlib.pyplot as plt
    plt.show()
    
    print("\n" + "="*60)
    print("SUCCESS: Terrain demo completed using extracted modules")
    print("="*60)
    print("+ Functions imported from vulkan_forge.terrain")
    print("+ Validation imported from vulkan_forge.testing")
    print("+ All library code properly extracted and organized")
    print(f"+ Output saved: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
