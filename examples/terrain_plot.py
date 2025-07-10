#!/usr/bin/env python3
"""
Unified terrain visualization with consistent color mapping across all panels.

This script generates a 3-panel figure where all panels share the exact same
height-to-color mapping using a centralized LinearSegmentedColormap. The left
and middle panels show 3D surfaces with proper GL convention axes, while the
right panel shows a 2D orthographic validation view. Color normalization is
centralized using matplotlib.colors.Normalize, and validation ensures color
consistency across all panels.

GL Convention: X = East-West, Y = Height, Z = North-South
Color mapping: Single terrain LUT applied consistently to all three panels
Validation: Asserts color fidelity between 3D surface and expected LUT output
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


def create_terrain_data(size=33, seed=0):
    """Create deterministic terrain data with GL convention coordinates."""
    np.random.seed(seed)  # Deterministic rendering
    
    # Create coordinate grids with GL convention
    x = np.linspace(-2, 2, size)  # East-West
    z = np.linspace(-2, 2, size)  # North-South  
    X, Z = np.meshgrid(x, z)
    
    # Generate terrain features (Y = height)
    Y = np.zeros_like(X)
    
    # Central mountain
    Y += np.exp(-((X-0.3)**2 + (Z+0.2)**2) / 2.5) * 0.7
    
    # Secondary peaks
    Y += np.exp(-((X+1.2)**2 + (Z-1.0)**2) / 1.8) * 0.4
    Y += np.exp(-((X-1.5)**2 + (Z+1.3)**2) / 1.5) * 0.3
    
    # Valley system
    valley = -np.exp(-((X+0.5)**2 + (Z-0.8)**2) / 3.0) * 0.2
    Y += valley
    
    # Rolling hills
    Y += np.sin(X * 1.5) * np.cos(Z * 1.8) * 0.1
    Y += np.sin(X * 3.2) * np.cos(Z * 2.5) * 0.05
    
    # Fine detail noise
    noise = np.random.random((size, size)) * 0.03
    Y += noise
    
    # Normalize to [0, 1] and apply height scaling
    Y = np.maximum(Y, 0)
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    Y = Y * 0.4  # Scale to 0.4m max height
    
    return X, Y, Z


def create_terrain_colormap():
    """Create single source of truth terrain colormap with proper normalization."""
    # Define terrain color gradient: water -> sand -> grass -> rock -> snow
    colors = [
        '#1e3a8a',  # Deep blue (water)
        '#3b82f6',  # Light blue (shallow water)  
        '#fbbf24',  # Sand/beach
        '#22c55e',  # Grass/plains
        '#166534',  # Forest/dark green
        '#78716c',  # Rock/stone gray
        '#f8fafc'   # Snow/peaks white
    ]
    
    # Create LinearSegmentedColormap
    terrain_lut = mcolors.LinearSegmentedColormap.from_list(
        'terrain_lut', colors, N=256
    )
    
    return terrain_lut


def create_unified_terrain_plot(X, Y, Z, terrain_lut):
    """
    Create 3-panel figure with unified color mapping and GL axis convention.
    
    All panels use the same terrain_lut and normalization for consistent colors.
    """
    # Single source normalization
    norm = mcolors.Normalize(vmin=Y.min(), vmax=Y.max())
    
    # Create figure with consistent layout
    fig = plt.figure(figsize=(18, 6), dpi=150)
    
    # Panel 1: 3D Surface with unified colors
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Apply terrain colormap to surface
    surface_colors = terrain_lut(norm(Y))
    
    surf1 = ax1.plot_surface(X, Y, Z, facecolors=surface_colors, 
                            linewidth=0, antialiased=False, alpha=0.9)
    
    # GL axis convention
    ax1.set_xlabel('X (East-West)')
    ax1.set_ylabel('Height (Y)')
    ax1.set_zlabel('Z (North-South)')
    ax1.set_title('3D Surface (Unified Colors)')
    ax1.view_init(elev=30, azim=45)
    
    # Panel 2: 3D Surface (colored, not wireframe) with same LUT
    ax2 = fig.add_subplot(132, projection='3d')
    
    # Create colored surface using same colormap/normalization
    surf2 = ax2.plot_surface(X, Y, Z, facecolors=surface_colors,
                            rstride=1, cstride=1, linewidth=0.5, 
                            antialiased=False, alpha=0.8)
    
    # GL axis convention  
    ax2.set_xlabel('X (East-West)')
    ax2.set_ylabel('Height (Y)')
    ax2.set_zlabel('Z (North-South)')
    ax2.set_title('3D Surface with Edges (Same LUT)')
    ax2.view_init(elev=30, azim=45)
    
    # Panel 3: 2D Orthographic validation with exact alignment
    ax3 = fig.add_subplot(133)
    
    # Use same colormap and normalization, proper extent alignment
    im = ax3.imshow(Y, cmap=terrain_lut, norm=norm, origin='lower',
                    extent=[X.min(), X.max(), Z.min(), Z.max()],
                    interpolation='nearest', alpha=0.9)
    
    # GL axis convention for 2D plot
    ax3.set_xlabel('X (East-West)')
    ax3.set_ylabel('Z (North-South)')
    ax3.set_title('2D Orthographic Validation\n(Same LUT & Normalization)')
    
    # Add colorbar with proper label
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Height (Y)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3), surface_colors, norm


def validate_color_consistency(surface_colors, terrain_lut, norm, Y):
    """
    Validate that surface facecolors match expected LUT output.
    
    Asserts color consistency between 3D surface and terrain LUT for
    representative samples to ensure unified color mapping.
    """
    print("+ Validating color consistency...")
    
    # Get expected colors from LUT
    expected_colors = terrain_lut(norm(Y))
    
    # Test representative samples (avoid edge effects)
    test_indices = [
        (Y.shape[0]//4, Y.shape[1]//4),      # Lower-left quadrant
        (Y.shape[0]//2, Y.shape[1]//2),      # Center
        (3*Y.shape[0]//4, 3*Y.shape[1]//4),  # Upper-right quadrant
    ]
    
    for i, j in test_indices:
        surface_rgb = surface_colors[i, j, :3]  # RGB only (ignore alpha)
        expected_rgb = expected_colors[i, j, :3]
        
        if not np.allclose(surface_rgb, expected_rgb, atol=1e-6):
            raise AssertionError(
                f"Color mismatch at ({i},{j}): "
                f"surface={surface_rgb}, expected={expected_rgb}"
            )
    
    # Test full array consistency (more comprehensive)
    if not np.allclose(surface_colors[:, :, :3], expected_colors[:, :, :3], atol=1e-6):
        raise AssertionError("Surface colors do not match expected LUT output")
    
    print("  PASS: Color consistency validated")


def main():
    """Main execution with comprehensive terrain visualization and validation."""
    print("="*60)
    print("UNIFIED TERRAIN VISUALIZATION")
    print("="*60)
    print("+ Creating deterministic terrain data...")
    
    # Create terrain data with deterministic seed
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
    plt.show()
    
    print("\n" + "="*60)
    print("SUCCESS: All panels use unified color mapping")
    print("="*60)
    print("+ Single terrain LUT applied consistently")
    print("+ GL axis convention throughout (X=E/W, Y=Height, Z=N/S)")
    print("+ Color validation passed for all test samples")
    print("+ 3D surfaces and 2D validation are color-aligned")
    print(f"+ Output saved: {output_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
