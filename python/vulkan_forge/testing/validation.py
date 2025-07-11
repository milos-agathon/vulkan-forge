"""
Validation utilities for terrain visualization testing in VulkanForge.

This module provides functions for validating color consistency, terrain data
integrity, and other quality assurance checks for terrain visualizations.
"""

import numpy as np
import matplotlib.colors as mcolors
from typing import List, Tuple, Optional, Any


def validate_color_consistency(surface_colors: np.ndarray, 
                             terrain_lut: mcolors.LinearSegmentedColormap, 
                             norm: mcolors.Normalize, 
                             Y: np.ndarray) -> None:
    """
    Validate that surface facecolors match expected LUT output.
    
    Asserts color consistency between 3D surface and terrain LUT for
    representative samples to ensure unified color mapping.
    
    Args:
        surface_colors: RGBA color array from 3D surface
        terrain_lut: Terrain colormap used
        norm: Normalization used for colors
        Y: Height array
        
    Raises:
        AssertionError: If colors don't match expected values
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


def validate_terrain_data(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    """
    Validate terrain data arrays for consistency and correctness.
    
    Args:
        X: East-West coordinate array
        Y: Height array
        Z: North-South coordinate array
        
    Raises:
        AssertionError: If data validation fails
    """
    print("+ Validating terrain data...")
    
    # Check array shapes
    if not (X.shape == Y.shape == Z.shape):
        raise AssertionError(f"Array shapes don't match: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)) or np.any(np.isnan(Z)):
        raise AssertionError("Terrain data contains NaN values")
    
    if np.any(np.isinf(X)) or np.any(np.isinf(Y)) or np.any(np.isinf(Z)):
        raise AssertionError("Terrain data contains infinite values")
    
    # Check reasonable value ranges
    if Y.min() < 0:
        raise AssertionError(f"Height values should be non-negative, got min={Y.min()}")
    
    if Y.max() - Y.min() < 1e-6:
        raise AssertionError("Height data has no variation (flat terrain)")
    
    print("  PASS: Terrain data validated")


def validate_colormap(colormap: mcolors.LinearSegmentedColormap) -> None:
    """
    Validate that a colormap is properly configured.
    
    Args:
        colormap: Colormap to validate
        
    Raises:
        AssertionError: If colormap validation fails
    """
    print("+ Validating colormap...")
    
    # Test colormap at various points
    test_values = np.linspace(0, 1, 10)
    colors = colormap(test_values)
    
    # Check that we get RGBA arrays
    if colors.shape[1] != 4:
        raise AssertionError(f"Colormap should return RGBA values, got shape {colors.shape}")
    
    # Check that colors are in valid range [0, 1]
    if np.any(colors < 0) or np.any(colors > 1):
        raise AssertionError("Colormap returns values outside [0, 1] range")
    
    # Check that alpha channel is reasonable
    if np.any(colors[:, 3] < 0.5):  # Alpha should be reasonably opaque
        raise AssertionError("Colormap alpha values are too transparent")
    
    print("  PASS: Colormap validated")


def validate_gl_coordinate_system(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    """
    Validate that coordinate arrays follow GL convention.
    
    Args:
        X: East-West coordinate array
        Y: Height array
        Z: North-South coordinate array
        
    Raises:
        AssertionError: If coordinate system validation fails
    """
    print("+ Validating GL coordinate system...")
    
    # Check that X increases from west to east (left to right)
    if X[0, 0] >= X[0, -1]:
        raise AssertionError("X coordinates should increase from west to east")
    
    # Check that Z increases from south to north (bottom to top)
    if Z[0, 0] >= Z[-1, 0]:
        raise AssertionError("Z coordinates should increase from south to north")
    
    # Check that Y represents height (should be non-negative)
    if Y.min() < 0:
        raise AssertionError("Y (height) should be non-negative in GL convention")
    
    print("  PASS: GL coordinate system validated")


def validate_normalization(norm: mcolors.Normalize, Y: np.ndarray) -> None:
    """
    Validate that normalization is correctly configured for height data.
    
    Args:
        norm: Normalization object
        Y: Height array
        
    Raises:
        AssertionError: If normalization validation fails
    """
    print("+ Validating normalization...")
    
    # Check that normalization range matches data range
    if abs(norm.vmin - Y.min()) > 1e-6:
        raise AssertionError(f"Normalization vmin ({norm.vmin}) doesn't match data min ({Y.min()})")
    
    if abs(norm.vmax - Y.max()) > 1e-6:
        raise AssertionError(f"Normalization vmax ({norm.vmax}) doesn't match data max ({Y.max()})")
    
    # Test normalization at various points
    test_values = np.array([Y.min(), Y.max(), Y.mean()])
    normalized = norm(test_values)
    
    if not np.allclose(normalized[0], 0.0, atol=1e-6):
        raise AssertionError("Normalization should map minimum to 0")
    
    if not np.allclose(normalized[1], 1.0, atol=1e-6):
        raise AssertionError("Normalization should map maximum to 1")
    
    print("  PASS: Normalization validated")


def validate_plot_consistency(fig: Any, axes: Tuple[Any, Any, Any]) -> None:
    """
    Validate that plot elements are consistent across panels.
    
    Args:
        fig: Matplotlib figure
        axes: Tuple of axes objects
        
    Raises:
        AssertionError: If plot consistency validation fails
    """
    print("+ Validating plot consistency...")
    
    ax1, ax2, ax3 = axes
    
    # Check that all axes have proper labels
    required_labels = ['X (East-West)', 'Height (Y)', 'Z (North-South)']
    
    # Check 3D axes
    for ax in [ax1, ax2]:
        if not hasattr(ax, 'get_xlabel'):
            raise AssertionError("3D axes missing xlabel method")
        if ax.get_xlabel() not in required_labels:
            raise AssertionError(f"Invalid xlabel: {ax.get_xlabel()}")
    
    # Check 2D axes
    if ax3.get_xlabel() != 'X (East-West)':
        raise AssertionError(f"Invalid 2D xlabel: {ax3.get_xlabel()}")
    if ax3.get_ylabel() != 'Z (North-South)':
        raise AssertionError(f"Invalid 2D ylabel: {ax3.get_ylabel()}")
    
    print("  PASS: Plot consistency validated")


def validate_terrain_statistics(stats: dict) -> None:
    """
    Validate terrain statistics for reasonableness.
    
    Args:
        stats: Dictionary of terrain statistics
        
    Raises:
        AssertionError: If statistics validation fails
    """
    print("+ Validating terrain statistics...")
    
    required_keys = ['min_height', 'max_height', 'mean_height', 'std_height', 'height_range']
    
    for key in required_keys:
        if key not in stats:
            raise AssertionError(f"Missing statistic: {key}")
    
    # Check logical relationships
    if stats['min_height'] > stats['max_height']:
        raise AssertionError("min_height > max_height")
    
    if stats['height_range'] != stats['max_height'] - stats['min_height']:
        raise AssertionError("height_range != max_height - min_height")
    
    if stats['mean_height'] < stats['min_height'] or stats['mean_height'] > stats['max_height']:
        raise AssertionError("mean_height outside [min_height, max_height] range")
    
    if stats['std_height'] < 0:
        raise AssertionError("std_height should be non-negative")
    
    print("  PASS: Terrain statistics validated")


def run_comprehensive_validation(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                               colormap: mcolors.LinearSegmentedColormap,
                               norm: mcolors.Normalize,
                               surface_colors: np.ndarray,
                               fig: Any, axes: Tuple[Any, Any, Any]) -> None:
    """
    Run comprehensive validation of terrain visualization.
    
    Args:
        X, Y, Z: Coordinate arrays
        colormap: Terrain colormap
        norm: Normalization object
        surface_colors: Surface colors array
        fig: Matplotlib figure
        axes: Tuple of axes objects
        
    Raises:
        AssertionError: If any validation fails
    """
    print("="*60)
    print("COMPREHENSIVE TERRAIN VALIDATION")
    print("="*60)
    
    validate_terrain_data(X, Y, Z)
    validate_gl_coordinate_system(X, Y, Z)
    validate_colormap(colormap)
    validate_normalization(norm, Y)
    validate_color_consistency(surface_colors, colormap, norm, Y)
    validate_plot_consistency(fig, axes)
    
    # Validate terrain statistics
    from ..terrain.data import get_terrain_statistics
    stats = get_terrain_statistics(Y)
    validate_terrain_statistics(stats)
    
    print("\n" + "="*60)
    print("ALL VALIDATIONS PASSED")
    print("="*60)
