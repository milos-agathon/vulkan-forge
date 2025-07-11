"""
Testing utilities for VulkanForge.

This package provides validation and testing utilities for terrain visualization
and other VulkanForge components.
"""

from .validation import (
    validate_color_consistency,
    validate_terrain_data,
    validate_colormap,
    validate_gl_coordinate_system,
    validate_normalization,
    validate_plot_consistency,
    validate_terrain_statistics,
    run_comprehensive_validation
)

__all__ = [
    'validate_color_consistency',
    'validate_terrain_data',
    'validate_colormap',
    'validate_gl_coordinate_system',
    'validate_normalization',
    'validate_plot_consistency',
    'validate_terrain_statistics',
    'run_comprehensive_validation'
]
