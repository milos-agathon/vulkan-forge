# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New terrain data generation utilities in `vulkan_forge.terrain.data`
- Comprehensive terrain colormap utilities in `vulkan_forge.terrain.colormap`
- 3D and 2D terrain plotting functions in `vulkan_forge.terrain.plot3d`
- Terrain validation utilities in `vulkan_forge.testing.validation`
- Slim `terrain_demo.py` example demonstrating extracted library functions
- Enhanced Camera class with GL projection methods in `vulkan_forge.terrain.Camera`
- Complete terrain mesh building and rendering pipeline functions
- 16-bit heightmap loading and saving utilities
- Automated validation system with SHA-256 hash verification
- Color space conversion utilities (linear RGB ↔ sRGB)

### Changed
- Refactored terrain plotting code from `examples/terrain_plot.py` into proper library modules
- Enhanced terrain module with additional data generation and visualization functions
- Improved code organization and maintainability
- Migrated all reusable logic from `examples/10_terrain_gpu_render.py` to appropriate modules
- Created slim `terrain_gpu_demo.py` example (≤40 lines) using refactored modules

### Internal
- Extracted `create_terrain_data()` from example to `vulkan_forge.terrain.data`
- Extracted `create_terrain_colormap()` from example to `vulkan_forge.terrain.colormap`
- Extracted `create_unified_terrain_plot()` from example to `vulkan_forge.terrain.plot3d`
- Extracted `validate_color_consistency()` from example to `vulkan_forge.testing.validation`
- Migrated heightmap functions to `vulkan_forge.terrain.data`
- Migrated LUT and color functions to `vulkan_forge.terrain.colormap`
- Migrated rendering functions to `vulkan_forge.terrain.plot3d`
- Migrated validation functions to `vulkan_forge.testing.validation`
- Extended existing Camera class with GL projection methods
- Added comprehensive type hints and documentation to all extracted functions
- Maintained full backward compatibility with existing APIs

### Notes
- No breaking changes: all existing functions and classes maintain their original signatures
- The original `terrain_plot.py` and `10_terrain_gpu_render.py` examples continue to work unchanged
- All existing tests pass without modification
- New slim examples demonstrate proper library usage with minimal code
