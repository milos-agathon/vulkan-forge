# vulkan-forge <img src="man/figures/logo.png" align="right" height="139" />

<!-- badges: start -->

[![R-CMD-check](https://github.com/milos-agathon/vulkan-forge/workflows/R-CMD-check/badge.svg)](https://github.com/milos-agathon/vulkan-forge/actions)
[![Codecov test coverage](https://codecov.io/gh/milos-agathon/vulkan-forge/branch/main/graph/badge.svg)](https://app.codecov.io/gh/milos-agathon/vulkan-forge?branch=main)
[![CRAN status](https://www.r-pkg.org/badges/version/vulkan-forge)](https://CRAN.R-project.org/package=vulkan-forge)
[![Lifecycle: stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://cranlogs.r-pkg.org/badges/vulkan-forge)](https://CRAN.R-project.org/package=vulkan-forge)

<!-- badges: end -->

> **Comprehensive GPU Backend for Vulkan Ray-Tracing in Python**

**vulkan-forge** provides Python users with a high-performance Vulkan GPU backend for ray-tracing and real-time rendering. It handles device enumeration, automatic backend selection (including CPU fallback), memory management, multi-GPU support, and advanced performance tuning—so you can focus on visualization rather than low-level graphics setup.

## Key Features

* **Vulkan Integration**
  Full support for Vulkan API and SPIR-V shaders on compatible GPUs
* **Compute Fallback**
  Seamless CPU path-tracing when no Vulkan device is available
* **Multi-GPU Systems**
  Enumerate, filter, and select among multiple GPUs automatically
* **Automatic Backend Selection**
  Intelligent scoring based on memory, ray-tracing support, and workload
* **Performance Optimization**
  Memory budgeting, queue configuration, and shader cache strategies
* **Real-Time Monitoring**
  GPU metrics (memory, utilization, temperature, power) and plotting
* **Extensible Mesh & Material System**
  Height-map, point-cloud mesh generation, PBR materials, and denoising

# Vulkan Installation Guide for VulkanForge

## Prerequisites

To use GPU rendering with VulkanForge, you need:

1. **Vulkan-capable GPU** (NVIDIA, AMD, or Intel)
2. **Updated GPU drivers** with Vulkan support
3. **Vulkan SDK** (optional but recommended for development)
4. **Python vulkan package**

## Step 1: Install/Update GPU Drivers

### NVIDIA
- Download from: https://www.nvidia.com/Download/index.aspx
- Ensure driver version 390+ for Vulkan support

### AMD
- Download from: https://www.amd.com/en/support
- Use latest Adrenalin drivers

### Intel
- Download from: https://www.intel.com/content/www/us/en/support
- Use latest graphics drivers

## Step 2: Install Vulkan Runtime

### Windows
The Vulkan runtime is usually included with GPU drivers. If not:

1. Download Vulkan SDK from: https://vulkan.lunarg.com/sdk/home#windows
2. Run the installer
3. Add to PATH: `C:\VulkanSDK\<version>\Bin`

### Verify Installation
Open PowerShell and run:
```powershell
vulkaninfo
```

## Step 3: Install Python Vulkan Package

```powershell
# Install the vulkan package with all dependencies
pip install vulkan-forge[vulkan]

# Or install vulkan separately
pip install vulkan
```

## Step 4: Test GPU Detection

Create `test_gpu.py`:

```python
import vulkan_forge
from vulkan_forge import create_renderer

# This will show if Vulkan is available
print(f"Vulkan available: {vulkan_forge.backend.VULKAN_AVAILABLE}")

# Create renderer with GPU preference
renderer = create_renderer(prefer_gpu=True)
print(f"Renderer type: {type(renderer).__name__}")

if hasattr(renderer, 'gpu_active'):
    print(f"GPU active: {renderer.gpu_active}")
```

## Troubleshooting

### "Vulkan not available"
- Ensure GPU drivers are updated
- Check if `vulkaninfo` works in terminal
- Reinstall the vulkan Python package

### "No GPU devices found"
- Update GPU drivers
- Ensure your GPU supports Vulkan
- Check Device Manager for GPU issues

### "Failed to create logical device"
- This usually means the GPU doesn't support required features
- Try disabling validation layers: `create_renderer(enable_validation=False)`

## Environment Variables

You may need to set:
```powershell
$env:VK_ICD_FILENAMES = "C:\Windows\System32\nv-vk64.json"  # For NVIDIA
# or
$env:VK_ICD_FILENAMES = "C:\Windows\System32\amd-vulkan64.json"  # For AMD
```

## System Requirements

* **Vulkan SDK** (LunarG) installed and on your `PATH`
* **Graphics Card** with Vulkan support and ≥2 GB VRAM (NVIDIA, AMD, Intel)
* **R** ≥ 4.0
* **C++17 toolchain** (for building from source)

## Basic Usage


## Vignettes & Examples

* **Getting Started**: `vignette("00-intro-getting-started")`
* **ggplot → 3D Conversion**: `vignette("01-ggplot-workflow")`
* **tmap → 3D Spatial Viz**: `vignette("02-tmap-spatial-viz")`
* **Hardware Setup**: `vignette("03-hardware-setup")`
* **Performance Tuning**: `vignette("04-performance-tuning")`
* **Advanced Materials**: `vignette("05-advanced-materials")`
* **Technical Details**: `vignette("99-technical-details")`

## Documentation

Full reference documentation is at [https://milos-agathon.github.io/vulkan-forge/](https://milos-agathon.github.io/vulkan-forge/)
Browse function help pages with `?vk_list_backends`, `?vk_mesh_heightmap`, etc.

## Memory allocator

GPU buffers are now backed by AMD's [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator).
This greatly simplifies memory management and improves performance on discrete GPUs.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and coding standards.

## Citation

If you use **vulkan-forge** in published research, please cite:

> Milos Popovic (2025). *vulkan-forge: GPU Backend for Vulkan Ray-Tracing in Python*. Python package version 1.0.0. [https://github.com/milos-agathon/vulkan-forge](https://github.com/milos-agathon/vulkan-forge)

## License

**vulkan-forge** is released under the [MIT License](LICENSE).
