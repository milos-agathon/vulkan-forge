# vulkan-forge <img src="man/figures/logo.png" align="right" height="139" />

<!-- badges: start -->

[![R-CMD-check](https://github.com/milos-agathon/vulkan-forge/workflows/R-CMD-check/badge.svg)](https://github.com/milos-agathon/vulkan-forge/actions)
[![Codecov test coverage](https://codecov.io/gh/milos-agathon/vulkan-forge/branch/main/graph/badge.svg)](https://app.codecov.io/gh/milos-agathon/vulkan-forge?branch=main)
[![CRAN status](https://www.r-pkg.org/badges/version/vulkan-forge)](https://CRAN.R-project.org/package=vulkan-forge)
[![Lifecycle: stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://cranlogs.r-pkg.org/badges/vulkan-forge)](https://CRAN.R-project.org/package=vulkan-forge)

<!-- badges: end -->

> **Comprehensive GPU Backend for Vulkan Ray-Tracing in R**

**vulkan-forge** provides R users with a high-performance Vulkan GPU backend for ray-tracing and real-time rendering. It handles device enumeration, automatic backend selection (including CPU fallback), memory management, multi-GPU support, and advanced performance tuning—so you can focus on visualization rather than low-level graphics setup.

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
* **R Interfaces**
  High-level R functions with `.Call()` bridges to native Vulkan code

### Install Vulkan SDK

Before using vulkan-forge, you must install the **Vulkan SDK** from LunarG. Follow the steps for your operating system:

#### Windows

1. **Download the SDK**

   * Visit [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) and download the latest Windows installer.
2. **Run the Installer**

   * Execute the downloaded `.exe` and follow the prompts.
3. **Update PATH**

   * Ensure the SDK `Bin` directory (e.g. `C:\VulkanSDK\<version>\Bin`) is added to your **System** `PATH` environment variable.

#### macOS

1. **Download the SDK**

   * Visit [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home#macos) and download the macOS `.dmg` package.
2. **Install**

   * Open the `.dmg` and drag the SDK folder to `/usr/local/` (or a location of your choice).
3. **Update PATH**

   * Add the SDK binaries to your shell profile, e.g.:

     ```bash
     export VULKAN_SDK="/usr/local/VulkanSDK/<version>/macOS"
     export PATH="$VULKAN_SDK/bin:$PATH"
     ```

#### Linux

1. **Download & Extract**

   ```bash
   # Replace <version> with the downloaded version
   wget https://sdk.lunarg.com/sdk/download/<version>/linux/vulkansdk-linux-x86_64-<version>.tar.gz
   tar -xzf vulkansdk-linux-x86_64-<version>.tar.gz -C $HOME
   ```
2. **Source Environment**

   ```bash
   source $HOME/VulkanSDK/<version>/setup-env.sh
   ```
3. **Persist Across Sessions**

   * Add the above `source` line to your `~/.bashrc` or `~/.zshrc`.

#### Verification

After installation, confirm everything is working by running:

```bash
vulkaninfo | head -n 10
```

You should see your GPU and driver details printed to the console.

---

## Installation

Install the released version from CRAN:

```r
install.packages("vulkan-forge")
```

Or install the development version from GitHub:

```r
# install.packages("devtools")
devtools::install_github("milos-agathon/vulkan-forge")
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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and coding standards.

## Citation

If you use **vulkan-forge** in published research, please cite:

> Milos Popovic (2025). *vulkan-forge: GPU Backend for Vulkan Ray-Tracing in Python*. Python package version 1.0.0. [https://github.com/milos-agathon/vulkan-forge](https://github.com/milos-agathon/vulkan-forge)

## License

**vulkan-forge** is released under the [MIT License](LICENSE).
