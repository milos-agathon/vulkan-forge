# Tracy Profiler Integration for Vulkan-Forge

This directory contains the Tracy profiler integration for Vulkan-Forge, providing both CPU and GPU profiling capabilities with minimal overhead when disabled.

## Quick Start

### 1. Initialize Git Submodule

```bash
# From repo root
git submodule add https://github.com/wolfpld/tracy.git third_party/tracy/upstream
git submodule update --init --recursive
```

### 2. Enable Profiling in CMake

```bash
# Development build with profiling
cmake -B build -DVULKAN_FORGE_ENABLE_PROFILING=ON

# Production build (profiling disabled)
cmake -B build -DVULKAN_FORGE_ENABLE_PROFILING=OFF
```

### 3. Basic Usage in Code

```cpp
#include "third_party/tracy/vulkan_forge_tracy.hpp"

void render_frame() {
    VF_TRACY_ZONE_RENDER("RenderFrame");
    
    // Your rendering code here
    draw_terrain();
    draw_water();
    
    VF_TRACY_FRAME_MARK();
}

void draw_terrain() {
    VF_TRACY_ZONE_NAMED("DrawTerrain");
    // Terrain rendering...
}
```

### 4. Vulkan GPU Profiling

```cpp
// Initialize once during renderer setup
VF_TRACY_VK_INIT(physical_device, device, graphics_queue, setup_cmd_buffer);

// In render loop
void record_commands(VkCommandBuffer cmd_buffer) {
    VF_TRACY_VK_ZONE_RENDER(cmd_buffer, "Terrain Pass");
    
    // Record Vulkan commands...
    vkCmdDrawIndexed(cmd_buffer, ...);
}

// Collect GPU data once per frame
VF_TRACY_VK_COLLECT(cmd_buffer);

// Cleanup during shutdown
VF_TRACY_VK_DESTROY();
```

## Available Macros

### CPU Profiling

| Macro | Description | Example |
|-------|-------------|---------|
| `VF_TRACY_ZONE()` | Profile current scope | `VF_TRACY_ZONE()` |
| `VF_TRACY_ZONE_NAMED(name)` | Profile with custom name | `VF_TRACY_ZONE_NAMED("LoadMesh")` |
| `VF_TRACY_ZONE_RENDER(name)` | Rendering operations (blue) | `VF_TRACY_ZONE_RENDER("Draw")` |
| `VF_TRACY_ZONE_MEMORY(name)` | Memory operations (green) | `VF_TRACY_ZONE_MEMORY("Alloc")` |
| `VF_TRACY_ZONE_IO(name)` | I/O operations (orange) | `VF_TRACY_ZONE_IO("LoadFile")` |
| `VF_TRACY_ZONE_COMPUTE(name)` | Compute operations (purple) | `VF_TRACY_ZONE_COMPUTE("Physics")` |
| `VF_TRACY_FRAME_MARK()` | Mark frame boundary | `VF_TRACY_FRAME_MARK()` |

### GPU Profiling (Vulkan)

| Macro | Description | Example |
|-------|-------------|---------|
| `VF_TRACY_VK_ZONE(cmd, name)` | Profile GPU zone | `VF_TRACY_VK_ZONE(cmd, "DrawCall")` |
| `VF_TRACY_VK_ZONE_RENDER(cmd, name)` | Rendering GPU zone | `VF_TRACY_VK_ZONE_RENDER(cmd, "Terrain")` |
| `VF_TRACY_VK_ZONE_COMPUTE(cmd, name)` | Compute GPU zone | `VF_TRACY_VK_ZONE_COMPUTE(cmd, "Culling")` |
| `VF_TRACY_VK_COLLECT(cmd)` | Collect GPU data | `VF_TRACY_VK_COLLECT(cmd_buffer)` |

### Memory & Data

| Macro | Description | Example |
|-------|-------------|---------|
| `VF_TRACY_ALLOC(ptr, size)` | Track allocation | `VF_TRACY_ALLOC(buffer, 1024)` |
| `VF_TRACY_FREE(ptr)` | Track deallocation | `VF_TRACY_FREE(buffer)` |
| `VF_TRACY_PLOT(name, value)` | Plot value over time | `VF_TRACY_PLOT("FPS", fps)` |
| `VF_TRACY_MESSAGE(text)` | Log message | `VF_TRACY_MESSAGE("Frame rendered")` |

## Configuration Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `VULKAN_FORGE_ENABLE_PROFILING` | `OFF` | Enable Tracy profiling |
| `VULKAN_FORGE_TRACY_ON_DEMAND` | `ON` | Connect only when profiler attached |
| `VULKAN_FORGE_TRACY_CALLSTACK` | `OFF` | Enable callstack collection (slower) |

### Build Configurations

```bash
# Development: Full profiling with callstacks
cmake -B build-dev \
  -DVULKAN_FORGE_ENABLE_PROFILING=ON \
  -DVULKAN_FORGE_TRACY_CALLSTACK=ON

# Performance: Profiling without callstacks  
cmake -B build-perf \
  -DVULKAN_FORGE_ENABLE_PROFILING=ON \
  -DVULKAN_FORGE_TRACY_CALLSTACK=OFF

# Release: No profiling overhead
cmake -B build-release \
  -DVULKAN_FORGE_ENABLE_PROFILING=OFF
```

## Using the Tracy Profiler

### 1. Download Tracy

```bash
# Get Tracy profiler GUI
wget https://github.com/wolfpld/tracy/releases/latest/download/Tracy-x.x.x.7z
```

### 2. Connect to Your Application

1. Build your application with profiling enabled
2. Run your application
3. Launch Tracy profiler GUI
4. Click "Connect" to attach to your running application

### 3. Key Views

- **Timeline**: See CPU and GPU zones over time
- **Memory**: Track allocation patterns and leaks
- **Statistics**: Frame time distribution, zone statistics
- **Compare**: Compare performance between runs

## Performance Impact

| Configuration | CPU Overhead | Memory Overhead | Build Time |
|---------------|--------------|-----------------|------------|
| Disabled | 0% | 0% | +0s |
| On-demand | <0.1% | ~10MB | +30s |
| Always-on | <1% | ~50MB | +30s |
| With callstacks | 2-5% | ~100MB | +45s |

## Best Practices

### 1. Zone Granularity
```cpp
// ✅ Good: Meaningful zones
VF_TRACY_ZONE_RENDER("DrawTerrain");
VF_TRACY_ZONE_MEMORY("AllocateVertexBuffer");

// ❌ Too fine-grained
VF_TRACY_ZONE_NAMED("i++");  // Don't profile individual operations
```

### 2. GPU Profiling
```cpp
// ✅ Profile complete render passes
VF_TRACY_VK_ZONE_RENDER(cmd, "Shadow Pass");
VF_TRACY_VK_ZONE_RENDER(cmd, "Main Pass");

// ❌ Don't profile individual draw calls (too much overhead)
```

### 3. Memory Tracking
```cpp
// ✅ Track significant allocations
if (size >= VF_TRACY_ALLOC_THRESHOLD_BYTES) {
    VF_TRACY_ALLOC(ptr, size);
}

// ❌ Don't track tiny allocations (noise)
```

### 4. Conditional Profiling
```cpp
// Profile expensive operations only in debug builds
#ifdef DEBUG
    VF_TRACY_ZONE_NAMED("ExpensiveValidation");
    validate_all_buffers();
#endif
```

## Integration with CI/CD

### GitHub Actions

```yaml
# .github/workflows/profiling.yml
- name: Build with profiling
  run: |
    cmake -B build -DVULKAN_FORGE_ENABLE_PROFILING=ON
    cmake --build build
    
- name: Run performance tests  
  run: |
    ./build/tests/perf_tests --tracy-capture=results.tracy
    
- name: Upload Tracy capture
  uses: actions/upload-artifact@v3
  with:
    name: tracy-capture
    path: results.tracy
```

## Troubleshooting

### Common Issues

1. **Tracy not connecting**
   - Ensure firewall allows Tracy connections (port 8086)
   - Check that `TRACY_ON_DEMAND=1` is set
   - Verify application is running when connecting

2. **Missing GPU data**  
   - Call `VF_TRACY_VK_INIT()` during setup
   - Ensure `VF_TRACY_VK_COLLECT()` is called each frame
   - Check Vulkan validation layers aren't interfering

3. **High overhead**
   - Disable callstack collection: `VULKAN_FORGE_TRACY_CALLSTACK=OFF`
   - Use on-demand mode: `VULKAN_FORGE_TRACY_ON_DEMAND=ON`
   - Reduce zone granularity

### Debug Build Configuration

```cpp
// Add to debug builds for extra validation
#ifdef DEBUG
    static_assert(vulkan_forge::profiling::is_enabled(), 
                  "Profiling should be enabled in debug builds");
#endif
```

## Version Compatibility

| Tracy Version | Vulkan API | Notes |
|---------------|------------|-------|
| v0.10.x | 1.0-1.3 | Recommended for Vulkan-Forge |
| v0.9.x | 1.0-1.2 | Legacy support |
| v0.11.x | 1.0-1.3 | Development branch |

Update Tracy version in the submodule:
```bash
cd third_party/tracy/upstream
git checkout v0.10  # Use desired version
cd ../../..
git add third_party/tracy/upstream
git commit -m "Update Tracy to v0.10"
```