/**
 * @file tracy_config.hpp
 * @brief Custom Tracy configuration for Vulkan-Forge
 * 
 * This file must be included before Tracy headers to apply custom settings.
 * It provides a centralized place for Tracy configuration that doesn't
 * interfere with the upstream Tracy submodule.
 */

#pragma once

// Only apply configuration if Tracy is enabled
#if defined(VULKAN_FORGE_PROFILING_ENABLED) && VULKAN_FORGE_PROFILING_ENABLED

// Tracy memory profiling settings
#ifndef TRACY_NO_STATISTICS
#define TRACY_NO_STATISTICS 0  // Enable memory statistics by default
#endif

// Enhanced Vulkan profiling
#ifndef TRACY_VK_USE_SYMBOL_TABLE
#define TRACY_VK_USE_SYMBOL_TABLE 1  // Enable Vulkan symbol resolution
#endif

// Performance settings for development builds
#ifndef TRACY_NO_FRAME_IMAGE
#define TRACY_NO_FRAME_IMAGE 1  // Disable frame screenshots by default (performance)
#endif

// Sampling settings
#ifndef TRACY_NO_SAMPLING
#define TRACY_NO_SAMPLING 0  // Enable sampling profiler
#endif

// Network settings for remote profiling
#ifndef TRACY_NO_BROADCAST
#define TRACY_NO_BROADCAST 0  // Enable network discovery
#endif

// Custom zone color definitions for Vulkan-Forge
#define VF_TRACY_COLOR_RENDER    0x0066CC    // Blue for rendering
#define VF_TRACY_COLOR_MEMORY    0x00CC66    // Green for memory operations  
#define VF_TRACY_COLOR_IO        0xCC6600    // Orange for I/O operations
#define VF_TRACY_COLOR_COMPUTE   0xCC0066    // Purple for compute shaders
#define VF_TRACY_COLOR_TRANSFER  0x66CC00    // Lime for data transfers
#define VF_TRACY_COLOR_SYNC      0xFF3300    // Red for synchronization

// Frame rate targets for profiling
#define VF_TRACY_TARGET_FPS_60   16666666   // 60 FPS in nanoseconds
#define VF_TRACY_TARGET_FPS_120  8333333    // 120 FPS in nanoseconds
#define VF_TRACY_TARGET_FPS_144  6944444    // 144 FPS in nanoseconds

// Memory allocation tracking thresholds
#define VF_TRACY_ALLOC_THRESHOLD_BYTES  1024    // Track allocations >= 1KB
#define VF_TRACY_LARGE_ALLOC_BYTES      1048576 // Flag allocations >= 1MB

// GPU profiling configuration
#define VF_TRACY_VK_MAX_QUERIES         65536   // Maximum GPU queries per frame
#define VF_TRACY_VK_QUERY_POOL_SIZE     1024    // Query pool size

#endif // VULKAN_FORGE_PROFILING_ENABLED