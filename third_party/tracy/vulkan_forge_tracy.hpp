/**
 * @file vulkan_forge_tracy.hpp  
 * @brief Vulkan-Forge Tracy Integration Wrapper (CPU Profiling Only)
 * 
 * This header provides a clean interface for Tracy CPU profiling in Vulkan-Forge.
 * GPU profiling will be added later once basic profiling is working.
 */

#pragma once

#include "tracy_config.hpp"

#if defined(VULKAN_FORGE_PROFILING_ENABLED) && VULKAN_FORGE_PROFILING_ENABLED

// Include Tracy CPU profiling only
#include "upstream/public/tracy/Tracy.hpp"

namespace vulkan_forge::profiling {

// Simple stub for future Vulkan integration
class VulkanTracyContext {
public:
    bool initialize(...) { return false; }  // Not implemented yet
    void destroy() {}
    bool is_valid() const { return false; }
};

extern VulkanTracyContext g_tracy_context;

} // namespace vulkan_forge::profiling

// ============================================================================
// CPU Profiling Macros
// ============================================================================

/// Mark frame boundaries for frame-based profiling
#define VF_TRACY_FRAME_MARK() FrameMark

/// Profile current scope with function name
#define VF_TRACY_ZONE() ZoneScoped

/// Profile current scope with custom name and color
#define VF_TRACY_ZONE_NAMED(name) ZoneScopedN(name)
#define VF_TRACY_ZONE_COLORED(name, color) ZoneScopedNC(name, color)

/// Profile current scope with Vulkan-Forge predefined colors
#define VF_TRACY_ZONE_RENDER(name) VF_TRACY_ZONE_COLORED(name, VF_TRACY_COLOR_RENDER)
#define VF_TRACY_ZONE_MEMORY(name) VF_TRACY_ZONE_COLORED(name, VF_TRACY_COLOR_MEMORY)
#define VF_TRACY_ZONE_IO(name) VF_TRACY_ZONE_COLORED(name, VF_TRACY_COLOR_IO)
#define VF_TRACY_ZONE_COMPUTE(name) VF_TRACY_ZONE_COLORED(name, VF_TRACY_COLOR_COMPUTE)

/// Add text to current zone
#define VF_TRACY_ZONE_TEXT(text) ZoneText(text, strlen(text))
#define VF_TRACY_ZONE_VALUE(value) ZoneValue(value)

/// Memory allocation tracking
#define VF_TRACY_ALLOC(ptr, size) TracyAlloc(ptr, size)
#define VF_TRACY_FREE(ptr) TracyFree(ptr)

/// Custom plotting
#define VF_TRACY_PLOT(name, value) TracyPlot(name, value)
#define VF_TRACY_PLOT_CONFIG(name, format, step, fill, color) TracyPlotConfig(name, format, step, fill, color)

/// Message logging
#define VF_TRACY_MESSAGE(text) TracyMessage(text, strlen(text))
#define VF_TRACY_MESSAGE_COLOR(text, color) TracyMessageC(text, strlen(text), color)

// ============================================================================
// GPU Profiling Macros (Stubs - Not Implemented Yet)
// ============================================================================

/// GPU profiling stubs - will be implemented later
#define VF_TRACY_VK_INIT(physical_device, device, queue, cmd_buffer) false
#define VF_TRACY_VK_DESTROY()
#define VF_TRACY_VK_COLLECT(cmd_buffer)
#define VF_TRACY_VK_ZONE(cmd_buffer, name)
#define VF_TRACY_VK_ZONE_COLORED(cmd_buffer, name, color)
#define VF_TRACY_VK_ZONE_RENDER(cmd_buffer, name)
#define VF_TRACY_VK_ZONE_COMPUTE(cmd_buffer, name)
#define VF_TRACY_VK_ZONE_TRANSFER(cmd_buffer, name)

#else // VULKAN_FORGE_PROFILING_ENABLED

// ============================================================================
// No-op macros when profiling is disabled
// ============================================================================

namespace vulkan_forge::profiling {
    class VulkanTracyContext {
    public:
        bool initialize(...) { return false; }
        void destroy() {}
        bool is_valid() const { return false; }
    };
    
    extern VulkanTracyContext g_tracy_context;
}

// CPU profiling no-ops
#define VF_TRACY_FRAME_MARK()
#define VF_TRACY_ZONE()
#define VF_TRACY_ZONE_NAMED(name)
#define VF_TRACY_ZONE_COLORED(name, color)
#define VF_TRACY_ZONE_RENDER(name)
#define VF_TRACY_ZONE_MEMORY(name)
#define VF_TRACY_ZONE_IO(name)
#define VF_TRACY_ZONE_COMPUTE(name)
#define VF_TRACY_ZONE_TEXT(text)
#define VF_TRACY_ZONE_VALUE(value)
#define VF_TRACY_ALLOC(ptr, size)
#define VF_TRACY_FREE(ptr)
#define VF_TRACY_PLOT(name, value)
#define VF_TRACY_PLOT_CONFIG(name, format, step, fill, color)
#define VF_TRACY_MESSAGE(text)
#define VF_TRACY_MESSAGE_COLOR(text, color)

// GPU profiling no-ops
#define VF_TRACY_VK_INIT(physical_device, device, queue, cmd_buffer) false
#define VF_TRACY_VK_DESTROY()
#define VF_TRACY_VK_COLLECT(cmd_buffer)
#define VF_TRACY_VK_ZONE(cmd_buffer, name)
#define VF_TRACY_VK_ZONE_COLORED(cmd_buffer, name, color)
#define VF_TRACY_VK_ZONE_RENDER(cmd_buffer, name)
#define VF_TRACY_VK_ZONE_COMPUTE(cmd_buffer, name)
#define VF_TRACY_VK_ZONE_TRANSFER(cmd_buffer, name)

#endif // VULKAN_FORGE_PROFILING_ENABLED

// ============================================================================
// Convenience utilities (always available)
// ============================================================================

namespace vulkan_forge::profiling {

/**
 * @brief Check if profiling is enabled at compile time
 */
constexpr bool is_enabled() {
#if defined(VULKAN_FORGE_PROFILING_ENABLED) && VULKAN_FORGE_PROFILING_ENABLED
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get profiling configuration as a string
 */
const char* get_config_string();

} // namespace vulkan_forge::profiling