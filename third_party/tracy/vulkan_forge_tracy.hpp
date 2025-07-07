/**
 * @file vulkan_forge_tracy.hpp  
 * @brief Vulkan-Forge Tracy Integration Wrapper
 * 
 * This header provides a clean interface for Tracy profiling in Vulkan-Forge.
 * It includes both CPU and GPU profiling macros, with no-op fallbacks when
 * profiling is disabled.
 */

#pragma once

#include "tracy_config.hpp"

#if defined(VULKAN_FORGE_PROFILING_ENABLED) && VULKAN_FORGE_PROFILING_ENABLED

// IMPORTANT: Include Vulkan headers BEFORE Tracy headers
#include <vulkan/vulkan.h>

// Now include Tracy headers
#include "upstream/public/tracy/Tracy.hpp"
#include "upstream/public/tracy/TracyVulkan.hpp"

#include <memory>
#include <string_view>

namespace vulkan_forge::profiling {

/**
 * @brief Tracy context wrapper for Vulkan GPU profiling
 */
class VulkanTracyContext {
private:
    TracyVkCtx m_context = nullptr;
    
public:
    VulkanTracyContext() = default;
    
    bool initialize(VkPhysicalDevice physical_device, VkDevice device, 
                   VkQueue queue, VkCommandBuffer setup_cmd_buffer) {
        // Use the correct TracyVkContext macro signature
        m_context = TracyVkContextCalibrated(physical_device, device, queue, setup_cmd_buffer, 
                                           vkGetPhysicalDeviceProperties, vkGetInstanceProcAddr);
        return m_context != nullptr;
    }
    
    void destroy() {
        if (m_context) {
            TracyVkDestroy(m_context);
            m_context = nullptr;
        }
    }
    
    TracyVkCtx get() const { return m_context; }
    bool is_valid() const { return m_context != nullptr; }
    
    ~VulkanTracyContext() {
        destroy();
    }
    
    // Non-copyable, movable
    VulkanTracyContext(const VulkanTracyContext&) = delete;
    VulkanTracyContext& operator=(const VulkanTracyContext&) = delete;
    VulkanTracyContext(VulkanTracyContext&& other) noexcept : m_context(other.m_context) {
        other.m_context = nullptr;
    }
    VulkanTracyContext& operator=(VulkanTracyContext&& other) noexcept {
        if (this != &other) {
            destroy();
            m_context = other.m_context;
            other.m_context = nullptr;
        }
        return *this;
    }
};

/**
 * @brief Global Tracy context for the renderer
 */
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
// GPU Profiling Macros (Vulkan)
// ============================================================================

/// Initialize Tracy Vulkan context (call once during renderer setup)
#define VF_TRACY_VK_INIT(physical_device, device, queue, cmd_buffer) \
    vulkan_forge::profiling::g_tracy_context.initialize(physical_device, device, queue, cmd_buffer)

/// Destroy Tracy Vulkan context (call during renderer shutdown)  
#define VF_TRACY_VK_DESTROY() \
    vulkan_forge::profiling::g_tracy_context.destroy()

/// Collect GPU profiling data (call once per frame)
#define VF_TRACY_VK_COLLECT(cmd_buffer) \
    do { \
        if (vulkan_forge::profiling::g_tracy_context.is_valid()) { \
            TracyVkCollect(vulkan_forge::profiling::g_tracy_context.get(), cmd_buffer); \
        } \
    } while(0)

/// Profile GPU zone with custom name
#define VF_TRACY_VK_ZONE(cmd_buffer, name) \
    TracyVkZone(vulkan_forge::profiling::g_tracy_context.get(), cmd_buffer, name)

/// Profile GPU zone with custom name and color
#define VF_TRACY_VK_ZONE_COLORED(cmd_buffer, name, color) \
    TracyVkZoneC(vulkan_forge::profiling::g_tracy_context.get(), cmd_buffer, name, color)

/// Profile GPU zones with Vulkan-Forge predefined colors
#define VF_TRACY_VK_ZONE_RENDER(cmd_buffer, name) \
    VF_TRACY_VK_ZONE_COLORED(cmd_buffer, name, VF_TRACY_COLOR_RENDER)
#define VF_TRACY_VK_ZONE_COMPUTE(cmd_buffer, name) \
    VF_TRACY_VK_ZONE_COLORED(cmd_buffer, name, VF_TRACY_COLOR_COMPUTE)
#define VF_TRACY_VK_ZONE_TRANSFER(cmd_buffer, name) \
    VF_TRACY_VK_ZONE_COLORED(cmd_buffer, name, VF_TRACY_COLOR_TRANSFER)

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