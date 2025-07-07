/**
 * @file vulkan_forge_tracy.cpp
 * @brief Implementation of Vulkan-Forge Tracy integration
 */

#include "vulkan_forge_tracy.hpp"

namespace vulkan_forge::profiling {

// Global Tracy context instance
VulkanTracyContext g_tracy_context;

const char* get_config_string() {
#if defined(VULKAN_FORGE_PROFILING_ENABLED) && VULKAN_FORGE_PROFILING_ENABLED
    static const char* config = 
        "Tracy Profiling: ENABLED\n"
        #if defined(TRACY_ON_DEMAND)
        "Mode: On-demand\n"
        #else
        "Mode: Always-on\n"
        #endif
        #if defined(TRACY_CALLSTACK)
        "Callstacks: Enabled\n"
        #else
        "Callstacks: Disabled\n"  
        #endif
        #if defined(TRACY_NO_VULKAN)
        "Vulkan GPU Profiling: Disabled\n"
        #else
        "Vulkan GPU Profiling: Enabled\n"
        #endif
        #if defined(TRACY_NO_SAMPLING)
        "Sampling: Disabled\n"
        #else  
        "Sampling: Enabled\n"
        #endif
        ;
    return config;
#else
    return "Tracy Profiling: DISABLED";
#endif
}

} // namespace vulkan_forge::profiling

#if defined(VULKAN_FORGE_PROFILING_ENABLED) && VULKAN_FORGE_PROFILING_ENABLED

// Include Tracy implementation (this must be in exactly one source file)
#ifdef TRACY_ENABLE
#include "upstream/public/TracyClient.cpp"
#endif

#endif // VULKAN_FORGE_PROFILING_ENABLED