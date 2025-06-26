// cpp/include/vf/vk_common.hpp — updated again to expose chooseType
// -----------------------------------------------------------------
#ifndef VF_VK_COMMON_HPP
#define VF_VK_COMMON_HPP

#include <vulkan/vulkan.h>
#include <cstdint>

namespace vf {

// ------------------------------------------------------------------
// Global GPU context — one per process
// ------------------------------------------------------------------
struct GpuContext {
    VkInstance       instance   = VK_NULL_HANDLE;
    VkPhysicalDevice phys       = VK_NULL_HANDLE;
    VkDevice         device     = VK_NULL_HANDLE;
    uint32_t         gfxFamily  = 0;               // graphics-capable family index
    VkQueue          graphicsQ  = VK_NULL_HANDLE;  // graphics queue handle
};

// single global instance declared in vk_common.cpp
extern GpuContext g_ctx;
inline GpuContext& ctx() { return g_ctx; }

// ------------------------------------------------------------------
// Helper API (implemented in cpp/src/vk_common.cpp)
// ------------------------------------------------------------------

/* Memory-type chooser */
uint32_t chooseType(uint32_t bits, VkMemoryPropertyFlags props);

/* One-shot command buffers */
VkCommandBuffer beginSingleTimeCmd();
void            endSingleTimeCmd(VkCommandBuffer cb);

/* Buffer utilities */
VkBuffer allocDeviceLocal(
    VkDeviceSize       bytes,
    VkBufferUsageFlags usage,
    VkDeviceMemory&    outMem);

VkBuffer allocHostVisible(
    VkDeviceSize       bytes,
    VkBufferUsageFlags usage,
    VkDeviceMemory&    outMem);

void uploadToBuffer(
    VkBuffer     dst,
    const void*  src,
    VkDeviceSize bytes);

} // namespace vf

#endif // VF_VK_COMMON_HPP
