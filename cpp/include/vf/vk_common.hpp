#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

namespace vf {

struct GpuContext {
    VkInstance instance {};
    VkPhysicalDevice phys {};
    VkDevice device {};
    VkQueue  queue {};
    uint32_t queueFamily {};
    VkCommandPool cmdPool {};
    // RAII ctor/dtor omitted for brevity
};

inline GpuContext& ctx();        // singleton accessor

/* Helpers */
VkCommandBuffer beginSingleTimeCmd();
void endSingleTimeCmd(VkCommandBuffer);

VkBuffer allocHostVisible(size_t bytes, VkBufferUsageFlags, VkDeviceMemory&);
VkBuffer allocDeviceLocal(size_t bytes, VkBufferUsageFlags, VkDeviceMemory&);
void    uploadToBuffer(VkBuffer dst, const void* src, size_t bytes);

} // namespace vf
