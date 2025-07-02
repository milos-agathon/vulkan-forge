#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace vf {

struct VulkanForgeError : std::runtime_error {
    VkResult result;
    VulkanForgeError(const char* msg, VkResult r)
        : std::runtime_error(msg), result(r) {}
};

VmaAllocator create_allocator(VkInstance instance,
                              VkPhysicalDevice phys,
                              VkDevice device);

VmaAllocation allocate_buffer(VmaAllocator allocator,
                              const VkBufferCreateInfo* info,
                              VkBuffer* out_buffer);

void destroy_allocator(VmaAllocator allocator);

} // namespace vf
