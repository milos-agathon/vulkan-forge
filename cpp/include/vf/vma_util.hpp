#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdint>
#include <stdexcept>

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

// Zero-copy buffer creation for NumPy integration
struct MappedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    void* mapped_data;
    VkDeviceSize size;
};

MappedBuffer create_mapped_buffer(VmaAllocator allocator,
                                  VkDeviceSize size,
                                  VkBufferUsageFlags usage);

void destroy_mapped_buffer(VmaAllocator allocator,
                          const MappedBuffer& buffer);

// Get buffer device address for GPU access
VkDeviceAddress get_buffer_device_address(VkDevice device, VkBuffer buffer);

} // namespace vf
