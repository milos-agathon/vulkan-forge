#include "vf/vma_util.hpp"

namespace vf {

#define VMA_CHECK(call, msg) \
    do { VkResult _r = (call); if (_r != VK_SUCCESS) throw VulkanForgeError(msg, _r); } while (0)

VmaAllocator create_allocator(VkInstance instance,
                              VkPhysicalDevice phys,
                              VkDevice device)
{
    VmaAllocatorCreateInfo info{};
    info.instance = instance;
    info.physicalDevice = phys;
    info.device = device;
    info.vulkanApiVersion = VK_API_VERSION_1_0;
    VmaAllocator allocator{};
    VMA_CHECK(vmaCreateAllocator(&info, &allocator), "vmaCreateAllocator");
    return allocator;
}

VmaAllocation allocate_buffer(VmaAllocator allocator,
                              const VkBufferCreateInfo* info,
                              VkBuffer* out_buffer)
{
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocation allocation{};
    VMA_CHECK(vmaCreateBuffer(allocator, info, &ai, out_buffer, &allocation, nullptr),
              "vmaCreateBuffer");
    return allocation;
}

void destroy_allocator(VmaAllocator allocator)
{
    if (allocator) vmaDestroyAllocator(allocator);
}

} // namespace vf
