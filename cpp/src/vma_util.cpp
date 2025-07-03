#include "vf/vma_util.hpp"
#include <cstring>

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
    info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
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

MappedBuffer create_mapped_buffer(VmaAllocator allocator,
                                  VkDeviceSize size,
                                  VkBufferUsageFlags usage)
{
    MappedBuffer result{};
    result.size = size;
    
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    VmaAllocationInfo allocationInfo{};
    VMA_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                              &result.buffer, &result.allocation, &allocationInfo),
              "vmaCreateBuffer for mapped buffer");
    
    result.mapped_data = allocationInfo.pMappedData;
    if (!result.mapped_data) {
        vmaDestroyBuffer(allocator, result.buffer, result.allocation);
        throw VulkanForgeError("Failed to get mapped pointer", VK_ERROR_MEMORY_MAP_FAILED);
    }
    
    return result;
}

void destroy_mapped_buffer(VmaAllocator allocator, const MappedBuffer& buffer)
{
    if (buffer.buffer && buffer.allocation) {
        vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
    }
}

VkDeviceAddress get_buffer_device_address(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &addressInfo);
}

} // namespace vf
