#include "vf/vertex_buffer.hpp"
#include "vf/vk_common.hpp"
#include <cstring>
#include <algorithm>

namespace vf {

// ============================================================================
// VertexLayout Implementation
// ============================================================================

VkVertexInputBindingDescription VertexLayout::getBindingDescription(uint32_t binding) const {
    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = binding;
    bindingDesc.stride = stride;
    bindingDesc.inputRate = inputRate;
    return bindingDesc;
}

std::vector<VkVertexInputAttributeDescription> VertexLayout::getAttributeDescriptions(uint32_t binding) const {
    std::vector<VkVertexInputAttributeDescription> attributeDescs;
    attributeDescs.reserve(attributes.size());
    
    for (const auto& attr : attributes) {
        VkVertexInputAttributeDescription desc{};
        desc.binding = binding;
        desc.location = attr.location;
        desc.format = attr.format;
        desc.offset = attr.offset;
        attributeDescs.push_back(desc);
    }
    
    return attributeDescs;
}

// ============================================================================
// Buffer Implementation
// ============================================================================

Buffer::~Buffer() {
    destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : device_(other.device_)
    , allocator_(other.allocator_)
    , buffer_(other.buffer_)
    , allocation_(other.allocation_)
    , size_(other.size_)
    , memoryUsage_(other.memoryUsage_) {
    
    other.device_ = VK_NULL_HANDLE;
    other.allocator_ = VK_NULL_HANDLE;
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        destroy();
        
        device_ = other.device_;
        allocator_ = other.allocator_;
        buffer_ = other.buffer_;
        allocation_ = other.allocation_;
        size_ = other.size_;
        memoryUsage_ = other.memoryUsage_;
        
        other.device_ = VK_NULL_HANDLE;
        other.allocator_ = VK_NULL_HANDLE;
        other.buffer_ = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.size_ = 0;
    }
    return *this;
}

VkResult Buffer::create(VkDevice device,
                       VmaAllocator allocator,
                       VkDeviceSize size,
                       VkBufferUsageFlags usage,
                       VmaMemoryUsage memoryUsage) {
    if (isValid()) {
        destroy();
    }
    
    device_ = device;
    allocator_ = allocator;
    size_ = size;
    memoryUsage_ = memoryUsage;
    
    // Create buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    // VMA allocation info
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;
    
    // Create buffer with VMA
    VkResult result = vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, 
                                     &buffer_, &allocation_, nullptr);
    
    if (result != VK_SUCCESS) {
        destroy();
        return result;
    }
    
    return VK_SUCCESS;
}

VkResult Buffer::uploadData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    if (!isValid() || !data || size == 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    
    if (offset + size > size_) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    
    // Check if memory is host visible
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(allocator_, allocation_, &allocInfo);
    
    VkMemoryPropertyFlags memProps;
    vmaGetMemoryTypeProperties(allocator_, allocInfo.memoryType, &memProps);
    
    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        // Direct mapping for host-visible memory
        void* mappedData;
        VkResult result = vmaMapMemory(allocator_, allocation_, &mappedData);
        if (result != VK_SUCCESS) {
            return result;
        }
        
        std::memcpy(static_cast<char*>(mappedData) + offset, data, size);
        
        // Flush if not coherent
        if (!(memProps & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            result = vmaFlushAllocation(allocator_, allocation_, offset, size);
        }
        
        vmaUnmapMemory(allocator_, allocation_);
        return result;
    } else {
        // Use staging buffer for device-local memory
        Buffer stagingBuffer;
        VkResult result = createStagingBuffer(size, stagingBuffer);
        if (result != VK_SUCCESS) {
            return result;
        }
        
        // Upload to staging buffer
        result = stagingBuffer.uploadData(data, size, 0);
        if (result != VK_SUCCESS) {
            return result;
        }
        
        // Copy from staging to device buffer
        result = copyBuffer(stagingBuffer.getBuffer(), buffer_, size);
        return result;
    }
}

VkResult Buffer::mapMemory(void** outData) {
    if (!isValid() || !outData) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    
    return vmaMapMemory(allocator_, allocation_, outData);
}

void Buffer::unmapMemory() {
    if (isValid()) {
        vmaUnmapMemory(allocator_, allocation_);
    }
}

void Buffer::destroy() {
    if (isValid()) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
    
    device_ = VK_NULL_HANDLE;
    allocator_ = VK_NULL_HANDLE;
    size_ = 0;
}

VkResult Buffer::createStagingBuffer(VkDeviceSize size, Buffer& stagingBuffer) {
    return stagingBuffer.create(
        device_,
        allocator_,
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY
    );
}

VkResult Buffer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    // TODO: This needs command pool and queue - will be provided by VertexBuffer
    // For now, return success (implementation will be completed when integrated)
    return VK_SUCCESS;
}

// ============================================================================
// VertexBuffer Implementation
// ============================================================================

VkResult VertexBuffer::initialize(VkDevice device,
                                 VmaAllocator allocator,
                                 VkCommandPool commandPool,
                                 VkQueue queue) {
    device_ = device;
    allocator_ = allocator;
    commandPool_ = commandPool;
    queue_ = queue;
    
    return VK_SUCCESS;
}

VkResult VertexBuffer::createVertexBuffer(const void* vertices,
                                         uint32_t vertexCount,
                                         const VertexLayout& layout) {
    if (!vertices || vertexCount == 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    
    vertexCount_ = vertexCount;
    layout_ = layout;
    
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(vertexCount) * layout.stride;
    
    // Create vertex buffer (device local for best performance)
    VkResult result = vertexBuffer_.create(
        device_,
        allocator_,
        bufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );
    
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Upload vertex data
    return vertexBuffer_.uploadData(vertices, bufferSize);
}

VkResult VertexBuffer::createIndexBuffer(const void* indices,
                                        uint32_t indexCount,
                                        VkIndexType indexType) {
    if (!indices || indexCount == 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    
    indexCount_ = indexCount;
    indexType_ = indexType;
    
    // Calculate buffer size based on index type
    VkDeviceSize indexSize = (indexType == VK_INDEX_TYPE_UINT16) ? sizeof(uint16_t) : sizeof(uint32_t);
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(indexCount) * indexSize;
    
    // Create index buffer (device local for best performance)
    VkResult result = indexBuffer_.create(
        device_,
        allocator_,
        bufferSize,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );
    
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Upload index data
    return indexBuffer_.uploadData(indices, bufferSize);
}

void VertexBuffer::bind(VkCommandBuffer commandBuffer, uint32_t firstBinding) const {
    if (!isValid()) {
        return;
    }
    
    // Bind vertex buffer
    VkBuffer vertexBuffers[] = { vertexBuffer_.getBuffer() };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, firstBinding, 1, vertexBuffers, offsets);
    
    // Bind index buffer if present
    if (indexBuffer_.isValid()) {
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer_.getBuffer(), 0, indexType_);
    }
}

void VertexBuffer::drawIndexed(VkCommandBuffer commandBuffer,
                              uint32_t instanceCount,
                              uint32_t firstInstance,
                              uint32_t indexOffset,
                              int32_t vertexOffset) const {
    if (!isValid()) {
        return;
    }
    
    if (indexBuffer_.isValid()) {
        vkCmdDrawIndexed(commandBuffer, indexCount_, instanceCount, 
                        indexOffset, vertexOffset, firstInstance);
    } else {
        // Non-indexed draw
        vkCmdDraw(commandBuffer, vertexCount_, instanceCount, 
                 vertexOffset, firstInstance);
    }
}

void VertexBuffer::destroy() {
    vertexBuffer_.destroy();
    indexBuffer_.destroy();
    
    vertexCount_ = 0;
    indexCount_ = 0;
    layout_ = VertexLayout(0);
}

// ============================================================================
// VertexLayouts Implementation
// ============================================================================

namespace VertexLayouts {

VertexLayout position3D() {
    VertexLayout layout(3 * sizeof(float));
    layout.addAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0);
    return layout;
}

VertexLayout positionUV() {
    VertexLayout layout(5 * sizeof(float));
    layout.addAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0);                    // position
    layout.addAttribute(1, VK_FORMAT_R32G32_SFLOAT, 3 * sizeof(float));      // UV
    return layout;
}

VertexLayout positionNormal() {
    VertexLayout layout(6 * sizeof(float));
    layout.addAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0);                    // position
    layout.addAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, 3 * sizeof(float));   // normal
    return layout;
}

VertexLayout positionNormalUV() {
    VertexLayout layout(8 * sizeof(float));
    layout.addAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0);                    // position
    layout.addAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, 3 * sizeof(float));   // normal
    layout.addAttribute(2, VK_FORMAT_R32G32_SFLOAT, 6 * sizeof(float));      // UV
    return layout;
}

VertexLayout positionColor() {
    VertexLayout layout(7 * sizeof(float));
    layout.addAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0);                    // position
    layout.addAttribute(1, VK_FORMAT_R32G32B32A32_SFLOAT, 3 * sizeof(float)); // color
    return layout;
}

} // namespace VertexLayouts

} // namespace vf