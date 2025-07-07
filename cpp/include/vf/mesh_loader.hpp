#pragma once

#include "vk_common.hpp"
#include "vma_util.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <cstdint>

namespace vf {

/**
 * @brief Vertex attribute description for pipeline creation
 */
struct VertexAttribute {
    uint32_t location;      ///< Shader attribute location
    VkFormat format;        ///< Vulkan format (e.g., VK_FORMAT_R32G32B32_SFLOAT)
    uint32_t offset;        ///< Byte offset in vertex structure
    
    VertexAttribute(uint32_t loc, VkFormat fmt, uint32_t off)
        : location(loc), format(fmt), offset(off) {}
};

/**
 * @brief Vertex input layout description
 */
struct VertexLayout {
    uint32_t stride;                        ///< Size of one vertex in bytes
    std::vector<VertexAttribute> attributes; ///< Vertex attributes
    VkVertexInputRate inputRate;            ///< Per-vertex or per-instance
    
    VertexLayout(uint32_t vertex_stride, 
                VkVertexInputRate rate = VK_VERTEX_INPUT_RATE_VERTEX)
        : stride(vertex_stride), inputRate(rate) {}
    
    /// Add a vertex attribute to the layout
    void addAttribute(uint32_t location, VkFormat format, uint32_t offset) {
        attributes.emplace_back(location, format, offset);
    }
    
    /// Get Vulkan vertex input binding description
    VkVertexInputBindingDescription getBindingDescription(uint32_t binding = 0) const;
    
    /// Get Vulkan vertex input attribute descriptions
    std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions(uint32_t binding = 0) const;
};

/**
 * @brief GPU buffer wrapper with VMA integration
 * 
 * Manages Vulkan buffers with automatic memory allocation using VMA.
 * Supports staging uploads, memory mapping, and proper cleanup.
 */
class Buffer {
public:
    Buffer() = default;
    ~Buffer();
    
    // Non-copyable but movable
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    
    /**
     * @brief Create a buffer with specified usage and memory properties
     * 
     * @param device Vulkan logical device
     * @param allocator VMA allocator
     * @param size Buffer size in bytes
     * @param usage Buffer usage flags
     * @param memoryUsage VMA memory usage pattern
     * @return VkResult creation result
     */
    VkResult create(VkDevice device,
                   VmaAllocator allocator,
                   VkDeviceSize size,
                   VkBufferUsageFlags usage,
                   VmaMemoryUsage memoryUsage);
    
    /**
     * @brief Upload data to buffer using staging if needed
     * 
     * @param data Source data pointer
     * @param size Data size in bytes
     * @param offset Offset in buffer (default: 0)
     * @return VkResult upload result
     */
    VkResult uploadData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    
    /**
     * @brief Map buffer memory for CPU access
     * 
     * @param outData Output pointer to mapped memory
     * @return VkResult mapping result
     */
    VkResult mapMemory(void** outData);
    
    /**
     * @brief Unmap buffer memory
     */
    void unmapMemory();
    
    /**
     * @brief Destroy buffer and free memory
     */
    void destroy();
    
    // Getters
    VkBuffer getBuffer() const { return buffer_; }
    VkDeviceSize getSize() const { return size_; }
    bool isValid() const { return buffer_ != VK_NULL_HANDLE; }
    
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VmaMemoryUsage memoryUsage_ = VMA_MEMORY_USAGE_UNKNOWN;
    
    /// Create staging buffer for uploads
    VkResult createStagingBuffer(VkDeviceSize size, Buffer& stagingBuffer);
    
    /// Copy data between buffers using command buffer
    VkResult copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
};

/**
 * @brief Vertex buffer manager for mesh rendering
 * 
 * Handles vertex and index buffer creation, uploads, and binding.
 * Optimized for static mesh data with staging buffer uploads.
 */
class VertexBuffer {
public:
    VertexBuffer() = default;
    ~VertexBuffer() = default;
    
    // Non-copyable but movable
    VertexBuffer(const VertexBuffer&) = delete;
    VertexBuffer& operator=(const VertexBuffer&) = delete;
    VertexBuffer(VertexBuffer&& other) noexcept = default;
    VertexBuffer& operator=(VertexBuffer&& other) noexcept = default;
    
    /**
     * @brief Initialize vertex buffer with device and allocator
     * 
     * @param device Vulkan logical device
     * @param allocator VMA allocator
     * @param commandPool Command pool for staging uploads
     * @param queue Graphics queue for staging uploads
     * @return VkResult initialization result
     */
    VkResult initialize(VkDevice device,
                       VmaAllocator allocator,
                       VkCommandPool commandPool,
                       VkQueue queue);
    
    /**
     * @brief Create vertex buffer from vertex data
     * 
     * @param vertices Pointer to vertex data
     * @param vertexCount Number of vertices
     * @param layout Vertex layout description
     * @return VkResult creation result
     */
    VkResult createVertexBuffer(const void* vertices,
                               uint32_t vertexCount,
                               const VertexLayout& layout);
    
    /**
     * @brief Create index buffer from index data
     * 
     * @param indices Pointer to index data
     * @param indexCount Number of indices
     * @param indexType Index type (VK_INDEX_TYPE_UINT16 or VK_INDEX_TYPE_UINT32)
     * @return VkResult creation result
     */
    VkResult createIndexBuffer(const void* indices,
                              uint32_t indexCount,
                              VkIndexType indexType);
    
    /**
     * @brief Bind vertex and index buffers to command buffer
     * 
     * @param commandBuffer Command buffer to bind to
     * @param firstBinding First vertex binding point (default: 0)
     */
    void bind(VkCommandBuffer commandBuffer, uint32_t firstBinding = 0) const;
    
    /**
     * @brief Draw indexed primitives
     * 
     * @param commandBuffer Command buffer to record draw command
     * @param instanceCount Number of instances to draw (default: 1)
     * @param firstInstance First instance ID (default: 0)
     * @param indexOffset Offset into index buffer (default: 0)
     * @param vertexOffset Offset added to vertex indices (default: 0)
     */
    void drawIndexed(VkCommandBuffer commandBuffer,
                    uint32_t instanceCount = 1,
                    uint32_t firstInstance = 0,
                    uint32_t indexOffset = 0,
                    int32_t vertexOffset = 0) const;
    
    /**
     * @brief Destroy all buffers and free memory
     */
    void destroy();
    
    // Getters
    uint32_t getVertexCount() const { return vertexCount_; }
    uint32_t getIndexCount() const { return indexCount_; }
    const VertexLayout& getLayout() const { return layout_; }
    VkIndexType getIndexType() const { return indexType_; }
    bool isValid() const { return vertexBuffer_.isValid(); }
    
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    
    Buffer vertexBuffer_;
    Buffer indexBuffer_;
    VertexLayout layout_{0};
    
    uint32_t vertexCount_ = 0;
    uint32_t indexCount_ = 0;
    VkIndexType indexType_ = VK_INDEX_TYPE_UINT32;
};

/**
 * @brief Predefined vertex layouts for common mesh formats
 */
namespace VertexLayouts {
    /// Position only (3x float32)
    VertexLayout position3D();
    
    /// Position + UV (3x float32 + 2x float32)
    VertexLayout positionUV();
    
    /// Position + Normal (3x float32 + 3x float32)
    VertexLayout positionNormal();
    
    /// Position + Normal + UV (3x float32 + 3x float32 + 2x float32)
    VertexLayout positionNormalUV();
    
    /// Position + Color (3x float32 + 4x float32)
    VertexLayout positionColor();
}

} // namespace vf