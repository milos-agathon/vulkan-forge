#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace vf {

// Format conversion utilities
VkFormat dtype_to_vk_format(const std::string& dtype_str, size_t itemsize, size_t components = 1);
std::string vk_format_to_dtype(VkFormat format);

// Stride and offset calculation for vertex attributes
struct VertexAttribute {
    uint32_t location;
    VkFormat format;
    uint32_t offset;
    uint32_t binding;
};

// Buffer view for typed access
struct BufferView {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
    VkFormat format;
    uint32_t count;  // Number of elements
};

// Memory barrier helper
struct MemoryBarrier {
    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;
    VkAccessFlags srcAccess;
    VkAccessFlags dstAccess;
};

// NumPy array info
struct ArrayInfo {
    void* data;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::string dtype;
    size_t itemsize;
    bool readonly;
    size_t total_bytes;
    
    bool is_contiguous() const;
    size_t element_count() const;
};

// GPU buffer with CPU mapping support
class MappedGPUBuffer {
public:
    MappedGPUBuffer(VmaAllocator allocator, 
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    bool host_visible = true);
    ~MappedGPUBuffer();
    
    // Non-copyable
    MappedGPUBuffer(const MappedGPUBuffer&) = delete;
    MappedGPUBuffer& operator=(const MappedGPUBuffer&) = delete;
    
    // Move semantics
    MappedGPUBuffer(MappedGPUBuffer&& other) noexcept;
    MappedGPUBuffer& operator=(MappedGPUBuffer&& other) noexcept;
    
    // Buffer access
    VkBuffer get_buffer() const { return m_buffer; }
    VkDeviceSize get_size() const { return m_size; }
    void* get_mapped_data() const { return m_mapped_data; }
    VkDeviceAddress get_device_address() const;
    
    // Memory operations
    void write(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    void read(void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    
    // Synchronization
    void barrier(VkCommandBuffer cmd, const MemoryBarrier& barrier);
    
private:
    VmaAllocator m_allocator = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = nullptr;
    void* m_mapped_data = nullptr;
    VkDeviceSize m_size = 0;
    bool m_host_visible = true;
    VkDevice m_device = VK_NULL_HANDLE;
};

// NumPy array to GPU buffer bridge
class NumpyBridge {
public:
    static std::unique_ptr<MappedGPUBuffer> create_from_array(
        VmaAllocator allocator,
        const ArrayInfo& info,
        VkBufferUsageFlags usage);
    
    static BufferView create_buffer_view(
        const MappedGPUBuffer& buffer,
        const ArrayInfo& info,
        uint32_t component_count = 1);
    
    static std::vector<VertexAttribute> create_vertex_attributes(
        const std::vector<ArrayInfo>& arrays,
        const std::vector<uint32_t>& locations,
        uint32_t binding = 0);
    
    // Utility to check if direct mapping is possible
    static bool can_zero_copy(const ArrayInfo& info);
    
    // Copy utilities for non-contiguous arrays
    static void copy_strided_to_packed(
        const ArrayInfo& info,
        void* dest);
    
    static void copy_packed_to_strided(
        const void* src,
        const ArrayInfo& info);
};

// RAII wrapper for command buffer recording
class ScopedCommandBuffer {
public:
    ScopedCommandBuffer(VkCommandPool pool, VkQueue queue);
    ~ScopedCommandBuffer();
    
    VkCommandBuffer get() const { return m_cmd; }
    operator VkCommandBuffer() const { return m_cmd; }
    
    void submit_and_wait();
    
private:
    VkDevice m_device;
    VkCommandPool m_pool;
    VkQueue m_queue;
    VkCommandBuffer m_cmd;
    bool m_submitted = false;
};

} // namespace vf