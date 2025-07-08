/**
 * @file terrain_memory_pool.hpp
 * @brief Specialized memory allocator for terrain rendering using VMA
 * 
 * Provides high-performance memory management for terrain data with:
 * - Dedicated memory pools for different terrain resource types
 * - Automatic memory pressure handling and defragmentation
 * - Streaming-friendly allocation patterns
 * - GPU memory bandwidth optimization
 * - Integration with Vulkan Memory Allocator (VMA)
 */

#pragma once

#include "vf/vk_common.hpp"
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>

namespace vf {

/**
 * @brief Types of terrain memory allocations
 */
enum class TerrainMemoryType {
    VertexBuffer,       // Vertex data for terrain meshes
    IndexBuffer,        // Index data for terrain meshes
    HeightTexture,      // Height map textures
    ColorTexture,       // Color/albedo textures
    NormalTexture,      // Normal map textures
    UniformBuffer,      // Uniform/constant buffers
    StagingBuffer,      // Temporary staging buffers
    ComputeBuffer       // Compute shader buffers
};

/**
 * @brief Memory allocation statistics
 */
struct TerrainMemoryStats {
    size_t totalAllocated = 0;      // Total memory allocated
    size_t totalUsed = 0;           // Total memory in use
    size_t totalFree = 0;           // Total free memory
    uint32_t activeAllocations = 0;  // Number of active allocations
    uint32_t poolCount = 0;         // Number of memory pools
    
    // Per-type statistics
    std::unordered_map<TerrainMemoryType, size_t> allocatedByType;
    std::unordered_map<TerrainMemoryType, size_t> usedByType;
    std::unordered_map<TerrainMemoryType, uint32_t> countByType;
    
    // Performance metrics
    std::chrono::milliseconds averageAllocTime{0};
    std::chrono::milliseconds averageFreeTime{0};
    uint64_t totalAllocations = 0;
    uint64_t totalDeallocations = 0;
    uint64_t failedAllocations = 0;
    
    double fragmentation = 0.0; // 0.0 = no fragmentation, 1.0 = highly fragmented
};

/**
 * @brief Configuration for terrain memory pools
 */
struct TerrainMemoryConfig {
    size_t maxTotalMemory = 2ULL * 1024 * 1024 * 1024; // 2GB default
    size_t maxPoolSize = 256ULL * 1024 * 1024;         // 256MB per pool
    size_t minBlockSize = 4 * 1024;                    // 4KB minimum
    size_t maxBlockSize = 64 * 1024 * 1024;            // 64MB maximum
    
    // Per-type configurations
    struct TypeConfig {
        size_t preferredPoolSize = 128ULL * 1024 * 1024; // 128MB
        size_t minPoolSize = 16ULL * 1024 * 1024;        // 16MB
        size_t allocationAlignment = 256;                // Default alignment
        bool enableDefragmentation = true;
        float growthFactor = 1.5f;                      // Pool growth multiplier
    };
    
    std::unordered_map<TerrainMemoryType, TypeConfig> typeConfigs;
    
    // Memory pressure thresholds
    float warningThreshold = 0.8f;    // Warn when 80% full
    float criticalThreshold = 0.95f;  // Critical when 95% full
    
    // Defragmentation settings
    bool enableAutoDefragmentation = true;
    float defragmentationThreshold = 0.3f; // Defrag when 30% fragmented
    size_t maxDefragmentationTime = 16;    // Max 16ms per frame
    
    TerrainMemoryConfig();
};

/**
 * @brief Memory allocation handle
 */
struct TerrainMemoryAllocation {
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo info = {};
    VkBuffer buffer = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    
    TerrainMemoryType type = TerrainMemoryType::VertexBuffer;
    size_t size = 0;
    size_t alignment = 0;
    uint32_t poolId = 0;
    
    std::chrono::high_resolution_clock::time_point allocTime;
    std::chrono::high_resolution_clock::time_point lastAccessTime;
    uint32_t accessCount = 0;
    
    bool isValid() const {
        return allocation != VK_NULL_HANDLE && 
               (buffer != VK_NULL_HANDLE || image != VK_NULL_HANDLE);
    }
    
    void markAccessed() {
        lastAccessTime = std::chrono::high_resolution_clock::now();
        accessCount++;
    }
};

/**
 * @brief Memory pool for specific terrain memory type
 */
class TerrainMemoryPool {
public:
    TerrainMemoryPool(TerrainMemoryType type, const TerrainMemoryConfig::TypeConfig& config);
    ~TerrainMemoryPool();
    
    // Allocation/deallocation
    VkResult allocateBuffer(VkBufferCreateInfo bufferInfo, 
                           VmaMemoryUsage usage,
                           std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    VkResult allocateImage(VkImageCreateInfo imageInfo,
                          VmaMemoryUsage usage,
                          std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    void deallocate(std::shared_ptr<TerrainMemoryAllocation> allocation);
    
    // Pool management
    VkResult resize(size_t newSize);
    VkResult defragment(size_t maxTimeMs);
    void cleanup();
    
    // Statistics
    size_t getTotalSize() const { return m_totalSize; }
    size_t getUsedSize() const { return m_usedSize; }
    size_t getFreeSize() const { return m_totalSize - m_usedSize; }
    float getFragmentation() const;
    uint32_t getActiveAllocations() const { return m_activeAllocations; }
    
    TerrainMemoryType getType() const { return m_type; }
    bool canAllocate(size_t size, size_t alignment) const;
    
private:
    TerrainMemoryType m_type;
    TerrainMemoryConfig::TypeConfig m_config;
    
    VmaPool m_vmaPool = VK_NULL_HANDLE;
    size_t m_totalSize = 0;
    std::atomic<size_t> m_usedSize{0};
    std::atomic<uint32_t> m_activeAllocations{0};
    
    mutable std::mutex m_mutex;
    std::vector<std::weak_ptr<TerrainMemoryAllocation>> m_allocations;
    
    VkResult createPool();
    void destroyPool();
    void updateStatistics();
};

/**
 * @brief Main terrain memory allocator
 */
class TerrainMemoryAllocator {
public:
    TerrainMemoryAllocator();
    ~TerrainMemoryAllocator();
    
    // Initialization
    VkResult initialize(const TerrainMemoryConfig& config = {});
    void destroy();
    
    // Buffer allocation
    VkResult allocateVertexBuffer(size_t size, 
                                 std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    VkResult allocateIndexBuffer(size_t size,
                                std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    VkResult allocateUniformBuffer(size_t size,
                                  std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    VkResult allocateStagingBuffer(size_t size,
                                  std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    // Texture allocation
    VkResult allocateTexture2D(uint32_t width, uint32_t height, VkFormat format,
                              VkImageUsageFlags usage, TerrainMemoryType type,
                              std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    VkResult allocateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers,
                                   VkFormat format, VkImageUsageFlags usage,
                                   TerrainMemoryType type,
                                   std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    // Generic allocation
    VkResult allocateBuffer(const VkBufferCreateInfo& bufferInfo,
                           VmaMemoryUsage usage, TerrainMemoryType type,
                           std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    VkResult allocateImage(const VkImageCreateInfo& imageInfo,
                          VmaMemoryUsage usage, TerrainMemoryType type,
                          std::shared_ptr<TerrainMemoryAllocation>& allocation);
    
    // Memory management
    void deallocate(std::shared_ptr<TerrainMemoryAllocation> allocation);
    void garbageCollect();
    VkResult defragment(size_t maxTimeMs = 16);
    
    // Memory pressure handling
    void handleMemoryPressure();
    bool isMemoryPressure() const;
    bool isCriticalMemoryPressure() const;
    
    // Statistics and monitoring
    TerrainMemoryStats getStats() const;
    void resetStats();
    float getMemoryUsageRatio() const;
    
    // Configuration
    void updateConfig(const TerrainMemoryConfig& config);
    const TerrainMemoryConfig& getConfig() const { return m_config; }
    
    // Callbacks for memory events
    using MemoryPressureCallback = std::function<void(float usageRatio)>;
    using AllocationFailedCallback = std::function<void(TerrainMemoryType, size_t)>;
    
    void setMemoryPressureCallback(MemoryPressureCallback callback) {
        m_memoryPressureCallback = callback;
    }
    
    void setAllocationFailedCallback(AllocationFailedCallback callback) {
        m_allocationFailedCallback = callback;
    }
    
    // Debugging
    void dumpMemoryInfo() const;
    std::vector<std::string> getMemoryReport() const;
    
private:
    bool m_initialized = false;
    TerrainMemoryConfig m_config;
    
    VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
    
    // Memory pools by type
    std::unordered_map<TerrainMemoryType, std::unique_ptr<TerrainMemoryPool>> m_pools;
    
    // Global statistics
    mutable std::mutex m_statsMutex;
    TerrainMemoryStats m_stats;
    
    // Callbacks
    MemoryPressureCallback m_memoryPressureCallback;
    AllocationFailedCallback m_allocationFailedCallback;
    
    // Background management
    std::atomic<bool> m_backgroundThreadRunning{false};
    std::thread m_backgroundThread;
    
    // Internal methods
    TerrainMemoryPool* getPool(TerrainMemoryType type);
    VkResult createPool(TerrainMemoryType type);
    void updateGlobalStats();
    void backgroundWorker();
    void startBackgroundThread();
    void stopBackgroundThread();
    
    // Memory pressure detection
    void checkMemoryPressure();
    void notifyMemoryPressure(float ratio);
    void notifyAllocationFailed(TerrainMemoryType type, size_t size);
};

/**
 * @brief RAII helper for terrain memory allocation
 */
class TerrainMemoryScope {
public:
    TerrainMemoryScope(TerrainMemoryAllocator& allocator);
    ~TerrainMemoryScope();
    
    // Buffer allocation helpers
    VkResult allocateVertexBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory);
    VkResult allocateIndexBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory);
    VkResult allocateUniformBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory);
    VkResult allocateStagingBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory);
    
    // Texture allocation helpers
    VkResult allocateTexture2D(uint32_t width, uint32_t height, VkFormat format,
                              VkImageUsageFlags usage, TerrainMemoryType type,
                              VkImage& image, VkDeviceMemory& memory);
    
    // Automatic cleanup on scope exit
    void setAutoCleanup(bool enable) { m_autoCleanup = enable; }
    
private:
    TerrainMemoryAllocator& m_allocator;
    std::vector<std::shared_ptr<TerrainMemoryAllocation>> m_allocations;
    bool m_autoCleanup = true;
};

/**
 * @brief Global terrain memory allocator instance
 */
class GlobalTerrainMemory {
public:
    static TerrainMemoryAllocator& getInstance();
    static VkResult initialize(const TerrainMemoryConfig& config = {});
    static void destroy();
    
private:
    static std::unique_ptr<TerrainMemoryAllocator> s_instance;
    static std::mutex s_mutex;
};

/**
 * @brief Utility functions for terrain memory management
 */
namespace TerrainMemoryUtils {
    
    /**
     * @brief Calculate optimal pool size based on usage patterns
     */
    size_t calculateOptimalPoolSize(TerrainMemoryType type, 
                                   const std::vector<size_t>& allocationSizes);
    
    /**
     * @brief Get recommended configuration for specific hardware
     */
    TerrainMemoryConfig getRecommendedConfig(VkPhysicalDevice physicalDevice);
    
    /**
     * @brief Analyze allocation patterns and suggest optimizations
     */
    struct AllocationAnalysis {
        float averageSize = 0.0f;
        float medianSize = 0.0f;
        size_t maxSize = 0;
        size_t minSize = 0;
        float fragmentation = 0.0f;
        std::vector<std::string> recommendations;
    };
    
    AllocationAnalysis analyzeAllocations(const TerrainMemoryStats& stats,
                                        TerrainMemoryType type);
    
    /**
     * @brief Memory alignment utilities
     */
    size_t getOptimalAlignment(TerrainMemoryType type);
    size_t alignSize(size_t size, size_t alignment);
    bool isAligned(size_t offset, size_t alignment);
    
    /**
     * @brief Convert between different units
     */
    std::string formatBytes(size_t bytes);
    size_t parseMemorySize(const std::string& sizeStr);
    
    /**
     * @brief Performance monitoring
     */
    struct PerformanceMetrics {
        double allocationsPerSecond = 0.0;
        double deallocationsPerSecond = 0.0;
        double averageAllocationTime = 0.0; // milliseconds
        double averageDeallocationTime = 0.0; // milliseconds
        double memoryBandwidth = 0.0; // GB/s
    };
    
    PerformanceMetrics calculatePerformanceMetrics(const TerrainMemoryStats& stats,
                                                 std::chrono::duration<double> timespan);
}

} // namespace vf