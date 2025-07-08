/**
 * @file terrain_memory_pool.cpp
 * @brief Implementation of specialized memory allocator for terrain rendering
 */

#include "vf/terrain_memory_pool.hpp"
#include "vf/vk_common.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace vf {

// TerrainMemoryConfig implementation
TerrainMemoryConfig::TerrainMemoryConfig() {
    // Initialize default configurations for each memory type
    
    // Vertex buffers - frequent access, GPU-only
    typeConfigs[TerrainMemoryType::VertexBuffer] = {
        .preferredPoolSize = 128ULL * 1024 * 1024,  // 128MB
        .minPoolSize = 16ULL * 1024 * 1024,         // 16MB
        .allocationAlignment = 256,
        .enableDefragmentation = true,
        .growthFactor = 1.5f
    };
    
    // Index buffers - similar to vertex buffers
    typeConfigs[TerrainMemoryType::IndexBuffer] = {
        .preferredPoolSize = 64ULL * 1024 * 1024,   // 64MB
        .minPoolSize = 8ULL * 1024 * 1024,          // 8MB
        .allocationAlignment = 256,
        .enableDefragmentation = true,
        .growthFactor = 1.5f
    };
    
    // Height textures - large, infrequent changes
    typeConfigs[TerrainMemoryType::HeightTexture] = {
        .preferredPoolSize = 256ULL * 1024 * 1024,  // 256MB
        .minPoolSize = 32ULL * 1024 * 1024,         // 32MB
        .allocationAlignment = 1024,
        .enableDefragmentation = false,             // Large allocations, avoid defrag
        .growthFactor = 2.0f
    };
    
    // Color textures
    typeConfigs[TerrainMemoryType::ColorTexture] = {
        .preferredPoolSize = 512ULL * 1024 * 1024,  // 512MB
        .minPoolSize = 64ULL * 1024 * 1024,         // 64MB
        .allocationAlignment = 1024,
        .enableDefragmentation = false,
        .growthFactor = 2.0f
    };
    
    // Normal textures
    typeConfigs[TerrainMemoryType::NormalTexture] = {
        .preferredPoolSize = 256ULL * 1024 * 1024,  // 256MB
        .minPoolSize = 32ULL * 1024 * 1024,         // 32MB
        .allocationAlignment = 1024,
        .enableDefragmentation = false,
        .growthFactor = 2.0f
    };
    
    // Uniform buffers - small, frequent updates
    typeConfigs[TerrainMemoryType::UniformBuffer] = {
        .preferredPoolSize = 16ULL * 1024 * 1024,   // 16MB
        .minPoolSize = 2ULL * 1024 * 1024,          // 2MB
        .allocationAlignment = 256,
        .enableDefragmentation = true,
        .growthFactor = 1.5f
    };
    
    // Staging buffers - temporary, CPU accessible
    typeConfigs[TerrainMemoryType::StagingBuffer] = {
        .preferredPoolSize = 64ULL * 1024 * 1024,   // 64MB
        .minPoolSize = 8ULL * 1024 * 1024,          // 8MB
        .allocationAlignment = 256,
        .enableDefragmentation = true,
        .growthFactor = 1.5f
    };
    
    // Compute buffers
    typeConfigs[TerrainMemoryType::ComputeBuffer] = {
        .preferredPoolSize = 32ULL * 1024 * 1024,   // 32MB
        .minPoolSize = 4ULL * 1024 * 1024,          // 4MB
        .allocationAlignment = 256,
        .enableDefragmentation = true,
        .growthFactor = 1.5f
    };
}

// TerrainMemoryPool implementation
TerrainMemoryPool::TerrainMemoryPool(TerrainMemoryType type, 
                                   const TerrainMemoryConfig::TypeConfig& config)
    : m_type(type), m_config(config) {
}

TerrainMemoryPool::~TerrainMemoryPool() {
    destroyPool();
}

VkResult TerrainMemoryPool::allocateBuffer(VkBufferCreateInfo bufferInfo,
                                         VmaMemoryUsage usage,
                                         std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Create pool if it doesn't exist
    if (m_vmaPool == VK_NULL_HANDLE) {
        VkResult result = createPool();
        if (result != VK_SUCCESS) {
            return result;
        }
    }
    
    // Check if we can allocate
    VkDeviceSize requiredSize = bufferInfo.size;
    VkDeviceSize alignment = m_config.allocationAlignment;
    
    if (!canAllocate(requiredSize, alignment)) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    
    auto& ctx = vk_common::context();
    auto allocator = ctx.vmaAllocator; // Assuming VMA allocator is available
    
    // Set up allocation info
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = usage;
    allocInfo.pool = m_vmaPool;
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    // Create allocation
    allocation = std::make_shared<TerrainMemoryAllocation>();
    allocation->type = m_type;
    allocation->size = requiredSize;
    allocation->alignment = alignment;
    allocation->allocTime = std::chrono::high_resolution_clock::now();
    
    VkResult result = vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                                    &allocation->buffer, &allocation->allocation,
                                    &allocation->info);
    
    if (result != VK_SUCCESS) {
        allocation.reset();
        return result;
    }
    
    // Update tracking
    m_usedSize += requiredSize;
    m_activeAllocations++;
    m_allocations.push_back(allocation);
    
    updateStatistics();
    
    return VK_SUCCESS;
}

VkResult TerrainMemoryPool::allocateImage(VkImageCreateInfo imageInfo,
                                        VmaMemoryUsage usage,
                                        std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Create pool if it doesn't exist
    if (m_vmaPool == VK_NULL_HANDLE) {
        VkResult result = createPool();
        if (result != VK_SUCCESS) {
            return result;
        }
    }
    
    auto& ctx = vk_common::context();
    auto allocator = ctx.vmaAllocator;
    
    // Set up allocation info
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = usage;
    allocInfo.pool = m_vmaPool;
    
    // Create allocation
    allocation = std::make_shared<TerrainMemoryAllocation>();
    allocation->type = m_type;
    allocation->allocTime = std::chrono::high_resolution_clock::now();
    
    VkResult result = vmaCreateImage(allocator, &imageInfo, &allocInfo,
                                   &allocation->image, &allocation->allocation,
                                   &allocation->info);
    
    if (result != VK_SUCCESS) {
        allocation.reset();
        return result;
    }
    
    allocation->size = allocation->info.size;
    allocation->alignment = allocation->info.alignment;
    
    // Update tracking
    m_usedSize += allocation->size;
    m_activeAllocations++;
    m_allocations.push_back(allocation);
    
    updateStatistics();
    
    return VK_SUCCESS;
}

void TerrainMemoryPool::deallocate(std::shared_ptr<TerrainMemoryAllocation> allocation) {
    if (!allocation || !allocation->isValid()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto& ctx = vk_common::context();
    auto allocator = ctx.vmaAllocator;
    
    // Destroy VMA allocation
    if (allocation->buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, allocation->buffer, allocation->allocation);
    } else if (allocation->image != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator, allocation->image, allocation->allocation);
    }
    
    // Update tracking
    m_usedSize -= allocation->size;
    m_activeAllocations--;
    
    // Remove from allocations list
    auto it = std::find_if(m_allocations.begin(), m_allocations.end(),
        [&allocation](const std::weak_ptr<TerrainMemoryAllocation>& weak) {
            auto shared = weak.lock();
            return shared && shared.get() == allocation.get();
        });
    
    if (it != m_allocations.end()) {
        m_allocations.erase(it);
    }
    
    updateStatistics();
}

VkResult TerrainMemoryPool::resize(size_t newSize) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (newSize < m_usedSize) {
        return VK_ERROR_VALIDATION_FAILED_EXT; // Cannot shrink below used size
    }
    
    // For VMA pools, we would need to recreate the pool
    // This is a simplified implementation
    m_totalSize = newSize;
    
    return VK_SUCCESS;
}

VkResult TerrainMemoryPool::defragment(size_t maxTimeMs) {
    if (!m_config.enableDefragmentation) {
        return VK_SUCCESS;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Clean up expired weak pointers
    m_allocations.erase(
        std::remove_if(m_allocations.begin(), m_allocations.end(),
            [](const std::weak_ptr<TerrainMemoryAllocation>& weak) {
                return weak.expired();
            }),
        m_allocations.end()
    );
    
    // VMA handles internal defragmentation
    // We just clean up our tracking
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    if (duration.count() > static_cast<long>(maxTimeMs)) {
        return VK_TIMEOUT; // Took too long
    }
    
    return VK_SUCCESS;
}

void TerrainMemoryPool::cleanup() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Remove expired allocations
    m_allocations.erase(
        std::remove_if(m_allocations.begin(), m_allocations.end(),
            [](const std::weak_ptr<TerrainMemoryAllocation>& weak) {
                return weak.expired();
            }),
        m_allocations.end()
    );
    
    updateStatistics();
}

float TerrainMemoryPool::getFragmentation() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_totalSize == 0) {
        return 0.0f;
    }
    
    // Simple fragmentation calculation
    // In a real implementation, this would analyze actual memory layout
    size_t largestFreeBlock = m_totalSize - m_usedSize;
    size_t totalFreeMemory = m_totalSize - m_usedSize;
    
    if (totalFreeMemory == 0) {
        return 0.0f;
    }
    
    return 1.0f - (static_cast<float>(largestFreeBlock) / totalFreeMemory);
}

bool TerrainMemoryPool::canAllocate(size_t size, size_t alignment) const {
    // Align size
    size_t alignedSize = (size + alignment - 1) & ~(alignment - 1);
    
    return (m_usedSize + alignedSize) <= m_totalSize;
}

VkResult TerrainMemoryPool::createPool() {
    auto& ctx = vk_common::context();
    auto allocator = ctx.vmaAllocator;
    
    VmaPoolCreateInfo poolInfo = {};
    poolInfo.memoryTypeIndex = 0; // Would need proper memory type selection
    poolInfo.blockSize = m_config.preferredPoolSize;
    poolInfo.minBlockCount = 1;
    poolInfo.maxBlockCount = 8; // Allow some growth
    
    VkResult result = vmaCreatePool(allocator, &poolInfo, &m_vmaPool);
    if (result == VK_SUCCESS) {
        m_totalSize = m_config.preferredPoolSize;
    }
    
    return result;
}

void TerrainMemoryPool::destroyPool() {
    if (m_vmaPool != VK_NULL_HANDLE) {
        auto& ctx = vk_common::context();
        vmaDestroyPool(ctx.vmaAllocator, m_vmaPool);
        m_vmaPool = VK_NULL_HANDLE;
    }
    
    m_totalSize = 0;
    m_usedSize = 0;
    m_activeAllocations = 0;
    m_allocations.clear();
}

void TerrainMemoryPool::updateStatistics() {
    // Statistics are updated by member variables
    // This method can be used for more complex calculations
}

// TerrainMemoryAllocator implementation
TerrainMemoryAllocator::TerrainMemoryAllocator() {
}

TerrainMemoryAllocator::~TerrainMemoryAllocator() {
    destroy();
}

VkResult TerrainMemoryAllocator::initialize(const TerrainMemoryConfig& config) {
    if (m_initialized) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    m_config = config;
    
    // Initialize VMA allocator (would be done in vk_common)
    // For now, assume it's already initialized
    
    // Create pools for each memory type
    for (const auto& [type, typeConfig] : config.typeConfigs) {
        VkResult result = createPool(type);
        if (result != VK_SUCCESS) {
            destroy();
            return result;
        }
    }
    
    // Start background management thread
    startBackgroundThread();
    
    m_initialized = true;
    return VK_SUCCESS;
}

void TerrainMemoryAllocator::destroy() {
    if (!m_initialized) {
        return;
    }
    
    // Stop background thread
    stopBackgroundThread();
    
    // Destroy all pools
    m_pools.clear();
    
    m_initialized = false;
}

VkResult TerrainMemoryAllocator::allocateVertexBuffer(size_t size,
                                                    std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    return allocateBuffer(bufferInfo, VMA_MEMORY_USAGE_GPU_ONLY, 
                         TerrainMemoryType::VertexBuffer, allocation);
}

VkResult TerrainMemoryAllocator::allocateIndexBuffer(size_t size,
                                                   std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    return allocateBuffer(bufferInfo, VMA_MEMORY_USAGE_GPU_ONLY,
                         TerrainMemoryType::IndexBuffer, allocation);
}

VkResult TerrainMemoryAllocator::allocateUniformBuffer(size_t size,
                                                     std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    return allocateBuffer(bufferInfo, VMA_MEMORY_USAGE_CPU_TO_GPU,
                         TerrainMemoryType::UniformBuffer, allocation);
}

VkResult TerrainMemoryAllocator::allocateStagingBuffer(size_t size,
                                                     std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    return allocateBuffer(bufferInfo, VMA_MEMORY_USAGE_CPU_ONLY,
                         TerrainMemoryType::StagingBuffer, allocation);
}

VkResult TerrainMemoryAllocator::allocateTexture2D(uint32_t width, uint32_t height, VkFormat format,
                                                  VkImageUsageFlags usage, TerrainMemoryType type,
                                                  std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    return allocateImage(imageInfo, VMA_MEMORY_USAGE_GPU_ONLY, type, allocation);
}

VkResult TerrainMemoryAllocator::allocateTexture2DArray(uint32_t width, uint32_t height, uint32_t layers,
                                                       VkFormat format, VkImageUsageFlags usage,
                                                       TerrainMemoryType type,
                                                       std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = layers;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    return allocateImage(imageInfo, VMA_MEMORY_USAGE_GPU_ONLY, type, allocation);
}

VkResult TerrainMemoryAllocator::allocateBuffer(const VkBufferCreateInfo& bufferInfo,
                                              VmaMemoryUsage usage, TerrainMemoryType type,
                                              std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    TerrainMemoryPool* pool = getPool(type);
    if (!pool) {
        if (notifyAllocationFailed) {
            notifyAllocationFailed(type, bufferInfo.size);
        }
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    
    VkBufferCreateInfo mutableBufferInfo = bufferInfo; // Copy for modification if needed
    
    VkResult result = pool->allocateBuffer(mutableBufferInfo, usage, allocation);
    
    if (result != VK_SUCCESS) {
        if (m_allocationFailedCallback) {
            m_allocationFailedCallback(type, bufferInfo.size);
        }
    } else {
        // Update global statistics
        updateGlobalStats();
        checkMemoryPressure();
    }
    
    return result;
}

VkResult TerrainMemoryAllocator::allocateImage(const VkImageCreateInfo& imageInfo,
                                             VmaMemoryUsage usage, TerrainMemoryType type,
                                             std::shared_ptr<TerrainMemoryAllocation>& allocation) {
    TerrainMemoryPool* pool = getPool(type);
    if (!pool) {
        if (m_allocationFailedCallback) {
            m_allocationFailedCallback(type, 0); // Unknown size for images
        }
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    
    VkImageCreateInfo mutableImageInfo = imageInfo; // Copy for modification if needed
    
    VkResult result = pool->allocateImage(mutableImageInfo, usage, allocation);
    
    if (result != VK_SUCCESS) {
        if (m_allocationFailedCallback) {
            m_allocationFailedCallback(type, 0);
        }
    } else {
        updateGlobalStats();
        checkMemoryPressure();
    }
    
    return result;
}

void TerrainMemoryAllocator::deallocate(std::shared_ptr<TerrainMemoryAllocation> allocation) {
    if (!allocation) {
        return;
    }
    
    TerrainMemoryPool* pool = getPool(allocation->type);
    if (pool) {
        pool->deallocate(allocation);
        updateGlobalStats();
    }
}

void TerrainMemoryAllocator::garbageCollect() {
    for (auto& [type, pool] : m_pools) {
        pool->cleanup();
    }
    updateGlobalStats();
}

VkResult TerrainMemoryAllocator::defragment(size_t maxTimeMs) {
    if (!m_config.enableAutoDefragmentation) {
        return VK_SUCCESS;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    size_t timePerPool = maxTimeMs / std::max(1ul, m_pools.size());
    
    for (auto& [type, pool] : m_pools) {
        auto poolStartTime = std::chrono::high_resolution_clock::now();
        
        VkResult result = pool->defragment(timePerPool);
        if (result == VK_TIMEOUT) {
            break; // Stop if we're taking too long
        }
        
        auto poolEndTime = std::chrono::high_resolution_clock::now();
        auto poolDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            poolEndTime - poolStartTime);
        
        if (poolDuration.count() > static_cast<long>(maxTimeMs)) {
            break; // Spent too much time already
        }
    }
    
    updateGlobalStats();
    return VK_SUCCESS;
}

void TerrainMemoryAllocator::handleMemoryPressure() {
    // Perform aggressive cleanup
    garbageCollect();
    
    // Force defragmentation
    defragment(50); // Allow up to 50ms for emergency defragmentation
    
    // Additional pressure relief strategies could include:
    // - Evicting least recently used allocations
    // - Reducing quality settings
    // - Requesting application to free resources
}

bool TerrainMemoryAllocator::isMemoryPressure() const {
    return getMemoryUsageRatio() > m_config.warningThreshold;
}

bool TerrainMemoryAllocator::isCriticalMemoryPressure() const {
    return getMemoryUsageRatio() > m_config.criticalThreshold;
}

TerrainMemoryStats TerrainMemoryAllocator::getStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

void TerrainMemoryAllocator::resetStats() {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    m_stats = TerrainMemoryStats{};
}

float TerrainMemoryAllocator::getMemoryUsageRatio() const {
    if (m_config.maxTotalMemory == 0) {
        return 0.0f;
    }
    
    size_t totalUsed = 0;
    for (const auto& [type, pool] : m_pools) {
        totalUsed += pool->getUsedSize();
    }
    
    return static_cast<float>(totalUsed) / m_config.maxTotalMemory;
}

void TerrainMemoryAllocator::updateConfig(const TerrainMemoryConfig& config) {
    m_config = config;
    
    // Update individual pool configurations
    for (auto& [type, pool] : m_pools) {
        auto it = config.typeConfigs.find(type);
        if (it != config.typeConfigs.end()) {
            // Pool config update would require pool recreation in full implementation
        }
    }
}

void TerrainMemoryAllocator::dumpMemoryInfo() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    
    printf("=== Terrain Memory Allocator Info ===\n");
    printf("Total Allocated: %s\n", TerrainMemoryUtils::formatBytes(m_stats.totalAllocated).c_str());
    printf("Total Used: %s\n", TerrainMemoryUtils::formatBytes(m_stats.totalUsed).c_str());
    printf("Total Free: %s\n", TerrainMemoryUtils::formatBytes(m_stats.totalFree).c_str());
    printf("Active Allocations: %u\n", m_stats.activeAllocations);
    printf("Memory Usage Ratio: %.2f%%\n", getMemoryUsageRatio() * 100.0f);
    printf("Fragmentation: %.2f%%\n", m_stats.fragmentation * 100.0f);
    
    printf("\nPer-Type Statistics:\n");
    for (const auto& [type, used] : m_stats.usedByType) {
        printf("  %s: %s used\n", 
               TerrainMemoryUtils::getTypeName(type).c_str(),
               TerrainMemoryUtils::formatBytes(used).c_str());
    }
}

std::vector<std::string> TerrainMemoryAllocator::getMemoryReport() const {
    std::vector<std::string> report;
    std::lock_guard<std::mutex> lock(m_statsMutex);
    
    report.push_back("Terrain Memory Allocator Report");
    report.push_back("================================");
    
    report.push_back("Global Statistics:");
    report.push_back("  Total Allocated: " + TerrainMemoryUtils::formatBytes(m_stats.totalAllocated));
    report.push_back("  Total Used: " + TerrainMemoryUtils::formatBytes(m_stats.totalUsed));
    report.push_back("  Usage Ratio: " + std::to_string(getMemoryUsageRatio() * 100.0f) + "%");
    report.push_back("  Fragmentation: " + std::to_string(m_stats.fragmentation * 100.0f) + "%");
    
    report.push_back("\nPool Statistics:");
    for (const auto& [type, pool] : m_pools) {
        std::string typeName = TerrainMemoryUtils::getTypeName(type);
        report.push_back("  " + typeName + ":");
        report.push_back("    Total Size: " + TerrainMemoryUtils::formatBytes(pool->getTotalSize()));
        report.push_back("    Used Size: " + TerrainMemoryUtils::formatBytes(pool->getUsedSize()));
        report.push_back("    Free Size: " + TerrainMemoryUtils::formatBytes(pool->getFreeSize()));
        report.push_back("    Fragmentation: " + std::to_string(pool->getFragmentation() * 100.0f) + "%");
        report.push_back("    Active Allocations: " + std::to_string(pool->getActiveAllocations()));
    }
    
    return report;
}

TerrainMemoryPool* TerrainMemoryAllocator::getPool(TerrainMemoryType type) {
    auto it = m_pools.find(type);
    return (it != m_pools.end()) ? it->second.get() : nullptr;
}

VkResult TerrainMemoryAllocator::createPool(TerrainMemoryType type) {
    auto it = m_config.typeConfigs.find(type);
    if (it == m_config.typeConfigs.end()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    auto pool = std::make_unique<TerrainMemoryPool>(type, it->second);
    m_pools[type] = std::move(pool);
    
    return VK_SUCCESS;
}

void TerrainMemoryAllocator::updateGlobalStats() {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    
    m_stats.totalAllocated = 0;
    m_stats.totalUsed = 0;
    m_stats.totalFree = 0;
    m_stats.activeAllocations = 0;
    m_stats.poolCount = static_cast<uint32_t>(m_pools.size());
    
    m_stats.allocatedByType.clear();
    m_stats.usedByType.clear();
    m_stats.countByType.clear();
    
    float totalFragmentation = 0.0f;
    
    for (const auto& [type, pool] : m_pools) {
        size_t poolTotal = pool->getTotalSize();
        size_t poolUsed = pool->getUsedSize();
        size_t poolFree = pool->getFreeSize();
        uint32_t poolCount = pool->getActiveAllocations();
        
        m_stats.totalAllocated += poolTotal;
        m_stats.totalUsed += poolUsed;
        m_stats.totalFree += poolFree;
        m_stats.activeAllocations += poolCount;
        
        m_stats.allocatedByType[type] = poolTotal;
        m_stats.usedByType[type] = poolUsed;
        m_stats.countByType[type] = poolCount;
        
        totalFragmentation += pool->getFragmentation();
    }
    
    // Average fragmentation across pools
    m_stats.fragmentation = m_pools.empty() ? 0.0f : (totalFragmentation / m_pools.size());
}

void TerrainMemoryAllocator::backgroundWorker() {
    auto lastDefragTime = std::chrono::high_resolution_clock::now();
    
    while (m_backgroundThreadRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Periodic garbage collection
        garbageCollect();
        
        // Periodic defragmentation
        auto now = std::chrono::high_resolution_clock::now();
        auto timeSinceDefrag = std::chrono::duration_cast<std::chrono::seconds>(now - lastDefragTime);
        
        if (timeSinceDefrag.count() > 5) { // Every 5 seconds
            if (m_stats.fragmentation > m_config.defragmentationThreshold) {
                defragment(m_config.maxDefragmentationTime);
            }
            lastDefragTime = now;
        }
        
        // Check memory pressure
        checkMemoryPressure();
    }
}

void TerrainMemoryAllocator::startBackgroundThread() {
    m_backgroundThreadRunning = true;
    m_backgroundThread = std::thread(&TerrainMemoryAllocator::backgroundWorker, this);
}

void TerrainMemoryAllocator::stopBackgroundThread() {
    m_backgroundThreadRunning = false;
    if (m_backgroundThread.joinable()) {
        m_backgroundThread.join();
    }
}

void TerrainMemoryAllocator::checkMemoryPressure() {
    float usageRatio = getMemoryUsageRatio();
    
    if (usageRatio > m_config.criticalThreshold) {
        handleMemoryPressure();
        notifyMemoryPressure(usageRatio);
    } else if (usageRatio > m_config.warningThreshold) {
        notifyMemoryPressure(usageRatio);
    }
}

void TerrainMemoryAllocator::notifyMemoryPressure(float ratio) {
    if (m_memoryPressureCallback) {
        m_memoryPressureCallback(ratio);
    }
}

void TerrainMemoryAllocator::notifyAllocationFailed(TerrainMemoryType type, size_t size) {
    if (m_allocationFailedCallback) {
        m_allocationFailedCallback(type, size);
    }
}

// TerrainMemoryScope implementation
TerrainMemoryScope::TerrainMemoryScope(TerrainMemoryAllocator& allocator)
    : m_allocator(allocator) {
}

TerrainMemoryScope::~TerrainMemoryScope() {
    if (m_autoCleanup) {
        for (auto& allocation : m_allocations) {
            m_allocator.deallocate(allocation);
        }
    }
}

VkResult TerrainMemoryScope::allocateVertexBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory) {
    std::shared_ptr<TerrainMemoryAllocation> allocation;
    VkResult result = m_allocator.allocateVertexBuffer(size, allocation);
    
    if (result == VK_SUCCESS) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        m_allocations.push_back(allocation);
    }
    
    return result;
}

VkResult TerrainMemoryScope::allocateIndexBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory) {
    std::shared_ptr<TerrainMemoryAllocation> allocation;
    VkResult result = m_allocator.allocateIndexBuffer(size, allocation);
    
    if (result == VK_SUCCESS) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        m_allocations.push_back(allocation);
    }
    
    return result;
}

VkResult TerrainMemoryScope::allocateUniformBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory) {
    std::shared_ptr<TerrainMemoryAllocation> allocation;
    VkResult result = m_allocator.allocateUniformBuffer(size, allocation);
    
    if (result == VK_SUCCESS) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        m_allocations.push_back(allocation);
    }
    
    return result;
}

VkResult TerrainMemoryScope::allocateStagingBuffer(size_t size, VkBuffer& buffer, VkDeviceMemory& memory) {
    std::shared_ptr<TerrainMemoryAllocation> allocation;
    VkResult result = m_allocator.allocateStagingBuffer(size, allocation);
    
    if (result == VK_SUCCESS) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        m_allocations.push_back(allocation);
    }
    
    return result;
}

VkResult TerrainMemoryScope::allocateTexture2D(uint32_t width, uint32_t height, VkFormat format,
                                              VkImageUsageFlags usage, TerrainMemoryType type,
                                              VkImage& image, VkDeviceMemory& memory) {
    std::shared_ptr<TerrainMemoryAllocation> allocation;
    VkResult result = m_allocator.allocateTexture2D(width, height, format, usage, type, allocation);
    
    if (result == VK_SUCCESS) {
        image = allocation->image;
        memory = allocation->memory;
        m_allocations.push_back(allocation);
    }
    
    return result;
}

// GlobalTerrainMemory implementation
std::unique_ptr<TerrainMemoryAllocator> GlobalTerrainMemory::s_instance;
std::mutex GlobalTerrainMemory::s_mutex;

TerrainMemoryAllocator& GlobalTerrainMemory::getInstance() {
    std::lock_guard<std::mutex> lock(s_mutex);
    
    if (!s_instance) {
        s_instance = std::make_unique<TerrainMemoryAllocator>();
    }
    
    return *s_instance;
}

VkResult GlobalTerrainMemory::initialize(const TerrainMemoryConfig& config) {
    std::lock_guard<std::mutex> lock(s_mutex);
    
    if (!s_instance) {
        s_instance = std::make_unique<TerrainMemoryAllocator>();
    }
    
    return s_instance->initialize(config);
}

void GlobalTerrainMemory::destroy() {
    std::lock_guard<std::mutex> lock(s_mutex);
    
    if (s_instance) {
        s_instance->destroy();
        s_instance.reset();
    }
}

// TerrainMemoryUtils implementation
namespace TerrainMemoryUtils {

size_t calculateOptimalPoolSize(TerrainMemoryType type, 
                               const std::vector<size_t>& allocationSizes) {
    if (allocationSizes.empty()) {
        return 64ULL * 1024 * 1024; // Default 64MB
    }
    
    // Calculate statistics
    size_t total = 0;
    size_t maxSize = 0;
    
    for (size_t size : allocationSizes) {
        total += size;
        maxSize = std::max(maxSize, size);
    }
    
    size_t avgSize = total / allocationSizes.size();
    
    // Pool should accommodate at least 10-20 average allocations
    // or 2-3 maximum allocations, whichever is larger
    size_t poolSize = std::max(avgSize * 15, maxSize * 3);
    
    // Apply type-specific multipliers
    switch (type) {
        case TerrainMemoryType::HeightTexture:
        case TerrainMemoryType::ColorTexture:
            poolSize *= 2; // Textures tend to be larger
            break;
        case TerrainMemoryType::UniformBuffer:
            poolSize = std::max(poolSize, 16ULL * 1024 * 1024); // Minimum 16MB
            break;
        default:
            break;
    }
    
    // Clamp to reasonable bounds
    poolSize = std::max(poolSize, 16ULL * 1024 * 1024);      // Min 16MB
    poolSize = std::min(poolSize, 1024ULL * 1024 * 1024);    // Max 1GB
    
    return poolSize;
}

TerrainMemoryConfig getRecommendedConfig(VkPhysicalDevice physicalDevice) {
    TerrainMemoryConfig config;
    
    // Query device memory properties
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    // Calculate total available device memory
    VkDeviceSize totalDeviceMemory = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceMemory += memProps.memoryHeaps[i].size;
        }
    }
    
    // Use conservative percentage of total memory
    config.maxTotalMemory = static_cast<size_t>(totalDeviceMemory * 0.25); // 25% of total
    
    // Adjust pool sizes based on available memory
    float scaleFactor = static_cast<float>(config.maxTotalMemory) / (2ULL * 1024 * 1024 * 1024);
    scaleFactor = std::max(0.25f, std::min(4.0f, scaleFactor)); // Clamp to reasonable range
    
    for (auto& [type, typeConfig] : config.typeConfigs) {
        typeConfig.preferredPoolSize = static_cast<size_t>(typeConfig.preferredPoolSize * scaleFactor);
        typeConfig.minPoolSize = static_cast<size_t>(typeConfig.minPoolSize * scaleFactor);
    }
    
    return config;
}

AllocationAnalysis analyzeAllocations(const TerrainMemoryStats& stats,
                                    TerrainMemoryType type) {
    AllocationAnalysis analysis;
    
    // This would analyze allocation patterns in a real implementation
    // For now, provide basic analysis based on available stats
    
    auto it = stats.usedByType.find(type);
    if (it != stats.usedByType.end()) {
        auto countIt = stats.countByType.find(type);
        if (countIt != stats.countByType.end() && countIt->second > 0) {
            analysis.averageSize = static_cast<float>(it->second) / countIt->second;
        }
    }
    
    // Generate recommendations based on usage patterns
    if (analysis.averageSize > 0) {
        analysis.recommendations.push_back("Average allocation size: " + 
                                         formatBytes(static_cast<size_t>(analysis.averageSize)));
    }
    
    if (stats.fragmentation > 0.3f) {
        analysis.recommendations.push_back("High fragmentation detected - consider defragmentation");
    }
    
    return analysis;
}

size_t getOptimalAlignment(TerrainMemoryType type) {
    switch (type) {
        case TerrainMemoryType::VertexBuffer:
        case TerrainMemoryType::IndexBuffer:
            return 256; // Good for vertex data
        case TerrainMemoryType::UniformBuffer:
            return 256; // Required by many GPUs
        case TerrainMemoryType::HeightTexture:
        case TerrainMemoryType::ColorTexture:
        case TerrainMemoryType::NormalTexture:
            return 1024; // Texture alignment
        case TerrainMemoryType::StagingBuffer:
            return 64; // Less strict for staging
        case TerrainMemoryType::ComputeBuffer:
            return 256; // Compute shader alignment
        default:
            return 256; // Safe default
    }
}

size_t alignSize(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

bool isAligned(size_t offset, size_t alignment) {
    return (offset & (alignment - 1)) == 0;
}

std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }
    
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
    return ss.str();
}

size_t parseMemorySize(const std::string& sizeStr) {
    // Simple parser for strings like "128MB", "2GB", etc.
    std::string str = sizeStr;
    std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    
    size_t value = std::stoull(str);
    
    if (str.find("KB") != std::string::npos) {
        value *= 1024;
    } else if (str.find("MB") != std::string::npos) {
        value *= 1024 * 1024;
    } else if (str.find("GB") != std::string::npos) {
        value *= 1024ULL * 1024 * 1024;
    } else if (str.find("TB") != std::string::npos) {
        value *= 1024ULL * 1024 * 1024 * 1024;
    }
    
    return value;
}

std::string getTypeName(TerrainMemoryType type) {
    switch (type) {
        case TerrainMemoryType::VertexBuffer: return "VertexBuffer";
        case TerrainMemoryType::IndexBuffer: return "IndexBuffer";
        case TerrainMemoryType::HeightTexture: return "HeightTexture";
        case TerrainMemoryType::ColorTexture: return "ColorTexture";
        case TerrainMemoryType::NormalTexture: return "NormalTexture";
        case TerrainMemoryType::UniformBuffer: return "UniformBuffer";
        case TerrainMemoryType::StagingBuffer: return "StagingBuffer";
        case TerrainMemoryType::ComputeBuffer: return "ComputeBuffer";
        default: return "Unknown";
    }
}

PerformanceMetrics calculatePerformanceMetrics(const TerrainMemoryStats& stats,
                                              std::chrono::duration<double> timespan) {
    PerformanceMetrics metrics;
    
    double seconds = timespan.count();
    if (seconds > 0.0) {
        metrics.allocationsPerSecond = static_cast<double>(stats.totalAllocations) / seconds;
        metrics.deallocationsPerSecond = static_cast<double>(stats.totalDeallocations) / seconds;
    }
    
    metrics.averageAllocationTime = static_cast<double>(stats.averageAllocTime.count());
    metrics.averageDeallocationTime = static_cast<double>(stats.averageFreeTime.count());
    
    // Estimate memory bandwidth (simplified)
    if (seconds > 0.0) {
        double bytesTransferred = static_cast<double>(stats.totalAllocated + stats.totalFree);
        metrics.memoryBandwidth = (bytesTransferred / seconds) / (1024.0 * 1024.0 * 1024.0);
    }
    
    return metrics;
}

} // namespace TerrainMemoryUtils

} // namespace vf