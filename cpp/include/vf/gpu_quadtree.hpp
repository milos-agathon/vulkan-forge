/**
 * @file gpu_quadtree.hpp
 * @brief GPU-based spatial indexing and frustum culling for terrain rendering
 * 
 * Provides high-performance spatial operations using compute shaders:
 * - GPU-based quadtree construction and maintenance
 * - Parallel frustum culling using compute shaders
 * - GPU-driven rendering with indirect draw commands
 * - Level-of-detail (LOD) selection on GPU
 * - Occlusion culling integration
 * - Memory-efficient GPU data structures
 */

#pragma once

#include "vf/vk_common.hpp"
#include "vf/terrain_tile.hpp"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace vf {

// Forward declarations
class TerrainMemoryAllocator;

/**
 * @brief GPU quadtree node structure (must match shader layout)
 */
struct alignas(16) GPUQuadtreeNode {
    glm::vec4 bounds;           // (min_x, min_z, max_x, max_z)
    glm::vec2 elevationRange;   // (min_y, max_y)
    uint32_t childIndices[4];   // Indices to child nodes (0 = no child)
    uint32_t tileIndex;         // Index to terrain tile data
    uint32_t level;             // Tree depth level
    uint32_t flags;             // Various flags (visible, has_children, etc.)
    float lodDistance;          // Distance for LOD calculations
    uint32_t padding[3];        // Align to 64 bytes
    
    enum Flags {
        VISIBLE = 1 << 0,
        HAS_CHILDREN = 1 << 1,
        HAS_TILE = 1 << 2,
        REQUIRES_UPDATE = 1 << 3,
        CULLED = 1 << 4,
        HIGH_PRIORITY = 1 << 5
    };
    
    bool hasFlag(uint32_t flag) const { return (flags & flag) != 0; }
    void setFlag(uint32_t flag) { flags |= flag; }
    void clearFlag(uint32_t flag) { flags &= ~flag; }
};

static_assert(sizeof(GPUQuadtreeNode) == 64, "GPUQuadtreeNode must be 64 bytes");

/**
 * @brief GPU tile data structure (must match shader layout)
 */
struct alignas(16) GPUTileData {
    glm::mat4 modelMatrix;      // Tile transformation matrix
    glm::vec4 bounds;           // (min_x, min_z, max_x, max_z)
    glm::vec2 elevationRange;   // (min_y, max_y)
    glm::vec2 texCoordOffset;   // Texture coordinate offset
    glm::vec2 texCoordScale;    // Texture coordinate scale
    uint32_t textureIndex;      // Index into texture array
    uint32_t levelOfDetail;     // Current LOD level
    uint32_t vertexOffset;      // Offset in vertex buffer
    uint32_t indexOffset;       // Offset in index buffer
    uint32_t indexCount;        // Number of indices
    uint32_t instanceCount;     // Number of instances (usually 1)
    float distanceToCamera;     // Distance for sorting
    uint32_t padding;           // Align to 64 bytes
};

static_assert(sizeof(GPUTileData) == 128, "GPUTileData must be 128 bytes");

/**
 * @brief GPU culling data structure (must match shader layout)
 */
struct alignas(16) GPUCullingData {
    glm::mat4 viewMatrix;
    glm::mat4 projMatrix;
    glm::mat4 viewProjMatrix;
    glm::vec4 frustumPlanes[6]; // Frustum planes for culling
    glm::vec4 cameraPosition;   // Camera position and near plane
    glm::vec4 cameraDirection;  // Camera direction and far plane
    glm::vec4 lodDistances;     // LOD transition distances
    glm::vec4 cullingParams;    // Various culling parameters
    uint32_t frameIndex;        // Current frame index
    uint32_t maxTiles;          // Maximum number of tiles
    uint32_t enableOcclusion;   // Enable occlusion culling
    uint32_t padding;
};

static_assert(sizeof(GPUCullingData) == 384, "GPUCullingData must be 384 bytes");

/**
 * @brief GPU draw command structure (VkDrawIndexedIndirectCommand compatible)
 */
struct alignas(16) GPUDrawCommand {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
    uint32_t tileIndex;         // Additional data for identification
    uint32_t lodLevel;          // LOD level for this draw
    uint32_t padding;
};

static_assert(sizeof(GPUDrawCommand) == 32, "GPUDrawCommand must be 32 bytes");

/**
 * @brief GPU quadtree statistics and counters
 */
struct alignas(16) GPUQuadtreeStats {
    uint32_t totalNodes;
    uint32_t visibleNodes;
    uint32_t culledNodes;
    uint32_t totalTiles;
    uint32_t visibleTiles;
    uint32_t culledTiles;
    uint32_t drawCommands;
    uint32_t triangles;
    float avgNodeDepth;
    float cullingTime;      // In milliseconds
    float buildTime;        // In milliseconds
    uint32_t padding;
};

/**
 * @brief Configuration for GPU quadtree
 */
struct GPUQuadtreeConfig {
    uint32_t maxNodes = 16384;          // Maximum number of nodes
    uint32_t maxTiles = 4096;           // Maximum number of tiles
    uint32_t maxDrawCommands = 2048;    // Maximum draw commands
    uint32_t maxDepth = 8;              // Maximum tree depth
    
    // LOD parameters
    float nearLODDistance = 100.0f;
    float farLODDistance = 2000.0f;
    float lodBias = 1.0f;
    
    // Culling parameters
    bool enableFrustumCulling = true;
    bool enableOcclusionCulling = false;
    bool enableDistanceCulling = true;
    float maxRenderDistance = 5000.0f;
    
    // Performance parameters
    uint32_t computeGroupSize = 64;     // Compute shader work group size
    bool enableAsyncCompute = true;     // Use async compute queue
    bool enableGPUDrivenRendering = true;
    
    // Memory management
    bool enableCompaction = true;       // Compact memory periodically
    uint32_t compactionFrameInterval = 60; // Frames between compaction
};

/**
 * @brief GPU quadtree implementation for spatial indexing and culling
 */
class GPUQuadtree {
public:
    GPUQuadtree();
    ~GPUQuadtree();
    
    // Initialization
    VkResult initialize(const GPUQuadtreeConfig& config = {});
    void destroy();
    
    // Tree management
    VkResult buildTree(const std::vector<TileCoordinate>& tiles, 
                      const glm::vec4& worldBounds);
    VkResult updateTree(VkCommandBuffer commandBuffer);
    VkResult compactTree(VkCommandBuffer commandBuffer);
    
    // Tile management
    VkResult addTile(const TileCoordinate& coord, const GPUTileData& tileData);
    VkResult removeTile(const TileCoordinate& coord);
    VkResult updateTile(const TileCoordinate& coord, const GPUTileData& tileData);
    
    // Culling operations
    VkResult performCulling(VkCommandBuffer commandBuffer, 
                           const GPUCullingData& cullingData);
    VkResult generateDrawCommands(VkCommandBuffer commandBuffer);
    
    // GPU-driven rendering
    VkResult bindForRendering(VkCommandBuffer commandBuffer, 
                            VkPipelineLayout pipelineLayout);
    VkResult executeGPUDrivenRender(VkCommandBuffer commandBuffer);
    
    // LOD management
    VkResult updateLOD(VkCommandBuffer commandBuffer, 
                      const glm::vec3& cameraPosition);
    
    // Occlusion culling
    VkResult performOcclusionCulling(VkCommandBuffer commandBuffer,
                                   VkImage depthBuffer);
    
    // Statistics and debugging
    GPUQuadtreeStats getStats() const;
    void resetStats();
    VkResult readbackStats(VkCommandBuffer commandBuffer);
    
    // Memory management
    size_t getMemoryUsage() const;
    VkResult defragment();
    
    // Configuration
    void updateConfig(const GPUQuadtreeConfig& config);
    const GPUQuadtreeConfig& getConfig() const { return m_config; }
    
    // GPU resources access
    VkBuffer getNodeBuffer() const { return m_nodeBuffer; }
    VkBuffer getTileBuffer() const { return m_tileBuffer; }
    VkBuffer getDrawCommandBuffer() const { return m_drawCommandBuffer; }
    VkDescriptorSet getDescriptorSet() const { return m_descriptorSet; }
    
    // Synchronization
    VkResult waitForGPU();
    bool isGPUWorkComplete() const;
    
private:
    // GPU resources
    struct GPUResources {
        // Node data
        VkBuffer nodeBuffer = VK_NULL_HANDLE;
        VkDeviceMemory nodeMemory = VK_NULL_HANDLE;
        void* nodeMappedMemory = nullptr;
        
        // Tile data
        VkBuffer tileBuffer = VK_NULL_HANDLE;
        VkDeviceMemory tileMemory = VK_NULL_HANDLE;
        void* tileMappedMemory = nullptr;
        
        // Draw commands
        VkBuffer drawCommandBuffer = VK_NULL_HANDLE;
        VkDeviceMemory drawCommandMemory = VK_NULL_HANDLE;
        
        // Culling data
        VkBuffer cullingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory cullingMemory = VK_NULL_HANDLE;
        void* cullingMappedMemory = nullptr;
        
        // Statistics
        VkBuffer statsBuffer = VK_NULL_HANDLE;
        VkDeviceMemory statsMemory = VK_NULL_HANDLE;
        void* statsMappedMemory = nullptr;
        
        // Atomic counters
        VkBuffer counterBuffer = VK_NULL_HANDLE;
        VkDeviceMemory counterMemory = VK_NULL_HANDLE;
        
        // Indirect draw buffer
        VkBuffer indirectBuffer = VK_NULL_HANDLE;
        VkDeviceMemory indirectMemory = VK_NULL_HANDLE;
        
        void destroy();
    };
    
    // Compute pipelines
    struct ComputePipelines {
        // Tree operations
        VkComputePipelineEXT buildTreePipeline = VK_NULL_HANDLE;
        VkComputePipelineEXT updateTreePipeline = VK_NULL_HANDLE;
        VkComputePipelineEXT compactTreePipeline = VK_NULL_HANDLE;
        
        // Culling operations
        VkComputePipelineEXT frustumCullingPipeline = VK_NULL_HANDLE;
        VkComputePipelineEXT occlusionCullingPipeline = VK_NULL_HANDLE;
        VkComputePipelineEXT lodUpdatePipeline = VK_NULL_HANDLE;
        
        // Draw command generation
        VkComputePipelineEXT generateDrawsPipeline = VK_NULL_HANDLE;
        VkComputePipelineEXT compactDrawsPipeline = VK_NULL_HANDLE;
        
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        
        void destroy();
    };
    
    // Core data
    GPUQuadtreeConfig m_config;
    bool m_initialized = false;
    
    // GPU resources
    GPUResources m_gpuResources;
    ComputePipelines m_computePipelines;
    
    // Descriptor sets
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    
    // CPU-side data for synchronization
    std::vector<GPUQuadtreeNode> m_cpuNodes;
    std::vector<GPUTileData> m_cpuTiles;
    std::unordered_map<TileCoordinate, uint32_t, TileCoordinateHash> m_tileIndexMap;
    
    // Statistics
    mutable std::mutex m_statsMutex;
    GPUQuadtreeStats m_stats;
    std::atomic<bool> m_statsNeedUpdate{false};
    
    // Synchronization
    VkFence m_computeFence = VK_NULL_HANDLE;
    VkSemaphore m_computeSemaphore = VK_NULL_HANDLE;
    std::atomic<bool> m_gpuWorkInProgress{false};
    
    // Memory management
    std::shared_ptr<TerrainMemoryAllocator> m_memoryAllocator;
    
    // Internal methods
    VkResult createGPUResources();
    VkResult createComputePipelines();
    VkResult createDescriptorSets();
    VkResult updateDescriptorSets();
    
    VkResult loadComputeShader(const std::string& shaderPath, 
                              VkShaderModule& shaderModule);
    VkResult createComputePipeline(VkShaderModule shaderModule,
                                  VkComputePipelineEXT& pipeline);
    
    // Tree building helpers
    uint32_t buildTreeRecursive(const std::vector<uint32_t>& tileIndices,
                               const glm::vec4& bounds, uint32_t depth);
    void calculateNodeBounds(const std::vector<uint32_t>& tileIndices,
                           glm::vec4& bounds, glm::vec2& elevationRange);
    
    // GPU synchronization
    VkResult submitComputeWork(VkCommandBuffer commandBuffer, 
                              bool waitForCompletion = false);
    VkResult waitForComputeCompletion();
    
    // Memory management helpers
    VkResult allocateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                           VmaMemoryUsage memoryUsage,
                           VkBuffer& buffer, VkDeviceMemory& memory,
                           void** mappedMemory = nullptr);
    
    // Debug and validation
    VkResult validateTreeStructure() const;
    void dumpTreeStatistics() const;
};

/**
 * @brief GPU quadtree manager for multiple datasets
 */
class GPUQuadtreeManager {
public:
    GPUQuadtreeManager();
    ~GPUQuadtreeManager();
    
    // Dataset management
    VkResult createQuadtree(const std::string& datasetId, 
                           const GPUQuadtreeConfig& config);
    void destroyQuadtree(const std::string& datasetId);
    GPUQuadtree* getQuadtree(const std::string& datasetId);
    
    // Batch operations
    VkResult performBatchCulling(VkCommandBuffer commandBuffer,
                                const GPUCullingData& cullingData);
    VkResult generateBatchDrawCommands(VkCommandBuffer commandBuffer);
    
    // Statistics
    struct ManagerStats {
        uint32_t activeQuadtrees = 0;
        uint32_t totalNodes = 0;
        uint32_t totalTiles = 0;
        size_t totalMemoryUsage = 0;
        float averageCullingTime = 0.0f;
    };
    
    ManagerStats getStats() const;
    void resetAllStats();
    
    // Memory management
    VkResult defragmentAll();
    size_t getTotalMemoryUsage() const;
    
    // Configuration
    void updateAllConfigs(const GPUQuadtreeConfig& config);
    
private:
    std::unordered_map<std::string, std::unique_ptr<GPUQuadtree>> m_quadtrees;
    mutable std::shared_mutex m_quadtreesMutex;
    
    // Shared resources for batch operations
    VkBuffer m_batchCullingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_batchCullingMemory = VK_NULL_HANDLE;
    VkBuffer m_batchDrawBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_batchDrawMemory = VK_NULL_HANDLE;
    
    VkResult createBatchResources();
    void destroyBatchResources();
};

/**
 * @brief Utility functions for GPU quadtree operations
 */
namespace GPUQuadtreeUtils {
    
    /**
     * @brief Calculate optimal configuration based on terrain size
     */
    GPUQuadtreeConfig calculateOptimalConfig(const glm::vec4& worldBounds,
                                            uint32_t expectedTileCount);
    
    /**
     * @brief Estimate memory usage for given configuration
     */
    size_t estimateMemoryUsage(const GPUQuadtreeConfig& config);
    
    /**
     * @brief Convert CPU tile data to GPU format
     */
    GPUTileData convertTileData(const TerrainTile& tile, 
                               const glm::mat4& transform);
    
    /**
     * @brief Generate frustum planes from view-projection matrix
     */
    void extractFrustumPlanes(const glm::mat4& viewProj, 
                            glm::vec4 planes[6]);
    
    /**
     * @brief Calculate LOD level based on distance and tile size
     */
    uint32_t calculateLODLevel(float distance, float tileSize,
                              float nearDistance, float farDistance);
    
    /**
     * @brief Validate GPU data structures for correctness
     */
    bool validateGPUStructures();
    
    /**
     * @brief Performance analysis tools
     */
    struct PerformanceAnalysis {
        float cullingEfficiency = 0.0f;    // Percentage of tiles culled
        float gpuUtilization = 0.0f;       // GPU compute utilization
        float memoryBandwidth = 0.0f;      // Memory bandwidth usage
        float overdrawRatio = 0.0f;        // Rendering overdraw
        std::vector<std::string> bottlenecks;
        std::vector<std::string> recommendations;
    };
    
    PerformanceAnalysis analyzePerformance(const GPUQuadtreeStats& stats,
                                         std::chrono::duration<float> frameTime);
}

/**
 * @brief RAII helper for GPU quadtree operations
 */
class GPUQuadtreeScope {
public:
    GPUQuadtreeScope(GPUQuadtree& quadtree, VkCommandBuffer commandBuffer);
    ~GPUQuadtreeScope();
    
    VkResult performCulling(const GPUCullingData& cullingData);
    VkResult generateDrawCommands();
    VkResult executeGPUDrivenRender();
    
private:
    GPUQuadtree& m_quadtree;
    VkCommandBuffer m_commandBuffer;
    bool m_bound = false;
};

} // namespace vf