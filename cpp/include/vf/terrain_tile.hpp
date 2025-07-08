/**
 * @file terrain_tile.hpp
 * @brief Individual terrain tile management with streaming and LOD
 * 
 * Provides comprehensive tile management including:
 * - Hierarchical level-of-detail (LOD) system
 * - Async loading and GPU upload
 * - Memory-efficient storage
 * - Spatial indexing and neighbor queries
 * - Integration with terrain cache system
 */

#pragma once

#include "vf/vk_common.hpp"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <functional>

namespace vf {

/**
 * @brief Tile coordinate in hierarchical grid system
 */
struct TileCoordinate {
    int32_t x = 0;
    int32_t y = 0;
    uint32_t level = 0;      // LOD level (0 = highest detail)
    std::string datasetId;
    
    TileCoordinate() = default;
    TileCoordinate(int32_t x_, int32_t y_, uint32_t level_, const std::string& dataset)
        : x(x_), y(y_), level(level_), datasetId(dataset) {}
    
    bool operator==(const TileCoordinate& other) const {
        return x == other.x && y == other.y && level == other.level && datasetId == other.datasetId;
    }
    
    bool operator<(const TileCoordinate& other) const {
        if (datasetId != other.datasetId) return datasetId < other.datasetId;
        if (level != other.level) return level < other.level;
        if (y != other.y) return y < other.y;
        return x < other.x;
    }
    
    // Get parent tile coordinate (lower detail)
    TileCoordinate getParent() const {
        return TileCoordinate(x / 2, y / 2, level + 1, datasetId);
    }
    
    // Get child tile coordinates (higher detail)
    std::vector<TileCoordinate> getChildren() const {
        if (level == 0) return {}; // Already at highest detail
        
        return {
            TileCoordinate(x * 2,     y * 2,     level - 1, datasetId),
            TileCoordinate(x * 2 + 1, y * 2,     level - 1, datasetId),
            TileCoordinate(x * 2,     y * 2 + 1, level - 1, datasetId),
            TileCoordinate(x * 2 + 1, y * 2 + 1, level - 1, datasetId)
        };
    }
    
    // Get neighboring tiles at same level
    std::vector<TileCoordinate> getNeighbors() const {
        return {
            TileCoordinate(x - 1, y,     level, datasetId), // Left
            TileCoordinate(x + 1, y,     level, datasetId), // Right
            TileCoordinate(x,     y - 1, level, datasetId), // Bottom
            TileCoordinate(x,     y + 1, level, datasetId), // Top
            TileCoordinate(x - 1, y - 1, level, datasetId), // Bottom-left
            TileCoordinate(x + 1, y - 1, level, datasetId), // Bottom-right
            TileCoordinate(x - 1, y + 1, level, datasetId), // Top-left
            TileCoordinate(x + 1, y + 1, level, datasetId)  // Top-right
        };
    }
    
    std::string toString() const {
        return datasetId + "_" + std::to_string(level) + "_" + std::to_string(x) + "_" + std::to_string(y);
    }
};

/**
 * @brief Hash function for TileCoordinate
 */
struct TileCoordinateHash {
    std::size_t operator()(const TileCoordinate& coord) const {
        std::size_t h1 = std::hash<std::string>{}(coord.datasetId);
        std::size_t h2 = std::hash<int32_t>{}(coord.x);
        std::size_t h3 = std::hash<int32_t>{}(coord.y);
        std::size_t h4 = std::hash<uint32_t>{}(coord.level);
        
        // Combine hashes
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

/**
 * @brief Tile loading and rendering states
 */
enum class TileState {
    Empty,           // Not loaded, no data
    Loading,         // Currently loading from disk
    Loaded,          // Data loaded in system memory
    Uploading,       // Uploading to GPU
    Ready,           // GPU buffers ready for rendering
    Error,           // Failed to load
    Evicted          // Data evicted from memory/GPU
};

/**
 * @brief Terrain tile bounds in world coordinates
 */
struct TileBounds {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    float minElevation = 0.0f;
    float maxElevation = 0.0f;
    
    glm::vec3 center() const { return (min + max) * 0.5f; }
    glm::vec3 size() const { return max - min; }
    float radius() const { return glm::length(size()) * 0.5f; }
    
    bool intersects(const TileBounds& other) const {
        return !(max.x < other.min.x || min.x > other.max.x ||
                 max.y < other.min.y || min.y > other.max.y ||
                 max.z < other.min.z || min.z > other.max.z);
    }
    
    bool contains(const glm::vec3& point) const {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z;
    }
};

/**
 * @brief GPU buffer resources for a terrain tile
 */
struct TileGPUResources {
    // Vertex data
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    uint32_t vertexCount = 0;
    
    // Index data (if using indexed rendering)
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
    
    // Height texture
    VkImage heightTexture = VK_NULL_HANDLE;
    VkDeviceMemory heightTextureMemory = VK_NULL_HANDLE;
    VkImageView heightTextureView = VK_NULL_HANDLE;
    
    // Normal texture (optional, for precomputed normals)
    VkImage normalTexture = VK_NULL_HANDLE;
    VkDeviceMemory normalTextureMemory = VK_NULL_HANDLE;
    VkImageView normalTextureView = VK_NULL_HANDLE;
    
    // Descriptor set for this tile's textures
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    
    size_t totalMemoryUsage = 0;
    
    bool isValid() const {
        return vertexBuffer != VK_NULL_HANDLE && heightTexture != VK_NULL_HANDLE;
    }
    
    void destroy();
};

/**
 * @brief CPU data for terrain tile
 */
struct TileCPUData {
    std::vector<float> heightData;    // Height values
    std::vector<float> normalData;    // Optional precomputed normals
    std::vector<uint8_t> textureData; // Optional color/material data
    
    uint32_t width = 0;
    uint32_t height = 0;
    float heightScale = 1.0f;
    
    size_t getMemoryUsage() const {
        return heightData.size() * sizeof(float) +
               normalData.size() * sizeof(float) +
               textureData.size() * sizeof(uint8_t);
    }
    
    void clear() {
        heightData.clear();
        normalData.clear();
        textureData.clear();
        heightData.shrink_to_fit();
        normalData.shrink_to_fit();
        textureData.shrink_to_fit();
    }
};

/**
 * @brief Terrain tile with LOD and streaming support
 */
class TerrainTile {
public:
    explicit TerrainTile(const TileCoordinate& coordinate);
    ~TerrainTile();
    
    // Basic properties
    const TileCoordinate& getCoordinate() const { return m_coordinate; }
    TileState getState() const { return m_state.load(); }
    const TileBounds& getBounds() const { return m_bounds; }
    
    // Data management
    VkResult loadData(const std::string& dataPath);
    VkResult uploadToGPU();
    void unloadFromGPU();
    void evictFromMemory();
    
    // GPU resources
    const TileGPUResources& getGPUResources() const { return m_gpuResources; }
    bool hasValidGPUResources() const { return m_gpuResources.isValid(); }
    
    // Rendering
    VkResult render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout);
    
    // LOD and visibility
    void updateLOD(const glm::vec3& cameraPosition);
    bool isVisible(const glm::vec4 frustumPlanes[6]) const;
    float getDistanceToCamera(const glm::vec3& cameraPosition) const;
    uint32_t getRecommendedLOD(const glm::vec3& cameraPosition, 
                              float nearDistance, float farDistance) const;
    
    // Priority and scheduling
    float getPriority() const { return m_priority; }
    void setPriority(float priority) { m_priority = priority; }
    void updatePriority(const glm::vec3& cameraPosition, float time);
    
    // Statistics and debugging
    size_t getMemoryUsage() const;
    size_t getGPUMemoryUsage() const;
    std::chrono::milliseconds getLoadTime() const { return m_loadTime; }
    uint32_t getFramesSinceAccess() const { return m_framesSinceAccess; }
    void markAccessed() { m_framesSinceAccess = 0; }
    void incrementFrameCounter() { m_framesSinceAccess++; }
    
    // Neighbors and hierarchical relationships
    void setNeighbors(const std::vector<std::shared_ptr<TerrainTile>>& neighbors);
    std::vector<std::weak_ptr<TerrainTile>> getNeighbors() const { return m_neighbors; }
    
    void setParent(std::shared_ptr<TerrainTile> parent) { m_parent = parent; }
    std::weak_ptr<TerrainTile> getParent() const { return m_parent; }
    
    void addChild(std::shared_ptr<TerrainTile> child) { m_children.push_back(child); }
    std::vector<std::weak_ptr<TerrainTile>> getChildren() const { return m_children; }
    
    // Error handling
    bool hasError() const { return m_state.load() == TileState::Error; }
    const std::string& getErrorMessage() const { return m_errorMessage; }
    
private:
    // Internal loading methods
    VkResult loadFromCache();
    VkResult loadFromDisk(const std::string& dataPath);
    VkResult createVertexBuffer();
    VkResult createHeightTexture();
    VkResult createNormalTexture();
    VkResult createDescriptorSet();
    
    // Utility methods
    void calculateBounds();
    void generateVertices();
    void setState(TileState newState);
    void setError(const std::string& errorMessage);
    
private:
    // Core properties
    TileCoordinate m_coordinate;
    std::atomic<TileState> m_state{TileState::Empty};
    TileBounds m_bounds;
    
    // Data
    TileCPUData m_cpuData;
    TileGPUResources m_gpuResources;
    
    // LOD and visibility
    float m_priority = 0.0f;
    uint32_t m_framesSinceAccess = 0;
    float m_lastDistanceToCamera = 0.0f;
    
    // Timing
    std::chrono::high_resolution_clock::time_point m_loadStartTime;
    std::chrono::milliseconds m_loadTime{0};
    
    // Hierarchical relationships
    std::vector<std::weak_ptr<TerrainTile>> m_neighbors;
    std::weak_ptr<TerrainTile> m_parent;
    std::vector<std::weak_ptr<TerrainTile>> m_children;
    
    // Thread safety
    mutable std::mutex m_mutex;
    
    // Error handling
    std::string m_errorMessage;
    
    // Static configuration
    static constexpr uint32_t DEFAULT_TILE_SIZE = 512;
    static constexpr uint32_t VERTICES_PER_SIDE = 64; // For base mesh before tessellation
};

/**
 * @brief Tile manager for handling collections of terrain tiles
 */
class TerrainTileManager {
public:
    TerrainTileManager();
    ~TerrainTileManager();
    
    // Tile creation and management
    std::shared_ptr<TerrainTile> getTile(const TileCoordinate& coordinate);
    std::shared_ptr<TerrainTile> createTile(const TileCoordinate& coordinate);
    void removeTile(const TileCoordinate& coordinate);
    void removeAllTiles();
    
    // Batch operations
    std::vector<std::shared_ptr<TerrainTile>> getTilesInBounds(const TileBounds& bounds);
    std::vector<std::shared_ptr<TerrainTile>> getVisibleTiles(const glm::vec4 frustumPlanes[6]);
    std::vector<std::shared_ptr<TerrainTile>> getTilesByLOD(uint32_t level);
    
    // LOD management
    void updateLOD(const glm::vec3& cameraPosition);
    void scheduleLODTransition(const TileCoordinate& coordinate, uint32_t targetLOD);
    
    // Streaming and priority
    void updatePriorities(const glm::vec3& cameraPosition, float deltaTime);
    std::vector<TileCoordinate> getHighPriorityLoadingQueue(size_t maxCount);
    
    // Memory management
    void performMemoryCleanup(size_t targetMemoryUsage);
    size_t getTotalMemoryUsage() const;
    size_t getTotalGPUMemoryUsage() const;
    
    // Statistics
    struct TileStats {
        uint32_t totalTiles = 0;
        uint32_t readyTiles = 0;
        uint32_t loadingTiles = 0;
        uint32_t errorTiles = 0;
        size_t memoryUsage = 0;
        size_t gpuMemoryUsage = 0;
    };
    
    TileStats getStats() const;
    
    // Configuration
    void setMaxTiles(uint32_t maxTiles) { m_maxTiles = maxTiles; }
    void setMaxMemoryUsage(size_t maxMemory) { m_maxMemoryUsage = maxMemory; }
    
private:
    // Internal data structures
    std::unordered_map<TileCoordinate, std::shared_ptr<TerrainTile>, TileCoordinateHash> m_tiles;
    
    // Configuration
    uint32_t m_maxTiles = 1000;
    size_t m_maxMemoryUsage = 1024 * 1024 * 1024; // 1GB default
    
    // Thread safety
    mutable std::shared_mutex m_tilesMutex;
    
    // Helper methods
    void enforceLimits();
    std::vector<std::shared_ptr<TerrainTile>> getLRUTiles(size_t count);
};

/**
 * @brief RAII helper for tile loading operations
 */
class TileLoadingScope {
public:
    TileLoadingScope(std::shared_ptr<TerrainTile> tile);
    ~TileLoadingScope();
    
    VkResult loadAndUpload(const std::string& dataPath);
    
private:
    std::shared_ptr<TerrainTile> m_tile;
    TileState m_originalState;
    bool m_success = false;
};

} // namespace vf