/**
 * @file terrain_renderer.hpp
 * @brief High-performance terrain rendering with GPU tessellation
 * 
 * Provides GPU-driven terrain rendering with:
 * - Adaptive tessellation based on distance and curvature
 * - Tile-based streaming for massive datasets
 * - Multi-threaded tile loading and GPU upload
 * - Frustum culling and LOD management
 * - Integration with GeoTIFF data sources
 */

#pragma once

#include "vf/vk_common.hpp"
#include "vf/terrain_tile.hpp"
#include "vf/tessellation_pipeline.hpp"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>

namespace vf {

// Forward declarations
class TerrainCache;
class GeoTiffLoader;

/**
 * @brief Camera parameters for terrain rendering
 */
struct TerrainCamera {
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, 0.0f, -1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 10000.0f;
    float aspect = 1.0f;
    
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getViewProjectionMatrix() const;
};

/**
 * @brief Terrain rendering configuration
 */
struct TerrainRenderConfig {
    // Tessellation parameters
    float tessellationScale = 1.0f;
    float minTessLevel = 1.0f;
    float maxTessLevel = 64.0f;
    float nearDistance = 50.0f;
    float farDistance = 2000.0f;
    
    // Quality settings
    uint32_t tileSize = 512;
    uint32_t maxVisibleTiles = 256;
    bool enableFrustumCulling = true;
    bool enableLODMorphing = true;
    bool enableWireframe = false;
    
    // Lighting parameters
    glm::vec3 sunDirection{-0.5f, -0.8f, -0.3f};
    glm::vec3 sunColor{1.0f, 0.95f, 0.8f};
    glm::vec3 ambientColor{0.2f, 0.25f, 0.3f};
    
    // Fog parameters
    glm::vec3 fogColor{0.7f, 0.8f, 0.9f};
    float fogDensity = 0.0001f;
    float fogStart = 1000.0f;
    float fogEnd = 5000.0f;
    
    // Material parameters
    float roughness = 0.8f;
    float metallic = 0.1f;
};

/**
 * @brief Terrain rendering statistics
 */
struct TerrainRenderStats {
    uint32_t tilesRendered = 0;
    uint32_t tilesCulled = 0;
    uint32_t tilesLoading = 0;
    uint32_t trianglesRendered = 0;
    uint32_t drawCalls = 0;
    float frameTime = 0.0f;
    float cullingTime = 0.0f;
    float renderTime = 0.0f;
    size_t memoryUsage = 0;
    size_t gpuMemoryUsage = 0;
};

/**
 * @brief Terrain bounds in world coordinates
 */
struct TerrainBounds {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    
    bool intersects(const TerrainBounds& other) const;
    bool contains(const glm::vec3& point) const;
    glm::vec3 center() const { return (min + max) * 0.5f; }
    glm::vec3 size() const { return max - min; }
};

/**
 * @brief Frustum for culling calculations
 */
struct Frustum {
    glm::vec4 planes[6]; // Left, Right, Bottom, Top, Near, Far
    
    void update(const glm::mat4& viewProjection);
    bool intersects(const TerrainBounds& bounds) const;
    bool intersects(const glm::vec3& center, float radius) const;
};

/**
 * @brief High-performance terrain renderer with GPU tessellation
 */
class TerrainRenderer {
public:
    TerrainRenderer();
    ~TerrainRenderer();
    
    // Initialization
    VkResult initialize(const TerrainRenderConfig& config = {});
    void destroy();
    
    // Dataset management
    VkResult loadDataset(const std::string& datasetId, const std::string& geoTiffPath);
    void unloadDataset(const std::string& datasetId);
    void setActiveDataset(const std::string& datasetId);
    
    // Rendering
    VkResult render(const TerrainCamera& camera, 
                   VkCommandBuffer commandBuffer,
                   VkRenderPass renderPass,
                   uint32_t subpass = 0);
    
    // Configuration
    void updateConfig(const TerrainRenderConfig& config);
    const TerrainRenderConfig& getConfig() const { return m_config; }
    
    // Statistics
    const TerrainRenderStats& getStats() const { return m_stats; }
    void resetStats();
    
    // Utility functions
    void setViewport(uint32_t width, uint32_t height);
    glm::vec3 worldToTerrain(const glm::vec3& worldPos) const;
    glm::vec3 terrainToWorld(const glm::vec3& terrainPos) const;
    float getHeightAtPosition(const glm::vec2& position) const;
    
    // Debug and profiling
    void enableDebugMode(bool enable) { m_debugMode = enable; }
    void setDebugVisualization(uint32_t mode) { m_debugVisualizationMode = mode; }
    
private:
    // Internal structures
    struct DatasetInfo {
        std::string path;
        std::unique_ptr<GeoTiffLoader> loader;
        TerrainBounds bounds;
        glm::mat4 transform;
        float heightScale;
        uint32_t width, height;
    };
    
    struct RenderFrame {
        TerrainCamera camera;
        Frustum frustum;
        std::vector<std::shared_ptr<TerrainTile>> visibleTiles;
        uint32_t frameIndex;
    };
    
    // Core rendering
    VkResult performCulling(const TerrainCamera& camera, RenderFrame& frame);
    VkResult updateTileStreaming(const RenderFrame& frame);
    VkResult renderTiles(const RenderFrame& frame, VkCommandBuffer commandBuffer);
    
    // Tile management
    std::shared_ptr<TerrainTile> getTile(const TileCoordinate& coord);
    void scheduleLoading(const TileCoordinate& coord);
    void updateTileStates();
    
    // GPU resource management
    VkResult createUniformBuffers();
    VkResult updateUniformBuffers(const TerrainCamera& camera);
    VkResult createDescriptorSets();
    
    // Threading
    void startBackgroundThreads();
    void stopBackgroundThreads();
    void tileLoadingWorker();
    void streamingWorker();
    
    // Utility
    TileCoordinate worldToTileCoordinate(const glm::vec3& worldPos, uint32_t level) const;
    TerrainBounds getTileBounds(const TileCoordinate& coord) const;
    uint32_t calculateLODLevel(const glm::vec3& tileCenter, const glm::vec3& cameraPos) const;
    
private:
    // Configuration
    TerrainRenderConfig m_config;
    bool m_initialized = false;
    bool m_debugMode = false;
    uint32_t m_debugVisualizationMode = 0;
    
    // Vulkan resources
    std::unique_ptr<TessellationPipeline> m_tessellationPipeline;
    std::unique_ptr<TessellationPipeline> m_wireframePipeline;
    
    VkBuffer m_uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_uniformBufferMemory = VK_NULL_HANDLE;
    void* m_uniformBufferMapped = nullptr;
    
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    
    // Datasets and tiles
    std::unordered_map<std::string, DatasetInfo> m_datasets;
    std::string m_activeDataset;
    std::unique_ptr<TerrainCache> m_tileCache;
    
    // Tile management
    std::unordered_map<TileCoordinate, std::shared_ptr<TerrainTile>, TileCoordinateHash> m_tiles;
    std::queue<TileCoordinate> m_loadingQueue;
    std::mutex m_tilesMutex;
    std::mutex m_loadingQueueMutex;
    
    // Threading
    std::vector<std::thread> m_loadingThreads;
    std::thread m_streamingThread;
    std::atomic<bool> m_threadsRunning{false};
    
    // Frame data
    RenderFrame m_currentFrame;
    uint32_t m_frameCounter = 0;
    uint32_t m_viewportWidth = 1920;
    uint32_t m_viewportHeight = 1080;
    
    // Statistics
    mutable TerrainRenderStats m_stats;
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    
    // Push constants structure matching shader
    struct PushConstants {
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
        glm::mat4 projMatrix;
        glm::mat4 mvpMatrix;
        
        glm::vec3 cameraPosition;
        float tessellationScale;
        
        glm::vec2 heightmapSize;
        glm::vec2 terrainScale;
        float heightScale;
        float time;
        
        float nearDistance;
        float farDistance;
        float minTessLevel;
        float maxTessLevel;
        
        // Lighting
        glm::vec3 sunDirection;
        float padding1;
        glm::vec3 sunColor;
        float padding2;
        glm::vec3 ambientColor;
        float shadowIntensity;
        
        // Fog
        glm::vec3 fogColor;
        float fogDensity;
        float fogStart;
        float fogEnd;
        float padding3;
        float padding4;
        
        // Material
        float roughness;
        float metallic;
        float specularPower;
        float padding5;
        
        // Wireframe (for debug mode)
        glm::vec3 wireframeColor;
        float wireframeThickness;
        float wireframeOpacity;
        int visualizationMode;
        float padding6;
        float padding7;
    };
    
    static_assert(sizeof(PushConstants) <= 256, "Push constants too large");
};

/**
 * @brief RAII helper for terrain rendering scope
 */
class TerrainRenderScope {
public:
    TerrainRenderScope(TerrainRenderer& renderer, const TerrainCamera& camera);
    ~TerrainRenderScope();
    
    VkResult render(VkCommandBuffer commandBuffer, VkRenderPass renderPass);
    
private:
    TerrainRenderer& m_renderer;
    TerrainCamera m_camera;
    bool m_rendered = false;
};

} // namespace vf