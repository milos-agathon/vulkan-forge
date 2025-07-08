/**
 * @file terrain_renderer.cpp
 * @brief Implementation of high-performance terrain renderer with GPU tessellation
 */

#include "vf/terrain_renderer.hpp"
#include "vf/vma_util.hpp"
#include <algorithm>
#include <execution>
#include <future>
#include <cmath>

namespace vf {

// TerrainCamera implementation
glm::mat4 TerrainCamera::getViewMatrix() const {
    return glm::lookAt(position, position + direction, up);
}

glm::mat4 TerrainCamera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(fov), aspect, nearPlane, farPlane);
}

glm::mat4 TerrainCamera::getViewProjectionMatrix() const {
    return getProjectionMatrix() * getViewMatrix();
}

// TerrainBounds implementation
bool TerrainBounds::intersects(const TerrainBounds& other) const {
    return !(max.x < other.min.x || min.x > other.max.x ||
             max.y < other.min.y || min.y > other.max.y ||
             max.z < other.min.z || min.z > other.max.z);
}

bool TerrainBounds::contains(const glm::vec3& point) const {
    return point.x >= min.x && point.x <= max.x &&
           point.y >= min.y && point.y <= max.y &&
           point.z >= min.z && point.z <= max.z;
}

// Frustum implementation
void Frustum::update(const glm::mat4& viewProjection) {
    const glm::mat4& m = viewProjection;
    
    // Extract frustum planes from view-projection matrix
    // Left plane
    planes[0] = glm::vec4(m[0][3] + m[0][0], m[1][3] + m[1][0], 
                         m[2][3] + m[2][0], m[3][3] + m[3][0]);
    
    // Right plane
    planes[1] = glm::vec4(m[0][3] - m[0][0], m[1][3] - m[1][0], 
                         m[2][3] - m[2][0], m[3][3] - m[3][0]);
    
    // Bottom plane
    planes[2] = glm::vec4(m[0][3] + m[0][1], m[1][3] + m[1][1], 
                         m[2][3] + m[2][1], m[3][3] + m[3][1]);
    
    // Top plane
    planes[3] = glm::vec4(m[0][3] - m[0][1], m[1][3] - m[1][1], 
                         m[2][3] - m[2][1], m[3][3] - m[3][1]);
    
    // Near plane
    planes[4] = glm::vec4(m[0][3] + m[0][2], m[1][3] + m[1][2], 
                         m[2][3] + m[2][2], m[3][3] + m[3][2]);
    
    // Far plane
    planes[5] = glm::vec4(m[0][3] - m[0][2], m[1][3] - m[1][2], 
                         m[2][3] - m[2][2], m[3][3] - m[3][2]);
    
    // Normalize planes
    for (int i = 0; i < 6; ++i) {
        float length = glm::length(glm::vec3(planes[i]));
        planes[i] /= length;
    }
}

bool Frustum::intersects(const TerrainBounds& bounds) const {
    for (int i = 0; i < 6; ++i) {
        const glm::vec4& plane = planes[i];
        glm::vec3 normal = glm::vec3(plane);
        float distance = plane.w;
        
        // Find the positive vertex (farthest along plane normal)
        glm::vec3 positiveVertex = bounds.min;
        if (normal.x >= 0) positiveVertex.x = bounds.max.x;
        if (normal.y >= 0) positiveVertex.y = bounds.max.y;
        if (normal.z >= 0) positiveVertex.z = bounds.max.z;
        
        // Test if positive vertex is behind plane
        if (glm::dot(normal, positiveVertex) + distance < 0) {
            return false; // AABB is completely behind this plane
        }
    }
    return true; // AABB intersects or is inside frustum
}

bool Frustum::intersects(const glm::vec3& center, float radius) const {
    for (int i = 0; i < 6; ++i) {
        const glm::vec4& plane = planes[i];
        float distance = glm::dot(glm::vec3(plane), center) + plane.w;
        if (distance < -radius) {
            return false; // Sphere is completely behind this plane
        }
    }
    return true; // Sphere intersects or is inside frustum
}

// TerrainRenderer implementation
TerrainRenderer::TerrainRenderer() {
    m_lastFrameTime = std::chrono::high_resolution_clock::now();
}

TerrainRenderer::~TerrainRenderer() {
    destroy();
}

VkResult TerrainRenderer::initialize(const TerrainRenderConfig& config) {
    if (m_initialized) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    m_config = config;
    
    // Initialize tessellation pipelines
    auto& ctx = vk_common::context();
    
    // Create solid pipeline for normal rendering
    m_tessellationPipeline = std::make_unique<TessellationPipeline>();
    VkResult result = m_tessellationPipeline->initialize(
        ctx.defaultRenderPass, 0, TessellationPipelineFactory::getDefaultConfig()
    );
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Create wireframe pipeline for debug rendering
    m_wireframePipeline = std::make_unique<TessellationPipeline>();
    result = m_wireframePipeline->initialize(
        ctx.defaultRenderPass, 0, TessellationPipelineFactory::getWireframeConfig()
    );
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Create uniform buffers
    result = createUniformBuffers();
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Create descriptor sets
    result = createDescriptorSets();
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Initialize tile cache (this would integrate with Python cache)
    // For now, create a placeholder
    // m_tileCache = std::make_unique<TerrainCache>();
    
    // Start background threads
    startBackgroundThreads();
    
    m_initialized = true;
    return VK_SUCCESS;
}

void TerrainRenderer::destroy() {
    if (!m_initialized) {
        return;
    }
    
    // Stop background threads
    stopBackgroundThreads();
    
    // Destroy Vulkan resources
    auto& ctx = vk_common::context();
    
    if (m_uniformBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, m_uniformBuffer, nullptr);
        m_uniformBuffer = VK_NULL_HANDLE;
    }
    
    if (m_uniformBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, m_uniformBufferMemory, nullptr);
        m_uniformBufferMemory = VK_NULL_HANDLE;
    }
    
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, m_descriptorSetLayout, nullptr);
        m_descriptorSetLayout = VK_NULL_HANDLE;
    }
    
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    
    // Destroy pipelines
    m_tessellationPipeline.reset();
    m_wireframePipeline.reset();
    
    // Clear datasets and tiles
    m_datasets.clear();
    m_tiles.clear();
    m_tileCache.reset();
    
    m_initialized = false;
}

VkResult TerrainRenderer::loadDataset(const std::string& datasetId, const std::string& geoTiffPath) {
    // This would integrate with the Python GeoTIFF loader
    // For now, create a placeholder dataset
    
    DatasetInfo dataset;
    dataset.path = geoTiffPath;
    // dataset.loader = std::make_unique<GeoTiffLoader>();
    
    // Load metadata
    // VkResult result = dataset.loader->open(geoTiffPath);
    // if (result != VK_SUCCESS) {
    //     return result;
    // }
    
    // Placeholder values
    dataset.bounds = {{-1000.0f, 0.0f, -1000.0f}, {1000.0f, 200.0f, 1000.0f}};
    dataset.transform = glm::mat4(1.0f);
    dataset.heightScale = 1.0f;
    dataset.width = 4096;
    dataset.height = 4096;
    
    m_datasets[datasetId] = std::move(dataset);
    
    if (m_activeDataset.empty()) {
        m_activeDataset = datasetId;
    }
    
    return VK_SUCCESS;
}

void TerrainRenderer::unloadDataset(const std::string& datasetId) {
    auto it = m_datasets.find(datasetId);
    if (it != m_datasets.end()) {
        // Remove all tiles from this dataset
        std::lock_guard<std::mutex> lock(m_tilesMutex);
        auto tileIt = m_tiles.begin();
        while (tileIt != m_tiles.end()) {
            if (tileIt->first.datasetId == datasetId) {
                tileIt = m_tiles.erase(tileIt);
            } else {
                ++tileIt;
            }
        }
        
        m_datasets.erase(it);
        
        if (m_activeDataset == datasetId) {
            m_activeDataset = m_datasets.empty() ? "" : m_datasets.begin()->first;
        }
    }
}

void TerrainRenderer::setActiveDataset(const std::string& datasetId) {
    if (m_datasets.find(datasetId) != m_datasets.end()) {
        m_activeDataset = datasetId;
    }
}

VkResult TerrainRenderer::render(const TerrainCamera& camera, 
                                VkCommandBuffer commandBuffer,
                                VkRenderPass renderPass,
                                uint32_t subpass) {
    if (!m_initialized || m_activeDataset.empty()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    auto frameStart = std::chrono::high_resolution_clock::now();
    
    // Update frame data
    m_currentFrame.camera = camera;
    m_currentFrame.frameIndex = m_frameCounter++;
    m_currentFrame.frustum.update(camera.getViewProjectionMatrix());
    
    // Perform culling and LOD selection
    auto cullingStart = std::chrono::high_resolution_clock::now();
    VkResult result = performCulling(camera, m_currentFrame);
    if (result != VK_SUCCESS) {
        return result;
    }
    auto cullingEnd = std::chrono::high_resolution_clock::now();
    
    // Update tile streaming
    result = updateTileStreaming(m_currentFrame);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Update uniform buffers
    result = updateUniformBuffers(camera);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Render tiles
    auto renderStart = std::chrono::high_resolution_clock::now();
    result = renderTiles(m_currentFrame, commandBuffer);
    if (result != VK_SUCCESS) {
        return result;
    }
    auto renderEnd = std::chrono::high_resolution_clock::now();
    
    // Update statistics
    auto frameEnd = std::chrono::high_resolution_clock::now();
    m_stats.frameTime = std::chrono::duration<float, std::milli>(frameEnd - frameStart).count();
    m_stats.cullingTime = std::chrono::duration<float, std::milli>(cullingEnd - cullingStart).count();
    m_stats.renderTime = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();
    m_stats.tilesRendered = static_cast<uint32_t>(m_currentFrame.visibleTiles.size());
    
    return VK_SUCCESS;
}

VkResult TerrainRenderer::performCulling(const TerrainCamera& camera, RenderFrame& frame) {
    frame.visibleTiles.clear();
    
    if (m_activeDataset.empty()) {
        return VK_SUCCESS;
    }
    
    const auto& dataset = m_datasets[m_activeDataset];
    
    // Generate tiles based on camera position and LOD
    std::vector<TileCoordinate> candidateTiles;
    
    // Calculate visible region based on camera frustum
    float maxDistance = std::min(camera.farPlane, m_config.farDistance);
    glm::vec3 cameraPos = camera.position;
    
    // Determine LOD levels to consider
    for (uint32_t level = 0; level < 8; ++level) { // Max 8 LOD levels
        float levelDistance = m_config.nearDistance * std::pow(2.0f, level);
        if (levelDistance > maxDistance) break;
        
        // Calculate tile coverage for this level
        uint32_t tilesPerSide = static_cast<uint32_t>(std::pow(2, level));
        float tileSize = 2000.0f / tilesPerSide; // 2km total coverage
        
        int minTileX = static_cast<int>((cameraPos.x - maxDistance) / tileSize) - 1;
        int maxTileX = static_cast<int>((cameraPos.x + maxDistance) / tileSize) + 1;
        int minTileY = static_cast<int>((cameraPos.z - maxDistance) / tileSize) - 1;
        int maxTileY = static_cast<int>((cameraPos.z + maxDistance) / tileSize) + 1;
        
        for (int y = minTileY; y <= maxTileY; ++y) {
            for (int x = minTileX; x <= maxTileX; ++x) {
                TileCoordinate coord(x, y, level, m_activeDataset);
                TerrainBounds tileBounds = getTileBounds(coord);
                
                // Frustum culling
                if (m_config.enableFrustumCulling && !frame.frustum.intersects(tileBounds)) {
                    continue;
                }
                
                // Distance culling
                float distance = glm::length(cameraPos - tileBounds.center());
                if (distance > maxDistance) {
                    continue;
                }
                
                candidateTiles.push_back(coord);
            }
        }
    }
    
    // Sort by priority (distance to camera)
    std::sort(candidateTiles.begin(), candidateTiles.end(),
        [&](const TileCoordinate& a, const TileCoordinate& b) {
            TerrainBounds boundsA = getTileBounds(a);
            TerrainBounds boundsB = getTileBounds(b);
            float distA = glm::length(cameraPos - boundsA.center());
            float distB = glm::length(cameraPos - boundsB.center());
            return distA < distB;
        });
    
    // Limit to maximum visible tiles
    if (candidateTiles.size() > m_config.maxVisibleTiles) {
        candidateTiles.resize(m_config.maxVisibleTiles);
    }
    
    // Get or create tiles
    for (const auto& coord : candidateTiles) {
        auto tile = getTile(coord);
        if (tile && tile->getState() == TileState::Ready) {
            frame.visibleTiles.push_back(tile);
        } else if (!tile) {
            scheduleLoading(coord);
        }
    }
    
    m_stats.tilesCulled = static_cast<uint32_t>(candidateTiles.size() - frame.visibleTiles.size());
    
    return VK_SUCCESS;
}

VkResult TerrainRenderer::updateTileStreaming(const RenderFrame& frame) {
    // Update tile loading queue and priorities
    updateTileStates();
    
    // This would integrate with the Python terrain cache
    // For now, just placeholder logic
    
    return VK_SUCCESS;
}

VkResult TerrainRenderer::renderTiles(const RenderFrame& frame, VkCommandBuffer commandBuffer) {
    if (frame.visibleTiles.empty()) {
        return VK_SUCCESS;
    }
    
    // Choose pipeline based on configuration
    TessellationPipeline* pipeline = m_config.enableWireframe ? 
        m_wireframePipeline.get() : m_tessellationPipeline.get();
    
    // Bind pipeline
    pipeline->bind(commandBuffer);
    
    // Bind descriptor sets
    if (m_descriptorSet != VK_NULL_HANDLE) {
        std::vector<VkDescriptorSet> descriptorSets = { m_descriptorSet };
        pipeline->bindDescriptorSets(commandBuffer, descriptorSets);
    }
    
    // Render each tile
    uint32_t totalTriangles = 0;
    for (const auto& tile : frame.visibleTiles) {
        if (!tile->hasValidGPUResources()) {
            continue;
        }
        
        // Update push constants for this tile
        TessellationPushConstants pushConstants = {};
        pushConstants.modelMatrix = glm::mat4(1.0f);
        pushConstants.viewMatrix = frame.camera.getViewMatrix();
        pushConstants.projMatrix = frame.camera.getProjectionMatrix();
        pushConstants.mvpMatrix = pushConstants.projMatrix * pushConstants.viewMatrix * pushConstants.modelMatrix;
        
        pushConstants.cameraPosition = frame.camera.position;
        pushConstants.tessellationScale = m_config.tessellationScale;
        pushConstants.heightmapSize = glm::vec2(512.0f, 512.0f); // Default tile size
        pushConstants.terrainScale = glm::vec2(1.0f, 1.0f);
        pushConstants.heightScale = m_config.heightScale;
        pushConstants.time = static_cast<float>(frame.frameIndex) * 0.016f; // Assume 60 FPS
        
        pushConstants.nearDistance = m_config.nearDistance;
        pushConstants.farDistance = m_config.farDistance;
        pushConstants.minTessLevel = m_config.minTessLevel;
        pushConstants.maxTessLevel = m_config.maxTessLevel;
        
        pushConstants.sunDirection = glm::normalize(m_config.sunDirection);
        pushConstants.sunColor = m_config.sunColor;
        pushConstants.ambientColor = m_config.ambientColor;
        
        pushConstants.fogColor = m_config.fogColor;
        pushConstants.fogDensity = m_config.fogDensity;
        pushConstants.fogStart = m_config.fogStart;
        pushConstants.fogEnd = m_config.fogEnd;
        
        pushConstants.roughness = m_config.roughness;
        pushConstants.metallic = m_config.metallic;
        
        // Wireframe parameters for debug mode
        if (m_config.enableWireframe) {
            pushConstants.wireframeColor = glm::vec3(1.0f, 1.0f, 0.0f);
            pushConstants.wireframeThickness = 1.0f;
            pushConstants.wireframeOpacity = 0.8f;
            pushConstants.visualizationMode = m_debugVisualizationMode;
        }
        
        pipeline->updatePushConstants(commandBuffer, pushConstants);
        
        // Render tile
        VkResult result = tile->render(commandBuffer, pipeline->getPipelineLayout());
        if (result != VK_SUCCESS) {
            continue; // Skip this tile but continue rendering others
        }
        
        // Estimate triangle count (this would be calculated based on tessellation level)
        totalTriangles += 1000; // Placeholder
        m_stats.drawCalls++;
    }
    
    m_stats.trianglesRendered = totalTriangles;
    
    return VK_SUCCESS;
}

VkResult TerrainRenderer::createUniformBuffers() {
    auto& ctx = vk_common::context();
    
    VkDeviceSize bufferSize = sizeof(TessellationPushConstants);
    
    VkResult result = vma_util::createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU,
        m_uniformBuffer,
        m_uniformBufferMemory
    );
    
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Map buffer for persistent mapping
    vkMapMemory(ctx.device, m_uniformBufferMemory, 0, bufferSize, 0, &m_uniformBufferMapped);
    
    return VK_SUCCESS;
}

VkResult TerrainRenderer::updateUniformBuffers(const TerrainCamera& camera) {
    // For now, uniform data is passed via push constants
    // This function is kept for future expansion
    return VK_SUCCESS;
}

VkResult TerrainRenderer::createDescriptorSets() {
    auto& ctx = vk_common::context();
    
    // Get descriptor set layout from pipeline
    m_descriptorSetLayout = m_tessellationPipeline->getDescriptorSetLayout();
    
    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 5 }
    };
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 100;
    
    VkResult result = vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_descriptorPool);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;
    
    result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &m_descriptorSet);
    return result;
}

std::shared_ptr<TerrainTile> TerrainRenderer::getTile(const TileCoordinate& coord) {
    std::lock_guard<std::mutex> lock(m_tilesMutex);
    
    auto it = m_tiles.find(coord);
    if (it != m_tiles.end()) {
        return it->second;
    }
    
    return nullptr;
}

void TerrainRenderer::scheduleLoading(const TileCoordinate& coord) {
    std::lock_guard<std::mutex> lock(m_loadingQueueMutex);
    m_loadingQueue.push(coord);
}

void TerrainRenderer::updateTileStates() {
    std::lock_guard<std::mutex> lock(m_tilesMutex);
    
    for (auto& [coord, tile] : m_tiles) {
        tile->incrementFrameCounter();
        
        // Remove tiles that haven't been accessed recently
        if (tile->getFramesSinceAccess() > 300) { // 5 seconds at 60 FPS
            tile->evictFromMemory();
        }
    }
}

void TerrainRenderer::startBackgroundThreads() {
    m_threadsRunning = true;
    
    // Start tile loading threads
    uint32_t numLoadingThreads = std::max(1u, std::thread::hardware_concurrency() / 4);
    for (uint32_t i = 0; i < numLoadingThreads; ++i) {
        m_loadingThreads.emplace_back(&TerrainRenderer::tileLoadingWorker, this);
    }
    
    // Start streaming thread
    m_streamingThread = std::thread(&TerrainRenderer::streamingWorker, this);
}

void TerrainRenderer::stopBackgroundThreads() {
    m_threadsRunning = false;
    
    // Wait for loading threads
    for (auto& thread : m_loadingThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    m_loadingThreads.clear();
    
    // Wait for streaming thread
    if (m_streamingThread.joinable()) {
        m_streamingThread.join();
    }
}

void TerrainRenderer::tileLoadingWorker() {
    while (m_threadsRunning) {
        TileCoordinate coord;
        bool hasWork = false;
        
        // Get next tile to load
        {
            std::lock_guard<std::mutex> lock(m_loadingQueueMutex);
            if (!m_loadingQueue.empty()) {
                coord = m_loadingQueue.front();
                m_loadingQueue.pop();
                hasWork = true;
            }
        }
        
        if (!hasWork) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Create and load tile
        auto tile = std::make_shared<TerrainTile>(coord);
        
        // This would integrate with the Python GeoTIFF loader
        // For now, just placeholder
        std::string dataPath = m_datasets[coord.datasetId].path;
        VkResult result = tile->loadData(dataPath);
        
        if (result == VK_SUCCESS) {
            result = tile->uploadToGPU();
        }
        
        if (result == VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(m_tilesMutex);
            m_tiles[coord] = tile;
        }
    }
}

void TerrainRenderer::streamingWorker() {
    while (m_threadsRunning) {
        // Perform periodic cleanup and optimization
        updateTileStates();
        
        // Sleep for a short time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

TileCoordinate TerrainRenderer::worldToTileCoordinate(const glm::vec3& worldPos, uint32_t level) const {
    // Convert world position to tile coordinate
    // This would depend on the specific coordinate system being used
    
    float tileSize = 1000.0f / std::pow(2.0f, level); // Adjust based on actual tile size
    int32_t x = static_cast<int32_t>(std::floor(worldPos.x / tileSize));
    int32_t y = static_cast<int32_t>(std::floor(worldPos.z / tileSize));
    
    return TileCoordinate(x, y, level, m_activeDataset);
}

TerrainBounds TerrainRenderer::getTileBounds(const TileCoordinate& coord) const {
    // Calculate bounds for a tile
    float tileSize = 1000.0f / std::pow(2.0f, coord.level);
    
    TerrainBounds bounds;
    bounds.min = glm::vec3(coord.x * tileSize, 0.0f, coord.y * tileSize);
    bounds.max = glm::vec3((coord.x + 1) * tileSize, 200.0f, (coord.y + 1) * tileSize);
    
    return bounds;
}

uint32_t TerrainRenderer::calculateLODLevel(const glm::vec3& tileCenter, const glm::vec3& cameraPos) const {
    float distance = glm::length(cameraPos - tileCenter);
    
    // Calculate LOD level based on distance
    if (distance < m_config.nearDistance) {
        return 0; // Highest detail
    } else if (distance > m_config.farDistance) {
        return 7; // Lowest detail
    } else {
        float ratio = (distance - m_config.nearDistance) / (m_config.farDistance - m_config.nearDistance);
        return static_cast<uint32_t>(ratio * 7.0f);
    }
}

void TerrainRenderer::updateConfig(const TerrainRenderConfig& config) {
    m_config = config;
}

void TerrainRenderer::resetStats() {
    m_stats = TerrainRenderStats{};
}

void TerrainRenderer::setViewport(uint32_t width, uint32_t height) {
    m_viewportWidth = width;
    m_viewportHeight = height;
}

glm::vec3 TerrainRenderer::worldToTerrain(const glm::vec3& worldPos) const {
    // Convert world coordinates to terrain-local coordinates
    // This would depend on the terrain's coordinate system
    return worldPos;
}

glm::vec3 TerrainRenderer::terrainToWorld(const glm::vec3& terrainPos) const {
    // Convert terrain-local coordinates to world coordinates
    return terrainPos;
}

float TerrainRenderer::getHeightAtPosition(const glm::vec2& position) const {
    // This would query the height from the active dataset
    // For now, return a placeholder value
    return 0.0f;
}

// TerrainRenderScope implementation
TerrainRenderScope::TerrainRenderScope(TerrainRenderer& renderer, const TerrainCamera& camera)
    : m_renderer(renderer), m_camera(camera) {
}

TerrainRenderScope::~TerrainRenderScope() {
    if (!m_rendered) {
        // Log warning about unused scope
    }
}

VkResult TerrainRenderScope::render(VkCommandBuffer commandBuffer, VkRenderPass renderPass) {
    m_rendered = true;
    return m_renderer.render(m_camera, commandBuffer, renderPass);
}

} // namespace vf