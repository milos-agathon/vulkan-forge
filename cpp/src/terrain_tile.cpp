/**
 * @file terrain_tile.cpp
 * @brief Implementation of terrain tile management with streaming and LOD
 */

#include "vf/terrain_tile.hpp"
#include "vf/vk_common.hpp"
#include "vf/vma_util.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>

namespace vf {

// TileGPUResources implementation
void TileGPUResources::destroy() {
    auto& ctx = vk_common::context();
    
    if (vertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, vertexBuffer, nullptr);
        vertexBuffer = VK_NULL_HANDLE;
    }
    
    if (vertexMemory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, vertexMemory, nullptr);
        vertexMemory = VK_NULL_HANDLE;
    }
    
    if (indexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx.device, indexBuffer, nullptr);
        indexBuffer = VK_NULL_HANDLE;
    }
    
    if (indexMemory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, indexMemory, nullptr);
        indexMemory = VK_NULL_HANDLE;
    }
    
    if (heightTextureView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, heightTextureView, nullptr);
        heightTextureView = VK_NULL_HANDLE;
    }
    
    if (heightTexture != VK_NULL_HANDLE) {
        vkDestroyImage(ctx.device, heightTexture, nullptr);
        heightTexture = VK_NULL_HANDLE;
    }
    
    if (heightTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, heightTextureMemory, nullptr);
        heightTextureMemory = VK_NULL_HANDLE;
    }
    
    if (normalTextureView != VK_NULL_HANDLE) {
        vkDestroyImageView(ctx.device, normalTextureView, nullptr);
        normalTextureView = VK_NULL_HANDLE;
    }
    
    if (normalTexture != VK_NULL_HANDLE) {
        vkDestroyImage(ctx.device, normalTexture, nullptr);
        normalTexture = VK_NULL_HANDLE;
    }
    
    if (normalTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(ctx.device, normalTextureMemory, nullptr);
        normalTextureMemory = VK_NULL_HANDLE;
    }
    
    totalMemoryUsage = 0;
}

// TerrainTile implementation
TerrainTile::TerrainTile(const TileCoordinate& coordinate) 
    : m_coordinate(coordinate) {
    calculateBounds();
}

TerrainTile::~TerrainTile() {
    unloadFromGPU();
    evictFromMemory();
}

VkResult TerrainTile::loadData(const std::string& dataPath) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_state.load() != TileState::Empty) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    setState(TileState::Loading);
    m_loadStartTime = std::chrono::high_resolution_clock::now();
    
    // Try loading from cache first
    VkResult result = loadFromCache();
    if (result == VK_SUCCESS) {
        auto endTime = std::chrono::high_resolution_clock::now();
        m_loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_loadStartTime);
        setState(TileState::Loaded);
        return VK_SUCCESS;
    }
    
    // Load from disk
    result = loadFromDisk(dataPath);
    if (result != VK_SUCCESS) {
        setError("Failed to load tile data from disk");
        return result;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    m_loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_loadStartTime);
    setState(TileState::Loaded);
    
    return VK_SUCCESS;
}

VkResult TerrainTile::uploadToGPU() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_state.load() != TileState::Loaded) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    setState(TileState::Uploading);
    
    // Create vertex buffer
    VkResult result = createVertexBuffer();
    if (result != VK_SUCCESS) {
        setError("Failed to create vertex buffer");
        return result;
    }
    
    // Create height texture
    result = createHeightTexture();
    if (result != VK_SUCCESS) {
        setError("Failed to create height texture");
        return result;
    }
    
    // Create normal texture if we have normal data
    if (!m_cpuData.normalData.empty()) {
        result = createNormalTexture();
        if (result != VK_SUCCESS) {
            setError("Failed to create normal texture");
            return result;
        }
    }
    
    // Create descriptor set
    result = createDescriptorSet();
    if (result != VK_SUCCESS) {
        setError("Failed to create descriptor set");
        return result;
    }
    
    setState(TileState::Ready);
    return VK_SUCCESS;
}

void TerrainTile::unloadFromGPU() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_gpuResources.destroy();
    
    if (m_state.load() == TileState::Ready) {
        setState(TileState::Loaded);
    }
}

void TerrainTile::evictFromMemory() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    unloadFromGPU();
    m_cpuData.clear();
    
    if (m_state.load() != TileState::Error) {
        setState(TileState::Evicted);
    }
}

VkResult TerrainTile::render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout) {
    if (!hasValidGPUResources()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    markAccessed();
    
    // Bind vertex buffer
    VkBuffer vertexBuffers[] = { m_gpuResources.vertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    
    // Bind descriptor set (textures)
    if (m_gpuResources.descriptorSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipelineLayout, 0, 1, &m_gpuResources.descriptorSet, 0, nullptr);
    }
    
    // Draw patches for tessellation
    uint32_t patchCount = (VERTICES_PER_SIDE - 1) * (VERTICES_PER_SIDE - 1); // Quad patches
    vkCmdDraw(commandBuffer, patchCount * 4, 1, 0, 0); // 4 vertices per patch
    
    return VK_SUCCESS;
}

void TerrainTile::updateLOD(const glm::vec3& cameraPosition) {
    float distance = getDistanceToCamera(cameraPosition);
    m_lastDistanceToCamera = distance;
    
    // Update priority based on distance
    updatePriority(cameraPosition, 0.0f);
}

bool TerrainTile::isVisible(const glm::vec4 frustumPlanes[6]) const {
    // Test AABB against frustum planes
    for (int i = 0; i < 6; ++i) {
        const glm::vec4& plane = frustumPlanes[i];
        glm::vec3 normal = glm::vec3(plane);
        float distance = plane.w;
        
        // Find the positive vertex (farthest along plane normal)
        glm::vec3 positiveVertex = m_bounds.min;
        if (normal.x >= 0) positiveVertex.x = m_bounds.max.x;
        if (normal.y >= 0) positiveVertex.y = m_bounds.max.y;
        if (normal.z >= 0) positiveVertex.z = m_bounds.max.z;
        
        // Test if positive vertex is behind plane
        if (glm::dot(normal, positiveVertex) + distance < 0) {
            return false; // AABB is completely behind this plane
        }
    }
    return true; // AABB intersects or is inside frustum
}

float TerrainTile::getDistanceToCamera(const glm::vec3& cameraPosition) const {
    return glm::length(cameraPosition - m_bounds.center());
}

uint32_t TerrainTile::getRecommendedLOD(const glm::vec3& cameraPosition, 
                                       float nearDistance, float farDistance) const {
    float distance = getDistanceToCamera(cameraPosition);
    
    if (distance < nearDistance) {
        return 0; // Highest detail
    } else if (distance > farDistance) {
        return 7; // Lowest detail
    } else {
        float ratio = (distance - nearDistance) / (farDistance - nearDistance);
        return static_cast<uint32_t>(ratio * 7.0f);
    }
}

void TerrainTile::updatePriority(const glm::vec3& cameraPosition, float time) {
    float distance = getDistanceToCamera(cameraPosition);
    
    // Priority calculation: closer tiles have higher priority
    // Also consider LOD level and recency of access
    float basePriority = 1000.0f / (distance + 1.0f);
    float lodBonus = (8 - m_coordinate.level) * 10.0f; // Higher LOD gets bonus
    float accessBonus = std::max(0.0f, 100.0f - static_cast<float>(m_framesSinceAccess));
    
    m_priority = basePriority + lodBonus + accessBonus;
}

size_t TerrainTile::getMemoryUsage() const {
    return m_cpuData.getMemoryUsage();
}

size_t TerrainTile::getGPUMemoryUsage() const {
    return m_gpuResources.totalMemoryUsage;
}

void TerrainTile::setNeighbors(const std::vector<std::shared_ptr<TerrainTile>>& neighbors) {
    m_neighbors.clear();
    for (const auto& neighbor : neighbors) {
        m_neighbors.push_back(neighbor);
    }
}

VkResult TerrainTile::loadFromCache() {
    // This would integrate with the Python terrain cache
    // For now, return failure to force disk loading
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult TerrainTile::loadFromDisk(const std::string& dataPath) {
    // This would integrate with the Python GeoTIFF loader
    // For now, generate synthetic data
    
    uint32_t size = DEFAULT_TILE_SIZE;
    m_cpuData.width = size;
    m_cpuData.height = size;
    m_cpuData.heightScale = 100.0f;
    
    // Generate synthetic height data
    m_cpuData.heightData.resize(size * size);
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            float fx = static_cast<float>(x) / size;
            float fy = static_cast<float>(y) / size;
            
            // Simple sine wave pattern
            float height = std::sin(fx * 6.28f) * std::sin(fy * 6.28f) * 50.0f + 
                          std::sin(fx * 12.56f) * std::sin(fy * 12.56f) * 25.0f;
            
            m_cpuData.heightData[y * size + x] = height;
        }
    }
    
    // Update bounds with actual height data
    auto minMaxHeight = std::minmax_element(m_cpuData.heightData.begin(), m_cpuData.heightData.end());
    m_bounds.minElevation = *minMaxHeight.first;
    m_bounds.maxElevation = *minMaxHeight.second;
    m_bounds.min.y = m_bounds.minElevation;
    m_bounds.max.y = m_bounds.maxElevation;
    
    return VK_SUCCESS;
}

VkResult TerrainTile::createVertexBuffer() {
    // Generate vertices for tessellation base mesh
    generateVertices();
    
    if (m_cpuData.heightData.empty()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    // Calculate vertex data for base mesh (lower resolution for tessellation)
    struct Vertex {
        glm::vec3 position;
        glm::vec2 texCoord;
        glm::vec3 normal;
    };
    
    std::vector<Vertex> vertices;
    vertices.reserve(VERTICES_PER_SIDE * VERTICES_PER_SIDE);
    
    float tileSize = 1000.0f / std::pow(2.0f, m_coordinate.level);
    float vertexSpacing = tileSize / (VERTICES_PER_SIDE - 1);
    
    for (uint32_t y = 0; y < VERTICES_PER_SIDE; ++y) {
        for (uint32_t x = 0; x < VERTICES_PER_SIDE; ++x) {
            Vertex vertex;
            
            // Position in world space
            vertex.position.x = m_bounds.min.x + x * vertexSpacing;
            vertex.position.z = m_bounds.min.z + y * vertexSpacing;
            vertex.position.y = 0.0f; // Height will be sampled in tessellation shader
            
            // Texture coordinates
            vertex.texCoord.x = static_cast<float>(x) / (VERTICES_PER_SIDE - 1);
            vertex.texCoord.y = static_cast<float>(y) / (VERTICES_PER_SIDE - 1);
            
            // Normal (up vector, will be calculated in tessellation shader)
            vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            
            vertices.push_back(vertex);
        }
    }
    
    // Create vertex buffer
    VkDeviceSize bufferSize = vertices.size() * sizeof(Vertex);
    
    VkResult result = vma_util::createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        m_gpuResources.vertexBuffer,
        m_gpuResources.vertexMemory
    );
    
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Upload vertex data
    result = vma_util::uploadBufferData(
        m_gpuResources.vertexBuffer,
        m_gpuResources.vertexMemory,
        vertices.data(),
        bufferSize
    );
    
    if (result != VK_SUCCESS) {
        return result;
    }
    
    m_gpuResources.vertexCount = static_cast<uint32_t>(vertices.size());
    m_gpuResources.totalMemoryUsage += bufferSize;
    
    return VK_SUCCESS;
}

VkResult TerrainTile::createHeightTexture() {
    auto& ctx = vk_common::context();
    
    if (m_cpuData.heightData.empty()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    uint32_t width = m_cpuData.width;
    uint32_t height = m_cpuData.height;
    
    // Convert height data to texture format (R32_SFLOAT)
    VkDeviceSize imageSize = width * height * sizeof(float);
    
    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    
    VkResult result = vma_util::createBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY,
        stagingBuffer,
        stagingBufferMemory
    );
    
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Copy height data to staging buffer
    void* data;
    vkMapMemory(ctx.device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, m_cpuData.heightData.data(), imageSize);
    vkUnmapMemory(ctx.device, stagingBufferMemory);
    
    // Create image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R32_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    result = vkCreateImage(ctx.device, &imageInfo, nullptr, &m_gpuResources.heightTexture);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
        vkFreeMemory(ctx.device, stagingBufferMemory, nullptr);
        return result;
    }
    
    // Allocate image memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(ctx.device, m_gpuResources.heightTexture, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vma_util::findMemoryType(memRequirements.memoryTypeBits, 
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    result = vkAllocateMemory(ctx.device, &allocInfo, nullptr, &m_gpuResources.heightTextureMemory);
    if (result != VK_SUCCESS) {
        vkDestroyImage(ctx.device, m_gpuResources.heightTexture, nullptr);
        vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
        vkFreeMemory(ctx.device, stagingBufferMemory, nullptr);
        return result;
    }
    
    vkBindImageMemory(ctx.device, m_gpuResources.heightTexture, m_gpuResources.heightTextureMemory, 0);
    
    // Transition image layout and copy data
    vma_util::transitionImageLayout(m_gpuResources.heightTexture, VK_FORMAT_R32_SFLOAT,
                                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    
    vma_util::copyBufferToImage(stagingBuffer, m_gpuResources.heightTexture, width, height);
    
    vma_util::transitionImageLayout(m_gpuResources.heightTexture, VK_FORMAT_R32_SFLOAT,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_gpuResources.heightTexture;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    result = vkCreateImageView(ctx.device, &viewInfo, nullptr, &m_gpuResources.heightTextureView);
    
    // Cleanup staging buffer
    vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
    vkFreeMemory(ctx.device, stagingBufferMemory, nullptr);
    
    if (result == VK_SUCCESS) {
        m_gpuResources.totalMemoryUsage += memRequirements.size;
    }
    
    return result;
}

VkResult TerrainTile::createNormalTexture() {
    // Similar to createHeightTexture but for normal data
    // Implementation would be similar, just using different format and data
    return VK_SUCCESS; // Placeholder
}

VkResult TerrainTile::createDescriptorSet() {
    // This would be handled by the terrain renderer
    // Individual tiles don't manage their own descriptor sets
    return VK_SUCCESS;
}

void TerrainTile::calculateBounds() {
    // Calculate world-space bounds based on tile coordinate
    float tileSize = 1000.0f / std::pow(2.0f, m_coordinate.level);
    
    m_bounds.min = glm::vec3(
        m_coordinate.x * tileSize,
        0.0f, // Will be updated when height data is loaded
        m_coordinate.y * tileSize
    );
    
    m_bounds.max = glm::vec3(
        (m_coordinate.x + 1) * tileSize,
        200.0f, // Default max height, will be updated
        (m_coordinate.y + 1) * tileSize
    );
}

void TerrainTile::generateVertices() {
    // Vertex generation is handled in createVertexBuffer
    // This function can be used for additional vertex processing
}

void TerrainTile::setState(TileState newState) {
    m_state.store(newState);
}

void TerrainTile::setError(const std::string& errorMessage) {
    m_errorMessage = errorMessage;
    setState(TileState::Error);
}

// TerrainTileManager implementation
TerrainTileManager::TerrainTileManager() {
}

TerrainTileManager::~TerrainTileManager() {
    removeAllTiles();
}

std::shared_ptr<TerrainTile> TerrainTileManager::getTile(const TileCoordinate& coordinate) {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    auto it = m_tiles.find(coordinate);
    if (it != m_tiles.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::shared_ptr<TerrainTile> TerrainTileManager::createTile(const TileCoordinate& coordinate) {
    std::unique_lock<std::shared_mutex> lock(m_tilesMutex);
    
    // Check if tile already exists
    auto it = m_tiles.find(coordinate);
    if (it != m_tiles.end()) {
        return it->second;
    }
    
    // Create new tile
    auto tile = std::make_shared<TerrainTile>(coordinate);
    m_tiles[coordinate] = tile;
    
    // Enforce limits
    enforceLimits();
    
    return tile;
}

void TerrainTileManager::removeTile(const TileCoordinate& coordinate) {
    std::unique_lock<std::shared_mutex> lock(m_tilesMutex);
    
    auto it = m_tiles.find(coordinate);
    if (it != m_tiles.end()) {
        it->second->evictFromMemory();
        m_tiles.erase(it);
    }
}

void TerrainTileManager::removeAllTiles() {
    std::unique_lock<std::shared_mutex> lock(m_tilesMutex);
    
    for (auto& [coord, tile] : m_tiles) {
        tile->evictFromMemory();
    }
    
    m_tiles.clear();
}

std::vector<std::shared_ptr<TerrainTile>> TerrainTileManager::getTilesInBounds(const TileBounds& bounds) {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    std::vector<std::shared_ptr<TerrainTile>> result;
    
    for (const auto& [coord, tile] : m_tiles) {
        if (tile->getBounds().intersects(bounds)) {
            result.push_back(tile);
        }
    }
    
    return result;
}

std::vector<std::shared_ptr<TerrainTile>> TerrainTileManager::getVisibleTiles(const glm::vec4 frustumPlanes[6]) {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    std::vector<std::shared_ptr<TerrainTile>> result;
    
    for (const auto& [coord, tile] : m_tiles) {
        if (tile->isVisible(frustumPlanes)) {
            result.push_back(tile);
        }
    }
    
    return result;
}

std::vector<std::shared_ptr<TerrainTile>> TerrainTileManager::getTilesByLOD(uint32_t level) {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    std::vector<std::shared_ptr<TerrainTile>> result;
    
    for (const auto& [coord, tile] : m_tiles) {
        if (coord.level == level) {
            result.push_back(tile);
        }
    }
    
    return result;
}

void TerrainTileManager::updateLOD(const glm::vec3& cameraPosition) {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    for (const auto& [coord, tile] : m_tiles) {
        tile->updateLOD(cameraPosition);
    }
}

void TerrainTileManager::updatePriorities(const glm::vec3& cameraPosition, float deltaTime) {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    for (const auto& [coord, tile] : m_tiles) {
        tile->updatePriority(cameraPosition, deltaTime);
    }
}

std::vector<TileCoordinate> TerrainTileManager::getHighPriorityLoadingQueue(size_t maxCount) {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    std::vector<std::pair<TileCoordinate, float>> priorityList;
    
    for (const auto& [coord, tile] : m_tiles) {
        if (tile->getState() == TileState::Empty || tile->getState() == TileState::Evicted) {
            priorityList.emplace_back(coord, tile->getPriority());
        }
    }
    
    // Sort by priority (highest first)
    std::sort(priorityList.begin(), priorityList.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract coordinates
    std::vector<TileCoordinate> result;
    size_t count = std::min(maxCount, priorityList.size());
    result.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        result.push_back(priorityList[i].first);
    }
    
    return result;
}

void TerrainTileManager::performMemoryCleanup(size_t targetMemoryUsage) {
    size_t currentUsage = getTotalMemoryUsage();
    if (currentUsage <= targetMemoryUsage) {
        return;
    }
    
    // Get LRU tiles for eviction
    size_t tilesToRemove = (currentUsage - targetMemoryUsage) / (1024 * 1024); // Rough estimate
    auto lruTiles = getLRUTiles(tilesToRemove);
    
    std::unique_lock<std::shared_mutex> lock(m_tilesMutex);
    
    for (const auto& tile : lruTiles) {
        tile->evictFromMemory();
    }
}

size_t TerrainTileManager::getTotalMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    size_t total = 0;
    for (const auto& [coord, tile] : m_tiles) {
        total += tile->getMemoryUsage();
    }
    
    return total;
}

size_t TerrainTileManager::getTotalGPUMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    size_t total = 0;
    for (const auto& [coord, tile] : m_tiles) {
        total += tile->getGPUMemoryUsage();
    }
    
    return total;
}

TerrainTileManager::TileStats TerrainTileManager::getStats() const {
    std::shared_lock<std::shared_mutex> lock(m_tilesMutex);
    
    TileStats stats;
    stats.totalTiles = static_cast<uint32_t>(m_tiles.size());
    
    for (const auto& [coord, tile] : m_tiles) {
        switch (tile->getState()) {
            case TileState::Ready:
                stats.readyTiles++;
                break;
            case TileState::Loading:
            case TileState::Uploading:
                stats.loadingTiles++;
                break;
            case TileState::Error:
                stats.errorTiles++;
                break;
            default:
                break;
        }
        
        stats.memoryUsage += tile->getMemoryUsage();
        stats.gpuMemoryUsage += tile->getGPUMemoryUsage();
    }
    
    return stats;
}

void TerrainTileManager::enforceLimits() {
    if (m_tiles.size() <= m_maxTiles) {
        return;
    }
    
    // Remove LRU tiles
    size_t tilesToRemove = m_tiles.size() - m_maxTiles;
    auto lruTiles = getLRUTiles(tilesToRemove);
    
    for (const auto& tile : lruTiles) {
        m_tiles.erase(tile->getCoordinate());
    }
}

std::vector<std::shared_ptr<TerrainTile>> TerrainTileManager::getLRUTiles(size_t count) {
    std::vector<std::shared_ptr<TerrainTile>> allTiles;
    allTiles.reserve(m_tiles.size());
    
    for (const auto& [coord, tile] : m_tiles) {
        allTiles.push_back(tile);
    }
    
    // Sort by frames since access (highest first = least recently used)
    std::sort(allTiles.begin(), allTiles.end(),
             [](const auto& a, const auto& b) {
                 return a->getFramesSinceAccess() > b->getFramesSinceAccess();
             });
    
    // Return the requested number of LRU tiles
    count = std::min(count, allTiles.size());
    return std::vector<std::shared_ptr<TerrainTile>>(allTiles.begin(), allTiles.begin() + count);
}

// TileLoadingScope implementation
TileLoadingScope::TileLoadingScope(std::shared_ptr<TerrainTile> tile) 
    : m_tile(tile), m_originalState(tile->getState()) {
}

TileLoadingScope::~TileLoadingScope() {
    if (!m_success && m_tile) {
        // Restore original state on failure
        // Note: This is simplified - in practice, we'd need more sophisticated error handling
    }
}

VkResult TileLoadingScope::loadAndUpload(const std::string& dataPath) {
    if (!m_tile) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    VkResult result = m_tile->loadData(dataPath);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    result = m_tile->uploadToGPU();
    if (result != VK_SUCCESS) {
        return result;
    }
    
    m_success = true;
    return VK_SUCCESS;
}

} // namespace vf