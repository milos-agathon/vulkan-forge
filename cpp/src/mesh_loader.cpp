#include "vf/mesh_loader.hpp"
#include "vf/vk_common.hpp"
#include <cmath>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace vf {

// ============================================================================
// MeshHandle Implementation
// ============================================================================

VkResult MeshHandle::initialize(std::unique_ptr<VertexBuffer> vertexBuffer, const std::string& name) {
    if (!vertexBuffer || !vertexBuffer->isValid()) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    
    vertexBuffer_ = std::move(vertexBuffer);
    name_ = name;
    
    return VK_SUCCESS;
}

void MeshHandle::bind(VkCommandBuffer commandBuffer) const {
    if (isValid()) {
        vertexBuffer_->bind(commandBuffer);
    }
}

void MeshHandle::draw(VkCommandBuffer commandBuffer, uint32_t instanceCount) const {
    if (isValid()) {
        vertexBuffer_->drawIndexed(commandBuffer, instanceCount);
    }
}

void MeshHandle::destroy() {
    if (vertexBuffer_) {
        vertexBuffer_->destroy();
        vertexBuffer_.reset();
    }
    name_.clear();
}

uint32_t MeshHandle::getVertexCount() const {
    return isValid() ? vertexBuffer_->getVertexCount() : 0;
}

uint32_t MeshHandle::getTriangleCount() const {
    return isValid() ? vertexBuffer_->getIndexCount() / 3 : 0;
}

const VertexLayout& MeshHandle::getVertexLayout() const {
    static VertexLayout emptyLayout(0);
    return isValid() ? vertexBuffer_->getLayout() : emptyLayout;
}

std::string MeshHandle::getInfo() const {
    if (!isValid()) {
        return "Invalid mesh handle";
    }
    
    std::ostringstream oss;
    oss << "Mesh '" << name_ << "': "
        << getVertexCount() << " vertices, "
        << getTriangleCount() << " triangles";
    
    return oss.str();
}

// ============================================================================
// MeshLoader Implementation
// ============================================================================

MeshLoader::~MeshLoader() {
    destroy();
}

VkResult MeshLoader::initialize(VkDevice device,
                               VmaAllocator allocator,
                               VkCommandPool commandPool,
                               VkQueue queue) {
    device_ = device;
    allocator_ = allocator;
    commandPool_ = commandPool;
    queue_ = queue;
    
    return VK_SUCCESS;
}

VkResult MeshLoader::uploadMesh(const void* vertices,
                               uint32_t vertexCount,
                               uint32_t vertexStride,
                               const void* indices,
                               uint32_t indexCount,
                               VkIndexType indexType,
                               const VertexLayout& layout,
                               const std::string& name,
                               std::shared_ptr<MeshHandle>& outHandle) {
    if (!vertices || vertexCount == 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    
    // Validate layout matches stride
    if (layout.stride != vertexStride) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    
    // Create vertex buffer
    auto vertexBuffer = std::make_unique<VertexBuffer>();
    VkResult result = vertexBuffer->initialize(device_, allocator_, commandPool_, queue_);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Upload vertex data
    result = vertexBuffer->createVertexBuffer(vertices, vertexCount, layout);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Upload index data if present
    if (indices && indexCount > 0) {
        result = vertexBuffer->createIndexBuffer(indices, indexCount, indexType);
        if (result != VK_SUCCESS) {
            return result;
        }
    }
    
    // Create mesh handle
    auto meshHandle = std::make_shared<MeshHandle>();
    result = meshHandle->initialize(std::move(vertexBuffer), name);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Update statistics
    totalMeshesCreated_++;
    totalVerticesUploaded_ += vertexCount;
    totalMemoryUsed_ += MeshUtils::calculateMemoryUsage(vertexCount, vertexStride, indexCount, indexType);
    
    // Track mesh handle
    activeMeshes_.push_back(meshHandle);
    cleanupExpiredMeshes();
    
    outHandle = meshHandle;
    return VK_SUCCESS;
}

VkResult MeshLoader::uploadMeshAuto(const void* vertices,
                                   uint32_t vertexCount,
                                   uint32_t vertexStride,
                                   const void* indices,
                                   uint32_t indexCount,
                                   VkIndexType indexType,
                                   const std::string& name,
                                   std::shared_ptr<MeshHandle>& outHandle) {
    VertexLayout layout = detectVertexLayout(vertexStride);
    
    return uploadMesh(vertices, vertexCount, vertexStride, indices, indexCount, 
                     indexType, layout, name, outHandle);
}

VkResult MeshLoader::createPrimitive(const std::string& primitiveType,
                                    float size,
                                    uint32_t subdivisions,
                                    const std::string& name,
                                    std::shared_ptr<MeshHandle>& outHandle) {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    VertexLayout layout(0);
    
    VkResult result = VK_ERROR_FEATURE_NOT_PRESENT;
    
    if (primitiveType == "cube") {
        result = generateCube(size, vertices, indices, layout);
    } else if (primitiveType == "sphere") {
        result = generateSphere(size, subdivisions, vertices, indices, layout);
    }
    
    if (result != VK_SUCCESS) {
        return result;
    }
    
    return uploadMesh(vertices.data(), static_cast<uint32_t>(vertices.size() / (layout.stride / sizeof(float))),
                     layout.stride, indices.data(), static_cast<uint32_t>(indices.size()),
                     VK_INDEX_TYPE_UINT32, layout, name, outHandle);
}

void MeshLoader::removeMesh(std::shared_ptr<MeshHandle> handle) {
    if (!handle) return;
    
    // Remove from active list
    activeMeshes_.erase(
        std::remove_if(activeMeshes_.begin(), activeMeshes_.end(),
                      [&handle](const std::weak_ptr<MeshHandle>& weak) {
                          return weak.expired() || weak.lock() == handle;
                      }),
        activeMeshes_.end()
    );
    
    // Destroy the handle
    handle->destroy();
}

std::string MeshLoader::getStats() const {
    uint32_t activeMeshCount = 0;
    for (const auto& weak : activeMeshes_) {
        if (!weak.expired()) {
            activeMeshCount++;
        }
    }
    
    std::ostringstream oss;
    oss << "MeshLoader Statistics:\n"
        << "  Active meshes: " << activeMeshCount << "\n"
        << "  Total created: " << totalMeshesCreated_ << "\n"
        << "  Vertices uploaded: " << totalVerticesUploaded_ << "\n"
        << "  Memory used: " << (totalMemoryUsed_ / 1024 / 1024) << " MB";
    
    return oss.str();
}

void MeshLoader::destroy() {
    // Destroy all active meshes
    for (auto& weak : activeMeshes_) {
        if (auto mesh = weak.lock()) {
            mesh->destroy();
        }
    }
    activeMeshes_.clear();
    
    // Reset statistics
    totalMeshesCreated_ = 0;
    totalVerticesUploaded_ = 0;
    totalMemoryUsed_ = 0;
}

VertexLayout MeshLoader::detectVertexLayout(uint32_t stride) const {
    // Detect common vertex layouts based on stride
    switch (stride) {
        case 12: // 3 floats - position only
            return VertexLayouts::position3D();
        case 20: // 5 floats - position + UV
            return VertexLayouts::positionUV();
        case 24: // 6 floats - position + normal
            return VertexLayouts::positionNormal();
        case 28: // 7 floats - position + color
            return VertexLayouts::positionColor();
        case 32: // 8 floats - position + normal + UV
            return VertexLayouts::positionNormalUV();
        default:
            // Fallback to position-only if unknown
            return VertexLayouts::position3D();
    }
}

void MeshLoader::cleanupExpiredMeshes() {
    activeMeshes_.erase(
        std::remove_if(activeMeshes_.begin(), activeMeshes_.end(),
                      [](const std::weak_ptr<MeshHandle>& weak) {
                          return weak.expired();
                      }),
        activeMeshes_.end()
    );
}

VkResult MeshLoader::generateCube(float size, 
                                 std::vector<float>& vertices, 
                                 std::vector<uint32_t>& indices,
                                 VertexLayout& layout) const {
    layout = VertexLayouts::positionNormalUV();
    
    float half = size * 0.5f;
    
    // Cube vertices with normals and UVs (24 vertices - 4 per face)
    vertices = {
        // Front face (z = +half)
        -half, -half,  half,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,
         half, -half,  half,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f,
         half,  half,  half,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,
        -half,  half,  half,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f,
        
        // Back face (z = -half)
         half, -half, -half,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
        -half, -half, -half,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
        -half,  half, -half,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
         half,  half, -half,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
        
        // Left face (x = -half)
        -half, -half, -half, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        -half, -half,  half, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
        -half,  half,  half, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
        -half,  half, -half, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
        
        // Right face (x = +half)
         half, -half,  half,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
         half, -half, -half,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
         half,  half, -half,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
         half,  half,  half,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
        
        // Bottom face (y = -half)
        -half, -half, -half,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
         half, -half, -half,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
         half, -half,  half,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
        -half, -half,  half,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
        
        // Top face (y = +half)
        -half,  half,  half,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
         half,  half,  half,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
         half,  half, -half,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
        -half,  half, -half,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
    };
    
    // Indices for cube faces (36 indices)
    indices = {
        // Front face
        0, 1, 2,  0, 2, 3,
        // Back face
        4, 5, 6,  4, 6, 7,
        // Left face
        8, 9, 10,  8, 10, 11,
        // Right face
        12, 13, 14,  12, 14, 15,
        // Bottom face
        16, 17, 18,  16, 18, 19,
        // Top face
        20, 21, 22,  20, 22, 23
    };
    
    return VK_SUCCESS;
}

VkResult MeshLoader::generateSphere(float radius, 
                                   uint32_t subdivisions,
                                   std::vector<float>& vertices,
                                   std::vector<uint32_t>& indices,
                                   VertexLayout& layout) const {
    layout = VertexLayouts::positionNormalUV();
    
    vertices.clear();
    indices.clear();
    
    const uint32_t rings = subdivisions;
    const uint32_t sectors = subdivisions * 2;
    
    const float R = 1.0f / static_cast<float>(rings - 1);
    const float S = 1.0f / static_cast<float>(sectors - 1);
    
    // Generate vertices
    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < sectors; ++s) {
            const float y = std::sin(-M_PI / 2 + M_PI * r * R);
            const float x = std::cos(2 * M_PI * s * S) * std::sin(M_PI * r * R);
            const float z = std::sin(2 * M_PI * s * S) * std::sin(M_PI * r * R);
            
            // Position
            vertices.push_back(x * radius);
            vertices.push_back(y * radius);
            vertices.push_back(z * radius);
            
            // Normal (same as normalized position for sphere)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            
            // UV coordinates
            vertices.push_back(s * S);
            vertices.push_back(r * R);
        }
    }
    
    // Generate indices
    for (uint32_t r = 0; r < rings - 1; ++r) {
        for (uint32_t s = 0; s < sectors - 1; ++s) {
            const uint32_t curRow = r * sectors;
            const uint32_t nextRow = (r + 1) * sectors;
            
            // First triangle
            indices.push_back(curRow + s);
            indices.push_back(nextRow + s);
            indices.push_back(nextRow + s + 1);
            
            // Second triangle
            indices.push_back(curRow + s);
            indices.push_back(nextRow + s + 1);
            indices.push_back(curRow + s + 1);
        }
    }
    
    return VK_SUCCESS;
}

// ============================================================================
// MeshUtils Implementation
// ============================================================================

namespace MeshUtils {

void calculateBounds(const void* vertices,
                    uint32_t vertexCount,
                    uint32_t vertexStride,
                    float outMin[3],
                    float outMax[3]) {
    if (!vertices || vertexCount == 0) {
        outMin[0] = outMin[1] = outMin[2] = 0.0f;
        outMax[0] = outMax[1] = outMax[2] = 0.0f;
        return;
    }
    
    const uint8_t* vertexData = static_cast<const uint8_t*>(vertices);
    
    // Initialize with first vertex
    const float* firstPos = reinterpret_cast<const float*>(vertexData);
    outMin[0] = outMax[0] = firstPos[0];
    outMin[1] = outMax[1] = firstPos[1];
    outMin[2] = outMax[2] = firstPos[2];
    
    // Process remaining vertices
    for (uint32_t i = 1; i < vertexCount; ++i) {
        const float* pos = reinterpret_cast<const float*>(vertexData + i * vertexStride);
        
        for (int j = 0; j < 3; ++j) {
            outMin[j] = std::min(outMin[j], pos[j]);
            outMax[j] = std::max(outMax[j], pos[j]);
        }
    }
}

bool validateMeshData(const void* vertices,
                     uint32_t vertexCount,
                     const void* indices,
                     uint32_t indexCount) {
    // Check basic parameters
    if (!vertices || vertexCount == 0) {
        return false;
    }
    
    // If indices are provided, validate them
    if (indices && indexCount > 0) {
        // Check triangle count
        if (indexCount % 3 != 0) {
            return false;
        }
        
        // Check index range (assuming uint32 indices for now)
        const uint32_t* indexData = static_cast<const uint32_t*>(indices);
        for (uint32_t i = 0; i < indexCount; ++i) {
            if (indexData[i] >= vertexCount) {
                return false;
            }
        }
    }
    
    return true;
}

std::string describeVertexLayout(const VertexLayout& layout) {
    std::ostringstream oss;
    oss << "Vertex Layout (stride: " << layout.stride << " bytes):\n";
    
    for (const auto& attr : layout.attributes) {
        oss << "  Location " << attr.location << ": ";
        
        switch (attr.format) {
            case VK_FORMAT_R32G32B32_SFLOAT:
                oss << "vec3";
                break;
            case VK_FORMAT_R32G32_SFLOAT:
                oss << "vec2";
                break;
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                oss << "vec4";
                break;
            default:
                oss << "unknown format";
                break;
        }
        
        oss << " at offset " << attr.offset << "\n";
    }
    
    return oss.str();
}

uint64_t calculateMemoryUsage(uint32_t vertexCount,
                             uint32_t vertexStride,
                             uint32_t indexCount,
                             VkIndexType indexType) {
    uint64_t vertexSize = static_cast<uint64_t>(vertexCount) * vertexStride;
    
    uint64_t indexSize = 0;
    if (indexCount > 0) {
        uint32_t indexStride = (indexType == VK_INDEX_TYPE_UINT16) ? 2 : 4;
        indexSize = static_cast<uint64_t>(indexCount) * indexStride;
    }
    
    return vertexSize + indexSize;
}

} // namespace MeshUtils

} // namespace vf