#include "vf/gpu_quadtree.hpp"
#include "vf/vulkan_context.hpp"
#include "vf/utils.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <execution>

namespace vf {

// GPU compute shader for culling (SPIR-V would be loaded from file in practice)
static const char* CULLING_COMPUTE_SHADER_SOURCE = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct CullableObject {
    vec3 bounds_min;
    uint id;
    vec3 bounds_max;
    uint lod_level;
    uint visible;
    uint padding[3];
};

struct FrustumPlane {
    vec3 normal;
    float distance;
};

layout(std430, binding = 0) readonly buffer ObjectBuffer {
    CullableObject objects[];
};

layout(std430, binding = 1) writeonly buffer ResultBuffer {
    uint visible_objects[];
};

layout(std430, binding = 2) readonly buffer FrustumBuffer {
    FrustumPlane frustum_planes[6];
};

layout(std430, binding = 3) readonly buffer CullingParams {
    vec3 camera_position;
    uint object_count;
    vec4 lod_distances;
    uint enable_frustum_culling;
    uint enable_lod_culling;
    uint padding[2];
} params;

bool sphere_in_frustum(vec3 center, float radius) {
    for (int i = 0; i < 6; i++) {
        float distance = dot(frustum_planes[i].normal, center) + frustum_planes[i].distance;
        if (distance < -radius) {
            return false;
        }
    }
    return true;
}

uint calculate_lod_level(vec3 object_center) {
    float distance = length(object_center - params.camera_position);
    
    if (distance <= params.lod_distances.x) return 0;
    if (distance <= params.lod_distances.y) return 1;
    if (distance <= params.lod_distances.z) return 2;
    if (distance <= params.lod_distances.w) return 3;
    return 4; // Beyond max distance
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    
    if (index >= params.object_count) {
        return;
    }
    
    CullableObject obj = objects[index];
    bool visible = true;
    
    // Frustum culling
    if (params.enable_frustum_culling != 0) {
        vec3 center = (obj.bounds_min + obj.bounds_max) * 0.5;
        float radius = length(obj.bounds_max - obj.bounds_min) * 0.5;
        
        visible = sphere_in_frustum(center, radius);
    }
    
    // LOD culling
    if (visible && params.enable_lod_culling != 0) {
        vec3 center = (obj.bounds_min + obj.bounds_max) * 0.5;
        uint required_lod = calculate_lod_level(center);
        
        // Only render objects at the appropriate LOD level
        visible = (obj.lod_level == required_lod);
    }
    
    visible_objects[index] = visible ? 1 : 0;
}
)";

GPUQuadtree::GPUQuadtree(VulkanContext& context, const GPUQuadtreeConfig& config)
    : context_(context), config_(config), object_count_(0) {
    
    VF_LOG_INFO("Initializing GPU quadtree with max depth {} and {} objects per node", 
                config_.max_depth, config_.max_objects_per_node);
    
    // Initialize root node
    root_node_ = std::make_unique<QuadtreeNode>();
    root_node_->bounds = config_.bounds;
    root_node_->depth = 0;
    root_node_->is_leaf = true;
    
    // Initialize GPU resources if GPU culling is enabled
    if (config_.enable_gpu_culling) {
        initialize_gpu_resources();
    }
    
    VF_LOG_INFO("GPU quadtree initialized successfully");
}

GPUQuadtree::~GPUQuadtree() {
    cleanup_gpu_resources();
}

bool GPUQuadtree::insert(const CullableObject& object) {
    std::lock_guard<std::shared_mutex> lock(quadtree_mutex_);
    
    // Check if object bounds intersect with quadtree bounds
    if (!config_.bounds.intersects(object.bounds)) {
        VF_LOG_WARN("Object {} bounds do not intersect with quadtree bounds", object.id);
        return false;
    }
    
    // Store object in our flat array for GPU access
    objects_[object.id] = object;
    object_count_++;
    
    // Insert into spatial hierarchy
    return insert_recursive(root_node_.get(), object);
}

bool GPUQuadtree::remove(uint32_t object_id) {
    std::lock_guard<std::shared_mutex> lock(quadtree_mutex_);
    
    auto it = objects_.find(object_id);
    if (it == objects_.end()) {
        return false;
    }
    
    // Remove from spatial hierarchy
    remove_recursive(root_node_.get(), object_id);
    
    // Remove from flat array
    objects_.erase(it);
    object_count_--;
    
    return true;
}

void GPUQuadtree::clear() {
    std::lock_guard<std::shared_mutex> lock(quadtree_mutex_);
    
    objects_.clear();
    object_count_ = 0;
    
    // Reset root node
    root_node_ = std::make_unique<QuadtreeNode>();
    root_node_->bounds = config_.bounds;
    root_node_->depth = 0;
    root_node_->is_leaf = true;
}

CullResults GPUQuadtree::cull_frustum(const std::array<Eigen::Vector4f, 6>& frustum_planes) {
    std::shared_lock<std::shared_mutex> lock(quadtree_mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CullResults results;
    results.visible_objects.reserve(object_count_ / 4); // Rough estimate
    results.culled_objects.reserve(object_count_);
    
    // Perform CPU-based frustum culling using spatial hierarchy
    cull_frustum_recursive(root_node_.get(), frustum_planes, results);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.cull_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    VF_LOG_DEBUG("Frustum culling: {} visible, {} culled in {:.2f}ms", 
                 results.visible_objects.size(), results.culled_objects.size(), results.cull_time_ms);
    
    return results;
}

CullResults GPUQuadtree::cull_lod(const Eigen::Vector3f& camera_position, 
                                 const std::vector<float>& lod_distances) {
    std::shared_lock<std::shared_mutex> lock(quadtree_mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CullResults results;
    results.visible_objects.reserve(object_count_);
    
    // Perform LOD-based culling
    for (const auto& [id, object] : objects_) {
        Eigen::Vector3f object_center = object.bounds.center();
        float distance = (object_center - camera_position).norm();
        
        // Determine required LOD level based on distance
        uint32_t required_lod = 0;
        for (size_t i = 0; i < lod_distances.size(); ++i) {
            if (distance > lod_distances[i]) {
                required_lod = static_cast<uint32_t>(i + 1);
            }
        }
        
        // Only include objects at the correct LOD level
        if (object.lod_level == required_lod) {
            results.visible_objects.push_back(object);
        } else {
            results.culled_objects.push_back(object);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.cull_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    return results;
}

CullResults GPUQuadtree::cull_gpu_compute(const std::array<Eigen::Vector4f, 6>& frustum_planes,
                                         const Eigen::Vector3f& camera_position) {
    if (!config_.enable_gpu_culling || !compute_pipeline_initialized_) {
        VF_LOG_WARN("GPU culling not available, falling back to CPU culling");
        return cull_frustum(frustum_planes);
    }
    
    std::shared_lock<std::shared_mutex> lock(quadtree_mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CullResults results;
    
    // Upload object data to GPU
    upload_objects_to_gpu();
    
    // Upload frustum planes
    upload_frustum_planes(frustum_planes);
    
    // Upload culling parameters
    CullingParams params{};
    params.camera_position = camera_position;
    params.object_count = static_cast<uint32_t>(objects_.size());
    params.lod_distances = Eigen::Vector4f(500.0f, 1000.0f, 2500.0f, 5000.0f); // Default LOD distances
    params.enable_frustum_culling = 1;
    params.enable_lod_culling = 0;
    
    void* mapped_data;
    vkMapMemory(context_.get_device(), culling_params_buffer_memory_, 0, sizeof(CullingParams), 0, &mapped_data);
    memcpy(mapped_data, &params, sizeof(CullingParams));
    vkUnmapMemory(context_.get_device(), culling_params_buffer_memory_);
    
    // Dispatch compute shader
    VkCommandBuffer cmd_buffer = begin_single_time_commands();
    
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 
                           0, 1, &compute_descriptor_set_, 0, nullptr);
    
    uint32_t dispatch_size = (static_cast<uint32_t>(objects_.size()) + config_.cull_threads - 1) / config_.cull_threads;
    vkCmdDispatch(cmd_buffer, dispatch_size, 1, 1);
    
    // Memory barrier
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    
    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &barrier, 0, nullptr, 0, nullptr);
    
    end_single_time_commands(cmd_buffer);
    
    // Read back results
    vkMapMemory(context_.get_device(), result_buffer_memory_, 0, VK_WHOLE_SIZE, 0, &mapped_data);
    uint32_t* visibility_results = static_cast<uint32_t*>(mapped_data);
    
    size_t i = 0;
    for (const auto& [id, object] : objects_) {
        if (visibility_results[i] == 1) {
            results.visible_objects.push_back(object);
        } else {
            results.culled_objects.push_back(object);
        }
        i++;
    }
    
    vkUnmapMemory(context_.get_device(), result_buffer_memory_);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.cull_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    VF_LOG_DEBUG("GPU culling: {} visible, {} culled in {:.2f}ms", 
                 results.visible_objects.size(), results.culled_objects.size(), results.cull_time_ms);
    
    return results;
}

uint32_t GPUQuadtree::get_object_count() const {
    std::shared_lock<std::shared_mutex> lock(quadtree_mutex_);
    return object_count_;
}

uint32_t GPUQuadtree::get_max_depth() const {
    return config_.max_depth;
}

uint32_t GPUQuadtree::get_max_objects_per_node() const {
    return config_.max_objects_per_node;
}

bool GPUQuadtree::is_gpu_culling_enabled() const {
    return config_.enable_gpu_culling && compute_pipeline_initialized_;
}

QuadtreeStatistics GPUQuadtree::get_statistics() const {
    std::shared_lock<std::shared_mutex> lock(quadtree_mutex_);
    
    QuadtreeStatistics stats{};
    stats.total_objects = object_count_;
    stats.total_nodes = count_nodes_recursive(root_node_.get());
    stats.max_depth_used = get_max_depth_used_recursive(root_node_.get());
    stats.leaf_nodes = count_leaf_nodes_recursive(root_node_.get());
    stats.avg_objects_per_leaf = stats.leaf_nodes > 0 ? 
        static_cast<float>(object_count_) / stats.leaf_nodes : 0.0f;
    
    return stats;
}

// Private implementation methods

bool GPUQuadtree::insert_recursive(QuadtreeNode* node, const CullableObject& object) {
    // If this is a leaf node and has room, add the object
    if (node->is_leaf) {
        if (node->objects.size() < config_.max_objects_per_node || node->depth >= config_.max_depth) {
            node->objects.push_back(object.id);
            return true;
        } else {
            // Need to subdivide
            subdivide_node(node);
        }
    }
    
    // Find which child nodes the object intersects with
    bool inserted = false;
    for (auto& child : node->children) {
        if (child && child->bounds.intersects(object.bounds)) {
            if (insert_recursive(child.get(), object)) {
                inserted = true;
            }
        }
    }
    
    return inserted;
}

void GPUQuadtree::remove_recursive(QuadtreeNode* node, uint32_t object_id) {
    if (node->is_leaf) {
        auto it = std::find(node->objects.begin(), node->objects.end(), object_id);
        if (it != node->objects.end()) {
            node->objects.erase(it);
        }
    } else {
        for (auto& child : node->children) {
            if (child) {
                remove_recursive(child.get(), object_id);
            }
        }
    }
}

void GPUQuadtree::subdivide_node(QuadtreeNode* node) {
    if (!node->is_leaf || node->depth >= config_.max_depth) {
        return;
    }
    
    node->is_leaf = false;
    
    Eigen::Vector3f center = node->bounds.center();
    Eigen::Vector3f min = node->bounds.min();
    Eigen::Vector3f max = node->bounds.max();
    
    // Create 4 child nodes
    node->children[0] = std::make_unique<QuadtreeNode>(); // Bottom-left
    node->children[0]->bounds = AxisAlignedBoundingBox(
        Eigen::Vector3f(min.x(), min.y(), min.z()),
        Eigen::Vector3f(center.x(), center.y(), max.z())
    );
    
    node->children[1] = std::make_unique<QuadtreeNode>(); // Bottom-right
    node->children[1]->bounds = AxisAlignedBoundingBox(
        Eigen::Vector3f(center.x(), min.y(), min.z()),
        Eigen::Vector3f(max.x(), center.y(), max.z())
    );
    
    node->children[2] = std::make_unique<QuadtreeNode>(); // Top-left
    node->children[2]->bounds = AxisAlignedBoundingBox(
        Eigen::Vector3f(min.x(), center.y(), min.z()),
        Eigen::Vector3f(center.x(), max.y(), max.z())
    );
    
    node->children[3] = std::make_unique<QuadtreeNode>(); // Top-right
    node->children[3]->bounds = AxisAlignedBoundingBox(
        Eigen::Vector3f(center.x(), center.y(), min.z()),
        Eigen::Vector3f(max.x(), max.y(), max.z())
    );
    
    // Set child properties
    for (int i = 0; i < 4; ++i) {
        node->children[i]->depth = node->depth + 1;
        node->children[i]->is_leaf = true;
    }
    
    // Redistribute objects to children
    std::vector<uint32_t> objects_to_redistribute = std::move(node->objects);
    node->objects.clear();
    
    for (uint32_t object_id : objects_to_redistribute) {
        auto it = objects_.find(object_id);
        if (it != objects_.end()) {
            insert_recursive(node, it->second);
        }
    }
}

void GPUQuadtree::cull_frustum_recursive(QuadtreeNode* node, 
                                        const std::array<Eigen::Vector4f, 6>& frustum_planes,
                                        CullResults& results) {
    // Check if node bounds intersect with frustum
    if (!is_aabb_in_frustum(node->bounds, frustum_planes)) {
        // Entire node is outside frustum, cull all objects
        add_node_objects_to_culled(node, results);
        return;
    }
    
    if (node->is_leaf) {
        // Test individual objects in leaf node
        for (uint32_t object_id : node->objects) {
            auto it = objects_.find(object_id);
            if (it != objects_.end()) {
                const CullableObject& object = it->second;
                
                if (is_sphere_in_frustum(object.bounds.center(), object.bounds.radius(), frustum_planes)) {
                    results.visible_objects.push_back(object);
                } else {
                    results.culled_objects.push_back(object);
                }
            }
        }
    } else {
        // Recursively test child nodes
        for (auto& child : node->children) {
            if (child) {
                cull_frustum_recursive(child.get(), frustum_planes, results);
            }
        }
    }
}

bool GPUQuadtree::is_aabb_in_frustum(const AxisAlignedBoundingBox& aabb,
                                    const std::array<Eigen::Vector4f, 6>& frustum_planes) {
    Eigen::Vector3f center = aabb.center();
    Eigen::Vector3f extents = aabb.extents();
    
    for (const auto& plane : frustum_planes) {
        Eigen::Vector3f normal = plane.head<3>();
        float distance = plane.w();
        
        // Calculate the distance from the center to the plane
        float center_distance = normal.dot(center) + distance;
        
        // Calculate the effective radius of the AABB along the plane normal
        float radius = std::abs(normal.x() * extents.x()) + 
                      std::abs(normal.y() * extents.y()) + 
                      std::abs(normal.z() * extents.z());
        
        // If the AABB is completely on the negative side of the plane, it's outside
        if (center_distance < -radius) {
            return false;
        }
    }
    
    return true;
}

bool GPUQuadtree::is_sphere_in_frustum(const Eigen::Vector3f& center, float radius,
                                      const std::array<Eigen::Vector4f, 6>& frustum_planes) {
    for (const auto& plane : frustum_planes) {
        Eigen::Vector3f normal = plane.head<3>();
        float distance = plane.w();
        
        float sphere_distance = normal.dot(center) + distance;
        if (sphere_distance < -radius) {
            return false;
        }
    }
    
    return true;
}

void GPUQuadtree::add_node_objects_to_culled(QuadtreeNode* node, CullResults& results) {
    if (node->is_leaf) {
        for (uint32_t object_id : node->objects) {
            auto it = objects_.find(object_id);
            if (it != objects_.end()) {
                results.culled_objects.push_back(it->second);
            }
        }
    } else {
        for (auto& child : node->children) {
            if (child) {
                add_node_objects_to_culled(child.get(), results);
            }
        }
    }
}

uint32_t GPUQuadtree::count_nodes_recursive(QuadtreeNode* node) const {
    if (!node) return 0;
    
    uint32_t count = 1;
    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            count += count_nodes_recursive(child.get());
        }
    }
    
    return count;
}

uint32_t GPUQuadtree::get_max_depth_used_recursive(QuadtreeNode* node) const {
    if (!node) return 0;
    
    uint32_t max_depth = node->depth;
    if (!node->is_leaf) {
        for (const auto& child : node->children) {
            if (child) {
                max_depth = std::max(max_depth, get_max_depth_used_recursive(child.get()));
            }
        }
    }
    
    return max_depth;
}

uint32_t GPUQuadtree::count_leaf_nodes_recursive(QuadtreeNode* node) const {
    if (!node) return 0;
    
    if (node->is_leaf) {
        return 1;
    } else {
        uint32_t count = 0;
        for (const auto& child : node->children) {
            count += count_leaf_nodes_recursive(child.get());
        }
        return count;
    }
}

// GPU resource management

void GPUQuadtree::initialize_gpu_resources() {
    VF_LOG_INFO("Initializing GPU culling resources");
    
    try {
        create_compute_pipeline();
        create_gpu_buffers();
        create_descriptor_sets();
        
        compute_pipeline_initialized_ = true;
        VF_LOG_INFO("GPU culling resources initialized successfully");
    } catch (const std::exception& e) {
        VF_LOG_ERROR("Failed to initialize GPU culling resources: {}", e.what());
        config_.enable_gpu_culling = false;
        compute_pipeline_initialized_ = false;
    }
}

void GPUQuadtree::cleanup_gpu_resources() {
    if (!compute_pipeline_initialized_) return;
    
    VkDevice device = context_.get_device();
    
    if (compute_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, compute_pipeline_, nullptr);
    }
    
    if (compute_pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, compute_pipeline_layout_, nullptr);
    }
    
    if (compute_shader_module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, compute_shader_module_, nullptr);
    }
    
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
    }
    
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
    }
    
    // Clean up buffers
    destroy_buffer(object_buffer_, object_buffer_memory_);
    destroy_buffer(result_buffer_, result_buffer_memory_);
    destroy_buffer(frustum_buffer_, frustum_buffer_memory_);
    destroy_buffer(culling_params_buffer_, culling_params_buffer_memory_);
}

void GPUQuadtree::create_compute_pipeline() {
    // In a real implementation, this would load SPIR-V from file
    // For now, we'll create a minimal pipeline setup
    
    VkDevice device = context_.get_device();
    
    // Create shader module (simplified - would normally load compiled SPIR-V)
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    
    // This is a placeholder - real implementation would have compiled SPIR-V
    std::vector<uint32_t> dummy_spirv = {0x07230203}; // SPIR-V magic number
    shader_info.codeSize = dummy_spirv.size() * sizeof(uint32_t);
    shader_info.pCode = dummy_spirv.data();
    
    VkResult result = vkCreateShaderModule(device, &shader_info, nullptr, &compute_shader_module_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute shader module");
    }
    
    // Create descriptor set layout
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    
    // Object buffer
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Result buffer
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Frustum buffer
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    // Culling parameters
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    
    result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    
    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout_;
    
    result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &compute_pipeline_layout_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
    
    // Create compute pipeline
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = compute_shader_module_;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = compute_pipeline_layout_;
    
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute_pipeline_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
}

void GPUQuadtree::create_gpu_buffers() {
    constexpr size_t max_objects = 100000; // Maximum objects for GPU culling
    
    // Object buffer
    VkDeviceSize object_buffer_size = max_objects * sizeof(GPUCullableObject);
    create_buffer(object_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 object_buffer_, object_buffer_memory_);
    
    // Result buffer
    VkDeviceSize result_buffer_size = max_objects * sizeof(uint32_t);
    create_buffer(result_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 result_buffer_, result_buffer_memory_);
    
    // Frustum buffer
    VkDeviceSize frustum_buffer_size = 6 * sizeof(GPUFrustumPlane);
    create_buffer(frustum_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 frustum_buffer_, frustum_buffer_memory_);
    
    // Culling parameters buffer
    VkDeviceSize params_buffer_size = sizeof(CullingParams);
    create_buffer(params_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 culling_params_buffer_, culling_params_buffer_memory_);
}

void GPUQuadtree::create_descriptor_sets() {
    VkDevice device = context_.get_device();
    
    // Create descriptor pool
    std::array<VkDescriptorPoolSize, 1> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 4;
    
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 1;
    
    VkResult result = vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;
    
    result = vkAllocateDescriptorSets(device, &alloc_info, &compute_descriptor_set_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }
    
    // Update descriptor set
    std::array<VkWriteDescriptorSet, 4> descriptor_writes{};
    
    VkDescriptorBufferInfo object_buffer_info{};
    object_buffer_info.buffer = object_buffer_;
    object_buffer_info.offset = 0;
    object_buffer_info.range = VK_WHOLE_SIZE;
    
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = compute_descriptor_set_;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &object_buffer_info;
    
    VkDescriptorBufferInfo result_buffer_info{};
    result_buffer_info.buffer = result_buffer_;
    result_buffer_info.offset = 0;
    result_buffer_info.range = VK_WHOLE_SIZE;
    
    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstSet = compute_descriptor_set_;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pBufferInfo = &result_buffer_info;
    
    VkDescriptorBufferInfo frustum_buffer_info{};
    frustum_buffer_info.buffer = frustum_buffer_;
    frustum_buffer_info.offset = 0;
    frustum_buffer_info.range = VK_WHOLE_SIZE;
    
    descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[2].dstSet = compute_descriptor_set_;
    descriptor_writes[2].dstBinding = 2;
    descriptor_writes[2].dstArrayElement = 0;
    descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[2].descriptorCount = 1;
    descriptor_writes[2].pBufferInfo = &frustum_buffer_info;
    
    VkDescriptorBufferInfo params_buffer_info{};
    params_buffer_info.buffer = culling_params_buffer_;
    params_buffer_info.offset = 0;
    params_buffer_info.range = VK_WHOLE_SIZE;
    
    descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[3].dstSet = compute_descriptor_set_;
    descriptor_writes[3].dstBinding = 3;
    descriptor_writes[3].dstArrayElement = 0;
    descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[3].descriptorCount = 1;
    descriptor_writes[3].pBufferInfo = &params_buffer_info;
    
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), 
                          descriptor_writes.data(), 0, nullptr);
}

void GPUQuadtree::upload_objects_to_gpu() {
    void* mapped_data;
    vkMapMemory(context_.get_device(), object_buffer_memory_, 0, VK_WHOLE_SIZE, 0, &mapped_data);
    
    GPUCullableObject* gpu_objects = static_cast<GPUCullableObject*>(mapped_data);
    
    size_t i = 0;
    for (const auto& [id, object] : objects_) {
        gpu_objects[i].bounds_min = object.bounds.min();
        gpu_objects[i].id = object.id;
        gpu_objects[i].bounds_max = object.bounds.max();
        gpu_objects[i].lod_level = object.lod_level;
        gpu_objects[i].visible = object.visible ? 1 : 0;
        i++;
    }
    
    vkUnmapMemory(context_.get_device(), object_buffer_memory_);
}

void GPUQuadtree::upload_frustum_planes(const std::array<Eigen::Vector4f, 6>& frustum_planes) {
    void* mapped_data;
    vkMapMemory(context_.get_device(), frustum_buffer_memory_, 0, VK_WHOLE_SIZE, 0, &mapped_data);
    
    GPUFrustumPlane* gpu_planes = static_cast<GPUFrustumPlane*>(mapped_data);
    
    for (size_t i = 0; i < 6; ++i) {
        gpu_planes[i].normal = frustum_planes[i].head<3>();
        gpu_planes[i].distance = frustum_planes[i].w();
    }
    
    vkUnmapMemory(context_.get_device(), frustum_buffer_memory_);
}

void GPUQuadtree::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                               VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
    VkDevice device = context_.get_device();
    
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }
    
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);
    
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }
    
    vkBindBufferMemory(device, buffer, buffer_memory, 0);
}

void GPUQuadtree::destroy_buffer(VkBuffer& buffer, VkDeviceMemory& buffer_memory) {
    VkDevice device = context_.get_device();
    
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }
    
    if (buffer_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer_memory, nullptr);
        buffer_memory = VK_NULL_HANDLE;
    }
}

uint32_t GPUQuadtree::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(context_.get_physical_device(), &mem_properties);
    
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}

VkCommandBuffer GPUQuadtree::begin_single_time_commands() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = context_.get_command_pool();
    alloc_info.commandBufferCount = 1;
    
    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(context_.get_device(), &alloc_info, &command_buffer);
    
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(command_buffer, &begin_info);
    
    return command_buffer;
}

void GPUQuadtree::end_single_time_commands(VkCommandBuffer command_buffer) {
    vkEndCommandBuffer(command_buffer);
    
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    
    vkQueueSubmit(context_.get_graphics_queue(), 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(context_.get_graphics_queue());
    
    vkFreeCommandBuffers(context_.get_device(), context_.get_command_pool(), 1, &command_buffer);
}

} // namespace vf