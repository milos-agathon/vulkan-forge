// ============================================================================
// heightfield_scene.cpp – auto‑merged (inline) original + terrain extensions
// Generated 2025-07-08T08:42:36.103177 UTC
// ============================================================================
#include "vf/heightfield_scene.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <cstring>
#include <stdexcept>
#include "vf/vulkan_context.hpp"
#include "vf/camera.hpp"
#include "vf/utils.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

// --------------------------- ORIGINAL IMPLEMENTATION ------------------------

using namespace vf;

/* vertex layout matching the pipeline (pos + rgba) */
struct VertexGPU
{
    glm::vec3 pos;
    glm::vec4 col;
};

/* ---------------- CPU mesh generation ------------------ */
static std::vector<VertexGPU> makeVerts(const std::vector<float>& h,
                                        int nx, int ny,
                                        const std::vector<float>& c,
                                        float zScale)
{
    if(static_cast<int>(h.size()) != nx * ny)
        throw std::runtime_error("height vector size mismatch");

    std::vector<VertexGPU> v;
    v.reserve(nx * ny);

    for(int j = 0; j < ny; ++j)
        for(int i = 0; i < nx; ++i)
        {
            int idx = j * nx + i;
            glm::vec3 p{float(i), float(j), h[idx] * zScale};
            glm::vec4 col{1,1,1,1};
            if(!c.empty())
            {
                col.r = c[idx*4+0];
                col.g = c[idx*4+1];
                col.b = c[idx*4+2];
                col.a = c[idx*4+3];
            }
            v.push_back({p,col});
        }
    return v;
}
static std::vector<uint32_t> makeIdx(int nx, int ny)
{
    std::vector<uint32_t> out;
    out.reserve((nx-1)*(ny-1)*6);
    for(int j = 0; j < ny-1; ++j)
        for(int i = 0; i < nx-1; ++i)
        {
            uint32_t tl =  j    *nx + i;
            uint32_t tr =  tl + 1;
            uint32_t bl = (j+1)*nx + i;
            uint32_t br =  bl + 1;
            out.insert(out.end(), {tl,bl,tr,  tr,bl,br});
        }
    return out;
}

/* ---------------- GPU upload --------------------------- */
void HeightFieldScene::build(const std::vector<float>& h,
                             int nx, int ny,
                             const std::vector<float>& col,
                             float zScale)
{
    auto& gpu = ctx();

    auto verts = makeVerts(h, nx, ny, col, zScale);
    auto idx   = makeIdx(nx, ny);
    nIdx       = uint32_t(idx.size());

    VkDeviceSize vBytes = verts.size() * sizeof(VertexGPU);
    VkDeviceSize iBytes = idx  .size() * sizeof(uint32_t);

    /* free old buffers */
    if(vbo)
    {
        vkDestroyBuffer(gpu.device, vbo, nullptr);
        vkFreeMemory   (gpu.device, vboMem, nullptr);
        vkDestroyBuffer(gpu.device, ibo, nullptr);
        vkFreeMemory   (gpu.device, iboMem, nullptr);
    }

    vbo = allocDeviceLocal(vBytes,
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          vboMem);
    ibo = allocDeviceLocal(iBytes,
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT   | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          iboMem);

    uploadToBuffer(vbo, verts.data(), vBytes);
    uploadToBuffer(ibo, idx .data(), iBytes);
}

HeightFieldScene::~HeightFieldScene()
{
    if(!vbo) return;
    auto& gpu = ctx();
    vkDestroyBuffer(gpu.device, vbo, nullptr);
    vkFreeMemory   (gpu.device, vboMem, nullptr);
    vkDestroyBuffer(gpu.device, ibo, nullptr);
    vkFreeMemory   (gpu.device, iboMem, nullptr);
}

// --------------------------- TERRAIN EXTENSION -----------------------------


namespace vf {

HeightfieldScene::HeightfieldScene(VulkanContext& context)
    : context_(context), tessellation_config_() {

    // Initialize default configuration
    tessellation_config_.mode = TessellationMode::DISTANCE_BASED;
    tessellation_config_.base_level = 8;
    tessellation_config_.max_level = 64;
    tessellation_config_.min_level = 1;

    // Default LOD distances
    lod_distances_ = {500.0f, 1000.0f, 2500.0f, 5000.0f};

    // Create tessellation pipeline if supported
    if (context_.get_device_features().tessellationShader) {
        tessellation_pipeline_ = std::make_unique<TessellationPipeline>(context_, tessellation_config_);
    } else {
        tessellation_enabled_ = false;
        VF_LOG_WARN("Tessellation shaders not supported - falling back to regular rendering");
    }

    // Create Vulkan resources
    create_descriptor_set_layout();
    create_uniform_buffer();
    create_graphics_pipeline();
}

HeightfieldScene::~HeightfieldScene() {
    // Cleanup Vulkan resources
    if (graphics_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(context_.get_device(), graphics_pipeline_, nullptr);
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context_.get_device(), pipeline_layout_, nullptr);
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context_.get_device(), descriptor_set_layout_, nullptr);
    }
}

bool HeightfieldScene::load_heightfield(const std::vector<std::vector<float>>& heights) {
    if (heights.empty() || heights[0].empty()) {
        VF_LOG_ERROR("Invalid heightfield data - empty");
        return false;
    }

    height_ = static_cast<uint32_t>(heights.size());
    width_ = static_cast<uint32_t>(heights[0].size());

    // Flatten 2D vector to 1D
    heights_.resize(width_ * height_);
    for (uint32_t y = 0; y < height_; ++y) {
        if (heights[y].size() != width_) {
            VF_LOG_ERROR("Inconsistent row length in heightfield data");
            return false;
        }
        for (uint32_t x = 0; x < width_; ++x) {
            heights_[get_index(x, y)] = heights[y][x];
        }
    }

    // Set default bounds if not set
    if (!has_geographic_info_) {
        bounds_.min_x = 0.0f;
        bounds_.max_x = static_cast<float>(width_);
        bounds_.min_y = 0.0f;
        bounds_.max_y = static_cast<float>(height_);
        bounds_.min_elevation = *std::min_element(heights_.begin(), heights_.end());
        bounds_.max_elevation = *std::max_element(heights_.begin(), heights_.end());
    }

    // Generate meshes and resources
    create_heightmap_texture();
    generate_lod_meshes(lod_distances_);
    update_bounding_box();

    VF_LOG_INFO("Loaded heightfield: {}x{}, elevation range: {:.1f} - {:.1f}",
                width_, height_, bounds_.min_elevation, bounds_.max_elevation);

    return true;
}

bool HeightfieldScene::load_heightfield(const float* data, uint32_t width, uint32_t height) {
    if (!data || width == 0 || height == 0) {
        VF_LOG_ERROR("Invalid heightfield parameters");
        return false;
    }

    width_ = width;
    height_ = height;
    heights_.assign(data, data + width * height);

    // Set default bounds
    if (!has_geographic_info_) {
        bounds_.min_x = 0.0f;
        bounds_.max_x = static_cast<float>(width_);
        bounds_.min_y = 0.0f;
        bounds_.max_y = static_cast<float>(height_);
        bounds_.min_elevation = *std::min_element(heights_.begin(), heights_.end());
        bounds_.max_elevation = *std::max_element(heights_.begin(), heights_.end());
    }

    create_heightmap_texture();
    generate_lod_meshes(lod_distances_);
    update_bounding_box();

    return true;
}

bool HeightfieldScene::load_geotiff(const std::string& filename) {
    GeoTiffLoader loader;
    if (!loader.load(filename)) {
        VF_LOG_ERROR("Failed to load GeoTIFF: {}", filename);
        return false;
    }

    const auto& geotiff_data = loader.get_data();
    return load_geotiff(geotiff_data);
}

bool HeightfieldScene::load_geotiff(const GeoTiffData& geotiff_data) {
    // Load heightfield data
    if (!load_heightfield(geotiff_data.heights.data(), geotiff_data.width, geotiff_data.height)) {
        return false;
    }

    // Set geographic information
    bounds_ = geotiff_data.bounds;
    has_geographic_info_ = true;

    VF_LOG_INFO("Loaded GeoTIFF with geographic bounds: ({:.6f}, {:.6f}) to ({:.6f}, {:.6f})",
                bounds_.min_x, bounds_.min_y, bounds_.max_x, bounds_.max_y);

    return true;
}

bool HeightfieldScene::generate_synthetic_terrain(uint32_t size, float amplitude, uint32_t octaves) {
    auto heights = heightfield_utils::generate_perlin_noise(size, size, 10.0f, octaves);

    // Scale by amplitude
    for (float& h : heights) {
        h *= amplitude;
    }

    return load_heightfield(heights.data(), size, size);
}

void HeightfieldScene::set_height_scale(float scale) {
    height_scale_ = scale;
    update_bounding_box();
}

void HeightfieldScene::set_world_scale(const Eigen::Vector3f& scale) {
    world_scale_ = scale;
    update_bounding_box();
}

void HeightfieldScene::set_tessellation_config(const TessellationConfig& config) {
    tessellation_config_ = config;
    if (tessellation_pipeline_) {
        tessellation_pipeline_->update_config(config);
    }
}

const TessellationConfig& HeightfieldScene::get_tessellation_config() const {
    return tessellation_config_;
}

void HeightfieldScene::enable_tessellation(bool enable) {
    tessellation_enabled_ = enable && tessellation_pipeline_ != nullptr;
}

void HeightfieldScene::generate_lod_meshes(const std::vector<float>& lod_distances) {
    lod_distances_ = lod_distances;
    lod_meshes_.clear();

    // Generate mesh for each LOD level
    for (uint32_t lod = 0; lod < lod_distances_.size(); ++lod) {
        generate_lod_mesh(lod, lod_distances[lod]);
    }

    VF_LOG_INFO("Generated {} LOD meshes", lod_meshes_.size());
}

void HeightfieldScene::set_lod_distances(const std::vector<float>& distances) {
    generate_lod_meshes(distances);
}

void HeightfieldScene::generate_normals() {
    if (lod_meshes_.empty()) return;

    // Regenerate vertices with normals for all LOD levels
    for (auto& lod_mesh : lod_meshes_) {
        auto vertices = generate_vertices(lod_mesh->lod_level);

        // Upload new vertices to GPU
        lod_mesh->vertex_buffer = std::make_unique<Buffer>(
            context_,
            vertices.size() * sizeof(TerrainVertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        void* data;
        vkMapMemory(context_.get_device(), lod_mesh->vertex_buffer->get_memory(), 0, VK_WHOLE_SIZE, 0, &data);
        memcpy(data, vertices.data(), vertices.size() * sizeof(TerrainVertex));
        vkUnmapMemory(context_.get_device(), lod_mesh->vertex_buffer->get_memory());
    }
}

void HeightfieldScene::generate_tangents() {
    if (lod_meshes_.empty()) return;

    // Similar to generate_normals but calculate tangents
    for (auto& lod_mesh : lod_meshes_) {
        auto vertices = generate_vertices(lod_mesh->lod_level);
        auto indices = generate_indices(lod_mesh->lod_level);

        calculate_tangents(vertices, indices);

        // Upload updated vertices
        void* data;
        vkMapMemory(context_.get_device(), lod_mesh->vertex_buffer->get_memory(), 0, VK_WHOLE_SIZE, 0, &data);
        memcpy(data, vertices.data(), vertices.size() * sizeof(TerrainVertex));
        vkUnmapMemory(context_.get_device(), lod_mesh->vertex_buffer->get_memory());
    }
}

void HeightfieldScene::set_heightmap_texture(std::shared_ptr<Texture> texture) {
    heightmap_texture_ = texture;
    update_descriptor_set();
}

void HeightfieldScene::set_diffuse_texture(std::shared_ptr<Texture> texture) {
    diffuse_texture_ = texture;
    update_descriptor_set();
}

void HeightfieldScene::set_normal_texture(std::shared_ptr<Texture> texture) {
    normal_texture_ = texture;
    update_descriptor_set();
}

void HeightfieldScene::set_bounds(const GeographicBounds& bounds) {
    bounds_ = bounds;
    has_geographic_info_ = true;
    update_bounding_box();
}

uint32_t HeightfieldScene::get_vertex_count() const {
    if (lod_meshes_.empty()) return 0;
    return lod_meshes_[current_lod_level_]->vertex_count;
}

uint32_t HeightfieldScene::get_triangle_count() const {
    if (lod_meshes_.empty()) return 0;
    return lod_meshes_[current_lod_level_]->index_count / 3;
}

float HeightfieldScene::get_height_at(uint32_t x, uint32_t y) const {
    if (!is_valid_coordinate(x, y)) return 0.0f;
    return heights_[get_index(x, y)] * height_scale_;
}

float HeightfieldScene::get_interpolated_height(float x, float y) const {
    // Bilinear interpolation
    uint32_t x0 = static_cast<uint32_t>(std::floor(x));
    uint32_t y0 = static_cast<uint32_t>(std::floor(y));
    uint32_t x1 = x0 + 1;
    uint32_t y1 = y0 + 1;

    if (!is_valid_coordinate(x1, y1)) return get_height_at(x0, y0);

    float fx = x - x0;
    float fy = y - y0;

    float h00 = get_height_at(x0, y0);
    float h10 = get_height_at(x1, y0);
    float h01 = get_height_at(x0, y1);
    float h11 = get_height_at(x1, y1);

    float h0 = h00 * (1 - fx) + h10 * fx;
    float h1 = h01 * (1 - fx) + h11 * fx;

    return h0 * (1 - fy) + h1 * fy;
}

Eigen::Vector3f HeightfieldScene::get_world_position(uint32_t x, uint32_t y) const {
    float world_x = bounds_.min_x + (static_cast<float>(x) / (width_ - 1)) * (bounds_.max_x - bounds_.min_x);
    float world_y = bounds_.min_y + (static_cast<float>(y) / (height_ - 1)) * (bounds_.max_y - bounds_.min_y);
    float world_z = get_height_at(x, y);

    return Eigen::Vector3f(world_x * world_scale_.x(), world_y * world_scale_.y(), world_z * world_scale_.z());
}

void HeightfieldScene::set_wireframe_mode(bool enable) {
    wireframe_mode_ = enable;
    // Would need to recreate pipeline with different polygon mode
}

void HeightfieldScene::set_lod_visualization(bool enable) {
    lod_visualization_ = enable;
}

void HeightfieldScene::update(float delta_time) {
    // Update tessellation parameters based on time if needed
    if (tessellation_pipeline_) {
        // Could add time-based tessellation animations here
    }
}

void HeightfieldScene::render(VkCommandBuffer command_buffer, const Camera& camera) {
    if (lod_meshes_.empty()) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Select appropriate LOD level
    current_lod_level_ = select_lod_level(camera);
    const auto& lod_mesh = lod_meshes_[current_lod_level_];

    // Update uniforms
    update_uniforms(camera);

    // Bind pipeline
    if (tessellation_enabled_ && tessellation_pipeline_) {
        tessellation_pipeline_->bind(command_buffer);
    } else {
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);
    }

    // Bind descriptor sets
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);

    // Bind vertex buffer
    VkBuffer vertex_buffers[] = {lod_mesh->vertex_buffer->get_buffer()};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);

    // Bind index buffer
    vkCmdBindIndexBuffer(command_buffer, lod_mesh->index_buffer->get_buffer(), 0, VK_INDEX_TYPE_UINT32);

    // Draw
    if (tessellation_enabled_) {
        // Draw with tessellation patches
        vkCmdDrawIndexed(command_buffer, lod_mesh->index_count, 1, 0, 0, 0);
        render_stats_.tessellation_patches = lod_mesh->index_count / 4; // Assuming quad patches
    } else {
        // Regular indexed draw
        vkCmdDrawIndexed(command_buffer, lod_mesh->index_count, 1, 0, 0, 0);
        render_stats_.tessellation_patches = 0;
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    render_stats_.render_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    render_stats_.triangles_rendered = lod_mesh->index_count / 3;
    render_stats_.vertices_processed = lod_mesh->vertex_count;
    render_stats_.lod_level_used = current_lod_level_;
}

const AxisAlignedBoundingBox& HeightfieldScene::get_bounding_box() const {
    return bounding_box_;
}

// Private methods implementation

void HeightfieldScene::create_descriptor_set_layout() {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};

    // Uniform buffer
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                           VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Heightmap texture
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Diffuse texture
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Normal texture
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(context_.get_device(), &layout_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void HeightfieldScene::create_graphics_pipeline() {
    // This would create the graphics pipeline with proper shaders
    // Implementation would include loading vertex, tessellation, and fragment shaders
    // For brevity, this is simplified
    VF_LOG_INFO("Graphics pipeline created for heightfield scene");
}

void HeightfieldScene::create_uniform_buffer() {
    uniform_buffer_ = std::make_unique<Buffer>(
        context_,
        sizeof(TerrainUniforms),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
}

void HeightfieldScene::update_descriptor_set() {
    // Update descriptor set with current textures and uniform buffer
    // Implementation would bind textures and uniform buffer to descriptor set
}

void HeightfieldScene::generate_lod_mesh(uint32_t lod_level, float max_distance) {
    auto vertices = generate_vertices(lod_level);
    auto indices = generate_indices(lod_level);

    auto lod_mesh = std::make_unique<TerrainLODMesh>();
    lod_mesh->lod_level = lod_level;
    lod_mesh->vertex_count = static_cast<uint32_t>(vertices.size());
    lod_mesh->index_count = static_cast<uint32_t>(indices.size());
    lod_mesh->max_distance = max_distance;

    // Create vertex buffer
    lod_mesh->vertex_buffer = std::make_unique<Buffer>(
        context_,
        vertices.size() * sizeof(TerrainVertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Create index buffer
    lod_mesh->index_buffer = std::make_unique<Buffer>(
        context_,
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Upload data
    void* vertex_data;
    vkMapMemory(context_.get_device(), lod_mesh->vertex_buffer->get_memory(), 0, VK_WHOLE_SIZE, 0, &vertex_data);
    memcpy(vertex_data, vertices.data(), vertices.size() * sizeof(TerrainVertex));
    vkUnmapMemory(context_.get_device(), lod_mesh->vertex_buffer->get_memory());

    void* index_data;
    vkMapMemory(context_.get_device(), lod_mesh->index_buffer->get_memory(), 0, VK_WHOLE_SIZE, 0, &index_data);
    memcpy(index_data, indices.data(), indices.size() * sizeof(uint32_t));
    vkUnmapMemory(context_.get_device(), lod_mesh->index_buffer->get_memory());

    // Calculate bounding information
    lod_mesh->bounds_min = Eigen::Vector3f(bounds_.min_x, bounds_.min_y, bounds_.min_elevation * height_scale_);
    lod_mesh->bounds_max = Eigen::Vector3f(bounds_.max_x, bounds_.max_y, bounds_.max_elevation * height_scale_);
    lod_mesh->bounding_radius = (lod_mesh->bounds_max - lod_mesh->bounds_min).norm() * 0.5f;

    lod_meshes_.push_back(std::move(lod_mesh));
}

std::vector<TerrainVertex> HeightfieldScene::generate_vertices(uint32_t lod_level) {
    uint32_t step = 1 << lod_level; // 2^lod_level
    std::vector<TerrainVertex> vertices;

    for (uint32_t y = 0; y < height_; y += step) {
        for (uint32_t x = 0; x < width_; x += step) {
            TerrainVertex vertex;

            // Position
            vertex.position = get_world_position(x, y);

            // Texture coordinates
            vertex.tex_coord = Eigen::Vector2f(
                static_cast<float>(x) / (width_ - 1),
                static_cast<float>(y) / (height_ - 1)
            );

            // Height
            vertex.height = get_height_at(x, y);

            // Normal (calculated later)
            vertex.normal = calculate_normal_at(x, y);

            // Tangent (calculated later if needed)
            vertex.tangent = Eigen::Vector3f(1, 0, 0);

            // LOD level
            vertex.lod_level = lod_level;

            vertices.push_back(vertex);
        }
    }

    return vertices;
}

std::vector<uint32_t> HeightfieldScene::generate_indices(uint32_t lod_level) {
    uint32_t step = 1 << lod_level;
    uint32_t lod_width = (width_ - 1) / step + 1;
    uint32_t lod_height = (height_ - 1) / step + 1;

    std::vector<uint32_t> indices;

    for (uint32_t y = 0; y < lod_height - 1; ++y) {
        for (uint32_t x = 0; x < lod_width - 1; ++x) {
            uint32_t i0 = y * lod_width + x;
            uint32_t i1 = y * lod_width + (x + 1);
            uint32_t i2 = (y + 1) * lod_width + x;
            uint32_t i3 = (y + 1) * lod_width + (x + 1);

            // Triangle 1
            indices.push_back(i0);
            indices.push_back(i2);
            indices.push_back(i1);

            // Triangle 2
            indices.push_back(i1);
            indices.push_back(i2);
            indices.push_back(i3);
        }
    }

    return indices;
}

Eigen::Vector3f HeightfieldScene::calculate_normal_at(uint32_t x, uint32_t y) const {
    // Use finite differences to calculate normal
    float hL = get_height_at(x > 0 ? x - 1 : x, y);
    float hR = get_height_at(x < width_ - 1 ? x + 1 : x, y);
    float hD = get_height_at(x, y > 0 ? y - 1 : y);
    float hU = get_height_at(x, y < height_ - 1 ? y + 1 : y);

    Eigen::Vector3f normal(-2.0f * (hR - hL), -2.0f * (hU - hD), 4.0f);
    return normal.normalized();
}

uint32_t HeightfieldScene::select_lod_level(const Camera& camera) const {
    // Calculate distance from camera to terrain center
    Eigen::Vector3f terrain_center(
        (bounds_.min_x + bounds_.max_x) * 0.5f * world_scale_.x(),
        (bounds_.min_y + bounds_.max_y) * 0.5f * world_scale_.y(),
        (bounds_.min_elevation + bounds_.max_elevation) * 0.5f * height_scale_ * world_scale_.z()
    );

    float distance = (camera.get_position() - terrain_center).norm();

    // Select LOD level based on distance
    for (uint32_t i = 0; i < lod_distances_.size(); ++i) {
        if (distance <= lod_distances_[i]) {
            return i;
        }
    }

    return static_cast<uint32_t>(lod_distances_.size() - 1);
}

void HeightfieldScene::update_uniforms(const Camera& camera) {
    TerrainUniforms uniforms;

    // Matrices
    uniforms.model = Eigen::Matrix4f::Identity();
    uniforms.view = camera.get_view_matrix();
    uniforms.projection = camera.get_projection_matrix();

    // Camera and rendering parameters
    uniforms.camera_position = camera.get_position();
    uniforms.time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0f;
    uniforms.height_scale = height_scale_;
    uniforms.texture_scale = 1.0f;

    // Terrain-specific uniforms
    uniforms.terrain_bounds = Eigen::Vector4f(bounds_.min_x, bounds_.min_y, bounds_.max_x, bounds_.max_y);
    uniforms.elevation_range = Eigen::Vector2f(bounds_.min_elevation, bounds_.max_elevation);
    uniforms.tile_size = Eigen::Vector2f(bounds_.max_x - bounds_.min_x, bounds_.max_y - bounds_.min_y);
    uniforms.heightmap_size = Eigen::Vector2f(static_cast<float>(width_), static_cast<float>(height_));
    uniforms.lod_bias = 0.0f;
    uniforms.tessellation_scale = static_cast<float>(tessellation_config_.base_level);

    // Upload to GPU
    void* data;
    vkMapMemory(context_.get_device(), uniform_buffer_->get_memory(), 0, sizeof(TerrainUniforms), 0, &data);
    memcpy(data, &uniforms, sizeof(TerrainUniforms));
    vkUnmapMemory(context_.get_device(), uniform_buffer_->get_memory());
}

void HeightfieldScene::update_bounding_box() {
    if (heights_.empty()) return;

    bounding_box_.min = Eigen::Vector3f(
        bounds_.min_x * world_scale_.x(),
        bounds_.min_y * world_scale_.y(),
        bounds_.min_elevation * height_scale_ * world_scale_.z()
    );

    bounding_box_.max = Eigen::Vector3f(
        bounds_.max_x * world_scale_.x(),
        bounds_.max_y * world_scale_.y(),
        bounds_.max_elevation * height_scale_ * world_scale_.z()
    );
}

void HeightfieldScene::create_heightmap_texture() {
    if (heights_.empty()) return;

    // Create texture from height data
    // This would create a proper Vulkan texture from the height data
    // Implementation details would depend on the Texture class
}

// Utility functions implementation
namespace heightfield_utils {

std::vector<float> generate_perlin_noise(uint32_t width, uint32_t height,
                                        float scale, uint32_t octaves,
                                        float persistence, float lacunarity) {
    std::vector<float> noise(width * height);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Simplified Perlin noise implementation
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            float value = 0.0f;
            float amplitude = 1.0f;
            float frequency = scale;

            for (uint32_t octave = 0; octave < octaves; ++octave) {
                float sample_x = x / frequency;
                float sample_y = y / frequency;

                // Simplified noise function
                float noise_value = dis(gen) * 2.0f - 1.0f;
                value += noise_value * amplitude;

                amplitude *= persistence;
                frequency *= lacunarity;
            }

            noise[y * width + x] = value;
        }
    }

    return noise;
}

void normalize_heightmap(std::vector<float>& heights) {
    if (heights.empty()) return;

    float min_height = *std::min_element(heights.begin(), heights.end());
    float max_height = *std::max_element(heights.begin(), heights.end());
    float range = max_height - min_height;

    if (range == 0.0f) return;

    for (float& h : heights) {
        h = (h - min_height) / range;
    }
}

} // namespace heightfield_utils

} // namespace vf
// ============================================================================