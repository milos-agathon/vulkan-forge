// ============================================================================
// heightfield_scene.hpp – inline‑merged original + enhanced terrain header
// Generated 2025-07-08T09:17:23.748035 UTC
// ============================================================================
#pragma once
#include "vf/vk_common.hpp"
#include <glm/glm.hpp>     //  ← brings in glm::vec3 / mat4, etc.
#include <vulkan/vulkan.h>
#include <vector>
#include "scene.hpp"
#include "terrain_config.hpp"
#include "tessellation_pipeline.hpp"
#include "geotiff_loader.hpp"
#include "buffer.hpp"
#include "texture.hpp"
#include "mesh.hpp"
#include <Eigen/Dense>
#include <memory>
#include <array>
#include <optional>

// --------------------------- ORIGINAL DECLARATIONS --------------------------


namespace vf
{

struct HeightFieldScene
{
    /* public API --------------------------------------------------------- */
    void build(const std::vector<float>& heights,
               int nx, int ny,
               const std::vector<float>& colours = {},
               float zScale = 1.0f);

    ~HeightFieldScene();

    /* GPU buffers -------------------------------------------------------- */
    VkBuffer       vbo     = VK_NULL_HANDLE;
    VkDeviceMemory vboMem  = VK_NULL_HANDLE;
    VkBuffer       ibo     = VK_NULL_HANDLE;
    VkDeviceMemory iboMem  = VK_NULL_HANDLE;
    uint32_t       nIdx    = 0;

    /* A tiny default camera (handy for Python bindings) ------------------ */
    glm::vec3 eye{0.0f, 0.0f, 5.0f};
    glm::vec3 target{0.0f, 0.0f, 0.0f};
};

} // namespace vf

// --------------------------- TERRAIN EXTENSIONS ----------------------------

// Enhanced Heightfield Scene with Tessellation and Terrain Support
// Extends the original heightfield scene with GPU tessellation, multi-resolution
// mesh generation, and terrain-specific vertex attributes



namespace vf {

// Forward declarations
class VulkanContext;
class Camera;
class Renderer;

/**
 * Vertex structure for terrain rendering with tessellation support
 */
struct TerrainVertex {
    Eigen::Vector3f position;      // Base grid position (x, y, base_height)
    Eigen::Vector2f tex_coord;     // Texture coordinates for heightmap sampling
    Eigen::Vector3f normal;        // Pre-computed normal for lighting
    Eigen::Vector3f tangent;       // Tangent vector for normal mapping
    float height;                  // Actual height value from heightmap
    uint32_t lod_level;           // LOD level for this vertex

    static VkVertexInputBindingDescription get_binding_description() {
        VkVertexInputBindingDescription binding_desc{};
        binding_desc.binding = 0;
        binding_desc.stride = sizeof(TerrainVertex);
        binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding_desc;
    }

    static std::array<VkVertexInputAttributeDescription, 6> get_attribute_descriptions() {
        std::array<VkVertexInputAttributeDescription, 6> attr_descs{};

        // Position
        attr_descs[0].binding = 0;
        attr_descs[0].location = 0;
        attr_descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attr_descs[0].offset = offsetof(TerrainVertex, position);

        // Texture coordinates
        attr_descs[1].binding = 0;
        attr_descs[1].location = 1;
        attr_descs[1].format = VK_FORMAT_R32G32_SFLOAT;
        attr_descs[1].offset = offsetof(TerrainVertex, tex_coord);

        // Normal
        attr_descs[2].binding = 0;
        attr_descs[2].location = 2;
        attr_descs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attr_descs[2].offset = offsetof(TerrainVertex, normal);

        // Tangent
        attr_descs[3].binding = 0;
        attr_descs[3].location = 3;
        attr_descs[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        attr_descs[3].offset = offsetof(TerrainVertex, tangent);

        // Height
        attr_descs[4].binding = 0;
        attr_descs[4].location = 4;
        attr_descs[4].format = VK_FORMAT_R32_SFLOAT;
        attr_descs[4].offset = offsetof(TerrainVertex, height);

        // LOD level
        attr_descs[5].binding = 0;
        attr_descs[5].location = 5;
        attr_descs[5].format = VK_FORMAT_R32_UINT;
        attr_descs[5].offset = offsetof(TerrainVertex, lod_level);

        return attr_descs;
    }
};

/**
 * Uniform buffer for terrain rendering
 */
struct TerrainUniforms {
    Eigen::Matrix4f model;
    Eigen::Matrix4f view;
    Eigen::Matrix4f projection;
    Eigen::Vector3f camera_position;
    float time;
    Eigen::Vector2f viewport_size;
    float height_scale;
    float texture_scale;

    // Terrain-specific uniforms
    Eigen::Vector4f terrain_bounds;     // min_x, min_y, max_x, max_y
    Eigen::Vector2f elevation_range;    // min_elevation, max_elevation
    Eigen::Vector2f tile_size;          // width, height in world units
    Eigen::Vector2f heightmap_size;     // texture width, height
    float lod_bias;
    float tessellation_scale;
    Eigen::Vector2f padding;
};

/**
 * Multi-resolution terrain mesh for different LOD levels
 */
struct TerrainLODMesh {
    uint32_t lod_level;
    uint32_t vertex_count;
    uint32_t index_count;
    float max_distance;
    std::unique_ptr<Buffer> vertex_buffer;
    std::unique_ptr<Buffer> index_buffer;

    // Bounding information for culling
    Eigen::Vector3f bounds_min;
    Eigen::Vector3f bounds_max;
    float bounding_radius;
};

/**
 * Enhanced HeightfieldScene with tessellation and terrain features
 */
class HeightfieldScene : public Scene {
public:
    explicit HeightfieldScene(VulkanContext& context);
    ~HeightfieldScene() override;

    // Basic heightfield loading
    bool load_heightfield(const std::vector<std::vector<float>>& heights);
    bool load_heightfield(const float* data, uint32_t width, uint32_t height);

    // GeoTIFF loading with geographic information
    bool load_geotiff(const std::string& filename);
    bool load_geotiff(const GeoTiffData& geotiff_data);

    // Synthetic terrain generation
    bool generate_synthetic_terrain(uint32_t size, float amplitude = 100.0f, uint32_t octaves = 6);

    // Height scaling and transformation
    void set_height_scale(float scale);
    float get_height_scale() const { return height_scale_; }

    void set_world_scale(const Eigen::Vector3f& scale);
    const Eigen::Vector3f& get_world_scale() const { return world_scale_; }

    // Tessellation configuration
    void set_tessellation_config(const TessellationConfig& config);
    const TessellationConfig& get_tessellation_config() const;

    void enable_tessellation(bool enable);
    bool is_tessellation_enabled() const { return tessellation_enabled_; }

    // Multi-resolution mesh generation
    void generate_lod_meshes(const std::vector<float>& lod_distances);
    void set_lod_distances(const std::vector<float>& distances);
    const std::vector<float>& get_lod_distances() const { return lod_distances_; }

    // Normal and tangent generation
    void generate_normals();
    void generate_tangents();
    void smooth_normals(float smoothing_factor = 0.5f);

    // Texture support
    void set_heightmap_texture(std::shared_ptr<Texture> texture);
    void set_diffuse_texture(std::shared_ptr<Texture> texture);
    void set_normal_texture(std::shared_ptr<Texture> texture);

    std::shared_ptr<Texture> get_heightmap_texture() const { return heightmap_texture_; }
    std::shared_ptr<Texture> get_diffuse_texture() const { return diffuse_texture_; }
    std::shared_ptr<Texture> get_normal_texture() const { return normal_texture_; }

    // Geographic bounds
    void set_bounds(const GeographicBounds& bounds);
    const GeographicBounds& get_bounds() const { return bounds_; }

    // Mesh information
    uint32_t get_width() const { return width_; }
    uint32_t get_height() const { return height_; }
    uint32_t get_vertex_count() const;
    uint32_t get_triangle_count() const;

    // Height queries
    float get_height_at(uint32_t x, uint32_t y) const;
    float get_interpolated_height(float x, float y) const;
    Eigen::Vector3f get_world_position(uint32_t x, uint32_t y) const;

    // Rendering options
    void set_wireframe_mode(bool enable);
    bool is_wireframe_mode() const { return wireframe_mode_; }

    void set_lod_visualization(bool enable);
    bool is_lod_visualization() const { return lod_visualization_; }

    // Performance and statistics
    struct RenderStats {
        uint32_t triangles_rendered;
        uint32_t vertices_processed;
        uint32_t tessellation_patches;
        uint32_t lod_level_used;
        float render_time_ms;
    };

    const RenderStats& get_render_stats() const { return render_stats_; }

    // Scene interface implementation
    void update(float delta_time) override;
    void render(VkCommandBuffer command_buffer, const Camera& camera) override;
    const AxisAlignedBoundingBox& get_bounding_box() const override;

private:
    VulkanContext& context_;

    // Heightfield data
    std::vector<float> heights_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    float height_scale_ = 1.0f;
    Eigen::Vector3f world_scale_ = Eigen::Vector3f(1.0f, 1.0f, 1.0f);

    // Geographic information
    GeographicBounds bounds_;
    bool has_geographic_info_ = false;

    // Tessellation support
    std::unique_ptr<TessellationPipeline> tessellation_pipeline_;
    TessellationConfig tessellation_config_;
    bool tessellation_enabled_ = true;

    // Multi-resolution meshes
    std::vector<std::unique_ptr<TerrainLODMesh>> lod_meshes_;
    std::vector<float> lod_distances_;
    uint32_t current_lod_level_ = 0;

    // Textures
    std::shared_ptr<Texture> heightmap_texture_;
    std::shared_ptr<Texture> diffuse_texture_;
    std::shared_ptr<Texture> normal_texture_;

    // Rendering resources
    std::unique_ptr<Buffer> uniform_buffer_;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkPipeline graphics_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;

    // Rendering options
    bool wireframe_mode_ = false;
    bool lod_visualization_ = false;

    // Statistics
    mutable RenderStats render_stats_{};
    mutable AxisAlignedBoundingBox bounding_box_;

    // Private methods
    void create_descriptor_set_layout();
    void create_graphics_pipeline();
    void create_uniform_buffer();
    void update_descriptor_set();

    void generate_base_mesh();
    void generate_lod_mesh(uint32_t lod_level, float max_distance);

    std::vector<TerrainVertex> generate_vertices(uint32_t lod_level = 0);
    std::vector<uint32_t> generate_indices(uint32_t lod_level = 0);

    void calculate_normals(std::vector<TerrainVertex>& vertices, const std::vector<uint32_t>& indices);
    void calculate_tangents(std::vector<TerrainVertex>& vertices, const std::vector<uint32_t>& indices);

    uint32_t select_lod_level(const Camera& camera) const;
    void update_uniforms(const Camera& camera);
    void update_bounding_box();

    // Heightmap processing
    void create_heightmap_texture();
    void smooth_heightmap(float factor);

    // Utility functions
    inline uint32_t get_index(uint32_t x, uint32_t y) const {
        return y * width_ + x;
    }

    inline bool is_valid_coordinate(uint32_t x, uint32_t y) const {
        return x < width_ && y < height_;
    }

    Eigen::Vector3f calculate_normal_at(uint32_t x, uint32_t y) const;
    Eigen::Vector2f world_to_texture_coords(const Eigen::Vector3f& world_pos) const;
    Eigen::Vector3f texture_to_world_coords(const Eigen::Vector2f& tex_coords) const;
};

// Utility functions for heightfield processing
namespace heightfield_utils {
    /**
     * Generate Perlin noise heightmap
     */
    std::vector<float> generate_perlin_noise(uint32_t width, uint32_t height,
                                           float scale = 10.0f, uint32_t octaves = 6,
                                           float persistence = 0.5f, float lacunarity = 2.0f);

    /**
     * Apply gaussian blur to heightmap
     */
    void gaussian_blur(std::vector<float>& heights, uint32_t width, uint32_t height,
                      float radius = 1.0f);

    /**
     * Normalize heightmap to [0, 1] range
     */
    void normalize_heightmap(std::vector<float>& heights);

    /**
     * Apply erosion simulation
     */
    void apply_hydraulic_erosion(std::vector<float>& heights, uint32_t width, uint32_t height,
                               uint32_t iterations = 50000, float evaporation_rate = 0.01f);

    /**
     * Calculate slope map from heightmap
     */
    std::vector<float> calculate_slope_map(const std::vector<float>& heights,
                                         uint32_t width, uint32_t height);

    /**
     * Calculate aspect map (slope direction) from heightmap
     */
    std::vector<float> calculate_aspect_map(const std::vector<float>& heights,
                                          uint32_t width, uint32_t height);
}

} // namespace vf
// ============================================================================