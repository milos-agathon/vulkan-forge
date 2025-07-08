// ============================================================================
// renderer.hpp – inline‑merged (original + enhanced)
// Generated 2025-07-08T09:19:21.418802 UTC
// ============================================================================
#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include "camera.hpp"
#include "scene.hpp"
#include "terrain_config.hpp"
#include "tessellation_pipeline.hpp"
#include <Eigen/Dense>
#include <chrono>

// --------------------------- ORIGINAL DECLARATIONS --------------------------
// cpp/include/vf/renderer.hpp
// Enhanced renderer with mesh pipeline support

namespace vf
{
struct HeightFieldScene;              // fwd-declared here, defined elsewhere
class MeshHandle;                     // Forward declaration for mesh rendering
class MeshPipeline;                   // Forward declaration for mesh pipeline
class MeshLoader;                     // Forward declaration for mesh loader
struct VertexLayout;                  // Forward declaration for vertex layout

class Renderer
{
public:
    Renderer(uint32_t w, uint32_t h); // off-screen resolution
    ~Renderer();

    /** Renders the given scene ➜ returns RGBA8 pixels (row-major, bottom-left origin). */
    std::vector<std::uint8_t> render(const HeightFieldScene& scene, uint32_t frameIdx = 0);

    /** Render a single mesh ➜ returns RGBA8 pixels */
    std::vector<std::uint8_t> render_mesh(std::shared_ptr<MeshHandle> mesh,
                                         const float* mvp_matrix = nullptr,
                                         uint32_t instance_count = 1);

    /** Render multiple meshes in a single frame */
    std::vector<std::uint8_t> render_meshes(const std::vector<std::shared_ptr<MeshHandle>>& meshes,
                                           const std::vector<float*>& mvp_matrices = {},
                                           const std::vector<uint32_t>& instance_counts = {});

    uint32_t width()  const { return m_width;  }
    uint32_t height() const { return m_height; }

    /** Set external vertex buffer for rendering */
    void set_vertex_buffer(VkBuffer buffer, uint32_t binding = 0);

    /** Clear external buffers */
    void clear_external_buffers();

    /** Get mesh loader for uploading meshes */
    MeshLoader& get_mesh_loader() { return *m_mesh_loader; }

    /** Get render pass for pipeline creation */
    VkRenderPass get_render_pass() const { return m_rpass; }

    /** Get Vulkan device */
    VkDevice get_device() const;

    /** Get command pool for mesh operations */
    VkCommandPool get_command_pool() const { return m_pool; }

    /** Get graphics queue */
    VkQueue get_graphics_queue() const;

    /** Performance statistics */
    struct RenderStats {
        float last_frame_time_ms = 0.0f;
        float avg_fps = 0.0f;
        uint32_t vertices_rendered = 0;
        uint32_t triangles_rendered = 0;
        uint32_t draw_calls = 0;
        uint64_t memory_used_bytes = 0;
    };

    const RenderStats& get_stats() const { return m_stats; }
    void reset_stats();

private:
    void createColorTarget();
    void createDepthTarget();
    void createRenderPass();
    void createFramebuffer();
    void createCommandObjects();
    void createReadbackBuffer();
    void createMeshPipeline();
    void createMeshLoader();

    /** Setup viewport and scissor for command buffer */
    void setupViewport(VkCommandBuffer cmd);

    /** Begin render pass with clear values */
    void beginRenderPass(VkCommandBuffer cmd);

    /** End render pass */
    void endRenderPass(VkCommandBuffer cmd);

    /** Copy rendered image to readback buffer */
    void copyToReadback(VkCommandBuffer cmd);

    /** Update performance statistics */
    void updateStats(const std::chrono::high_resolution_clock::time_point& start_time,
                    uint32_t vertices, uint32_t triangles, uint32_t draws);

    uint32_t           m_width  = 0;
    uint32_t           m_height = 0;

    /* GPU objects */
    VkImage            m_colorImg   = VK_NULL_HANDLE;
    VkDeviceMemory     m_colorMem   = VK_NULL_HANDLE;
    VkImageView        m_colorView  = VK_NULL_HANDLE;
    VkImage            m_depthImg   = VK_NULL_HANDLE;
    VkDeviceMemory     m_depthMem   = VK_NULL_HANDLE;
    VkImageView        m_depthView  = VK_NULL_HANDLE;
    VkRenderPass       m_rpass      = VK_NULL_HANDLE;
    VkFramebuffer      m_fb         = VK_NULL_HANDLE;
    VkCommandPool      m_pool       = VK_NULL_HANDLE;
    VkCommandBuffer    m_cmd        = VK_NULL_HANDLE;
    VkBuffer           m_readback   = VK_NULL_HANDLE;
    VkDeviceMemory     m_readMem    = VK_NULL_HANDLE;

    /* External buffer bindings */
    std::unordered_map<uint32_t, VkBuffer> m_external_buffers;

    /* Mesh rendering components */
    std::unique_ptr<MeshPipeline> m_mesh_pipeline;
    std::unique_ptr<MeshLoader> m_mesh_loader;

    /* Performance tracking */
    RenderStats m_stats;
    std::vector<float> m_frame_times;  // Rolling window for FPS calculation
    static constexpr size_t MAX_FRAME_TIMES = 60;  // Track last 60 frames

    /* Default matrices */
    std::vector<float> m_identity_matrix;
    std::vector<float> m_default_view_matrix;
    std::vector<float> m_default_proj_matrix;

    void updateDefaultMatrices();
};

} // namespace vf

// --------------------------- ENHANCED EXTENSIONS ---------------------------

// Enhanced Renderer with Tessellation and GPU-Driven Terrain Support
// Extends the base renderer with terrain-specific rendering capabilities,
// tessellation pipeline support, and GPU-driven culling



namespace vf {

// Forward declarations
class VulkanContext;
class TerrainRenderer;
class HeightfieldScene;

/**
 * Render statistics for performance monitoring
 */
struct RenderStats {
    // Timing
    float frame_time_ms = 0.0f;
    float gpu_time_ms = 0.0f;
    float cpu_time_ms = 0.0f;

    // Geometry
    uint64_t triangles_rendered = 0;
    uint64_t vertices_processed = 0;
    uint32_t draw_calls = 0;
    uint32_t tessellation_patches = 0;

    // Culling
    uint32_t objects_total = 0;
    uint32_t objects_rendered = 0;
    uint32_t objects_frustum_culled = 0;
    uint32_t objects_occlusion_culled = 0;
    uint32_t culled_objects = 0;

    // Memory
    uint64_t gpu_memory_used = 0;
    uint64_t vertex_buffer_memory = 0;
    uint64_t texture_memory = 0;

    // Terrain-specific
    uint32_t terrain_tiles_rendered = 0;
    uint32_t terrain_tiles_culled = 0;
    uint32_t active_lod_levels = 0;
    float tessellation_level_avg = 0.0f;

    void reset() {
        frame_time_ms = 0.0f;
        gpu_time_ms = 0.0f;
        cpu_time_ms = 0.0f;
        triangles_rendered = 0;
        vertices_processed = 0;
        draw_calls = 0;
        tessellation_patches = 0;
        objects_total = 0;
        objects_rendered = 0;
        objects_frustum_culled = 0;
        objects_occlusion_culled = 0;
        culled_objects = 0;
        gpu_memory_used = 0;
        vertex_buffer_memory = 0;
        texture_memory = 0;
        terrain_tiles_rendered = 0;
        terrain_tiles_culled = 0;
        active_lod_levels = 0;
        tessellation_level_avg = 0.0f;
    }
};

/**
 * Render pass configuration
 */
enum class RenderPassType {
    FORWARD,           // Forward rendering
    DEFERRED,          // Deferred rendering
    SHADOW_MAP,        // Shadow map generation
    DEPTH_PREPASS,     // Z-prepass for occlusion culling
    TERRAIN_ONLY,      // Terrain-only rendering
    WIREFRAME,         // Wireframe rendering
    DEBUG              // Debug visualization
};

/**
 * GPU-driven rendering support
 */
struct GPUDrivenConfig {
    bool enable_indirect_drawing = true;
    bool enable_gpu_culling = true;
    bool enable_occlusion_culling = false;
    bool enable_mesh_shaders = false;  // If supported
    uint32_t max_draw_commands = 10000;
    uint32_t culling_threads = 64;
};

/**
 * Enhanced Renderer class with terrain and tessellation support
 */
class Renderer {
public:
    explicit Renderer(VulkanContext& context);
    ~Renderer();

    // Disable copy constructor and assignment
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Scene management
    void add_scene(std::shared_ptr<Scene> scene);
    void remove_scene(std::shared_ptr<Scene> scene);
    void clear_scenes();
    const std::vector<std::shared_ptr<Scene>>& get_scenes() const { return scenes_; }

    // Camera management
    void set_camera(std::shared_ptr<Camera> camera);
    std::shared_ptr<Camera> get_camera() const { return camera_; }

    // Rendering
    void render();
    void render_frame(VkCommandBuffer command_buffer);
    void render_scene(VkCommandBuffer command_buffer, Scene& scene, RenderPassType pass_type = RenderPassType::FORWARD);

    // Tessellation support
    void enable_tessellation(bool enable);
    bool is_tessellation_supported() const;
    bool is_tessellation_enabled() const { return tessellation_enabled_; }

    void set_tessellation_config(const TessellationConfig& config);
    const TessellationConfig& get_tessellation_config() const;

    // GPU-driven rendering
    void enable_gpu_driven_rendering(bool enable);
    bool is_gpu_driven_rendering_enabled() const { return gpu_driven_enabled_; }

    void set_gpu_driven_config(const GPUDrivenConfig& config);
    const GPUDrivenConfig& get_gpu_driven_config() const { return gpu_driven_config_; }

    // Culling
    void set_culling_enabled(bool enable);
    bool is_culling_enabled() const { return culling_enabled_; }

    void set_occlusion_culling_enabled(bool enable);
    bool is_occlusion_culling_enabled() const { return occlusion_culling_enabled_; }

    // Render passes
    void set_render_pass_type(RenderPassType type);
    RenderPassType get_render_pass_type() const { return current_render_pass_; }

    // Multi-pass rendering
    void enable_depth_prepass(bool enable);
    void enable_shadow_mapping(bool enable);
    bool is_depth_prepass_enabled() const { return depth_prepass_enabled_; }
    bool is_shadow_mapping_enabled() const { return shadow_mapping_enabled_; }

    // Debug and visualization
    void set_wireframe_mode(bool enable);
    bool is_wireframe_mode() const { return wireframe_mode_; }

    void set_debug_visualization(bool enable);
    bool is_debug_visualization_enabled() const { return debug_visualization_; }

    void set_show_bounding_boxes(bool enable);
    void set_show_lod_levels(bool enable);
    void set_show_tessellation_levels(bool enable);

    // Performance and statistics
    const RenderStats& get_render_stats() const { return render_stats_; }
    void reset_render_stats() { render_stats_.reset(); }

    float get_fps() const;
    float get_frame_time_ms() const { return render_stats_.frame_time_ms; }

    // Viewport and resolution
    void resize(uint32_t width, uint32_t height);
    void set_viewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);
    Eigen::Vector2i get_viewport_size() const { return viewport_size_; }

    // Screenshot and frame capture
    std::vector<uint8_t> screenshot();
    void save_screenshot(const std::string& filename);

    // Resource management
    void cleanup();
    uint64_t get_memory_usage() const;

    // Advanced features
    void enable_temporal_upsampling(bool enable);
    void set_render_scale(float scale);  // For dynamic resolution scaling

    // Terrain-specific methods
    void register_terrain_renderer(std::shared_ptr<TerrainRenderer> terrain_renderer);
    void unregister_terrain_renderer(std::shared_ptr<TerrainRenderer> terrain_renderer);

private:
    VulkanContext& context_;

    // Scenes and camera
    std::vector<std::shared_ptr<Scene>> scenes_;
    std::shared_ptr<Camera> camera_;

    // Tessellation support
    std::unique_ptr<TessellationPipeline> tessellation_pipeline_;
    TessellationConfig tessellation_config_;
    bool tessellation_enabled_ = true;
    bool tessellation_supported_ = false;

    // GPU-driven rendering
    GPUDrivenConfig gpu_driven_config_;
    bool gpu_driven_enabled_ = false;

    // Culling
    bool culling_enabled_ = true;
    bool occlusion_culling_enabled_ = false;

    // Render passes
    RenderPassType current_render_pass_ = RenderPassType::FORWARD;
    bool depth_prepass_enabled_ = false;
    bool shadow_mapping_enabled_ = false;

    // Debug and visualization
    bool wireframe_mode_ = false;
    bool debug_visualization_ = false;
    bool show_bounding_boxes_ = false;
    bool show_lod_levels_ = false;
    bool show_tessellation_levels_ = false;

    // Viewport
    Eigen::Vector2i viewport_size_;
    VkViewport viewport_{};
    VkRect2D scissor_{};

    // Performance tracking
    RenderStats render_stats_;
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    float fps_smoothing_factor_ = 0.9f;
    float current_fps_ = 0.0f;

    // Vulkan resources
    VkRenderPass main_render_pass_ = VK_NULL_HANDLE;
    VkRenderPass depth_prepass_render_pass_ = VK_NULL_HANDLE;
    VkRenderPass shadow_map_render_pass_ = VK_NULL_HANDLE;

    std::vector<VkFramebuffer> main_framebuffers_;
    std::vector<VkFramebuffer> depth_prepass_framebuffers_;
    std::vector<VkFramebuffer> shadow_map_framebuffers_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;

    // Synchronization
    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;
    uint32_t current_frame_ = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    // Query pools for performance measurements
    VkQueryPool timestamp_query_pool_ = VK_NULL_HANDLE;
    VkQueryPool occlusion_query_pool_ = VK_NULL_HANDLE;

    // GPU-driven rendering resources
    std::unique_ptr<Buffer> draw_commands_buffer_;
    std::unique_ptr<Buffer> instance_data_buffer_;
    std::unique_ptr<Buffer> culling_data_buffer_;
    VkPipeline culling_compute_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout culling_pipeline_layout_ = VK_NULL_HANDLE;

    // Terrain renderers
    std::vector<std::weak_ptr<TerrainRenderer>> terrain_renderers_;

    // Advanced rendering features
    bool temporal_upsampling_enabled_ = false;
    float render_scale_ = 1.0f;

    // Private methods
    void initialize_vulkan_resources();
    void create_render_passes();
    void create_framebuffers();
    void create_command_pool();
    void create_command_buffers();
    void create_synchronization_objects();
    void create_query_pools();
    void setup_gpu_driven_rendering();
    void cleanup_vulkan_resources();

    // Rendering pipeline methods
    void begin_frame();
    void end_frame();
    void update_render_stats();

    void record_command_buffer(VkCommandBuffer command_buffer, uint32_t image_index);
    void record_depth_prepass(VkCommandBuffer command_buffer);
    void record_shadow_pass(VkCommandBuffer command_buffer);
    void record_main_pass(VkCommandBuffer command_buffer);
    void record_debug_pass(VkCommandBuffer command_buffer);

    // Culling methods
    void perform_frustum_culling();
    void perform_occlusion_culling();
    void update_gpu_culling_data();

    // Tessellation methods
    void setup_tessellation_pipeline();
    void update_tessellation_parameters();

    // Utility methods
    void transition_image_layout(VkCommandBuffer command_buffer, VkImage image,
                               VkImageLayout old_layout, VkImageLayout new_layout);
    void insert_debug_marker(VkCommandBuffer command_buffer, const std::string& name);
    void begin_debug_region(VkCommandBuffer command_buffer, const std::string& name);
    void end_debug_region(VkCommandBuffer command_buffer);

    uint32_t get_current_image_index() const;
    VkCommandBuffer get_current_command_buffer() const;

    // Memory management
    void track_gpu_memory_usage();
    void cleanup_unused_resources();
};

/**
 * Render queue for managing multiple render operations
 */
class RenderQueue {
public:
    struct RenderItem {
        std::shared_ptr<Scene> scene;
        RenderPassType pass_type;
        float depth;  // For depth sorting
        uint32_t priority;
    };

    void add_item(const RenderItem& item);
    void clear();
    void sort_by_depth();
    void sort_by_priority();

    const std::vector<RenderItem>& get_items() const { return items_; }

private:
    std::vector<RenderItem> items_;
};

/**
 * Render context for passing rendering state
 */
struct RenderContext {
    VkCommandBuffer command_buffer;
    const Camera* camera;
    RenderPassType pass_type;
    uint32_t frame_index;
    float delta_time;
    const RenderStats* stats;

    // Terrain-specific context
    bool tessellation_enabled;
    const TessellationConfig* tessellation_config;
    bool gpu_driven_enabled;
    bool wireframe_mode;
};

/**
 * Utility functions for renderer
 */
namespace renderer_utils {
    /**
     * Check if tessellation is supported on the current device
     */
    bool is_tessellation_supported(const VulkanContext& context);

    /**
     * Check if mesh shaders are supported
     */
    bool is_mesh_shader_supported(const VulkanContext& context);

    /**
     * Get optimal tessellation configuration for device
     */
    TessellationConfig get_optimal_tessellation_config(const VulkanContext& context);

    /**
     * Calculate LOD level based on distance and screen size
     */
    uint32_t calculate_lod_level(float distance, float screen_size, const TessellationConfig& config);

    /**
     * Estimate GPU memory usage for given configuration
     */
    uint64_t estimate_gpu_memory_usage(uint32_t vertex_count, uint32_t texture_count,
                                     const TessellationConfig& tessellation_config);
}

} // namespace vf
// ============================================================================