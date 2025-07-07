// cpp/include/vf/renderer.hpp
// Enhanced renderer with mesh pipeline support
#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <memory>

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