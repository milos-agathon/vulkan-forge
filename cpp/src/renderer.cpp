// ============================================================================
// renderer_merged_full.cpp  (auto‑generated 2025-07-08T08:37:52.169119 UTC)
// Concatenation: original renderer.cpp  +  renderer_cpp_enhanced.txt
// No lines removed.
// ============================================================================

/* ----------------------------- ORIGINAL BEGIN ----------------------------- */
// cpp/src/renderer.cpp
// Enhanced renderer with mesh pipeline support
#include "vf/renderer.hpp"
#include "vf/heightfield_scene.hpp"
#include "vf/mesh_loader.hpp"
#include "vf/mesh_pipeline.hpp"
#include "vf/vertex_buffer.hpp"
#include "vf/vk_common.hpp"
#include "vf/vma_util.hpp"

#include <cstring>   // memcpy
#include <stdexcept>
#include <chrono>
#include <algorithm>

#define VK_CHECK(call)                                                              \
    do { VkResult _r = (call); if (_r != VK_SUCCESS) throw std::runtime_error("VK"); } while (0)

using namespace vf;

Renderer::Renderer(uint32_t w, uint32_t h)
    : m_width{ w }, m_height{ h }
{
    createColorTarget();
    createDepthTarget();
    createRenderPass();
    createFramebuffer();
    createCommandObjects();
    createReadbackBuffer();
    createMeshLoader();
    createMeshPipeline();
    updateDefaultMatrices();
    
    // Initialize frame times tracking
    m_frame_times.reserve(MAX_FRAME_TIMES);
}

Renderer::~Renderer()
{
    auto& c = ctx();
    vkDeviceWaitIdle(c.device);

    // Destroy mesh components
    m_mesh_pipeline.reset();
    m_mesh_loader.reset();

    vkDestroyBuffer     (c.device, m_readback,  nullptr);
    vkFreeMemory        (c.device, m_readMem,   nullptr);

    vkDestroyFramebuffer(c.device, m_fb,        nullptr);
    vkDestroyRenderPass (c.device, m_rpass,     nullptr);

    vkDestroyImageView  (c.device, m_depthView, nullptr);
    vkDestroyImage      (c.device, m_depthImg,  nullptr);
    vkFreeMemory        (c.device, m_depthMem,  nullptr);

    vkDestroyImageView  (c.device, m_colorView, nullptr);
    vkDestroyImage      (c.device, m_colorImg,  nullptr);
    vkFreeMemory        (c.device, m_colorMem,  nullptr);

    vkFreeCommandBuffers(c.device, m_pool, 1, &m_cmd);
    vkDestroyCommandPool(c.device, m_pool, nullptr);
}

/* --------------------------------------------------------------------- */
/*  External buffer management                                           */
/* --------------------------------------------------------------------- */
void Renderer::set_vertex_buffer(VkBuffer buffer, uint32_t binding)
{
    m_external_buffers[binding] = buffer;
}

void Renderer::clear_external_buffers()
{
    m_external_buffers.clear();
}

VkDevice Renderer::get_device() const
{
    return ctx().device;
}

VkQueue Renderer::get_graphics_queue() const
{
    return ctx().graphicsQ;
}

void Renderer::reset_stats()
{
    m_stats = {};
    m_frame_times.clear();
}

/* --------------------------------------------------------------------- */
/*  Mesh rendering API                                                   */
/* --------------------------------------------------------------------- */
std::vector<std::uint8_t> Renderer::render_mesh(std::shared_ptr<MeshHandle> mesh, 
                                               const float* mvp_matrix,
                                               uint32_t instance_count)
{
    if (!mesh || !mesh->isValid()) {
        throw std::runtime_error("Invalid mesh handle");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto& c = ctx();

    /* --- record ------------------------------------------------------ */
    VK_CHECK(vkResetCommandBuffer(m_cmd, 0));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(m_cmd, &bi));

    setupViewport(m_cmd);
    beginRenderPass(m_cmd);

    // Bind mesh pipeline
    if (m_mesh_pipeline && m_mesh_pipeline->isValid()) {
        m_mesh_pipeline->bind(m_cmd);
        
        // Set MVP matrix (use default if none provided)
        const float* matrix = mvp_matrix ? mvp_matrix : m_identity_matrix.data();
        m_mesh_pipeline->pushConstants(m_cmd, VK_SHADER_STAGE_VERTEX_BIT, 0, 
                                      16 * sizeof(float), matrix);
        
        // Bind and draw mesh
        mesh->bind(m_cmd);
        mesh->draw(m_cmd, instance_count);
    }

    endRenderPass(m_cmd);
    copyToReadback(m_cmd);

    VK_CHECK(vkEndCommandBuffer(m_cmd));

    /* --- submit & wait ---------------------------------------------- */
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;

    VK_CHECK(vkQueueSubmit(c.graphicsQ, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.graphicsQ));

    /* --- read out ---------------------------------------------------- */
    std::vector<std::uint8_t> pixels(m_width * m_height * 4);
    void* map = nullptr;
    vkMapMemory(c.device, m_readMem, 0, VK_WHOLE_SIZE, 0, &map);
    std::memcpy(pixels.data(), map, pixels.size());
    vkUnmapMemory(c.device, m_readMem);

    // Update performance statistics
    updateStats(start_time, mesh->getVertexCount() * instance_count, 
               mesh->getTriangleCount() * instance_count, 1);

    return pixels;
}

std::vector<std::uint8_t> Renderer::render_meshes(const std::vector<std::shared_ptr<MeshHandle>>& meshes,
                                                 const std::vector<float*>& mvp_matrices,
                                                 const std::vector<uint32_t>& instance_counts)
{
    if (meshes.empty()) {
        throw std::runtime_error("No meshes to render");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto& c = ctx();
    
    uint32_t total_vertices = 0;
    uint32_t total_triangles = 0;
    uint32_t draw_calls = 0;

    /* --- record ------------------------------------------------------ */
    VK_CHECK(vkResetCommandBuffer(m_cmd, 0));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(m_cmd, &bi));

    setupViewport(m_cmd);
    beginRenderPass(m_cmd);

    // Bind mesh pipeline once
    if (m_mesh_pipeline && m_mesh_pipeline->isValid()) {
        m_mesh_pipeline->bind(m_cmd);
        
        // Render each mesh
        for (size_t i = 0; i < meshes.size(); ++i) {
            const auto& mesh = meshes[i];
            if (!mesh || !mesh->isValid()) {
                continue;
            }
            
            // Get MVP matrix for this mesh
            const float* matrix = m_identity_matrix.data();
            if (i < mvp_matrices.size() && mvp_matrices[i]) {
                matrix = mvp_matrices[i];
            }
            
            // Get instance count for this mesh
            uint32_t instances = 1;
            if (i < instance_counts.size()) {
                instances = instance_counts[i];
            }
            
            // Update push constants
            m_mesh_pipeline->pushConstants(m_cmd, VK_SHADER_STAGE_VERTEX_BIT, 0, 
                                          16 * sizeof(float), matrix);
            
            // Bind and draw mesh
            mesh->bind(m_cmd);
            mesh->draw(m_cmd, instances);
            
            // Update statistics
            total_vertices += mesh->getVertexCount() * instances;
            total_triangles += mesh->getTriangleCount() * instances;
            draw_calls++;
        }
    }

    endRenderPass(m_cmd);
    copyToReadback(m_cmd);

    VK_CHECK(vkEndCommandBuffer(m_cmd));

    /* --- submit & wait ---------------------------------------------- */
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;

    VK_CHECK(vkQueueSubmit(c.graphicsQ, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.graphicsQ));

    /* --- read out ---------------------------------------------------- */
    std::vector<std::uint8_t> pixels(m_width * m_height * 4);
    void* map = nullptr;
    vkMapMemory(c.device, m_readMem, 0, VK_WHOLE_SIZE, 0, &map);
    std::memcpy(pixels.data(), map, pixels.size());
    vkUnmapMemory(c.device, m_readMem);

    // Update performance statistics
    updateStats(start_time, total_vertices, total_triangles, draw_calls);

    return pixels;
}

/* --------------------------------------------------------------------- */
/*  Original heightfield rendering                                      */
/* --------------------------------------------------------------------- */
std::vector<std::uint8_t> Renderer::render(const HeightFieldScene& scene,
                                           uint32_t /*frameIdx*/)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    auto& c = ctx();

    /* --- record ------------------------------------------------------ */
    VK_CHECK(vkResetCommandBuffer(m_cmd, 0));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(m_cmd, &bi));

    setupViewport(m_cmd);
    beginRenderPass(m_cmd);

    // Bind external vertex buffers if any
    for (const auto& [binding, buffer] : m_external_buffers) {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(m_cmd, binding, 1, &buffer, &offset);
    }
    
    // TODO: Render heightfield scene here
    // For now, just clear to background color

    endRenderPass(m_cmd);
    copyToReadback(m_cmd);

    VK_CHECK(vkEndCommandBuffer(m_cmd));

    /* --- submit & wait ---------------------------------------------- */
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;

    VK_CHECK(vkQueueSubmit(c.graphicsQ, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.graphicsQ));

    /* --- read out ---------------------------------------------------- */
    std::vector<std::uint8_t> pixels(m_width * m_height * 4);
    void* map = nullptr;
    vkMapMemory(c.device, m_readMem, 0, VK_WHOLE_SIZE, 0, &map);
    std::memcpy(pixels.data(), map, pixels.size());
    vkUnmapMemory(c.device, m_readMem);

    // Update statistics for heightfield
    updateStats(start_time, 0, 0, 0);

    return pixels;
}

/* --------------------------------------------------------------------- */
/*  Helper functions                                                     */
/* --------------------------------------------------------------------- */
void Renderer::setupViewport(VkCommandBuffer cmd)
{
    VkViewport viewport{};
    viewport.width = static_cast<float>(m_width);
    viewport.height = static_cast<float>(m_height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = { m_width, m_height };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void Renderer::beginRenderPass(VkCommandBuffer cmd)
{
    VkClearValue clearValues[2];
    clearValues[0].color = { { 0.1f, 0.2f, 0.3f, 1.0f } };   // background colour
    clearValues[1].depthStencil = { 1.0f, 0 };                // clear depth

    VkRenderPassBeginInfo rbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rbi.renderPass  = m_rpass;
    rbi.framebuffer = m_fb;
    rbi.renderArea  = { {0, 0}, { m_width, m_height } };
    rbi.clearValueCount = 2;
    rbi.pClearValues    = clearValues;

    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
}

void Renderer::endRenderPass(VkCommandBuffer cmd)
{
    vkCmdEndRenderPass(cmd);
}

void Renderer::copyToReadback(VkCommandBuffer cmd)
{
    /* Transition so the image can be copied */
    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.srcAccessMask    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.image            = m_colorImg;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    /* Copy image → staging buffer */
    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent                 = { m_width, m_height, 1 };

    vkCmdCopyImageToBuffer(cmd,
        m_colorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_readback, 1, &copy);
}

void Renderer::updateStats(const std::chrono::high_resolution_clock::time_point& start_time,
                          uint32_t vertices, uint32_t triangles, uint32_t draws)
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float frame_time_ms = duration.count() / 1000.0f;
    
    m_stats.last_frame_time_ms = frame_time_ms;
    m_stats.vertices_rendered = vertices;
    m_stats.triangles_rendered = triangles;
    m_stats.draw_calls = draws;
    
    // Update rolling average FPS
    m_frame_times.push_back(frame_time_ms);
    if (m_frame_times.size() > MAX_FRAME_TIMES) {
        m_frame_times.erase(m_frame_times.begin());
    }
    
    if (!m_frame_times.empty()) {
        float avg_frame_time = 0.0f;
        for (float time : m_frame_times) {
            avg_frame_time += time;
        }
        avg_frame_time /= m_frame_times.size();
        m_stats.avg_fps = avg_frame_time > 0.0f ? 1000.0f / avg_frame_time : 0.0f;
    }
}

void Renderer::updateDefaultMatrices()
{
    // Identity matrix (4x4)
    m_identity_matrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    // Simple view matrix (looking down negative Z)
    m_default_view_matrix = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, -5.0f,  // Move back 5 units
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    // Simple perspective projection
    float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    float fov = 45.0f * (M_PI / 180.0f);  // 45 degrees in radians
    float near_plane = 0.1f;
    float far_plane = 100.0f;
    
    float f = 1.0f / std::tan(fov / 2.0f);
    m_default_proj_matrix = {
        f / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, (far_plane + near_plane) / (near_plane - far_plane), 
                    (2.0f * far_plane * near_plane) / (near_plane - far_plane),
        0.0f, 0.0f, -1.0f, 0.0f
    };
}

/* --------------------------------------------------------------------- */
/*  Initialization helpers                                               */
/* --------------------------------------------------------------------- */
void Renderer::createColorTarget()
{
    auto& c = ctx();

    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.extent      = { m_width, m_height, 1 };
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.format      = VK_FORMAT_R8G8B8A8_UNORM;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;

    VK_CHECK(vkCreateImage(c.device, &ici, nullptr, &m_colorImg));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c.device, m_colorImg, &req);

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = chooseType(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &m_colorMem));
    vkBindImageMemory(c.device, m_colorImg, m_colorMem, 0);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = m_colorImg;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(c.device, &vci, nullptr, &m_colorView));
}

void Renderer::createDepthTarget()
{
    auto& c = ctx();

    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.extent      = { m_width, m_height, 1 };
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.format      = VK_FORMAT_D32_SFLOAT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;

    VK_CHECK(vkCreateImage(c.device, &ici, nullptr, &m_depthImg));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c.device, m_depthImg, &req);

    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = chooseType(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(c.device, &mai, nullptr, &m_depthMem));
    vkBindImageMemory(c.device, m_depthImg, m_depthMem, 0);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = m_depthImg;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    VK_CHECK(vkCreateImageView(c.device, &vci, nullptr, &m_depthView));
}

void Renderer::createRenderPass()
{
    auto& c = ctx();

    VkAttachmentDescription attachments[2] = {};
    
    // Color attachment
    attachments[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    // Depth attachment
    attachments[1].format         = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 1;
    sub.pColorAttachments       = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpci.attachmentCount = 2;
    rpci.pAttachments    = attachments;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &sub;

    VK_CHECK(vkCreateRenderPass(c.device, &rpci, nullptr, &m_rpass));
}

void Renderer::createFramebuffer()
{
    auto& c = ctx();
    
    VkImageView attachments[] = { m_colorView, m_depthView };
    
    VkFramebufferCreateInfo fci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    fci.renderPass = m_rpass;
    fci.attachmentCount = 2;
    fci.pAttachments    = attachments;
    fci.width  = m_width;
    fci.height = m_height;
    fci.layers = 1;

    VK_CHECK(vkCreateFramebuffer(c.device, &fci, nullptr, &m_fb));
}

void Renderer::createCommandObjects()
{
    auto& c = ctx();

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c.gfxFamily;
    VK_CHECK(vkCreateCommandPool(c.device, &pci, nullptr, &m_pool));

    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool        = m_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &ai, &m_cmd));
}

void Renderer::createReadbackBuffer()
{
    const VkDeviceSize size = static_cast<VkDeviceSize>(m_width) * m_height * 4;
    m_readback = allocHostVisible(size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_readMem);
}

void Renderer::createMeshLoader()
{
    auto& c = ctx();
    
    m_mesh_loader = std::make_unique<MeshLoader>();
    VkResult result = m_mesh_loader->initialize(c.device, c.allocator, m_pool, c.graphicsQ);
    
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to initialize mesh loader");
    }
}

void Renderer::createMeshPipeline()
{
    auto& c = ctx();
    
    try {
        // Create basic mesh pipeline with position+normal+UV layout
        auto factory_result = MeshPipelineFactory::createTextured(c.device, m_rpass, m_mesh_pipeline);
        
        if (factory_result != VK_SUCCESS) {
            // Fallback to basic pipeline
            factory_result = MeshPipelineFactory::createBasic(c.device, m_rpass, m_mesh_pipeline);
            
            if (factory_result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create any mesh pipeline");
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create mesh pipeline: ") + e.what());
    }
}
/* ----------------------------- ORIGINAL END ------------------------------- */


/* ----------------------------- ENHANCED BEGIN ----------------------------- */
#include "vf/renderer.hpp"
#include "vf/vulkan_context.hpp"
#include "vf/terrain_renderer.hpp"
#include "vf/heightfield_scene.hpp"
#include "vf/utils.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <cstring>

namespace vf {

Renderer::Renderer(VulkanContext& context) 
    : context_(context), viewport_size_(1920, 1080) {
    
    // Check tessellation support
    VkPhysicalDeviceFeatures features = context_.get_device_features();
    tessellation_supported_ = features.tessellationShader;
    
    if (tessellation_supported_) {
        VF_LOG_INFO("Tessellation shaders supported");
        tessellation_enabled_ = true;
    } else {
        VF_LOG_WARN("Tessellation shaders not supported - falling back to regular rendering");
        tessellation_enabled_ = false;
    }
    
    // Initialize default configurations
    gpu_driven_config_.enable_indirect_drawing = true;
    gpu_driven_config_.enable_gpu_culling = true;
    gpu_driven_config_.enable_occlusion_culling = false;
    gpu_driven_config_.max_draw_commands = 10000;
    
    // Initialize Vulkan resources
    initialize_vulkan_resources();
    
    // Setup tessellation if supported
    if (tessellation_supported_) {
        setup_tessellation_pipeline();
    }
    
    // Setup GPU-driven rendering
    setup_gpu_driven_rendering();
    
    VF_LOG_INFO("Renderer initialized with tessellation: {}, GPU-driven: {}", 
                tessellation_enabled_, gpu_driven_enabled_);
}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::add_scene(std::shared_ptr<Scene> scene) {
    scenes_.push_back(scene);
    VF_LOG_DEBUG("Added scene, total scenes: {}", scenes_.size());
}

void Renderer::remove_scene(std::shared_ptr<Scene> scene) {
    auto it = std::find(scenes_.begin(), scenes_.end(), scene);
    if (it != scenes_.end()) {
        scenes_.erase(it);
        VF_LOG_DEBUG("Removed scene, total scenes: {}", scenes_.size());
    }
}

void Renderer::clear_scenes() {
    scenes_.clear();
    VF_LOG_DEBUG("Cleared all scenes");
}

void Renderer::set_camera(std::shared_ptr<Camera> camera) {
    camera_ = camera;
}

void Renderer::render() {
    if (!camera_) {
        VF_LOG_WARN("No camera set for rendering");
        return;
    }
    
    auto frame_start = std::chrono::high_resolution_clock::now();
    
    // Begin frame
    begin_frame();
    
    // Get current command buffer
    VkCommandBuffer command_buffer = get_current_command_buffer();
    
    // Record command buffer
    uint32_t image_index = get_current_image_index();
    record_command_buffer(command_buffer, image_index);
    
    // End frame
    end_frame();
    
    // Update performance statistics
    auto frame_end = std::chrono::high_resolution_clock::now();
    render_stats_.frame_time_ms = std::chrono::duration<float, std::milli>(frame_end - frame_start).count();
    update_render_stats();
    
    // Update FPS
    float current_frame_time = render_stats_.frame_time_ms;
    if (current_frame_time > 0.0f) {
        float instant_fps = 1000.0f / current_frame_time;
        current_fps_ = current_fps_ * fps_smoothing_factor_ + instant_fps * (1.0f - fps_smoothing_factor_);
    }
}

void Renderer::render_frame(VkCommandBuffer command_buffer) {
    if (!camera_) return;
    
    auto frame_start = std::chrono::high_resolution_clock::now();
    
    // Set viewport and scissor
    vkCmdSetViewport(command_buffer, 0, 1, &viewport_);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor_);
    
    // Perform culling if enabled
    if (culling_enabled_) {
        perform_frustum_culling();
        if (occlusion_culling_enabled_) {
            perform_occlusion_culling();
        }
    }
    
    // Render based on current render pass type
    switch (current_render_pass_) {
        case RenderPassType::FORWARD:
            if (depth_prepass_enabled_) {
                record_depth_prepass(command_buffer);
            }
            if (shadow_mapping_enabled_) {
                record_shadow_pass(command_buffer);
            }
            record_main_pass(command_buffer);
            break;
            
        case RenderPassType::DEFERRED:
            // Deferred rendering implementation
            record_main_pass(command_buffer);
            break;
            
        case RenderPassType::WIREFRAME:
            record_main_pass(command_buffer);
            break;
            
        case RenderPassType::DEBUG:
            record_debug_pass(command_buffer);
            break;
            
        default:
            record_main_pass(command_buffer);
            break;
    }
    
    auto frame_end = std::chrono::high_resolution_clock::now();
    render_stats_.frame_time_ms = std::chrono::duration<float, std::milli>(frame_end - frame_start).count();
}

void Renderer::render_scene(VkCommandBuffer command_buffer, Scene& scene, RenderPassType pass_type) {
    begin_debug_region(command_buffer, "Scene Render");
    
    // Create render context
    RenderContext context;
    context.command_buffer = command_buffer;
    context.camera = camera_.get();
    context.pass_type = pass_type;
    context.frame_index = current_frame_;
    context.delta_time = render_stats_.frame_time_ms / 1000.0f;
    context.stats = &render_stats_;
    context.tessellation_enabled = tessellation_enabled_;
    context.tessellation_config = &tessellation_config_;
    context.gpu_driven_enabled = gpu_driven_enabled_;
    context.wireframe_mode = wireframe_mode_;
    
    // Update scene
    scene.update(context.delta_time);
    
    // Check if scene is in frustum
    if (culling_enabled_) {
        const auto& scene_bounds = scene.get_bounding_box();
        if (!camera_->is_sphere_in_frustum(scene_bounds.center(), scene_bounds.radius())) {
            render_stats_.objects_frustum_culled++;
            end_debug_region(command_buffer);
            return;
        }
    }
    
    // Special handling for terrain scenes
    if (auto* heightfield_scene = dynamic_cast<HeightfieldScene*>(&scene)) {
        if (tessellation_enabled_ && tessellation_pipeline_) {
            tessellation_pipeline_->bind(command_buffer);
            update_tessellation_parameters();
        }
    }
    
    // Render the scene
    scene.render(command_buffer, *camera_);
    
    // Update statistics
    render_stats_.objects_rendered++;
    render_stats_.draw_calls++;
    
    // Add scene-specific statistics
    if (auto* heightfield_scene = dynamic_cast<HeightfieldScene*>(&scene)) {
        const auto& scene_stats = heightfield_scene->get_render_stats();
        render_stats_.triangles_rendered += scene_stats.triangles_rendered;
        render_stats_.vertices_processed += scene_stats.vertices_processed;
        render_stats_.tessellation_patches += scene_stats.tessellation_patches;
        render_stats_.terrain_tiles_rendered++;
    }
    
    end_debug_region(command_buffer);
}

void Renderer::enable_tessellation(bool enable) {
    if (enable && !tessellation_supported_) {
        VF_LOG_WARN("Cannot enable tessellation - not supported on this device");
        return;
    }
    
    tessellation_enabled_ = enable;
    VF_LOG_INFO("Tessellation {}", enable ? "enabled" : "disabled");
}

bool Renderer::is_tessellation_supported() const {
    return tessellation_supported_;
}

void Renderer::set_tessellation_config(const TessellationConfig& config) {
    tessellation_config_ = config;
    if (tessellation_pipeline_) {
        tessellation_pipeline_->update_config(config);
    }
}

const TessellationConfig& Renderer::get_tessellation_config() const {
    return tessellation_config_;
}

void Renderer::enable_gpu_driven_rendering(bool enable) {
    if (enable && !context_.get_device_features().multiDrawIndirect) {
        VF_LOG_WARN("Cannot enable GPU-driven rendering - indirect drawing not supported");
        return;
    }
    
    gpu_driven_enabled_ = enable;
    VF_LOG_INFO("GPU-driven rendering {}", enable ? "enabled" : "disabled");
}

void Renderer::set_gpu_driven_config(const GPUDrivenConfig& config) {
    gpu_driven_config_ = config;
    if (gpu_driven_enabled_) {
        setup_gpu_driven_rendering();
    }
}

void Renderer::set_culling_enabled(bool enable) {
    culling_enabled_ = enable;
}

void Renderer::set_occlusion_culling_enabled(bool enable) {
    occlusion_culling_enabled_ = enable && context_.get_device_features().occlusionQueryPrecise;
    if (enable && !context_.get_device_features().occlusionQueryPrecise) {
        VF_LOG_WARN("Occlusion culling requested but not supported");
    }
}

void Renderer::set_render_pass_type(RenderPassType type) {
    current_render_pass_ = type;
}

void Renderer::enable_depth_prepass(bool enable) {
    depth_prepass_enabled_ = enable;
}

void Renderer::enable_shadow_mapping(bool enable) {
    shadow_mapping_enabled_ = enable;
}

void Renderer::set_wireframe_mode(bool enable) {
    wireframe_mode_ = enable;
}

void Renderer::set_debug_visualization(bool enable) {
    debug_visualization_ = enable;
}

void Renderer::set_show_bounding_boxes(bool enable) {
    show_bounding_boxes_ = enable;
}

void Renderer::set_show_lod_levels(bool enable) {
    show_lod_levels_ = enable;
}

void Renderer::set_show_tessellation_levels(bool enable) {
    show_tessellation_levels_ = enable;
}

float Renderer::get_fps() const {
    return current_fps_;
}

void Renderer::resize(uint32_t width, uint32_t height) {
    viewport_size_ = Eigen::Vector2i(static_cast<int>(width), static_cast<int>(height));
    
    // Update viewport
    viewport_.x = 0.0f;
    viewport_.y = 0.0f;
    viewport_.width = static_cast<float>(width);
    viewport_.height = static_cast<float>(height);
    viewport_.minDepth = 0.0f;
    viewport_.maxDepth = 1.0f;
    
    // Update scissor
    scissor_.offset = {0, 0};
    scissor_.extent = {width, height};
    
    // Recreate framebuffers if needed
    create_framebuffers();
    
    VF_LOG_INFO("Renderer resized to {}x{}", width, height);
}

void Renderer::set_viewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    viewport_.x = static_cast<float>(x);
    viewport_.y = static_cast<float>(y);
    viewport_.width = static_cast<float>(width);
    viewport_.height = static_cast<float>(height);
    
    scissor_.offset = {static_cast<int32_t>(x), static_cast<int32_t>(y)};
    scissor_.extent = {width, height};
}

std::vector<uint8_t> Renderer::screenshot() {
    // Implementation for taking screenshot
    // This would read back the framebuffer data
    std::vector<uint8_t> data;
    // ... implementation details ...
    return data;
}

void Renderer::save_screenshot(const std::string& filename) {
    auto data = screenshot();
    // Save data to file
    VF_LOG_INFO("Screenshot saved to: {}", filename);
}

void Renderer::cleanup() {
    // Wait for device to finish
    vkDeviceWaitIdle(context_.get_device());
    
    cleanup_vulkan_resources();
    
    VF_LOG_INFO("Renderer cleanup completed");
}

uint64_t Renderer::get_memory_usage() const {
    return render_stats_.gpu_memory_used;
}

void Renderer::enable_temporal_upsampling(bool enable) {
    temporal_upsampling_enabled_ = enable;
}

void Renderer::set_render_scale(float scale) {
    render_scale_ = std::clamp(scale, 0.25f, 2.0f);
}

void Renderer::register_terrain_renderer(std::shared_ptr<TerrainRenderer> terrain_renderer) {
    terrain_renderers_.push_back(terrain_renderer);
}

void Renderer::unregister_terrain_renderer(std::shared_ptr<TerrainRenderer> terrain_renderer) {
    terrain_renderers_.erase(
        std::remove_if(terrain_renderers_.begin(), terrain_renderers_.end(),
                      [&](const std::weak_ptr<TerrainRenderer>& weak_ptr) {
                          return weak_ptr.expired() || weak_ptr.lock() == terrain_renderer;
                      }),
        terrain_renderers_.end()
    );
}

// Private method implementations

void Renderer::initialize_vulkan_resources() {
    create_render_passes();
    create_framebuffers();
    create_command_pool();
    create_command_buffers();
    create_synchronization_objects();
    create_query_pools();
    
    // Initialize viewport
    viewport_.x = 0.0f;
    viewport_.y = 0.0f;
    viewport_.width = static_cast<float>(viewport_size_.x());
    viewport_.height = static_cast<float>(viewport_size_.y());
    viewport_.minDepth = 0.0f;
    viewport_.maxDepth = 1.0f;
    
    scissor_.offset = {0, 0};
    scissor_.extent = {static_cast<uint32_t>(viewport_size_.x()), static_cast<uint32_t>(viewport_size_.y())};
}

void Renderer::create_render_passes() {
    // Create main render pass
    VkAttachmentDescription color_attachment{};
    color_attachment.format = context_.get_swapchain_image_format();
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = context_.get_depth_format();
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference depth_attachment_ref{};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;
    
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    
    std::array<VkAttachmentDescription, 2> attachments = {color_attachment, depth_attachment};
    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;
    
    if (vkCreateRenderPass(context_.get_device(), &render_pass_info, nullptr, &main_render_pass_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create main render pass");
    }
    
    // Create additional render passes (depth prepass, shadow map, etc.)
    // ... implementation details ...
}

void Renderer::create_framebuffers() {
    // Implementation for creating framebuffers
    // This would create framebuffers for all render passes
}

void Renderer::create_command_pool() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = context_.get_graphics_queue_family();
    
    if (vkCreateCommandPool(context_.get_device(), &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void Renderer::create_command_buffers() {
    command_buffers_.resize(MAX_FRAMES_IN_FLIGHT);
    
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());
    
    if (vkAllocateCommandBuffers(context_.get_device(), &alloc_info, command_buffers_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
}

void Renderer::create_synchronization_objects() {
    image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);
    
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(context_.get_device(), &semaphore_info, nullptr, &image_available_semaphores_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(context_.get_device(), &semaphore_info, nullptr, &render_finished_semaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(context_.get_device(), &fence_info, nullptr, &in_flight_fences_[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create synchronization objects");
        }
    }
}

void Renderer::create_query_pools() {
    // Create timestamp query pool for performance measurements
    VkQueryPoolCreateInfo timestamp_pool_info{};
    timestamp_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    timestamp_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    timestamp_pool_info.queryCount = 128; // Enough for multiple timing points
    
    if (vkCreateQueryPool(context_.get_device(), &timestamp_pool_info, nullptr, &timestamp_query_pool_) != VK_SUCCESS) {
        VF_LOG_WARN("Failed to create timestamp query pool");
    }
    
    // Create occlusion query pool if supported
    if (occlusion_culling_enabled_) {
        VkQueryPoolCreateInfo occlusion_pool_info{};
        occlusion_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        occlusion_pool_info.queryType = VK_QUERY_TYPE_OCCLUSION;
        occlusion_pool_info.queryCount = 1000; // Max objects for occlusion queries
        
        if (vkCreateQueryPool(context_.get_device(), &occlusion_pool_info, nullptr, &occlusion_query_pool_) != VK_SUCCESS) {
            VF_LOG_WARN("Failed to create occlusion query pool");
        }
    }
}

void Renderer::setup_gpu_driven_rendering() {
    if (!gpu_driven_enabled_) return;
    
    // Create buffers for GPU-driven rendering
    size_t draw_commands_size = gpu_driven_config_.max_draw_commands * sizeof(VkDrawIndexedIndirectCommand);
    draw_commands_buffer_ = std::make_unique<Buffer>(
        context_,
        draw_commands_size,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    
    // Create culling compute pipeline if GPU culling is enabled
    if (gpu_driven_config_.enable_gpu_culling) {
        // Implementation for culling compute pipeline
        // This would create a compute shader for GPU-based culling
    }
}

void Renderer::setup_tessellation_pipeline() {
    if (!tessellation_supported_) return;
    
    tessellation_pipeline_ = std::make_unique<TessellationPipeline>(context_, tessellation_config_);
}

void Renderer::cleanup_vulkan_resources() {
    // Cleanup in reverse order of creation
    
    // Synchronization objects
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (image_available_semaphores_[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(context_.get_device(), image_available_semaphores_[i], nullptr);
        }
        if (render_finished_semaphores_[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(context_.get_device(), render_finished_semaphores_[i], nullptr);
        }
        if (in_flight_fences_[i] != VK_NULL_HANDLE) {
            vkDestroyFence(context_.get_device(), in_flight_fences_[i], nullptr);
        }
    }
    
    // Query pools
    if (timestamp_query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(context_.get_device(), timestamp_query_pool_, nullptr);
    }
    if (occlusion_query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(context_.get_device(), occlusion_query_pool_, nullptr);
    }
    
    // Command pool (this also destroys command buffers)
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context_.get_device(), command_pool_, nullptr);
    }
    
    // Framebuffers
    for (auto framebuffer : main_framebuffers_) {
        vkDestroyFramebuffer(context_.get_device(), framebuffer, nullptr);
    }
    
    // Render passes
    if (main_render_pass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(context_.get_device(), main_render_pass_, nullptr);
    }
    if (depth_prepass_render_pass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(context_.get_device(), depth_prepass_render_pass_, nullptr);
    }
    if (shadow_map_render_pass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(context_.get_device(), shadow_map_render_pass_, nullptr);
    }
    
    // GPU-driven rendering resources
    if (culling_compute_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(context_.get_device(), culling_compute_pipeline_, nullptr);
    }
    if (culling_pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context_.get_device(), culling_pipeline_layout_, nullptr);
    }
}

void Renderer::begin_frame() {
    // Wait for fence
    vkWaitForFences(context_.get_device(), 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);
    
    // Reset stats for new frame
    render_stats_.reset();
    render_stats_.objects_total = static_cast<uint32_t>(scenes_.size());
    
    // Reset fence
    vkResetFences(context_.get_device(), 1, &in_flight_fences_[current_frame_]);
    
    // Reset command buffer
    vkResetCommandBuffer(command_buffers_[current_frame_], 0);
}

void Renderer::end_frame() {
    // Submit command buffer
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
    VkSemaphore wait_semaphores[] = {image_available_semaphores_[current_frame_]};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffers_[current_frame_];
    
    VkSemaphore signal_semaphores[] = {render_finished_semaphores_[current_frame_]};
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;
    
    if (vkQueueSubmit(context_.get_graphics_queue(), 1, &submit_info, in_flight_fences_[current_frame_]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }
    
    // Advance frame
    current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::record_command_buffer(VkCommandBuffer command_buffer, uint32_t image_index) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }
    
    begin_debug_region(command_buffer, "Main Render Pass");
    
    // Begin render pass
    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = main_render_pass_;
    render_pass_info.framebuffer = main_framebuffers_[image_index];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = {static_cast<uint32_t>(viewport_size_.x()), static_cast<uint32_t>(viewport_size_.y())};
    
    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color = {{0.0f, 0.0f, 0.2f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};
    render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();
    
    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    
    // Set viewport and scissor
    vkCmdSetViewport(command_buffer, 0, 1, &viewport_);
    vkCmdSetScissor(command_buffer, 0, 1, &scissor_);
    
    // Render all scenes
    for (auto& scene : scenes_) {
        if (scene) {
            render_scene(command_buffer, *scene, current_render_pass_);
        }
    }
    
    vkCmdEndRenderPass(command_buffer);
    end_debug_region(command_buffer);
    
    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer");
    }
}

void Renderer::record_depth_prepass(VkCommandBuffer command_buffer) {
    begin_debug_region(command_buffer, "Depth Prepass");
    // Implementation for depth prepass
    end_debug_region(command_buffer);
}

void Renderer::record_shadow_pass(VkCommandBuffer command_buffer) {
    begin_debug_region(command_buffer, "Shadow Pass");
    // Implementation for shadow mapping
    end_debug_region(command_buffer);
}

void Renderer::record_main_pass(VkCommandBuffer command_buffer) {
    begin_debug_region(command_buffer, "Main Pass");
    // Main rendering pass is handled in record_command_buffer
    end_debug_region(command_buffer);
}

void Renderer::record_debug_pass(VkCommandBuffer command_buffer) {
    begin_debug_region(command_buffer, "Debug Pass");
    // Debug visualization rendering
    end_debug_region(command_buffer);
}

void Renderer::perform_frustum_culling() {
    if (!camera_) return;
    
    render_stats_.objects_frustum_culled = 0;
    
    for (auto& scene : scenes_) {
        if (!scene) continue;
        
        const auto& bounds = scene->get_bounding_box();
        if (!camera_->is_sphere_in_frustum(bounds.center(), bounds.radius())) {
            render_stats_.objects_frustum_culled++;
        }
    }
}

void Renderer::perform_occlusion_culling() {
    // Implementation for occlusion culling
    render_stats_.objects_occlusion_culled = 0;
}

void Renderer::update_tessellation_parameters() {
    if (!tessellation_pipeline_ || !camera_) return;
    
    // Update tessellation parameters based on camera distance
    // This would update uniform buffers for tessellation shaders
}

void Renderer::update_render_stats() {
    render_stats_.culled_objects = render_stats_.objects_frustum_culled + render_stats_.objects_occlusion_culled;
    
    // Update memory usage statistics
    track_gpu_memory_usage();
}

void Renderer::track_gpu_memory_usage() {
    // Implementation for tracking GPU memory usage
    // This would query Vulkan memory usage
}

uint32_t Renderer::get_current_image_index() const {
    // In a real implementation, this would return the current swapchain image index
    return current_frame_;
}

VkCommandBuffer Renderer::get_current_command_buffer() const {
    return command_buffers_[current_frame_];
}

void Renderer::insert_debug_marker(VkCommandBuffer command_buffer, const std::string& name) {
    if (context_.is_debug_enabled()) {
        // Insert debug marker if validation layers are enabled
        // Implementation would use VK_EXT_debug_utils
    }
}

void Renderer::begin_debug_region(VkCommandBuffer command_buffer, const std::string& name) {
    if (context_.is_debug_enabled()) {
        // Begin debug region
        // Implementation would use VK_EXT_debug_utils
    }
}

void Renderer::end_debug_region(VkCommandBuffer command_buffer) {
    if (context_.is_debug_enabled()) {
        // End debug region
        // Implementation would use VK_EXT_debug_utils
    }
}

// Utility functions implementation
namespace renderer_utils {

bool is_tessellation_supported(const VulkanContext& context) {
    return context.get_device_features().tessellationShader;
}

bool is_mesh_shader_supported(const VulkanContext& context) {
    // Check for mesh shader extension support
    // This would query device extensions
    return false; // Placeholder
}

TessellationConfig get_optimal_tessellation_config(const VulkanContext& context) {
    TessellationConfig config;
    
    // Get device limits
    const auto& limits = context.get_device_limits();
    
    config.base_level = 8;
    config.max_level = std::min(64u, limits.maxTessellationGenerationLevel);
    config.min_level = 1;
    config.mode = TessellationMode::DISTANCE_BASED;
    
    return config;
}

uint32_t calculate_lod_level(float distance, float screen_size, const TessellationConfig& config) {
    // Simple LOD calculation based on distance
    for (uint32_t level = 0; level < 8; ++level) {
        float threshold = 500.0f * (1 << level);
        if (distance <= threshold) {
            return level;
        }
    }
    return 7; // Maximum LOD level
}

uint64_t estimate_gpu_memory_usage(uint32_t vertex_count, uint32_t texture_count, 
                                 const TessellationConfig& tessellation_config) {
    // Rough estimation of GPU memory usage
    uint64_t vertex_memory = vertex_count * sizeof(TerrainVertex);
    uint64_t texture_memory = texture_count * 4 * 1024 * 1024; // Assume 4MB per texture
    uint64_t tessellation_overhead = tessellation_config.max_level * tessellation_config.max_level * vertex_memory / 100;
    
    return vertex_memory + texture_memory + tessellation_overhead;
}

} // namespace renderer_utils

} // namespace vf

/* ----------------------------- ENHANCED END ------------------------------- */
