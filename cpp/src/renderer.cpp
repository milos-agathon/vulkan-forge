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