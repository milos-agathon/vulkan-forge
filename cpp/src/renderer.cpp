// cpp/src/renderer.cpp
// --------------------
#include "vf/renderer.hpp"
#include "vf/vk_common.hpp"

#include <cstring>   // memcpy
#include <stdexcept>

#define VK_CHECK(call)                                                              \
    do { VkResult _r = (call); if (_r != VK_SUCCESS) throw std::runtime_error("VK"); } while (0)

using namespace vf;

Renderer::Renderer(uint32_t w, uint32_t h)
    : m_width{ w }, m_height{ h }
{
    createColorTarget();
    createRenderPass();
    createFramebuffer();
    createCommandObjects();
    createReadbackBuffer();
}

Renderer::~Renderer()
{
    auto& c = ctx();
    vkDeviceWaitIdle(c.device);

    vkDestroyBuffer     (c.device, m_readback,  nullptr);
    vkFreeMemory        (c.device, m_readMem,   nullptr);

    vkDestroyFramebuffer(c.device, m_fb,        nullptr);
    vkDestroyRenderPass (c.device, m_rpass,     nullptr);

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

/* --------------------------------------------------------------------- */
/*  public API                                                           */
/* --------------------------------------------------------------------- */
std::vector<std::uint8_t> Renderer::render(const HeightFieldScene& /*scene*/,
                                           uint32_t /*frameIdx*/)
{
    auto& c = ctx();

    /* --- record ------------------------------------------------------ */
    VK_CHECK(vkResetCommandBuffer(m_cmd, 0));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(m_cmd, &bi));

    VkClearValue clear;
    clear.color = { { 0.1f, 0.2f, 0.3f, 1.0f } };   // background colour

    VkRenderPassBeginInfo rbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rbi.renderPass  = m_rpass;
    rbi.framebuffer = m_fb;
    rbi.renderArea  = { {0, 0}, { m_width, m_height } };
    rbi.clearValueCount = 1;
    rbi.pClearValues    = &clear;

    vkCmdBeginRenderPass(m_cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        // Bind external vertex buffers if any
        for (const auto& [binding, buffer] : m_external_buffers) {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(m_cmd, binding, 1, &buffer, &offset);
        }
        // TODO: bind pipeline, descriptor sets, draw meshes here
    vkCmdEndRenderPass(m_cmd);

    /* Transition so the image can be copied */
    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.srcAccessMask    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.image            = m_colorImg;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(m_cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    /* Copy image → staging buffer */
    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent                 = { m_width, m_height, 1 };

    vkCmdCopyImageToBuffer(m_cmd,
        m_colorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_readback, 1, &copy);

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

    return pixels;
}

/* --------------------------------------------------------------------- */
/*  helpers                                                              */
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

void Renderer::createRenderPass()
{
    auto& c = ctx();

    VkAttachmentDescription ad{};
    ad.format         = VK_FORMAT_R8G8B8A8_UNORM;
    ad.samples        = VK_SAMPLE_COUNT_1_BIT;
    ad.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ad.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    ad.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    ad.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference cref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &cref;

    VkRenderPassCreateInfo rpci{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &ad;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &sub;

    VK_CHECK(vkCreateRenderPass(c.device, &rpci, nullptr, &m_rpass));
}

void Renderer::createFramebuffer()
{
    auto& c = ctx();
    VkFramebufferCreateInfo fci{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    fci.renderPass = m_rpass;
    fci.attachmentCount = 1;
    fci.pAttachments    = &m_colorView;
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
