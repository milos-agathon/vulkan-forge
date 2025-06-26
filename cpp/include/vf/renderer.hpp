// cpp/include/vf/renderer.hpp
// ---------------------------
#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace vf
{
struct HeightFieldScene;              // fwd-declared here, defined elsewhere

class Renderer
{
public:
    Renderer(uint32_t w, uint32_t h); // off-screen resolution
    ~Renderer();

    /** Renders the given scene ➜ returns RGBA8 pixels (row-major, bottom-left origin). */
    std::vector<std::uint8_t> render(const HeightFieldScene& scene, uint32_t frameIdx = 0);

    uint32_t width()  const { return m_width;  }
    uint32_t height() const { return m_height; }

private:
    void createColorTarget();
    void createRenderPass();
    void createFramebuffer();
    void createCommandObjects();
    void createReadbackBuffer();

    uint32_t           m_width  = 0;
    uint32_t           m_height = 0;

    /* GPU objects */
    VkImage            m_colorImg   = VK_NULL_HANDLE;
    VkDeviceMemory     m_colorMem   = VK_NULL_HANDLE;
    VkImageView        m_colorView  = VK_NULL_HANDLE;
    VkRenderPass       m_rpass      = VK_NULL_HANDLE;
    VkFramebuffer      m_fb         = VK_NULL_HANDLE;
    VkCommandPool      m_pool       = VK_NULL_HANDLE;
    VkCommandBuffer    m_cmd        = VK_NULL_HANDLE;
    VkBuffer           m_readback   = VK_NULL_HANDLE;
    VkDeviceMemory     m_readMem    = VK_NULL_HANDLE;
};
} // namespace vf
