#pragma once
#include "heightfield_scene.hpp"

namespace vf {

class Renderer {
    VkRenderPass   pass{};
    VkPipelineLayout pipeLayout{};
    VkPipeline     pipe{};
    VkFramebuffer  fb{};
    VkImage        color{};
    VkImageView    colorView{};
    VkDeviceMemory colorMem{};

    uint32_t width{}, height{};
public:
    Renderer(uint32_t w, uint32_t h);
    ~Renderer();

    /* returns RGBA8 host buffer */
    std::vector<uint8_t> render(const HeightFieldScene&, uint32_t spp = 16);
};

} // ns
