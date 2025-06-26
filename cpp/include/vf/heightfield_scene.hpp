#pragma once
#include "vf/vk_common.hpp"

#include <glm/glm.hpp>     //  ← brings in glm::vec3 / mat4, etc.
#include <vulkan/vulkan.h>
#include <vector>

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
