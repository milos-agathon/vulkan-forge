#pragma once
#include "vk_common.hpp"
#include <vector>

namespace vf {

struct HeightFieldScene {
    /* raw data copied from Python */
    std::vector<float> vertices;   // x,y,z,r,g,b,a   (N*7)
    std::vector<uint32_t> indices; // 3*K

    /* GPU buffers */
    VkBuffer vbo{}, ibo{};
    VkDeviceMemory vboMem{}, iboMem{};
    uint32_t       nIdx{0};

    /* camera */
    glm::vec3 eye {0,0,5}, target{0,0,0};
    float fov = 45.f;

    void upload();                 // CPU→GPU
    void destroy();                // free buffers
};

} // ns
