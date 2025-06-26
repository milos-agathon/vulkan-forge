#include "vf/heightfield_scene.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <cstring>
#include <stdexcept>

using namespace vf;

/* vertex layout matching the pipeline (pos + rgba) */
struct VertexGPU
{
    glm::vec3 pos;
    glm::vec4 col;
};

/* ---------------- CPU mesh generation ------------------ */
static std::vector<VertexGPU> makeVerts(const std::vector<float>& h,
                                        int nx, int ny,
                                        const std::vector<float>& c,
                                        float zScale)
{
    if(static_cast<int>(h.size()) != nx * ny)
        throw std::runtime_error("height vector size mismatch");

    std::vector<VertexGPU> v;
    v.reserve(nx * ny);

    for(int j = 0; j < ny; ++j)
        for(int i = 0; i < nx; ++i)
        {
            int idx = j * nx + i;
            glm::vec3 p{float(i), float(j), h[idx] * zScale};
            glm::vec4 col{1,1,1,1};
            if(!c.empty())
            {
                col.r = c[idx*4+0];
                col.g = c[idx*4+1];
                col.b = c[idx*4+2];
                col.a = c[idx*4+3];
            }
            v.push_back({p,col});
        }
    return v;
}
static std::vector<uint32_t> makeIdx(int nx, int ny)
{
    std::vector<uint32_t> out;
    out.reserve((nx-1)*(ny-1)*6);
    for(int j = 0; j < ny-1; ++j)
        for(int i = 0; i < nx-1; ++i)
        {
            uint32_t tl =  j    *nx + i;
            uint32_t tr =  tl + 1;
            uint32_t bl = (j+1)*nx + i;
            uint32_t br =  bl + 1;
            out.insert(out.end(), {tl,bl,tr,  tr,bl,br});
        }
    return out;
}

/* ---------------- GPU upload --------------------------- */
void HeightFieldScene::build(const std::vector<float>& h,
                             int nx, int ny,
                             const std::vector<float>& col,
                             float zScale)
{
    auto& gpu = ctx();

    auto verts = makeVerts(h, nx, ny, col, zScale);
    auto idx   = makeIdx(nx, ny);
    nIdx       = uint32_t(idx.size());

    VkDeviceSize vBytes = verts.size() * sizeof(VertexGPU);
    VkDeviceSize iBytes = idx  .size() * sizeof(uint32_t);

    /* free old buffers */
    if(vbo)
    {
        vkDestroyBuffer(gpu.device, vbo, nullptr);
        vkFreeMemory   (gpu.device, vboMem, nullptr);
        vkDestroyBuffer(gpu.device, ibo, nullptr);
        vkFreeMemory   (gpu.device, iboMem, nullptr);
    }

    vbo = allocDeviceLocal(vBytes,
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          vboMem);
    ibo = allocDeviceLocal(iBytes,
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT   | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          iboMem);

    uploadToBuffer(vbo, verts.data(), vBytes);
    uploadToBuffer(ibo, idx .data(), iBytes);
}

HeightFieldScene::~HeightFieldScene()
{
    if(!vbo) return;
    auto& gpu = ctx();
    vkDestroyBuffer(gpu.device, vbo, nullptr);
    vkFreeMemory   (gpu.device, vboMem, nullptr);
    vkDestroyBuffer(gpu.device, ibo, nullptr);
    vkFreeMemory   (gpu.device, iboMem, nullptr);
}
