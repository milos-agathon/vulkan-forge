#include "vf/heightfield_scene.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <stdexcept>

using namespace vf;

/* local vertex that matches renderer pipeline (vec3 pos + vec4 rgba) */
struct VertexGPU {
    glm::vec3 pos;
    glm::vec4 col;
};

static std::vector<VertexGPU>
cpuGenerateMesh(const std::vector<float>& h, int nx, int ny,
                const std::vector<float>& c, float zScale)
{
    if (int(h.size()) != nx*ny)
        throw std::runtime_error("height vector size mismatch");

    std::vector<VertexGPU> verts;
    verts.reserve(nx*ny);

    /* create vertices row-major, unit grid in X/Y, height in Z */
    for (int j=0;j<ny;++j)
        for (int i=0;i<nx;++i)
        {
            int idx = j*nx + i;
            glm::vec3 p{float(i), float(j), h[idx]*zScale};
            glm::vec4 col{1,1,1,1};
            if (!c.empty()) {
                col.r = c[idx*4+0];
                col.g = c[idx*4+1];
                col.b = c[idx*4+2];
                col.a = c[idx*4+3];
            }
            verts.push_back({p,col});
        }

    return verts;
}

static std::vector<uint32_t>
cpuGenerateIndices(int nx, int ny)
{
    std::vector<uint32_t> idx;
    idx.reserve( (nx-1)*(ny-1)*6 );
    for (int j=0;j<ny-1;++j)
        for (int i=0;i<nx-1;++i)
        {
            uint32_t tl =  j    *nx + i;
            uint32_t tr =  tl + 1;
            uint32_t bl = (j+1)*nx + i;
            uint32_t br =  bl + 1;
            /* two triangles per quad */
            idx.insert(idx.end(), {tl,bl,tr,  tr,bl,br});
        }
    return idx;
}

/* ───────────────────────────────────────────────────────────── */

void HeightFieldScene::build(const std::vector<float>& heights,
                             int nx, int ny,
                             const std::vector<float>& colours,
                             float zScale)
{
    auto& gpu = ctx();

    /* 1. generate CPU mesh ------------------------------------------------ */
    auto verts = cpuGenerateMesh(heights,nx,ny,colours,zScale);
    auto idx   = cpuGenerateIndices(nx,ny);
    nIdx = uint32_t(idx.size());

    VkDeviceSize vBytes = verts.size()*sizeof(VertexGPU);
    VkDeviceSize iBytes = idx  .size()*sizeof(uint32_t);

    /* destroy previous GPU buffers (if any) ------------------------------ */
    if (vbo) {
        vkDestroyBuffer (gpu.device,vbo,nullptr);
        vkFreeMemory    (gpu.device,vboMem,nullptr);
        vkDestroyBuffer (gpu.device,ibo,nullptr);
        vkFreeMemory    (gpu.device,iboMem,nullptr);
    }

    /* 2. allocate device-local buffers ----------------------------------- */
    vbo = allocDeviceLocal(vBytes,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vboMem);
    ibo = allocDeviceLocal(iBytes,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT   |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            iboMem);

    /* 3. stage upload ----------------------------------------------------- */
    uploadToBuffer(vbo, verts.data(), vBytes);
    uploadToBuffer(ibo, idx .data(), iBytes);
}

HeightFieldScene::~HeightFieldScene()
{
    if (!vbo) return;                // nothing allocated
    auto& gpu = ctx();
    vkDestroyBuffer (gpu.device,vbo,nullptr);
    vkFreeMemory    (gpu.device,vboMem,nullptr);
    vkDestroyBuffer (gpu.device,ibo,nullptr);
    vkFreeMemory    (gpu.device,iboMem,nullptr);
}
