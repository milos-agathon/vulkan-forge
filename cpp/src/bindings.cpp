#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "vf/renderer.hpp"
#include "vf/heightfield_scene.hpp"
#include "vf/vma_util.hpp"

namespace py = pybind11;
using vf::HeightFieldScene;
using vf::Renderer;

/* -------------------------------------------------------------------------
   Convert a 2-D NumPy float32 array → std::vector<float>
   --------------------------------------------------------------------- */
static std::vector<float>
numpyToVector(py::array_t<float, py::array::c_style | py::array::forcecast> arr)
{
    py::buffer_info bi = arr.request();
    if (bi.ndim != 2)
        throw std::runtime_error("height map must be 2-D numpy array");

    size_t n   = static_cast<size_t>(bi.shape[0]) * static_cast<size_t>(bi.shape[1]);
    auto *src  = static_cast<float *>(bi.ptr);
    return { src, src + n };
}

/* --------------------------------------------------------------------- */
PYBIND11_MODULE(_vulkan_forge_native, m)
{
    m.doc() = "Minimal Vulkan height-field renderer";

    py::register_exception<vf::VulkanForgeError>(m, "VulkanForgeError");

    /* ------------------------- HeightFieldScene ---------------------- */
    py::class_<HeightFieldScene>(m, "HeightFieldScene")
        .def(py::init<>())
        .def("build",
             [](HeightFieldScene& self,
                py::array_t<float> heights,
                py::object         colours,   // None or (ny,nx,4) float32
                float              zscale)
             {
                 auto hvec = numpyToVector(heights);
                 int ny = static_cast<int>(heights.shape(0));
                 int nx = static_cast<int>(heights.shape(1));

                 std::vector<float> cvec;
                 if (!colours.is_none())
                 {
                     py::array_t<float, py::array::c_style |
                                         py::array::forcecast> arr(colours);
                     if (arr.ndim() != 3 || arr.shape(2) != 4)
                         throw std::runtime_error("colours must be (ny,nx,4) float32");

                     const float* src = static_cast<const float*>(arr.data());
                     cvec.assign(src, src + nx * ny * 4);
                 }

                 self.build(hvec, nx, ny, cvec, zscale);
             },
             py::arg("heights"),
             py::arg("colours") = py::none(),
             py::arg("zscale")  = 1.0f)
        .def_readonly("n_indices", &HeightFieldScene::nIdx);

    /* ----------------------------- Renderer -------------------------- */
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("width"), py::arg("height"))

        .def_property_readonly("width",  &Renderer::width)
        .def_property_readonly("height", &Renderer::height)

        .def("render",
             [](Renderer& r, const HeightFieldScene& scn, uint32_t spp)
             {
                 auto img = r.render(scn, spp);

                 /* expose as NumPy (h,w,4) uint8 without an extra copy */
                 py::capsule free_when_done(
                     img.data(),
                     [](void* p){ delete[] static_cast<uint8_t*>(p); });

                 return py::array_t<uint8_t>(
                     { static_cast<int>(r.height()),
                       static_cast<int>(r.width()),
                       4 },
                     { static_cast<int>(r.width()) * 4, 4, 1 },
                     img.data(),
                     free_when_done);
             },
             py::arg("scene"),
             py::arg("spp") = 1);

    m.def("create_allocator",
          [](std::uintptr_t inst, std::uintptr_t phys, std::uintptr_t dev) {
              return reinterpret_cast<std::uintptr_t>(
                  vf::create_allocator(reinterpret_cast<VkInstance>(inst),
                                       reinterpret_cast<VkPhysicalDevice>(phys),
                                       reinterpret_cast<VkDevice>(dev)));
          },
          py::arg("instance"), py::arg("physical_device"), py::arg("device"));

    m.def("allocate_buffer",
          [](std::uintptr_t allocator, std::size_t size, uint32_t usage) {
              VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
              info.size = size;
              info.usage = usage;
              info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
              VkBuffer buf{};
              VmaAllocation alloc = vf::allocate_buffer(
                  reinterpret_cast<VmaAllocator>(allocator), &info, &buf);
              return py::make_tuple(reinterpret_cast<std::uintptr_t>(buf),
                                   reinterpret_cast<std::uintptr_t>(alloc));
          },
          py::arg("allocator"), py::arg("size"), py::arg("usage"));

    m.def("destroy_allocator",
          [](std::uintptr_t allocator) {
              vf::destroy_allocator(reinterpret_cast<VmaAllocator>(allocator));
          },
          py::arg("allocator"));
}
