#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vf/renderer.hpp"
#include "vf/heightfield_scene.hpp"

namespace py  = pybind11;
using    vf::HeightFieldScene;
using    vf::Renderer;

/* helper: convert Python (nx,ny) height numpy array -> std::vector<float> */
static std::vector<float>
numpyToVector(py::array_t<float,py::array::c_style | py::array::forcecast> arr)
{
    py::buffer_info bi = arr.request();
    if (bi.ndim != 2)
        throw std::runtime_error("height map must be 2-D numpy array");
    size_t n = bi.shape[0]*bi.shape[1];
    const float* src = static_cast<float*>(bi.ptr);
    return std::vector<float>(src, src+n);
}

PYBIND11_MODULE(vulkan_forge, m)
{
    m.doc() = "Minimal Vulkan height-field renderer";

    /* ---------------------------------------------------- HeightFieldScene */
    py::class_<HeightFieldScene>(m,"HeightFieldScene")
        .def(py::init<>())
        .def("build",
             [](HeightFieldScene& self,
                py::array_t<float> heights,
                py::object colours,          // None or (ny,nx,4) float32
                float zscale)
             {
                 auto hvec = numpyToVector(heights);
                 int ny = int(heights.shape(0));
                 int nx = int(heights.shape(1));

                 std::vector<float> cvec;
                 if (!colours.is_none()) {
                     auto arr = py::array_t<float, py::array::c_style |
                                                    py::array::forcecast>(colours);
                     if (arr.ndim()!=3 || arr.shape(2)!=4)
                         throw std::runtime_error("colours must be (ny,nx,4) float32");
                     cvec.assign(static_cast<const float*>(arr.data()),
                                 static_cast<const float*>(arr.data())+nx*ny*4);
                 }
                 self.build(hvec,nx,ny,cvec,zscale);
             },
             py::arg("heights"),
             py::arg("colours")=py::none(),
             py::arg("zscale") = 1.0f
        )
        .def_readonly("n_indices",&HeightFieldScene::nIdx);

    /* -------------------------------------------------------------- Renderer */
    py::class_<Renderer>(m,"Renderer")
        .def(py::init<uint32_t,uint32_t>(),
             py::arg("width"),py::arg("height"))
        .def("render",
             [](Renderer& r, const HeightFieldScene& scn, uint32_t spp)
             {
                 auto img = r.render(scn,spp);
                 /* wrap result as NumPy (h,w,4) uint8 view */
                 py::capsule free_when_done(img.data(),[](void* p){ delete[] static_cast<uint8_t*>(p); });
                 return py::array_t<uint8_t>(
                     {int(r.height), int(r.width), 4},
                     {int(r.width*4), 4, 1},
                     img.data(), free_when_done);
             },
             py::arg("scene"),
             py::arg("spp") = 1
        );
}
