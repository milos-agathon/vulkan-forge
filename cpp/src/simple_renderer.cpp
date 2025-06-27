// src/simple_renderer.cpp - Simplified Vulkan renderer for testing build
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>
#include <cmath>

namespace py = pybind11;

class HeightFieldScene {
private:
    std::vector<float> vertices;
    uint32_t indexCount = 0;
    
public:
    void build(py::array_t<float> heights, py::object colors_obj, float zscale) {
        py::buffer_info buf = heights.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Heights must be a 2D array");
        }
        
        int ny = static_cast<int>(buf.shape[0]);
        int nx = static_cast<int>(buf.shape[1]);
        
        // Generate simple mesh
        vertices.clear();
        vertices.reserve(nx * ny * 3);
        
        float* h_ptr = static_cast<float*>(buf.ptr);
        
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                float xf = static_cast<float>(x) / (nx - 1) * 2.0f - 1.0f;
                float yf = static_cast<float>(y) / (ny - 1) * 2.0f - 1.0f;
                float zf = h_ptr[y * nx + x] * zscale;
                
                vertices.push_back(xf);
                vertices.push_back(yf);
                vertices.push_back(zf);
            }
        }
        
        // Calculate index count for triangle grid
        indexCount = static_cast<uint32_t>((nx - 1) * (ny - 1) * 6);
    }
    
    uint32_t getIndexCount() const { return indexCount; }
};

class Renderer {
private:
    uint32_t width_val, height_val;
    
public:
    Renderer(uint32_t w, uint32_t h) : width_val(w), height_val(h) {}
    
    py::array_t<uint8_t> render(const HeightFieldScene& scene) {
        // Create output array - correct syntax for shape
        std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(height_val), 
                                          static_cast<py::ssize_t>(width_val), 
                                          static_cast<py::ssize_t>(4)};
        auto result = py::array_t<uint8_t>(shape);
        py::buffer_info buf = result.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        
        // Simple gradient render for testing
        for (uint32_t y = 0; y < height_val; y++) {
            for (uint32_t x = 0; x < width_val; x++) {
                uint32_t idx = (y * width_val + x) * 4;
                
                // Create a simple gradient pattern
                float fx = static_cast<float>(x) / width_val;
                float fy = static_cast<float>(y) / height_val;
                
                // Simulate some 3D shading
                float shade = (fx + fy) * 0.5f;
                
                ptr[idx + 0] = static_cast<uint8_t>(shade * 200 + 55);  // R
                ptr[idx + 1] = static_cast<uint8_t>(shade * 180 + 60);  // G  
                ptr[idx + 2] = static_cast<uint8_t>(shade * 160 + 70);  // B
                ptr[idx + 3] = 255;                                      // A
            }
        }
        
        return result;
    }
    
    uint32_t width() const { return width_val; }
    uint32_t height() const { return height_val; }
};

PYBIND11_MODULE(_vulkan_forge_native, m) {
    m.doc() = "Vulkan height field renderer";
    
    py::class_<HeightFieldScene>(m, "HeightFieldScene")
        .def(py::init<>())
        .def("build", &HeightFieldScene::build,
             py::arg("heights"),
             py::arg("colors") = py::none(),
             py::arg("zscale") = 1.0f)
        .def_property_readonly("n_indices", &HeightFieldScene::getIndexCount);
    
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("width"), py::arg("height"))
        .def_property_readonly("width", &Renderer::width)
        .def_property_readonly("height", &Renderer::height)
        .def("render", &Renderer::render);
}