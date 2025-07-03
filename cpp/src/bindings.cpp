#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>

#include "vf/renderer.hpp"
#include "vf/heightfield_scene.hpp"
#include "vf/vma_util.hpp"
#include "vf/vk_common.hpp"
#include <unordered_map>
#include <memory>

namespace py = pybind11;
using vf::HeightFieldScene;
using vf::Renderer;
using vf::MappedBuffer;

/* -------------------------------------------------------------------------
   NumPy format to Vulkan format mapping
   --------------------------------------------------------------------- */
static VkFormat numpy_dtype_to_vk_format(const std::string& format, size_t itemsize) {
    if (format == "f" && itemsize == 4) return VK_FORMAT_R32_SFLOAT;
    if (format == "f" && itemsize == 8) return VK_FORMAT_R64_SFLOAT;
    if (format == "i" && itemsize == 4) return VK_FORMAT_R32_SINT;
    if (format == "u" && itemsize == 4) return VK_FORMAT_R32_UINT;
    if (format == "i" && itemsize == 2) return VK_FORMAT_R16_SINT;
    if (format == "u" && itemsize == 2) return VK_FORMAT_R16_UINT;
    if (format == "i" && itemsize == 1) return VK_FORMAT_R8_SINT;
    if (format == "u" && itemsize == 1) return VK_FORMAT_R8_UINT;
    throw std::runtime_error("Unsupported NumPy dtype for Vulkan");
}

/* -------------------------------------------------------------------------
   Zero-copy NumPy buffer wrapper
   --------------------------------------------------------------------- */
class NumpyBuffer {
public:
    NumpyBuffer(VmaAllocator allocator, py::buffer& buf, VkBufferUsageFlags usage)
        : m_allocator(allocator) {
        py::buffer_info info = buf.request();
        
        // Calculate total size
        m_size = info.itemsize;
        for (auto s : info.shape) {
            m_size *= s;
        }
        
        // Create mapped buffer
        m_buffer = vf::create_mapped_buffer(m_allocator, m_size, usage);
        
        // Copy data if not writable or has strides
        if (info.readonly || !is_contiguous(info)) {
            std::memcpy(m_buffer.mapped_data, info.ptr, m_size);
            m_owns_data = true;
        } else {
            // Zero-copy: just reference the data
            m_numpy_data = info.ptr;
            m_buffer.mapped_data = info.ptr;
        }
        
        // Store format info
        m_format = numpy_dtype_to_vk_format(info.format, info.itemsize);
        m_shape = info.shape;
        m_strides = info.strides;
    }
    
    ~NumpyBuffer() {
        if (m_buffer.buffer) {
            vf::destroy_mapped_buffer(m_allocator, m_buffer);
        }
    }
    
    VkBuffer get_buffer() const { return m_buffer.buffer; }
    VkDeviceSize get_size() const { return m_size; }
    VkFormat get_format() const { return m_format; }
    void* get_mapped_data() const { return m_buffer.mapped_data; }
    
    void sync_to_gpu() {
        if (m_numpy_data && !m_owns_data) {
            // Ensure CPU writes are visible to GPU
            VmaAllocationInfo info;
            vmaGetAllocationInfo(m_allocator, m_buffer.allocation, &info);
            vmaFlushAllocation(m_allocator, m_buffer.allocation, 0, VK_WHOLE_SIZE);
        }
    }
    
    void sync_from_gpu() {
        if (m_numpy_data && !m_owns_data) {
            // Ensure GPU writes are visible to CPU
            VmaAllocationInfo info;
            vmaGetAllocationInfo(m_allocator, m_buffer.allocation, &info);
            vmaInvalidateAllocation(m_allocator, m_buffer.allocation, 0, VK_WHOLE_SIZE);
        }
    }
    
private:
    bool is_contiguous(const py::buffer_info& info) {
        size_t expected_stride = info.itemsize;
        for (int i = info.ndim - 1; i >= 0; --i) {
            if (info.strides[i] != expected_stride) return false;
            expected_stride *= info.shape[i];
        }
        return true;
    }
    
    VmaAllocator m_allocator;
    MappedBuffer m_buffer{};
    VkDeviceSize m_size = 0;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    std::vector<ssize_t> m_shape;
    std::vector<ssize_t> m_strides;
    void* m_numpy_data = nullptr;
    bool m_owns_data = false;
};

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

    py::register_exception<vf::VulkanForgeError>(m, "VulkanForgeError");

    +    /* ------------------------- NumpyBuffer ---------------------- */
    py::class_<NumpyBuffer, std::shared_ptr<NumpyBuffer>>(m, "NumpyBuffer")
        .def(py::init([](std::uintptr_t allocator_ptr, py::buffer buf, uint32_t usage) {
            auto allocator = reinterpret_cast<VmaAllocator>(allocator_ptr);
            return std::make_shared<NumpyBuffer>(allocator, buf, usage);
        }),
        py::arg("allocator"),
        py::arg("buffer"),
        py::arg("usage") = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
        .def("get_buffer", [](const NumpyBuffer& self) {
            return reinterpret_cast<std::uintptr_t>(self.get_buffer());
        })
        .def("get_size", &NumpyBuffer::get_size)
        .def("get_format", &NumpyBuffer::get_format)
        .def("sync_to_gpu", &NumpyBuffer::sync_to_gpu)
        .def("sync_from_gpu", &NumpyBuffer::sync_from_gpu)
        .def("__enter__", [](std::shared_ptr<NumpyBuffer> self) { return self; })
        .def("__exit__", [](NumpyBuffer& self, py::args) { });

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

        
        .def("set_vertex_buffer", [](Renderer& r, std::shared_ptr<NumpyBuffer> buffer, uint32_t binding) {
            r.set_vertex_buffer(buffer->get_buffer(), binding);
        },
        py::arg("buffer"),
        py::arg("binding") = 0)

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

    m.def(
        "create_allocator",
        [](std::uintptr_t inst, std::uintptr_t phys, std::uintptr_t dev) {
            VmaAllocator alloc = vf::create_allocator(
                reinterpret_cast<VkInstance>(inst),
                reinterpret_cast<VkPhysicalDevice>(phys),
                reinterpret_cast<VkDevice>(dev));
            return py::capsule(
                alloc,
                "VmaAllocator",
                [](PyObject* cap) {
                    vf::destroy_allocator(reinterpret_cast<VmaAllocator>(
                        PyCapsule_GetPointer(cap, "VmaAllocator")));
                });
        },
        py::arg("instance"),
        py::arg("physical_device"),
        py::arg("device"));

    m.def(
        "allocate_buffer",
        [](std::uintptr_t allocator_ptr, std::size_t size, uint32_t usage) {
            auto allocator = reinterpret_cast<VmaAllocator>(allocator_ptr);
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = size;
            info.usage = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkBuffer buf{};
            VmaAllocation alloc =
                vf::allocate_buffer(allocator, &info, &buf);
            return py::make_tuple(
                reinterpret_cast<std::uintptr_t>(buf),
                reinterpret_cast<std::uintptr_t>(alloc));
        },
        py::arg("allocator"),
        py::arg("size"),
        py::arg("usage"));

    m.def(
        "destroy_allocator",
        [](std::uintptr_t allocator_ptr) {
            vf::destroy_allocator(reinterpret_cast<VmaAllocator>(allocator_ptr));
        },
        py::arg("allocator"));

    /* Buffer usage flags */
    m.attr("BUFFER_USAGE_VERTEX_BUFFER") = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    m.attr("BUFFER_USAGE_INDEX_BUFFER") = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    m.attr("BUFFER_USAGE_UNIFORM_BUFFER") = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    m.attr("BUFFER_USAGE_STORAGE_BUFFER") = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    m.attr("BUFFER_USAGE_TRANSFER_SRC") = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    m.attr("BUFFER_USAGE_TRANSFER_DST") = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
}
