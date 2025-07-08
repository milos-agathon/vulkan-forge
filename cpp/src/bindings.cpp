// ============================================================================
// bindings.cpp – auto‑merged original + terrain bindings (inline)
// Generated 2025-07-08T08:48:47.275016 UTC
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>
#include "vf/renderer.hpp"
#include "vf/heightfield_scene.hpp"
#include "vf/mesh_loader.hpp"
#include "vf/mesh_pipeline.hpp"
#include "vf/vertex_buffer.hpp"
#include "vf/vma_util.hpp"
#include "vf/vk_common.hpp"
#include <unordered_map>
#include <memory>
#include <pybind11/eigen.h>
#include "vf/terrain_renderer.hpp"
#include "vf/geotiff_loader.hpp"
#include "vf/tessellation_pipeline.hpp"
#include "vf/camera.hpp"
#include "vf/vulkan_context.hpp"

// --------------------------- ORIGINAL BINDINGS -----------------------------


namespace py = pybind11;
using vf::HeightFieldScene;
using vf::Renderer;
using vf::MappedBuffer;
using vf::MeshLoader;
using vf::MeshHandle;
using vf::MeshPipeline;
using vf::VertexBuffer;
using vf::VertexLayout;

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

/* -------------------------------------------------------------------------
   Convert vertex format string to enum
   --------------------------------------------------------------------- */
static vf::VertexLayout string_to_vertex_layout(const std::string& format) {
    if (format == "position_3f") return vf::VertexLayouts::position3D();
    if (format == "position_uv") return vf::VertexLayouts::positionUV();
    if (format == "position_normal") return vf::VertexLayouts::positionNormal();
    if (format == "position_normal_uv") return vf::VertexLayouts::positionNormalUV();
    if (format == "position_color") return vf::VertexLayouts::positionColor();
    throw std::runtime_error("Unsupported vertex format: " + format);
}

/* --------------------------------------------------------------------- */
PYBIND11_MODULE(_vulkan_forge_native, m)
{
    m.doc() = "Vulkan-Forge: High-performance mesh rendering with Vulkan";

    py::register_exception<vf::VulkanForgeError>(m, "VulkanForgeError");

    /* ------------------------- VertexLayout ---------------------- */
    py::class_<vf::VertexAttribute>(m, "VertexAttribute")
        .def(py::init<uint32_t, VkFormat, uint32_t>(),
             py::arg("location"), py::arg("format"), py::arg("offset"))
        .def_readonly("location", &vf::VertexAttribute::location)
        .def_readonly("format", &vf::VertexAttribute::format)
        .def_readonly("offset", &vf::VertexAttribute::offset);

    py::class_<vf::VertexLayout>(m, "VertexLayout")
        .def(py::init<uint32_t, VkVertexInputRate>(),
             py::arg("stride"), py::arg("input_rate") = VK_VERTEX_INPUT_RATE_VERTEX)
        .def_readonly("stride", &vf::VertexLayout::stride)
        .def("add_attribute", &vf::VertexLayout::addAttribute,
             py::arg("location"), py::arg("format"), py::arg("offset"))
        .def_property_readonly("attributes", [](const vf::VertexLayout& self) {
            return self.attributes;
        });

    /* ------------------------- NumpyBuffer ---------------------- */
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

    /* ------------------------- MeshHandle ---------------------- */
    py::class_<vf::MeshHandle, std::shared_ptr<vf::MeshHandle>>(m, "MeshHandle")
        .def("bind", &vf::MeshHandle::bind, py::arg("command_buffer"))
        .def("draw", &vf::MeshHandle::draw,
             py::arg("command_buffer"), py::arg("instance_count") = 1)
        .def("destroy", &vf::MeshHandle::destroy)
        .def_property_readonly("name", &vf::MeshHandle::getName)
        .def_property_readonly("vertex_count", &vf::MeshHandle::getVertexCount)
        .def_property_readonly("triangle_count", &vf::MeshHandle::getTriangleCount)
        .def_property_readonly("vertex_layout", &vf::MeshHandle::getVertexLayout)
        .def_property_readonly("is_valid", &vf::MeshHandle::isValid)
        .def("get_info", &vf::MeshHandle::getInfo)
        .def("__str__", &vf::MeshHandle::getInfo)
        .def("__repr__", [](const vf::MeshHandle& self) {
            return "<MeshHandle '" + self.getName() + "'>";
        });

    /* ------------------------- MeshLoader ---------------------- */
    py::class_<vf::MeshLoader>(m, "MeshLoader")
        .def(py::init<>())
        .def("initialize",
             [](vf::MeshLoader& self, std::uintptr_t device, std::uintptr_t allocator,
                std::uintptr_t command_pool, std::uintptr_t queue) {
                 VkResult result = self.initialize(
                     reinterpret_cast<VkDevice>(device),
                     reinterpret_cast<VmaAllocator>(allocator),
                     reinterpret_cast<VkCommandPool>(command_pool),
                     reinterpret_cast<VkQueue>(queue)
                 );
                 if (result != VK_SUCCESS) {
                     throw std::runtime_error("Failed to initialize MeshLoader: " +
                                            std::to_string(result));
                 }
             },
             py::arg("device"), py::arg("allocator"),
             py::arg("command_pool"), py::arg("queue"))
        .def("upload_mesh",
             [](vf::MeshLoader& self, py::array_t<float> vertices,
                py::array_t<uint32_t> indices, const std::string& vertex_format,
                const std::string& name) {

                 py::buffer_info vertex_info = vertices.request();
                 py::buffer_info index_info = indices.request();

                 if (vertex_info.ndim != 2) {
                     throw std::runtime_error("Vertices must be 2D array (N, components)");
                 }
                 if (index_info.ndim != 1) {
                     throw std::runtime_error("Indices must be 1D array");
                 }

                 uint32_t vertex_count = static_cast<uint32_t>(vertex_info.shape[0]);
                 uint32_t vertex_stride = static_cast<uint32_t>(vertex_info.shape[1] * sizeof(float));
                 uint32_t index_count = static_cast<uint32_t>(index_info.shape[0]);

                 vf::VertexLayout layout = string_to_vertex_layout(vertex_format);

                 std::shared_ptr<vf::MeshHandle> handle;
                 VkResult result = self.uploadMesh(
                     vertex_info.ptr, vertex_count, vertex_stride,
                     index_info.ptr, index_count, VK_INDEX_TYPE_UINT32,
                     layout, name, handle
                 );

                 if (result != VK_SUCCESS) {
                     throw std::runtime_error("Failed to upload mesh: " +
                                            std::to_string(result));
                 }

                 return handle;
             },
             py::arg("vertices"), py::arg("indices"),
             py::arg("vertex_format"), py::arg("name"))
        .def("upload_mesh_auto",
             [](vf::MeshLoader& self, py::array_t<float> vertices,
                py::array_t<uint32_t> indices, const std::string& name) {

                 py::buffer_info vertex_info = vertices.request();
                 py::buffer_info index_info = indices.request();

                 if (vertex_info.ndim != 2) {
                     throw std::runtime_error("Vertices must be 2D array (N, components)");
                 }
                 if (index_info.ndim != 1) {
                     throw std::runtime_error("Indices must be 1D array");
                 }

                 uint32_t vertex_count = static_cast<uint32_t>(vertex_info.shape[0]);
                 uint32_t vertex_stride = static_cast<uint32_t>(vertex_info.shape[1] * sizeof(float));
                 uint32_t index_count = static_cast<uint32_t>(index_info.shape[0]);

                 std::shared_ptr<vf::MeshHandle> handle;
                 VkResult result = self.uploadMeshAuto(
                     vertex_info.ptr, vertex_count, vertex_stride,
                     index_info.ptr, index_count, VK_INDEX_TYPE_UINT32,
                     name, handle
                 );

                 if (result != VK_SUCCESS) {
                     throw std::runtime_error("Failed to upload mesh: " +
                                            std::to_string(result));
                 }

                 return handle;
             },
             py::arg("vertices"), py::arg("indices"), py::arg("name"))
        .def("create_primitive",
             [](vf::MeshLoader& self, const std::string& primitive_type,
                float size, uint32_t subdivisions, const std::string& name) {

                 std::shared_ptr<vf::MeshHandle> handle;
                 VkResult result = self.createPrimitive(
                     primitive_type, size, subdivisions, name, handle
                 );

                 if (result != VK_SUCCESS) {
                     throw std::runtime_error("Failed to create primitive: " +
                                            std::to_string(result));
                 }

                 return handle;
             },
             py::arg("primitive_type"), py::arg("size"),
             py::arg("subdivisions") = 16, py::arg("name") = "primitive")
        .def("remove_mesh", &vf::MeshLoader::removeMesh, py::arg("handle"))
        .def("get_stats", &vf::MeshLoader::getStats)
        .def("destroy", &vf::MeshLoader::destroy);

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

    /* ----------------------------- Utility Functions -------------------------- */
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

    /* Vertex layout factory functions */
    m.def("vertex_layout_position_3d", &vf::VertexLayouts::position3D);
    m.def("vertex_layout_position_uv", &vf::VertexLayouts::positionUV);
    m.def("vertex_layout_position_normal", &vf::VertexLayouts::positionNormal);
    m.def("vertex_layout_position_normal_uv", &vf::VertexLayouts::positionNormalUV);
    m.def("vertex_layout_position_color", &vf::VertexLayouts::positionColor);

    /* Buffer usage flags */
    m.attr("BUFFER_USAGE_VERTEX_BUFFER") = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    m.attr("BUFFER_USAGE_INDEX_BUFFER") = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    m.attr("BUFFER_USAGE_UNIFORM_BUFFER") = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    m.attr("BUFFER_USAGE_STORAGE_BUFFER") = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    m.attr("BUFFER_USAGE_TRANSFER_SRC") = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    m.attr("BUFFER_USAGE_TRANSFER_DST") = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    /* Vulkan format constants */
    m.attr("FORMAT_R32G32B32_SFLOAT") = VK_FORMAT_R32G32B32_SFLOAT;
    m.attr("FORMAT_R32G32_SFLOAT") = VK_FORMAT_R32G32_SFLOAT;
    m.attr("FORMAT_R32G32B32A32_SFLOAT") = VK_FORMAT_R32G32B32A32_SFLOAT;

    /* Index types */
    m.attr("INDEX_TYPE_UINT16") = VK_INDEX_TYPE_UINT16;
    m.attr("INDEX_TYPE_UINT32") = VK_INDEX_TYPE_UINT32;
}

// --------------------------- TERRAIN EXTENSIONS ---------------------------
// Vulkan-Forge Python Bindings with Terrain Support
// Enhanced to include TerrainRenderer, GeoTIFF loading, and tessellation controls



namespace py = pybind11;

// Terrain configuration bindings
void bind_terrain_config(py::module& m) {
    py::enum_<vf::TessellationMode>(m, "TessellationMode")
        .value("DISABLED", vf::TessellationMode::DISABLED)
        .value("UNIFORM", vf::TessellationMode::UNIFORM)
        .value("DISTANCE_BASED", vf::TessellationMode::DISTANCE_BASED)
        .value("SCREEN_SPACE", vf::TessellationMode::SCREEN_SPACE);

    py::enum_<vf::LODAlgorithm>(m, "LODAlgorithm")
        .value("DISTANCE", vf::LODAlgorithm::DISTANCE)
        .value("SCREEN_ERROR", vf::LODAlgorithm::SCREEN_ERROR)
        .value("FRUSTUM_SIZE", vf::LODAlgorithm::FRUSTUM_SIZE);

    py::class_<vf::TessellationConfig>(m, "TessellationConfig")
        .def(py::init<>())
        .def_readwrite("mode", &vf::TessellationConfig::mode)
        .def_readwrite("base_level", &vf::TessellationConfig::base_level)
        .def_readwrite("max_level", &vf::TessellationConfig::max_level)
        .def_readwrite("min_level", &vf::TessellationConfig::min_level)
        .def_readwrite("near_distance", &vf::TessellationConfig::near_distance)
        .def_readwrite("far_distance", &vf::TessellationConfig::far_distance)
        .def_readwrite("falloff_exponent", &vf::TessellationConfig::falloff_exponent)
        .def_readwrite("target_triangle_size", &vf::TessellationConfig::target_triangle_size)
        .def("get_tessellation_level", &vf::TessellationConfig::get_tessellation_level);

    py::class_<vf::LODConfig>(m, "LODConfig")
        .def(py::init<>())
        .def_readwrite("algorithm", &vf::LODConfig::algorithm)
        .def_readwrite("distances", &vf::LODConfig::distances)
        .def_readwrite("screen_error_threshold", &vf::LODConfig::screen_error_threshold)
        .def_readwrite("enable_morphing", &vf::LODConfig::enable_morphing)
        .def_readwrite("morph_distance", &vf::LODConfig::morph_distance);

    py::class_<vf::TerrainConfig>(m, "TerrainConfig")
        .def(py::init<>())
        .def_readwrite("tile_size", &vf::TerrainConfig::tile_size)
        .def_readwrite("height_scale", &vf::TerrainConfig::height_scale)
        .def_readwrite("max_render_distance", &vf::TerrainConfig::max_render_distance)
        .def_readwrite("tessellation", &vf::TerrainConfig::tessellation)
        .def_readwrite("lod", &vf::TerrainConfig::lod)
        .def_static("from_preset", &vf::TerrainConfig::from_preset)
        .def("validate", &vf::TerrainConfig::validate)
        .def("optimize_for_hardware", &vf::TerrainConfig::optimize_for_hardware);
}

// GeoTIFF and terrain data bindings
void bind_geotiff_loader(py::module& m) {
    py::class_<vf::GeographicBounds>(m, "GeographicBounds")
        .def(py::init<>())
        .def_readwrite("min_x", &vf::GeographicBounds::min_x)
        .def_readwrite("max_x", &vf::GeographicBounds::max_x)
        .def_readwrite("min_y", &vf::GeographicBounds::min_y)
        .def_readwrite("max_y", &vf::GeographicBounds::max_y)
        .def_readwrite("min_elevation", &vf::GeographicBounds::min_elevation)
        .def_readwrite("max_elevation", &vf::GeographicBounds::max_elevation);

    py::class_<vf::TerrainTile>(m, "TerrainTile")
        .def(py::init<>())
        .def_readwrite("tile_id", &vf::TerrainTile::tile_id)
        .def_readwrite("lod_level", &vf::TerrainTile::lod_level)
        .def_readwrite("bounds", &vf::TerrainTile::bounds)
        .def_readwrite("is_loaded", &vf::TerrainTile::is_loaded)
        .def("get_heightmap", &vf::TerrainTile::get_heightmap,
             py::return_value_policy::reference_internal);

    py::class_<vf::GeoTiffLoader>(m, "GeoTiffLoader")
        .def(py::init<>())
        .def("load", &vf::GeoTiffLoader::load)
        .def("get_bounds", &vf::GeoTiffLoader::get_bounds)
        .def("get_heightmap", &vf::GeoTiffLoader::get_heightmap,
             py::return_value_policy::reference_internal)
        .def("get_transform", &vf::GeoTiffLoader::get_transform)
        .def("get_projection", &vf::GeoTiffLoader::get_projection)
        .def("get_statistics", &vf::GeoTiffLoader::get_statistics);
}

// Tessellation pipeline bindings
void bind_tessellation_pipeline(py::module& m) {
    py::class_<vf::TessellationPipeline>(m, "TessellationPipeline")
        .def(py::init<vf::VulkanContext&, const vf::TessellationConfig&>())
        .def("update_config", &vf::TessellationPipeline::update_config)
        .def("bind", &vf::TessellationPipeline::bind)
        .def("set_tessellation_level", &vf::TessellationPipeline::set_tessellation_level)
        .def("get_pipeline", &vf::TessellationPipeline::get_pipeline,
             py::return_value_policy::reference_internal)
        .def("is_supported", &vf::TessellationPipeline::is_supported);

    py::class_<vf::TessellationUniformBuffer>(m, "TessellationUniformBuffer")
        .def(py::init<>())
        .def_readwrite("tessellation_level", &vf::TessellationUniformBuffer::tessellation_level)
        .def_readwrite("screen_size", &vf::TessellationUniformBuffer::screen_size)
        .def_readwrite("lod_bias", &vf::TessellationUniformBuffer::lod_bias)
        .def_readwrite("distance_scale", &vf::TessellationUniformBuffer::distance_scale);
}

// Enhanced terrain renderer bindings
void bind_terrain_renderer(py::module& m) {
    py::class_<vf::TerrainRenderer>(m, "TerrainRenderer")
        .def(py::init<vf::VulkanContext&, const vf::TerrainConfig&>())
        .def("load_geotiff", &vf::TerrainRenderer::load_geotiff)
        .def("load_heightmap", &vf::TerrainRenderer::load_heightmap)
        .def("update_camera", &vf::TerrainRenderer::update_camera)
        .def("render", &vf::TerrainRenderer::render)
        .def("get_bounds", &vf::TerrainRenderer::get_bounds,
             py::return_value_policy::reference_internal)
        .def("get_tiles", &vf::TerrainRenderer::get_tiles,
             py::return_value_policy::reference_internal)
        .def("get_performance_stats", &vf::TerrainRenderer::get_performance_stats)
        .def("set_config", &vf::TerrainRenderer::set_config)
        .def("get_config", &vf::TerrainRenderer::get_config,
             py::return_value_policy::reference_internal)
        .def("enable_wireframe", &vf::TerrainRenderer::enable_wireframe)
        .def("enable_lod_visualization", &vf::TerrainRenderer::enable_lod_visualization)
        .def("get_triangle_count", &vf::TerrainRenderer::get_triangle_count)
        .def("get_visible_tile_count", &vf::TerrainRenderer::get_visible_tile_count);

    py::class_<vf::TerrainStreamer>(m, "TerrainStreamer")
        .def(py::init<vf::TerrainRenderer&, uint32_t>())
        .def("update", &vf::TerrainStreamer::update)
        .def("set_max_loaded_tiles", &vf::TerrainStreamer::set_max_loaded_tiles)
        .def("get_loaded_tile_count", &vf::TerrainStreamer::get_loaded_tile_count)
        .def("get_loading_stats", &vf::TerrainStreamer::get_loading_stats);
}

// Enhanced heightfield scene bindings
void bind_enhanced_heightfield_scene(py::module& m) {
    py::class_<vf::HeightfieldScene>(m, "HeightfieldScene")
        .def(py::init<vf::VulkanContext&>())
        .def("load_heightfield", &vf::HeightfieldScene::load_heightfield)
        .def("load_geotiff", &vf::HeightfieldScene::load_geotiff)
        .def("set_height_scale", &vf::HeightfieldScene::set_height_scale)
        .def("get_height_scale", &vf::HeightfieldScene::get_height_scale)
        .def("set_tessellation_config", &vf::HeightfieldScene::set_tessellation_config)
        .def("get_tessellation_config", &vf::HeightfieldScene::get_tessellation_config,
             py::return_value_policy::reference_internal)
        .def("enable_tessellation", &vf::HeightfieldScene::enable_tessellation)
        .def("is_tessellation_enabled", &vf::HeightfieldScene::is_tessellation_enabled)
        .def("generate_normals", &vf::HeightfieldScene::generate_normals)
        .def("generate_tangents", &vf::HeightfieldScene::generate_tangents)
        .def("set_texture", &vf::HeightfieldScene::set_texture)
        .def("get_bounds", &vf::HeightfieldScene::get_bounds,
             py::return_value_policy::reference_internal)
        .def("get_vertex_count", &vf::HeightfieldScene::get_vertex_count)
        .def("get_triangle_count", &vf::HeightfieldScene::get_triangle_count)
        .def("update", &vf::HeightfieldScene::update)
        .def("render", &vf::HeightfieldScene::render);
}

// Enhanced renderer bindings
void bind_enhanced_renderer(py::module& m) {
    py::class_<vf::Renderer>(m, "Renderer")
        .def(py::init<vf::VulkanContext&>())
        .def("add_scene", &vf::Renderer::add_scene)
        .def("remove_scene", &vf::Renderer::remove_scene)
        .def("render", &vf::Renderer::render)
        .def("set_camera", &vf::Renderer::set_camera)
        .def("get_camera", &vf::Renderer::get_camera,
             py::return_value_policy::reference_internal)
        .def("enable_tessellation", &vf::Renderer::enable_tessellation)
        .def("is_tessellation_supported", &vf::Renderer::is_tessellation_supported)
        .def("set_tessellation_config", &vf::Renderer::set_tessellation_config)
        .def("enable_gpu_driven_rendering", &vf::Renderer::enable_gpu_driven_rendering)
        .def("is_gpu_driven_rendering_enabled", &vf::Renderer::is_gpu_driven_rendering_enabled)
        .def("set_culling_enabled", &vf::Renderer::set_culling_enabled)
        .def("is_culling_enabled", &vf::Renderer::is_culling_enabled)
        .def("get_render_stats", &vf::Renderer::get_render_stats)
        .def("screenshot", &vf::Renderer::screenshot)
        .def("resize", &vf::Renderer::resize);

    py::class_<vf::RenderStats>(m, "RenderStats")
        .def(py::init<>())
        .def_readwrite("frame_time_ms", &vf::RenderStats::frame_time_ms)
        .def_readwrite("triangles_rendered", &vf::RenderStats::triangles_rendered)
        .def_readwrite("draw_calls", &vf::RenderStats::draw_calls)
        .def_readwrite("vertices_processed", &vf::RenderStats::vertices_processed)
        .def_readwrite("tessellation_patches", &vf::RenderStats::tessellation_patches)
        .def_readwrite("culled_objects", &vf::RenderStats::culled_objects);
}

// Camera enhancements for terrain
void bind_enhanced_camera(py::module& m) {
    py::class_<vf::Camera>(m, "Camera")
        .def(py::init<>())
        .def(py::init<const Eigen::Vector3f&, const Eigen::Vector3f&>())
        .def("set_position", &vf::Camera::set_position)
        .def("get_position", &vf::Camera::get_position)
        .def("set_target", &vf::Camera::set_target)
        .def("get_target", &vf::Camera::get_target)
        .def("set_up", &vf::Camera::set_up)
        .def("get_up", &vf::Camera::get_up)
        .def("set_fov", &vf::Camera::set_fov)
        .def("get_fov", &vf::Camera::get_fov)
        .def("set_aspect_ratio", &vf::Camera::set_aspect_ratio)
        .def("get_aspect_ratio", &vf::Camera::get_aspect_ratio)
        .def("set_near_far", &vf::Camera::set_near_far)
        .def("get_near", &vf::Camera::get_near)
        .def("get_far", &vf::Camera::get_far)
        .def("get_view_matrix", &vf::Camera::get_view_matrix)
        .def("get_projection_matrix", &vf::Camera::get_projection_matrix)
        .def("move_forward", &vf::Camera::move_forward)
        .def("move_right", &vf::Camera::move_right)
        .def("move_up", &vf::Camera::move_up)
        .def("rotate", &vf::Camera::rotate)
        .def("look_at", &vf::Camera::look_at)
        .def("orbit", &vf::Camera::orbit)
        .def("zoom", &vf::Camera::zoom)
        .def("get_frustum_planes", &vf::Camera::get_frustum_planes)
        .def("is_point_in_frustum", &vf::Camera::is_point_in_frustum)
        .def("is_sphere_in_frustum", &vf::Camera::is_sphere_in_frustum);
}

// Main module binding
PYBIND11_MODULE(vulkan_forge_core, m) {
    m.doc() = "Vulkan-Forge: High-performance terrain rendering with Vulkan";

    // Bind terrain configuration classes
    bind_terrain_config(m);

    // Bind GeoTIFF loader and terrain data structures
    bind_geotiff_loader(m);

    // Bind tessellation pipeline
    bind_tessellation_pipeline(m);

    // Bind terrain renderer
    bind_terrain_renderer(m);

    // Bind enhanced heightfield scene
    bind_enhanced_heightfield_scene(m);

    // Bind enhanced renderer
    bind_enhanced_renderer(m);

    // Bind enhanced camera
    bind_enhanced_camera(m);

    // Utility functions
    m.def("is_tessellation_supported", &vf::is_tessellation_supported,
          "Check if tessellation shaders are supported on the current GPU");

    m.def("get_gpu_info", &vf::get_gpu_info,
          "Get GPU information including memory and feature support");

    m.def("optimize_terrain_config_for_gpu", &vf::optimize_terrain_config_for_gpu,
          "Automatically optimize terrain configuration for current GPU");

    m.def("generate_synthetic_heightmap", &vf::generate_synthetic_heightmap,
          "Generate synthetic heightmap for testing and demonstration");

    // Constants
    m.attr("MAX_TESSELLATION_LEVEL") = vf::MAX_TESSELLATION_LEVEL;
    m.attr("MIN_TESSELLATION_LEVEL") = vf::MIN_TESSELLATION_LEVEL;
    m.attr("DEFAULT_TILE_SIZE") = vf::DEFAULT_TILE_SIZE;
    m.attr("MAX_LOD_LEVELS") = vf::MAX_LOD_LEVELS;

    // Version information
    m.attr("__version__") = "0.2.0";
    m.attr("VULKAN_API_VERSION") = VK_API_VERSION_1_3;
}
// ============================================================================