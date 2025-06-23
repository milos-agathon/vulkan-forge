cmake_minimum_required(VERSION 3.14)
project(vulkan_renderer_py)

find_package(pybind11 REQUIRED)
add_library(vulkan_renderer_py MODULE bindings.cpp)
target_link_libraries(vulkan_renderer_py PRIVATE vulkan_renderer pybind11::module)
set_target_properties(vulkan_renderer_py PROPERTIES
  PREFIX ""
  OUTPUT_NAME "vulkan_renderer"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
