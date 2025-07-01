cmake_minimum_required(VERSION 3.24)
project(vulkan_forge_native LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Vulkan      REQUIRED)
find_package(glm         CONFIG REQUIRED)

add_library(vulkan_forge_native MODULE
    ../src/bindings.cpp
    ../src/vk_common.cpp
    ../src/heightfield_scene.cpp
    ../src/renderer.cpp
)

target_include_directories(vulkan_forge_native PRIVATE include)
target_link_libraries(vulkan_forge_native PRIVATE
    pybind11::module Vulkan::Vulkan glm::glm
)
# on Linux/macOS add -fvisibility=hidden
set_target_properties(vulkan_forge_native PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
