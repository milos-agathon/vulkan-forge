# cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

# ----- project -----
project(vulkan_forge_native LANGUAGES CXX)

# Let pybind11 provide include dirs & helper macros
find_package(pybind11 CONFIG REQUIRED)

# Use pybind11_add_module instead of add_library
pybind11_add_module(vulkan_forge_native MODULE
    src/vk_common.cpp
    src/renderer.cpp
    src/heightfield_scene.cpp
    src/bindings.cpp
)

# C++17 (or later)
target_compile_features(vulkan_forge_native PUBLIC cxx_std_17)

# Link against Vulkan
find_package(Vulkan REQUIRED)
target_link_libraries(vulkan_forge_native PRIVATE Vulkan::Vulkan)

# Set the output name
set_target_properties(vulkan_forge_native PROPERTIES
    OUTPUT_NAME "_native"
)

# ----- install rule -----
include(GNUInstallDirs)
install(
    TARGETS vulkan_forge_native
    COMPONENT vulkan_forge_native
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/vulkan_forge
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/vulkan_forge
)