# cpp/CMakeLists.txt ───────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.24)
project(vulkan_forge LANGUAGES CXX)

# ───────────────────────── core deps ──────────────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)
find_package(Vulkan      REQUIRED)          # brings GLSLC

# ─────────────────────── glm (headers-only fallback) ──────────────────────────
find_package(glm CONFIG QUIET)

if (NOT glm_FOUND)
    include(FetchContent)
    message(STATUS "glm not found — falling back to header-only download …")

    # 1.  Get the sources but *do not* call glm’s own CMakeLists.
    FetchContent_Declare(
        glm_headers
        URL https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.zip
    )
    FetchContent_Populate(glm_headers)   # produces glm_headers_SOURCE_DIR

    # 2.  Lightweight interface target that only adds the include dir.
    add_library(glm_interface INTERFACE)
    target_include_directories(glm_interface INTERFACE
        "${glm_headers_SOURCE_DIR}"
    )

    # 3.  Canonical alias so we can `target_link_libraries(... glm::glm)`.
    add_library(glm::glm ALIAS glm_interface)
endif()

# ───────────────────────── directories ────────────────────────────────────────
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

# ─────────────── GLSL → SPIR-V → C-array headers ─────────────────────────────
set(GLSLC "${Vulkan_GLSLC_EXECUTABLE}")

# vertex shader
set(VERT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_passthrough.vert.glsl")
add_custom_command(
    OUTPUT  "${GENERATED_DIR}/vtx.inc"
    COMMAND "${GLSLC}" -fshader-stage=vert "${VERT_SRC}" -o "${GENERATED_DIR}/vtx.spv"
    COMMAND "${CMAKE_COMMAND}"
            -DINPUT="${GENERATED_DIR}/vtx.spv"
            -DOUTPUT="${GENERATED_DIR}/vtx.inc"
            -DVAR=VTX_SPV
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpirvToHeader.cmake"
    DEPENDS "${VERT_SRC}"
    COMMENT "Compiling vertex shader → header"
)

# fragment shader
set(FRAG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader_color.frag.glsl")
add_custom_command(
    OUTPUT  "${GENERATED_DIR}/frg.inc"
    COMMAND "${GLSLC}" -fshader-stage=frag "${FRAG_SRC}" -o "${GENERATED_DIR}/frg.spv"
    COMMAND "${CMAKE_COMMAND}"
            -DINPUT="${GENERATED_DIR}/frg.spv"
            -DOUTPUT="${GENERATED_DIR}/frg.inc"
            -DVAR=FRG_SPV
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpirvToHeader.cmake"
    DEPENDS "${FRAG_SRC}"
    COMMENT "Compiling fragment shader → header"
)

set(SHADER_INCS "${GENERATED_DIR}/vtx.inc" "${GENERATED_DIR}/frg.inc")

# ───────────── native extension module (pybind11) ────────────────────────────
# *Target is called **vulkan_forge** — keep that name consistent everywhere*
pybind11_add_module(vulkan_forge MODULE
    src/vk_common.cpp
    src/heightfield_scene.cpp
    src/renderer.cpp
    src/bindings.cpp
    ${SHADER_INCS}
)

target_include_directories(vulkan_forge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GENERATED_DIR}"
)

target_compile_features(vulkan_forge PUBLIC cxx_std_17)

set_target_properties(vulkan_forge PROPERTIES
    OUTPUT_NAME "vulkan_forge"  # final .pyd / .so name
    PREFIX ""                   # no “lib” prefix on *nix
)

target_link_libraries(vulkan_forge PRIVATE
    Vulkan::Vulkan
    glm::glm                          # either system or header-only alias
)

# ───────────────────────── install rule ───────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS vulkan_forge
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/vulkan_forge
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/vulkan_forge)
