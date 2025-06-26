# cpp/CMakeLists.txt ───────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.24)
project(vulkan_forge_native LANGUAGES CXX)

# ───────────────────────── core deps ──────────────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)
find_package(Vulkan      REQUIRED)          # brings GLSLC

# ─────────────────────── glm (headers-only) ───────────────────────────────────
find_package(glm CONFIG QUIET)

if (NOT glm_FOUND)
    include(FetchContent)
    message(STATUS "glm not found – downloading headers only …")

    # 1. Download the archive but DO NOT call add_subdirectory / configure GLM.
    FetchContent_Declare(
        glm_headers
        URL https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.zip
        # No SHA check – GitHub archives change their hash from time to time.
    )
    FetchContent_Populate(glm_headers)  # produces glm_headers_SOURCE_DIR

    # 2. Provide a lightweight INTERFACE target that just adds the include dir.
    add_library(glm_interface INTERFACE)
    target_include_directories(glm_interface INTERFACE
        "${glm_headers_SOURCE_DIR}"
    )

    # 3. Alias it to the canonical name so downstream code can `link` normally.
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
pybind11_add_module(vulkan_forge_native MODULE
    src/vk_common.cpp
    src/heightfield_scene.cpp
    src/renderer.cpp
    src/bindings.cpp
    ${SHADER_INCS}
)

target_include_directories(vulkan_forge_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GENERATED_DIR}"
)

target_compile_features(vulkan_forge_native PUBLIC cxx_std_17)

target_link_libraries(vulkan_forge_native PRIVATE
    Vulkan::Vulkan
    glm::glm                      # always available (system or header-only)
)

# ───────────────────────── install rule ───────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS vulkan_forge_native
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/vulkan_forge
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/vulkan_forge)
