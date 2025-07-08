# =============================================================================
# CMakeLists.txt – unified build for Vulkan-Forge core + terrain renderer
#  • Base: cmake_terrain1  (cross-platform wheel + shader pipeline)
#  • Add-ons: cmake_terrain2 (terrain options, extra libs, profiling, tests)
# =============================================================================

cmake_minimum_required(VERSION 3.24)

# -----------------------------------------------------------------------------
# Project setup
# -----------------------------------------------------------------------------
project(vulkan_forge
        VERSION 0.2.0        # ← bumped from 0.1.0
        DESCRIPTION "High-performance GPU & terrain renderer using Vulkan"
        LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Global build settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)          # terrain renderer needs C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS   OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release MinSizeRel RelWithDebInfo)
endif()

# -----------------------------------------------------------------------------
# Feature / option matrix  (original + new)
# -----------------------------------------------------------------------------
option(VF_ENABLE_VALIDATION_LAYERS "Enable Vulkan validation layers"       OFF)
option(VF_ENABLE_GPU_DEBUG         "Enable GPU debugging features"        OFF)
option(VF_STATIC_VULKAN            "Link Vulkan statically"               OFF)
option(VF_ENABLE_SIMD              "Enable SIMD optimisations"            ON)
option(VF_ENABLE_LTO               "Enable Link-Time Optimisation"        ON)
option(VF_STATIC_LINK              "Static-link MSVC runtime"             OFF)
option(VF_USE_MOLTEN_VK            "Use MoltenVK on macOS"                ON)
option(VF_USE_SYSTEM_VULKAN        "Use system Vulkan"                    ON)
option(VF_ENABLE_MARCH_NATIVE      "Enable -march=native"                 OFF)

#  New terrain features
option(VF_BUILD_TESTS        "Build unit tests"                ON)
option(VF_BUILD_EXAMPLES     "Build example apps"              ON)
option(VF_ENABLE_TESSELLATION "Enable tessellation shaders"    ON)
option(VF_ENABLE_RAY_TRACING  "Enable ray-tracing paths"       OFF)
option(VF_ENABLE_PROFILING    "Enable built-in GPU/CPU timing" ON)

# -----------------------------------------------------------------------------
# Platform detection (kept from terrain1)
# -----------------------------------------------------------------------------
if(WIN32)
    set(VF_PLATFORM "Windows")
elseif(APPLE)
    set(VF_PLATFORM "macOS")
elseif(UNIX)
    set(VF_PLATFORM "Linux")
else()
    set(VF_PLATFORM "Unknown")
endif()

message(STATUS "==== Vulkan-Forge Build ====")
message(STATUS "Platform         : ${VF_PLATFORM}")
message(STATUS "Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard     : C++${CMAKE_CXX_STANDARD}")

# -----------------------------------------------------------------------------
# Compiler-specific flags (kept, plus small updates)
# -----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /bigobj /MP /permissive- /wd4127 /wd4100)
    if(VF_STATIC_LINK)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    add_link_options(/LARGEADDRESSAWARE)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter
                        -Wno-missing-field-initializers)
    if(VF_ENABLE_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
    # SIMD
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND VF_ENABLE_SIMD)
        add_compile_options(-msse4.2 -mavx)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$" AND VF_ENABLE_SIMD)
        add_compile_options(-march=armv8-a+simd)
    endif()
endif()

# -----------------------------------------------------------------------------
# Required packages – Python + pybind11 (unchanged)
# -----------------------------------------------------------------------------
find_package(Python          3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11        2.11.0 REQUIRED CONFIG)

# -----------------------------------------------------------------------------
# Required packages – Vulkan + friends
# -----------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# GLFW / Eigen / GDAL / optional PROJ & GEOS (from terrain2)
find_package(glfw3   REQUIRED)
find_package(glm     QUIET CONFIG)              # terrain1 already handled glm fetch
find_package(Eigen3  REQUIRED)
find_package(GDAL    QUIET)

# Optional PROJ & GEOS via pkg-config
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PROJ proj>=7.0)
    pkg_check_modules(GEOS geos>=3.8)
endif()

# -----------------------------------------------------------------------------
# VMA fetch (original)
# -----------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    vma_headers
    URL https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/heads/master.zip
)
FetchContent_MakeAvailable(vma_headers)

# -----------------------------------------------------------------------------
# Shader compilation (original function + extended terrain shader list)
# -----------------------------------------------------------------------------
set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SHADER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/shaders")
file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}" "${SHADER_INCLUDE_DIR}")

# Helper to compile a single shader (uses improved SpirvToHeader from terrain1)
function(compile_shader GLSL_SOURCE)
    get_filename_component(NAME ${GLSL_SOURCE} NAME)
    set(SPIRV "${SHADER_BINARY_DIR}/${NAME}.spv")
    set(HEADER "${SHADER_INCLUDE_DIR}/${NAME}.hpp")

    if(GLSLC_EXECUTABLE AND EXISTS "${GLSL_SOURCE}")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND "${GLSLC_EXECUTABLE}" -O -I "${SHADER_SOURCE_DIR}"
                    "${GLSL_SOURCE}" -o "${SPIRV}"
            DEPENDS "${GLSL_SOURCE}"
            COMMENT "Compiling ${NAME}"
        )
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND ${CMAKE_COMMAND}
                -DSPIRV_FILE="${SPIRV}"
                -HEADER_FILE="${HEADER}"
                -ARRAY_NAME="${NAME}_spirv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cmake/SpirvToHeader.cmake"
            DEPENDS "${SPIRV}"
            COMMENT "Embedding ${NAME} to C++ header"
        )
        list(APPEND EMBEDDED_SHADER_HEADERS "${HEADER}")
        set(EMBEDDED_SHADER_HEADERS "${EMBEDDED_SHADER_HEADERS}"
            PARENT_SCOPE)
    endif()
endfunction()

# Aggregate shader lists (terrain1 + terrain2)
set(BASE_SHADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/height_field.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/height_field.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/point_cloud.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders/point_cloud.frag")

set(TERRAIN_SHADERS
    "${SHADER_SOURCE_DIR}/terrain.vert"
    "${SHADER_SOURCE_DIR}/terrain.tesc"
    "${SHADER_SOURCE_DIR}/terrain.tese"
    "${SHADER_SOURCE_DIR}/terrain.frag"
    "${SHADER_SOURCE_DIR}/heightfield.vert"
    "${SHADER_SOURCE_DIR}/heightfield.frag"
    "${SHADER_SOURCE_DIR}/culling.comp"
    "${SHADER_SOURCE_DIR}/lod.comp")

set(ALL_SHADERS ${BASE_SHADERS} ${TERRAIN_SHADERS})
foreach(S ${ALL_SHADERS})
    compile_shader(${S})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${EMBEDDED_SHADER_HEADERS})

# -----------------------------------------------------------------------------
# Source lists
# -----------------------------------------------------------------------------
# Core/scene/renderer files from original build
set(VF_BASE_SOURCES
    cpp/src/bindings.cpp
    cpp/src/vk_common.cpp
    cpp/src/heightfield_scene.cpp
    cpp/src/renderer.cpp
    cpp/src/mesh_loader.cpp)

# New terrain engine + utilities (from cmake_terrain2)
set(VF_TERRAIN_SOURCES
    src/vulkan_context.cpp
    src/renderer.cpp            # terrain renderer variant
    src/camera.cpp
    src/scene.cpp
    src/mesh.cpp
    src/texture.cpp
    src/shader.cpp
    src/buffer.cpp
    src/pipeline.cpp
    src/command_buffer.cpp
    src/descriptor_set.cpp
    src/framebuffer.cpp
    src/render_pass.cpp
    src/utils.cpp
    # Terrain-specific
    src/terrain_renderer.cpp
    src/geotiff_loader.cpp
    src/tessellation_pipeline.cpp
    src/terrain_tile.cpp
    src/terrain_streaming.cpp
    src/terrain_culling.cpp
    src/terrain_lod.cpp
    src/heightfield_scene.cpp      # enhanced version
    src/terrain_materials.cpp
    src/gpu_driven_terrain.cpp)

set(VF_ALL_SOURCES ${VF_BASE_SOURCES} ${VF_TERRAIN_SOURCES})

# -----------------------------------------------------------------------------
# Build static library (terrain core) then Python module
# -----------------------------------------------------------------------------
add_library(vulkan_forge_core STATIC ${VF_ALL_SOURCES})
target_include_directories(vulkan_forge_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    ${SHADER_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${vma_headers_SOURCE_DIR}/include
    $<TARGET_PROPERTY:pybind11::module,INTERFACE_INCLUDE_DIRECTORIES>)

# Link third-party libs
target_link_libraries(vulkan_forge_core PUBLIC
    ${Vulkan_LIBRARIES}
    glfw
    Eigen3::Eigen
    glm::glm
    Threads::Threads             # on Linux
    ${CMAKE_DL_LIBS})
if(GDAL_FOUND)
    target_link_libraries(vulkan_forge_core PUBLIC ${GDAL_LIBRARIES})
    target_include_directories(vulkan_forge_core PUBLIC ${GDAL_INCLUDE_DIRS})
endif()
if(PROJ_FOUND)
    target_link_libraries(vulkan_forge_core PUBLIC ${PROJ_LIBRARIES})
    target_include_directories(vulkan_forge_core PUBLIC ${PROJ_INCLUDE_DIRS})
endif()
if(GEOS_FOUND)
    target_link_libraries(vulkan_forge_core PUBLIC ${GEOS_LIBRARIES})
    target_include_directories(vulkan_forge_core PUBLIC ${GEOS_INCLUDE_DIRS})
endif()

# Platform-specific links remain unchanged (Windows/macOS/Linux logic)

# Existing Python extension now links to core
pybind11_add_module(_vulkan_forge_native ${VF_BASE_SOURCES})  # keep original name
target_link_libraries(_vulkan_forge_native PRIVATE vulkan_forge_core)

add_dependencies(_vulkan_forge_native compile_shaders)

# -----------------------------------------------------------------------------
# Compile definitions
# -----------------------------------------------------------------------------
target_compile_definitions(vulkan_forge_core PUBLIC
    $<$<BOOL:${VF_ENABLE_VALIDATION_LAYERS}>:VF_ENABLE_VALIDATION_LAYERS>
    $<$<BOOL:${VF_ENABLE_TESSELLATION}>:VF_ENABLE_TESSELLATION>
    $<$<BOOL:${VF_ENABLE_RAY_TRACING}>:VF_ENABLE_RAY_TRACING>
    $<$<BOOL:${VF_ENABLE_PROFILING}>:VF_ENABLE_PROFILING>)

target_compile_definitions(_vulkan_forge_native PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
    VMA_IMPLEMENTATION
    VK_NO_PROTOTYPES=1)

# -----------------------------------------------------------------------------
# LTO check (kept)
# -----------------------------------------------------------------------------
if(VF_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_ok OUTPUT ipo_err)
    if(ipo_ok)
        set_property(TARGET vulkan_forge_core _vulkan_forge_native
                     PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO/LTO not supported: ${ipo_err}")
    endif()
endif()

# -----------------------------------------------------------------------------
# Tests & examples (from terrain2)
# -----------------------------------------------------------------------------
if(VF_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_subdirectory(tests)
    else()
        message(STATUS "GTest not found – tests skipped")
    endif()
endif()

if(VF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Docs (terrain2 Doxygen)
# -----------------------------------------------------------------------------
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME        "Vulkan-Forge")
    set(DOXYGEN_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/docs)
    doxygen_add_docs(docs ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# -----------------------------------------------------------------------------
# Install rules (original + static lib)
# -----------------------------------------------------------------------------
install(TARGETS vulkan_forge_core _vulkan_forge_native
        LIBRARY DESTINATION lib/vulkan_forge
        ARCHIVE DESTINATION lib/vulkan_forge
        RUNTIME DESTINATION lib/vulkan_forge)

install(DIRECTORY include/vf  DESTINATION include)
install(DIRECTORY ${SHADER_BINARY_DIR} DESTINATION share/vulkan_forge
        FILES_MATCHING PATTERN "*.spv")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "------------------------------------------")
message(STATUS "GLM            : ${glm_FOUND}")
message(STATUS "GDAL           : ${GDAL_FOUND}")
message(STATUS "PROJ           : ${PROJ_FOUND}")
message(STATUS "GEOS           : ${GEOS_FOUND}")
message(STATUS "Validation     : ${VF_ENABLE_VALIDATION_LAYERS}")
message(STATUS "Tessellation   : ${VF_ENABLE_TESSELLATION}")
message(STATUS "Ray tracing    : ${VF_ENABLE_RAY_TRACING}")
message(STATUS "Profiling      : ${VF_ENABLE_PROFILING}")
message(STATUS "Tests enabled  : ${VF_BUILD_TESTS}")
message(STATUS "Examples enabled: ${VF_BUILD_EXAMPLES}")
message(STATUS "------------------------------------------")
