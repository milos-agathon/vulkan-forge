# CMakeLists.txt - Complete build configuration for Vulkan-Forge mesh renderer
cmake_minimum_required(VERSION 3.24)
project(vulkan_forge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set policy for compatibility with newer CMake versions
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

# Find required packages
find_package(pybind11 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

# Fetch VulkanMemoryAllocator header
include(FetchContent)
FetchContent_Declare(
    vma_headers
    URL https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/heads/master.zip
)
FetchContent_MakeAvailable(vma_headers)

# Find or download GLM with better error handling
find_package(glm 1.0 CONFIG QUIET)

if (NOT glm_FOUND)
    include(FetchContent)
    message(STATUS "glm ≥ 1.0 not found — downloading header-only copy …")

    # Pull GLM 1.0.1 (latest stable) — header-only, no build step needed
    FetchContent_Declare(
        glm_headers
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
    )
    FetchContent_Populate(glm_headers)

    add_library(glm_interface INTERFACE)
    target_include_directories(glm_interface INTERFACE
        "$<BUILD_INTERFACE:${glm_headers_SOURCE_DIR}>")
    add_library(glm::glm ALIAS glm_interface)
endif()

# Shader compilation setup
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

# Find glslc compiler
find_program(GLSLC_EXECUTABLE NAMES glslc
    HINTS 
        $ENV{VULKAN_SDK}/bin
        $ENV{VULKAN_SDK}/Bin
    PATHS
        /usr/bin
        /usr/local/bin
)

if(GLSLC_EXECUTABLE)
    message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")
else()
    message(WARNING "glslc not found. Shaders will use embedded fallbacks.")
endif()

# Create shader directory
file(MAKE_DIRECTORY "${SHADER_DIR}")

# Basic mesh vertex shader
file(WRITE "${SHADER_DIR}/mesh.vert" "
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragPos;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec3 lightDir;
    float time;
} pc;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    gl_Position = pc.projection * pc.view * worldPos;
    
    fragNormal = mat3(transpose(inverse(pc.model))) * inNormal;
    fragTexCoord = inTexCoord;
    fragPos = worldPos.xyz;
}
")

# Basic mesh fragment shader
file(WRITE "${SHADER_DIR}/mesh.frag" "
#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec3 lightDir;
    float time;
} pc;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(-pc.lightDir);
    
    // Simple Phong shading
    float ambient = 0.2;
    float diffuse = max(dot(normal, lightDir), 0.0) * 0.8;
    
    // Basic material color (can be replaced with textures later)
    vec3 baseColor = vec3(0.7, 0.6, 0.8);
    
    vec3 color = baseColor * (ambient + diffuse);
    outColor = vec4(color, 1.0);
}
")

# Wireframe fragment shader
file(WRITE "${SHADER_DIR}/wireframe.frag" "
#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple wireframe color
    outColor = vec4(0.0, 1.0, 0.0, 1.0);  // Green wireframe
}
")

# Compile shaders if glslc is available
set(COMPILED_SHADERS)
if(GLSLC_EXECUTABLE)
    set(SHADER_SOURCES
        "${SHADER_DIR}/mesh.vert"
        "${SHADER_DIR}/mesh.frag"
        "${SHADER_DIR}/wireframe.frag"
    )
    
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        set(SHADER_OUTPUT "${GENERATED_DIR}/${SHADER_NAME}.spv")
        
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE} -o ${SHADER_OUTPUT}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling ${SHADER_NAME}"
        )
        
        list(APPEND COMPILED_SHADERS ${SHADER_OUTPUT})
    endforeach()
    
    # Create custom target for shader compilation
    add_custom_target(compile_shaders ALL DEPENDS ${COMPILED_SHADERS})
endif()

# Generate embedded shader headers
file(WRITE "${GENERATED_DIR}/embedded_shaders.hpp" "
#pragma once
#include <vector>
#include <cstdint>

namespace vf {
namespace EmbeddedShaders {
    // Placeholder SPIR-V data - replace with actual compiled shaders
    extern const std::vector<uint32_t> mesh_vert_spv;
    extern const std::vector<uint32_t> mesh_frag_spv;
    extern const std::vector<uint32_t> wireframe_frag_spv;
}
}
")

file(WRITE "${GENERATED_DIR}/embedded_shaders.cpp" "
#include \"embedded_shaders.hpp\"

namespace vf {
namespace EmbeddedShaders {
    // Minimal valid SPIR-V headers - replace with actual compiled shaders
    const std::vector<uint32_t> mesh_vert_spv = {
        0x07230203, 0x00010000, 0x00080001, 0x00000006, 0x00000000
    };
    
    const std::vector<uint32_t> mesh_frag_spv = {
        0x07230203, 0x00010000, 0x00080001, 0x00000006, 0x00000000
    };
    
    const std::vector<uint32_t> wireframe_frag_spv = {
        0x07230203, 0x00010000, 0x00080001, 0x00000006, 0x00000000
    };
}
}
")

# Create the Python extension module
pybind11_add_module(_vulkan_forge_native
    # Core source files
    src/bindings.cpp
    src/vk_common.cpp
    src/heightfield_scene.cpp
    src/renderer.cpp
    src/vma_util.cpp
    
    # New mesh pipeline source files
    src/vertex_buffer.cpp
    src/mesh_loader.cpp
    src/mesh_pipeline.cpp
    
    # Generated embedded shaders
    ${GENERATED_DIR}/embedded_shaders.cpp
)

# Set include directories
target_include_directories(_vulkan_forge_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GENERATED_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${vma_headers_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(_vulkan_forge_native PRIVATE
    ${Vulkan_LIBRARIES}
    glm::glm
)

# Set compile definitions
target_compile_definitions(_vulkan_forge_native PRIVATE
    VERSION_INFO=${PROJECT_VERSION}
    VMA_IMPLEMENTATION
)

# Platform-specific settings
if(MSVC)
    target_compile_options(_vulkan_forge_native PRIVATE /W4 /bigobj)
    # Disable specific MSVC warnings for VMA
    target_compile_options(_vulkan_forge_native PRIVATE /wd4127 /wd4100)
else()
    target_compile_options(_vulkan_forge_native PRIVATE -Wall -Wextra -Wpedantic)
    # Disable specific warnings for VMA
    target_compile_options(_vulkan_forge_native PRIVATE -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# Add shader compilation dependency
if(GLSLC_EXECUTABLE)
    add_dependencies(_vulkan_forge_native compile_shaders)
endif()

# Installation rules
install(TARGETS _vulkan_forge_native
    LIBRARY DESTINATION lib/vulkan_forge
    ARCHIVE DESTINATION lib/vulkan_forge
    RUNTIME DESTINATION lib/vulkan_forge
)

# Install shader files
install(DIRECTORY ${SHADER_DIR}/
    DESTINATION share/vulkan_forge/shaders
    FILES_MATCHING PATTERN "*.vert" PATTERN "*.frag" PATTERN "*.spv"
)

# Install Python files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/vulkan_forge
    DESTINATION .
    FILES_MATCHING PATTERN "*.py"
)

# Create __init__.py if it doesn't exist
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/vulkan_forge/__init__.py")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/python/vulkan_forge/__init__.py" "")
endif()

# Print configuration summary
message(STATUS "==== Vulkan-Forge Configuration ====")
message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan library: ${Vulkan_LIBRARIES}")
message(STATUS "GLM found: ${glm_FOUND}")
message(STATUS "GLSLC found: ${GLSLC_EXECUTABLE}")
message(STATUS "VMA headers: ${vma_headers_SOURCE_DIR}")
message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "======================================")