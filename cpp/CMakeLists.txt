# CMakeLists.txt - Complete build configuration for Vulkan height field renderer
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
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip        # ← new
    )
    FetchContent_Populate(glm_headers)

    add_library(glm_interface INTERFACE)
    target_include_directories(glm_interface INTERFACE
        "$<BUILD_INTERFACE:${glm_headers_SOURCE_DIR}>")
    add_library(glm::glm ALIAS glm_interface)
endif()

# Generate shader includes (simplified - in production use glslangValidator)
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

# Vertex shader GLSL
file(WRITE "${SHADER_DIR}/height_field.vert" "
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec4 fragColor;
layout(location = 2) out vec3 fragPos;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec3 lightDir;
    float time;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    fragNormal = inNormal;
    fragColor = inColor;
    fragPos = inPosition;
}
")

# Fragment shader GLSL
file(WRITE "${SHADER_DIR}/height_field.frag" "
#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec4 fragColor;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec3 lightDir;
    float time;
} pc;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(pc.lightDir);
    
    // Simple Phong shading
    float ambient = 0.3;
    float diffuse = max(dot(normal, -lightDir), 0.0) * 0.7;
    
    vec3 viewDir = normalize(vec3(0, 0, 1) - fragPos);
    vec3 reflectDir = reflect(lightDir, normal);
    float specular = pow(max(dot(viewDir, reflectDir), 0.0), 32) * 0.5;
    
    vec3 lighting = vec3(ambient + diffuse + specular);
    outColor = vec4(fragColor.rgb * lighting, fragColor.a);
}
")

# Compile shaders to SPIR-V (simplified - using pre-compiled for example)
# In production, you would use glslangValidator or similar
file(WRITE "${GENERATED_DIR}/vertex_shader.inc" "
// Placeholder - in production, compile vertex shader to SPIR-V
0x07230203, 0x00010000, 0x00080001, 0x00000001, // ... rest of SPIR-V data
")

file(WRITE "${GENERATED_DIR}/fragment_shader.inc" "
// Placeholder - in production, compile fragment shader to SPIR-V
0x07230203, 0x00010000, 0x00080001, 0x00000001, // ... rest of SPIR-V data
")

# Create the Python extension module
pybind11_add_module(_vulkan_forge_native
    src/bindings.cpp
    src/vk_common.cpp
    src/heightfield_scene.cpp
    src/renderer.cpp
    src/vma_util.cpp
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
)

# Platform-specific settings
if(MSVC)
    target_compile_options(_vulkan_forge_native PRIVATE /W4 /bigobj)
else()
    target_compile_options(_vulkan_forge_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation rules
install(TARGETS _vulkan_forge_native
    LIBRARY DESTINATION lib/vulkan_forge
    ARCHIVE DESTINATION lib/vulkan_forge
    RUNTIME DESTINATION lib/vulkan_forge
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
message(STATUS "==== Vulkan Forge Configuration ====")
message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan library: ${Vulkan_LIBRARIES}")
message(STATUS "GLM found: ${glm_FOUND}")
message(STATUS "Python: ${PYTHON_EXECUTABLE}")
message(STATUS "=====================================")