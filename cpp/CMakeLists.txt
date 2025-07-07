# ============================================================================
# CMakeLists.txt - Cross-platform build configuration for vulkan-forge
# ============================================================================
# Handles Vulkan SDK detection, wheel building, static linking, and
# architecture-specific optimizations for Windows, macOS, and Linux
# ============================================================================

cmake_minimum_required(VERSION 3.24)

# ============================================================================
# Project Setup and Policy Configuration
# ============================================================================

project(vulkan_forge
    VERSION 0.1.0
    DESCRIPTION "High-performance GPU renderer using Vulkan"
    LANGUAGES CXX C
)

# Set policies for compatibility
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)  # FindPythonInterp
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)  # find_package uses <PackageName>_ROOT variables
endif()
if(POLICY CMP0094)
    cmake_policy(SET CMP0094 NEW)  # FindPython should return the first matching Python
endif()

# ============================================================================
# Build Configuration
# ============================================================================

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Project Options
# ============================================================================

option(VF_ENABLE_VALIDATION_LAYERS "Enable Vulkan validation layers" OFF)
option(VF_ENABLE_GPU_DEBUG "Enable GPU debugging features" OFF)
option(VF_STATIC_VULKAN "Link Vulkan statically" OFF)
option(VF_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(VF_ENABLE_LTO "Enable Link Time Optimization" ON)
option(VF_STATIC_LINK "Static link runtime libraries" OFF)
option(VF_USE_MOLTEN_VK "Use MoltenVK on macOS" ON)
option(VF_USE_SYSTEM_VULKAN "Use system Vulkan installation" ON)
option(VF_ENABLE_MARCH_NATIVE "Enable -march=native" OFF)

# Platform detection
if(WIN32)
    set(VF_PLATFORM "Windows")
elseif(APPLE)
    set(VF_PLATFORM "macOS")
elseif(UNIX)
    set(VF_PLATFORM "Linux")
else()
    set(VF_PLATFORM "Unknown")
endif()

message(STATUS "Building vulkan-forge for ${VF_PLATFORM}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# Compiler-Specific Settings
# ============================================================================

if(MSVC)
    # Visual Studio specific settings
    add_compile_options(
        /W4                     # High warning level
        /bigobj                 # Large object file support
        /MP                     # Multi-processor compilation
        /permissive-           # Conformance mode
    )
    
    # Disable specific warnings for external libraries
    add_compile_options(
        /wd4127                 # conditional expression is constant (VMA)
        /wd4100                 # unreferenced formal parameter
    )
    
    # Static runtime linking for wheels
    if(VF_STATIC_LINK)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    
    # Enable large address aware
    add_link_options(/LARGEADDRESSAWARE)
    
else()
    # GCC/Clang settings
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )
    
    # Architecture-specific optimizations
    if(VF_ENABLE_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
    
    # Platform-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        if(VF_ENABLE_SIMD)
            add_compile_options(-msse4.2 -mavx)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        if(VF_ENABLE_SIMD)
            add_compile_options(-march=armv8-a+simd)
        endif()
    endif()
endif()

# ============================================================================
# Find Required Packages - Python
# ============================================================================

# Find Python and pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 2.11.0 REQUIRED CONFIG)

message(STATUS "Python: ${Python_VERSION}")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "pybind11: ${pybind11_VERSION}")

# ============================================================================
# Find Required Packages - Vulkan SDK
# ============================================================================

# Platform-specific Vulkan SDK detection
if(VF_PLATFORM STREQUAL "Windows")
    # Windows Vulkan SDK detection
    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
        message(STATUS "Using VULKAN_SDK environment variable: ${VULKAN_SDK_PATH}")
        
        # Set hints for find_package
        list(APPEND CMAKE_PREFIX_PATH "${VULKAN_SDK_PATH}")
        set(Vulkan_INCLUDE_DIR "${VULKAN_SDK_PATH}/Include")
        set(Vulkan_LIBRARY "${VULKAN_SDK_PATH}/Lib/vulkan-1.lib")
        
        # Find glslc
        find_program(GLSLC_EXECUTABLE 
            NAMES glslc
            HINTS "${VULKAN_SDK_PATH}/Bin"
            REQUIRED
        )
    else()
        message(WARNING "VULKAN_SDK environment variable not set on Windows")
    endif()
    
elseif(VF_PLATFORM STREQUAL "macOS")
    # macOS MoltenVK detection
    if(VF_USE_MOLTEN_VK)
        if(DEFINED ENV{VULKAN_SDK})
            set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
            message(STATUS "Using VULKAN_SDK environment variable: ${VULKAN_SDK_PATH}")
            
            # MoltenVK specific paths
            list(APPEND CMAKE_PREFIX_PATH "${VULKAN_SDK_PATH}")
            set(Vulkan_INCLUDE_DIR "${VULKAN_SDK_PATH}/include")
            
            # Find MoltenVK library
            find_library(MoltenVK_LIBRARY
                NAMES MoltenVK
                HINTS "${VULKAN_SDK_PATH}/lib"
                REQUIRED
            )
            
            # Set Vulkan library to MoltenVK
            set(Vulkan_LIBRARY ${MoltenVK_LIBRARY})
            
        else()
            # Try Homebrew or system installation
            find_library(MoltenVK_LIBRARY
                NAMES MoltenVK
                PATHS 
                    /usr/local/lib
                    /opt/homebrew/lib
                    /usr/lib
            )
            
            if(MoltenVK_LIBRARY)
                set(Vulkan_LIBRARY ${MoltenVK_LIBRARY})
                message(STATUS "Found MoltenVK: ${MoltenVK_LIBRARY}")
            endif()
        endif()
        
        # Find glslc
        find_program(GLSLC_EXECUTABLE 
            NAMES glslc
            HINTS 
                "${VULKAN_SDK_PATH}/bin"
                "/usr/local/bin"
                "/opt/homebrew/bin"
        )
    endif()
    
elseif(VF_PLATFORM STREQUAL "Linux")
    # Linux system Vulkan detection
    if(VF_USE_SYSTEM_VULKAN)
        # Use pkg-config to find Vulkan
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(VULKAN_PKG QUIET vulkan)
            if(VULKAN_PKG_FOUND)
                set(Vulkan_INCLUDE_DIR ${VULKAN_PKG_INCLUDE_DIRS})
                set(Vulkan_LIBRARY ${VULKAN_PKG_LIBRARIES})
                message(STATUS "Found Vulkan via pkg-config")
            endif()
        endif()
        
        # Find glslc
        find_program(GLSLC_EXECUTABLE 
            NAMES glslc
            HINTS 
                "$ENV{VULKAN_SDK}/bin"
                "/usr/bin"
                "/usr/local/bin"
        )
    endif()
endif()

# Standard Vulkan find_package as fallback
find_package(Vulkan REQUIRED)

message(STATUS "Vulkan found: ${Vulkan_FOUND}")
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan library: ${Vulkan_LIBRARIES}")

if(GLSLC_EXECUTABLE)
    message(STATUS "glslc found: ${GLSLC_EXECUTABLE}")
else()
    message(WARNING "glslc not found - shader compilation may not work")
endif()

# ============================================================================
# Find Required Packages - GLM
# ============================================================================

find_package(glm QUIET CONFIG)

if(NOT glm_FOUND)
    message(STATUS "GLM not found locally - downloading header-only copy")
    
    include(FetchContent)
    FetchContent_Declare(
        glm_headers
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
        URL_HASH SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
    )
    FetchContent_MakeAvailable(glm_headers)
    
    # Create interface library
    add_library(glm_interface INTERFACE)
    target_include_directories(glm_interface INTERFACE
        "$<BUILD_INTERFACE:${glm_headers_SOURCE_DIR}>"
    )
    add_library(glm::glm ALIAS glm_interface)
endif()

# ============================================================================
# Find Required Packages - VulkanMemoryAllocator
# ============================================================================

include(FetchContent)
FetchContent_Declare(
    vma_headers
    URL https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/heads/master.zip
    URL_HASH SHA256=eb4c16b78e7a3a0b7cc9cb430c7b9cad99a171b0dd8a96b21ceae03e1f5ff4fe
)
FetchContent_MakeAvailable(vma_headers)

message(STATUS "VMA headers: ${vma_headers_SOURCE_DIR}")

# ============================================================================
# Shader Compilation Setup
# ============================================================================

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp/shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SHADER_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include/shaders")

# Create directories
file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}")
file(MAKE_DIRECTORY "${SHADER_INCLUDE_DIR}")

# Function to compile shaders
function(compile_shader SHADER_NAME)
    set(SHADER_SOURCE "${SHADER_SOURCE_DIR}/${SHADER_NAME}")
    set(SHADER_BINARY "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")
    set(SHADER_HEADER "${SHADER_INCLUDE_DIR}/${SHADER_NAME}.h")
    
    if(GLSLC_EXECUTABLE AND EXISTS "${SHADER_SOURCE}")
        # Compile GLSL to SPIR-V
        add_custom_command(
            OUTPUT "${SHADER_BINARY}"
            COMMAND "${GLSLC_EXECUTABLE}" 
                -O 
                -I "${SHADER_SOURCE_DIR}"
                "${SHADER_SOURCE}" 
                -o "${SHADER_BINARY}"
            DEPENDS "${SHADER_SOURCE}"
            COMMENT "Compiling shader ${SHADER_NAME}"
        )
        
        # Convert SPIR-V to C++ header
        add_custom_command(
            OUTPUT "${SHADER_HEADER}"
            COMMAND ${CMAKE_COMMAND}
                -DSHADER_BINARY="${SHADER_BINARY}"
                -DSHADER_HEADER="${SHADER_HEADER}"
                -DSHADER_NAME="${SHADER_NAME}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpirvToHeader.cmake"
            DEPENDS "${SHADER_BINARY}"
            COMMENT "Generating header for ${SHADER_NAME}"
        )
        
        list(APPEND SHADER_HEADERS "${SHADER_HEADER}")
        set(SHADER_HEADERS ${SHADER_HEADERS} PARENT_SCOPE)
    else()
        # Create dummy header for missing shaders
        set(DUMMY_CONTENT "// Dummy shader header for ${SHADER_NAME}\n")
        string(REPLACE "." "_" SHADER_VAR "${SHADER_NAME}")
        string(APPEND DUMMY_CONTENT "static const uint32_t ${SHADER_VAR}_spirv[] = {0x07230203};\n")
        string(APPEND DUMMY_CONTENT "static const size_t ${SHADER_VAR}_spirv_size = sizeof(${SHADER_VAR}_spirv);\n")
        
        file(WRITE "${SHADER_HEADER}" "${DUMMY_CONTENT}")
        message(STATUS "Created dummy header for ${SHADER_NAME}")
    endif()
endfunction()

# Compile all shaders
set(SHADER_NAMES
    "height_field.vert"
    "height_field.frag"
    "point_cloud.vert"
    "point_cloud.frag"
)

foreach(SHADER_NAME ${SHADER_NAMES})
    compile_shader(${SHADER_NAME})
endforeach()

# Create shader compilation target
add_custom_target(compile_shaders ALL
    DEPENDS ${SHADER_HEADERS}
    COMMENT "Compiling all shaders"
)

# ============================================================================
# Main Library Target
# ============================================================================

# Collect source files
set(VF_SOURCES
    cpp/src/bindings.cpp
    cpp/src/vk_common.cpp
    cpp/src/heightfield_scene.cpp
    cpp/src/renderer.cpp
    cpp/src/mesh_loader.cpp
)

# Collect header files
set(VF_HEADERS
    cpp/include/vk_common.hpp
    cpp/include/heightfield_scene.hpp
    cpp/include/renderer.hpp
    cpp/include/mesh_loader.hpp
)

# Create pybind11 module
pybind11_add_module(_vulkan_forge_native
    ${VF_SOURCES}
    ${VF_HEADERS}
)

# Add shader compilation dependency
add_dependencies(_vulkan_forge_native compile_shaders)

# ============================================================================
# Include Directories
# ============================================================================

target_include_directories(_vulkan_forge_native PRIVATE
    # Project headers
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    
    # Generated shader headers
    ${SHADER_INCLUDE_DIR}
    
    # Vulkan
    ${Vulkan_INCLUDE_DIRS}
    
    # VulkanMemoryAllocator
    ${vma_headers_SOURCE_DIR}/include
)

# ============================================================================
# Link Libraries
# ============================================================================

target_link_libraries(_vulkan_forge_native PRIVATE
    # GLM (header-only)
    glm::glm
    
    # Vulkan
    ${Vulkan_LIBRARIES}
)

# Platform-specific linking
if(VF_PLATFORM STREQUAL "Windows")
    # Windows-specific libraries
    target_link_libraries(_vulkan_forge_native PRIVATE
        user32.lib
        gdi32.lib
        winmm.lib
    )
    
elseif(VF_PLATFORM STREQUAL "macOS")
    # macOS-specific frameworks
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
    find_library(IOKIT_LIBRARY IOKit REQUIRED)
    find_library(CORE_VIDEO_LIBRARY CoreVideo REQUIRED)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(QUARTZ_CORE_LIBRARY QuartzCore REQUIRED)
    
    target_link_libraries(_vulkan_forge_native PRIVATE
        ${COCOA_LIBRARY}
        ${IOKIT_LIBRARY}
        ${CORE_VIDEO_LIBRARY}
        ${METAL_LIBRARY}
        ${QUARTZ_CORE_LIBRARY}
    )
    
elseif(VF_PLATFORM STREQUAL "Linux")
    # Linux-specific libraries
    find_package(Threads REQUIRED)
    target_link_libraries(_vulkan_forge_native PRIVATE
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )
endif()

# ============================================================================
# Compile Definitions
# ============================================================================

target_compile_definitions(_vulkan_forge_native PRIVATE
    # Version information
    VERSION_INFO="${PROJECT_VERSION}"
    
    # VMA implementation
    VMA_IMPLEMENTATION
    
    # Platform detection
    VF_PLATFORM_${VF_PLATFORM}=1
    
    # Vulkan features
    VK_NO_PROTOTYPES=1
    $<$<BOOL:${VF_ENABLE_VALIDATION_LAYERS}>:VF_ENABLE_VALIDATION_LAYERS>
    $<$<BOOL:${VF_ENABLE_GPU_DEBUG}>:VF_ENABLE_GPU_DEBUG>
    
    # Platform-specific definitions
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Darwin>:VK_USE_PLATFORM_MACOS_MVK>
    $<$<PLATFORM_ID:Linux>:VK_USE_PLATFORM_XLIB_KHR>
)

# ============================================================================
# Compiler and Linker Options
# ============================================================================

# Link Time Optimization
if(VF_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
    
    if(LTO_SUPPORTED)
        set_property(TARGET _vulkan_forge_native PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "Link Time Optimization enabled")
    else()
        message(STATUS "Link Time Optimization not supported: ${LTO_ERROR}")
    endif()
endif()

# Platform-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(_vulkan_forge_native PRIVATE /O2 /Ob2)
    else()
        target_compile_options(_vulkan_forge_native PRIVATE -O3 -DNDEBUG)
        
        # Additional GCC/Clang optimizations
        target_compile_options(_vulkan_forge_native PRIVATE
            -ffast-math
            -fomit-frame-pointer
            -fno-signed-zeros
            -fno-trapping-math
        )
    endif()
endif()

# ============================================================================
# Module Properties
# ============================================================================

# Set module properties for Python import
set_target_properties(_vulkan_forge_native PROPERTIES
    # Module naming
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    
    # Output directory
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/vulkan_forge"
    
    # Version
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    
    # Visibility
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ============================================================================
# Installation Rules
# ============================================================================

# Install the compiled module
install(TARGETS _vulkan_forge_native
    COMPONENT vulkan_forge
    LIBRARY DESTINATION vulkan_forge
    ARCHIVE DESTINATION vulkan_forge
    RUNTIME DESTINATION vulkan_forge
)

# Install Python package files
install(DIRECTORY python/vulkan_forge/
    DESTINATION vulkan_forge
    COMPONENT vulkan_forge
    FILES_MATCHING PATTERN "*.py"
)

# Install shader files for development
install(DIRECTORY ${SHADER_BINARY_DIR}/
    DESTINATION share/vulkan_forge/shaders
    COMPONENT vulkan_forge
    FILES_MATCHING PATTERN "*.spv"
    OPTIONAL
)

# ============================================================================
# Development Utilities
# ============================================================================

# Create compile_commands.json for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Summary
message(STATUS "==== vulkan-forge Configuration Summary ====")
message(STATUS "Platform: ${VF_PLATFORM}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python: ${Python_VERSION}")
message(STATUS "Vulkan SDK: ${Vulkan_FOUND}")
message(STATUS "GLM: ${glm_FOUND}")
message(STATUS "VMA: ${vma_headers_SOURCE_DIR}")
message(STATUS "GLSLC: ${GLSLC_EXECUTABLE}")
message(STATUS "LTO: ${VF_ENABLE_LTO}")
message(STATUS "SIMD: ${VF_ENABLE_SIMD}")
message(STATUS "Validation: ${VF_ENABLE_VALIDATION_LAYERS}")
message(STATUS "==========================================")

# ============================================================================
# Create cmake/SpirvToHeader.cmake helper script
# ============================================================================

file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpirvToHeader.cmake" "
# Convert SPIR-V binary to C++ header
file(READ \"\${SHADER_BINARY}\" SHADER_DATA HEX)

# Convert hex string to array
string(REGEX REPLACE \"([0-9a-f][0-9a-f])\" \"0x\\\\1, \" SHADER_ARRAY \"\${SHADER_DATA}\")
string(REGEX REPLACE \", $\" \"\" SHADER_ARRAY \"\${SHADER_ARRAY}\")

# Calculate size
string(LENGTH \"\${SHADER_DATA}\" HEX_LENGTH)
math(EXPR ARRAY_SIZE \"\${HEX_LENGTH} / 2\")

# Generate variable name
string(REPLACE \".\" \"_\" SHADER_VAR \"\${SHADER_NAME}\")

# Generate header content
set(HEADER_CONTENT \"// Generated shader header for \${SHADER_NAME}
#pragma once
#include <cstdint>
#include <cstddef>

static const uint32_t \${SHADER_VAR}_spirv[] = {
    \${SHADER_ARRAY}
};

static const size_t \${SHADER_VAR}_spirv_size = \${ARRAY_SIZE};
\")

# Write header file
file(WRITE \"\${SHADER_HEADER}\" \"\${HEADER_CONTENT}\")
")