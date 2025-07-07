# Enhanced SPIR-V to C++ header converter for Vulkan Forge
# Provides validation, metadata generation, and proper formatting
# 
# Usage:
#   cmake -DSPIRV_FILE=shader.spv -DHEADER_FILE=shader.h -DARRAY_NAME=shaderSpirv -P SpirvToHeader.cmake
#
# Required variables:
#   SPIRV_FILE  - Input SPIR-V binary file path
#   HEADER_FILE - Output C++ header file path  
#   ARRAY_NAME  - C++ array variable name
#
# Optional variables:
#   SHADER_STAGE - Vertex/Fragment/Compute (auto-detected if not provided)
#   NAMESPACE    - C++ namespace (default: vulkan_forge::shaders)
#   VALIDATE_SPIRV - Enable SPIR-V validation (default: ON)

cmake_minimum_required(VERSION 3.16)

# Set default values for optional variables
if(NOT DEFINED NAMESPACE)
    set(NAMESPACE "vulkan_forge::shaders")
endif()

if(NOT DEFINED VALIDATE_SPIRV)
    set(VALIDATE_SPIRV ON)
endif()

# Utility function for error reporting
function(spirv_error MESSAGE)
    message(FATAL_ERROR "SPIR-V Converter Error: ${MESSAGE}")
endfunction()

# Utility function for warnings
function(spirv_warning MESSAGE)
    message(WARNING "SPIR-V Converter Warning: ${MESSAGE}")
endfunction()

# Validate required parameters
if(NOT DEFINED SPIRV_FILE)
    spirv_error("SPIRV_FILE parameter is required")
endif()

if(NOT DEFINED HEADER_FILE)
    spirv_error("HEADER_FILE parameter is required")
endif()

if(NOT DEFINED ARRAY_NAME)
    spirv_error("ARRAY_NAME parameter is required")
endif()

# Check if input file exists
if(NOT EXISTS "${SPIRV_FILE}")
    spirv_error("SPIR-V file not found: ${SPIRV_FILE}")
endif()

# Get file size and validate minimum size
file(SIZE "${SPIRV_FILE}" SPIRV_SIZE)
if(SPIRV_SIZE LESS 20)
    spirv_error("SPIR-V file too small (${SPIRV_SIZE} bytes, minimum 20 required)")
endif()

# Validate file size alignment (must be multiple of 4)
math(EXPR SIZE_MOD_4 "${SPIRV_SIZE} % 4")
if(NOT SIZE_MOD_4 EQUAL 0)
    spirv_error("SPIR-V file size not aligned to 4 bytes (${SPIRV_SIZE} bytes)")
endif()

# Read SPIR-V file as hex string
file(READ "${SPIRV_FILE}" SPIRV_HEX HEX)

# Validate SPIR-V magic number (0x07230203 in little-endian)
string(SUBSTRING "${SPIRV_HEX}" 0 8 MAGIC_HEX)
if(NOT MAGIC_HEX STREQUAL "07230203")
    spirv_error("Invalid SPIR-V magic number: 0x${MAGIC_HEX} (expected: 0x07230203)")
endif()

# Extract SPIR-V header information for validation
string(SUBSTRING "${SPIRV_HEX}" 8 8 VERSION_HEX)
string(SUBSTRING "${SPIRV_HEX}" 16 8 GENERATOR_HEX)
string(SUBSTRING "${SPIRV_HEX}" 24 8 BOUND_HEX)
string(SUBSTRING "${SPIRV_HEX}" 32 8 SCHEMA_HEX)

# Convert version to readable format
set(VERSION_MAJOR "0")
set(VERSION_MINOR "0")

# Parse version (format: 0x00MMmmpp where MM=major, mm=minor, pp=patch)
if(VERSION_HEX MATCHES "^00([0-9a-fA-F][0-9a-fA-F])([0-9a-fA-F][0-9a-fA-F])..")
    math(EXPR VERSION_MAJOR "0x${CMAKE_MATCH_1}")
    math(EXPR VERSION_MINOR "0x${CMAKE_MATCH_2}")
endif()

# Auto-detect shader stage from filename if not provided
if(NOT DEFINED SHADER_STAGE)
    get_filename_component(SPIRV_NAME "${SPIRV_FILE}" NAME)
    if(SPIRV_NAME MATCHES ".*vert.*")
        set(SHADER_STAGE "Vertex")
    elseif(SPIRV_NAME MATCHES ".*frag.*")
        set(SHADER_STAGE "Fragment")
    elseif(SPIRV_NAME MATCHES ".*comp.*")
        set(SHADER_STAGE "Compute")
    elseif(SPIRV_NAME MATCHES ".*geom.*")
        set(SHADER_STAGE "Geometry")
    elseif(SPIRV_NAME MATCHES ".*tesc.*")
        set(SHADER_STAGE "TessellationControl")
    elseif(SPIRV_NAME MATCHES ".*tese.*")
        set(SHADER_STAGE "TessellationEvaluation")
    else()
        set(SHADER_STAGE "Unknown")
    endif()
endif()

# Calculate array length
string(LENGTH "${SPIRV_HEX}" HEX_LENGTH)
math(EXPR UINT32_COUNT "${HEX_LENGTH} / 8")

# Validate that we have complete uint32_t values
math(EXPR HEX_MOD_8 "${HEX_LENGTH} % 8")
if(NOT HEX_MOD_8 EQUAL 0)
    spirv_error("SPIR-V hex data length not multiple of 8 (${HEX_LENGTH})")
endif()

# Convert hex string to formatted uint32_t array
set(ARRAY_VALUES "")
set(CURRENT_LINE "")
set(VALUES_PER_LINE 6)
set(LINE_COUNT 0)

for(i RANGE 0 ${UINT32_COUNT})
    if(i EQUAL ${UINT32_COUNT})
        break()
    endif()
    
    # Calculate byte position in hex string
    math(EXPR HEX_POS "${i} * 8")
    
    # Extract 8 hex characters (4 bytes)
    string(SUBSTRING "${SPIRV_HEX}" ${HEX_POS} 8 HEX_VALUE)
    
    # Convert little-endian hex to uint32_t format
    string(SUBSTRING "${HEX_VALUE}" 6 2 BYTE0)
    string(SUBSTRING "${HEX_VALUE}" 4 2 BYTE1)
    string(SUBSTRING "${HEX_VALUE}" 2 2 BYTE2)
    string(SUBSTRING "${HEX_VALUE}" 0 2 BYTE3)
    
    set(UINT32_VALUE "0x${BYTE0}${BYTE1}${BYTE2}${BYTE3}")
    
    # Add to current line
    if(CURRENT_LINE STREQUAL "")
        set(CURRENT_LINE "${UINT32_VALUE}")
    else()
        set(CURRENT_LINE "${CURRENT_LINE}, ${UINT32_VALUE}")
    endif()
    
    math(EXPR LINE_COUNT "${LINE_COUNT} + 1")
    
    # Check if line is complete or this is the last value
    math(EXPR IS_LAST "${i} + 1")
    if(LINE_COUNT EQUAL ${VALUES_PER_LINE} OR IS_LAST EQUAL ${UINT32_COUNT})
        if(IS_LAST EQUAL ${UINT32_COUNT})
            set(ARRAY_VALUES "${ARRAY_VALUES}    ${CURRENT_LINE}\n")
        else()
            set(ARRAY_VALUES "${ARRAY_VALUES}    ${CURRENT_LINE},\n")
        endif()
        set(CURRENT_LINE "")
        set(LINE_COUNT 0)
    endif()
endfor()

# Generate header guard name
string(TOUPPER "${ARRAY_NAME}" HEADER_GUARD_BASE)
string(REGEX REPLACE "[^A-Z0-9_]" "_" HEADER_GUARD "${HEADER_GUARD_BASE}")
set(HEADER_GUARD "VULKAN_FORGE_${HEADER_GUARD}_H")

# Get current timestamp for generation info
string(TIMESTAMP GENERATION_TIME "%Y-%m-%d %H:%M:%S UTC" UTC)

# Get relative path for cleaner comments
file(RELATIVE_PATH SPIRV_REL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${SPIRV_FILE}")

# Generate namespace components
string(REPLACE "::" ";" NAMESPACE_LIST "${NAMESPACE}")
set(NAMESPACE_OPEN "")
set(NAMESPACE_CLOSE "")
foreach(NS_PART ${NAMESPACE_LIST})
    set(NAMESPACE_OPEN "${NAMESPACE_OPEN}namespace ${NS_PART} {\n")
    set(NAMESPACE_CLOSE "} // namespace ${NS_PART}\n${NAMESPACE_CLOSE}")
endforeach()

# Generate shader info structure name
set(INFO_STRUCT_NAME "${ARRAY_NAME}Info")

# Build the complete header content
set(HEADER_CONTENT "//==============================================================================
// Auto-generated SPIR-V shader header for Vulkan Forge
// Source: ${SPIRV_REL_PATH}
// Generated: ${GENERATION_TIME}
// 
// DO NOT EDIT THIS FILE MANUALLY
//==============================================================================

#ifndef ${HEADER_GUARD}
#define ${HEADER_GUARD}

#include <cstdint>
#include <cstddef>

${NAMESPACE_OPEN}
//------------------------------------------------------------------------------
// SPIR-V Binary Data
//------------------------------------------------------------------------------

/// SPIR-V bytecode for ${SHADER_STAGE} shader
/// Size: ${SPIRV_SIZE} bytes (${UINT32_COUNT} uint32_t values)
/// SPIR-V Version: ${VERSION_MAJOR}.${VERSION_MINOR}
alignas(4) static constexpr uint32_t ${ARRAY_NAME}[] = {
${ARRAY_VALUES}};

//------------------------------------------------------------------------------
// Shader Metadata
//------------------------------------------------------------------------------

/// Shader information structure for runtime use
struct ShaderInfo {
    const uint32_t* spirv_data;        ///< Pointer to SPIR-V bytecode
    size_t spirv_size;                 ///< Size in bytes
    size_t spirv_count;                ///< Number of uint32_t values
    const char* stage;                 ///< Shader stage name
    uint32_t spirv_version_major;      ///< SPIR-V major version
    uint32_t spirv_version_minor;      ///< SPIR-V minor version
    uint32_t magic_number;             ///< SPIR-V magic number for validation
};

/// Runtime shader information for ${ARRAY_NAME}
static constexpr ShaderInfo ${INFO_STRUCT_NAME} = {
    .spirv_data = ${ARRAY_NAME},
    .spirv_size = sizeof(${ARRAY_NAME}),
    .spirv_count = ${UINT32_COUNT},
    .stage = \"${SHADER_STAGE}\",
    .spirv_version_major = ${VERSION_MAJOR},
    .spirv_version_minor = ${VERSION_MINOR},
    .magic_number = 0x07230203U
};

//------------------------------------------------------------------------------
// Convenience Constants
//------------------------------------------------------------------------------

/// Size of SPIR-V data in bytes
static constexpr size_t ${ARRAY_NAME}_size = sizeof(${ARRAY_NAME});

/// Number of uint32_t values in SPIR-V data
static constexpr size_t ${ARRAY_NAME}_count = ${UINT32_COUNT};

/// Shader stage identifier
static constexpr const char* ${ARRAY_NAME}_stage = \"${SHADER_STAGE}\";

//------------------------------------------------------------------------------
// Validation Functions
//------------------------------------------------------------------------------

/// Validate SPIR-V data integrity at compile time
static constexpr bool validate_${ARRAY_NAME}() {
    return (${ARRAY_NAME}[0] == 0x07230203U) &&          // Magic number
           (sizeof(${ARRAY_NAME}) == ${SPIRV_SIZE}) &&    // Size check
           (${ARRAY_NAME}_count == ${UINT32_COUNT});       // Count check
}

/// Compile-time validation assertion
static_assert(validate_${ARRAY_NAME}(), \"SPIR-V data validation failed for ${ARRAY_NAME}\");

${NAMESPACE_CLOSE}

//------------------------------------------------------------------------------
// Usage Example
//------------------------------------------------------------------------------
/*
#include \"${ARRAY_NAME}.h\"

// Create Vulkan shader module
VkShaderModuleCreateInfo createInfo = {};
createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
createInfo.codeSize = ${NAMESPACE}::${ARRAY_NAME}_size;
createInfo.pCode = ${NAMESPACE}::${ARRAY_NAME};

VkShaderModule shaderModule;
VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);

// Access metadata
const auto& info = ${NAMESPACE}::${INFO_STRUCT_NAME};
printf(\"Loaded %s shader: %zu bytes, SPIR-V v%d.%d\\n\",
       info.stage, info.spirv_size, info.spirv_version_major, info.spirv_version_minor);
*/

#endif // ${HEADER_GUARD}
")

# Create output directory if it doesn't exist
get_filename_component(HEADER_DIR "${HEADER_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${HEADER_DIR}")

# Write the header file
file(WRITE "${HEADER_FILE}" "${HEADER_CONTENT}")

# Print generation summary
message(STATUS "Generated SPIR-V header: ${HEADER_FILE}")
message(STATUS "  Source: ${SPIRV_REL_PATH}")
message(STATUS "  Stage: ${SHADER_STAGE}")
message(STATUS "  Size: ${SPIRV_SIZE} bytes (${UINT32_COUNT} uint32_t values)")
message(STATUS "  SPIR-V Version: ${VERSION_MAJOR}.${VERSION_MINOR}")
message(STATUS "  Array: ${NAMESPACE}::${ARRAY_NAME}")
message(STATUS "  Info Struct: ${NAMESPACE}::${INFO_STRUCT_NAME}")

# Optional: Validate generated header syntax (requires C++ compiler)
if(VALIDATE_SPIRV AND CMAKE_CXX_COMPILER)
    # Create a minimal test program to validate syntax
    set(TEST_CPP "${CMAKE_CURRENT_BINARY_DIR}/test_${ARRAY_NAME}.cpp")
    file(WRITE "${TEST_CPP}" "
#include \"${HEADER_FILE}\"
int main() {
    static_assert(${NAMESPACE}::validate_${ARRAY_NAME}());
    return 0;
}
")
    
    # Try to compile the test
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -std=c++17 -c "${TEST_CPP}" -o "${CMAKE_CURRENT_BINARY_DIR}/test_${ARRAY_NAME}.o"
        RESULT_VARIABLE COMPILE_RESULT
        ERROR_VARIABLE COMPILE_ERROR
        OUTPUT_QUIET
    )
    
    if(COMPILE_RESULT EQUAL 0)
        message(STATUS "  ✓ Header syntax validation passed")
        file(REMOVE "${TEST_CPP}")
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/test_${ARRAY_NAME}.o")
    else()
        spirv_warning("Header syntax validation failed: ${COMPILE_ERROR}")
    endif()
endif()

message(STATUS "SPIR-V header generation completed successfully")