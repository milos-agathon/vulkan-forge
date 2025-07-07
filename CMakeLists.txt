cmake_minimum_required(VERSION 3.20)
project(vulkan-forge VERSION 0.1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Vulkan REQUIRED)

# Add Tracy profiling
add_subdirectory(third_party/tracy)

# Main executable (placeholder for now)
add_executable(vulkan_forge_test
    src/main.cpp  # You'll create this next
)

target_link_libraries(vulkan_forge_test PRIVATE 
    VulkanForge::Tracy
    Vulkan::Vulkan
)

# Show Tracy configuration
message(STATUS "Tracy profiling enabled: ${VULKAN_FORGE_ENABLE_PROFILING}")