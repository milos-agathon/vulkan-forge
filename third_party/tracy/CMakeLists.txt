# Tracy Profiler Integration for Vulkan-Forge
# Handles conditional compilation and Vulkan integration

# Check if Tracy submodule exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/upstream/CMakeLists.txt")
    message(FATAL_ERROR 
        "Tracy submodule not found. Run: git submodule update --init --recursive")
endif()

# Tracy build options
option(VULKAN_FORGE_ENABLE_PROFILING "Enable Tracy profiling" OFF)
option(VULKAN_FORGE_TRACY_ON_DEMAND "Tracy on-demand mode" ON)
option(VULKAN_FORGE_TRACY_CALLSTACK "Enable Tracy callstack collection" OFF)

if(VULKAN_FORGE_ENABLE_PROFILING)
    message(STATUS "Tracy profiling enabled")
    
    # Set Tracy configuration
    set(TRACY_ENABLE ON CACHE BOOL "Enable Tracy")
    set(TRACY_ON_DEMAND ${VULKAN_FORGE_TRACY_ON_DEMAND} CACHE BOOL "Tracy on demand")
    set(TRACY_CALLSTACK ${VULKAN_FORGE_TRACY_CALLSTACK} CACHE BOOL "Tracy callstack")
    set(TRACY_NO_VULKAN OFF CACHE BOOL "Disable Vulkan profiling")
    
    # Platform-specific settings
    if(WIN32)
        set(TRACY_TIMER_QPC ON CACHE BOOL "Use QueryPerformanceCounter on Windows")
    endif()
    
    # Add Tracy client library
    add_subdirectory(upstream)
    
    # Create Tracy wrapper library
    add_library(vulkan_forge_tracy 
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_forge_tracy.cpp
    )
    
    target_link_libraries(vulkan_forge_tracy PUBLIC Tracy::TracyClient)
    
    target_compile_definitions(vulkan_forge_tracy PUBLIC
        TRACY_ENABLE=1
        $<$<BOOL:${VULKAN_FORGE_TRACY_ON_DEMAND}>:TRACY_ON_DEMAND=1>
        $<$<BOOL:${VULKAN_FORGE_TRACY_CALLSTACK}>:TRACY_CALLSTACK=1>
        VULKAN_FORGE_PROFILING_ENABLED=1
    )
    
    # Include our custom Tracy headers
    target_include_directories(vulkan_forge_tracy PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Vulkan integration (if Vulkan is available)
    if(TARGET Vulkan::Vulkan)
        target_compile_definitions(vulkan_forge_tracy PUBLIC 
            TRACY_VK_USE_SYMBOL_TABLE=1
        )
        target_link_libraries(vulkan_forge_tracy PUBLIC Vulkan::Vulkan)
    endif()
    
else()
    message(STATUS "Tracy profiling disabled")
    
    # Create empty interface library for disabled profiling
    add_library(vulkan_forge_tracy INTERFACE)
    
    target_compile_definitions(vulkan_forge_tracy INTERFACE
        VULKAN_FORGE_PROFILING_ENABLED=0
        TRACY_ENABLE=0
    )
    
    # Include directory for no-op macros
    target_include_directories(vulkan_forge_tracy INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# Create alias for consistent naming
add_library(VulkanForge::Tracy ALIAS vulkan_forge_tracy)

# Export configuration for build summary
get_target_property(TRACY_DEFINITIONS vulkan_forge_tracy INTERFACE_COMPILE_DEFINITIONS)
message(STATUS "Tracy configuration: ${TRACY_DEFINITIONS}")

# Optional: Build Tracy verification program
option(VULKAN_FORGE_BUILD_TRACY_VERIFY "Build Tracy verification program" OFF)

if(VULKAN_FORGE_BUILD_TRACY_VERIFY AND VULKAN_FORGE_ENABLE_PROFILING)
    add_executable(verify_tracy 
        ${CMAKE_CURRENT_SOURCE_DIR}/verify_tracy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_forge_tracy.cpp
    )
    
    target_link_libraries(verify_tracy PRIVATE VulkanForge::Tracy)
    
    # Ensure it gets built with the main project
    if(TARGET vulkan_forge)
        add_dependencies(vulkan_forge verify_tracy)
    endif()
    
    message(STATUS "Building Tracy verification program")
endif()