/**
 * @file verify_tracy.cpp
 * @brief Simple verification program to test Tracy integration
 * 
 * Compile with:
 *   g++ -I. verify_tracy.cpp vulkan_forge_tracy.cpp -o verify_tracy
 * 
 * Run with profiling enabled:
 *   g++ -DVULKAN_FORGE_PROFILING_ENABLED=1 -I. -Iupstream/public verify_tracy.cpp vulkan_forge_tracy.cpp -o verify_tracy
 */

#include "vulkan_forge_tracy.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <random>

using namespace vulkan_forge::profiling;

void simulate_work(int duration_ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
}

void test_memory_tracking() {
    VF_TRACY_ZONE_MEMORY("MemoryTest");
    
    // Allocate some memory
    constexpr size_t buffer_size = 1024 * 1024; // 1MB
    void* buffer = malloc(buffer_size);
    VF_TRACY_ALLOC(buffer, buffer_size);
    
    simulate_work(10);
    
    free(buffer);
    VF_TRACY_FREE(buffer);
    
    VF_TRACY_MESSAGE("Memory allocation test completed");
}

void test_nested_zones() {
    VF_TRACY_ZONE_RENDER("NestedTest");
    
    {
        VF_TRACY_ZONE_NAMED("Level1");
        simulate_work(5);
        
        {
            VF_TRACY_ZONE_NAMED("Level2");
            simulate_work(3);
            
            {
                VF_TRACY_ZONE_COMPUTE("Level3_Compute");
                simulate_work(2);
            }
        }
    }
}

void test_plotting() {
    VF_TRACY_ZONE_IO("PlottingTest");
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(30.0f, 120.0f);
    
    // Simulate FPS measurements
    for (int i = 0; i < 10; ++i) {
        float fps = dis(gen);
        VF_TRACY_PLOT("FPS", fps);
        VF_TRACY_PLOT("Frame Time (ms)", 1000.0f / fps);
        simulate_work(16); // ~60 FPS
        VF_TRACY_FRAME_MARK();
    }
}

void simulate_render_loop() {
    VF_TRACY_ZONE_RENDER("RenderLoop");
    
    for (int frame = 0; frame < 60; ++frame) {
        VF_TRACY_ZONE_NAMED("Frame");
        VF_TRACY_ZONE_TEXT("Frame processing");
        VF_TRACY_ZONE_VALUE(frame);
        
        // Simulate variable frame time
        int frame_time = 14 + (frame % 5); // 14-18ms
        
        {
            VF_TRACY_ZONE_RENDER("Geometry Pass");
            simulate_work(frame_time / 3);
        }
        
        {
            VF_TRACY_ZONE_RENDER("Lighting Pass");  
            simulate_work(frame_time / 3);
        }
        
        {
            VF_TRACY_ZONE_RENDER("Post Processing");
            simulate_work(frame_time / 3);
        }
        
        // Plot frame metrics
        float fps = 1000.0f / frame_time;
        VF_TRACY_PLOT("Simulated FPS", fps);
        VF_TRACY_PLOT("Frame Time (ms)", frame_time);
        
        VF_TRACY_FRAME_MARK();
    }
}

int main() {
    std::cout << "Tracy Integration Verification\n";
    std::cout << "==============================\n\n";
    
    // Show configuration
    std::cout << get_config_string() << "\n\n";
    
    if (is_enabled()) {
        std::cout << "✅ Tracy profiling is ENABLED\n";
        std::cout << "   Connect with Tracy profiler to see live data\n\n";
    } else {
        std::cout << "ℹ️  Tracy profiling is DISABLED (no overhead)\n\n";
    }
    
    VF_TRACY_MESSAGE_COLOR("Starting Tracy verification", 0x00FF00);
    
    std::cout << "Running tests...\n";
    
    // Test 1: Basic zones and timing
    std::cout << "  🔍 Testing basic profiling zones...\n";
    {
        VF_TRACY_ZONE_NAMED("MainTest");
        simulate_work(50);
    }
    
    // Test 2: Nested zones
    std::cout << "  🔍 Testing nested zones...\n";
    test_nested_zones();
    
    // Test 3: Memory tracking
    std::cout << "  🔍 Testing memory tracking...\n"; 
    test_memory_tracking();
    
    // Test 4: Plotting
    std::cout << "  🔍 Testing plotting functionality...\n";
    test_plotting();
    
    // Test 5: Render loop simulation
    std::cout << "  🔍 Testing render loop simulation (60 frames)...\n";
    simulate_render_loop();
    
    VF_TRACY_MESSAGE_COLOR("Tracy verification completed", 0x0000FF);
    
    std::cout << "\n✅ All tests completed!\n\n";
    
    if (is_enabled()) {
        std::cout << "To view profiling data:\n";
        std::cout << "  1. Download Tracy from: https://github.com/wolfpld/tracy/releases\n";
        std::cout << "  2. Run this program\n";
        std::cout << "  3. Open Tracy profiler and click 'Connect'\n";
        std::cout << "  4. You should see zones, memory allocations, and plots\n\n";
        
        std::cout << "Expected profiling data:\n";
        std::cout << "  - 60 frames of render loop simulation\n";
        std::cout << "  - Memory allocation/deallocation events\n"; 
        std::cout << "  - FPS and frame time plots\n";
        std::cout << "  - Nested zone hierarchies\n";
        std::cout << "  - Color-coded zones by category\n\n";
        
        // Keep running for Tracy connection
        std::cout << "Program will run for 30 seconds to allow Tracy connection...\n";
        for (int i = 30; i > 0; --i) {
            VF_TRACY_ZONE_NAMED("KeepAlive");
            std::cout << "\r⏱️  " << i << " seconds remaining... " << std::flush;
            simulate_work(1000);
            VF_TRACY_FRAME_MARK();
        }
        std::cout << "\n";
    } else {
        std::cout << "Tracy is disabled - no profiling overhead!\n";
        std::cout << "To enable profiling, compile with -DVULKAN_FORGE_PROFILING_ENABLED=1\n\n";
    }
    
    return 0;
}