/**
 * @file main.cpp
 * @brief Vulkan-Forge main entry point with Tracy profiling
 * 
 * This is a minimal starter to test Tracy integration.
 * Eventually this will become the main Vulkan renderer.
 */

#include "third_party/tracy/vulkan_forge_tracy.hpp"
#include <iostream>
#include <chrono>
#include <thread>

using namespace vulkan_forge::profiling;

void simulate_render_frame() {
    VF_TRACY_ZONE_RENDER("RenderFrame");
    
    // Simulate some render work
    {
        VF_TRACY_ZONE_RENDER("GeometryPass");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    {
        VF_TRACY_ZONE_RENDER("LightingPass");
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    
    {
        VF_TRACY_ZONE_RENDER("PostProcess");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

int main() {
    std::cout << "Vulkan-Forge v0.1.0 - Performance Baseline Test\n";
    std::cout << "===============================================\n\n";
    
    // Show Tracy configuration
    std::cout << get_config_string() << "\n\n";
    
    if (is_enabled()) {
        std::cout << "✅ Tracy profiling ENABLED\n";
        std::cout << "   Connect Tracy profiler to see live performance data\n\n";
        
        VF_TRACY_MESSAGE_COLOR("Vulkan-Forge startup", VF_TRACY_COLOR_RENDER);
    } else {
        std::cout << "ℹ️  Tracy profiling DISABLED (no overhead)\n\n";
    }
    
    std::cout << "Running 144 FPS simulation for 10 seconds...\n";
    std::cout << "Target: 6.94ms per frame (144 FPS)\n\n";
    
    // Simulate 144 FPS for 10 seconds (1440 frames)
    constexpr int target_fps = 144;
    constexpr int frame_time_ms = 1000 / target_fps; // ~6.94ms
    constexpr int total_frames = target_fps * 10; // 10 seconds
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int frame = 0; frame < total_frames; ++frame) {
        VF_TRACY_ZONE_NAMED("MainLoop");
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Simulate frame work
        simulate_render_frame();
        
        // Calculate frame time
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        
        float frame_time_actual = frame_duration.count() / 1000.0f; // Convert to ms
        float fps_actual = 1000.0f / frame_time_actual;
        
        // Plot metrics
        VF_TRACY_PLOT("Frame Time (ms)", frame_time_actual);
        VF_TRACY_PLOT("FPS", fps_actual);
        VF_TRACY_PLOT("Target FPS", static_cast<float>(target_fps));
        
        // Progress indicator every second
        if (frame % target_fps == 0) {
            int seconds = frame / target_fps;
            std::cout << "  " << seconds << "s - Frame " << frame 
                      << " - " << fps_actual << " FPS\n";
        }
        
        // Frame boundary marker
        VF_TRACY_FRAME_MARK();
        
        // Sleep to maintain target framerate
        std::this_thread::sleep_for(std::chrono::milliseconds(frame_time_ms));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n✅ Performance baseline test completed!\n";
    std::cout << "   Total time: " << total_duration.count() << " seconds\n";
    std::cout << "   Frames rendered: " << total_frames << "\n";
    std::cout << "   Average FPS: " << total_frames / total_duration.count() << "\n\n";
    
    if (is_enabled()) {
        std::cout << "📊 Tracy profiling data captured:\n";
        std::cout << "   - Frame timing for " << total_frames << " frames\n";
        std::cout << "   - CPU zones for render passes\n";
        std::cout << "   - FPS and frame time plots\n\n";
        
        std::cout << "🔗 Connect Tracy profiler to view detailed performance data\n";
        std::cout << "   Download from: https://github.com/wolfpld/tracy/releases\n\n";
    }
    
    VF_TRACY_MESSAGE_COLOR("Vulkan-Forge performance test completed", VF_TRACY_COLOR_RENDER);
    
    return 0;
}