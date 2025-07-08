#!/usr/bin/env python3
"""
Terrain Performance Test Example - Fixed Version

This example demonstrates terrain rendering performance with proper error handling
and fallback implementations when terrain modules are not available.
"""

import numpy as np
import time
import sys
from typing import List, Tuple, Optional, Any
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Global variables to store imported classes
TerrainRenderer = None
TerrainStreamer = None
TerrainConfig = None
TessellationConfig = None
TessellationMode = None
TERRAIN_AVAILABLE = False

# Try to import real terrain modules
try:
    from vulkan_forge.terrain import TerrainRenderer, TerrainStreamer
    from vulkan_forge.terrain_config import TerrainConfig, TessellationConfig, TessellationMode
    TERRAIN_AVAILABLE = True
    print("✓ Terrain modules loaded successfully")
except ImportError as e:
    print(f"⚠ Terrain modules not available: {e}")
    print("Using mock implementations for demonstration...")
    TERRAIN_AVAILABLE = False

# Create mock implementations if real ones aren't available
if not TERRAIN_AVAILABLE:
    # Mock TessellationMode enum
    class TessellationMode:
        DISABLED = "disabled"
        UNIFORM = "uniform"
        DISTANCE_BASED = "distance_based"
    
    # Mock TessellationConfig
    class TessellationConfig:
        def __init__(self, mode=None, base_level=8, max_level=64, min_level=1):
            self.mode = mode or TessellationMode.DISTANCE_BASED
            self.base_level = base_level
            self.max_level = max_level
            self.min_level = min_level
            self.near_distance = 100.0
            self.far_distance = 5000.0
            self.falloff_exponent = 1.5
            
        def get_tessellation_level(self, distance: float) -> int:
            """Mock tessellation level calculation."""
            if distance < self.near_distance:
                return self.max_level
            elif distance > self.far_distance:
                return self.min_level
            else:
                # Linear interpolation
                t = (distance - self.near_distance) / (self.far_distance - self.near_distance)
                return int(self.max_level * (1 - t) + self.min_level * t)
    
    # Mock LODConfig
    class LODConfig:
        def __init__(self):
            self.distances = [200.0, 500.0, 1000.0, 2000.0]
            self.screen_error_threshold = 2.0
            self.enable_morphing = True
    
    # Mock PerformanceConfig
    class PerformanceConfig:
        def __init__(self):
            self.target_fps = 144
            self.enable_multithreading = True
            self.worker_threads = 4
    
    # Mock TerrainConfig
    class TerrainConfig:
        def __init__(self):
            self.tile_size = 256
            self.height_scale = 1.0
            self.max_render_distance = 10000.0
            self.tessellation = TessellationConfig()
            self.lod = LODConfig()
            self.performance = PerformanceConfig()
        
        @classmethod
        def from_preset(cls, preset_name: str):
            """Create configuration from preset."""
            config = cls()
            
            if preset_name == 'high_performance':
                config.tessellation.base_level = 4
                config.tessellation.max_level = 16
                config.performance.target_fps = 144
            elif preset_name == 'balanced':
                config.tessellation.base_level = 8
                config.tessellation.max_level = 32
                config.performance.target_fps = 60
            elif preset_name == 'high_quality':
                config.tessellation.base_level = 16
                config.tessellation.max_level = 64
                config.performance.target_fps = 30
            elif preset_name == 'mobile':
                config.tessellation.base_level = 2
                config.tessellation.max_level = 8
                config.performance.target_fps = 30
            
            return config
    
    # Mock TerrainRenderer
    class TerrainRenderer:
        def __init__(self, config: Optional[TerrainConfig] = None):
            self.config = config or TerrainConfig()
            self.is_initialized = False
            
        def initialize(self, device: Any = None, allocator: Any = None) -> bool:
            """Mock initialization."""
            print("Mock TerrainRenderer initialized")
            self.is_initialized = True
            return True
            
        def render_heightfield(self, heightmap: np.ndarray, **kwargs) -> bool:
            """Mock rendering."""
            if not self.is_initialized:
                return False
            height, width = heightmap.shape
            print(f"Mock rendering heightfield: {width}x{height}")
            # Simulate some processing time
            time.sleep(0.001)
            return True
        
        def update_lod(self, camera_position: Tuple[float, float, float]) -> None:
            """Mock LOD update."""
            pass
            
        def cleanup(self) -> None:
            """Mock cleanup."""
            self.is_initialized = False
    
    # Mock TerrainStreamer
    class TerrainStreamer:
        def __init__(self, cache_size_mb: int = 512):
            self.cache_size_mb = cache_size_mb
            self.loaded_tiles = {}
            
        def load_tile(self, x: int, y: int, lod: int = 0) -> Optional[np.ndarray]:
            """Mock tile loading."""
            tile_key = f"{x}_{y}_{lod}"
            if tile_key not in self.loaded_tiles:
                tile_size = 64 >> lod if lod > 0 else 64
                self.loaded_tiles[tile_key] = np.random.random((tile_size, tile_size)).astype(np.float32)
            return self.loaded_tiles[tile_key]
        
        def update_streaming(self, camera_position: Tuple[float, float, float], 
                           view_distance: float = 1000.0) -> None:
            """Mock streaming update."""
            pass
        
        def get_memory_usage(self) -> dict:
            """Mock memory usage stats."""
            return {
                'loaded_tiles': len(self.loaded_tiles),
                'estimated_memory_mb': len(self.loaded_tiles) * 0.5,
                'cache_limit_mb': self.cache_size_mb,
                'utilization': 0.5
            }


class TerrainBenchmark:
    """Comprehensive terrain rendering benchmark suite."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {}
        self.test_configs = self.create_test_configurations()
        
    def create_test_configurations(self) -> List[Tuple[str, TerrainConfig]]:
        """Create different test configurations for benchmarking."""
        configs = []
        
        # Preset configurations
        presets = ['high_performance', 'balanced', 'high_quality', 'mobile']
        
        for preset in presets:
            try:
                config = TerrainConfig.from_preset(preset)
                configs.append((f"preset_{preset}", config))
            except Exception as e:
                print(f"Failed to create {preset} preset: {e}")
                # Create fallback config
                config = TerrainConfig()
                configs.append((f"fallback_{preset}", config))
        
        return configs
    
    def generate_test_heightmap(self, size: int, complexity: str = 'medium') -> np.ndarray:
        """Generate test heightmap with specified complexity."""
        if complexity == 'simple':
            # Simple sine wave pattern
            x = np.linspace(0, 4 * np.pi, size)
            y = np.linspace(0, 4 * np.pi, size)
            X, Y = np.meshgrid(x, y)
            heightmap = np.sin(X) * np.cos(Y)
            
        elif complexity == 'medium':
            # Multiple octaves of noise
            heightmap = np.zeros((size, size))
            for octave in range(4):
                freq = 2 ** octave
                amp = 1.0 / (2 ** octave)
                
                x = np.linspace(0, freq * np.pi, size)
                y = np.linspace(0, freq * np.pi, size)
                X, Y = np.meshgrid(x, y)
                
                heightmap += amp * np.sin(X) * np.cos(Y)
                
        elif complexity == 'complex':
            # High-frequency detail with multiple patterns
            heightmap = np.zeros((size, size))
            
            # Base terrain
            x = np.linspace(0, 2 * np.pi, size)
            y = np.linspace(0, 2 * np.pi, size)
            X, Y = np.meshgrid(x, y)
            heightmap += np.sin(X) * np.cos(Y)
            
            # Add noise
            noise = np.random.random((size, size)) * 0.1
            heightmap += noise
            
            # Add ridges
            ridge_x = np.sin(X * 4) * 0.3
            ridge_y = np.cos(Y * 4) * 0.3
            heightmap += ridge_x + ridge_y
            
        else:
            # Default: random heightmap
            heightmap = np.random.random((size, size))
        
        # Normalize to [0, 1] and convert to float32
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        return heightmap.astype(np.float32)
    
    def benchmark_rendering(self, config_name: str, config: TerrainConfig, 
                           heightmap_sizes: List[int], num_frames: int = 100) -> dict:
        """Benchmark terrain rendering performance."""
        print(f"\n🔬 Benchmarking {config_name}")
        print("-" * 50)
        
        # Initialize renderer
        renderer = TerrainRenderer(config)
        if not renderer.initialize():
            print(f"❌ Failed to initialize renderer for {config_name}")
            return {}
        
        results = {}
        
        for size in heightmap_sizes:
            print(f"Testing {size}x{size} heightmap...")
            
            # Generate test heightmap
            heightmap = self.generate_test_heightmap(size, 'medium')
            
            # Warmup
            for _ in range(5):
                renderer.render_heightfield(heightmap)
            
            # Benchmark
            frame_times = []
            start_time = time.perf_counter()
            
            for frame in range(num_frames):
                frame_start = time.perf_counter()
                
                success = renderer.render_heightfield(heightmap)
                if not success:
                    print(f"❌ Rendering failed at frame {frame}")
                    break
                
                frame_end = time.perf_counter()
                frame_times.append(frame_end - frame_start)
            
            total_time = time.perf_counter() - start_time
            
            # Calculate statistics
            if frame_times:
                avg_frame_time = np.mean(frame_times)
                min_frame_time = np.min(frame_times)
                max_frame_time = np.max(frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                
                results[size] = {
                    'avg_frame_time_ms': avg_frame_time * 1000,
                    'min_frame_time_ms': min_frame_time * 1000,
                    'max_frame_time_ms': max_frame_time * 1000,
                    'fps': fps,
                    'total_time_s': total_time,
                    'frames_rendered': len(frame_times)
                }
                
                print(f"  {size}x{size}: {fps:.1f} FPS (avg: {avg_frame_time*1000:.2f}ms)")
                
                # Check if target FPS is met
                target_fps = config.performance.target_fps
                if fps >= target_fps:
                    print(f"  ✅ Meets target {target_fps} FPS")
                else:
                    print(f"  ⚠️  Below target {target_fps} FPS")
            else:
                print(f"  ❌ No valid frames rendered for {size}x{size}")
        
        renderer.cleanup()
        return results
    
    def benchmark_streaming(self, config: TerrainConfig) -> dict:
        """Benchmark terrain streaming performance."""
        print(f"\n🌊 Benchmarking Terrain Streaming")
        print("-" * 50)
        
        streamer = TerrainStreamer(cache_size_mb=512)
        
        # Test different camera movement patterns
        results = {}
        
        # Static camera test
        print("Testing static camera...")
        start_time = time.perf_counter()
        
        camera_pos = (0, 0, 100)
        for _ in range(100):
            streamer.update_streaming(camera_pos, view_distance=1000.0)
        
        static_time = time.perf_counter() - start_time
        results['static_camera'] = {
            'time_s': static_time,
            'updates_per_second': 100 / static_time
        }
        
        # Moving camera test
        print("Testing moving camera...")
        start_time = time.perf_counter()
        
        for i in range(100):
            # Simulate camera movement
            x = i * 10
            y = i * 5
            z = 100 + np.sin(i * 0.1) * 20
            camera_pos = (x, y, z)
            streamer.update_streaming(camera_pos, view_distance=1000.0)
        
        moving_time = time.perf_counter() - start_time
        results['moving_camera'] = {
            'time_s': moving_time,
            'updates_per_second': 100 / moving_time
        }
        
        # Memory usage test
        memory_stats = streamer.get_memory_usage()
        results['memory'] = memory_stats
        
        print(f"Static camera: {results['static_camera']['updates_per_second']:.1f} updates/sec")
        print(f"Moving camera: {results['moving_camera']['updates_per_second']:.1f} updates/sec")
        print(f"Memory usage: {memory_stats['estimated_memory_mb']:.1f} MB ({memory_stats['loaded_tiles']} tiles)")
        
        return results
    
    def run_full_benchmark(self) -> dict:
        """Run complete benchmark suite."""
        print("🏁 Starting Terrain Performance Benchmark Suite")
        print("=" * 60)
        
        if not TERRAIN_AVAILABLE:
            print("⚠️  Using mock implementations - results are for demonstration only")
        
        all_results = {}
        
        # Heightmap sizes to test
        sizes_to_test = [128, 256, 512, 1024] if TERRAIN_AVAILABLE else [64, 128, 256]
        
        # Benchmark each configuration
        for config_name, config in self.test_configs:
            try:
                rendering_results = self.benchmark_rendering(
                    config_name, config, sizes_to_test, num_frames=50
                )
                
                streaming_results = self.benchmark_streaming(config)
                
                all_results[config_name] = {
                    'rendering': rendering_results,
                    'streaming': streaming_results
                }
                
            except Exception as e:
                print(f"❌ Benchmark failed for {config_name}: {e}")
                all_results[config_name] = {'error': str(e)}
        
        return all_results
    
    def print_summary(self, results: dict) -> None:
        """Print benchmark summary."""
        print("\n📊 Benchmark Summary")
        print("=" * 60)
        
        for config_name, config_results in results.items():
            if 'error' in config_results:
                print(f"❌ {config_name}: {config_results['error']}")
                continue
                
            print(f"\n🔧 {config_name}")
            print("-" * 30)
            
            # Rendering results
            if 'rendering' in config_results:
                rendering = config_results['rendering']
                for size, stats in rendering.items():
                    fps = stats['fps']
                    frame_time = stats['avg_frame_time_ms']
                    print(f"  {size}x{size}: {fps:.1f} FPS ({frame_time:.2f}ms)")
            
            # Streaming results
            if 'streaming' in config_results:
                streaming = config_results['streaming']
                if 'static_camera' in streaming:
                    static_ups = streaming['static_camera']['updates_per_second']
                    moving_ups = streaming['moving_camera']['updates_per_second']
                    memory_mb = streaming['memory']['estimated_memory_mb']
                    
                    print(f"  Streaming: {static_ups:.1f}/{moving_ups:.1f} UPS, {memory_mb:.1f}MB")


def main():
    """Main benchmark execution."""
    print("🌄 Vulkan-Forge Terrain Performance Test")
    print("=" * 60)
    
    # Quick functionality test
    print("\n🧪 Quick Functionality Test")
    print("-" * 30)
    
    # Test basic terrain functionality
    config = TerrainConfig.from_preset('balanced')
    renderer = TerrainRenderer(config)
    streamer = TerrainStreamer(cache_size_mb=256)
    
    if renderer.initialize():
        print("✅ TerrainRenderer initialized")
        
        # Test rendering
        test_heightmap = np.random.random((256, 256)).astype(np.float32)
        start_time = time.perf_counter()
        
        success = renderer.render_heightfield(test_heightmap)
        render_time = time.perf_counter() - start_time
        
        if success:
            print(f"✅ Rendered 256x256 heightmap in {render_time*1000:.2f}ms")
        else:
            print("❌ Rendering failed")
        
        renderer.cleanup()
    else:
        print("❌ Failed to initialize TerrainRenderer")
    
    # Test streaming
    camera_pos = (0, 0, 100)
    streamer.update_streaming(camera_pos)
    memory_stats = streamer.get_memory_usage()
    print(f"✅ Streaming test: {memory_stats['loaded_tiles']} tiles loaded")
    
    # Run full benchmark suite
    print("\n🚀 Running Full Benchmark Suite")
    print("-" * 30)
    
    benchmark = TerrainBenchmark()
    results = benchmark.run_full_benchmark()
    benchmark.print_summary(results)
    
    print("\n✅ Benchmark Complete!")
    print("\nFor production use:")
    print("- Install Vulkan SDK for hardware acceleration")
    print("- Enable multi-threading in terrain configuration") 
    print("- Tune tessellation levels based on target hardware")


if __name__ == "__main__":
    main()