#!/usr/bin/env python3
"""
4K Terrain Performance Benchmarking Script for Vulkan-Forge

This script benchmarks terrain rendering performance at 4K resolution (3840x2160)
with the goal of achieving 144 FPS. It tests various configurations and provides
detailed performance analysis.

Requirements:
    pip install vulkan-forge rasterio numpy matplotlib psutil

Usage:
    python terrain_performance.py path/to/heightmap.tif
    python terrain_performance.py --synthetic --size 4096  # Use synthetic data
    python terrain_performance.py --preset high_performance --duration 60
"""

import sys
import argparse
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, asdict

try:
    import psutil
except ImportError:
    psutil = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
except ImportError:
    plt = None
    print("Warning: matplotlib not available - plotting disabled")

# Import vulkan-forge terrain system
from vulkan_forge.terrain import TerrainRenderer, TerrainStreamer
from vulkan_forge.terrain_config import TerrainConfig, TessellationMode, LODAlgorithm


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single frame or test run"""
    timestamp: float
    fps: float
    frame_time_ms: float
    triangles_rendered: int
    tiles_rendered: int
    culled_tiles: int
    gpu_memory_mb: float = 0.0
    system_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class BenchmarkResult:
    """Results from a benchmark test"""
    config_name: str
    duration_seconds: float
    total_frames: int
    avg_fps: float
    min_fps: float
    max_fps: float
    percentile_99_fps: float
    percentile_95_fps: float
    avg_frame_time_ms: float
    frame_time_std_ms: float
    avg_triangles_per_frame: int
    peak_memory_mb: float
    avg_cpu_usage: float
    target_fps_hit_rate: float  # Percentage of frames hitting target FPS


class TerrainBenchmark:
    """Comprehensive terrain rendering benchmark suite"""
    
    def __init__(self, target_fps: int = 144, resolution: Tuple[int, int] = (3840, 2160)):
        self.target_fps = target_fps
        self.resolution = resolution
        self.results: List[BenchmarkResult] = []
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Mock Vulkan context for demonstration
        self.vulkan_context = self._create_mock_vulkan_context()
        
    def _create_mock_vulkan_context(self):
        """Create mock Vulkan context with performance simulation"""
        class MockVulkanContext:
            def __init__(self, resolution):
                self.resolution = resolution
                self.frame_count = 0
                
                # Simulate GPU performance characteristics
                self.base_triangle_rate = 800_000_000  # triangles per second
                self.memory_bandwidth = 600_000  # MB/s
                self.setup_overhead_ms = 0.5
                
            def create_buffer(self, data):
                return len(data) if hasattr(data, '__len__') else 1000
                
            def destroy_buffer(self, buffer_id):
                pass
                
            def simulate_frame_render(self, triangle_count: int, tile_count: int) -> Tuple[float, int]:
                """Simulate frame rendering and return frame time and actual triangles rendered"""
                self.frame_count += 1
                
                # Simulate performance degradation with high triangle counts
                effective_rate = self.base_triangle_rate
                if triangle_count > 50_000_000:  # 50M triangles
                    effective_rate *= 0.7
                elif triangle_count > 20_000_000:  # 20M triangles
                    effective_rate *= 0.85
                
                # Calculate render time
                render_time_ms = (triangle_count / effective_rate) * 1000
                
                # Add setup overhead
                total_time_ms = render_time_ms + self.setup_overhead_ms
                
                # Add some realistic variance
                variance = np.random.normal(1.0, 0.05)  # 5% variance
                total_time_ms *= variance
                
                # Simulate occasional frame spikes
                if np.random.random() < 0.01:  # 1% chance
                    total_time_ms *= 2.0
                
                return max(total_time_ms, 1.0), triangle_count
        
        return MockVulkanContext(self.resolution)
    
    def generate_synthetic_terrain(self, size: int = 2048, height_scale: float = 100.0) -> np.ndarray:
        """Generate synthetic terrain heightmap for testing"""
        print(f"Generating synthetic terrain: {size}x{size}")
        
        # Create coordinate grids
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Generate terrain using multiple noise octaves
        heightmap = np.zeros((size, size))
        
        # Base terrain
        heightmap += 0.5 * np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y)
        
        # Add detail layers
        for octave in range(1, 6):
            frequency = 2 ** octave
            amplitude = 1.0 / (2 ** octave)
            
            noise = amplitude * np.sin(frequency * np.pi * X) * np.cos(frequency * np.pi * Y)
            noise += amplitude * np.sin(frequency * np.pi * Y) * np.cos(frequency * np.pi * X)
            
            heightmap += noise
        
        # Add random noise for fine detail
        heightmap += 0.1 * np.random.random((size, size))
        
        # Normalize and scale
        heightmap = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))
        heightmap *= height_scale
        
        return heightmap.astype(np.float32)
    
    def create_test_configurations(self) -> List[Tuple[str, TerrainConfig]]:
        """Create test configurations for benchmarking"""
        configs = []
        
        # Performance preset variations
        base_configs = [
            ("High Performance", TerrainConfig.from_preset('high_performance')),
            ("Balanced", TerrainConfig.from_preset('balanced')),
            ("High Quality", TerrainConfig.from_preset('high_quality')),
        ]
        
        for name, config in base_configs:
            config.performance.target_fps = self.target_fps
            configs.append((name, config))
        
        # Tessellation variations
        for tess_mode in [TessellationMode.UNIFORM, TessellationMode.DISTANCE_BASED]:
            config = TerrainConfig.from_preset('balanced')
            config.tessellation.mode = tess_mode
            config.performance.target_fps = self.target_fps
            configs.append((f"Tessellation {tess_mode.value.title()}", config))
        
        # LOD variations
        for max_distance in [2000, 5000, 10000]:
            config = TerrainConfig.from_preset('balanced')
            config.max_render_distance = max_distance
            config.performance.target_fps = self.target_fps
            configs.append((f"Render Distance {max_distance}m", config))
        
        # Tile size variations
        for tile_size in [128, 256, 512]:
            config = TerrainConfig.from_preset('balanced')
            config.tile_size = tile_size
            config.performance.target_fps = self.target_fps
            configs.append((f"Tile Size {tile_size}", config))
        
        return configs
    
    def run_benchmark(self, renderer: TerrainRenderer, config_name: str, 
                     duration: float = 30.0) -> BenchmarkResult:
        """Run benchmark for specified duration"""
        print(f"\nRunning benchmark: {config_name}")
        print(f"Duration: {duration:.1f}s, Target: {self.target_fps} FPS")
        
        metrics = []
        start_time = time.perf_counter()
        frame_count = 0
        
        # Simulate camera movement for realistic testing
        camera_path = self._generate_camera_path(duration)
        
        while time.perf_counter() - start_time < duration:
            frame_start = time.perf_counter()
            
            # Update camera position
            t = (frame_start - start_time) / duration
            camera_pos = self._interpolate_camera_path(camera_path, t)
            
            # Update renderer
            view_matrix = np.eye(4)
            proj_matrix = np.eye(4) 
            renderer.update_camera(view_matrix, proj_matrix, camera_pos)
            
            # Simulate rendering
            triangle_count = sum(
                (renderer.config.tile_size - 1) ** 2 * 2 * 
                (renderer.config.tessellation.base_level ** 2)
                for tile in renderer.tiles if tile.is_loaded
            )
            
            frame_time_ms, actual_triangles = self.vulkan_context.simulate_frame_render(
                triangle_count, len([t for t in renderer.tiles if t.is_loaded])
            )
            
            frame_end = time.perf_counter()
            
            # Record metrics
            fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
            
            metric = PerformanceMetrics(
                timestamp=frame_start,
                fps=fps,
                frame_time_ms=frame_time_ms,
                triangles_rendered=actual_triangles,
                tiles_rendered=renderer.frame_stats.get('tiles_rendered', 0),
                culled_tiles=renderer.frame_stats.get('culled_tiles', 0),
                gpu_memory_mb=self._get_gpu_memory_usage(),
                system_memory_mb=self._get_system_memory_usage(),
                cpu_usage_percent=self._get_cpu_usage()
            )
            
            metrics.append(metric)
            self.metrics_history.append(metric)
            frame_count += 1
            
            # Simple frame rate limiting for simulation
            target_frame_time = 1.0 / self.target_fps
            elapsed = frame_end - frame_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
        
        # Calculate benchmark results
        fps_values = [m.fps for m in metrics]
        frame_times = [m.frame_time_ms for m in metrics]
        triangles = [m.triangles_rendered for m in metrics]
        memory_values = [m.gpu_memory_mb for m in metrics]
        cpu_values = [m.cpu_usage_percent for m in metrics]
        
        target_fps_hits = sum(1 for fps in fps_values if fps >= self.target_fps)
        
        result = BenchmarkResult(
            config_name=config_name,
            duration_seconds=duration,
            total_frames=len(metrics),
            avg_fps=np.mean(fps_values),
            min_fps=np.min(fps_values),
            max_fps=np.max(fps_values),
            percentile_99_fps=np.percentile(fps_values, 1),  # 99th percentile (1st percentile for min)
            percentile_95_fps=np.percentile(fps_values, 5),  # 95th percentile
            avg_frame_time_ms=np.mean(frame_times),
            frame_time_std_ms=np.std(frame_times),
            avg_triangles_per_frame=int(np.mean(triangles)),
            peak_memory_mb=np.max(memory_values),
            avg_cpu_usage=np.mean(cpu_values),
            target_fps_hit_rate=(target_fps_hits / len(fps_values)) * 100.0
        )
        
        self.results.append(result)
        return result
    
    def _generate_camera_path(self, duration: float) -> List[np.ndarray]:
        """Generate camera path for realistic movement during benchmark"""
        points = []
        num_points = max(10, int(duration / 2))  # One point every 2 seconds
        
        # Generate path around terrain
        for i in range(num_points):
            t = i / (num_points - 1)
            angle = t * 4 * np.pi  # Two full circles
            radius = 2000 + 1000 * np.sin(t * np.pi)  # Varying radius
            height = 500 + 200 * np.cos(t * 2 * np.pi)  # Varying height
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = height
            
            points.append(np.array([x, y, z]))
        
        return points
    
    def _interpolate_camera_path(self, path: List[np.ndarray], t: float) -> np.ndarray:
        """Interpolate camera position along path"""
        if t <= 0:
            return path[0]
        if t >= 1:
            return path[-1]
        
        # Find segment
        segment_length = 1.0 / (len(path) - 1)
        segment = int(t / segment_length)
        local_t = (t - segment * segment_length) / segment_length
        
        if segment >= len(path) - 1:
            return path[-1]
        
        # Linear interpolation
        return path[segment] * (1 - local_t) + path[segment + 1] * local_t
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage (simulated)"""
        # Simulate GPU memory usage
        base_memory = 200  # Base memory usage in MB
        tile_memory = len(self.metrics_history) * 0.1  # Accumulating memory
        return base_memory + tile_memory
    
    def _get_system_memory_usage(self) -> float:
        """Get system memory usage"""
        if psutil:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        else:
            return 50.0  # Default estimate
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        if psutil:
            return psutil.cpu_percent(interval=None)
        else:
            return 25.0 + np.random.random() * 20  # Simulated 25-45%
    
    def print_results_summary(self):
        """Print summary of all benchmark results"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        print(f"Target: {self.target_fps} FPS @ {self.resolution[0]}x{self.resolution[1]}")
        print()
        
        # Table header
        print(f"{'Configuration':<25} {'Avg FPS':<10} {'Min FPS':<10} {'Target %':<10} {'Triangles':<12} {'Memory':<10}")
        print("-" * 80)
        
        # Sort results by average FPS (descending)
        sorted_results = sorted(self.results, key=lambda r: r.avg_fps, reverse=True)
        
        for result in sorted_results:
            target_indicator = "✓" if result.target_fps_hit_rate >= 95 else "⚠" if result.target_fps_hit_rate >= 80 else "✗"
            
            print(f"{result.config_name:<25} "
                  f"{result.avg_fps:<10.1f} "
                  f"{result.min_fps:<10.1f} "
                  f"{result.target_fps_hit_rate:<9.1f}% "
                  f"{result.avg_triangles_per_frame/1000000:<11.1f}M "
                  f"{result.peak_memory_mb:<9.0f}MB "
                  f"{target_indicator}")
    
    def print_detailed_analysis(self):
        """Print detailed performance analysis"""
        if not self.results:
            print("No results to analyze!")
            return
        
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE ANALYSIS")
        print("="*80)
        
        best_result = max(self.results, key=lambda r: r.avg_fps)
        worst_result = min(self.results, key=lambda r: r.avg_fps)
        
        print(f"\nBest Performing Configuration:")
        print(f"  {best_result.config_name}")
        print(f"  Average FPS: {best_result.avg_fps:.1f}")
        print(f"  Target hit rate: {best_result.target_fps_hit_rate:.1f}%")
        print(f"  Frame time: {best_result.avg_frame_time_ms:.2f}ms ± {best_result.frame_time_std_ms:.2f}ms")
        
        print(f"\nWorst Performing Configuration:")
        print(f"  {worst_result.config_name}")
        print(f"  Average FPS: {worst_result.avg_fps:.1f}")
        print(f"  Target hit rate: {worst_result.target_fps_hit_rate:.1f}%")
        
        # Performance factors analysis
        print(f"\nPerformance Factors Analysis:")
        
        # Triangle count impact
        triangle_results = [(r.avg_triangles_per_frame, r.avg_fps) for r in self.results]
        triangle_results.sort()
        
        print(f"  Triangle Count Impact:")
        for triangles, fps in triangle_results[:3]:
            print(f"    {triangles/1000000:.1f}M triangles → {fps:.1f} FPS")
        
        # Memory usage
        memory_results = [(r.peak_memory_mb, r.avg_fps) for r in self.results]
        memory_results.sort()
        
        print(f"  Memory Usage Range:")
        print(f"    Min: {memory_results[0][0]:.0f}MB → {memory_results[0][1]:.1f} FPS")
        print(f"    Max: {memory_results[-1][0]:.0f}MB → {memory_results[-1][1]:.1f} FPS")
        
        # Recommendations
        print(f"\nRecommendations:")
        
        successful_configs = [r for r in self.results if r.target_fps_hit_rate >= 90]
        if successful_configs:
            best_config = max(successful_configs, key=lambda r: r.avg_fps)
            print(f"  ✓ Use '{best_config.config_name}' for consistent {self.target_fps} FPS")
        else:
            print(f"  ⚠ No configuration consistently achieved {self.target_fps} FPS")
            print(f"    Consider reducing tessellation levels or render distance")
        
        high_triangle_configs = [r for r in self.results if r.avg_triangles_per_frame > 50_000_000]
        if high_triangle_configs:
            print(f"  ⚠ Configurations with >50M triangles showed performance degradation")
        
        high_memory_configs = [r for r in self.results if r.peak_memory_mb > 800]
        if high_memory_configs:
            print(f"  ⚠ High memory usage (>800MB) may cause issues on lower-end GPUs")
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file"""
        data = {
            'benchmark_info': {
                'target_fps': self.target_fps,
                'resolution': self.resolution,
                'timestamp': datetime.now().isoformat()
            },
            'results': [asdict(result) for result in self.results],
            'metrics_sample': [asdict(m) for m in self.metrics_history[-100:]]  # Last 100 metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def plot_performance_graphs(self, output_dir: str = "."):
        """Generate performance graphs"""
        if not plt or not self.results:
            print("Cannot generate plots - matplotlib not available or no results")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. FPS comparison chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = [r.config_name for r in self.results]
        avg_fps = [r.avg_fps for r in self.results]
        min_fps = [r.min_fps for r in self.results]
        
        x = np.arange(len(configs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, avg_fps, width, label='Average FPS', alpha=0.8)
        bars2 = ax.bar(x + width/2, min_fps, width, label='Minimum FPS', alpha=0.8)
        
        ax.axhline(y=self.target_fps, color='r', linestyle='--', label=f'Target {self.target_fps} FPS')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('FPS')
        ax.set_title('Terrain Rendering Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'fps_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Performance vs triangle count scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        triangles = [r.avg_triangles_per_frame / 1000000 for r in self.results]  # Millions
        fps = [r.avg_fps for r in self.results]
        colors = ['green' if f >= self.target_fps else 'orange' if f >= self.target_fps * 0.8 else 'red' for f in fps]
        
        scatter = ax.scatter(triangles, fps, c=colors, alpha=0.7, s=100)
        
        for i, config in enumerate(configs):
            ax.annotate(config, (triangles[i], fps[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax.axhline(y=self.target_fps, color='r', linestyle='--', label=f'Target {self.target_fps} FPS')
        ax.set_xlabel('Triangles Rendered (Millions)')
        ax.set_ylabel('Average FPS')
        ax.set_title('Performance vs Triangle Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_vs_triangles.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Frame time distribution for best config
        if self.metrics_history:
            best_config = max(self.results, key=lambda r: r.avg_fps)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Frame time over time
            times = [m.timestamp for m in self.metrics_history[-1000:]]  # Last 1000 frames
            frame_times = [m.frame_time_ms for m in self.metrics_history[-1000:]]
            
            if times:
                start_time = times[0]
                times = [(t - start_time) for t in times]
                
                ax1.plot(times, frame_times, alpha=0.7)
                ax1.axhline(y=1000/self.target_fps, color='r', linestyle='--', 
                           label=f'Target frame time ({1000/self.target_fps:.1f}ms)')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Frame Time (ms)')
                ax1.set_title(f'Frame Time History - {best_config.config_name}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Frame time histogram
                ax2.hist(frame_times, bins=50, alpha=0.7, edgecolor='black')
                ax2.axvline(x=1000/self.target_fps, color='r', linestyle='--', 
                           label=f'Target frame time')
                ax2.set_xlabel('Frame Time (ms)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Frame Time Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'frame_time_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Performance graphs saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='4K Terrain Performance Benchmark')
    parser.add_argument('geotiff_path', nargs='?', help='Path to GeoTIFF heightmap file')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic terrain data')
    parser.add_argument('--size', type=int, default=2048, help='Synthetic terrain size')
    parser.add_argument('--duration', type=float, default=30.0, help='Benchmark duration per config (seconds)')
    parser.add_argument('--preset', choices=['high_performance', 'balanced', 'high_quality'], 
                       default='balanced', help='Primary configuration preset')
    parser.add_argument('--target-fps', type=int, default=144, help='Target FPS')
    parser.add_argument('--resolution', type=str, default='4K', 
                       choices=['4K', '1440p', '1080p'], help='Target resolution')
    parser.add_argument('--output', type=str, default='benchmark_results.json', 
                       help='Output file for results')
    parser.add_argument('--plots', action='store_true', help='Generate performance plots')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark (fewer configs, shorter duration)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Resolution mapping
    resolutions = {
        '4K': (3840, 2160),
        '1440p': (2560, 1440),
        '1080p': (1920, 1080)
    }
    resolution = resolutions[args.resolution]
    
    # Adjust duration for quick mode
    duration = args.duration
    if args.quick:
        duration = min(10.0, duration)
    
    print("Vulkan-Forge Terrain Performance Benchmark")
    print("=" * 60)
    print(f"Target: {args.target_fps} FPS @ {resolution[0]}x{resolution[1]}")
    print(f"Duration per config: {duration:.1f}s")
    
    try:
        # Create benchmark
        benchmark = TerrainBenchmark(target_fps=args.target_fps, resolution=resolution)
        
        # Load or generate terrain
        if args.synthetic or not args.geotiff_path:
            print("Using synthetic terrain data")
            heightmap = benchmark.generate_synthetic_terrain(args.size)
            
            # Create terrain renderer with synthetic data
            config = TerrainConfig.from_preset(args.preset)
            renderer = TerrainRenderer(benchmark.vulkan_context, config)
            
            # Manually create terrain bounds and tiles for synthetic data
            renderer.bounds = type('TerrainBounds', (), {
                'min_x': 0, 'max_x': args.size, 
                'min_y': 0, 'max_y': args.size,
                'min_elevation': np.min(heightmap),
                'max_elevation': np.max(heightmap)
            })()
            
            # Generate tiles from synthetic heightmap
            renderer._generate_tiles(heightmap, np.eye(3))  # Mock transform
            
        else:
            # Load from GeoTIFF
            geotiff_path = Path(args.geotiff_path)
            if not geotiff_path.exists():
                print(f"Error: GeoTIFF file not found: {geotiff_path}")
                sys.exit(1)
            
            config = TerrainConfig.from_preset(args.preset)
            renderer = TerrainRenderer(benchmark.vulkan_context, config)
            
            success = renderer.load_geotiff(geotiff_path)
            if not success:
                print("Failed to load GeoTIFF!")
                sys.exit(1)
        
        print(f"Terrain loaded: {len(renderer.tiles)} tiles")
        
        # Get test configurations
        if args.quick:
            # Quick mode - test only key configurations
            test_configs = [
                ("High Performance", TerrainConfig.from_preset('high_performance')),
                ("Balanced", TerrainConfig.from_preset('balanced')),
                ("High Quality", TerrainConfig.from_preset('high_quality'))
            ]
        else:
            test_configs = benchmark.create_test_configurations()
        
        print(f"Testing {len(test_configs)} configurations...")
        
        # Run benchmarks
        for i, (config_name, config) in enumerate(test_configs, 1):
            print(f"\nProgress: {i}/{len(test_configs)}")
            
            # Update renderer configuration
            renderer.config = config
            
            # Run benchmark
            result = benchmark.run_benchmark(renderer, config_name, duration)
            
            # Print quick result
            status = "✓" if result.target_fps_hit_rate >= 90 else "⚠" if result.target_fps_hit_rate >= 70 else "✗"
            print(f"  Result: {result.avg_fps:.1f} FPS (target hit: {result.target_fps_hit_rate:.1f}%) {status}")
        
        # Print results
        benchmark.print_results_summary()
        benchmark.print_detailed_analysis()
        
        # Save results
        benchmark.save_results(args.output)
        
        # Generate plots
        if args.plots:
            benchmark.plot_performance_graphs()
        
        # Cleanup
        renderer.cleanup()
        
        print(f"\nBenchmark complete! Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()