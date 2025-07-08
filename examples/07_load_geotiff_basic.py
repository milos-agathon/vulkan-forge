#!/usr/bin/env python3
"""
Simple GeoTIFF Loading Example for Vulkan-Forge Terrain System

This example demonstrates the basic usage of the terrain system:
1. Loading a GeoTIFF heightmap
2. Creating a basic terrain renderer  
3. Displaying basic terrain information

Requirements:
    pip install vulkan-forge rasterio numpy

Usage:
    python load_geotiff_basic.py path/to/heightmap.tif
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np

# Import vulkan-forge terrain system
from vulkan_forge.terrain import TerrainRenderer
from vulkan_forge.terrain_config import TerrainConfig

# Mock Vulkan context for this example
class MockVulkanContext:
    """Mock Vulkan context for demonstration purposes"""
    def __init__(self):
        self.device = None
        self.command_pool = None
        self.queue = None
        
    def create_buffer(self, data):
        return len(data)  # Return size as mock buffer ID
        
    def destroy_buffer(self, buffer_id):
        pass


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('terrain_basic.log')
        ]
    )


def print_terrain_info(renderer: TerrainRenderer):
    """Print detailed information about loaded terrain"""
    if not renderer.bounds:
        print("No terrain loaded!")
        return
    
    bounds = renderer.bounds
    tile_count = len(renderer.tiles)
    
    print("\n" + "="*60)
    print("TERRAIN INFORMATION")
    print("="*60)
    
    # Geographic bounds
    print(f"Geographic Bounds:")
    print(f"  X: {bounds.min_x:.6f} to {bounds.max_x:.6f}")
    print(f"  Y: {bounds.min_y:.6f} to {bounds.max_y:.6f}")
    print(f"  Width:  {bounds.max_x - bounds.min_x:.2f} units")
    print(f"  Height: {bounds.max_y - bounds.min_y:.2f} units")
    
    # Elevation info
    print(f"\nElevation:")
    print(f"  Min: {bounds.min_elevation:.1f}m")
    print(f"  Max: {bounds.max_elevation:.1f}m")
    print(f"  Range: {bounds.max_elevation - bounds.min_elevation:.1f}m")
    
    # Tile information
    print(f"\nTiles:")
    print(f"  Total tiles: {tile_count}")
    if tile_count > 0:
        # Calculate tile grid dimensions
        tile_ids = [tile.tile_id for tile in renderer.tiles]
        max_x = max(tid[0] for tid in tile_ids) + 1
        max_y = max(tid[1] for tid in tile_ids) + 1
        print(f"  Grid size: {max_x} x {max_y}")
        print(f"  Tile size: {renderer.config.tile_size} vertices")
        
        # Sample tile info
        sample_tile = renderer.tiles[0]
        tile_height, tile_width = sample_tile.heightmap.shape
        print(f"  Heightmap resolution: {tile_width} x {tile_height} per tile")
    
    # Configuration
    config = renderer.config
    print(f"\nConfiguration:")
    print(f"  Height scale: {config.height_scale}")
    print(f"  Max render distance: {config.max_render_distance}m")
    print(f"  Tessellation mode: {config.tessellation.mode.value}")
    print(f"  Tessellation level: {config.tessellation.base_level}")
    print(f"  LOD distances: {config.lod.distances}")


def analyze_heightmap_statistics(renderer: TerrainRenderer):
    """Analyze and display heightmap statistics"""
    if not renderer.tiles:
        return
    
    print("\n" + "="*60)
    print("HEIGHTMAP ANALYSIS")
    print("="*60)
    
    # Collect all heightmap data
    all_heights = []
    for tile in renderer.tiles:
        all_heights.extend(tile.heightmap.flatten())
    
    heights = np.array(all_heights)
    
    # Basic statistics
    print(f"Total pixels: {len(heights):,}")
    print(f"Mean elevation: {np.mean(heights):.1f}m")
    print(f"Std deviation: {np.std(heights):.1f}m")
    print(f"Median elevation: {np.median(heights):.1f}m")
    
    # Percentiles
    percentiles = [5, 25, 75, 95]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(heights, p)
        print(f"  {p:2d}th percentile: {value:.1f}m")
    
    # Elevation distribution
    print(f"\nElevation Distribution:")
    hist, bins = np.histogram(heights, bins=10)
    for i, (count, bin_start, bin_end) in enumerate(zip(hist, bins[:-1], bins[1:])):
        percent = (count / len(heights)) * 100
        bar = "█" * int(percent / 2)  # Scale bar length
        print(f"  {bin_start:7.1f}m - {bin_end:7.1f}m: {percent:5.1f}% {bar}")
    
    # Slope analysis (simplified)
    if len(renderer.tiles) > 0:
        sample_tile = renderer.tiles[len(renderer.tiles)//2]  # Middle tile
        heightmap = sample_tile.heightmap
        
        # Calculate gradients
        grad_x = np.gradient(heightmap, axis=1)
        grad_y = np.gradient(heightmap, axis=0)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        print(f"\nSlope Analysis (sample tile):")
        print(f"  Mean slope: {np.mean(slope_magnitude):.3f}")
        print(f"  Max slope: {np.max(slope_magnitude):.3f}")
        print(f"  Steep areas (>0.5): {np.sum(slope_magnitude > 0.5) / slope_magnitude.size * 100:.1f}%")


def estimate_performance(renderer: TerrainRenderer):
    """Estimate rendering performance characteristics"""
    print("\n" + "="*60)
    print("PERFORMANCE ESTIMATION")
    print("="*60)
    
    config = renderer.config
    tile_count = len(renderer.tiles)
    
    if tile_count == 0:
        print("No tiles to analyze!")
        return
    
    # Estimate geometry complexity
    vertices_per_tile = config.tile_size * config.tile_size
    triangles_per_tile = (config.tile_size - 1) * (config.tile_size - 1) * 2
    
    # With tessellation
    tess_multiplier = config.tessellation.base_level * config.tessellation.base_level
    tessellated_triangles_per_tile = triangles_per_tile * tess_multiplier
    
    print(f"Geometry Complexity:")
    print(f"  Base vertices per tile: {vertices_per_tile:,}")
    print(f"  Base triangles per tile: {triangles_per_tile:,}")
    print(f"  Total base triangles: {triangles_per_tile * tile_count:,}")
    print(f"  With tessellation (level {config.tessellation.base_level}): {tessellated_triangles_per_tile * tile_count:,}")
    
    # Memory estimation
    vertex_size = 5 * 4  # 5 floats (x,y,z,u,v) * 4 bytes
    index_size = 4  # 4 bytes per index
    
    vertex_memory_mb = (vertices_per_tile * vertex_size * tile_count) / (1024 * 1024)
    index_memory_mb = (triangles_per_tile * 3 * index_size * tile_count) / (1024 * 1024)
    total_geometry_mb = vertex_memory_mb + index_memory_mb
    
    print(f"\nMemory Usage (estimated):")
    print(f"  Vertex buffers: {vertex_memory_mb:.1f} MB")
    print(f"  Index buffers: {index_memory_mb:.1f} MB")
    print(f"  Total geometry: {total_geometry_mb:.1f} MB")
    print(f"  With tile cache limit: {config.memory.max_tile_cache_mb} MB")
    
    # Performance estimates for different GPUs
    gpu_configs = [
        ("RTX 4090", 1000000000, "Very High"),
        ("RTX 3070", 500000000, "High"),
        ("GTX 1660", 200000000, "Medium"),
        ("Integrated", 50000000, "Low")
    ]
    
    print(f"\nEstimated Performance (4K resolution):")
    print(f"  Target: {tessellated_triangles_per_tile * tile_count:,} triangles")
    
    for gpu_name, triangles_per_sec, quality in gpu_configs:
        if tessellated_triangles_per_tile * tile_count > 0:
            fps = triangles_per_sec / (tessellated_triangles_per_tile * tile_count)
            fps = min(fps, 300)  # Cap at reasonable max
            
            performance_level = "✓" if fps >= config.performance.target_fps else "⚠" if fps >= 30 else "✗"
            print(f"  {gpu_name:12}: {fps:6.1f} FPS {performance_level} ({quality} settings)")


def demo_camera_movement(renderer: TerrainRenderer):
    """Demonstrate camera movement and LOD updates"""
    print("\n" + "="*60)
    print("CAMERA MOVEMENT DEMO")
    print("="*60)
    
    if not renderer.bounds:
        print("No terrain loaded for camera demo!")
        return
    
    bounds = renderer.bounds
    
    # Define camera positions to test
    camera_positions = [
        # High altitude overview
        ((bounds.min_x + bounds.max_x) / 2, (bounds.min_y + bounds.max_y) / 2, bounds.max_elevation + 5000),
        # Medium altitude
        ((bounds.min_x + bounds.max_x) / 2, (bounds.min_y + bounds.max_y) / 2, bounds.max_elevation + 1000),
        # Low altitude - corner
        (bounds.min_x + (bounds.max_x - bounds.min_x) * 0.1, bounds.min_y + (bounds.max_y - bounds.min_y) * 0.1, bounds.max_elevation + 100),
        # Low altitude - center
        ((bounds.min_x + bounds.max_x) / 2, (bounds.min_y + bounds.max_y) / 2, bounds.max_elevation + 100),
    ]
    
    camera_names = ["High Altitude", "Medium Altitude", "Low Corner", "Low Center"]
    
    # Mock view and projection matrices
    view_matrix = np.eye(4)
    proj_matrix = np.eye(4)
    
    for name, position in zip(camera_names, camera_positions):
        print(f"\n{name} View:")
        print(f"  Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
        
        # Update camera (this would trigger LOD updates)
        renderer.update_camera(view_matrix, proj_matrix, np.array(position))
        
        # Analyze tile visibility and LOD
        visible_tiles = sum(1 for tile in renderer.tiles if tile.is_loaded)
        culled_tiles = len(renderer.tiles) - visible_tiles
        
        if len(renderer.tiles) > 0:
            lod_distribution = {}
            for tile in renderer.tiles:
                if tile.is_loaded:
                    lod = tile.lod_level
                    lod_distribution[lod] = lod_distribution.get(lod, 0) + 1
            
            print(f"  Visible tiles: {visible_tiles} / {len(renderer.tiles)}")
            print(f"  Culled tiles: {culled_tiles}")
            print(f"  LOD distribution: {lod_distribution}")
            
            # Estimate triangles for this view
            total_triangles = 0
            for tile in renderer.tiles:
                if tile.is_loaded:
                    tile_triangles = (renderer.config.tile_size - 1) ** 2 * 2
                    tess_level = renderer.config.tessellation.get_tessellation_level(100.0)  # Simplified
                    total_triangles += tile_triangles * (tess_level ** 2)
            
            print(f"  Estimated triangles: {total_triangles:,}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Basic GeoTIFF terrain loading example')
    parser.add_argument('geotiff_path', help='Path to GeoTIFF heightmap file')
    parser.add_argument('--config', choices=['high_performance', 'balanced', 'high_quality', 'mobile'], 
                       default='balanced', help='Configuration preset')
    parser.add_argument('--tile-size', type=int, default=256, help='Tile size in vertices')
    parser.add_argument('--height-scale', type=float, default=1.0, help='Height scaling factor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()
    
    # Validate input file
    geotiff_path = Path(args.geotiff_path)
    if not geotiff_path.exists():
        print(f"Error: GeoTIFF file not found: {geotiff_path}")
        sys.exit(1)
    
    print("Vulkan-Forge Terrain System - Basic GeoTIFF Loading Example")
    print("=" * 60)
    print(f"Loading: {geotiff_path}")
    print(f"Config preset: {args.config}")
    
    try:
        # Create terrain configuration
        config = TerrainConfig.from_preset(args.config)
        config.tile_size = args.tile_size
        config.height_scale = args.height_scale
        
        # Validate configuration
        issues = config.validate()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            print("Continuing with current configuration...\n")
        
        # Create mock Vulkan context
        vulkan_context = MockVulkanContext()
        
        # Create terrain renderer
        print("Initializing terrain renderer...")
        renderer = TerrainRenderer(vulkan_context, config)
        
        # Load GeoTIFF
        print(f"Loading GeoTIFF: {geotiff_path}")
        success = renderer.load_geotiff(geotiff_path)
        
        if not success:
            print("Failed to load GeoTIFF!")
            sys.exit(1)
        
        print("GeoTIFF loaded successfully!")
        
        # Display terrain information
        print_terrain_info(renderer)
        
        # Analyze heightmap
        analyze_heightmap_statistics(renderer)
        
        # Performance estimation
        estimate_performance(renderer)
        
        # Camera movement demo
        demo_camera_movement(renderer)
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETE")
        print("="*60)
        print("Next steps:")
        print("  1. Run terrain_performance.py for detailed benchmarking")
        print("  2. Run terrain_viewer.py for interactive exploration")
        print("  3. Try different configuration presets with --config")
        print("  4. Experiment with different tile sizes and height scales")
        
        # Cleanup
        renderer.cleanup()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()