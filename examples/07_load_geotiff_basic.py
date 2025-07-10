#!/usr/bin/env python3
"""
Simple GeoTIFF Loading Example for Vulkan-Forge Terrain System

Fixed version that works with the actual TerrainRenderer API
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from types import SimpleNamespace

# Import vulkan-forge terrain system
from vulkan_forge.terrain import TerrainRenderer
from vulkan_forge.terrain_config import TerrainConfig

# Import GeoTIFF loader if available
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. Install with: pip install rasterio")


def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_synthetic_heightmap(size=512):
    """Create synthetic terrain data for demonstration"""
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Create interesting terrain
    Z = np.sin(5 * X) * np.cos(5 * Y) * 0.1
    Z += np.exp(-(X**2 + Y**2) / 0.5) * 0.5
    Z += np.sin(X * 3) * 0.05
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * 1000
    
    return Z.astype(np.float32)


def load_geotiff(filepath):
    """Load a GeoTIFF file and return heightmap data and bounds"""
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required to load GeoTIFF files")
    
    with rasterio.open(filepath) as src:
        # Read the first band as height data
        heightmap = src.read(1).astype(np.float32)
        
        # Get bounds
        bounds = SimpleNamespace(
            min_x=src.bounds.left,
            max_x=src.bounds.right,
            min_y=src.bounds.bottom,
            max_y=src.bounds.top,
            min_elevation=float(np.min(heightmap)),
            max_elevation=float(np.max(heightmap))
        )
        
        # Get metadata
        metadata = {
            'crs': str(src.crs) if src.crs else 'Unknown',
            'width': src.width,
            'height': src.height,
            'transform': src.transform
        }
        
        return heightmap, bounds, metadata


def print_terrain_info(heightmap, bounds, metadata=None):
    """Print detailed information about terrain data"""
    print("\n" + "="*60)
    print("TERRAIN INFORMATION")
    print("="*60)
    
    # Data dimensions
    print(f"Data Size: {heightmap.shape[1]}x{heightmap.shape[0]} pixels")
    
    # Geographic bounds
    print(f"\nGeographic Bounds:")
    print(f"  X: {bounds.min_x:.6f} to {bounds.max_x:.6f}")
    print(f"  Y: {bounds.min_y:.6f} to {bounds.max_y:.6f}")
    print(f"  Width:  {bounds.max_x - bounds.min_x:.2f} units")
    print(f"  Height: {bounds.max_y - bounds.min_y:.2f} units")
    
    # Elevation info
    print(f"\nElevation:")
    print(f"  Min: {bounds.min_elevation:.2f} meters")
    print(f"  Max: {bounds.max_elevation:.2f} meters")
    print(f"  Range: {bounds.max_elevation - bounds.min_elevation:.2f} meters")
    
    # Metadata if available
    if metadata:
        print(f"\nMetadata:")
        print(f"  CRS: {metadata.get('crs', 'Unknown')}")


def main():
    """Main entry point"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load and display GeoTIFF terrain data"
    )
    
    parser.add_argument(
        'geotiff_path',
        nargs='?',
        help='Path to GeoTIFF file (optional, uses synthetic data if not provided)'
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic terrain data'
    )
    
    parser.add_argument(
        '--config',
        choices=['high_performance', 'balanced', 'high_quality', 'mobile'],
        default='balanced',
        help='Terrain configuration preset'
    )
    
    parser.add_argument(
        '--tile-size',
        type=int,
        default=256,
        help='Tile size in vertices'
    )
    
    parser.add_argument(
        '--height-scale',
        type=float,
        default=1.0,
        help='Height scaling factor'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Create terrain configuration
    config = TerrainConfig.from_preset(args.config)
    config.tile_size = args.tile_size
    config.height_scale = args.height_scale
    
    # Create terrain renderer
    renderer = TerrainRenderer(config)
    
    # Initialize the renderer
    if not renderer.initialize():
        logging.error("Failed to initialize terrain renderer")
        return 1
    
    # Load or create heightmap data
    if args.synthetic or args.geotiff_path is None:
        print("Using synthetic terrain data...")
        
        heightmap = create_synthetic_heightmap()
        bounds = SimpleNamespace(
            min_x=-122.5, 
            max_x=-122.0,
            min_y=37.5, 
            max_y=38.0,
            min_elevation=float(heightmap.min()),
            max_elevation=float(heightmap.max())
        )
        metadata = None
        
    else:
        # Load actual GeoTIFF
        geotiff_path = Path(args.geotiff_path)
        
        if not geotiff_path.exists():
            logging.error(f"File not found: {geotiff_path}")
            return 1
        
        logging.info(f"Loading GeoTIFF: {geotiff_path}")
        
        try:
            heightmap, bounds, metadata = load_geotiff(str(geotiff_path))
            logging.info("Successfully loaded GeoTIFF data")
            
        except ImportError:
            logging.error("rasterio is required to load GeoTIFF files")
            logging.info("Install with: pip install rasterio")
            logging.info("Or use --synthetic flag for synthetic data")
            return 1
        except Exception as e:
            logging.error(f"Error loading GeoTIFF: {e}")
            return 1
    
    # Print terrain information
    print_terrain_info(heightmap, bounds, metadata)
    
    # Demonstrate rendering the heightfield
    print("\nAttempting to render heightfield...")
    success = renderer.render_heightfield(heightmap)
    
    if success:
        print("✓ Heightfield rendered successfully")
    else:
        print("✗ Failed to render heightfield")
    
    # Display renderer configuration
    print("\n" + "="*60)
    print("RENDERER CONFIGURATION")
    print("="*60)
    print(f"Preset: {args.config}")
    print(f"Tile Size: {config.tile_size}")
    print(f"Height Scale: {config.height_scale}")
    print(f"Max Render Distance: {config.max_render_distance}")
    print(f"Tessellation Mode: {config.tessellation.mode.value}")
    print(f"Target FPS: {config.performance.target_fps}")
    
    # Demonstrate LOD calculation
    print("\n" + "="*60)
    print("LOD TESSELLATION LEVELS")
    print("="*60)
    distances = [100, 500, 1000, 2500, 5000, 10000]
    for distance in distances:
        level = config.tessellation.get_tessellation_level(distance)
        print(f"  Distance {distance:5d}m → Tessellation Level: {level}")
    
    # Clean up
    renderer.cleanup()
    
    logging.info("Terrain loading example completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
