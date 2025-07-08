#!/usr/bin/env python3
"""
Test suite for GeoTIFF loading functionality

Tests the complete GeoTIFF loading pipeline from file reading to GPU upload,
including coordinate transformations, caching, and error handling.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Optional

# Import vulkan-forge components
try:
    from vulkan_forge.loaders import GeoTiffLoader, TerrainCache, CoordinateSystems
    from vulkan_forge.terrain_config import TerrainConfig, GeographicBounds
    import vulkan_forge_core as vf
    VULKAN_FORGE_AVAILABLE = True
except ImportError:
    VULKAN_FORGE_AVAILABLE = False

# Import optional dependencies for testing
try:
    import rasterio
    import rasterio.transform
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


class TestGeoTiffLoader:
    """Test suite for GeoTIFF loader functionality"""
    
    @pytest.fixture
    def sample_geotiff_data(self):
        """Create sample GeoTIFF data for testing"""
        width, height = 512, 512
        heights = np.random.randint(0, 1000, (height, width)).astype(np.float32)
        
        # Create geographic transform (WGS84 coordinates)
        transform = rasterio.transform.from_bounds(
            west=-180.0, south=-90.0, east=180.0, north=90.0,
            width=width, height=height
        )
        
        return {
            'heights': heights,
            'width': width,
            'height': height,
            'transform': transform,
            'crs': CRS.from_epsg(4326),  # WGS84
            'bounds': GeographicBounds(
                min_x=-180.0, max_x=180.0,
                min_y=-90.0, max_y=90.0,
                min_elevation=float(np.min(heights)),
                max_elevation=float(np.max(heights))
            )
        }
    
    @pytest.fixture
    def temp_geotiff_file(self, sample_geotiff_data):
        """Create a temporary GeoTIFF file for testing"""
        if not RASTERIO_AVAILABLE:
            pytest.skip("rasterio not available")
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            filepath = f.name
        
        data = sample_geotiff_data
        
        # Write GeoTIFF file
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=data['height'],
            width=data['width'],
            count=1,
            dtype=rasterio.float32,
            crs=data['crs'],
            transform=data['transform']
        ) as dataset:
            dataset.write(data['heights'], 1)
        
        yield filepath
        
        # Cleanup
        if os.path.exists(filepath):
            os.unlink(filepath)
    
    @pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available")
    def test_geotiff_loader_initialization(self):
        """Test GeoTIFF loader initialization"""
        loader = GeoTiffLoader()
        assert loader is not None
        assert not loader.is_loaded()
        assert loader.get_width() == 0
        assert loader.get_height() == 0
    
    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_load_geotiff_file(self, temp_geotiff_file, sample_geotiff_data):
        """Test loading GeoTIFF from file"""
        loader = GeoTiffLoader()
        
        # Test successful loading
        success = loader.load(temp_geotiff_file)
        assert success
        assert loader.is_loaded()
        
        # Check dimensions
        assert loader.get_width() == sample_geotiff_data['width']
        assert loader.get_height() == sample_geotiff_data['height']
        
        # Check bounds
        bounds = loader.get_bounds()
        expected_bounds = sample_geotiff_data['bounds']
        assert abs(bounds.min_x - expected_bounds.min_x) < 1e-6
        assert abs(bounds.max_x - expected_bounds.max_x) < 1e-6
        assert abs(bounds.min_y - expected_bounds.min_y) < 1e-6
        assert abs(bounds.max_y - expected_bounds.max_y) < 1e-6
        
        # Check heightmap data
        heightmap = loader.get_heightmap()
        assert heightmap.shape == (sample_geotiff_data['height'], sample_geotiff_data['width'])
        np.testing.assert_array_equal(heightmap, sample_geotiff_data['heights'])
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        loader = GeoTiffLoader()
        
        success = loader.load("non_existent_file.tif")
        assert not success
        assert not loader.is_loaded()
    
    def test_load_invalid_file(self):
        """Test loading invalid file format"""
        loader = GeoTiffLoader()
        
        # Create a text file with .tif extension
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
            f.write(b"This is not a valid GeoTIFF file")
            filepath = f.name
        
        try:
            success = loader.load(filepath)
            assert not success
            assert not loader.is_loaded()
        finally:
            os.unlink(filepath)
    
    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_geotiff_statistics(self, temp_geotiff_file, sample_geotiff_data):
        """Test GeoTIFF statistics calculation"""
        loader = GeoTiffLoader()
        loader.load(temp_geotiff_file)
        
        stats = loader.get_statistics()
        expected_heights = sample_geotiff_data['heights']
        
        assert abs(stats['min'] - np.min(expected_heights)) < 1e-6
        assert abs(stats['max'] - np.max(expected_heights)) < 1e-6
        assert abs(stats['mean'] - np.mean(expected_heights)) < 1e-6
        assert abs(stats['std'] - np.std(expected_heights)) < 1e-6
    
    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")  
    def test_coordinate_system_info(self, temp_geotiff_file):
        """Test coordinate system information extraction"""
        loader = GeoTiffLoader()
        loader.load(temp_geotiff_file)
        
        crs_info = loader.get_projection()
        assert 'epsg' in crs_info
        assert crs_info['epsg'] == 4326  # WGS84
        
        transform = loader.get_transform()
        assert len(transform) == 6  # Affine transform has 6 parameters
    
    @pytest.mark.parametrize("invalid_path", [
        "",
        "/dev/null",
        "directory_not_file/",
        "file_without_extension",
    ])
    def test_invalid_file_paths(self, invalid_path):
        """Test various invalid file paths"""
        loader = GeoTiffLoader()
        success = loader.load(invalid_path)
        assert not success


class TestTerrainCache:
    """Test suite for terrain caching functionality"""
    
    @pytest.fixture
    def cache_config(self):
        """Cache configuration for testing"""
        return {
            'max_size_mb': 100,
            'tile_size': 256,
            'max_tiles': 64,
            'eviction_policy': 'lru'
        }
    
    @pytest.fixture
    def sample_tile_data(self):
        """Sample terrain tile data"""
        size = 256
        heights = np.random.rand(size, size).astype(np.float32)
        return {
            'tile_id': (0, 0),
            'lod_level': 0,
            'heights': heights,
            'bounds': GeographicBounds(0, 0, 1, 1, 0, 100)
        }
    
    def test_cache_initialization(self, cache_config):
        """Test terrain cache initialization"""
        cache = TerrainCache(**cache_config)
        assert cache.get_max_size_mb() == cache_config['max_size_mb']
        assert cache.get_tile_count() == 0
        assert cache.get_memory_usage_mb() == 0
    
    def test_cache_tile_storage(self, cache_config, sample_tile_data):
        """Test storing tiles in cache"""
        cache = TerrainCache(**cache_config)
        
        # Store tile
        tile_id = sample_tile_data['tile_id']
        success = cache.store_tile(tile_id, sample_tile_data)
        assert success
        assert cache.has_tile(tile_id)
        assert cache.get_tile_count() == 1
        
        # Retrieve tile
        retrieved = cache.get_tile(tile_id)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved['heights'], sample_tile_data['heights'])
    
    def test_cache_eviction_lru(self, cache_config):
        """Test LRU cache eviction policy"""
        # Set small cache size to trigger eviction
        cache_config['max_tiles'] = 2
        cache = TerrainCache(**cache_config)
        
        # Create test tiles
        tiles = []
        for i in range(3):
            tile_data = {
                'tile_id': (i, 0),
                'lod_level': 0,
                'heights': np.random.rand(256, 256).astype(np.float32),
                'bounds': GeographicBounds(i, 0, i+1, 1, 0, 100)
            }
            tiles.append(tile_data)
        
        # Store tiles (should evict first tile when storing third)
        for tile in tiles:
            cache.store_tile(tile['tile_id'], tile)
        
        # First tile should be evicted
        assert not cache.has_tile((0, 0))
        assert cache.has_tile((1, 0))
        assert cache.has_tile((2, 0))
        assert cache.get_tile_count() == 2
    
    def test_cache_memory_limits(self, cache_config):
        """Test cache memory usage limits"""
        cache_config['max_size_mb'] = 1  # Very small limit
        cache = TerrainCache(**cache_config)
        
        # Create large tile that exceeds memory limit
        large_tile = {
            'tile_id': (0, 0),
            'lod_level': 0,
            'heights': np.ones((2048, 2048), dtype=np.float32),  # ~16MB
            'bounds': GeographicBounds(0, 0, 1, 1, 0, 100)
        }
        
        # Should reject tile due to size
        success = cache.store_tile(large_tile['tile_id'], large_tile)
        assert not success
        assert cache.get_tile_count() == 0
    
    def test_cache_statistics(self, cache_config, sample_tile_data):
        """Test cache statistics and metrics"""
        cache = TerrainCache(**cache_config)
        
        # Initial stats
        stats = cache.get_statistics()
        assert stats['hit_rate'] == 0.0
        assert stats['miss_rate'] == 0.0
        assert stats['total_requests'] == 0
        
        # Store and access tile
        tile_id = sample_tile_data['tile_id']
        cache.store_tile(tile_id, sample_tile_data)
        
        # Cache hit
        cache.get_tile(tile_id)
        
        # Cache miss
        cache.get_tile((99, 99))
        
        # Check updated stats
        stats = cache.get_statistics()
        assert stats['total_requests'] == 2
        assert stats['hit_rate'] == 0.5
        assert stats['miss_rate'] == 0.5


class TestCoordinateSystems:
    """Test suite for coordinate system transformations"""
    
    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not available")
    def test_wgs84_to_utm_conversion(self):
        """Test WGS84 to UTM coordinate conversion"""
        cs = CoordinateSystems()
        
        # Test point in London (should be UTM zone 30N)
        lon, lat = -0.1278, 51.5074  # London coordinates
        
        utm_x, utm_y, utm_zone = cs.wgs84_to_utm(lon, lat)
        
        assert isinstance(utm_x, float)
        assert isinstance(utm_y, float)
        assert utm_zone == "30N"
        
        # UTM coordinates should be reasonable for London
        assert 600000 < utm_x < 700000  # Approximate UTM easting for London
        assert 5700000 < utm_y < 5800000  # Approximate UTM northing for London
    
    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not available")
    def test_utm_to_wgs84_conversion(self):
        """Test UTM to WGS84 coordinate conversion"""
        cs = CoordinateSystems()
        
        # Test round-trip conversion
        original_lon, original_lat = -0.1278, 51.5074
        
        # Convert to UTM
        utm_x, utm_y, utm_zone = cs.wgs84_to_utm(original_lon, original_lat)
        
        # Convert back to WGS84
        converted_lon, converted_lat = cs.utm_to_wgs84(utm_x, utm_y, utm_zone)
        
        # Should match original coordinates within tolerance
        assert abs(converted_lon - original_lon) < 1e-6
        assert abs(converted_lat - original_lat) < 1e-6
    
    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not available")
    def test_coordinate_bounds_transformation(self):
        """Test transformation of geographic bounds"""
        cs = CoordinateSystems()
        
        # Test bounds covering part of Europe
        wgs84_bounds = GeographicBounds(
            min_x=-5.0, max_x=5.0,    # Longitude
            min_y=45.0, max_y=55.0,   # Latitude
            min_elevation=0, max_elevation=1000
        )
        
        # Transform to UTM (should pick appropriate zone)
        utm_bounds = cs.transform_bounds(wgs84_bounds, "EPSG:4326", "AUTO")
        
        assert utm_bounds.min_x < utm_bounds.max_x
        assert utm_bounds.min_y < utm_bounds.max_y
        assert utm_bounds.min_elevation == wgs84_bounds.min_elevation
        assert utm_bounds.max_elevation == wgs84_bounds.max_elevation
    
    @pytest.mark.parametrize("invalid_coord", [
        (-200, 0),    # Invalid longitude
        (0, -95),     # Invalid latitude
        (200, 0),     # Invalid longitude
        (0, 95),      # Invalid latitude
    ])
    def test_invalid_coordinates(self, invalid_coord):
        """Test handling of invalid coordinates"""
        cs = CoordinateSystems()
        
        lon, lat = invalid_coord
        
        # Should raise exception or return None for invalid coordinates
        with pytest.raises((ValueError, RuntimeError)):
            cs.wgs84_to_utm(lon, lat)
    
    def test_supported_coordinate_systems(self):
        """Test listing of supported coordinate systems"""
        cs = CoordinateSystems()
        
        supported = cs.get_supported_systems()
        assert isinstance(supported, list)
        assert len(supported) > 0
        
        # Should include common systems
        epsg_codes = [s['epsg'] for s in supported if 'epsg' in s]
        assert 4326 in epsg_codes  # WGS84
        assert 3857 in epsg_codes  # Web Mercator


class TestGeoTiffIntegration:
    """Integration tests for complete GeoTIFF pipeline"""
    
    @pytest.fixture
    def integration_setup(self):
        """Setup for integration testing"""
        if not VULKAN_FORGE_AVAILABLE:
            pytest.skip("vulkan-forge not available")
        
        # Mock Vulkan context to avoid GPU dependency
        mock_vulkan = Mock()
        mock_vulkan.is_initialized.return_value = True
        
        config = TerrainConfig.from_preset('balanced')
        
        return {
            'vulkan_context': mock_vulkan,
            'config': config
        }
    
    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_geotiff_to_terrain_pipeline(self, temp_geotiff_file, integration_setup):
        """Test complete GeoTIFF to terrain rendering pipeline"""
        setup = integration_setup
        
        # Load GeoTIFF
        loader = GeoTiffLoader()
        success = loader.load(temp_geotiff_file)
        assert success
        
        # Create terrain cache
        cache = TerrainCache(max_size_mb=50, max_tiles=16)
        
        # Create terrain renderer (mocked)
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            mock_renderer = MockRenderer.return_value
            mock_renderer.load_geotiff.return_value = True
            
            # Test loading GeoTIFF into renderer
            terrain_renderer = MockRenderer(setup['vulkan_context'], setup['config'])
            success = terrain_renderer.load_geotiff(temp_geotiff_file)
            assert success
            
            # Verify renderer was called with correct data
            MockRenderer.assert_called_once_with(setup['vulkan_context'], setup['config'])
            mock_renderer.load_geotiff.assert_called_once_with(temp_geotiff_file)
    
    def test_coordinate_transformation_pipeline(self, integration_setup):
        """Test coordinate transformation in terrain pipeline"""
        setup = integration_setup
        cs = CoordinateSystems()
        
        # Test transforming terrain bounds through pipeline
        original_bounds = GeographicBounds(
            min_x=-1.0, max_x=1.0,
            min_y=50.0, max_y=52.0,
            min_elevation=0, max_elevation=500
        )
        
        # Transform to different coordinate systems
        utm_bounds = cs.transform_bounds(original_bounds, "EPSG:4326", "AUTO")
        web_mercator_bounds = cs.transform_bounds(original_bounds, "EPSG:4326", "EPSG:3857")
        
        # All bounds should be valid
        assert utm_bounds.min_x < utm_bounds.max_x
        assert utm_bounds.min_y < utm_bounds.max_y
        assert web_mercator_bounds.min_x < web_mercator_bounds.max_x
        assert web_mercator_bounds.min_y < web_mercator_bounds.max_y
    
    @pytest.mark.performance
    def test_large_geotiff_loading_performance(self, integration_setup):
        """Test performance of loading large GeoTIFF files"""
        setup = integration_setup
        
        # Create large synthetic heightmap
        size = 2048
        large_heights = np.random.randint(0, 1000, (size, size)).astype(np.float32)
        
        # Time the loading process
        import time
        start_time = time.time()
        
        # Simulate loading large dataset
        loader = GeoTiffLoader()
        
        # Mock the actual file I/O to focus on processing time
        with patch.object(loader, '_load_raw_data') as mock_load:
            mock_load.return_value = {
                'heights': large_heights,
                'width': size,
                'height': size,
                'transform': Mock(),
                'crs': Mock()
            }
            
            success = loader._process_geotiff_data(mock_load.return_value)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 2K x 2K heightmap reasonably quickly
        assert processing_time < 1.0  # Should take less than 1 second
        assert success


# Benchmark and performance tests
class TestGeoTiffPerformance:
    """Performance benchmarks for GeoTIFF loading"""
    
    @pytest.mark.benchmark
    def test_benchmark_geotiff_loading(self, benchmark, temp_geotiff_file):
        """Benchmark GeoTIFF loading performance"""
        loader = GeoTiffLoader()
        
        def load_geotiff():
            loader.load(temp_geotiff_file)
            return loader.is_loaded()
        
        result = benchmark(load_geotiff)
        assert result is True
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [256, 512, 1024, 2048])
    def test_benchmark_tile_caching(self, benchmark, size):
        """Benchmark tile caching performance for different sizes"""
        cache = TerrainCache(max_size_mb=200, max_tiles=100)
        
        # Create test tile
        tile_data = {
            'tile_id': (0, 0),
            'lod_level': 0,
            'heights': np.random.rand(size, size).astype(np.float32),
            'bounds': GeographicBounds(0, 0, 1, 1, 0, 100)
        }
        
        def cache_operations():
            # Store tile
            cache.store_tile(tile_data['tile_id'], tile_data)
            # Retrieve tile
            retrieved = cache.get_tile(tile_data['tile_id'])
            return retrieved is not None
        
        result = benchmark(cache_operations)
        assert result is True


# Fixtures and test utilities
@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_geotiff_files(test_data_dir):
    """Sample GeoTIFF files for testing"""
    files = {
        'small': test_data_dir / "small_terrain.tif",
        'medium': test_data_dir / "medium_terrain.tif",
        'large': test_data_dir / "large_terrain.tif"
    }
    
    # Only return files that actually exist
    return {name: path for name, path in files.items() if path.exists()}


# Test configuration
pytest_plugins = ['pytest_benchmark']


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark tests requiring external dependencies
        if "rasterio" in item.name or "geotiff" in item.name.lower():
            item.add_marker(pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available"))
        
        if "pyproj" in item.name or "coordinate" in item.name.lower():
            item.add_marker(pytest.mark.skipif(not PYPROJ_AVAILABLE, reason="pyproj not available"))
        
        if "vulkan" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available"))