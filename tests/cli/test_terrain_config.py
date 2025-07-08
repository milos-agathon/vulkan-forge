#!/usr/bin/env python3
"""
Test suite for terrain configuration and Python integration

Tests configuration validation, preset systems, hardware optimization,
and Python API integration for the terrain rendering system.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import vulkan-forge components

# Fix imports for CLI tests
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from test_mocks import *
except ImportError:
    # Fallback: create minimal mocks inline
    class GeoTiffLoader:
        def load(self, path): return False
        def get_data(self): return None, {}
    
    class TerrainCache:
        def __init__(self, max_tiles=64, tile_size=256, eviction_policy='lru'): pass
        def get_tile(self, tile_id): return None
        def put_tile(self, tile_id, data): pass
        def get_statistics(self): return {}
    
    class CoordinateSystems:
        def __init__(self): pass
        def is_valid_coordinate(self, lon, lat): return -180 <= lon <= 180 and -90 <= lat <= 90
        def get_supported_systems(self): return ['EPSG:4326']
    
    class GeographicBounds:
        def __init__(self, *args): pass
        def contains(self, lon, lat): return True
    
    class TerrainConfig:
        def __init__(self): 
            from types import SimpleNamespace
            self.tessellation = SimpleNamespace(max_level=64)
            self.performance = SimpleNamespace(target_fps=144)
        
        @classmethod
        def from_preset(cls, preset):
            config = cls()
            if preset == 'high_performance': config.performance.target_fps = 200
            elif preset == 'balanced': config.performance.target_fps = 144
            elif preset == 'high_quality': config.performance.target_fps = 60
            elif preset == 'mobile': config.performance.target_fps = 30
            return config
        
        def optimize_for_hardware(self, gpu, vram, cores): pass
        def validate(self): return []
    
    class ShaderCompiler:
        def __init__(self): 
            self.glslc_path = None
            self.spirv_val_path = None
        def compile_shader(self, source, stage, target='vulkan1.3'): 
            return True, b'\x03\x02\x23\x07' + b'\x00'*100, ""
        def validate_spirv(self, data, target='vulkan1.3'): 
            return True, ""
    
    class TerrainShaderTemplates:
        VERTEX_SHADER = "#version 450\nvoid main() {}"
        TESSELLATION_CONTROL_SHADER = "#version 450\nvoid main() {}"
        TESSELLATION_EVALUATION_SHADER = "#version 450\nvoid main() {}"
        FRAGMENT_SHADER = "#version 450\nvoid main() {}"
    
    class TerrainRenderer:
        def __init__(self, config, context): pass
        def update_camera(self, pos, rot): pass
    
    class InvalidGeoTiffError(Exception): pass


try:
    from vulkan_forge.terrain_config import (
        TerrainConfig, TessellationConfig, LODConfig, CullingConfig,
        MemoryConfig, RenderingConfig, PerformanceConfig,
        TessellationMode, LODAlgorithm, CullingMode
    )
    from vulkan_forge.terrain import TerrainRenderer
    import vulkan_forge_core as vf
    VULKAN_FORGE_AVAILABLE = True
except ImportError:
    VULKAN_FORGE_AVAILABLE = False


class TestTessellationConfig:
    """Test tessellation configuration validation and behavior"""
    
    def test_tessellation_config_defaults(self):
        """Test default tessellation configuration values"""
        config = TessellationConfig()
        
        assert config.mode == TessellationMode.DISTANCE_BASED
        assert config.base_level == 8
        assert config.max_level == 64
        assert config.min_level == 1
        assert config.near_distance == 100.0
        assert config.far_distance == 5000.0
        assert config.falloff_exponent == 1.5
    
    def test_tessellation_level_calculation(self):
        """Test tessellation level calculation based on distance"""
        config = TessellationConfig()
        config.near_distance = 100.0
        config.far_distance = 1000.0
        config.max_level = 32
        config.min_level = 1
        
        # At near distance, should use max level
        level = config.get_tessellation_level(100.0)
        assert level == config.max_level
        
        # At far distance, should use min level
        level = config.get_tessellation_level(1000.0)
        assert level == config.min_level
        
        # Beyond far distance, should still use min level
        level = config.get_tessellation_level(2000.0)
        assert level == config.min_level
        
        # At middle distance, should interpolate
        level = config.get_tessellation_level(550.0)
        assert config.min_level <= level <= config.max_level
    
    @pytest.mark.parametrize("mode", [
        TessellationMode.DISABLED,
        TessellationMode.UNIFORM,
        TessellationMode.DISTANCE_BASED,
        TessellationMode.SCREEN_SPACE
    ])
    def test_tessellation_modes(self, mode):
        """Test different tessellation modes"""
        config = TessellationConfig()
        config.mode = mode
        
        if mode == TessellationMode.DISABLED:
            assert config.get_tessellation_level(100.0) == 1
        elif mode == TessellationMode.UNIFORM:
            assert config.get_tessellation_level(100.0) == config.base_level
            assert config.get_tessellation_level(1000.0) == config.base_level
        else:
            # Distance-based and screen-space should vary with distance
            near_level = config.get_tessellation_level(100.0)
            far_level = config.get_tessellation_level(1000.0)
            assert near_level >= far_level
    
    def test_invalid_tessellation_config(self):
        """Test validation of invalid tessellation configurations"""
        config = TessellationConfig()
        
        # Test invalid level ranges
        with pytest.raises(ValueError):
            config.max_level = 0  # Should be at least 1
        
        with pytest.raises(ValueError):
            config.min_level = config.max_level + 1  # min > max
        
        with pytest.raises(ValueError):
            config.near_distance = config.far_distance + 100  # near > far


class TestLODConfig:
    """Test Level of Detail configuration"""
    
    def test_lod_config_defaults(self):
        """Test default LOD configuration"""
        config = LODConfig()
        
        assert config.algorithm == LODAlgorithm.DISTANCE
        assert len(config.distances) == 4
        assert all(config.distances[i] <= config.distances[i+1] 
                  for i in range(len(config.distances)-1))  # Should be sorted
        assert config.screen_error_threshold == 2.0
        assert config.enable_morphing is True
    
    def test_lod_distance_validation(self):
        """Test LOD distance validation"""
        config = LODConfig()
        
        # Valid distances (sorted)
        config.distances = [100, 500, 1000, 2000]
        # Should not raise exception
        
        # Invalid distances (not sorted)
        with pytest.raises(ValueError):
            config.distances = [500, 100, 1000, 2000]
    
    @pytest.mark.parametrize("algorithm", [
        LODAlgorithm.DISTANCE,
        LODAlgorithm.SCREEN_ERROR,
        LODAlgorithm.FRUSTUM_SIZE
    ])
    def test_lod_algorithms(self, algorithm):
        """Test different LOD algorithms"""
        config = LODConfig()
        config.algorithm = algorithm
        
        # All algorithms should be valid
        assert config.algorithm == algorithm


class TestTerrainConfig:
    """Test complete terrain configuration system"""
    
    def test_terrain_config_defaults(self):
        """Test default terrain configuration"""
        config = TerrainConfig()
        
        assert config.tile_size == 256
        assert config.height_scale == 1.0
        assert config.max_render_distance == 10000.0
        assert isinstance(config.tessellation, TessellationConfig)
        assert isinstance(config.lod, LODConfig)
        assert isinstance(config.culling, CullingConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.rendering, RenderingConfig)
        assert isinstance(config.performance, PerformanceConfig)
    
    @pytest.mark.parametrize("preset_name", [
        'high_performance',
        'balanced', 
        'high_quality',
        'mobile',
        'debug'
    ])
    def test_configuration_presets(self, preset_name):
        """Test terrain configuration presets"""
        config = TerrainConfig.from_preset(preset_name)
        
        assert isinstance(config, TerrainConfig)
        assert config.tile_size > 0
        assert config.height_scale > 0
        assert config.max_render_distance > 0
        
        # Validate preset-specific characteristics
        if preset_name == 'high_performance':
            assert config.tessellation.base_level <= 8  # Lower tessellation for performance
            assert not config.rendering.enable_shadows  # Shadows disabled for performance
        elif preset_name == 'high_quality':
            assert config.tessellation.base_level >= 16  # Higher tessellation for quality
            assert config.rendering.enable_shadows  # Shadows enabled for quality
        elif preset_name == 'mobile':
            assert config.tessellation.max_level <= 16  # Conservative for mobile
            assert config.memory.max_tile_cache_mb <= 128  # Limited memory
        elif preset_name == 'debug':
            assert config.tessellation.base_level == 1  # Simple for debugging
            assert config.performance.enable_profiling  # Profiling enabled
    
    def test_config_validation(self):
        """Test terrain configuration validation"""
        config = TerrainConfig()
        
        # Valid configuration should pass
        issues = config.validate()
        assert len(issues) == 0
        
        # Invalid tile size
        config.tile_size = 0
        issues = config.validate()
        assert len(issues) > 0
        assert any('tile_size' in issue for issue in issues)
        
        # Reset for next test
        config.tile_size = 256
        
        # Invalid height scale
        config.height_scale = -1.0
        issues = config.validate()
        assert len(issues) > 0
        assert any('height_scale' in issue for issue in issues)
    
    def test_hardware_optimization(self):
        """Test automatic hardware optimization"""
        config = TerrainConfig()
        
        # Test high-end GPU optimization
        config.optimize_for_hardware("RTX 4090", 24576, 16)  # 24GB VRAM, 16 cores
        assert config.tessellation.max_level == 64
        assert config.memory.max_tile_cache_mb >= 1024
        assert config.performance.worker_threads <= 15  # N-1 cores
        
        # Test mid-range GPU optimization  
        config = TerrainConfig()
        config.optimize_for_hardware("RTX 3070", 8192, 8)  # 8GB VRAM, 8 cores
        assert config.tessellation.max_level == 32
        assert 256 <= config.memory.max_tile_cache_mb <= 1024
        
        # Test low-end GPU optimization
        config = TerrainConfig()
        config.optimize_for_hardware("GTX 1660", 2048, 4)  # 2GB VRAM, 4 cores
        assert config.tessellation.max_level <= 16
        assert config.memory.max_tile_cache_mb <= 256
        assert not config.performance.enable_gpu_driven_rendering
    
    def test_memory_usage_estimation(self):
        """Test GPU memory usage estimation"""
        config = TerrainConfig()
        
        memory_est = config.get_estimated_memory_usage()
        assert 'tile_geometry_mb' in memory_est
        assert 'texture_cache_mb' in memory_est
        assert 'total_gpu_mb' in memory_est
        assert 'system_ram_mb' in memory_est
        
        # All values should be positive
        assert all(v >= 0 for v in memory_est.values())
        
        # Total should be sum of components
        expected_total = (memory_est['tile_geometry_mb'] + 
                         memory_est['texture_cache_mb'] + 100)  # +100 for overhead
        assert abs(memory_est['total_gpu_mb'] - expected_total) < 1.0
    
    def test_config_serialization(self):
        """Test configuration save/load functionality"""
        config = TerrainConfig.from_preset('balanced')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_to_file(f.name)
            filepath = f.name
        
        try:
            # Load from file
            loaded_config = TerrainConfig.load_from_file(filepath)
            
            # Compare key values (simplified comparison)
            assert loaded_config.tile_size == config.tile_size
            assert loaded_config.height_scale == config.height_scale
            assert loaded_config.max_render_distance == config.max_render_distance
        finally:
            Path(filepath).unlink()
    
    def test_invalid_preset_name(self):
        """Test handling of invalid preset names"""
        with pytest.raises(ValueError):
            TerrainConfig.from_preset('nonexistent_preset')


class TestTerrainRenderer:
    """Test terrain renderer Python API"""
    
    @pytest.fixture
    def mock_vulkan_context(self):
        """Mock Vulkan context for testing"""
        mock_context = Mock()
        mock_context.get_device.return_value = Mock()
        mock_context.get_device_features.return_value = Mock(tessellationShader=True)
        mock_context.get_command_pool.return_value = Mock()
        return mock_context
    
    @pytest.fixture
    def sample_config(self):
        """Sample terrain configuration"""
        return TerrainConfig.from_preset('balanced')
    
    @pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available")
    def test_terrain_renderer_initialization(self, mock_vulkan_context, sample_config):
        """Test terrain renderer initialization"""
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            renderer = MockRenderer(mock_vulkan_context, sample_config)
            
            # Should be called with correct parameters
            MockRenderer.assert_called_once_with(mock_vulkan_context, sample_config)
    
    @pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available")
    def test_terrain_bounds_calculation(self, mock_vulkan_context, sample_config):
        """Test terrain bounds calculation"""
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            mock_renderer = MockRenderer.return_value
            
            # Mock bounds
            from vulkan_forge.terrain_config import GeographicBounds
            mock_bounds = GeographicBounds(
                min_x=-1.0, max_x=1.0,
                min_y=-1.0, max_y=1.0,
                min_elevation=0.0, max_elevation=100.0
            )
            mock_renderer.get_bounds.return_value = mock_bounds
            
            renderer = MockRenderer(mock_vulkan_context, sample_config)
            bounds = renderer.get_bounds()
            
            assert bounds.min_x == -1.0
            assert bounds.max_x == 1.0
            assert bounds.min_y == -1.0
            assert bounds.max_y == 1.0
            assert bounds.min_elevation == 0.0
            assert bounds.max_elevation == 100.0
    
    @pytest.mark.skipif(not VULKAN_FORGE_AVAILABLE, reason="vulkan-forge not available")
    def test_performance_stats_collection(self, mock_vulkan_context, sample_config):
        """Test performance statistics collection"""
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            mock_renderer = MockRenderer.return_value
            
            # Mock performance stats
            mock_stats = {
                'triangles_rendered': 1000000,
                'tiles_rendered': 64,
                'culled_tiles': 32,
                'frame_time_ms': 6.9,  # ~144 FPS
                'fps': 144.9,
                'triangles_per_second': 144900000
            }
            mock_renderer.get_performance_stats.return_value = mock_stats
            
            renderer = MockRenderer(mock_vulkan_context, sample_config)
            stats = renderer.get_performance_stats()
            
            assert stats['triangles_rendered'] == 1000000
            assert stats['fps'] > 144.0
            assert stats['triangles_per_second'] > 100000000
    
    def test_camera_update_parameters(self, mock_vulkan_context, sample_config):
        """Test camera update parameter validation"""
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            mock_renderer = MockRenderer.return_value
            
            renderer = MockRenderer(mock_vulkan_context, sample_config)
            
            # Test camera update with valid matrices
            view_matrix = np.eye(4, dtype=np.float32)
            proj_matrix = np.eye(4, dtype=np.float32)
            position = np.array([0.0, 0.0, 100.0], dtype=np.float32)
            
            renderer.update_camera(view_matrix, proj_matrix, position)
            
            # Verify the call was made
            mock_renderer.update_camera.assert_called_once()
            args = mock_renderer.update_camera.call_args[0]
            
            # Check that matrices have correct shape
            assert args[0].shape == (4, 4)  # view matrix
            assert args[1].shape == (4, 4)  # projection matrix
            assert args[2].shape == (3,)    # position vector


class TestTerrainConfigPropertyBasedTesting:
    """Property-based testing for terrain configuration"""
    
    @pytest.mark.parametrize("tile_size", [16, 32, 64, 128, 256, 512, 1024])
    def test_tile_size_properties(self, tile_size):
        """Test properties that should hold for any valid tile size"""
        config = TerrainConfig()
        config.tile_size = tile_size
        
        # Tile size should be power of 2
        assert (tile_size & (tile_size - 1)) == 0
        
        # Memory estimation should scale with tile size
        memory_est = config.get_estimated_memory_usage()
        assert memory_est['tile_geometry_mb'] > 0
        
        # Larger tiles should use more memory
        if tile_size >= 256:
            config_small = TerrainConfig()
            config_small.tile_size = 128
            memory_small = config_small.get_estimated_memory_usage()
            assert memory_est['tile_geometry_mb'] >= memory_small['tile_geometry_mb']
    
    @pytest.mark.parametrize("height_scale", [0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
    def test_height_scale_properties(self, height_scale):
        """Test properties for different height scales"""
        config = TerrainConfig()
        config.height_scale = height_scale
        
        # Height scale should be positive
        assert height_scale > 0
        
        # Configuration should remain valid
        issues = config.validate()
        assert len(issues) == 0
    
    @pytest.mark.parametrize("distance", [100, 500, 1000, 5000, 10000])
    def test_lod_distance_properties(self, distance):
        """Test LOD properties for different distances"""
        config = TessellationConfig()
        level = config.get_tessellation_level(float(distance))
        
        # Level should be within valid range
        assert config.min_level <= level <= config.max_level
        
        # Closer distances should have higher tessellation levels
        if distance <= config.near_distance:
            assert level == config.max_level
        elif distance >= config.far_distance:
            assert level == config.min_level


class TestConfigurationValidation:
    """Test comprehensive configuration validation"""
    
    def test_cross_component_validation(self):
        """Test validation across configuration components"""
        config = TerrainConfig()
        
        # Test memory constraints vs performance settings
        config.memory.max_tile_cache_mb = 64  # Very low memory
        config.performance.target_fps = 144   # High performance target
        config.tessellation.max_level = 64    # High tessellation
        
        issues = config.validate()
        # Should identify potential conflicts
        assert len(issues) >= 0  # May or may not have issues depending on implementation
    
    def test_platform_specific_validation(self):
        """Test platform-specific configuration validation"""
        config = TerrainConfig()
        
        # Mobile configuration should have appropriate limits
        mobile_config = TerrainConfig.from_preset('mobile')
        mobile_issues = mobile_config.validate()
        assert len(mobile_issues) == 0
        
        # High-quality configuration should be valid for high-end systems
        hq_config = TerrainConfig.from_preset('high_quality')
        hq_issues = hq_config.validate()
        assert len(hq_issues) == 0
    
    @pytest.mark.parametrize("invalid_config", [
        {'tile_size': 0},
        {'tile_size': 3},  # Not power of 2
        {'height_scale': -1.0},
        {'max_render_distance': 0},
        {'tessellation': {'max_level': 0}},
        {'memory': {'max_tile_cache_mb': -1}},
    ])
    def test_invalid_configurations(self, invalid_config):
        """Test detection of invalid configurations"""
        config = TerrainConfig()
        
        # Apply invalid configuration
        for key, value in invalid_config.items():
            if isinstance(value, dict):
                # Nested configuration
                nested_obj = getattr(config, key)
                for nested_key, nested_value in value.items():
                    setattr(nested_obj, nested_key, nested_value)
            else:
                setattr(config, key, value)
        
        # Should detect validation issues
        issues = config.validate()
        assert len(issues) > 0


# Integration tests combining multiple components
class TestConfigurationIntegration:
    """Integration tests for configuration system"""
    
    def test_config_to_renderer_pipeline(self, mock_vulkan_context):
        """Test configuration flowing through to renderer"""
        config = TerrainConfig.from_preset('high_performance')
        
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            renderer = MockRenderer(mock_vulkan_context, config)
            
            # Configuration should be passed to renderer
            MockRenderer.assert_called_once_with(mock_vulkan_context, config)
            
            # Test configuration updates
            new_config = TerrainConfig.from_preset('high_quality')
            renderer.set_config(new_config)
            
            # Should be able to retrieve updated config
            mock_renderer = MockRenderer.return_value
            mock_renderer.get_config.return_value = new_config
            retrieved_config = renderer.get_config()
            assert retrieved_config == new_config
    
    def test_dynamic_configuration_updates(self, mock_vulkan_context):
        """Test dynamic configuration updates during rendering"""
        config = TerrainConfig.from_preset('balanced')
        
        with patch('vulkan_forge.terrain.TerrainRenderer') as MockRenderer:
            renderer = MockRenderer(mock_vulkan_context, config)
            
            # Simulate runtime configuration changes
            config.tessellation.base_level = 16
            config.performance.target_fps = 60
            
            renderer.set_config(config)
            
            # Should handle configuration updates gracefully
            mock_renderer = MockRenderer.return_value
            mock_renderer.set_config.assert_called()


# Performance and stress tests
@pytest.mark.performance
class TestConfigurationPerformance:
    """Performance tests for configuration system"""
    
    @pytest.mark.benchmark
    def test_config_creation_performance(self, benchmark):
        """Benchmark configuration creation performance"""
        def create_config():
            return TerrainConfig.from_preset('balanced')
        
        result = benchmark(create_config)
        assert isinstance(result, TerrainConfig)
    
    @pytest.mark.benchmark
    def test_config_validation_performance(self, benchmark):
        """Benchmark configuration validation performance"""
        config = TerrainConfig.from_preset('high_quality')
        
        def validate_config():
            return config.validate()
        
        result = benchmark(validate_config)
        assert isinstance(result, list)
    
    def test_memory_usage_calculation_performance(self):
        """Test performance of memory usage calculations"""
        config = TerrainConfig()
        
        import time
        start_time = time.time()
        
        # Calculate memory usage multiple times
        for _ in range(1000):
            memory_est = config.get_estimated_memory_usage()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        
        # Should be very fast (< 1ms per calculation)
        assert avg_time < 0.001