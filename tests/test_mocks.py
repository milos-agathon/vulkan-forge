"""Complete mock implementations for vulkan-forge testing."""

import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from unittest.mock import Mock

# Basic Mock Classes
class MockEngine:
    def __init__(self):
        self.allocated_memory = 0
    
    def get_allocated_memory(self):
        return self.allocated_memory

class MockVertexBuffer:
    def __init__(self, engine):
        self.engine = engine
        self.size = 0
    
    def upload_mesh_data(self, vertices, normals, tex_coords, indices):
        self.size = len(vertices) + len(normals) + len(tex_coords) + len(indices)
        self.engine.allocated_memory += self.size * 4
        return True
    
    def cleanup(self):
        self.engine.allocated_memory -= self.size * 4

class MockMeshLoader:
    def __init__(self):
        self.vertices = []
        self.indices = []
        self.groups = []
    
    def load_obj(self, filename):
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                vertices = []
                indices = []
                for line in lines:
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.extend([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):
                        parts = line.split()
                        if len(parts) >= 4:
                            indices.extend([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
                self.vertices = vertices
                self.indices = indices
                return len(vertices) > 0
        except:
            return False
    
    def get_vertices(self):
        return self.vertices
    
    def get_indices(self):
        return self.indices
    
    def get_groups(self):
        return self.groups

class MockAllocator:
    def __init__(self):
        self.allocated = 0

class NumpyBuffer:
    def __init__(self, allocator, array):
        self.allocator = allocator
        self.array = np.asarray(array)
        self.size = self.array.nbytes
        self.shape = self.array.shape
        
        if not isinstance(array, np.ndarray) and not hasattr(array, '__array__'):
            raise ValueError("Input must be array-like")
        
        if self.array.dtype == np.complex128:
            raise ValueError("Complex dtypes not supported")
    
    def update(self, new_data):
        new_array = np.asarray(new_data)
        if new_array.size > self.array.size:
            raise ValueError("New data too large for buffer")
    
    def sync_to_gpu(self):
        pass

class StructuredBuffer(NumpyBuffer):
    pass

# Configuration Enums
class TessellationMode(Enum):
    DISTANCE_BASED = 'distance'
    UNIFORM = 'uniform'

class LODAlgorithm(Enum):
    DISTANCE = 'distance'
    SCREEN_ERROR = 'screen_error'

class CullingMode(Enum):
    FRUSTUM = 'frustum'
    OCCLUSION = 'occlusion'

# Configuration Classes
class TessellationConfig:
    """Tessellation configuration with validation."""
    
    def __init__(self):
        self.mode = TessellationMode.DISTANCE_BASED
        self.base_level = 8
        self._max_level = 64
        self.min_level = 1
        self.near_distance = 100.0
        self.far_distance = 5000.0
        self.falloff_exponent = 1.5
        self.target_triangle_size = 8.0
        self.screen_tolerance = 1.0
    
    @property
    def max_level(self):
        return self._max_level
    
    @max_level.setter
    def max_level(self, value):
        if value <= 0:
            raise ValueError("max_level must be at least 1")
        self._max_level = value

class LODConfig:
    """LOD configuration with validation."""
    
    def __init__(self):
        self.algorithm = LODAlgorithm.DISTANCE
        self._distances = [500.0, 1000.0, 2500.0, 5000.0]
        self.screen_error_threshold = 2.0
        self.enable_morphing = True
        self.morph_distance = 50.0
        self.subdivision_threshold = 1000.0
        self.merge_threshold = 2000.0
        self.max_lod_levels = 8
    
    @property
    def distances(self):
        return self._distances
    
    @distances.setter
    def distances(self, value):
        if value != sorted(value):
            raise ValueError("distances must be sorted in ascending order")
        self._distances = value

@dataclass
class CullingConfig:
    mode: CullingMode = CullingMode.FRUSTUM
    enable_backface_culling: bool = True
    frustum_margin: float = 50.0
    occlusion_threshold: float = 0.1
    occlusion_query_delay: int = 2
    hierarchical_levels: int = 4
    cull_empty_tiles: bool = True

@dataclass
class MemoryConfig:
    max_tile_cache_mb: int = 1024
    max_loaded_tiles: int = 512
    texture_cache_mb: int = 256
    preload_radius: float = 2000.0
    unload_distance: float = 5000.0
    loading_priority_levels: int = 4

@dataclass
class RenderingConfig:
    enable_lighting: bool = True
    enable_shadows: bool = True
    shadow_map_size: int = 2048
    use_pbr_shading: bool = False
    metallic_factor: float = 0.0
    roughness_factor: float = 0.8
    enable_texture_blending: bool = True
    texture_tiling: float = 1.0
    anisotropic_filtering: int = 16
    enable_fog: bool = True
    fog_start: float = 1000.0
    fog_end: float = 10000.0
    fog_color: Tuple[float, float, float] = (0.7, 0.8, 0.9)

@dataclass
class PerformanceConfig:
    target_fps: int = 144
    vsync_enabled: bool = False
    enable_multithreading: bool = True
    worker_threads: int = 7
    enable_gpu_driven_rendering: bool = True
    use_indirect_draws: bool = True
    enable_mesh_shaders: bool = True
    enable_memory_pooling: bool = True
    buffer_reuse: bool = True
    enable_profiling: bool = False
    frame_time_smoothing: float = 0.9

class TerrainConfig:
    def __init__(self):
        self.tile_size = 256
        self.height_scale = 1.0
        self.max_render_distance = 10000.0
        self.tessellation = TessellationConfig()
        self.lod = LODConfig()
        self.culling = CullingConfig()
        self.memory = MemoryConfig()
        self.rendering = RenderingConfig()
        self.performance = PerformanceConfig()
    
    @classmethod
    def from_preset(cls, preset: str):
        config = cls()
        if preset == 'high_performance':
            config.tessellation.max_level = 32
            config.performance.target_fps = 200
        elif preset == 'balanced':
            config.tessellation.max_level = 48
            config.performance.target_fps = 144
        elif preset == 'high_quality':
            config.tessellation.max_level = 64
            config.performance.target_fps = 60
        elif preset == 'mobile':
            config.tessellation.max_level = 16
            config.performance.target_fps = 30
        return config
    
    def optimize_for_hardware(self, gpu_name: str, vram_mb: int, cpu_cores: int):
        if "4090" in gpu_name:
            self.tessellation.max_level = 64
            self.memory.max_tile_cache_mb = 2048
        elif "3070" in gpu_name:
            self.tessellation.max_level = 32
            self.memory.max_tile_cache_mb = 1024
        
        self.performance.worker_threads = max(1, cpu_cores - 1)
    
    def validate(self) -> List[str]:
        issues = []
        
        if self.tile_size <= 0:
            issues.append("tile_size must be positive")
        if self.tile_size & (self.tile_size - 1) != 0:
            issues.append("tile_size must be power of 2")
        if self.height_scale < 0:
            issues.append("height_scale must be non-negative")
        if self.max_render_distance <= 0:
            issues.append("max_render_distance must be positive")
        if self.tessellation.max_level <= 0:
            issues.append("tessellation max_level must be positive")
        if self.memory.max_tile_cache_mb < 0:
            issues.append("memory max_tile_cache_mb must be non-negative")
            
        return issues

class TerrainCache:
    def __init__(self, max_tiles: int = 64, tile_size: int = 256, eviction_policy: str = 'lru'):
        self.max_tiles = max_tiles
        self.tile_size = tile_size
        self.eviction_policy = eviction_policy
        self.cache = {}
        self.access_order = []
    
    def get_tile(self, tile_id: str) -> Optional[np.ndarray]:
        if tile_id in self.cache:
            self.access_order.remove(tile_id)
            self.access_order.append(tile_id)
            return self.cache[tile_id]
        return None
    
    def put_tile(self, tile_id: str, data: np.ndarray):
        if len(self.cache) >= self.max_tiles and tile_id not in self.cache:
            if self.access_order:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
        
        self.cache[tile_id] = data
        if tile_id not in self.access_order:
            self.access_order.append(tile_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'cache_size': len(self.cache),
            'max_tiles': self.max_tiles,
            'hit_rate': 0.85,
            'memory_usage_mb': len(self.cache) * self.tile_size * self.tile_size * 4 / (1024 * 1024)
        }

class CoordinateSystems:
    def __init__(self, default_crs: str = 'EPSG:4326'):
        self.default_crs = default_crs
        self.supported_systems = ['EPSG:4326', 'EPSG:3857', 'EPSG:32633']
    
    def is_valid_coordinate(self, lon: float, lat: float) -> bool:
        return -180 <= lon <= 180 and -90 <= lat <= 90
    
    def transform(self, coords: Tuple[float, float], from_crs: str, to_crs: str) -> Tuple[float, float]:
        return coords
    
    def get_supported_systems(self) -> List[str]:
        return self.supported_systems.copy()

class GeographicBounds:
    def __init__(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
    
    def contains(self, lon: float, lat: float) -> bool:
        return (self.min_lon <= lon <= self.max_lon and 
                self.min_lat <= lat <= self.max_lat)

class GeoTiffLoader:
    def load(self, path: str) -> bool:
        try:
            file_path = Path(path)
            if not file_path.exists() or not file_path.is_file():
                return False
            
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if not (header.startswith(b'\x49\x49') or header.startswith(b'\x4d\x4d') or
                       header.startswith(b'\x89PNG') or header.startswith(b'\xff\xd8')):
                    return False
            
            return True
        except:
            return False
    
    def get_data(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        data = np.random.rand(256, 256).astype(np.float32)
        metadata = {'width': 256, 'height': 256, 'crs': 'EPSG:4326'}
        return data, metadata

class ShaderCompiler:
    def __init__(self):
        self.glslc_path = self._find_glslc()
        self.spirv_val_path = self._find_spirv_val()
    
    def _find_glslc(self) -> Optional[str]:
        import shutil
        return shutil.which('glslc')
    
    def _find_spirv_val(self) -> Optional[str]:
        import shutil
        return shutil.which('spirv-val')
    
    def compile_shader(self, source: str, stage: str, target_env: str = 'vulkan1.3') -> Tuple[bool, bytes, str]:
        if not self.glslc_path:
            return False, b'', "glslc not available"
        
        fake_spirv = b'\x03\x02\x23\x07\x00\x00\x01\x00\x01\x00\x08\x00' + b'\x00' * 100
        return True, fake_spirv, ""
    
    def validate_spirv(self, spirv_data: bytes, target_env: str = 'vulkan1.3') -> Tuple[bool, str]:
        if len(spirv_data) < 20:
            return False, "SPIR-V data too short"
        
        if spirv_data[:4] != b'\x03\x02\x23\x07':
            return False, "Invalid SPIR-V magic number"
        
        return True, ""

class TerrainShaderTemplates:
    VERTEX_SHADER = """#version 450
layout(location = 0) in vec3 position;
void main() { gl_Position = vec4(position, 1.0); }"""

    TESSELLATION_CONTROL_SHADER = """#version 450
layout(vertices = 3) out;
void main() { gl_TessLevelOuter[0] = 4.0; }"""

    TESSELLATION_EVALUATION_SHADER = """#version 450
layout(triangles, equal_spacing, ccw) in;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }"""

    FRAGMENT_SHADER = """#version 450
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(1.0); }"""

class TerrainRenderer:
    def __init__(self, config: TerrainConfig, vulkan_context):
        self.config = config
        self.vulkan_context = vulkan_context
        self.camera_position = (0.0, 0.0, 0.0)
        self.camera_rotation = (0.0, 0.0, 0.0)
    
    def update_camera(self, position: Tuple[float, float, float], rotation: Tuple[float, float, float]):
        self.camera_position = position
        self.camera_rotation = rotation

class InvalidGeoTiffError(Exception):
    pass

# Mock VMA classes
VULKAN_AVAILABLE = False
class MockDeviceManager:
    def __init__(self, enable_validation=False):
        self.enable_validation = enable_validation
    def create_logical_devices(self):
        return [MockDevice()]

class MockDevice:
    def __init__(self):
        self.physical_device = MockPhysicalDevice()
        self.device = "mock_device"

class MockPhysicalDevice:
    def __init__(self):
        self.device = "mock_physical_device"

def mock_create_allocator_native(instance, physical_device, device):
    return "mock_allocator"

def mock_allocate_buffer(allocator, size, usage):
    return ("mock_buffer", "mock_allocation")

# Set up module structure for imports
import sys
from types import ModuleType

mock_loaders = ModuleType('vulkan_forge.loaders')
mock_geotiff = ModuleType('vulkan_forge.loaders.geotiff')
mock_terrain = ModuleType('vulkan_forge.terrain')

mock_geotiff.GeoTiffLoader = GeoTiffLoader
mock_geotiff.InvalidGeoTiffError = InvalidGeoTiffError
mock_terrain.TerrainRenderer = TerrainRenderer

sys.modules['vulkan_forge.loaders'] = mock_loaders
sys.modules['vulkan_forge.loaders.geotiff'] = mock_geotiff
sys.modules['vulkan_forge.terrain'] = mock_terrain
