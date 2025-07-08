"""
Terrain Rendering Configuration Classes

This module defines configuration classes for the Vulkan-Forge terrain system,
allowing fine-tuned control over performance vs quality trade-offs.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
from enum import Enum
import json
import numpy as np


class TessellationMode(Enum):
    """Tessellation mode for terrain rendering"""

    DISABLED = "disabled"  # No tessellation, use base mesh
    UNIFORM = "uniform"  # Uniform tessellation level
    DISTANCE_BASED = "distance"  # Distance-based adaptive tessellation
    SCREEN_SPACE = "screen_space"  # Screen-space adaptive tessellation


class LODAlgorithm(Enum):
    """Level of Detail algorithm"""

    DISTANCE = "distance"  # Simple distance-based LOD
    SCREEN_ERROR = "screen_error"  # Screen-space error metric
    FRUSTUM_SIZE = "frustum_size"  # Based on projected frustum size


class CullingMode(Enum):
    """Culling modes for terrain tiles"""

    NONE = "none"  # No culling
    FRUSTUM = "frustum"  # Frustum culling only
    OCCLUSION = "occlusion"  # Frustum + occlusion culling
    HIERARCHICAL = "hierarchical"  # Hierarchical Z-buffer culling


@dataclass
class TessellationConfig:
    """Configuration for GPU tessellation"""

    mode: TessellationMode = TessellationMode.DISTANCE_BASED
    base_level: int = 8  # Base tessellation level (1-64)
    near_distance: float = 100.0  # Near tessellation distance
    far_distance: float = 5000.0  # Far tessellation distance
    falloff_exponent: float = 1.5  # Distance falloff curve
    target_triangle_size: float = 8.0  # Target triangle size in pixels
    screen_tolerance: float = 1.0  # Screen-space error tolerance

    def __post_init__(self):
        """Initialize private attributes after dataclass creation."""
        self._min_level = 1
        self._max_level = 64

    @property
    def max_level(self) -> int:
        """Get the maximum tessellation level."""
        return self._max_level

    @max_level.setter
    def max_level(self, value: int) -> None:
        """Set the maximum tessellation level."""
        if not isinstance(value, int):
            raise TypeError(f"max_level must be an integer, got {type(value).__name__}")
        if value < 1:
            raise ValueError(f"max_level must be at least 1, got {value}")
        if value > 64:
            raise ValueError(f"max_level must be <= 64, got {value}")
        if hasattr(self, "_min_level") and self._min_level > value:
            raise ValueError("min_level cannot exceed max_level")
        self._max_level = value

    @property
    def min_level(self) -> int:
        """Get the minimum tessellation level."""
        return self._min_level

    @min_level.setter
    def min_level(self, value: int) -> None:
        """Set the minimum tessellation level."""
        if not isinstance(value, int):
            raise TypeError("min_level must be int")
        if value < 1:
            raise ValueError("min_level must be >= 1")
        if value > 64:
            raise ValueError("min_level must be <= 64")
        if hasattr(self, "_max_level") and value > self._max_level:
            raise ValueError("min_level cannot exceed max_level")
        self._min_level = value

    def get_tessellation_level(self, distance: float) -> int:
        """Return tessellation level based on camera distance."""
        if distance < 0:
            raise ValueError("distance must be non-negative")

        if self.mode in (TessellationMode.DISABLED, TessellationMode.UNIFORM):
            return (
                self.base_level
                if self.mode is TessellationMode.UNIFORM
                else self.min_level
            )

        if distance <= self.near_distance:
            return int(self.base_level)
        if distance >= self.far_distance:
            return int(self.max_level)

        span = max(self.far_distance - self.near_distance, 1e-6)
        ratio = (distance - self.near_distance) / span
        level = self.base_level + ratio * (self.max_level - self.base_level)
        level = max(self.base_level, min(self.max_level, level))
        return int(round(level))


def _get_near_distance(self) -> float:
    return self._near_distance


def _set_near_distance(self, value: float) -> None:
    val = float(value)
    if getattr(self, "_far_distance", val + 1) <= val:
        raise ValueError("near_distance must be < far_distance")
    self._near_distance = val


def _get_far_distance(self) -> float:
    return self._far_distance


def _set_far_distance(self, value: float) -> None:
    val = float(value)
    if getattr(self, "_near_distance", val - 1) >= val:
        raise ValueError("far_distance must be > near_distance")
    self._far_distance = val


TessellationConfig.near_distance = property(_get_near_distance, _set_near_distance)
TessellationConfig.far_distance = property(_get_far_distance, _set_far_distance)


@dataclass
class LODConfig:
    """Level of Detail configuration - Fixed version"""

    algorithm: LODAlgorithm = LODAlgorithm.DISTANCE
    screen_error_threshold: float = 2.0
    enable_morphing: bool = True
    morph_distance: float = 50.0
    subdivision_threshold: float = 1000.0
    merge_threshold: float = 2000.0
    max_lod_levels: int = 8

    def __post_init__(self):
        """Initialize distances list after dataclass creation."""
        # Set as simple attribute to avoid property conflicts
        self._distances = [500.0, 1000.0, 2500.0, 5000.0]


def _get_distances(self) -> List[float]:
    return self._distances


def _set_distances(self, value: List[float]) -> None:
    vals = list(value)
    if vals != sorted(vals):
        raise ValueError("LOD distances must be sorted ascending")
    self._distances = vals


LODConfig.distances = property(_get_distances, _set_distances)


@dataclass
class CullingConfig:
    """Culling configuration"""

    mode: CullingMode = CullingMode.FRUSTUM
    enable_backface_culling: bool = True
    frustum_margin: float = 50.0
    occlusion_threshold: float = 0.1
    occlusion_query_delay: int = 2
    hierarchical_levels: int = 4
    cull_empty_tiles: bool = True


@dataclass
class MemoryConfig:
    """Memory management configuration"""

    max_tile_cache_mb: int = 512
    max_loaded_tiles: int = 256
    texture_cache_mb: int = 256
    preload_radius: float = 2000.0
    unload_distance: float = 5000.0
    loading_priority_levels: int = 4


@dataclass
class RenderingConfig:
    """General rendering configuration"""

    enable_lighting: bool = True
    enable_shadows: bool = False
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
    """Performance tuning configuration"""

    target_fps: int = 144
    vsync_enabled: bool = False
    enable_multithreading: bool = True
    worker_threads: int = 4
    enable_gpu_driven_rendering: bool = True
    use_indirect_draws: bool = True
    enable_mesh_shaders: bool = False
    enable_memory_pooling: bool = True
    buffer_reuse: bool = True
    enable_profiling: bool = False
    frame_time_smoothing: float = 0.9


@dataclass
class TerrainConfig:
    """Complete terrain rendering configuration"""

    tile_size: int = 256
    height_scale: float = 1.0
    max_render_distance: float = 10000.0
    tessellation: TessellationConfig = field(default_factory=TessellationConfig)
    lod: LODConfig = field(
        default_factory=LODConfig
    )  # Now LODConfig is properly defined
    culling: CullingConfig = field(default_factory=CullingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_preset(cls, preset_name: str) -> "TerrainConfig":
        """Create configuration from preset"""
        presets = {
            "high_performance": cls._high_performance_preset(),
            "balanced": cls._balanced_preset(),
            "high_quality": cls._high_quality_preset(),
            "mobile": cls._mobile_preset(),
            "debug": cls._debug_preset(),
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}"
            )

        return presets[preset_name]

    @classmethod
    def _high_performance_preset(cls) -> "TerrainConfig":
        """High performance preset"""
        config = cls()
        config.tessellation.base_level = 4
        config.tessellation.max_level = 16
        config.lod.distances = [200.0, 500.0, 1000.0, 2000.0]
        config.performance.target_fps = 144
        return config

    @classmethod
    def _balanced_preset(cls) -> "TerrainConfig":
        """Balanced preset"""
        return cls()  # Default values

    @classmethod
    def _high_quality_preset(cls) -> "TerrainConfig":
        """High quality preset"""
        config = cls()
        config.tessellation.base_level = 16
        config.tessellation.max_level = 64
        config.lod.distances = [1000.0, 2000.0, 4000.0, 8000.0]
        config.performance.target_fps = 60
        return config

    @classmethod
    def _mobile_preset(cls) -> "TerrainConfig":
        """Mobile preset"""
        config = cls()
        config.tessellation.base_level = 2
        config.tessellation.max_level = 8
        config.lod.distances = [100.0, 250.0, 500.0, 1000.0]
        config.performance.target_fps = 30
        return config

    @classmethod
    def _debug_preset(cls) -> "TerrainConfig":
        """Debug preset"""
        config = cls()
        config.tessellation.base_level = 1
        config.performance.enable_profiling = True
        return config


@dataclass
class CullingConfig:
    """Culling configuration"""

    mode: CullingMode = CullingMode.FRUSTUM
    enable_backface_culling: bool = True

    # Frustum culling
    frustum_margin: float = 50.0  # Extra margin for frustum culling

    # Occlusion culling
    occlusion_threshold: float = 0.1  # Occlusion query threshold
    occlusion_query_delay: int = 2  # Frames to delay occlusion queries

    # Hierarchical culling
    hierarchical_levels: int = 4  # Levels in hierarchy
    cull_empty_tiles: bool = True  # Cull tiles with no geometry


@dataclass
class MemoryConfig:
    """Memory management configuration"""

    max_tile_cache_mb: int = 512  # Maximum tile cache size in MB
    max_loaded_tiles: int = 256  # Maximum simultaneously loaded tiles
    texture_cache_mb: int = 256  # Texture atlas cache size

    # Streaming parameters
    preload_radius: float = 2000.0  # Preload tiles within this radius
    unload_distance: float = 5000.0  # Unload tiles beyond this distance
    loading_priority_levels: int = 4  # Number of loading priority levels


@dataclass
class RenderingConfig:
    """General rendering configuration"""

    # Shading
    enable_lighting: bool = True
    enable_shadows: bool = False
    shadow_map_size: int = 2048

    # Materials
    use_pbr_shading: bool = False
    metallic_factor: float = 0.0
    roughness_factor: float = 0.8

    # Texturing
    enable_texture_blending: bool = True
    texture_tiling: float = 1.0
    anisotropic_filtering: int = 16

    # Post-processing
    enable_fog: bool = True
    fog_start: float = 1000.0
    fog_end: float = 10000.0
    fog_color: Tuple[float, float, float] = (0.7, 0.8, 0.9)


@dataclass
class PerformanceConfig:
    """Performance tuning configuration"""

    target_fps: int = 144  # Target frame rate
    vsync_enabled: bool = False  # Enable V-Sync

    # Threading
    enable_multithreading: bool = True
    worker_threads: int = 4  # Number of worker threads

    # GPU optimization
    enable_gpu_driven_rendering: bool = True
    use_indirect_draws: bool = True
    enable_mesh_shaders: bool = False  # Enable if supported

    # Memory optimization
    enable_memory_pooling: bool = True
    buffer_reuse: bool = True

    # Debugging
    enable_profiling: bool = False
    frame_time_smoothing: float = 0.9  # Frame time smoothing factor


@dataclass
class TerrainConfig:
    """Complete terrain rendering configuration"""

    # Basic parameters
    tile_size: int = 256  # Tile size in vertices
    height_scale: float = 1.0  # Height scaling factor
    max_render_distance: float = 10000.0  # Maximum render distance

    # Sub-configurations
    tessellation: TessellationConfig = field(default_factory=TessellationConfig)
    lod: LODConfig = field(default_factory=LODConfig)
    culling: CullingConfig = field(default_factory=CullingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_preset(cls, preset_name: str) -> "TerrainConfig":
        """Create configuration from preset"""
        presets = {
            "high_performance": cls._high_performance_preset(),
            "balanced": cls._balanced_preset(),
            "high_quality": cls._high_quality_preset(),
            "mobile": cls._mobile_preset(),
            "debug": cls._debug_preset(),
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}"
            )

        return presets[preset_name]

    @classmethod
    def _high_performance_preset(cls) -> "TerrainConfig":
        """High performance preset - prioritizes FPS over visual quality"""
        config = cls()

        # Reduce tessellation for performance
        config.tessellation.base_level = 4
        config.tessellation.max_level = 16
        config.tessellation.mode = TessellationMode.DISTANCE_BASED

        # Aggressive LOD
        config.lod.distances = [200.0, 500.0, 1000.0, 2000.0]
        config.lod.enable_morphing = False

        # Enable all culling
        config.culling.mode = CullingMode.HIERARCHICAL

        # Reduce memory usage
        config.memory.max_loaded_tiles = 128
        config.memory.max_tile_cache_mb = 256

        # Performance optimizations
        config.performance.target_fps = 144
        config.performance.enable_gpu_driven_rendering = True
        config.performance.use_indirect_draws = True

        # Reduce visual quality
        config.rendering.enable_shadows = False
        config.rendering.enable_fog = False
        config.rendering.anisotropic_filtering = 4

        return config

    @classmethod
    def _balanced_preset(cls) -> "TerrainConfig":
        """Balanced preset - good compromise between performance and quality"""
        return cls()  # Default values are balanced

    @classmethod
    def _high_quality_preset(cls) -> "TerrainConfig":
        """High quality preset - prioritizes visual quality over performance"""
        config = cls()

        # Higher tessellation
        config.tessellation.base_level = 16
        config.tessellation.max_level = 64
        config.tessellation.mode = TessellationMode.SCREEN_SPACE

        # More conservative LOD
        config.lod.distances = [1000.0, 2000.0, 4000.0, 8000.0]
        config.lod.enable_morphing = True

        # More memory for quality
        config.memory.max_loaded_tiles = 512
        config.memory.max_tile_cache_mb = 1024

        # Enable advanced rendering features
        config.rendering.enable_shadows = True
        config.rendering.use_pbr_shading = True
        config.rendering.shadow_map_size = 4096
        config.rendering.anisotropic_filtering = 16

        # Lower target FPS for quality
        config.performance.target_fps = 60

        return config

    @classmethod
    def _mobile_preset(cls) -> "TerrainConfig":
        """Mobile preset - optimized for mobile/low-end devices"""
        config = cls()

        # Minimal tessellation
        config.tessellation.mode = TessellationMode.UNIFORM
        config.tessellation.base_level = 2
        config.tessellation.max_level = 4

        # Aggressive LOD
        config.lod.distances = [100.0, 250.0, 500.0, 1000.0]
        config.lod.enable_morphing = False

        # Minimal memory usage
        config.memory.max_loaded_tiles = 32
        config.memory.max_tile_cache_mb = 64
        config.memory.texture_cache_mb = 64

        # Disable expensive features
        config.rendering.enable_shadows = False
        config.rendering.enable_texture_blending = False
        config.rendering.anisotropic_filtering = 1

        # Mobile-friendly performance
        config.performance.target_fps = 30
        config.performance.enable_gpu_driven_rendering = False
        config.performance.use_indirect_draws = False

        return config

    @classmethod
    def _debug_preset(cls) -> "TerrainConfig":
        """Debug preset - optimized for development and debugging"""
        config = cls()

        # Simple rendering for debugging
        config.tessellation.mode = TessellationMode.UNIFORM
        config.tessellation.base_level = 1

        # Disable culling for debugging
        config.culling.mode = CullingMode.NONE

        # Enable debugging features
        config.performance.enable_profiling = True
        config.rendering.enable_shadows = False

        return config

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        # Convert to dictionary for JSON serialization
        config_dict = self._to_dict()

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "TerrainConfig":
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        return cls._from_dict(config_dict)

    def _to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        # This would need proper implementation for all nested dataclasses
        # For now, a simplified version
        return {
            "tile_size": self.tile_size,
            "height_scale": self.height_scale,
            "max_render_distance": self.max_render_distance,
            "tessellation_mode": self.tessellation.mode.value,
            "tessellation_base_level": self.tessellation.base_level,
            "lod_distances": self.lod.distances,
            "target_fps": self.performance.target_fps,
        }

    @classmethod
    def _from_dict(cls, config_dict: dict) -> "TerrainConfig":
        """Create configuration from dictionary"""
        # Simplified implementation
        config = cls()
        config.tile_size = config_dict.get("tile_size", config.tile_size)
        config.height_scale = config_dict.get("height_scale", config.height_scale)
        config.max_render_distance = config_dict.get(
            "max_render_distance", config.max_render_distance
        )

        # Would need to properly reconstruct all nested objects
        if "tessellation_mode" in config_dict:
            config.tessellation.mode = TessellationMode(
                config_dict["tessellation_mode"]
            )
        if "tessellation_base_level" in config_dict:
            config.tessellation.base_level = config_dict["tessellation_base_level"]
        if "lod_distances" in config_dict:
            config.lod.distances = config_dict["lod_distances"]
        if "target_fps" in config_dict:
            config.performance.target_fps = config_dict["target_fps"]

        return config

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Validate basic parameters
        if self.tile_size < 16 or self.tile_size > 1024:
            issues.append(f"tile_size ({self.tile_size}) should be between 16 and 1024")

        if not (self.tile_size & (self.tile_size - 1)) == 0:
            issues.append(f"tile_size ({self.tile_size}) should be power of 2")

        if self.height_scale <= 0:
            issues.append(f"height_scale ({self.height_scale}) must be positive")

        if self.max_render_distance <= 0:
            issues.append("max_render_distance must be positive")

        # Validate tessellation
        if self.tessellation.base_level < 1 or self.tessellation.base_level > 64:
            issues.append(
                f"tessellation base_level ({self.tessellation.base_level}) should be between 1 and 64"
            )

        if self.tessellation.max_level < self.tessellation.min_level:
            issues.append("tessellation max_level must be >= min_level")
        if self.tessellation.min_level < 1:
            issues.append("tessellation min_level must be >= 1")
        if self.tessellation.near_distance >= self.tessellation.far_distance:
            issues.append("near_distance must be less than far_distance")

        # Validate LOD distances are sorted
        if not all(
            self.lod.distances[i] <= self.lod.distances[i + 1]
            for i in range(len(self.lod.distances) - 1)
        ):
            issues.append("LOD distances must be in ascending order")

        # Validate memory limits
        if self.memory.max_tile_cache_mb < 64:
            issues.append("max_tile_cache_mb should be at least 64 MB")

        if self.memory.max_tile_cache_mb <= 0:
            issues.append("memory.max_tile_cache_mb must be positive")

        if self.memory.max_loaded_tiles < 16:
            issues.append("max_loaded_tiles should be at least 16")

        # Validate performance settings
        if self.performance.target_fps < 10 or self.performance.target_fps > 300:
            issues.append(
                f"target_fps ({self.performance.target_fps}) should be between 10 and 300"
            )

        return issues

    def optimize_for_hardware(
        self, gpu_name: str, vram_mb: int, cpu_cores: int
    ) -> None:
        """Automatically optimize configuration based on hardware specs"""
        if vram_mb >= 16384:
            self.tessellation.max_level = 64
        elif vram_mb >= 4096:
            self.tessellation.max_level = 32
        else:
            self.tessellation.max_level = 16

        cache_mb = max(vram_mb // 4, 256)
        self.memory.max_tile_cache_mb = min(cache_mb, 2048)

        self.performance.worker_threads = max(1, cpu_cores - 1)

    def get_estimated_memory_usage(self) -> dict:
        """Estimate memory usage for current configuration"""
        # Rough estimates in MB
        tile_memory = (
            self.memory.max_loaded_tiles
            * (self.tile_size * self.tile_size * 4 * 2)
            / (1024 * 1024)
        )  # vertices + indices
        texture_memory = self.memory.texture_cache_mb
        total_gpu_memory = (
            tile_memory + texture_memory + 100
        )  # +100MB for shaders, uniforms, etc.

        return {
            "tile_geometry_mb": tile_memory,
            "texture_cache_mb": texture_memory,
            "total_gpu_mb": total_gpu_memory,
            "system_ram_mb": 50,  # Estimated system RAM usage
        }
