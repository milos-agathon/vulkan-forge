"""Terrain configuration dataclasses with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List


@dataclass
class TessellationConfig:
    """GPU tessellation settings."""

    min_level: int = 1
    max_level: int = 8

    def __post_init__(self) -> None:
        if self.min_level < 1 or self.max_level < self.min_level:
            raise ValueError("Invalid tessellation levels")
        if self.max_level > 64:
            raise ValueError("max_level out of range")


@dataclass
class LODConfig:
    """Level-of-detail configuration."""

    distances: List[float] = field(default_factory=lambda: [100.0, 200.0, 400.0])
    max_lod_levels: int = 4

    def __post_init__(self) -> None:
        if any(d <= 0 for d in self.distances):
            raise ValueError("distances must be positive")
        if any(
            self.distances[i] >= self.distances[i + 1]
            for i in range(len(self.distances) - 1)
        ):
            raise ValueError("distances must be strictly ascending")


@dataclass
class CullingConfig:
    enable_frustum_culling: bool = True


@dataclass
class MemoryConfig:
    tile_cache_mb: int = 512


@dataclass
class RenderingConfig:
    enable_shadows: bool = False


@dataclass
class PerformanceConfig:
    target_fps: int = 60


@dataclass
class TerrainConfig:
    """Aggregate terrain configuration."""

    PRESETS: ClassVar[dict[str, "TerrainConfig"]]

    tessellation: TessellationConfig = field(default_factory=TessellationConfig)
    lod: LODConfig = field(default_factory=LODConfig)
    culling: CullingConfig = field(default_factory=CullingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def hardware_optimize(self, gpu_mem_mb: int) -> "TerrainConfig":
        if gpu_mem_mb < 1024:
            self.tessellation.max_level = min(self.tessellation.max_level, 8)
            self.lod.max_lod_levels = min(self.lod.max_lod_levels, 4)
        elif gpu_mem_mb < 4096:
            self.tessellation.max_level = min(self.tessellation.max_level, 16)
            self.lod.max_lod_levels = min(self.lod.max_lod_levels, 6)
        else:
            self.tessellation.max_level = min(self.tessellation.max_level, 32)
            self.lod.max_lod_levels = min(self.lod.max_lod_levels, 8)
        return self


TerrainConfig.PRESETS = {
    "high_performance": TerrainConfig(
        tessellation=TessellationConfig(max_level=16),
        lod=LODConfig(max_lod_levels=6),
    ),
    "balanced": TerrainConfig(),
    "high_quality": TerrainConfig(
        tessellation=TessellationConfig(max_level=32),
        lod=LODConfig(max_lod_levels=8),
        rendering=RenderingConfig(enable_shadows=True),
    ),
    "mobile": TerrainConfig(
        tessellation=TessellationConfig(max_level=4),
        lod=LODConfig(max_lod_levels=4),
        performance=PerformanceConfig(target_fps=30),
    ),
}
