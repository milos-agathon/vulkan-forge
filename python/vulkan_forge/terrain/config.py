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
        self._validate_levels()

    def _validate_levels(self) -> None:
        min_lv = getattr(self, "_min_level", None)
        max_lv = getattr(self, "_max_level", None)
        if min_lv is None or max_lv is None:
            return
        if max_lv < 1 or min_lv < 1 or min_lv > max_lv:
            raise ValueError("Invalid tessellation levels")
        if max_lv > 64:
            raise ValueError("max_level out of range")

    def get_tessellation_level(self, distance: float) -> int:
        """Compute tessellation level based on distance."""
        if distance < 0:
            raise ValueError("distance must be non-negative")
        ratio = max(0.0, min(distance / 1000.0, 1.0))
        level = round(self.max_level - (self.max_level - self.min_level) * ratio)
        return int(max(self.min_level, min(level, self.max_level)))


def _get_min_level(self) -> int:
    return self._min_level


def _set_min_level(self, value: int) -> None:
    old = getattr(self, "_min_level", None)
    self._min_level = value
    try:
        self._validate_levels()
    except Exception:
        if old is not None:
            self._min_level = old
        raise


def _get_max_level(self) -> int:
    return self._max_level


def _set_max_level(self, value: int) -> None:
    old = getattr(self, "_max_level", None)
    self._max_level = value
    try:
        self._validate_levels()
    except Exception:
        if old is not None:
            self._max_level = old
        raise


TessellationConfig.min_level = property(_get_min_level, _set_min_level)
TessellationConfig.max_level = property(_get_max_level, _set_max_level)


@dataclass
class LODConfig:
    """Level-of-detail configuration."""

    distances: List[float] = field(default_factory=lambda: [100.0, 200.0, 400.0])
    max_lod_levels: int = 4

    def __post_init__(self) -> None:
        self._validate_distances()

    def _validate_distances(self) -> None:
        if any(d <= 0 for d in self._distances):
            raise ValueError("distances must be positive")
        if any(
            self._distances[i] >= self._distances[i + 1]
            for i in range(len(self._distances) - 1)
        ):
            raise ValueError("distances must be strictly ascending")


def _get_distances(self) -> List[float]:
    return self._distances


def _set_distances(self, value: List[float]) -> None:
    old = getattr(self, "_distances", None)
    self._distances = list(value)
    try:
        self._validate_distances()
    except Exception:
        if old is not None:
            self._distances = old
        raise


LODConfig.distances = property(_get_distances, _set_distances)


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
        elif gpu_mem_mb < 2048:
            self.tessellation.max_level = min(self.tessellation.max_level, 32)
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
