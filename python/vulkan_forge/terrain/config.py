"""Terrain configuration dataclasses with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List


@dataclass(init=False)
class TessellationConfig:
    """GPU tessellation settings."""

    min_level: int = 1
    max_level: int = 8
    near_distance: float = 100.0
    far_distance: float = 1000.0
    _strict: bool = True

    def __init__(
        self,
        min_level: int = 1,
        max_level: int = 8,
        near_distance: float = 100.0,
        far_distance: float = 1000.0,
        *,
        strict: bool = True,
    ) -> None:
        self._strict = strict
        self.min_level = min_level
        self.max_level = max_level
        self.near_distance = near_distance
        self.far_distance = far_distance
        self._dirty = False
        if not self._validate_levels() and self._strict:
            raise ValueError("Invalid tessellation levels")

    def _validate_levels(self) -> bool:
        min_lv = getattr(self, "_min_level", None)
        max_lv = getattr(self, "_max_level", None)
        if min_lv is None or max_lv is None:
            return True
        if max_lv < 1 or min_lv < 1 or min_lv > max_lv or max_lv > 64:
            return False
        return True

    def get_tessellation_level(self, distance: float) -> int:
        """Compute tessellation level using linear fall-off."""
        if distance < 0:
            raise ValueError("distance must be non-negative")
        if distance <= self.near_distance:
            return self.max_level
        if distance >= self.far_distance:
            return self.min_level
        span = max(self.far_distance - self.near_distance, 1e-6)
        ratio = (distance - self.near_distance) / span
        level = self.max_level - (self.max_level - self.min_level) * ratio
        return int(round(max(self.min_level, min(level, self.max_level))))


def _get_min_level(self) -> int:
    return self._min_level


def _set_min_level(self, value: int) -> None:
    self._min_level = value
    if not self._validate_levels():
        if self._strict:
            raise ValueError("Invalid tessellation levels")
        self._dirty = True


def _get_max_level(self) -> int:
    return self._max_level


def _set_max_level(self, value: int) -> None:
    self._max_level = value
    if not self._validate_levels():
        if self._strict:
            raise ValueError("Invalid tessellation levels")
        self._dirty = True


TessellationConfig.min_level = property(_get_min_level, _set_min_level)
TessellationConfig.max_level = property(_get_max_level, _set_max_level)


def _get_near_distance(self) -> float:
    return self._near_distance


def _set_near_distance(self, value: float) -> None:
    self._near_distance = float(value)
    if getattr(self, "far_distance", value) < self._near_distance:
        if self._strict:
            raise ValueError("near_distance cannot exceed far_distance")
        self._dirty = True


def _get_far_distance(self) -> float:
    return self._far_distance


def _set_far_distance(self, value: float) -> None:
    self._far_distance = float(value)
    if getattr(self, "near_distance", value) > self._far_distance:
        if self._strict:
            raise ValueError("far_distance must be >= near_distance")
        self._dirty = True


TessellationConfig.near_distance = property(_get_near_distance, _set_near_distance)
TessellationConfig.far_distance = property(_get_far_distance, _set_far_distance)


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
    vals = list(value)
    if vals != sorted(vals):
        raise ValueError("distances must be strictly ascending")
    if any(d <= 0 for d in vals):
        raise ValueError("distances must be positive")
    self._distances = vals


LODConfig.distances = property(_get_distances, _set_distances)


@dataclass
class CullingConfig:
    enable_frustum_culling: bool = True


@dataclass
class MemoryConfig:
    max_tile_cache_mb: int = 512


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

    tile_size: int = 256
    height_scale: float = 1.0
    max_render_distance: float = 10000.0

    tessellation: TessellationConfig = field(
        default_factory=lambda: TessellationConfig(strict=False)
    )
    lod: LODConfig = field(default_factory=LODConfig)
    culling: CullingConfig = field(default_factory=CullingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def optimize_for_hardware(
        self, gpu_name: str, vram_mb: int, cpu_cores: int
    ) -> "TerrainConfig":
        if "4090" in gpu_name:
            self.tessellation.max_level = 64
            self.memory.max_tile_cache_mb = max(self.memory.max_tile_cache_mb, 2048)
        elif "3070" in gpu_name:
            self.tessellation.max_level = 32
            self.memory.max_tile_cache_mb = max(self.memory.max_tile_cache_mb, 1024)
        else:
            if vram_mb < 4096:
                self.tessellation.max_level = min(self.tessellation.max_level, 16)
                self.memory.max_tile_cache_mb = min(self.memory.max_tile_cache_mb, 256)
            elif vram_mb < 12288:
                self.tessellation.max_level = min(self.tessellation.max_level, 32)
                self.memory.max_tile_cache_mb = min(self.memory.max_tile_cache_mb, 512)
            else:
                self.tessellation.max_level = min(self.tessellation.max_level, 64)
                self.memory.max_tile_cache_mb = max(self.memory.max_tile_cache_mb, 1024)

        self.performance.worker_threads = max(1, cpu_cores - 1)
        if "1660" in gpu_name:
            self.performance.enable_gpu_driven_rendering = False
        return self

    def validate(self) -> List[str]:
        issues: List[str] = []
        if self.tile_size <= 0:
            issues.append("tile_size must be positive")
        elif self.tile_size & (self.tile_size - 1):
            issues.append("tile_size must be power of two")
        if self.height_scale <= 0:
            issues.append("height_scale must be positive")
        if self.max_render_distance <= 0:
            issues.append("max_render_distance must be positive")
        if not self.tessellation._validate_levels():
            issues.append("invalid tessellation levels")
        return issues


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
