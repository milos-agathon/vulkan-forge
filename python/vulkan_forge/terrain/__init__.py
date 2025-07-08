"""Public terrain API."""

from .config import (
    TerrainConfig,
    TessellationConfig,
    LODConfig,
    CullingConfig,
    MemoryConfig,
    PerformanceConfig,
    RenderingConfig,
)
from .cache import TerrainCache


def __getattr__(name: str):
    if name == "TerrainRenderer":
        from ..terrain import TerrainRenderer

        return TerrainRenderer
    raise AttributeError(name)


__all__ = [
    "TerrainConfig",
    "TessellationConfig",
    "LODConfig",
    "CullingConfig",
    "MemoryConfig",
    "PerformanceConfig",
    "RenderingConfig",
    "TerrainCache",
    "TerrainRenderer",
]

__all__ = [n for n in globals() if not n.startswith("_")]
