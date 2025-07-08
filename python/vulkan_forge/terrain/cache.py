"""In-memory terrain tile cache with LRU eviction."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple

import numpy as np


class TerrainCache:
    """Cache of terrain tiles with size limits."""

    def __init__(self, max_tiles: int = 512, max_mb: int = 1024) -> None:
        self.max_tiles = max_tiles
        self.max_mb = max_mb
        self._tiles: "OrderedDict[Any, Tuple[np.ndarray, Dict[str, Any]]]" = (
            OrderedDict()
        )
        self._hits = 0
        self._misses = 0
        self._size_bytes = 0

    # -----------------------------------------------------
    @property
    def memory_used_mb(self) -> float:
        return self._size_bytes / (1024 * 1024)

    # -----------------------------------------------------
    def stats(self) -> Dict[str, float]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        miss_rate = self._misses / total if total else 0.0
        return {"hit_rate": hit_rate, "miss_rate": miss_rate, "total_requests": total}

    # -----------------------------------------------------
    def _evict_if_needed(self) -> None:
        while (len(self._tiles) > self.max_tiles) or (
            self.memory_used_mb > self.max_mb
        ):
            _, (tile, _) = self._tiles.popitem(last=False)
            self._size_bytes -= tile.nbytes

    # -----------------------------------------------------
    def get_tile(
        self, tile_id: Any, loader: Callable[[Any], Tuple[np.ndarray, Dict[str, Any]]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if tile_id in self._tiles:
            self._hits += 1
            tile = self._tiles.pop(tile_id)
            self._tiles[tile_id] = tile
            return tile

        self._misses += 1
        tile = loader(tile_id)
        if not isinstance(tile, tuple) or not isinstance(tile[0], np.ndarray):
            raise TypeError("loader must return (array, metadata)")
        self._tiles[tile_id] = tile
        self._size_bytes += tile[0].nbytes
        self._evict_if_needed()
        return tile
