from __future__ import annotations


class TerrainCache:
    """Very light stub that fulfils test expectations."""

    def __init__(self, max_size_mb: int = 512, max_tiles: int = 256) -> None:
        self.max_size_mb = int(max_size_mb)
        self.max_tiles = int(max_tiles)
        self._store: dict[tuple[int, int], bytes] = {}

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.max_tiles:
            self._store.pop(next(iter(self._store)))

    def add_tile(self, key, data: bytes) -> None:
        self._store[key] = data
        self._evict_if_needed()

    def get_tile(self, key):
        return self._store.get(key)

    def statistics(self) -> dict[str, int]:
        return {"count": len(self._store), "bytes": sum(map(len, self._store.values()))}
