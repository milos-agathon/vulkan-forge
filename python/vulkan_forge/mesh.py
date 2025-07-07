from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .vertex_format import VertexFormat


@dataclass
class Mesh:
    """Minimal mesh container used for CPU-based tests."""

    vertices: np.ndarray
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    vertex_format: Optional[VertexFormat] = None
    _bounding_box: Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        if self.vertices.ndim != 2:
            raise ValueError("vertices must be a 2D array")

        if self.vertex_format is None:
            cols = self.vertices.shape[1]
            if cols == 3:
                self.vertex_format = VertexFormat.POSITION_3F
            elif cols in (6, 8):
                self.vertex_format = VertexFormat.POSITION_NORMAL_UV
            else:
                raise ValueError(f"Cannot infer vertex format from {cols} columns")

        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float32)
            if self.normals.shape != (self.vertices.shape[0], 3):
                raise ValueError(
                    "normals must match vertex count and have 3 components"
                )

        if self.uvs is not None:
            self.uvs = np.asarray(self.uvs, dtype=np.float32)
            if self.uvs.shape != (self.vertices.shape[0], 2):
                raise ValueError("uvs must match vertex count and have 2 components")

        if self.indices is not None:
            self.indices = np.asarray(self.indices, dtype=np.int32)
            if self.indices.ndim == 2 and self.indices.shape[1] == 3:
                self.indices = self.indices.reshape(-1)
            elif self.indices.ndim != 1:
                raise ValueError("indices must be 1D array or Nx3 triangle array")
            self.indices = np.ascontiguousarray(self.indices, dtype=np.int32)

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    @property
    def data(self) -> "Mesh":
        """Return self for backwards compatibility with old API."""
        return self

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def index_count(self) -> int:
        return 0 if self.indices is None else int(self.indices.size)

    @property
    def bounding_box(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if self._bounding_box is None:
            pos = self.vertices[:, :3]
            min_p = tuple(pos.min(axis=0))
            max_p = tuple(pos.max(axis=0))
            self._bounding_box = (min_p, max_p)
        return self._bounding_box

    @property
    def triangle_count(self) -> int:
        return self.index_count // 3

    @property
    def vertex_size_bytes(self) -> int:
        return self.vertex_count * self.vertices.shape[1] * 4

    @property
    def index_size_bytes(self) -> int:
        return self.index_count * 4

    def compute_bounding_box(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        self._bounding_box = None
        return self.bounding_box
