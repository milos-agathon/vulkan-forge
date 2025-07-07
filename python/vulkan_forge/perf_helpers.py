from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .vertex_format import VertexFormat


def create_performance_mesh(vertex_count: int, complexity: str = "simple") -> Mesh:
    """Generate a synthetic mesh for performance tests."""
    if complexity == "simple":
        vertices = np.random.random((vertex_count, 3)).astype(np.float32)
        fmt = VertexFormat.POSITION_3F
    elif complexity == "full":
        vertices = np.random.random((vertex_count, 8)).astype(np.float32)
        fmt = VertexFormat.POSITION_NORMAL_UV
    else:
        raise ValueError(f"Unknown complexity: {complexity}")

    triangle_count = vertex_count // 3
    indices = np.arange(triangle_count * 3, dtype=np.int32)
    return Mesh(vertices=vertices, indices=indices, vertex_format=fmt)
