from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np


class VertexFormat(Enum):
    """Supported vertex formats for simple CPU tests."""

    POSITION_3F = ("xyz", 3, np.float32)
    POSITION_NORMAL_UV = ("xyz nrm uv", 8, np.float32)

    def __init__(self, components_desc: str, components: int, dtype: np.dtype) -> None:
        self._components_desc = components_desc
        self._components = components
        self._dtype = dtype

    @property
    def components(self) -> int:
        """Number of numeric components per vertex."""
        return self._components

    @property
    def stride(self) -> int:
        """Vertex byte stride."""
        return self._components * np.dtype(self._dtype).itemsize

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"VertexFormat.{self.name}"
