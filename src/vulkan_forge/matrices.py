"""Matrix utilities for 3D transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Matrix4x4:
    """Simple 4x4 matrix wrapper."""

    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")

    def __matmul__(self, other: "Matrix4x4") -> "Matrix4x4":
        return Matrix4x4(self.data @ other.data)

    def to_list(self) -> list[list[float]]:
        from typing import cast

        return cast(list[list[float]], self.data.tolist())

    @staticmethod
    def identity() -> "Matrix4x4":
        return Matrix4x4(np.eye(4, dtype=np.float32))

    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> "Matrix4x4":
        f = 1.0 / np.tan(fov_y / 2)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1
        return Matrix4x4(m)

    @staticmethod
    def look_at(eye: Iterable[float], target: Iterable[float], up: Iterable[float]) -> "Matrix4x4":
        e = np.array(eye, dtype=np.float32)
        t = np.array(target, dtype=np.float32)
        u = np.array(up, dtype=np.float32)
        z = e - t
        z /= np.linalg.norm(z)
        x = np.cross(u, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = x
        m[1, :3] = y
        m[2, :3] = z
        m[:3, 3] = -e @ np.stack((x, y, z))
        return Matrix4x4(m)
