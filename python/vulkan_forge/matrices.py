# vulkan_forge/matrices.py
"""3D transformation matrix utilities."""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import math
@dataclass
class Matrix4x4:
    """4x4 transformation matrix."""
    
    data: np.ndarray
    
    def __init__(self, data: Optional[np.ndarray] = None):
        """Initialize matrix, defaults to identity."""
        if data is None:
            self.data = np.eye(4, dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32).reshape(4, 4)
    
    @classmethod
    def identity(cls) -> "Matrix4x4":
        """Create identity matrix."""
        return cls()
    
    @classmethod
    def translation(cls, x: float, y: float, z: float) -> "Matrix4x4":
        """Create translation matrix."""
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z
        return cls(mat)
    
    @classmethod
    def scale(cls, x: float, y: float, z: float) -> "Matrix4x4":
        """Create scale matrix."""
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = x
        mat[1, 1] = y
        mat[2, 2] = z
        return cls(mat)
    
    @classmethod
    def rotation_x(cls, angle: float) -> "Matrix4x4":
        """Create rotation matrix around X axis (angle in radians)."""
        c = math.cos(angle)
        s = math.sin(angle)
        mat = np.eye(4, dtype=np.float32)
        mat[1, 1] = c
        mat[1, 2] = -s
        mat[2, 1] = s
        mat[2, 2] = c
        return cls(mat)
    
    @classmethod
    def rotation_y(cls, angle: float) -> "Matrix4x4":
        """Create rotation matrix around Y axis (angle in radians)."""
        c = math.cos(angle)
        s = math.sin(angle)
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = c
        mat[0, 2] = s
        mat[2, 0] = -s
        mat[2, 2] = c
        return cls(mat)
    
    @classmethod
    def rotation_z(cls, angle: float) -> "Matrix4x4":
        """Create rotation matrix around Z axis (angle in radians)."""
        c = math.cos(angle)
        s = math.sin(angle)
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = c
        mat[0, 1] = -s
        mat[1, 0] = s
        mat[1, 1] = c
        return cls(mat)
    
    @classmethod
    def look_at(cls, eye: Tuple[float, float, float], 
                target: Tuple[float, float, float], 
                up: Tuple[float, float, float] = (0, 1, 0)) -> "Matrix4x4":
        """Create view matrix looking from eye to target."""
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        # Calculate basis vectors
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-10)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-10)
        
        up = np.cross(right, forward)
        
        # Build view matrix
        mat = np.eye(4, dtype=np.float32)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -forward
        mat[0, 3] = -np.dot(right, eye)
        mat[1, 3] = -np.dot(up, eye)
        mat[2, 3] = np.dot(forward, eye)
        
        return cls(mat)
    
    @classmethod
    def perspective(cls, fov: float, aspect: float, near: float, far: float) -> "Matrix4x4":
        """Create perspective projection matrix (fov in radians)."""
        f = 1.0 / math.tan(fov / 2.0)
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = (far + near) / (near - far)
        mat[2, 3] = (2 * far * near) / (near - far)
        mat[3, 2] = -1
        return cls(mat)
    
    @classmethod
    def orthographic(cls, left: float, right: float, bottom: float, 
                     top: float, near: float, far: float) -> "Matrix4x4":
        """Create orthographic projection matrix."""
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = 2 / (right - left)
        mat[1, 1] = 2 / (top - bottom)
        mat[2, 2] = -2 / (far - near)
        mat[0, 3] = -(right + left) / (right - left)
        mat[1, 3] = -(top + bottom) / (top - bottom)
        mat[2, 3] = -(far + near) / (far - near)
        return cls(mat)
    
    def __matmul__(self, other: "Matrix4x4") -> "Matrix4x4":
        """Matrix multiplication."""
        return Matrix4x4(self.data @ other.data)
    
    def inverse(self) -> "Matrix4x4":
        """Calculate inverse matrix."""
        return Matrix4x4(np.linalg.inv(self.data))
    
    def transpose(self) -> "Matrix4x4":
        """Calculate transpose matrix."""
        return Matrix4x4(self.data.T)