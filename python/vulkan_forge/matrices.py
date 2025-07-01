# vulkan_forge/matrices.py
"""Matrix operations for 3D transformations."""

import numpy as np
from typing import Tuple


class Matrix4x4:
    """4x4 transformation matrix."""
    
    def __init__(self, data: np.ndarray = None):
        """Initialize matrix with optional data."""
        if data is None:
            self.data = np.eye(4, dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
            if self.data.shape != (4, 4):
                raise ValueError("Matrix data must be 4x4")
    
    @classmethod
    def identity(cls) -> 'Matrix4x4':
        """Create identity matrix."""
        return cls()
    
    @classmethod
    def translation(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create translation matrix."""
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return cls(matrix)
    
    @classmethod
    def rotation_x(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around X axis (angle in radians)."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        matrix = np.eye(4, dtype=np.float32)
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_y(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Y axis (angle in radians)."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_z(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Z axis (angle in radians)."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        return cls(matrix)
    
    @classmethod
    def scale(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create scale matrix."""
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = x
        matrix[1, 1] = y
        matrix[2, 2] = z
        return cls(matrix)
    
    @classmethod
    def perspective(cls, fov: float, aspect: float, near: float, far: float) -> 'Matrix4x4':
        """Create perspective projection matrix."""
        f = 1.0 / np.tan(fov / 2.0)
        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = f / aspect
        matrix[1, 1] = f
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2 * far * near) / (near - far)
        matrix[3, 2] = -1
        return cls(matrix)
    
    @classmethod
    def look_at(cls, eye: Tuple[float, float, float], 
                target: Tuple[float, float, float],
                up: Tuple[float, float, float]) -> 'Matrix4x4':
        """Create look-at view matrix."""
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        # Calculate camera coordinate system
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        new_up = np.cross(right, forward)
        
        # Create view matrix
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, :3] = right
        matrix[1, :3] = new_up
        matrix[2, :3] = -forward
        matrix[0, 3] = -np.dot(right, eye)
        matrix[1, 3] = -np.dot(new_up, eye)
        matrix[2, 3] = np.dot(forward, eye)
        
        return cls(matrix)
    
    def __matmul__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        """Matrix multiplication."""
        return Matrix4x4(self.data @ other.data)
    
    def __mul__(self, scalar: float) -> 'Matrix4x4':
        """Scalar multiplication."""
        return Matrix4x4(self.data * scalar)
    
    def inverse(self) -> 'Matrix4x4':
        """Calculate matrix inverse."""
        return Matrix4x4(np.linalg.inv(self.data))
    
    def transpose(self) -> 'Matrix4x4':
        """Calculate matrix transpose."""
        return Matrix4x4(self.data.T)
    
    def transform_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform a 3D point."""
        p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        result = self.data @ p
        return (result[0], result[1], result[2])
    
    def transform_vector(self, vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform a 3D vector (no translation)."""
        v = np.array([vector[0], vector[1], vector[2], 0.0], dtype=np.float32)
        result = self.data @ v
        return (result[0], result[1], result[2])
    
    def __str__(self) -> str:
        """String representation of the matrix."""
        return str(self.data)