"""
Vulkan Forge - Mesh Data Structures

Defines standard mesh data formats and vertex layouts compatible
with Vulkan vertex input descriptions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from enum import Enum


class VertexFormat(Enum):
    """Supported vertex attribute formats for Vulkan compatibility."""

    POSITION_3F = "position_3f"  # 3x float32 position
    POSITION_UV = "position_uv"  # 3x float32 position + 2x float32 UV
    POSITION_NORMAL = "position_normal"  # 3x float32 position + 3x float32 normal
    POSITION_NORMAL_UV = "position_normal_uv"  # Full vertex: pos + normal + UV
    POSITION_COLOR = "position_color"  # 3x float32 position + 4x uint8 color


class IndexFormat(Enum):
    """Supported index buffer formats."""

    UINT16 = np.uint16  # Up to 65,535 vertices
    UINT32 = np.uint32  # Up to 4,294,967,295 vertices


@dataclass
class Vertex:
    """
    Standard vertex structure for Vulkan rendering.

    All coordinates are in right-handed coordinate system:
    - X: right
    - Y: up
    - Z: forward (out of screen)
    """

    position: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]] = None
    uv: Optional[Tuple[float, float]] = None
    color: Optional[Tuple[int, int, int, int]] = None  # RGBA, 0-255

    def to_array(self, format_type: VertexFormat) -> np.ndarray:
        """Convert vertex to numpy array based on format."""
        if format_type == VertexFormat.POSITION_3F:
            return np.array(self.position, dtype=np.float32)
        elif format_type == VertexFormat.POSITION_UV:
            if self.uv is None:
                raise ValueError("UV coordinates required for POSITION_UV format")
            return np.array([*self.position, *self.uv], dtype=np.float32)
        elif format_type == VertexFormat.POSITION_NORMAL:
            if self.normal is None:
                raise ValueError("Normal required for POSITION_NORMAL format")
            return np.array([*self.position, *self.normal], dtype=np.float32)
        elif format_type == VertexFormat.POSITION_NORMAL_UV:
            if self.normal is None or self.uv is None:
                raise ValueError("Normal and UV required for POSITION_NORMAL_UV format")
            return np.array([*self.position, *self.normal, *self.uv], dtype=np.float32)
        elif format_type == VertexFormat.POSITION_COLOR:
            if self.color is None:
                raise ValueError("Color required for POSITION_COLOR format")
            # Pack color as normalized floats for shader compatibility
            color_norm = [c / 255.0 for c in self.color]
            return np.array([*self.position, *color_norm], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported vertex format: {format_type}")


@dataclass
class MeshData:
    """
    Raw mesh data container optimized for Vulkan upload.

    Contains numpy arrays ready for direct GPU buffer upload.
    All arrays are contiguous and properly aligned.
    """

    vertices: np.ndarray  # Vertex data array
    indices: np.ndarray  # Index array (triangles)
    vertex_format: VertexFormat
    index_format: IndexFormat

    # Optional metadata
    material_name: Optional[str] = None
    bounding_box: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None

    def __post_init__(self):
        """Validate mesh data after creation."""
        # Ensure arrays are contiguous for GPU upload
        if not self.vertices.flags["C_CONTIGUOUS"]:
            self.vertices = np.ascontiguousarray(self.vertices)
        if not self.indices.flags["C_CONTIGUOUS"]:
            self.indices = np.ascontiguousarray(self.indices)

        # Validate data types
        if self.vertices.dtype != np.float32:
            raise ValueError("Vertex data must be float32")
        if self.indices.dtype not in (np.uint16, np.uint32):
            raise ValueError("Index data must be uint16 or uint32")

        # Validate index range
        max_index = np.max(self.indices) if len(self.indices) > 0 else 0
        vertex_count = len(self.vertices) // self._get_vertex_stride()
        if max_index >= vertex_count:
            raise ValueError(f"Index {max_index} exceeds vertex count {vertex_count}")

    def _get_vertex_stride(self) -> int:
        """Get the number of floats per vertex based on format."""
        strides = {
            VertexFormat.POSITION_3F: 3,
            VertexFormat.POSITION_UV: 5,
            VertexFormat.POSITION_NORMAL: 6,
            VertexFormat.POSITION_NORMAL_UV: 8,
            VertexFormat.POSITION_COLOR: 7,
        }
        return strides[self.vertex_format]

    @property
    def vertex_count(self) -> int:
        """Get number of vertices in the mesh."""
        return len(self.vertices) // self._get_vertex_stride()

    @property
    def triangle_count(self) -> int:
        """Get number of triangles in the mesh."""
        return len(self.indices) // 3

    @property
    def index_count(self) -> int:
        """Get number of indices in the mesh."""
        return len(self.indices)

    @property
    def vertex_size_bytes(self) -> int:
        """Get size of vertex data in bytes."""
        return self.vertices.nbytes

    @property
    def index_size_bytes(self) -> int:
        """Get size of index data in bytes."""
        return self.indices.nbytes

    def compute_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute axis-aligned bounding box from vertex positions."""
        stride = self._get_vertex_stride()
        positions = self.vertices.reshape(-1, stride)[:, :3]  # Extract x,y,z

        min_pos = tuple(positions.min(axis=0))
        max_pos = tuple(positions.max(axis=0))

        self.bounding_box = (min_pos, max_pos)
        return self.bounding_box

    def compute_normals(self) -> None:
        """
        Compute face normals for meshes that don't have them.
        Only works with POSITION_3F format - converts to POSITION_NORMAL.
        """
        if self.vertex_format != VertexFormat.POSITION_3F:
            raise ValueError("Normal computation only supported for POSITION_3F format")

        # Extract positions
        positions = self.vertices.reshape(-1, 3)
        vertex_normals = np.zeros_like(positions)

        # Compute face normals and accumulate at vertices
        for i in range(0, len(self.indices), 3):
            i0, i1, i2 = self.indices[i : i + 3]

            v0, v1, v2 = positions[i0], positions[i1], positions[i2]

            # Cross product for face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            # Normalize
            length = np.linalg.norm(face_normal)
            if length > 1e-6:
                face_normal /= length

            # Accumulate at vertices
            vertex_normals[i0] += face_normal
            vertex_normals[i1] += face_normal
            vertex_normals[i2] += face_normal

        # Normalize vertex normals
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        lengths = np.where(lengths < 1e-6, 1.0, lengths)
        vertex_normals /= lengths

        # Combine positions and normals
        combined = np.hstack([positions, vertex_normals])
        self.vertices = combined.astype(np.float32).flatten()
        self.vertex_format = VertexFormat.POSITION_NORMAL


class Mesh:
    """
    High-level mesh container with additional metadata and operations.

    This is the user-facing mesh class that contains MeshData plus
    additional functionality like transformations, materials, etc.
    """

    def __init__(self, mesh_data: MeshData, name: str = "mesh", transform: Optional[np.ndarray] = None):
        """
        Create a mesh from mesh data.

        Args:
            mesh_data: Raw mesh data for GPU upload
            name: Display name for the mesh
            transform: 4x4 transformation matrix (optional)
        """
        self.data = mesh_data
        self.name = name
        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)

        # Cached GPU resources (set by renderer)
        self._vertex_buffer_handle = None
        self._index_buffer_handle = None
        self._is_uploaded = False

    @classmethod
    def from_vertices(
        cls, vertices: List[Vertex], indices: List[int], vertex_format: VertexFormat, name: str = "mesh"
    ) -> "Mesh":
        """Create mesh from list of vertices and indices."""
        # Convert vertices to array
        vertex_arrays = [v.to_array(vertex_format) for v in vertices]
        vertex_data = np.concatenate(vertex_arrays).astype(np.float32)

        # Convert indices to appropriate format
        max_index = max(indices) if indices else 0
        index_format = IndexFormat.UINT16 if max_index < 65536 else IndexFormat.UINT32
        index_data = np.array(indices, dtype=index_format.value)

        mesh_data = MeshData(
            vertices=vertex_data, indices=index_data, vertex_format=vertex_format, index_format=index_format
        )

        return cls(mesh_data, name)

    @property
    def is_uploaded(self) -> bool:
        """Check if mesh data has been uploaded to GPU."""
        return self._is_uploaded

    def mark_uploaded(self, vertex_handle, index_handle):
        """Mark mesh as uploaded with GPU buffer handles."""
        self._vertex_buffer_handle = vertex_handle
        self._index_buffer_handle = index_handle
        self._is_uploaded = True

    def get_info(self) -> str:
        """Get human-readable mesh information."""
        data = self.data
        return (
            f"Mesh '{self.name}': "
            f"{data.vertex_count} vertices, "
            f"{data.triangle_count} triangles, "
            f"format={data.vertex_format.value}, "
            f"uploaded={self.is_uploaded}"
        )

    def __str__(self) -> str:
        return self.get_info()

    def __repr__(self) -> str:
        return f"Mesh(name='{self.name}', vertices={self.data.vertex_count})"

    @property
    def vertex_count(self) -> int:
        return self.data.vertex_count

    @property
    def triangle_count(self) -> int:
        return self.data.triangle_count

    @property
    def index_count(self) -> int:
        return self.data.index_count
