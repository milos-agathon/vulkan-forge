"""
Vulkan Forge - Mesh and Asset Loaders

This module provides loaders for various mesh formats and 3D assets.
Supports OBJ, PLY, and other common formats with efficient parsing
optimized for Vulkan buffer uploads.
"""

from .obj_loader import ObjLoader, load_obj
from .mesh_formats import (
    Vertex, 
    Mesh, 
    MeshData,
    VertexFormat,
    IndexFormat
)

__all__ = [
    'ObjLoader',
    'load_obj',
    'Vertex',
    'Mesh', 
    'MeshData',
    'VertexFormat',
    'IndexFormat'
]

# Version info for loaders module
__version__ = '0.1.0'