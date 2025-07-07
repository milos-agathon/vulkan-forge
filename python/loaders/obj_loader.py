"""
Vulkan Forge - OBJ File Loader

High-performance OBJ file parser optimized for Vulkan rendering.
Supports positions, normals, texture coordinates, and face indexing.

Features:
- Memory-efficient parsing with numpy arrays
- Automatic normal generation for meshes without normals
- Triangulation of quad faces
- Proper index handling for Vulkan
- Material group support
"""

import os
import re
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path

from .mesh_formats import Mesh, MeshData, Vertex, VertexFormat, IndexFormat


class ObjParseError(Exception):
    """Exception raised for OBJ parsing errors."""
    pass


class ObjLoader:
    """
    Fast OBJ file loader with Vulkan-optimized output.
    
    Parses OBJ files and converts them to MeshData structures
    ready for GPU upload. Handles complex OBJ features like
    multiple objects, materials, and mixed face types.
    """
    
    def __init__(self, 
                 generate_normals: bool = True,
                 triangulate_quads: bool = True,
                 merge_vertices: bool = True,
                 scale: float = 1.0):
        """
        Initialize OBJ loader with options.
        
        Args:
            generate_normals: Generate normals if not present in file
            triangulate_quads: Convert quad faces to triangles
            merge_vertices: Merge duplicate vertices to reduce memory
            scale: Uniform scale factor applied to all vertices
        """
        self.generate_normals = generate_normals
        self.triangulate_quads = triangulate_quads
        self.merge_vertices = merge_vertices
        self.scale = scale
        
        # Statistics from last load
        self.stats = {
            'vertices_loaded': 0,
            'faces_loaded': 0,
            'normals_loaded': 0,
            'uvs_loaded': 0,
            'materials_found': 0,
            'parsing_time': 0.0
        }
    
    def load(self, filepath: Union[str, Path]) -> Mesh:
        """
        Load a single mesh from OBJ file.
        
        Args:
            filepath: Path to .obj file
            
        Returns:
            Mesh object ready for rendering
            
        Raises:
            ObjParseError: If file cannot be parsed
            FileNotFoundError: If file doesn't exist
        """
        import time
        start_time = time.perf_counter()
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"OBJ file not found: {filepath}")
        
        try:
            # Parse the file
            vertices, normals, uvs, faces, materials = self._parse_obj_file(filepath)
            
            # Convert to mesh data
            mesh_data = self._build_mesh_data(vertices, normals, uvs, faces)
            
            # Create mesh
            mesh = Mesh(mesh_data, name=filepath.stem)
            
            # Update statistics
            self.stats['parsing_time'] = time.perf_counter() - start_time
            
            return mesh
            
        except Exception as e:
            raise ObjParseError(f"Failed to parse OBJ file {filepath}: {str(e)}") from e
    
    def load_multiple(self, filepath: Union[str, Path]) -> List[Mesh]:
        """
        Load multiple meshes from OBJ file (one per object/group).
        
        Args:
            filepath: Path to .obj file
            
        Returns:
            List of Mesh objects, one per object in file
        """
        # For now, return single mesh - can extend later for multi-object support
        return [self.load(filepath)]
    
    def _parse_obj_file(self, filepath: Path) -> Tuple[List, List, List, List, Dict]:
        """Parse OBJ file and extract raw data."""
        vertices = []      # Vertex positions
        normals = []       # Vertex normals  
        uvs = []          # Texture coordinates
        faces = []        # Face definitions
        materials = {}    # Material definitions
        
        current_material = None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                    
                cmd = parts[0]
                
                try:
                    if cmd == 'v':  # Vertex position
                        if len(parts) < 4:
                            raise ObjParseError(f"Invalid vertex at line {line_num}")
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append((x * self.scale, y * self.scale, z * self.scale))
                        
                    elif cmd == 'vn':  # Vertex normal
                        if len(parts) < 4:
                            raise ObjParseError(f"Invalid normal at line {line_num}")
                        nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                        normals.append((nx, ny, nz))
                        
                    elif cmd == 'vt':  # Texture coordinate
                        if len(parts) < 3:
                            raise ObjParseError(f"Invalid UV at line {line_num}")
                        u, v = float(parts[1]), float(parts[2])
                        uvs.append((u, v))
                        
                    elif cmd == 'f':  # Face definition
                        if len(parts) < 4:
                            raise ObjParseError(f"Invalid face at line {line_num}")
                        
                        face_vertices = []
                        for vertex_str in parts[1:]:
                            face_vertices.append(self._parse_face_vertex(vertex_str))
                        
                        # Handle triangulation
                        if len(face_vertices) == 3:
                            faces.append(face_vertices)
                        elif len(face_vertices) == 4 and self.triangulate_quads:
                            # Split quad into two triangles: (0,1,2) and (0,2,3)
                            faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
                            faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])
                        elif len(face_vertices) > 4:
                            # Fan triangulation for n-gons
                            for i in range(1, len(face_vertices) - 1):
                                faces.append([face_vertices[0], face_vertices[i], face_vertices[i + 1]])
                        else:
                            raise ObjParseError(f"Unsupported face type at line {line_num}")
                            
                    elif cmd == 'usemtl':  # Material usage
                        current_material = parts[1] if len(parts) > 1 else None
                        
                except (ValueError, IndexError) as e:
                    raise ObjParseError(f"Parse error at line {line_num}: {str(e)}")
        
        # Update statistics
        self.stats.update({
            'vertices_loaded': len(vertices),
            'faces_loaded': len(faces),
            'normals_loaded': len(normals),
            'uvs_loaded': len(uvs),
            'materials_found': len(materials)
        })
        
        return vertices, normals, uvs, faces, materials
    
    def _parse_face_vertex(self, vertex_str: str) -> Tuple[int, Optional[int], Optional[int]]:
        """
        Parse face vertex string like "v", "v/vt", "v/vt/vn", or "v//vn".
        
        Returns:
            Tuple of (vertex_idx, uv_idx, normal_idx) - 1-based indices
        """
        parts = vertex_str.split('/')
        
        # Vertex index (required)
        v_idx = int(parts[0])
        
        # UV index (optional)
        uv_idx = None
        if len(parts) > 1 and parts[1]:
            uv_idx = int(parts[1])
        
        # Normal index (optional)  
        normal_idx = None
        if len(parts) > 2 and parts[2]:
            normal_idx = int(parts[2])
            
        return (v_idx, uv_idx, normal_idx)
    
    def _build_mesh_data(self, 
                        vertices: List[Tuple[float, float, float]],
                        normals: List[Tuple[float, float, float]],
                        uvs: List[Tuple[float, float]],
                        faces: List[List[Tuple[int, Optional[int], Optional[int]]]]) -> MeshData:
        """Convert parsed OBJ data to MeshData structure."""
        
        if not vertices:
            raise ObjParseError("No vertices found in OBJ file")
        
        if not faces:
            raise ObjParseError("No faces found in OBJ file")
        
        # Determine vertex format based on available data
        has_normals = len(normals) > 0
        has_uvs = len(uvs) > 0
        
        if has_normals and has_uvs:
            vertex_format = VertexFormat.POSITION_NORMAL_UV
        elif has_normals:
            vertex_format = VertexFormat.POSITION_NORMAL
        elif has_uvs:
            vertex_format = VertexFormat.POSITION_UV
        else:
            vertex_format = VertexFormat.POSITION_3F
        
        # Build vertex list and index mapping
        unique_vertices = []
        vertex_indices = []
        vertex_map = {}  # Maps (v,vt,vn) tuple to final index
        
        for face in faces:
            face_indices = []
            
            for v_idx, uv_idx, normal_idx in face:
                # Convert to 0-based indices
                v_idx -= 1
                uv_idx = (uv_idx - 1) if uv_idx is not None else None
                normal_idx = (normal_idx - 1) if normal_idx is not None else None
                
                # Validate indices
                if v_idx < 0 or v_idx >= len(vertices):
                    raise ObjParseError(f"Vertex index {v_idx + 1} out of range")
                
                if uv_idx is not None and (uv_idx < 0 or uv_idx >= len(uvs)):
                    raise ObjParseError(f"UV index {uv_idx + 1} out of range")
                
                if normal_idx is not None and (normal_idx < 0 or normal_idx >= len(normals)):
                    raise ObjParseError(f"Normal index {normal_idx + 1} out of range")
                
                # Create vertex key for deduplication
                if self.merge_vertices:
                    vertex_key = (v_idx, uv_idx, normal_idx)
                    
                    if vertex_key in vertex_map:
                        # Reuse existing vertex
                        final_idx = vertex_map[vertex_key]
                    else:
                        # Create new vertex
                        final_idx = len(unique_vertices)
                        vertex_map[vertex_key] = final_idx
                        
                        # Build vertex
                        vertex = self._build_vertex(
                            vertices[v_idx],
                            normals[normal_idx] if normal_idx is not None else None,
                            uvs[uv_idx] if uv_idx is not None else None
                        )
                        unique_vertices.append(vertex)
                else:
                    # No deduplication - each face vertex is unique
                    final_idx = len(unique_vertices)
                    vertex = self._build_vertex(
                        vertices[v_idx],
                        normals[normal_idx] if normal_idx is not None else None,
                        uvs[uv_idx] if uv_idx is not None else None
                    )
                    unique_vertices.append(vertex)
                
                face_indices.append(final_idx)
            
            vertex_indices.extend(face_indices)
        
        # Convert vertices to numpy array
        vertex_arrays = [v.to_array(vertex_format) for v in unique_vertices]
        vertex_data = np.concatenate(vertex_arrays).astype(np.float32)
        
        # Convert indices to numpy array
        max_index = max(vertex_indices) if vertex_indices else 0
        index_format = IndexFormat.UINT16 if max_index < 65536 else IndexFormat.UINT32
        index_data = np.array(vertex_indices, dtype=index_format.value)
        
        # Create mesh data
        mesh_data = MeshData(
            vertices=vertex_data,
            indices=index_data,
            vertex_format=vertex_format,
            index_format=index_format
        )
        
        # Generate normals if needed and not present
        if vertex_format == VertexFormat.POSITION_3F and self.generate_normals:
            mesh_data.compute_normals()
        
        # Compute bounding box
        mesh_data.compute_bounding_box()
        
        return mesh_data
    
    def _build_vertex(self, 
                     position: Tuple[float, float, float],
                     normal: Optional[Tuple[float, float, float]] = None,
                     uv: Optional[Tuple[float, float]] = None) -> Vertex:
        """Build a Vertex object from components."""
        return Vertex(
            position=position,
            normal=normal,
            uv=uv
        )
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics from the last load operation."""
        return self.stats.copy()


def load_obj(filepath: Union[str, Path], 
             generate_normals: bool = True,
             triangulate_quads: bool = True,
             merge_vertices: bool = True,
             scale: float = 1.0) -> Mesh:
    """
    Convenience function to load a single mesh from OBJ file.
    
    Args:
        filepath: Path to .obj file
        generate_normals: Generate normals if not present
        triangulate_quads: Convert quad faces to triangles
        merge_vertices: Merge duplicate vertices
        scale: Uniform scale factor
        
    Returns:
        Mesh object ready for rendering
        
    Example:
        >>> mesh = load_obj("models/bunny.obj")
        >>> print(f"Loaded {mesh.data.vertex_count} vertices")
    """
    loader = ObjLoader(
        generate_normals=generate_normals,
        triangulate_quads=triangulate_quads,
        merge_vertices=merge_vertices,
        scale=scale
    )
    return loader.load(filepath)


def create_test_mesh() -> Mesh:
    """
    Create a simple test mesh (cube) for development and testing.
    
    Returns:
        Mesh containing a unit cube centered at origin
    """
    # Cube vertices
    vertices = [
        # Front face
        Vertex((-0.5, -0.5,  0.5), (0, 0, 1), (0, 0)),  # 0
        Vertex(( 0.5, -0.5,  0.5), (0, 0, 1), (1, 0)),  # 1
        Vertex(( 0.5,  0.5,  0.5), (0, 0, 1), (1, 1)),  # 2
        Vertex((-0.5,  0.5,  0.5), (0, 0, 1), (0, 1)),  # 3
        
        # Back face
        Vertex((-0.5, -0.5, -0.5), (0, 0, -1), (1, 0)),  # 4
        Vertex((-0.5,  0.5, -0.5), (0, 0, -1), (1, 1)),  # 5
        Vertex(( 0.5,  0.5, -0.5), (0, 0, -1), (0, 1)),  # 6
        Vertex(( 0.5, -0.5, -0.5), (0, 0, -1), (0, 0)),  # 7
        
        # Additional faces would go here...
        # Simplified cube for testing
    ]
    
    # Cube indices (triangles)
    indices = [
        # Front face
        0, 1, 2,  0, 2, 3,
        # Back face  
        4, 5, 6,  4, 6, 7,
        # More faces would go here for complete cube...
    ]
    
    return Mesh.from_vertices(
        vertices=vertices,
        indices=indices,
        vertex_format=VertexFormat.POSITION_NORMAL_UV,
        name="test_cube"
    )


if __name__ == "__main__":
    # Quick test of the loader
    test_mesh = create_test_mesh()
    print(f"Created test mesh: {test_mesh}")
    print(f"Vertex format: {test_mesh.data.vertex_format}")
    print(f"Vertices: {test_mesh.data.vertex_count}")
    print(f"Triangles: {test_mesh.data.triangle_count}")