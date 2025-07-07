#!/usr/bin/env python3
"""
Test mesh loading functionality for Vulkan-Forge
Tests the OBJ loader → vertex buffer pipeline from the roadmap deliverable
"""

import pytest
import numpy as np
import tempfile
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import vulkan_forge as vf
    from vulkan_forge.loaders import load_obj, ObjLoader, Mesh, MeshData, Vertex, VertexFormat, IndexFormat
except ImportError as e:
    pytest.skip(f"vulkan_forge not available: {e}", allow_module_level=True)


class TestObjLoader:
    """Test OBJ file loading functionality."""
    
    def create_test_obj(self, vertices, faces, normals=None, uvs=None):
        """Create a temporary OBJ file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False)
        
        with temp_file as f:
            f.write("# Test OBJ file\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write normals if provided
            if normals:
                for n in normals:
                    f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
            
            # Write UVs if provided
            if uvs:
                for uv in uvs:
                    f.write(f"vt {uv[0]} {uv[1]}\n")
            
            # Write faces
            for face in faces:
                if normals and uvs:
                    # v/vt/vn format
                    face_str = " ".join([f"{v}/{v}/{v}" for v in face])
                elif uvs:
                    # v/vt format
                    face_str = " ".join([f"{v}/{v}" for v in face])
                elif normals:
                    # v//vn format
                    face_str = " ".join([f"{v}//{v}" for v in face])
                else:
                    # v format
                    face_str = " ".join([str(v) for v in face])
                
                f.write(f"f {face_str}\n")
        
        return Path(temp_file.name)
    
    def test_simple_triangle(self):
        """Test loading a simple triangle."""
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[1, 2, 3]]  # OBJ uses 1-based indexing
        
        obj_file = self.create_test_obj(vertices, faces)
        
        try:
            mesh = load_obj(obj_file)
            
            assert mesh is not None
            assert mesh.data.vertex_count == 3
            assert mesh.data.triangle_count == 1
            assert mesh.data.vertex_format == VertexFormat.POSITION_3F
            
            # Verify vertex data
            vertices_reshaped = mesh.data.vertices.reshape(-1, 3)
            np.testing.assert_allclose(vertices_reshaped[0], [0, 0, 0])
            np.testing.assert_allclose(vertices_reshaped[1], [1, 0, 0])
            np.testing.assert_allclose(vertices_reshaped[2], [0, 1, 0])
            
            # Verify indices
            expected_indices = [0, 1, 2]  # Converted to 0-based
            np.testing.assert_array_equal(mesh.data.indices, expected_indices)
            
        finally:
            obj_file.unlink()
    
    def test_cube_with_normals(self):
        """Test loading a cube with normals."""
        # Simple cube vertices
        vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # back face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # front face
        ]
        
        normals = [
            [0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]
        ]
        
        # Two triangles for one face
        faces = [[1, 2, 3], [1, 3, 4]]  # back face
        
        obj_file = self.create_test_obj(vertices, faces, normals=normals)
        
        try:
            mesh = load_obj(obj_file)
            
            assert mesh is not None
            assert mesh.data.vertex_count == 4  # 4 unique vertices used
            assert mesh.data.triangle_count == 2
            assert mesh.data.vertex_format == VertexFormat.POSITION_NORMAL
            
        finally:
            obj_file.unlink()
    
    def test_quad_triangulation(self):
        """Test that quads are properly triangulated."""
        vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        faces = [[1, 2, 3, 4]]  # Quad face
        
        obj_file = self.create_test_obj(vertices, faces)
        
        try:
            mesh = load_obj(obj_file, triangulate_quads=True)
            
            assert mesh is not None
            assert mesh.data.vertex_count == 4
            assert mesh.data.triangle_count == 2  # Quad split into 2 triangles
            
            # Should have 6 indices (2 triangles * 3 vertices each)
            assert len(mesh.data.indices) == 6
        finally:
            obj_file.unlink()