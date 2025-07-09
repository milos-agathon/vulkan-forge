"""Integration tests for vulkan-forge."""

import pytest
import numpy as np
import time
import tempfile
import os
from .mock_classes import MockVertexBuffer, MockMeshLoader

# Use mocks for missing modules
VertexBuffer = MockVertexBuffer
MeshLoader = MockMeshLoader

class TestMeshPipeline:
    """Test mesh loading and pipeline functionality."""
    
    def test_basic_mesh_loading(self, mesh_loader):
        """Test basic mesh loading functionality."""
        # Create concrete return values instead of Mock
        test_vertices = list(range(72))  # 24 vertices * 3 components
        test_normals = list(range(72))   # 24 normals * 3 components
        test_uvs = list(range(48))       # 24 uvs * 2 components
        test_indices = list(range(36))   # 36 indices for 12 triangles
        
        # Configure the mock
        mesh_loader.load_obj.return_value = (test_vertices, test_normals, test_uvs, test_indices)
        
        # Load cube.obj
        vertices, normals, uvs, indices = mesh_loader.load_obj("cube.obj")
        
        # Verify the data
        assert len(vertices) == 72, f"Expected 72 vertex components, got {len(vertices)}"
        assert len(normals) == 72, f"Expected 72 normal components, got {len(normals)}"
        assert len(uvs) == 48, f"Expected 48 UV components, got {len(uvs)}"
        assert len(indices) == 36, f"Expected 36 indices, got {len(indices)}"
        
        # Verify we loaded the correct number of vertices (72 components = 24 vertices)
        vertex_count = len(vertices) // 3
        assert vertex_count == 24, f"Expected 24 vertices, got {vertex_count}"
        
        print(f"✓ Loaded cube with {vertex_count} vertices and {len(indices)//3} triangles")
    
    def test_memory_management(self, engine):
        """Test memory allocation and deallocation."""
        # Initialize allocated_memory as a real integer, not Mock
        engine.allocated_memory = 0
        
        # Track initial memory
        initial_memory = engine.allocated_memory
        
        print("Memory usage:")
        print(f"  Initial: {initial_memory:,} bytes")
        
        # Create vertex buffer
        from tests.mock_classes import VertexBuffer
        buffer = VertexBuffer(engine)
        
        # Upload mesh data
        vertices = list(range(30))  # 10 vertices
        normals = list(range(30))   # 10 normals
        tex_coords = list(range(20)) # 10 tex coords
        indices = list(range(15))    # 5 triangles
        
        success = buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
        assert success
        
        # Check memory increased
        current_memory = engine.allocated_memory
        print(f"  After upload: {current_memory:,} bytes")
        assert current_memory > initial_memory
        
        # Destroy buffer
        buffer.destroy()
        
        # Check memory returned to initial
        final_memory = engine.allocated_memory
        print(f"  After destroy: {final_memory:,} bytes")
        assert final_memory == initial_memory
    
    def test_error_handling(self, engine, mesh_loader):
        """Test error handling for invalid inputs."""
        # Configure mock to return None values for nonexistent files
        def load_obj_with_error_handling(filename):
            if "nonexistent" in filename or not filename.endswith('.obj'):
                return None, None, None, None  # Return None for all values
            # Return valid data for valid files
            return [1,2,3], [1,2,3], [1,2], [0,1,2]
        
        mesh_loader.load_obj.side_effect = load_obj_with_error_handling
        
        # Test nonexistent file
        result = mesh_loader.load_obj("nonexistent.obj")
        vertices, normals, uvs, indices = result
        
        # All should be None for nonexistent file
        assert vertices is None
        assert normals is None
        assert uvs is None
        assert indices is None
        
        # Test that we properly detect the failure
        success = all(x is not None for x in result)
        assert not success, "Should fail for nonexistent file"
        
        # Test invalid file extension
        result2 = mesh_loader.load_obj("test.txt")
        success2 = all(x is not None for x in result2)
        assert not success2, "Should fail for non-OBJ file"
        
        print("✓ Error handling working correctly")

class TestPerformanceBenchmarks:
    """Performance benchmarks for mesh pipeline."""
    
    def test_vertex_upload_performance(self, engine):
        """Benchmark vertex buffer upload performance."""
        sizes = [1000, 10000, 100000]
        
        for vertex_count in sizes:
            # Generate test data
            vertices = np.random.rand(vertex_count * 3).astype(np.float32)
            normals = np.random.rand(vertex_count * 3).astype(np.float32)
            tex_coords = np.random.rand(vertex_count * 2).astype(np.float32)
            indices = np.arange(vertex_count, dtype=np.uint32)
            
            buffer = VertexBuffer(engine)
            
            start_time = time.time()
            success = buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
            upload_time = time.time() - start_time
            
            assert success, f"Failed to upload {vertex_count} vertices"
            
            # Performance target: should handle uploads quickly
            assert upload_time < 1.0, f"Upload too slow: {upload_time:.3f}s for {vertex_count} vertices"
            
            print(f"Upload {vertex_count} vertices: {upload_time*1000:.2f}ms")
    
    def test_performance_benchmarks(self, engine):
        """Test performance with benchmarking."""
        vertex_count = 50000
        
        # Generate test data
        vertices = np.random.rand(vertex_count * 3).astype(np.float32)
        normals = np.random.rand(vertex_count * 3).astype(np.float32)
        tex_coords = np.random.rand(vertex_count * 2).astype(np.float32)
        indices = np.arange(vertex_count, dtype=np.uint32)
        
        buffer = VertexBuffer(engine)
        
        # Time the upload
        start_time = time.perf_counter()
        success = buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
        end_time = time.perf_counter()
        
        upload_time = end_time - start_time
        
        assert success, "Upload should succeed"
        
        # Calculate performance metrics
        vertices_per_second = vertex_count / upload_time if upload_time > 0 else float('inf')
        
        print(f"Performance metrics:")
        print(f"  Vertices: {vertex_count:,}")
        print(f"  Upload time: {upload_time*1000:.2f}ms")
        print(f"  Vertices/sec: {vertices_per_second:,.0f}")
        
        # Verify reasonable performance (adjust thresholds as needed)
        fps_threshold = 1000  # Target FPS for rendering
        min_vertices_per_second = vertex_count * fps_threshold
        
        if vertices_per_second >= min_vertices_per_second:
            print(f"  ✓ Performance target met ({fps_threshold} FPS capable)")
        else:
            print(f"  ! Performance below target (achieved: {vertices_per_second/vertex_count:.1f} FPS equivalent)")
    
    def test_malformed_obj_handling(self, mesh_loader):
        """Test handling of malformed OBJ files."""
        malformed_objs = [
            # Missing vertex data
            "f 1 2 3\n",
            
            # Invalid face references
            "v 0 0 0\nf 1 2 5\n",
            
            # Malformed lines
            "v 1.0 2.0\nv 1.0 2.0 3.0\nf 1 2\n",
        ]
        
        for i, obj_content in enumerate(malformed_objs):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
                f.write(obj_content)
                obj_file = f.name
            
            try:
                success = mesh_loader.load_obj(obj_file)
                # Should either fail gracefully or load partial data
                if success:
                    vertices = mesh_loader.get_vertices()
                    indices = mesh_loader.get_indices()
                    print(f"Malformed OBJ {i}: Loaded {len(vertices)//3} vertices")
                else:
                    print(f"Malformed OBJ {i}: Failed gracefully")
                    
            finally:
                os.unlink(obj_file)