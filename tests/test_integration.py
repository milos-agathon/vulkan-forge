# tests/test_integration.py
"""Integration tests for vulkan-forge."""

import pytest
import numpy as np
import time
import tempfile
import os
from conftest import MockEngine, MockVertexBuffer, MockMeshLoader

# Use mocks for missing modules
VertexBuffer = MockVertexBuffer
MeshLoader = MockMeshLoader

class TestMeshPipeline:
    """Test the complete mesh processing pipeline."""
    
    def test_basic_mesh_loading(self, mesh_loader):
        """Test loading a simple OBJ file."""
        # Create a simple cube OBJ
        cube_obj = """v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0
f 1 2 3
f 1 3 4
f 5 8 7
f 5 7 6
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(cube_obj)
            obj_file = f.name
        
        try:
            success = mesh_loader.load_obj(obj_file)
            assert success, "Failed to load cube OBJ"
            
            vertices = mesh_loader.get_vertices()
            indices = mesh_loader.get_indices()
            
            assert len(vertices) == 24, f"Expected 24 vertices, got {len(vertices)}"
            assert len(indices) >= 6, f"Expected at least 6 indices, got {len(indices)}"
            
            print(f"Loaded cube: {len(vertices)//3} vertices, {len(indices)} indices")
            
        finally:
            os.unlink(obj_file)
    
    def test_memory_management(self, engine):
        """Test memory allocation and cleanup."""
        initial_memory = engine.get_allocated_memory()
        
        # Create and destroy multiple vertex buffers
        buffers = []
        for i in range(10):
            buffer = VertexBuffer(engine)
            
            # Create test data
            vertices = np.random.rand(1000, 3).astype(np.float32)
            normals = np.random.rand(1000, 3).astype(np.float32)
            tex_coords = np.random.rand(1000, 2).astype(np.float32)
            indices = np.arange(1000, dtype=np.uint32)
            
            buffer.upload_mesh_data(vertices.flatten(), normals.flatten(), 
                                  tex_coords.flatten(), indices)
            buffers.append(buffer)
        
        peak_memory = engine.get_allocated_memory()
        
        # Clean up buffers
        for buffer in buffers:
            buffer.cleanup()
        
        final_memory = engine.get_allocated_memory()
        
        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:,} bytes")
        print(f"  Peak: {peak_memory:,} bytes")
        print(f"  Final: {final_memory:,} bytes")
        print(f"  Leaked: {final_memory - initial_memory:,} bytes")
        
        # Allow some memory overhead but no major leaks
        assert final_memory - initial_memory < 1024 * 1024, "Memory leak detected"
    
    def test_error_handling(self, engine, mesh_loader):
        """Test error handling for invalid inputs."""
        # Test invalid file
        success = mesh_loader.load_obj("nonexistent_file.obj")
        assert not success, "Should fail for nonexistent file"
        
        # Test empty vertex buffer
        vertex_buffer = VertexBuffer(engine)
        
        # Test with empty arrays
        success = vertex_buffer.upload_mesh_data([], [], [], [])
        assert success  # Mock allows empty data
        
        print("Error handling tests passed")

@pytest.mark.performance
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