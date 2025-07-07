#!/usr/bin/env python3
"""
Integration tests for Vulkan Forge mesh pipeline
Tests OBJ loading, vertex buffers, and high-performance rendering
"""

import pytest
import numpy as np
import time
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import vulkan_forge
    from vulkan_forge import VulkanEngine, MeshLoader, VertexBuffer
    from vulkan_forge.matrices import create_perspective_matrix, create_view_matrix, create_model_matrix
    VULKAN_FORGE_AVAILABLE = True
except ImportError as e:
    VULKAN_FORGE_AVAILABLE = False
    pytest.skip(f"Vulkan Forge not available: {e}", allow_module_level=True)


class TestMeshPipeline:
    """Test suite for mesh pipeline functionality"""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """Create and initialize Vulkan engine"""
        if not VULKAN_FORGE_AVAILABLE:
            pytest.skip("Vulkan Forge not available")
        
        try:
            engine = VulkanEngine()
            engine.initialize()
            yield engine
            engine.cleanup()
        except Exception as e:
            pytest.skip(f"Could not initialize Vulkan: {e}")
    
    @pytest.fixture
    def sample_obj_file(self):
        """Create a simple OBJ file for testing"""
        obj_content = """# Simple cube OBJ file for testing
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0

vn  0.0  0.0  1.0
vn  0.0  0.0 -1.0
vn  0.0  1.0  0.0
vn  0.0 -1.0  0.0
vn  1.0  0.0  0.0
vn -1.0  0.0  0.0

vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0

f 1/1/1 2/2/1 3/3/1 4/4/1
f 5/1/2 8/4/2 7/3/2 6/2/2
f 1/1/3 5/2/3 6/3/3 2/4/3
f 3/1/4 7/2/4 8/3/4 4/4/4
f 2/1/5 6/2/5 7/3/5 3/4/5
f 1/1/6 4/4/6 8/3/6 5/2/6
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(obj_content)
            obj_file = f.name
        
        yield obj_file
        
        # Cleanup
        try:
            os.unlink(obj_file)
        except OSError:
            pass
    
    @pytest.fixture
    def stanford_bunny_obj(self):
        """Create a simplified Stanford bunny for performance testing"""
        # Generate a simple bunny-like shape using parametric equations
        vertices = []
        normals = []
        tex_coords = []
        faces = []
        
        # Create a simplified bunny shape (ellipsoid with ears)
        phi_steps = 20
        theta_steps = 16
        
        for i in range(phi_steps):
            phi = (i / (phi_steps - 1)) * np.pi
            for j in range(theta_steps):
                theta = (j / theta_steps) * 2 * np.pi
                
                # Ellipsoid base shape
                x = 0.5 * np.sin(phi) * np.cos(theta)
                y = 0.7 * np.cos(phi) - 0.2  # Offset down slightly
                z = 0.5 * np.sin(phi) * np.sin(theta)
                
                # Add "ears" by modifying vertices near the top
                if phi < np.pi / 4:  # Top quarter
                    if abs(theta - np.pi/3) < np.pi/6:  # First ear
                        y += 0.3 * (1 - 4*phi/np.pi)
                        x += 0.2 * (1 - 4*phi/np.pi)
                    elif abs(theta - 5*np.pi/3) < np.pi/6:  # Second ear
                        y += 0.3 * (1 - 4*phi/np.pi)
                        x -= 0.2 * (1 - 4*phi/np.pi)
                
                vertices.append((x, y, z))
                
                # Calculate normal (approximate)
                normal_x = x / 0.5
                normal_y = (y + 0.2) / 0.7
                normal_z = z / 0.5
                length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
                if length > 0:
                    normals.append((normal_x/length, normal_y/length, normal_z/length))
                else:
                    normals.append((0, 1, 0))
                
                # Texture coordinates
                tex_coords.append((j / theta_steps, i / (phi_steps - 1)))
        
        # Generate faces (triangles)
        for i in range(phi_steps - 1):
            for j in range(theta_steps):
                # Current quad vertices
                v1 = i * theta_steps + j
                v2 = i * theta_steps + ((j + 1) % theta_steps)
                v3 = (i + 1) * theta_steps + j
                v4 = (i + 1) * theta_steps + ((j + 1) % theta_steps)
                
                # Two triangles per quad
                faces.append((v1 + 1, v2 + 1, v3 + 1))  # OBJ is 1-indexed
                faces.append((v2 + 1, v4 + 1, v3 + 1))
        
        # Write OBJ file
        obj_content = "# Simplified Stanford Bunny for testing\n"
        
        for x, y, z in vertices:
            obj_content += f"v {x:.6f} {y:.6f} {z:.6f}\n"
        
        for nx, ny, nz in normals:
            obj_content += f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n"
        
        for u, v in tex_coords:
            obj_content += f"vt {u:.6f} {v:.6f}\n"
        
        for v1, v2, v3 in faces:
            obj_content += f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(obj_content)
            bunny_file = f.name
        
        yield bunny_file, len(vertices), len(faces)
        
        # Cleanup
        try:
            os.unlink(bunny_file)
        except OSError:
            pass
    
    def test_mesh_loader_basic(self, sample_obj_file):
        """Test basic OBJ file loading"""
        loader = MeshLoader()
        
        # Load the mesh
        success = loader.load_obj(sample_obj_file)
        assert success, "Failed to load simple OBJ file"
        
        # Check mesh data
        vertices = loader.get_vertices()
        indices = loader.get_indices()
        normals = loader.get_normals()
        tex_coords = loader.get_tex_coords()
        
        assert len(vertices) > 0, "No vertices loaded"
        assert len(indices) > 0, "No indices loaded"
        assert len(normals) > 0, "No normals loaded"
        assert len(tex_coords) > 0, "No texture coordinates loaded"
        
        # Validate data shapes
        assert len(vertices) % 3 == 0, "Vertex data not multiple of 3"
        assert len(normals) % 3 == 0, "Normal data not multiple of 3"
        assert len(tex_coords) % 2 == 0, "Texture coordinate data not multiple of 2"
        
        print(f"Loaded mesh: {len(vertices)//3} vertices, {len(indices)//3} triangles")
    
    def test_vertex_buffer_creation(self, engine, sample_obj_file):
        """Test vertex buffer creation and upload"""
        loader = MeshLoader()
        loader.load_obj(sample_obj_file)
        
        # Create vertex buffer
        vertex_buffer = VertexBuffer(engine)
        
        # Upload mesh data
        vertices = np.array(loader.get_vertices(), dtype=np.float32)
        normals = np.array(loader.get_normals(), dtype=np.float32)
        tex_coords = np.array(loader.get_tex_coords(), dtype=np.float32)
        indices = np.array(loader.get_indices(), dtype=np.uint32)
        
        success = vertex_buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
        assert success, "Failed to upload mesh data to vertex buffer"
        
        # Validate buffer properties
        assert vertex_buffer.get_vertex_count() == len(vertices) // 3
        assert vertex_buffer.get_index_count() == len(indices)
        
        print(f"Uploaded to GPU: {vertex_buffer.get_vertex_count()} vertices")
    
    def test_mesh_pipeline_rendering(self, engine, sample_obj_file):
        """Test complete mesh rendering pipeline"""
        loader = MeshLoader()
        loader.load_obj(sample_obj_file)
        
        vertex_buffer = VertexBuffer(engine)
        
        # Upload mesh data
        vertices = np.array(loader.get_vertices(), dtype=np.float32)
        normals = np.array(loader.get_normals(), dtype=np.float32)
        tex_coords = np.array(loader.get_tex_coords(), dtype=np.float32)
        indices = np.array(loader.get_indices(), dtype=np.uint32)
        
        vertex_buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
        
        # Create matrices
        model_matrix = create_model_matrix(rotation_y=0.0)
        view_matrix = create_view_matrix(
            eye=[0, 0, 3],
            center=[0, 0, 0],
            up=[0, 1, 0]
        )
        proj_matrix = create_perspective_matrix(
            fov_degrees=60.0,
            aspect_ratio=16.0/9.0,
            near_plane=0.1,
            far_plane=100.0
        )
        
        # Render single frame
        start_time = time.time()
        
        success = engine.render_mesh(
            vertex_buffer=vertex_buffer,
            model_matrix=model_matrix,
            view_matrix=view_matrix,
            projection_matrix=proj_matrix,
            light_direction=[0.0, -1.0, -1.0]
        )
        
        frame_time = time.time() - start_time
        
        assert success, "Failed to render mesh"
        assert frame_time < 0.1, f"Frame time too high: {frame_time:.3f}s"
        
        print(f"Single frame render time: {frame_time*1000:.2f}ms")
    
    def test_performance_1000_fps(self, engine, stanford_bunny_obj):
        """Test high-performance rendering targeting 1000+ FPS"""
        bunny_file, vertex_count, triangle_count = stanford_bunny_obj
        
        loader = MeshLoader()
        loader.load_obj(bunny_file)
        
        vertex_buffer = VertexBuffer(engine)
        
        # Upload bunny mesh
        vertices = np.array(loader.get_vertices(), dtype=np.float32)
        normals = np.array(loader.get_normals(), dtype=np.float32)
        tex_coords = np.array(loader.get_tex_coords(), dtype=np.float32)
        indices = np.array(loader.get_indices(), dtype=np.uint32)
        
        vertex_buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
        
        # Use fast pipeline variant
        engine.set_pipeline_mode("fast")
        
        # Create matrices
        model_matrix = create_model_matrix()
        view_matrix = create_view_matrix([0, 0, 2], [0, 0, 0], [0, 1, 0])
        proj_matrix = create_perspective_matrix(60.0, 16.0/9.0, 0.1, 100.0)
        
        # Performance test - render many frames
        num_frames = 100
        start_time = time.time()
        
        for i in range(num_frames):
            # Slight rotation for each frame
            rotation = (i / num_frames) * 2 * np.pi
            current_model = create_model_matrix(rotation_y=rotation)
            
            success = engine.render_mesh(
                vertex_buffer=vertex_buffer,
                model_matrix=current_model,
                view_matrix=view_matrix,
                projection_matrix=proj_matrix,
                light_direction=[0.0, -1.0, -1.0]
            )
            
            assert success, f"Frame {i} render failed"
        
        total_time = time.time() - start_time
        fps = num_frames / total_time
        avg_frame_time = total_time / num_frames * 1000  # ms
        
        print(f"Performance test results:")
        print(f"  Mesh: {vertex_count} vertices, {triangle_count} triangles")
        print(f"  {num_frames} frames in {total_time:.3f}s")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average frame time: {avg_frame_time:.3f}ms")
        
        # Assert performance targets
        assert fps > 500, f"FPS too low: {fps:.1f} (target: >500)"
        
        # Stretch goal: 1000+ FPS
        if fps > 1000:
            print(f"  ✓ Achieved 1000+ FPS target!")
        else:
            print(f"  ! FPS below 1000 (achieved: {fps:.1f})")
    
    def test_memory_management(self, engine):
        """Test memory allocation and cleanup"""
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
    
    def test_error_handling(self, engine):
        """Test error handling for invalid inputs"""
        loader = MeshLoader()
        
        # Test invalid file
        success = loader.load_obj("nonexistent_file.obj")
        assert not success, "Should fail for nonexistent file"
        
        # Test empty vertex buffer
        vertex_buffer = VertexBuffer(engine)
        
        with pytest.raises((ValueError, RuntimeError)):
            vertex_buffer.upload_mesh_data([], [], [], [])
        
        # Test mismatched data sizes
        vertices = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 1 vertex
        normals = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 2 normals
        tex_coords = np.array([0.0, 0.0], dtype=np.float32)  # 1 tex coord
        indices = np.array([0], dtype=np.uint32)
        
        with pytest.raises((ValueError, RuntimeError)):
            vertex_buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
    
    def test_shader_validation(self, engine):
        """Test shader compilation and validation"""
        # Test that shaders are properly loaded
        shader_info = engine.get_shader_info()
        
        assert "mesh_vertex" in shader_info, "Mesh vertex shader not found"
        assert "mesh_fragment" in shader_info, "Mesh fragment shader not found"
        
        # Validate SPIR-V data
        for shader_name, info in shader_info.items():
            assert info["spirv_size"] > 0, f"Empty SPIR-V for {shader_name}"
            assert info["spirv_count"] > 0, f"No SPIR-V instructions for {shader_name}"
            
            # Check magic number
            if "spirv_data" in info:
                spirv_data = info["spirv_data"]
                assert len(spirv_data) >= 5, f"SPIR-V too short for {shader_name}"
                assert spirv_data[0] == 0x07230203, f"Invalid SPIR-V magic for {shader_name}"
        
        print(f"Validated {len(shader_info)} shaders")
    
    def test_multiple_meshes(self, engine, sample_obj_file):
        """Test rendering multiple meshes simultaneously"""
        loader = MeshLoader()
        loader.load_obj(sample_obj_file)
        
        # Create multiple vertex buffers with the same mesh
        num_meshes = 5
        vertex_buffers = []
        
        for i in range(num_meshes):
            buffer = VertexBuffer(engine)
            
            vertices = np.array(loader.get_vertices(), dtype=np.float32)
            normals = np.array(loader.get_normals(), dtype=np.float32)
            tex_coords = np.array(loader.get_tex_coords(), dtype=np.float32)
            indices = np.array(loader.get_indices(), dtype=np.uint32)
            
            buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
            vertex_buffers.append(buffer)
        
        # Render all meshes in one frame
        view_matrix = create_view_matrix([0, 0, 5], [0, 0, 0], [0, 1, 0])
        proj_matrix = create_perspective_matrix(60.0, 16.0/9.0, 0.1, 100.0)
        
        start_time = time.time()
        
        for i, buffer in enumerate(vertex_buffers):
            # Position meshes in a line
            x_offset = (i - num_meshes//2) * 2.0
            model_matrix = create_model_matrix(translation=[x_offset, 0, 0])
            
            success = engine.render_mesh(
                vertex_buffer=buffer,
                model_matrix=model_matrix,
                view_matrix=view_matrix,
                projection_matrix=proj_matrix,
                light_direction=[0.0, -1.0, -1.0]
            )
            
            assert success, f"Failed to render mesh {i}"
        
        render_time = time.time() - start_time
        
        print(f"Rendered {num_meshes} meshes in {render_time*1000:.2f}ms")
        
        # Cleanup
        for buffer in vertex_buffers:
            buffer.cleanup()
    
    def test_large_mesh_loading(self):
        """Test loading and processing large mesh data"""
        # Generate a large mesh (tessellated sphere)
        resolution = 50  # 50x50 grid on sphere
        vertices = []
        normals = []
        tex_coords = []
        indices = []
        
        # Generate sphere vertices
        for i in range(resolution + 1):
            lat = (i / resolution) * np.pi - np.pi/2  # -π/2 to π/2
            for j in range(resolution + 1):
                lon = (j / resolution) * 2 * np.pi  # 0 to 2π
                
                x = np.cos(lat) * np.cos(lon)
                y = np.sin(lat)
                z = np.cos(lat) * np.sin(lon)
                
                vertices.extend([x, y, z])
                normals.extend([x, y, z])  # Normal = position for unit sphere
                tex_coords.extend([j / resolution, i / resolution])
        
        # Generate indices
        for i in range(resolution):
            for j in range(resolution):
                # Current quad
                v0 = i * (resolution + 1) + j
                v1 = v0 + 1
                v2 = (i + 1) * (resolution + 1) + j
                v3 = v2 + 1
                
                # Two triangles
                indices.extend([v0, v1, v2])
                indices.extend([v1, v3, v2])
        
        vertex_count = len(vertices) // 3
        triangle_count = len(indices) // 3
        
        print(f"Generated large mesh: {vertex_count:,} vertices, {triangle_count:,} triangles")
        
        # Test data validation
        assert len(vertices) % 3 == 0
        assert len(normals) % 3 == 0
        assert len(tex_coords) % 2 == 0
        assert len(vertices) == len(normals)
        assert len(vertices) // 3 * 2 == len(tex_coords)
        
        # Test that all indices are valid
        max_index = max(indices)
        assert max_index < vertex_count, f"Invalid index {max_index} >= {vertex_count}"
        
        print("✓ Large mesh data validation passed")


class TestOBJLoader:
    """Specific tests for OBJ file parsing"""
    
    def test_complex_obj_features(self):
        """Test parsing OBJ files with various features"""
        complex_obj = """# Complex OBJ with groups and materials
mtllib materials.mtl

o Cube
g Front
usemtl FrontMaterial
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0

vn 0.0 0.0 1.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0

f 1/1/1 2/2/1 3/3/1 4/4/1

g Back
usemtl BackMaterial
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0

vn 0.0 0.0 -1.0
f 5/1/2 8/4/2 7/3/2 6/2/2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            f.write(complex_obj)
            obj_file = f.name
        
        try:
            loader = MeshLoader()
            success = loader.load_obj(obj_file)
            
            assert success, "Failed to load complex OBJ"
            
            # Check that groups are handled properly
            groups = loader.get_groups()
            assert len(groups) >= 2, "Groups not parsed correctly"
            
            print(f"Parsed {len(groups)} groups from complex OBJ")
            
        finally:
            os.unlink(obj_file)
    
    def test_malformed_obj_handling(self):
        """Test handling of malformed OBJ files"""
        malformed_objs = [
            # Missing vertex data
            "f 1 2 3\n",
            
            # Invalid face references
            "v 0 0 0\nf 1 2 5\n",
            
            # Malformed lines
            "v 1.0 2.0\nv 1.0 2.0 3.0\nf 1 2\n",
        ]
        
        loader = MeshLoader()
        
        for i, obj_content in enumerate(malformed_objs):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
                f.write(obj_content)
                obj_file = f.name
            
            try:
                success = loader.load_obj(obj_file)
                # Should either fail gracefully or load partial data
                if success:
                    vertices = loader.get_vertices()
                    indices = loader.get_indices()
                    print(f"Malformed OBJ {i}: Loaded {len(vertices)//3} vertices")
                else:
                    print(f"Malformed OBJ {i}: Failed gracefully")
                    
            finally:
                os.unlink(obj_file)


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for mesh pipeline"""
    
    def test_vertex_upload_performance(self, engine):
        """Benchmark vertex buffer upload performance"""
        sizes = [1000, 10000, 100000, 1000000]
        
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
            
            assert success, f"Upload failed for {vertex_count} vertices"
            
            data_size = vertices.nbytes + normals.nbytes + tex_coords.nbytes + indices.nbytes
            throughput = data_size / upload_time / (1024 * 1024)  # MB/s
            
            print(f"Upload {vertex_count:,} vertices: {upload_time*1000:.2f}ms, {throughput:.1f} MB/s")
            
            buffer.cleanup()
    
    def test_render_call_overhead(self, engine, sample_obj_file):
        """Benchmark render call overhead"""
        loader = MeshLoader()
        loader.load_obj(sample_obj_file)
        
        vertex_buffer = VertexBuffer(engine)
        vertices = np.array(loader.get_vertices(), dtype=np.float32)
        normals = np.array(loader.get_normals(), dtype=np.float32)
        tex_coords = np.array(loader.get_tex_coords(), dtype=np.float32)
        indices = np.array(loader.get_indices(), dtype=np.uint32)
        
        vertex_buffer.upload_mesh_data(vertices, normals, tex_coords, indices)
        
        # Prepare matrices
        model_matrix = create_model_matrix()
        view_matrix = create_view_matrix([0, 0, 2], [0, 0, 0], [0, 1, 0])
        proj_matrix = create_perspective_matrix(60.0, 16.0/9.0, 0.1, 100.0)
        
        # Benchmark many render calls
        num_calls = 1000
        start_time = time.time()
        
        for _ in range(num_calls):
            engine.render_mesh(
                vertex_buffer=vertex_buffer,
                model_matrix=model_matrix,
                view_matrix=view_matrix,
                projection_matrix=proj_matrix,
                light_direction=[0.0, -1.0, -1.0]
            )
        
        total_time = time.time() - start_time
        avg_call_time = total_time / num_calls * 1000000  # microseconds
        
        print(f"Render call overhead: {avg_call_time:.1f} μs per call")
        print(f"Theoretical max FPS: {1000000 / avg_call_time:.0f}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])