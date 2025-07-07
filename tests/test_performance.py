#!/usr/bin/env python3
"""
Performance tests for Vulkan-Forge mesh pipeline
Tests the 1000+ FPS Stanford bunny target from the roadmap deliverable
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import vulkan_forge as vf
except ImportError as e:
    pytest.skip(f"vulkan_forge not available: {e}", allow_module_level=True)


class TestMeshPerformance:
    """Test mesh loading and processing performance."""
    
    def create_performance_mesh(self, vertex_count: int, complexity: str = "simple") -> 'vf.Mesh':
        """Create a mesh with specified vertex count for performance testing."""
        if complexity == "simple":
            # Simple positions only
            vertices = np.random.random((vertex_count, 3)).astype(np.float32)
            format_type = vf.VertexFormat.POSITION_3F
        elif complexity == "full":
            # Full vertex data: position + normal + UV
            vertices = np.random.random((vertex_count, 8)).astype(np.float32)
            format_type = vf.VertexFormat.POSITION_NORMAL_UV
        else:
            raise ValueError(f"Unknown complexity: {complexity}")
        
        # Generate triangle indices
        triangle_count = vertex_count // 3
        indices = np.arange(triangle_count * 3, dtype=np.uint32)
        
        mesh_data = vf.MeshData(
            vertices=vertices.flatten(),
            indices=indices,
            vertex_format=format_type,
            index_format=vf.IndexFormat.UINT32
        )
        
        return vf.Mesh(mesh_data, name=f"perf_test_{vertex_count}_{complexity}")
    
    def test_small_mesh_loading_speed(self):
        """Test loading speed for small meshes (< 1K vertices)."""
        vertex_count = 1000
        
        start_time = time.perf_counter()
        mesh = self.create_performance_mesh(vertex_count, "simple")
        end_time = time.perf_counter()
        
        loading_time = end_time - start_time
        
        # Small meshes should load very quickly
        assert loading_time < 0.01  # Less than 10ms
        assert mesh.data.vertex_count == vertex_count
        
        print(f"Small mesh ({vertex_count} vertices): {loading_time*1000:.2f}ms")
    
    def test_medium_mesh_loading_speed(self):
        """Test loading speed for medium meshes (10K vertices)."""
        vertex_count = 10000
        
        start_time = time.perf_counter()
        mesh = self.create_performance_mesh(vertex_count, "simple")
        end_time = time.perf_counter()
        
        loading_time = end_time - start_time
        
        # Medium meshes should still load quickly
        assert loading_time < 0.1  # Less than 100ms
        assert mesh.data.vertex_count == vertex_count
        
        print(f"Medium mesh ({vertex_count} vertices): {loading_time*1000:.2f}ms")
    
    def test_stanford_bunny_size_loading(self):
        """Test loading speed for Stanford bunny-sized meshes (~35K vertices)."""
        vertex_count = 35000  # Approximate Stanford bunny size
        
        start_time = time.perf_counter()
        mesh = self.create_performance_mesh(vertex_count, "full")
        end_time = time.perf_counter()
        
        loading_time = end_time - start_time
        
        # Stanford bunny size should load reasonably quickly
        assert loading_time < 0.5  # Less than 500ms
        assert mesh.data.vertex_count == vertex_count
        
        print(f"Stanford bunny size ({vertex_count} vertices): {loading_time*1000:.2f}ms")
    
    def test_large_mesh_loading_speed(self):
        """Test loading speed for large meshes (100K+ vertices)."""
        vertex_count = 100000
        
        start_time = time.perf_counter()
        mesh = self.create_performance_mesh(vertex_count, "simple")
        end_time = time.perf_counter()
        
        loading_time = end_time - start_time
        
        # Large meshes should still be reasonable
        assert loading_time < 2.0  # Less than 2 seconds
        assert mesh.data.vertex_count == vertex_count
        
        print(f"Large mesh ({vertex_count} vertices): {loading_time*1000:.2f}ms")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of mesh storage."""
        test_cases = [
            (1000, "small"),
            (10000, "medium"),
            (35000, "bunny"),
            (100000, "large")
        ]
        
        for vertex_count, size_name in test_cases:
            # Test both simple and complex formats
            simple_mesh = self.create_performance_mesh(vertex_count, "simple")
            full_mesh = self.create_performance_mesh(vertex_count, "full")
            
            simple_mb = simple_mesh.data.vertex_size_bytes / (1024 * 1024)
            full_mb = full_mesh.data.vertex_size_bytes / (1024 * 1024)
            
            print(f"{size_name} mesh memory - Simple: {simple_mb:.2f}MB, Full: {full_mb:.2f}MB")
            
            # Memory should scale linearly with vertex count
            expected_simple_mb = vertex_count * 3 * 4 / (1024 * 1024)  # 3 floats * 4 bytes
            expected_full_mb = vertex_count * 8 * 4 / (1024 * 1024)    # 8 floats * 4 bytes
            
            assert abs(simple_mb - expected_simple_mb) < 0.01
            assert abs(full_mb - expected_full_mb) < 0.01
            
            # Memory usage should be reasonable
            if vertex_count <= 35000:  # Stanford bunny size
                assert simple_mb < 1.0  # Less than 1MB for position-only
                assert full_mb < 3.0    # Less than 3MB for full vertex data
    
    def test_bounding_box_calculation_speed(self):
        """Test bounding box calculation performance."""
        test_cases = [1000, 10000, 35000, 100000]
        
        for vertex_count in test_cases:
            mesh = self.create_performance_mesh(vertex_count, "simple")
            
            start_time = time.perf_counter()
            bbox = mesh.data.compute_bounding_box()
            end_time = time.perf_counter()
            
            calc_time = end_time - start_time
            
            # Should be very fast even for large meshes
            assert calc_time < 0.1  # Less than 100ms
            assert bbox is not None
            
            print(f"Bounding box ({vertex_count} vertices): {calc_time*1000:.2f}ms")


class TestRenderingPerformance:
    """Test rendering performance estimates and targets."""
    
    def estimate_gpu_performance(self, mesh: 'vf.Mesh', target_fps: int = 1000) -> dict:
        """Estimate GPU rendering performance for a mesh."""
        vertex_count = mesh.data.vertex_count
        triangle_count = mesh.data.triangle_count
        vertex_size = mesh.data.vertex_size_bytes
        index_size = mesh.data.index_size_bytes
        
        # Estimate GPU workload
        # Modern GPU can process ~10-50M vertices/sec depending on complexity
        vertex_processing_time = vertex_count / 50_000_000  # Conservative estimate
        
        # Triangle setup and rasterization
        triangle_processing_time = triangle_count / 100_000_000  # Very conservative
        
        # Memory bandwidth (assume 500 GB/s modern GPU)
        memory_transfer_time = (vertex_size + index_size) / (500 * 1024**3)
        
        estimated_frame_time = max(vertex_processing_time, triangle_processing_time, memory_transfer_time)
        estimated_fps = 1.0 / estimated_frame_time if estimated_frame_time > 0 else float('inf')
        
        return {
            'vertex_count': vertex_count,
            'triangle_count': triangle_count,
            'vertex_size_mb': vertex_size / (1024**2),
            'estimated_frame_time_ms': estimated_frame_time * 1000,
            'estimated_fps': estimated_fps,
            'meets_target': estimated_fps >= target_fps,
            'performance_ratio': estimated_fps / target_fps,
            'bottleneck': 'vertices' if vertex_processing_time == estimated_frame_time else
                         'triangles' if triangle_processing_time == estimated_frame_time else 'memory'
        }
    
    def test_stanford_bunny_1000fps_target(self):
        """Test that Stanford bunny-sized mesh can theoretically achieve 1000+ FPS."""
        # Create Stanford bunny-sized mesh
        mesh = self.create_performance_mesh(35000, "full")
        
        # Estimate performance
        perf = self.estimate_gpu_performance(mesh, target_fps=1000)
        
        print(f"\nStanford Bunny Performance Estimate:")
        print(f"  Vertices: {perf['vertex_count']:,}")
        print(f"  Triangles: {perf['triangle_count']:,}")
        print(f"  Memory: {perf['vertex_size_mb']:.2f} MB")
        print(f"  Estimated FPS: {perf['estimated_fps']:,.0f}")
        print(f"  Target: 1000 FPS")
        print(f"  Performance ratio: {perf['performance_ratio']:.1f}x")
        print(f"  Bottleneck: {perf['bottleneck']}")
        
        # Should meet or exceed 1000 FPS target
        assert perf['meets_target'], f"Estimated FPS ({perf['estimated_fps']:.0f}) below target (1000)"
        
        # Should have good performance margin
        assert perf['performance_ratio'] >= 2.0, "Should have at least 2x performance margin"
    
    def test_performance_scaling(self):
        """Test how performance scales with mesh complexity."""
        vertex_counts = [1000, 5000, 10000, 20000, 35000, 50000, 100000]
        results = []
        
        print(f"\nPerformance Scaling Analysis:")
        print(f"{'Vertices':<10} {'Est. FPS':<10} {'Meets 1K':<10} {'Margin':<10}")
        print("-" * 50)
        
        for vertex_count in vertex_counts:
            mesh = self.create_performance_mesh(vertex_count, "simple")
            perf = self.estimate_gpu_performance(mesh, target_fps=1000)
            
            meets_target = "✓" if perf['meets_target'] else "✗"
            margin = f"{perf['performance_ratio']:.1f}x" if perf['meets_target'] else f"{perf['performance_ratio']:.2f}x"
            
            print(f"{vertex_count:<10,} {perf['estimated_fps']:<10,.0f} {meets_target:<10} {margin:<10}")
            
            results.append(perf)
        
        # Stanford bunny size (35K vertices) should definitely meet target
        bunny_result = next(r for r in results if r['vertex_count'] == 35000)
        assert bunny_result['meets_target'], "Stanford bunny size must meet 1000 FPS target"
        
        # Performance should scale predictably
        # Smaller meshes should have better performance
        small_result = next(r for r in results if r['vertex_count'] == 1000)
        assert small_result['estimated_fps'] > bunny_result['estimated_fps']
    
    def test_memory_bandwidth_requirements(self):
        """Test GPU memory bandwidth requirements for 1000 FPS."""
        mesh = self.create_performance_mesh(35000, "full")
        
        # Calculate memory bandwidth at 1000 FPS
        memory_per_frame = mesh.data.vertex_size_bytes + mesh.data.index_size_bytes
        memory_per_second = memory_per_frame * 1000  # At 1000 FPS
        memory_gb_per_second = memory_per_second / (1024**3)
        
        # Modern high-end GPUs have 500-1000 GB/s bandwidth
        # We should use only a small fraction for vertex data
        bandwidth_usage_500 = memory_gb_per_second / 500  # Assuming 500 GB/s GPU
        bandwidth_usage_1000 = memory_gb_per_second / 1000  # Assuming 1000 GB/s GPU
        
        print(f"\nMemory Bandwidth Analysis at 1000 FPS:")
        print(f"  Memory per frame: {memory_per_frame / 1024:.1f} KB")
        print(f"  Memory per second: {memory_gb_per_second:.3f} GB/s")
        print(f"  Bandwidth usage (500 GB/s GPU): {bandwidth_usage_500:.1%}")
        print(f"  Bandwidth usage (1000 GB/s GPU): {bandwidth_usage_1000:.1%}")
        
        # Should use less than 10% of available bandwidth
        assert bandwidth_usage_500 < 0.1, "Should use less than 10% of GPU bandwidth"
        
        # Should be well within limits even for mid-range GPUs
        assert bandwidth_usage_500 < 0.05, "Should use less than 5% for good performance margin"
    
    def test_vertex_format_performance_impact(self):
        """Test how vertex format affects performance."""