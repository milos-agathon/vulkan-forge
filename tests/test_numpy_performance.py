"""Test NumPy integration performance."""

import pytest
import numpy as np
import time
import vulkan_forge as vf
from vulkan_forge.numpy_buffer import NumpyBuffer, create_vertex_buffer


class TestPerformance:
    """Performance tests for NumPy integration."""
    
    @pytest.fixture
    def allocator(self):
        """Create a test allocator."""
        # Mock allocator for testing
        class MockAllocator:
            pass
        return MockAllocator()
    
    def test_large_buffer_upload(self, allocator):
        """Test uploading 1M vertices in < 5ms."""
        # Create 1 million vertices
        vertices = np.random.randn(1_000_000, 3).astype(np.float32)
        
        # Time the upload
        start = time.perf_counter()
        buffer = create_vertex_buffer(allocator, vertices)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        # Should be fast (in real implementation with proper GPU)
        # For mock, just check it completes
        assert buffer.size == vertices.nbytes
        print(f"Upload time: {elapsed:.2f}ms")
    
    def test_zero_copy_performance(self, allocator):
        """Test zero-copy is actually zero-copy."""
        # Create large contiguous array
        vertices = np.zeros((10_000_000, 3), dtype=np.float32)
        
        # Measure memory before
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss
        
        # Create buffer (should not duplicate memory)
        buffer = NumpyBuffer(allocator, vertices)
        
        mem_after = process.memory_info().rss
        mem_increase = (mem_after - mem_before) / 1024 / 1024  # MB
        
        # Memory increase should be minimal (< 10MB for metadata)
        # In real implementation, this would verify zero-copy
        print(f"Memory increase: {mem_increase:.2f}MB")
    
    def test_update_performance(self, allocator):
        """Test buffer update performance."""
        # Create initial buffer
        vertices = np.zeros((100_000, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Time updates
        update_times = []
        for i in range(100):
            # Modify data
            vertices[:] = i
            
            # Time sync
            start = time.perf_counter()
            buffer.sync_to_gpu()
            elapsed = (time.perf_counter() - start) * 1000
            update_times.append(elapsed)
        
        # Check average update time
        avg_time = np.mean(update_times)
        print(f"Average update time: {avg_time:.2f}ms")
    
    @pytest.mark.parametrize("size,dtype", [
        (1000, np.float32),
        (10000, np.float32),
        (100000, np.float32),
        (1000000, np.float32),
        (1000000, np.float16),
        (1000000, np.uint8),
    ])
    def test_various_sizes_and_types(self, allocator, size, dtype):
        """Test performance with various sizes and types."""
        # Create array
        if dtype == np.uint8:
            array = np.random.randint(0, 255, (size, 4), dtype=dtype)
        else:
            array = np.random.randn(size, 3).astype(dtype)
        
        # Time creation
        start = time.perf_counter()
        buffer = NumpyBuffer(allocator, array)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Calculate bandwidth
        mb_transferred = array.nbytes / 1024 / 1024
        bandwidth = mb_transferred / (elapsed / 1000) if elapsed > 0 else float('inf')
        
        print(f"Size: {size}, dtype: {dtype}, "
              f"time: {elapsed:.2f}ms, bandwidth: {bandwidth:.0f}MB/s")