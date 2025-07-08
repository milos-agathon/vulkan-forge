"""Test NumPy integration performance."""

import pytest
import numpy as np
import time
from .mock_classes import NumpyBuffer

class TestPerformance:
    """Performance tests for NumPy integration."""
    
    def test_large_buffer_upload(self, allocator):
        """Test uploading large buffers."""
        vertices = np.random.randn(100_000, 3).astype(np.float32)
        
        start = time.perf_counter()
        buffer = NumpyBuffer(allocator, vertices)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert buffer.size == vertices.nbytes
        print(f"Upload time: {elapsed:.2f}ms")
    
    def test_update_performance(self, allocator):
        """Test buffer update performance."""
        vertices = np.zeros((100_000, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        update_times = []
        for i in range(10):  # Reduced for testing
            vertices[:] = i
            
            start = time.perf_counter()
            buffer.sync_to_gpu()
            elapsed = (time.perf_counter() - start) * 1000
            update_times.append(elapsed)
        
        avg_time = np.mean(update_times)
        print(f"Average update time: {avg_time:.2f}ms")
    
    @pytest.mark.parametrize("size,dtype", [
        (1000, np.float32),
        (10000, np.float32),
        (100000, np.float32),
    ])
    def test_various_sizes_and_types(self, allocator, size, dtype):
        """Test performance with various sizes and types."""
        data = np.random.randn(size, 3).astype(dtype)
        
        start = time.perf_counter()
        buffer = NumpyBuffer(allocator, data)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert buffer.size == data.nbytes
        print(f"Size {size}, dtype {dtype}: {elapsed:.2f}ms")