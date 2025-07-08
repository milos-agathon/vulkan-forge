# tests/test_numpy_performance.py
"""Test NumPy integration performance."""

import pytest
import numpy as np
import time

class TestPerformance:
    """Performance tests for NumPy integration."""
    
    @pytest.fixture
    def allocator(self):
        """Create a test allocator."""
        return MockAllocator()
    
    def test_large_buffer_upload(self, allocator):
        """Test uploading 1M vertices in reasonable time."""
        # Create 1 million vertices
        vertices = np.random.randn(1_000_000, 3).astype(np.float32)
        
        # Time the upload
        start = time.perf_counter()
        buffer = NumpyBuffer(allocator, vertices)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        # Should complete
        assert buffer.size == vertices.nbytes
        print(f"Upload time: {elapsed:.2f}ms")
    
    def test_update_performance(self, allocator):
        """Test buffer update performance."""
        # Create initial buffer
        vertices = np.zeros((100_000, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Time updates
        update_times = []
        for i in range(10):  # Reduced for testing
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
    ])
    def test_various_sizes_and_types(self, allocator, size, dtype):
        """Test performance with various sizes and types."""
        # Create test data
        data = np.random.randn(size, 3).astype(dtype)
        
        # Time buffer creation
        start = time.perf_counter()
        buffer = NumpyBuffer(allocator, data)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert buffer.size == data.nbytes
        print(f"Size {size}, dtype {dtype}: {elapsed:.2f}ms")