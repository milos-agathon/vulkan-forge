"""Test performance functionality."""

import pytest
import numpy as np
import time

class TestPerformance:
    """Test performance measurement and benchmarking."""
    
    def test_array_operations_performance(self):
        """Test NumPy array operations performance."""
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)
            
            start = time.perf_counter()
            c = a + b
            add_time = time.perf_counter() - start
            
            start = time.perf_counter()
            d = np.dot(a, b)
            dot_time = time.perf_counter() - start
            
            assert c.size == size
            assert isinstance(d, np.float32)
            
            print(f"Size {size}: add={add_time*1000:.2f}ms, dot={dot_time*1000:.2f}ms")
    
    def test_memory_bandwidth(self):
        """Test memory bandwidth with large arrays."""
        size = 1_000_000
        
        data = np.random.rand(size).astype(np.float32)
        
        start = time.perf_counter()
        total = np.sum(data)
        seq_time = time.perf_counter() - start
        
        indices = np.random.randint(0, size, size//10)
        start = time.perf_counter()
        subset = data[indices]
        random_time = time.perf_counter() - start
        
        assert total > 0
        assert len(subset) == len(indices)
        
        print(f"Sequential sum: {seq_time*1000:.2f}ms")
        print(f"Random access: {random_time*1000:.2f}ms")
    
    @pytest.mark.performance
    def test_large_array_creation(self):
        """Test performance of large array creation."""
        sizes = [100_000, 1_000_000, 10_000_000]
        
        for size in sizes:
            start = time.perf_counter()
            arr = np.zeros(size, dtype=np.float32)
            creation_time = time.perf_counter() - start
            
            start = time.perf_counter()
            arr.fill(1.0)
            fill_time = time.perf_counter() - start
            
            assert arr.size == size
            assert np.all(arr == 1.0)
            
            print(f"Size {size}: create={creation_time*1000:.2f}ms, fill={fill_time*1000:.2f}ms")
            
            del arr