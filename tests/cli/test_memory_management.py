"""Test memory management functionality."""

import pytest
import numpy as np
import tempfile
import os

class TestMemoryManagement:
    """Test memory allocation and deallocation."""
    
    def test_basic_allocation(self):
        """Test basic memory allocation works."""
        data = np.zeros(1000, dtype=np.float32)
        assert data.size == 1000
        assert data.dtype == np.float32
    
    def test_large_allocation(self):
        """Test large memory allocation."""
        size = 1_000_000
        data = np.random.randn(size).astype(np.float32)
        assert data.size == size
        del data
    
    def test_file_cleanup(self):
        """Test temporary file cleanup."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            f.write(b"test data")
        
        assert os.path.exists(temp_path)
        os.unlink(temp_path)
        assert not os.path.exists(temp_path)
    
    def test_array_memory_usage(self):
        """Test NumPy array memory usage patterns."""
        sizes = [100, 1000, 10000]
        
        for size in sizes:
            arr = np.random.rand(size, 3).astype(np.float32)
            expected_bytes = size * 3 * 4  # float32 = 4 bytes
            
            assert arr.nbytes == expected_bytes
            print(f"Array size {size}: {arr.nbytes} bytes")
    
    def test_memory_cleanup_with_references(self):
        """Test memory cleanup when arrays have references."""
        original = np.random.rand(10000).astype(np.float32)
        view = original[::2]  # Create a view
        copy = original.copy()  # Create a copy
        
        assert original.size == 10000
        assert view.size == 5000
        assert copy.size == 10000
        
        del original
        
        assert view.size == 5000
        assert copy.size == 10000
        
        del view, copy