# tests/test_memory_management.py
"""Test memory management functionality."""

import pytest
import numpy as np
import tempfile
import os

class TestMemoryManagement:
    """Test memory allocation and deallocation."""
    
    def test_basic_allocation(self):
        """Test basic memory allocation works."""
        # Simple allocation test
        data = np.zeros(1000, dtype=np.float32)
        assert data.size == 1000
        assert data.dtype == np.float32
    
    def test_large_allocation(self):
        """Test large memory allocation."""
        # Test with larger data
        size = 1_000_000
        data = np.random.randn(size).astype(np.float32)
        assert data.size == size
        
        # Cleanup
        del data
    
    def test_file_cleanup(self):
        """Test temporary file cleanup."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            f.write(b"test data")
        
        # Verify file exists
        assert os.path.exists(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        assert not os.path.exists(temp_path)