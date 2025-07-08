"""Test memory safety and error handling."""

import pytest
import numpy as np
import gc
import weakref
from .mock_classes import NumpyBuffer, StructuredBuffer

class TestMemorySafety:
    """Test memory safety and lifetime management."""
    
    def test_buffer_lifetime(self, allocator):
        """Test buffer properly manages memory lifetime."""
        vertices = np.random.randn(100, 3).astype(np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        buffer_ref = weakref.ref(buffer)
        del buffer
        gc.collect()
        
        assert buffer_ref() is None
    
    def test_invalid_array_types(self, allocator):
        """Test handling of invalid array types."""
        with pytest.raises(ValueError):
            NumpyBuffer(allocator, "not an array")
        
        complex_array = np.zeros(100, dtype=np.complex128)
        with pytest.raises(ValueError):
            NumpyBuffer(allocator, complex_array)
    
    def test_buffer_overflow_protection(self, allocator):
        """Test protection against buffer overflows."""
        vertices = np.zeros((100, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        large_data = np.zeros((200, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            buffer.update(large_data)
    
    def test_array_lifetime_management(self, allocator):
        """Test buffer keeps array alive when needed."""
        def create_buffer():
            vertices = np.random.randn(100, 3).astype(np.float32)
            buffer = NumpyBuffer(allocator, vertices)
            # Array goes out of scope but buffer should keep reference
            return buffer
        
        buffer = create_buffer()
        gc.collect()
        
        # Buffer should still be valid
        assert buffer.size > 0
        assert buffer.shape == (100, 3)
    
    def test_concurrent_access_safety(self, allocator):
        """Test safety with concurrent access patterns."""
        vertices = np.zeros((100, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Simulate concurrent modifications
        for i in range(10):
            vertices[i % 100] = i
            buffer.sync_to_gpu()
        
        assert buffer.size > 0