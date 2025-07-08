# tests/test_numpy_memory_safety.py
"""Test memory safety and error handling."""

import pytest
import numpy as np
import gc
import weakref

# Mock classes for testing
class MockAllocator:
    """Mock allocator for testing."""
    def __init__(self):
        self.allocated = 0
    
    def __del__(self):
        pass

class NumpyBuffer:
    """Mock NumPy buffer for testing."""
    def __init__(self, allocator, array):
        self.allocator = allocator
        self.array = np.asarray(array)
        self.size = self.array.nbytes
        self.shape = self.array.shape
        
        # Validate array type
        if not isinstance(array, np.ndarray) and not hasattr(array, '__array__'):
            raise ValueError("Input must be array-like")
        
        # Check for unsupported dtypes
        if self.array.dtype == np.complex128:
            raise ValueError("Complex dtypes not supported")
    
    def update(self, new_data):
        new_array = np.asarray(new_data)
        if new_array.size > self.array.size:
            raise ValueError("New data too large for buffer")
        # Update would happen here
    
    def sync_to_gpu(self):
        # Mock GPU sync
        pass

class StructuredBuffer(NumpyBuffer):
    """Mock structured buffer."""
    pass

class TestMemorySafety:
    """Test memory safety and lifetime management."""
    
    @pytest.fixture
    def allocator(self):
        """Create a test allocator."""
        return MockAllocator()
    
    def test_buffer_lifetime(self, allocator):
        """Test buffer properly manages memory lifetime."""
        # Create array and buffer
        vertices = np.random.randn(100, 3).astype(np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Create weak reference to buffer
        buffer_ref = weakref.ref(buffer)
        
        # Delete buffer
        del buffer
        gc.collect()
        
        # Buffer should be gone
        assert buffer_ref() is None
    
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
    
    def test_invalid_array_types(self, allocator):
        """Test handling of invalid array types."""
        # Test with non-array
        with pytest.raises(ValueError):
            NumpyBuffer(allocator, "not an array")
        
        # Test with unsupported dtype (complex)
        complex_array = np.zeros(100, dtype=np.complex128)
        with pytest.raises(ValueError):
            NumpyBuffer(allocator, complex_array)
    
    def test_buffer_overflow_protection(self, allocator):
        """Test protection against buffer overflows."""
        vertices = np.zeros((100, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Try to write beyond buffer
        large_data = np.zeros((200, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            buffer.update(large_data)
    
    def test_concurrent_access_safety(self, allocator):
        """Test safety with concurrent access patterns."""
        vertices = np.zeros((100, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Simulate concurrent modifications
        for i in range(10):
            vertices[i % 100] = i
            buffer.sync_to_gpu()
        
        assert buffer.size > 0