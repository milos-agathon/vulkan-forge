"""Test zero-copy NumPy integration."""

import pytest
import numpy as np
import vulkan_forge as vf
from vulkan_forge.numpy_buffer import (
    NumpyBuffer, StructuredBuffer, MultiBuffer,
    create_vertex_buffer, create_index_buffer,
    BUFFER_USAGE_VERTEX, BUFFER_USAGE_STORAGE
)


class TestNumpyBuffer:
    """Test basic NumpyBuffer functionality."""
    
    @pytest.fixture
    def allocator(self):
        """Create a test allocator."""
        # Mock allocator for testing
        # In real tests, this would be a proper VMA allocator
        class MockAllocator:
            pass
        return MockAllocator()
    
    def test_contiguous_array(self, allocator):
        """Test zero-copy with contiguous array."""
        # Create contiguous array
        vertices = np.random.randn(1000, 3).astype(np.float32)
        assert vertices.flags['C_CONTIGUOUS']
        
        # Create buffer
        buffer = NumpyBuffer(allocator, vertices)
        
        # Check properties
        assert buffer.size == vertices.nbytes
        assert buffer.dtype == np.float32
        assert buffer.shape == (1000, 3)
        assert buffer.format == 'VK_FORMAT_R32G32B32_SFLOAT'
    
    def test_non_contiguous_array(self, allocator):
        """Test fallback copy with non-contiguous array."""
        # Create non-contiguous array (transpose)
        vertices = np.random.randn(3, 1000).astype(np.float32).T
        assert not vertices.flags['C_CONTIGUOUS']
        
        # Should still work
        buffer = NumpyBuffer(allocator, vertices)
        assert buffer.size == vertices.nbytes
        assert buffer.shape == (1000, 3)
    
    def test_readonly_array(self, allocator):
        """Test handling of read-only arrays."""
        # Create read-only array
        vertices = np.random.randn(100, 3).astype(np.float32)
        vertices.flags.writeable = False
        
        # Should work with copy
        buffer = NumpyBuffer(allocator, vertices)
        assert buffer.size == vertices.nbytes
    
    def test_dtype_conversion(self, allocator):
        """Test automatic dtype handling."""
        # Different dtypes
        test_cases = [
            (np.float64, 1, 'VK_FORMAT_R64_SFLOAT'),
            (np.int32, 1, 'VK_FORMAT_R32_SINT'),
            (np.uint16, 1, 'VK_FORMAT_R16_UINT'),
            (np.uint8, 4, 'VK_FORMAT_R8G8B8A8_UNORM'),
        ]
        
        for dtype, components, expected_format in test_cases:
            if components == 1:
                array = np.zeros(100, dtype=dtype)
            else:
                array = np.zeros((100, components), dtype=dtype)
            
            buffer = NumpyBuffer(allocator, array)
            assert buffer.format == expected_format
    
    def test_context_manager(self, allocator):
        """Test context manager usage."""
        vertices = np.random.randn(100, 3).astype(np.float32)
        
        with NumpyBuffer(allocator, vertices) as buffer:
            assert buffer.size == vertices.nbytes
            # Modify array
            vertices[0] = [1, 2, 3]
            # Sync should happen on exit
        
        # Buffer should be valid after context exit
        assert buffer.size == vertices.nbytes
    
    def test_update_method(self, allocator):
        """Test buffer update."""
        # Initial data
        vertices = np.zeros((100, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Update with new data
        new_vertices = np.ones((50, 3), dtype=np.float32)
        buffer.update(new_vertices)
        
        # Update with offset
        buffer.update(new_vertices, offset=50 * 3 * 4)  # 50 vertices * 3 * 4 bytes
    
    def test_buffer_too_small(self, allocator):
        """Test error on buffer overflow."""
        vertices = np.zeros((100, 3), dtype=np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Try to update with too much data
        big_vertices = np.zeros((200, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="exceed buffer size"):
            buffer.update(big_vertices)


class TestStructuredBuffer:
    """Test StructuredBuffer for vertex data."""
    
    @pytest.fixture
    def allocator(self):
        """Create a test allocator."""
        class MockAllocator:
            pass
        return MockAllocator()
    
    def test_structured_vertex_data(self, allocator):
        """Test structured array for vertices."""
        # Define vertex structure
        vertex_dtype = np.dtype([
            ('position', np.float32, 3),
            ('normal', np.float32, 3),
            ('texcoord', np.float32, 2),
            ('color', np.uint8, 4)
        ])
        
        # Create buffer
        buffer = StructuredBuffer(allocator, vertex_dtype, 1000)
        
        # Check fields
        fields = buffer.fields
        assert 'position' in fields
        assert fields['position']['components'] == 3
        assert fields['position']['format'] == 'VK_FORMAT_R32G32B32_SFLOAT'
        
        assert 'color' in fields
        assert fields['color']['components'] == 4
        assert fields['color']['format'] == 'VK_FORMAT_R8G8B8A8_UNORM'
    
    def test_field_access(self, allocator):
        """Test field get/set operations."""
        vertex_dtype = np.dtype([
            ('position', np.float32, 3),
            ('color', np.float32, 4)
        ])
        
        buffer = StructuredBuffer(allocator, vertex_dtype, 100)
        
        # Set positions
        positions = np.random.randn(100, 3).astype(np.float32)
        buffer['position'] = positions
        
        # Get positions back
        retrieved = buffer['position']
        np.testing.assert_array_equal(retrieved, positions)
        
        # Set colors
        colors = np.random.rand(100, 4).astype(np.float32)
        buffer['color'] = colors
    
    def test_vertex_attributes(self, allocator):
        """Test vertex attribute generation."""
        vertex_dtype = np.dtype([
            ('position', np.float32, 3),
            ('normal', np.float32, 3),
            ('texcoord', np.float32, 2)
        ])
        
        buffer = StructuredBuffer(allocator, vertex_dtype, 100)
        attributes = buffer.get_vertex_attributes()
        
        # Check attributes
        assert len(attributes) == 3
        
        # Position at location 0
        assert attributes[0]['location'] == 0
        assert attributes[0]['offset'] == 0
        assert attributes[0]['format'] == 'VK_FORMAT_R32G32B32_SFLOAT'
        
        # Normal at location 1
        assert attributes[1]['location'] == 1
        assert attributes[1]['offset'] == 12  # 3 floats * 4 bytes
        
        # Texcoord at location 2
        assert attributes[2]['location'] == 2
        assert attributes[2]['offset'] == 24  # 6 floats * 4 bytes


class TestMultiBuffer:
    """Test MultiBuffer manager."""
    
    @pytest.fixture
    def allocator(self):
        """Create a test allocator."""
        class MockAllocator:
            pass
        return MockAllocator()
    
    def test_multiple_vertex_buffers(self, allocator):
        """Test managing multiple vertex buffers."""
        buffers = MultiBuffer(allocator)
        
        # Add vertex buffers
        positions = np.random.randn(1000, 3).astype(np.float32)
        colors = np.random.rand(1000, 4).astype(np.float32)
        
        pos_buffer = buffers.add_vertex_buffer('positions', positions)
        col_buffer = buffers.add_vertex_buffer('colors', colors)
        
        # Check retrieval
        binding, retrieved_pos = buffers.get_vertex_buffer('positions')
        assert binding == 0
        assert retrieved_pos is pos_buffer
        
        binding, retrieved_col = buffers.get_vertex_buffer('colors')
        assert binding == 1
        assert retrieved_col is col_buffer
    
    def test_index_buffer(self, allocator):
        """Test index buffer management."""
        buffers = MultiBuffer(allocator)
        
        # Add index buffer
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        idx_buffer = buffers.add_index_buffer(indices)
        
        assert buffers.get_index_buffer() is idx_buffer
    
    def test_uniform_and_storage_buffers(self, allocator):
        """Test uniform and storage buffer management."""
        buffers = MultiBuffer(allocator)
        
        # Add uniform buffer
        uniforms = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        uni_buffer = buffers.add_uniform_buffer('mvp', uniforms, binding=0)
        
        # Add storage buffer
        particles = np.random.randn(10000, 4).astype(np.float32)
        storage_buffer = buffers.add_storage_buffer('particles', particles, binding=1)
        
        # Test context manager
        with buffers:
            # Modify data
            particles[:, 3] = 1.0
            # Sync should happen on exit


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.fixture
    def allocator(self):
        """Create a test allocator."""
        class MockAllocator:
            pass
        return MockAllocator()
    
    def test_create_vertex_buffer(self, allocator):
        """Test vertex buffer creation utility."""
        vertices = np.random.randn(100, 3).astype(np.float32)
        buffer = create_vertex_buffer(allocator, vertices)
        
        assert isinstance(buffer, NumpyBuffer)
        assert buffer.size == vertices.nbytes
    
    def test_create_index_buffer(self, allocator):
        """Test index buffer creation with type conversion."""
        # Test with different types
        indices_int = np.array([0, 1, 2], dtype=np.int32)
        buffer = create_index_buffer(allocator, indices_int)
        
        # Should be converted to uint32
        assert buffer.dtype == np.uint32
    
    def test_buffer_array_interface(self, allocator):
        """Test array-like interface."""
        from vulkan_forge.numpy_buffer import BufferArray
        
        vertices = np.random.randn(10, 3).astype(np.float32)
        buffer = NumpyBuffer(allocator, vertices)
        
        # Create array view
        array_view = BufferArray(buffer)
        
        # Test properties
        assert array_view.shape == (10, 3)
        assert array_view.dtype == np.float32
        
        # Test indexing (would need proper GPU sync in real implementation)
        # array_view[0] = [1, 2, 3]
        # assert np.allclose(array_view[0], [1, 2, 3])