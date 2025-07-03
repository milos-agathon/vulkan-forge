"""Test NumPy dtype to Vulkan format conversions."""

import pytest
import numpy as np
from vulkan_forge.numpy_buffer import NUMPY_TO_VK_FORMAT
from vulkan_forge.vertex_input import (
    VertexInputDescription, describe_vertices,
    describe_instanced_vertices, VertexInputRate
)


class TestFormatConversions:
    """Test dtype to Vulkan format mapping."""
    
    def test_scalar_formats(self):
        """Test scalar format conversions."""
        test_cases = [
            (np.float32, 1, 'VK_FORMAT_R32_SFLOAT'),
            (np.float64, 1, 'VK_FORMAT_R64_SFLOAT'),
            (np.int32, 1, 'VK_FORMAT_R32_SINT'),
            (np.uint32, 1, 'VK_FORMAT_R32_UINT'),
            (np.int16, 1, 'VK_FORMAT_R16_SINT'),
            (np.uint16, 1, 'VK_FORMAT_R16_UINT'),
            (np.int8, 1, 'VK_FORMAT_R8_SINT'),
            (np.uint8, 1, 'VK_FORMAT_R8_UINT'),
        ]
        
        for dtype, components, expected in test_cases:
            key = (np.dtype(dtype), components)
            assert key in NUMPY_TO_VK_FORMAT
            assert NUMPY_TO_VK_FORMAT[key] == expected
    
    def test_vector_formats(self):
        """Test vector format conversions."""
        test_cases = [
            (np.float32, 2, 'VK_FORMAT_R32G32_SFLOAT'),
            (np.float32, 3, 'VK_FORMAT_R32G32B32_SFLOAT'),
            (np.float32, 4, 'VK_FORMAT_R32G32B32A32_SFLOAT'),
            (np.uint8, 4, 'VK_FORMAT_R8G8B8A8_UNORM'),
        ]
        
        for dtype, components, expected in test_cases:
            key = (np.dtype(dtype), components)
            assert key in NUMPY_TO_VK_FORMAT
            assert NUMPY_TO_VK_FORMAT[key] == expected
    
    def test_vertex_input_description(self):
        """Test vertex input description creation."""
        # Create description from arrays
        positions = np.zeros((100, 3), dtype=np.float32)
        colors = np.zeros((100, 4), dtype=np.float32)
        
        desc = VertexInputDescription()
        desc.add_attribute(0, positions)
        desc.add_attribute(1, colors)
        desc.finalize()
        
        # Check attributes
        assert len(desc.attributes) == 2
        assert desc.attributes[0].location == 0
        assert desc.attributes[0].format == 'VK_FORMAT_R32G32B32_SFLOAT'
        assert desc.attributes[1].location == 1
        assert desc.attributes[1].format == 'VK_FORMAT_R32G32B32A32_SFLOAT'
        
        # Check bindings
        assert len(desc.bindings) == 1
        assert desc.bindings[0].stride == 28  # 3*4 + 4*4
    
    def test_structured_dtype_description(self):
        """Test description from structured dtype."""
        vertex_dtype = np.dtype([
            ('position', np.float32, 3),
            ('normal', np.float32, 3),
            ('texcoord', np.float32, 2),
            ('color', np.uint8, 4)
        ])
        
        desc = VertexInputDescription.from_dtype(vertex_dtype)
        
        # Check all attributes created
        assert len(desc.attributes) == 4
        
        # Check offsets
        assert desc.attributes[0].offset == 0  # position
        assert desc.attributes[1].offset == 12  # normal (3 * 4)
        assert desc.attributes[2].offset == 24  # texcoord (6 * 4)
        assert desc.attributes[3].offset == 32  # color (8 * 4)
    
    def test_convenience_describe_vertices(self):
        """Test convenience function for common layouts."""
        positions = np.zeros((100, 3), dtype=np.float32)
        normals = np.zeros((100, 3), dtype=np.float32)
        texcoords = np.zeros((100, 2), dtype=np.float32)
        
        desc = describe_vertices(
            positions=positions,
            normals=normals,
            texcoords=texcoords
        )
        
        # Check created properly
        assert len(desc.attributes) == 3
        assert desc.attributes[0].location == 0  # positions
        assert desc.attributes[1].location == 1  # normals
        assert desc.attributes[2].location == 2  # texcoords
    
    def test_instanced_description(self):
        """Test instanced vertex description."""
        # Base vertices
        positions = np.zeros((100, 3), dtype=np.float32)
        base_desc = describe_vertices(positions=positions)
        
        # Instance data
        transforms = np.zeros((10, 4, 4), dtype=np.float32)
        colors = np.zeros((10, 4), dtype=np.float32)
        
        # Add instancing
        desc = describe_instanced_vertices(
            base_desc,
            transforms,
            colors
        )
        
        # Check instance attributes added
        assert len(desc.attributes) == 6  # 1 vertex + 4 matrix + 1 color
        
        # Check instance binding
        instance_bindings = [b for b in desc.bindings 
                           if b.input_rate == VertexInputRate.INSTANCE]
        assert len(instance_bindings) == 1