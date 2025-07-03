"""Vertex input description and layout utilities."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import IntEnum

# Import format constants from numpy_buffer
from .numpy_buffer import NUMPY_TO_VK_FORMAT


class VertexInputRate(IntEnum):
    """Vertex input rate for attribute data."""
    VERTEX = 0
    INSTANCE = 1


@dataclass
class VertexAttribute:
    """Description of a single vertex attribute.
    
    Parameters
    ----------
    location : int
        Shader location index
    binding : int
        Vertex buffer binding index
    format : str
        Vulkan format string
    offset : int
        Byte offset within vertex
    """
    location: int
    binding: int
    format: str
    offset: int


@dataclass
class VertexBinding:
    """Description of a vertex buffer binding.
    
    Parameters
    ----------
    binding : int
        Binding index
    stride : int
        Byte stride between vertices
    input_rate : VertexInputRate
        Per-vertex or per-instance
    """
    binding: int
    stride: int
    input_rate: VertexInputRate = VertexInputRate.VERTEX


class VertexInputDescription:
    """Complete vertex input state description.
    
    This class helps create vertex input descriptions from
    NumPy arrays or structured dtypes.
    
    Examples
    --------
    >>> # From separate arrays
    >>> desc = VertexInputDescription()
    >>> desc.add_attribute(0, positions, binding=0)
    >>> desc.add_attribute(1, colors, binding=1)
    >>> 
    >>> # From structured array
    >>> vertex_dtype = np.dtype([
    ...     ('position', np.float32, 3),
    ...     ('color', np.float32, 4)
    ... ])
    >>> desc = VertexInputDescription.from_dtype(vertex_dtype)
    """
    
    def __init__(self):
        self.attributes: List[VertexAttribute] = []
        self.bindings: List[VertexBinding] = []
        self._binding_strides: Dict[int, int] = {}
    
    def add_attribute(self, location: int, array: np.ndarray,
                     binding: int = 0, offset: int = 0) -> 'VertexInputDescription':
        """Add a vertex attribute from a NumPy array.
        
        Parameters
        ----------
        location : int
            Shader location
        array : np.ndarray
            Array with attribute data
        binding : int, optional
            Buffer binding index (default: 0)
        offset : int, optional
            Byte offset in vertex (default: 0)
        
        Returns
        -------
        VertexInputDescription
            Self for chaining
        """
        # Determine format from array
        components = 1
        if array.ndim > 1:
            components = array.shape[-1]
        
        format_key = (array.dtype, components)
        format_str = NUMPY_TO_VK_FORMAT.get(format_key)
        
        if not format_str:
            raise ValueError(f"Unsupported array format: dtype={array.dtype}, "
                           f"components={components}")
        
        # Add attribute
        attr = VertexAttribute(
            location=location,
            binding=binding,
            format=format_str,
            offset=offset
        )
        self.attributes.append(attr)
        
        # Update binding stride
        stride = array.strides[0] if array.ndim > 1 else array.itemsize * components
        if binding in self._binding_strides:
            self._binding_strides[binding] = max(self._binding_strides[binding], 
                                               offset + array.itemsize * components)
        else:
            self._binding_strides[binding] = stride
        
        return self
    
    def add_binding(self, binding: int, stride: int,
                   input_rate: VertexInputRate = VertexInputRate.VERTEX) -> 'VertexInputDescription':
        """Add a vertex buffer binding.
        
        Parameters
        ----------
        binding : int
            Binding index
        stride : int
            Byte stride between elements
        input_rate : VertexInputRate, optional
            Per-vertex or per-instance (default: VERTEX)
        
        Returns
        -------
        VertexInputDescription
            Self for chaining
        """
        bind = VertexBinding(
            binding=binding,
            stride=stride,
            input_rate=input_rate
        )
        self.bindings.append(bind)
        return self
    
    def finalize(self) -> 'VertexInputDescription':
        """Finalize the description by creating bindings from attributes.
        
        Returns
        -------
        VertexInputDescription
            Self for chaining
        """
        # Create bindings for any that weren't explicitly added
        existing_bindings = {b.binding for b in self.bindings}
        
        for binding, stride in self._binding_strides.items():
            if binding not in existing_bindings:
                self.add_binding(binding, stride)
        
        return self
    
    @classmethod
    def from_dtype(cls, dtype: np.dtype, binding: int = 0,
                  input_rate: VertexInputRate = VertexInputRate.VERTEX) -> 'VertexInputDescription':
        """Create description from a structured NumPy dtype.
        
        Parameters
        ----------
        dtype : np.dtype
            Structured dtype with fields
        binding : int, optional
            Buffer binding index (default: 0)
        input_rate : VertexInputRate, optional
            Per-vertex or per-instance (default: VERTEX)
        
        Returns
        -------
        VertexInputDescription
            The created description
        """
        desc = cls()
        location = 0
        
        for field_name, (field_dtype, field_shape) in dtype.fields.items():
            # Determine components
            if field_shape:
                components = field_shape[0] if isinstance(field_shape, tuple) else field_shape
            else:
                components = 1
            
            # Get format
            format_key = (field_dtype, components)
            format_str = NUMPY_TO_VK_FORMAT.get(format_key)
            
            if not format_str:
                raise ValueError(f"Unsupported field format: {field_name} with "
                               f"dtype={field_dtype}, components={components}")
            
            # Add attribute
            offset = dtype.fields[field_name][1]
            attr = VertexAttribute(
                location=location,
                binding=binding,
                format=format_str,
                offset=offset
            )
            desc.attributes.append(attr)
            location += 1
        
        # Add binding
        desc.add_binding(binding, dtype.itemsize, input_rate)
        
        return desc
    
    @classmethod
    def from_arrays(cls, arrays: List[Tuple[int, np.ndarray]],
                   binding: int = 0) -> 'VertexInputDescription':
        """Create description from multiple arrays packed together.
        
        Parameters
        ----------
        arrays : List[Tuple[int, np.ndarray]]
            List of (location, array) pairs
        binding : int, optional
            Buffer binding index (default: 0)
        
        Returns
        -------
        VertexInputDescription
            The created description
        """
        desc = cls()
        offset = 0
        
        for location, array in arrays:
            desc.add_attribute(location, array, binding, offset)
            
            # Calculate size
            components = 1 if array.ndim == 1 else array.shape[-1]
            offset += array.itemsize * components
        
        return desc.finalize()
    
    def get_native_description(self) -> Dict:
        """Get description in format suitable for native code.
        
        Returns
        -------
        Dict
            Dictionary with 'attributes' and 'bindings' lists
        """
        return {
            'attributes': [
                {
                    'location': attr.location,
                    'binding': attr.binding,
                    'format': attr.format,
                    'offset': attr.offset
                }
                for attr in self.attributes
            ],
            'bindings': [
                {
                    'binding': bind.binding,
                    'stride': bind.stride,
                    'inputRate': bind.input_rate.value
                }
                for bind in self.bindings
            ]
        }


# Convenience functions
def describe_vertices(positions: np.ndarray,
                     normals: Optional[np.ndarray] = None,
                     texcoords: Optional[np.ndarray] = None,
                     colors: Optional[np.ndarray] = None,
                     tangents: Optional[np.ndarray] = None) -> VertexInputDescription:
    """Create vertex description for common vertex layouts.
    
    Parameters
    ----------
    positions : np.ndarray
        Vertex positions (required)
    normals : np.ndarray, optional
        Vertex normals
    texcoords : np.ndarray, optional
        Texture coordinates
    colors : np.ndarray, optional
        Vertex colors
    tangents : np.ndarray, optional
        Vertex tangents
    
    Returns
    -------
    VertexInputDescription
        Complete vertex input description
    """
    desc = VertexInputDescription()
    location = 0
    offset = 0
    
    # Add position (always location 0)
    desc.add_attribute(location, positions, 0, offset)
    components = 1 if positions.ndim == 1 else positions.shape[-1]
    offset += positions.itemsize * components
    location += 1
    
    # Add optional attributes
    if normals is not None:
        desc.add_attribute(location, normals, 0, offset)
        components = 1 if normals.ndim == 1 else normals.shape[-1]
        offset += normals.itemsize * components
        location += 1
    
    if texcoords is not None:
        desc.add_attribute(location, texcoords, 0, offset)
        components = 1 if texcoords.ndim == 1 else texcoords.shape[-1]
        offset += texcoords.itemsize * components
        location += 1
    
    if colors is not None:
        desc.add_attribute(location, colors, 0, offset)
        components = 1 if colors.ndim == 1 else colors.shape[-1]
        offset += colors.itemsize * components
        location += 1
    
    if tangents is not None:
        desc.add_attribute(location, tangents, 0, offset)
        components = 1 if tangents.ndim == 1 else tangents.shape[-1]
        offset += tangents.itemsize * components
        location += 1
    
    return desc.finalize()


def describe_instanced_vertices(vertices: VertexInputDescription,
                               instance_transforms: np.ndarray,
                               instance_colors: Optional[np.ndarray] = None) -> VertexInputDescription:
    """Add instance data to vertex description.
    
    Parameters
    ----------
    vertices : VertexInputDescription
        Base vertex description
    instance_transforms : np.ndarray
        Per-instance transformation matrices (4x4)
    instance_colors : np.ndarray, optional
        Per-instance colors
    
    Returns
    -------
    VertexInputDescription
        Updated description with instance attributes
    """
    # Start location after vertex attributes
    location = len(vertices.attributes)
    binding = max(attr.binding for attr in vertices.attributes) + 1
    offset = 0
    
    # Add transform as 4 vec4 attributes
    for i in range(4):
        vertices.add_attribute(
            location + i,
            instance_transforms[:, i, :],
            binding,
            offset + i * 16  # 4 floats * 4 bytes
        )
    
    location += 4
    offset += 64  # Full 4x4 matrix
    
    # Add instance colors if provided
    if instance_colors is not None:
        vertices.add_attribute(location, instance_colors, binding, offset)
    
    # Set instance input rate
    vertices.add_binding(binding, offset, VertexInputRate.INSTANCE)
    
    return vertices