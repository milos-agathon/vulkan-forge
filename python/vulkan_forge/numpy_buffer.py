"""Zero-copy NumPy buffer integration for Vulkan Forge."""

import numpy as np
from typing import Optional, Union, Tuple, List
import weakref
from contextlib import contextmanager
try:
    from . import _vulkan_forge_native as native
except ImportError:
    from . import vulkan_forge_native as native

# Buffer usage flags
BUFFER_USAGE_VERTEX = native.BUFFER_USAGE_VERTEX_BUFFER
BUFFER_USAGE_INDEX = native.BUFFER_USAGE_INDEX_BUFFER
BUFFER_USAGE_UNIFORM = native.BUFFER_USAGE_UNIFORM_BUFFER
BUFFER_USAGE_STORAGE = native.BUFFER_USAGE_STORAGE_BUFFER
BUFFER_USAGE_TRANSFER_SRC = native.BUFFER_USAGE_TRANSFER_SRC
BUFFER_USAGE_TRANSFER_DST = native.BUFFER_USAGE_TRANSFER_DST

# Format mappings
NUMPY_TO_VK_FORMAT = {
    (np.dtype('float32'), 1): 'VK_FORMAT_R32_SFLOAT',
    (np.dtype('float32'), 2): 'VK_FORMAT_R32G32_SFLOAT',
    (np.dtype('float32'), 3): 'VK_FORMAT_R32G32B32_SFLOAT',
    (np.dtype('float32'), 4): 'VK_FORMAT_R32G32B32A32_SFLOAT',
    (np.dtype('float64'), 1): 'VK_FORMAT_R64_SFLOAT',
    (np.dtype('int32'), 1): 'VK_FORMAT_R32_SINT',
    (np.dtype('uint32'), 1): 'VK_FORMAT_R32_UINT',
    (np.dtype('int16'), 1): 'VK_FORMAT_R16_SINT',
    (np.dtype('uint16'), 1): 'VK_FORMAT_R16_UINT',
    (np.dtype('int8'), 1): 'VK_FORMAT_R8_SINT',
    (np.dtype('uint8'), 1): 'VK_FORMAT_R8_UINT',
    (np.dtype('uint8'), 4): 'VK_FORMAT_R8G8B8A8_UNORM',
}


class NumpyBuffer:
    """GPU buffer backed by NumPy array with zero-copy support when possible.
    
    This class provides a bridge between NumPy arrays and Vulkan buffers,
    enabling efficient data transfer and manipulation.
    
    Parameters
    ----------
    allocator : VmaAllocator
        The Vulkan memory allocator
    array : np.ndarray
        The NumPy array to upload
    usage : int, optional
        Vulkan buffer usage flags (default: BUFFER_USAGE_VERTEX)

    Yields
    ------
    NumpyBuffer
        The GPU buffer

    Examples
    --------
    >>> vertex_dtype = np.dtype([
    ...     ('position', np.float32, 3),
    ...     ('normal', np.float32, 3),
    ...     ('texcoord', np.float32, 2)
    ... ])
    >>> buffer = StructuredBuffer(allocator, vertex_dtype, 1000)
    >>> buffer['position'] = positions
    >>> buffer['normal'] = normals
    """
    
    def __init__(self, allocator, dtype: np.dtype, count: int,
                 usage: int = BUFFER_USAGE_VERTEX):
        self._allocator = allocator
        self._dtype = dtype
        self._count = count
        self._usage = usage
        
        # Create backing array
        self._array = np.zeros(count, dtype=dtype)
        
        # Create GPU buffer
        self._buffer = NumpyBuffer(allocator, self._array, usage)
        
        # Parse fields for attribute info
        self._fields = {}
        self._parse_fields()
    
    def _parse_fields(self):
        """Parse dtype fields to extract attribute information."""
        offset = 0
        for name, (dtype, shape) in self._dtype.fields.items():
            if shape:
                # Array field
                components = shape[0] if isinstance(shape, tuple) else shape
            else:
                components = 1
            
            self._fields[name] = {
                'dtype': dtype,
                'offset': offset,
                'components': components,
                'format': NUMPY_TO_VK_FORMAT.get((dtype, components))
            }
            
            offset += dtype.itemsize * components
    
    def __getitem__(self, field: str) -> np.ndarray:
        """Get field data."""
        if field not in self._fields:
            raise KeyError(f"Field '{field}' not found in structured buffer")
        self._buffer.sync_from_gpu()
        return self._array[field]
    
    def __setitem__(self, field: str, value):
        """Set field data."""
        if field not in self._fields:
            raise KeyError(f"Field '{field}' not found in structured buffer")
        self._array[field] = value
        self._buffer.sync_to_gpu()
    
    @property
    def buffer(self) -> 'NumpyBuffer':
        """Get the underlying buffer."""
        return self._buffer
    
    @property
    def fields(self) -> dict:
        """Get field information."""
        return self._fields.copy()
    
    def get_vertex_attributes(self, binding: int = 0) -> List[dict]:
        """Get vertex attribute descriptions for this buffer.
        
        Parameters
        ----------
        binding : int, optional
            Vertex buffer binding index (default: 0)
        
        Returns
        -------
        List[dict]
            List of vertex attribute descriptions
        """
        attributes = []
        location = 0
        
        for name, info in self._fields.items():
            attributes.append({
                'location': location,
                'binding': binding,
                'format': info['format'],
                'offset': info['offset']
            })
            location += 1
        
        return attributes


class MultiBuffer:
    """Manager for multiple related GPU buffers.
    
    Handles allocation and management of multiple buffers
    that are used together (e.g., vertex and index buffers).
    
    Parameters
    ----------
    allocator : VmaAllocator
        The Vulkan memory allocator
    
    Examples
    --------
    >>> buffers = MultiBuffer(allocator)
    >>> buffers.add_vertex_buffer('positions', positions)
    >>> buffers.add_vertex_buffer('colors', colors)
    >>> buffers.add_index_buffer(indices)
    >>> 
    >>> renderer.bind_buffers(buffers)
    >>> renderer.draw_indexed(len(indices))
    """
    
    def __init__(self, allocator):
        self._allocator = allocator
        self._vertex_buffers = {}
        self._index_buffer = None
        self._uniform_buffers = {}
        self._storage_buffers = {}
    
    def add_vertex_buffer(self, name: str, array: np.ndarray,
                         binding: Optional[int] = None) -> NumpyBuffer:
        """Add a vertex buffer.
        
        Parameters
        ----------
        name : str
            Buffer name for identification
        array : np.ndarray
            Vertex data
        binding : int, optional
            Explicit binding index (auto-assigned if None)
        
        Returns
        -------
        NumpyBuffer
            The created buffer
        """
        if binding is None:
            binding = len(self._vertex_buffers)
        
        buffer = NumpyBuffer(self._allocator, array, BUFFER_USAGE_VERTEX)
        self._vertex_buffers[name] = (binding, buffer)
        return buffer
    
    def add_index_buffer(self, indices: np.ndarray) -> NumpyBuffer:
        """Add an index buffer.
        
        Parameters
        ----------
        indices : np.ndarray
            Index data (uint16 or uint32)
        
        Returns
        -------
        NumpyBuffer
            The created buffer
        """
        # Ensure correct dtype
        if indices.dtype not in [np.uint16, np.uint32]:
            indices = indices.astype(np.uint32)
        
        self._index_buffer = NumpyBuffer(
            self._allocator, indices, BUFFER_USAGE_INDEX
        )
        return self._index_buffer
    
    def add_uniform_buffer(self, name: str, data: Union[np.ndarray, dict],
                          binding: int) -> NumpyBuffer:
        """Add a uniform buffer.
        
        Parameters
        ----------
        name : str
            Buffer name
        data : np.ndarray or dict
            Uniform data
        binding : int
            Descriptor set binding index
        
        Returns
        -------
        NumpyBuffer
            The created buffer
        """
        if isinstance(data, dict):
            # Convert dict to structured array
            # TODO: Implement structured uniform conversion
            raise NotImplementedError("Dict uniforms not yet supported")
        
        buffer = NumpyBuffer(self._allocator, data, BUFFER_USAGE_UNIFORM)
        self._uniform_buffers[name] = (binding, buffer)
        return buffer
    
    def add_storage_buffer(self, name: str, array: np.ndarray,
                          binding: int) -> NumpyBuffer:
        """Add a storage buffer.
        
        Parameters
        ----------
        name : str
            Buffer name
        array : np.ndarray
            Storage data
        binding : int
            Descriptor set binding index
        
        Returns
        -------
        NumpyBuffer
            The created buffer
        """
        buffer = NumpyBuffer(self._allocator, array, BUFFER_USAGE_STORAGE)
        self._storage_buffers[name] = (binding, buffer)
        return buffer
    
    def get_vertex_buffer(self, name: str) -> Tuple[int, NumpyBuffer]:
        """Get a vertex buffer by name."""
        return self._vertex_buffers.get(name)
    
    def get_index_buffer(self) -> Optional[NumpyBuffer]:
        """Get the index buffer."""
        return self._index_buffer
    
    def sync_all(self):
        """Sync all buffers to GPU."""
        for _, buffer in self._vertex_buffers.values():
            buffer.sync_to_gpu()
        if self._index_buffer:
            self._index_buffer.sync_to_gpu()
        for _, buffer in self._uniform_buffers.values():
            buffer.sync_to_gpu()
        for _, buffer in self._storage_buffers.values():
            buffer.sync_to_gpu()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.sync_all()
        return False


# Utility functions
def create_vertex_buffer(allocator, vertices: np.ndarray) -> NumpyBuffer:
    """Create a vertex buffer from a NumPy array.
    
    Parameters
    ----------
    allocator : VmaAllocator
        The Vulkan memory allocator
    vertices : np.ndarray
        Vertex data
    
    Returns
    -------
    NumpyBuffer
        The created vertex buffer
    """
    return NumpyBuffer(allocator, vertices, BUFFER_USAGE_VERTEX)


def create_index_buffer(allocator, indices: np.ndarray) -> NumpyBuffer:
    """Create an index buffer from a NumPy array.
    
    Parameters
    ----------
    allocator : VmaAllocator
        The Vulkan memory allocator
    indices : np.ndarray
        Index data (will be converted to uint32 if needed)
    
    Returns
    -------
    NumpyBuffer
        The created index buffer
    """
    if indices.dtype not in [np.uint16, np.uint32]:
        indices = indices.astype(np.uint32)
    return NumpyBuffer(allocator, indices, BUFFER_USAGE_INDEX)


def create_uniform_buffer(allocator, data: np.ndarray) -> NumpyBuffer:
    """Create a uniform buffer from a NumPy array.
    
    Parameters
    ----------
    allocator : VmaAllocator
        The Vulkan memory allocator
    data : np.ndarray
        Uniform data
    
    Returns
    -------
    NumpyBuffer
        The created uniform buffer
    """
    return NumpyBuffer(allocator, data, BUFFER_USAGE_UNIFORM)


def create_storage_buffer(allocator, data: np.ndarray,
                         read_only: bool = False) -> NumpyBuffer:
    """Create a storage buffer from a NumPy array.
    
    Parameters
    ----------
    allocator : VmaAllocator
        The Vulkan memory allocator
    data : np.ndarray
        Storage data
    read_only : bool, optional
        Whether the buffer is read-only (default: False)
    
    Returns
    -------
    NumpyBuffer
        The created storage buffer
    """
    usage = BUFFER_USAGE_STORAGE
    if not read_only:
        usage |= BUFFER_USAGE_TRANSFER_DST
    return NumpyBuffer(allocator, data, usage)