"""Zero-copy NumPy buffer integration for Vulkan Forge."""

import numpy as np
from typing import Optional, Union, Tuple, List
import weakref
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
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


@dataclass
class StructuredBuffer:
    """Simple structured buffer backed by a NumPy array and optional GPU memory."""

    data: np.ndarray
    gpu_buffer: int
    alloc: int
    _ptr: Optional[ctypes.c_void_p] = None

    def __init__(self, size_or_allocator, dtype: np.dtype = np.float32,
                 count: int = None, usage: int = BUFFER_USAGE_STORAGE):
        if isinstance(size_or_allocator, int) and count is None:
            size_bytes = size_or_allocator
            self.data = np.zeros(size_bytes // np.dtype(dtype).itemsize, dtype=dtype)
        else:
            if count is None:
                raise TypeError("count must be provided when allocator is given")
            allocator = size_or_allocator
            self.data = np.zeros(count, dtype=dtype)
            size_bytes = self.data.nbytes
        self.gpu_buffer = 0
        self.alloc = 0
        self._ptr = None

        if hasattr(native, 'create_structured_buffer'):
            buf, alloc, ptr = native.create_structured_buffer(size_bytes, usage)
            self.gpu_buffer = int(buf)
            self.alloc = int(alloc)
            if ptr:
                self._ptr = ctypes.c_void_p(ptr)
        else:
            # CPU fallback uses an in-memory buffer to mimic mapped GPU memory
            self._cpu_backing = (ctypes.c_ubyte * size_bytes)()
            self._ptr = ctypes.cast(self._cpu_backing, ctypes.c_void_p)

    def upload(self) -> None:
        """Copy data from host array to GPU mapping if available."""
        if self._ptr:
            ctypes.memmove(self._ptr.value, self.data.ctypes.data, self.data.nbytes)

    def download(self) -> np.ndarray:
        """Copy data from GPU mapping back to the host array."""
        if self._ptr:
            ctypes.memmove(self.data.ctypes.data, self._ptr.value, self.data.nbytes)
        return self.data

    def __getitem__(self, field: str) -> np.ndarray:
        if field not in self.data.dtype.fields:
            raise KeyError(field)
        return self.data[field]

    def __setitem__(self, field: str, value) -> None:
        if field not in self.data.dtype.fields:
            raise KeyError(field)
        self.data[field] = value
        self.upload()


class NumpyBuffer:
    """GPU buffer backed by a NumPy array.

    The constructor accepts either an existing ``np.ndarray`` which will be
    used directly (zero-copy) or a ``dtype`` and element ``count`` to allocate
    a new array.
    """

    def __init__(self, allocator, dtype_or_data, count: int = None,
                 usage: int = BUFFER_USAGE_VERTEX):
        self._allocator = allocator
        self._usage = usage

        if isinstance(dtype_or_data, np.ndarray):
            self._array = dtype_or_data
            self._dtype = self._array.dtype
            self._count = (
                self._array.size if self._array.ndim == 1 else self._array.shape[0]
            )
        else:
            if count is None:
                raise TypeError("count must be provided when dtype is given")
            self._dtype = np.dtype(dtype_or_data)
            self._count = count
            self._array = np.zeros(count, dtype=self._dtype)

        if self._dtype.kind == "c":
            raise TypeError("complex dtypes are not supported")

        self.gpu_buffer = 0
        self.allocation = 0
        if hasattr(native, "allocate_buffer"):
            try:
                buf, alloc = native.allocate_buffer(
                    int(getattr(self._allocator, "value", 0)),
                    self._array.nbytes,
                    usage,
                )
                self.gpu_buffer = int(buf)
                self.allocation = int(alloc)
            except Exception:
                self.gpu_buffer = 0
                self.allocation = 0

        self._fields = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Size of the underlying array in bytes."""
        return self._array.nbytes

    @property
    def shape(self) -> tuple:
        """Shape of the underlying array."""
        return self._array.shape

    # ------------------------------------------------------------------
    # Data transfer helpers
    # ------------------------------------------------------------------
    def upload(self) -> None:
        """Copy host data to GPU (if GPU buffer exists)."""
        if self.gpu_buffer and hasattr(native, "upload_buffer"):
            native.upload_buffer(int(self.gpu_buffer), self._array)

    def download(self) -> np.ndarray:
        """Copy GPU data back to host (if GPU buffer exists)."""
        if self.gpu_buffer and hasattr(native, "download_buffer"):
            native.download_buffer(int(self.gpu_buffer), self._array)
        return self._array

    # Backwards compatibility with older API used in tests
    def sync_to_gpu(self) -> None:
        self.upload()

    def sync_from_gpu(self) -> None:
        self.download()

    def update(self, data: np.ndarray) -> None:
        """Update buffer contents with ``data`` and upload to GPU."""
        if data.nbytes > self._array.nbytes:
            raise ValueError("Data too large for buffer")
        np.copyto(self._array, data, casting="no")
        self.upload()


class _NumpyBufferCtx(NumpyBuffer):
    """Context-managed NumpyBuffer that releases resources on exit."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


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
    return NumpyBuffer(allocator, vertices, usage=BUFFER_USAGE_VERTEX)


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
    return NumpyBuffer(allocator, indices, usage=BUFFER_USAGE_INDEX)


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
    return NumpyBuffer(allocator, data, usage=BUFFER_USAGE_UNIFORM)


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
    return NumpyBuffer(allocator, data, usage=usage)


def numpy_buffer(allocator, array, usage: int = BUFFER_USAGE_VERTEX) -> "NumpyBuffer":
    """Convenience wrapper returning a context-managed buffer."""
    return _NumpyBufferCtx(allocator, array, usage=usage)


__all__ = [
    "StructuredBuffer",
    "NumpyBuffer",
    "numpy_buffer",
    "create_vertex_buffer",
    "create_index_buffer",
    "create_uniform_buffer",
    "create_storage_buffer",
]
