"""
Vulkan-Forge Core Utilities

Core functionality for Vulkan initialization, device management,
and high-level rendering operations.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import sys
import platform

def _ensure_float32(a: np.ndarray) -> np.ndarray:
    """Ensure array is contiguous float32 for GPU upload."""
    return np.ascontiguousarray(a, dtype=np.float32)

def _ensure_uint32(a: np.ndarray) -> np.ndarray:
    """Ensure array is contiguous uint32 for index buffers."""
    return np.ascontiguousarray(a, dtype=np.uint32)

def get_vulkan_version() -> str:
    """Get Vulkan API version string."""
    try:
        # Try to import the native module to get version
        from . import _vulkan_forge_native as native
        if hasattr(native, 'get_vulkan_api_version'):
            return native.get_vulkan_api_version()
        else:
            return "1.2.0"  # Default target version
    except ImportError:
        return "Unknown"

def check_vulkan_support() -> Dict[str, Any]:
    """
    Check if Vulkan is available on the current system.
    
    Returns:
        Dictionary with support information
    """
    try:
        from . import _vulkan_forge_native as native
        
        # Try to create a minimal Vulkan instance
        if hasattr(native, 'check_vulkan_available'):
            return native.check_vulkan_available()
        
        # Fallback: assume available if module loads
        return {
            'vulkan_available': True,
            'api_version': get_vulkan_version(),
            'instance_extensions': [],
            'device_extensions': [],
            'validation_layers': False
        }
        
    except ImportError as e:
        return {
            'vulkan_available': False,
            'error': str(e),
            'api_version': 'None',
            'instance_extensions': [],
            'device_extensions': [],
            'validation_layers': False
        }

def list_vulkan_devices() -> List[Dict[str, Any]]:
    """
    List available Vulkan devices.
    
    Returns:
        List of device information dictionaries
    """
    try:
        from . import _vulkan_forge_native as native
        
        if hasattr(native, 'enumerate_vulkan_devices'):
            return native.enumerate_vulkan_devices()
        
        # Fallback: return mock device info
        return [{
            'name': 'Default Device',
            'type': 'Unknown',
            'memory_mb': 1000,
            'api_version': get_vulkan_version(),
            'driver_version': 'Unknown',
            'vendor_id': 0,
            'device_id': 0
        }]
        
    except ImportError:
        return []

def create_renderer_auto(width: int = 1920, height: int = 1080) -> 'Renderer':
    """
    Create a renderer with automatic device selection.
    
    Args:
        width: Render target width
        height: Render target height
        
    Returns:
        Configured Renderer instance
        
    Raises:
        RuntimeError: If no suitable Vulkan device found
    """
    try:
        from . import Renderer
        
        # Check Vulkan support first
        support = check_vulkan_support()
        if not support['vulkan_available']:
            raise RuntimeError(f"Vulkan not available: {support.get('error', 'Unknown error')}")
        
        # Create renderer
        renderer = Renderer(width, height)
        
        return renderer
        
    except ImportError as e:
        raise RuntimeError(f"Failed to import Renderer: {e}")

def get_system_info() -> Dict[str, Any]:
    """Get system information relevant to Vulkan rendering."""
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'vulkan_support': check_vulkan_support(),
        'numpy_version': np.__version__,
    }
    
    # Add GPU information if available
    try:
        devices = list_vulkan_devices()
        info['vulkan_devices'] = devices
        info['primary_device'] = devices[0] if devices else None
    except Exception:
        info['vulkan_devices'] = []
        info['primary_device'] = None
    
    return info

def benchmark_system() -> Dict[str, Any]:
    """
    Run a quick system benchmark for rendering performance.
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {
        'cpu_info': platform.processor(),
        'memory_test': {},
        'vulkan_test': {},
        'numpy_test': {}
    }
    
    # Memory allocation test
    try:
        start_time = time.perf_counter()
        test_array = np.random.random((1000, 1000)).astype(np.float32)
        alloc_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        _ = test_array @ test_array.T  # Matrix multiplication
        compute_time = time.perf_counter() - start_time
        
        results['memory_test'] = {
            'allocation_time_ms': alloc_time * 1000,
            'compute_time_ms': compute_time * 1000,
            'throughput_gflops': (2 * 1000**3) / (compute_time * 1e9)
        }
    except Exception as e:
        results['memory_test'] = {'error': str(e)}
    
    # Vulkan test
    try:
        support = check_vulkan_support()
        devices = list_vulkan_devices()
        
        results['vulkan_test'] = {
            'available': support['vulkan_available'],
            'device_count': len(devices),
            'api_version': support['api_version']
        }
        
        if devices:
            primary = devices[0]
            results['vulkan_test'].update({
                'primary_device': primary['name'],
                'device_memory_mb': primary.get('memory_mb', 0),
                'device_type': primary.get('type', 'Unknown')
            })
    except Exception as e:
        results['vulkan_test'] = {'error': str(e)}
    
    # NumPy optimization test
    try:
        start_time = time.perf_counter()
        test_data = np.random.random((10000, 3)).astype(np.float32)
        converted = _ensure_float32(test_data)
        conversion_time = time.perf_counter() - start_time
        
        results['numpy_test'] = {
            'conversion_time_ms': conversion_time * 1000,
            'contiguous': converted.flags['C_CONTIGUOUS'],
            'dtype_correct': converted.dtype == np.float32
        }
    except Exception as e:
        results['numpy_test'] = {'error': str(e)}
    
    return results

def validate_mesh_data(vertices: np.ndarray, indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Validate mesh data for GPU upload.
    
    Args:
        vertices: Vertex data array
        indices: Optional index array
        
    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'info': {}
    }
    
    # Validate vertices
    if vertices is None:
        results['errors'].append("Vertices array is None")
        results['valid'] = False
        return results
    
    if not isinstance(vertices, np.ndarray):
        results['errors'].append("Vertices must be numpy array")
        results['valid'] = False
        return results
    
    if vertices.ndim != 2:
        results['errors'].append(f"Vertices must be 2D array, got {vertices.ndim}D")
        results['valid'] = False
    
    if vertices.shape[0] == 0:
        results['errors'].append("No vertices provided")
        results['valid'] = False
    
    # Check vertex format
    vertex_count, components = vertices.shape
    results['info']['vertex_count'] = vertex_count
    results['info']['components'] = components
    
    if components not in [3, 5, 6, 8]:
        results['warnings'].append(f"Unusual vertex format: {components} components")
    
    # Check data type
    if vertices.dtype != np.float32:
        results['warnings'].append(f"Vertices should be float32, got {vertices.dtype}")
    
    if not vertices.flags['C_CONTIGUOUS']:
        results['warnings'].append("Vertices array is not contiguous")
    
    # Validate indices if provided
    if indices is not None:
        if not isinstance(indices, np.ndarray):
            results['errors'].append("Indices must be numpy array")
            results['valid'] = False
        elif indices.ndim != 1:
            results['errors'].append(f"Indices must be 1D array, got {indices.ndim}D")
            results['valid'] = False
        else:
            index_count = indices.shape[0]
            results['info']['index_count'] = index_count
            results['info']['triangle_count'] = index_count // 3
            
            if index_count % 3 != 0:
                results['warnings'].append("Index count not divisible by 3")
            
            if indices.dtype not in [np.uint16, np.uint32]:
                results['warnings'].append(f"Indices should be uint16 or uint32, got {indices.dtype}")
            
            # Check index range
            if np.any(indices >= vertex_count):
                results['errors'].append("Some indices exceed vertex count")
                results['valid'] = False
            
            if np.any(indices < 0):
                results['errors'].append("Negative indices found")
                results['valid'] = False
    
    # Memory usage estimation
    vertex_memory = vertices.nbytes
    index_memory = indices.nbytes if indices is not None else 0
    results['info']['memory_bytes'] = vertex_memory + index_memory
    results['info']['memory_mb'] = (vertex_memory + index_memory) / (1024 * 1024)
    
    return results

def optimize_mesh_data(vertices: np.ndarray, indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Optimize mesh data for GPU rendering.
    
    Args:
        vertices: Vertex data array
        indices: Optional index array
        
    Returns:
        Tuple of (optimized_vertices, optimized_indices)
    """
    # Ensure proper data types and layout
    opt_vertices = _ensure_float32(vertices)
    
    opt_indices = None
    if indices is not None:
        # Choose optimal index type based on vertex count
        if vertices.shape[0] < 65536:
            opt_indices = np.ascontiguousarray(indices, dtype=np.uint16)
        else:
            opt_indices = _ensure_uint32(indices)
    
    return opt_vertices, opt_indices

def create_test_matrices(width: int, height: int) -> Dict[str, np.ndarray]:
    """
    Create test transformation matrices for rendering.
    
    Args:
        width: Viewport width
        height: Viewport height
        
    Returns:
        Dictionary with model, view, and projection matrices
    """
    import math
    
    # Identity model matrix
    model = np.eye(4, dtype=np.float32)
    
    # Simple view matrix (camera at (0,0,5) looking at origin)
    view = np.eye(4, dtype=np.float32)
    view[2, 3] = -5.0  # Move camera back
    
    # Perspective projection matrix
    aspect = width / height
    fov = math.radians(45.0)
    near = 0.1
    far = 100.0
    
    f = 1.0 / math.tan(fov / 2.0)
    
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2.0 * far * near) / (near - far)
    projection[3, 2] = -1.0
    
    return {
        'model': model,
        'view': view,
        'projection': projection,
        'mvp': projection @ view @ model
    }

# Module-level cache for expensive operations
_system_info_cache = None
_vulkan_support_cache = None

def get_cached_system_info() -> Dict[str, Any]:
    """Get cached system information (computed once per session)."""
    global _system_info_cache
    if _system_info_cache is None:
        _system_info_cache = get_system_info()
    return _system_info_cache.copy()

def get_cached_vulkan_support() -> Dict[str, Any]:
    """Get cached Vulkan support info (computed once per session)."""
    global _vulkan_support_cache
    if _vulkan_support_cache is None:
        _vulkan_support_cache = check_vulkan_support()
    return _vulkan_support_cache.copy()

# Utility functions for common operations
def create_grid_vertices(width: int, height: int, spacing: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a grid of vertices for testing.
    
    Args:
        width: Grid width in vertices
        height: Grid height in vertices  
        spacing: Distance between vertices
        
    Returns:
        Tuple of (vertices, indices)
    """
    vertices = []
    indices = []
    
    # Generate vertices
    for y in range(height):
        for x in range(width):
            # Position
            pos_x = (x - width/2) * spacing
            pos_y = (y - height/2) * spacing
            pos_z = 0.0
            
            # Normal (pointing up)
            norm_x, norm_y, norm_z = 0.0, 0.0, 1.0
            
            # UV coordinates
            u = x / (width - 1) if width > 1 else 0.0
            v = y / (height - 1) if height > 1 else 0.0
            
            vertices.append([pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, u, v])
    
    # Generate indices (triangles)
    for y in range(height - 1):
        for x in range(width - 1):
            # Current quad vertices
            v0 = y * width + x
            v1 = y * width + (x + 1)
            v2 = (y + 1) * width + (x + 1)
            v3 = (y + 1) * width + x
            
            # Two triangles per quad
            indices.extend([v0, v1, v2, v0, v2, v3])
    
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

__all__ = [
    '_ensure_float32',
    '_ensure_uint32', 
    'get_vulkan_version',
    'check_vulkan_support',
    'list_vulkan_devices',
    'create_renderer_auto',
    'get_system_info',
    'benchmark_system',
    'validate_mesh_data',
    'optimize_mesh_data',
    'create_test_matrices',
    'get_cached_system_info',
    'get_cached_vulkan_support',
    'create_grid_vertices'
]