# vulkan_forge/__init__.py
"""VulkanForge: Multi-GPU renderer with automatic CPU fallback."""

print('DEBUG: Loading vulkan_forge/__init__.py from:', __file__)


import logging
import sys
import os
import importlib

# Set up module logger
logger = logging.getLogger(__name__)

# Version info
__version__ = "0.1.0"
__author__ = "VulkanForge Team"

# Ensure the directory containing this file is in the module search path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Try to import native extension if available
_native_available = False
_native_module = None
try:
    try:
        # Try relative import first
        from . import _vulkan_forge_native
        _native_module = _vulkan_forge_native
        _native_available = True
    except ImportError:
        # Fall back to direct import
        from . import vulkan_forge_native
        _native_module = vulkan_forge_native
        _native_available = True
    logger.info("Native extension loaded successfully")
except ImportError as e:
    logger.debug(f"Native extension not available: {e}")
    # This is fine - we'll use pure Python implementation
    _native_module = None

if _native_module is not None:
    sys.modules[f"{__name__}._vulkan_forge_native"] = _native_module
# Import Python modules with better error handling
_import_errors = []

try:
    # Try relative imports first
    from .matrices import Matrix4x4
except ImportError as e:
    _import_errors.append(f"matrices: {e}")
    try:
        # Fallback to direct import
        from matrices import Matrix4x4
    except ImportError as e2:
        _import_errors.append(f"matrices (direct): {e2}")
        Matrix4x4 = None

try:
    from .backend import (
        DeviceManager,
        VulkanForgeError,
        LogicalDevice,
        PhysicalDeviceInfo,
        create_allocator,
        create_allocator_native,
        allocate_buffer,
        destroy_allocator,
        BUFFER_USAGE_VERTEX,
        BUFFER_USAGE_STORAGE,
    )
except ImportError as e:
    _import_errors.append(f"backend: {e}")
    try:
        from backend import (
            DeviceManager,
            VulkanForgeError,
            LogicalDevice,
        PhysicalDeviceInfo,
        create_allocator,
        create_allocator_native,
        allocate_buffer,
        destroy_allocator,
        BUFFER_USAGE_VERTEX,
        BUFFER_USAGE_STORAGE,
        )
    except ImportError as e2:
        _import_errors.append(f"backend (direct): {e2}")
        DeviceManager = VulkanForgeError = LogicalDevice = PhysicalDeviceInfo = None

try:
    from .renderer import (
        create_renderer, RenderTarget, Mesh, Material, Light,
        Transform, Renderer, VulkanRenderer, CPURenderer,
        set_vertex_buffer, save_image,
    )
except ImportError as e:
    _import_errors.append(f"renderer: {e}")
    try:
        from renderer import (
            create_renderer, RenderTarget, Mesh, Material, Light,
            Transform, Renderer, VulkanRenderer, CPURenderer,
            set_vertex_buffer, save_image,
        )
    except ImportError as e2:
        _import_errors.append(f"renderer (direct): {e2}")
        create_renderer = RenderTarget = Mesh = Material = Light = None
        Transform = Renderer = VulkanRenderer = CPURenderer = None

try:
    from .numpy_buffer import (
        numpy_buffer,
        create_uniform_buffer,
        create_vertex_buffer,
        create_index_buffer,
        create_storage_buffer,
    )
except ImportError as e:
    _import_errors.append(f"numpy_buffer: {e}")
    try:
        from numpy_buffer import numpy_buffer, create_uniform_buffer, create_vertex_buffer, create_index_buffer, create_storage_buffer  # type: ignore
    except ImportError as e2:
        _import_errors.append(f"numpy_buffer (direct): {e2}")
        numpy_buffer = None
        create_uniform_buffer = None
        create_vertex_buffer = None
        create_index_buffer = None
        create_storage_buffer = None

# Check if any critical imports failed
if _import_errors and (Matrix4x4 is None or create_renderer is None):
    error_msg = "Failed to import VulkanForge Python modules:\n" + "\n".join(_import_errors)
    raise ImportError(error_msg)

# Export main API - only include successfully imported items
__all__ = ['__version__', '_native_available']

# Add successfully imported items to __all__
if Matrix4x4 is not None:
    __all__.extend(['Matrix4x4'])
    
if create_renderer is not None:
    __all__.extend([
        'create_renderer', 'RenderTarget', 'Mesh', 'Material', 'Light',
        'Transform', 'Renderer', 'VulkanRenderer', 'CPURenderer',
        'set_vertex_buffer', 'save_image'
    ])
    
if DeviceManager is not None:
    __all__.extend([
        'DeviceManager', 'VulkanForgeError', 'LogicalDevice', 'PhysicalDeviceInfo',
        'create_allocator', 'allocate_buffer', 'destroy_allocator',
        'BUFFER_USAGE_VERTEX', 'BUFFER_USAGE_STORAGE'
    ])

if numpy_buffer is not None:
    __all__.append('numpy_buffer')

if create_uniform_buffer is not None:
    __all__.extend(['create_uniform_buffer', 'create_vertex_buffer', 'create_index_buffer', 'create_storage_buffer'])

if 'BUFFER_USAGE_VERTEX' not in __all__:
    __all__.append('BUFFER_USAGE_VERTEX')

# Module initialization message
logger.debug(f"VulkanForge {__version__} initialized (native extension: {_native_available})")
logger.debug(f"Successfully imported: {__all__}")
if _import_errors:
    logger.debug(f"Import warnings: {_import_errors}")