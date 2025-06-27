# vulkan_forge/__init__.py
"""Vulkan-accelerated height-field renderer for Matplotlib"""

__version__ = "0.1.0"

import sys
from pathlib import Path
import importlib.util

# Import the native module
_native = None
HeightFieldScene = None
Renderer = None

try:
    # Find the .pyd file
    import site
    pyd_file = None
    
    for site_dir in site.getsitepackages():
        pyd_path = Path(site_dir) / "lib" / "vulkan_forge"
        if pyd_path.exists():
            pyd_files = list(pyd_path.glob("*.pyd"))
            if pyd_files:
                pyd_file = pyd_files[0]
                break
    
    if pyd_file:
        # Load the module directly from the file
        # The module internally identifies as _vulkan_forge_native
        spec = importlib.util.spec_from_file_location("_vulkan_forge_native", pyd_file)
        _native = importlib.util.module_from_spec(spec)
        sys.modules["_vulkan_forge_native"] = _native
        spec.loader.exec_module(_native)
        
        HeightFieldScene = _native.HeightFieldScene
        Renderer = _native.Renderer
        print(f"Successfully loaded native module from: {pyd_file}")
    else:
        raise ImportError("Could not find .pyd file")
        
except Exception as e:
    import warnings
    warnings.warn(f"Native module not found: {e}")
    
    # Provide dummy classes
    class HeightFieldScene:
        def __init__(self):
            raise RuntimeError("Native Vulkan module not available")
        def build(self, heights, colors=None, zscale=1.0):
            raise RuntimeError("Native Vulkan module not available")
    
    class Renderer:
        def __init__(self, width=800, height=600):
            raise RuntimeError("Native Vulkan module not available")
        def render(self, scene):
            raise RuntimeError("Native Vulkan module not available")

# Wrapper functions for compatibility
def new_scene():
    """Create a new HeightFieldScene"""
    return HeightFieldScene()

def render_to_array(scene, width, height):
    """Render a scene using the Renderer class"""
    renderer = Renderer(width, height)
    return renderer.render(scene)

# Import Python helper modules if needed
try:
    from .heightmap import axes_to_heightfield
except ImportError:
    pass

# Define public API
__all__ = [
    'HeightFieldScene',
    'Renderer',
    'new_scene',
    'render_to_array',
    '__version__',
]