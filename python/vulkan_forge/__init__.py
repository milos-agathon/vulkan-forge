# vulkan_forge/__init__.py
"""Vulkan-accelerated height-field renderer for Matplotlib"""

__version__ = "0.1.0"

import sys
import os
from pathlib import Path

# Find and load the native module
HeightFieldScene = None
Renderer = None

try:
    # In editable installs, the module is in site-packages/lib/vulkan_forge/
    import site
    for site_dir in site.getsitepackages():
        pyd_path = Path(site_dir) / "lib" / "vulkan_forge"
        if pyd_path.exists():
            # Add this path to sys.path so we can import the module
            if str(pyd_path) not in sys.path:
                sys.path.insert(0, str(pyd_path))
            
            # The module is named 'vulkan_forge' in the .pyd
            # We need to import it with a different name to avoid conflicts
            import importlib
            
            # First, remove any existing 'vulkan_forge' from sys.modules
            # to avoid conflicts with this package
            if 'vulkan_forge' in sys.modules:
                # Save reference to this module
                this_module = sys.modules['vulkan_forge']
                # Temporarily remove it
                del sys.modules['vulkan_forge']
                
                try:
                    # Import the native module
                    native = importlib.import_module('vulkan_forge')
                    
                    # Get the classes
                    HeightFieldScene = native.HeightFieldScene
                    Renderer = native.Renderer
                    
                    # Store the native module under a different name
                    sys.modules['_vulkan_forge_native'] = native
                finally:
                    # Restore this module
                    sys.modules['vulkan_forge'] = this_module
            
            break
    
    if HeightFieldScene is None:
        raise ImportError("Could not find native module classes")
        
except Exception as e:
    import warnings
    warnings.warn(f"Native Vulkan extension not found: {e}")
    
    # Provide dummy classes if native module not available
    class HeightFieldScene:
        def __init__(self):
            raise RuntimeError("Native Vulkan module not available")
    
    class Renderer:
        def __init__(self, width=800, height=600):
            raise RuntimeError("Native Vulkan module not available")

# For backward compatibility with the functional API
def new_scene():
    """Create a new HeightFieldScene"""
    return HeightFieldScene()

def render_to_array(scene, width, height):
    """Render a scene using the Renderer class"""
    renderer = Renderer(width, height)
    return renderer.render(scene)

# Define what gets imported with "from vulkan_forge import *"
__all__ = [
    'HeightFieldScene',
    'Renderer',
    'new_scene',
    'render_to_array',
]