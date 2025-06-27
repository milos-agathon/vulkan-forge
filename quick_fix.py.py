#!/usr/bin/env python3
"""Quick fix to set up the project structure correctly"""

import os
import shutil
from pathlib import Path

def setup_project():
    """Set up the project with the correct structure"""
    
    # Create directories
    dirs = [
        'cpp/src',
        'cpp/include/shaders',
        'cpp/shaders',
        'python/vulkan_forge'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")
    
    # Create a simple __init__.py if it doesn't exist
    init_path = Path('python/vulkan_forge/__init__.py')
    if not init_path.exists():
        init_path.write_text('''"""Vulkan Forge - GPU-accelerated height field renderer"""
__version__ = "2.0.0"

try:
    from _vulkan_forge_native import HeightFieldScene, Renderer
    _native_available = True
except ImportError as e:
    print(f"Warning: Native module not available: {e}")
    _native_available = False
    
    # Dummy classes for testing
    class HeightFieldScene:
        def __init__(self):
            self.n_indices = 0
        def build(self, heights, colors=None, zscale=1.0):
            self.n_indices = heights.size * 6
    
    class Renderer:
        def __init__(self, width, height):
            self.width = width
            self.height = height
        def render(self, scene):
            import numpy as np
            return np.ones((self.height, self.width, 4), dtype=np.uint8) * 128

def test():
    """Test the module"""
    print(f"Vulkan Forge {__version__}")
    print(f"Native module available: {_native_available}")
    
    if _native_available:
        import numpy as np
        scene = HeightFieldScene()
        heights = np.ones((10, 10), dtype=np.float32)
        scene.build(heights)
        print(f"Scene built with {scene.n_indices} indices")
        
        renderer = Renderer(100, 100)
        img = renderer.render(scene)
        print(f"Rendered image shape: {img.shape}")

if __name__ == '__main__':
    test()
''')
        print(f"Created: {init_path}")
    
    # Clean build directories
    for d in ['build', '_skbuild', 'dist', '*.egg-info']:
        for path in Path('.').glob(d):
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"Cleaned: {path}")

if __name__ == '__main__':
    setup_project()
    print("\nNow run: pip install -e .")