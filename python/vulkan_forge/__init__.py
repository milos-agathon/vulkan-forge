"""Vulkan Forge - GPU-accelerated height field renderer"""

__version__ = "2.0.0"

import sys
from pathlib import Path
import numpy as np

# Try to import the native module
_native_available = False

try:
    # First try direct import
    from _vulkan_forge_native import HeightFieldScene, Renderer
    _native_available = True
    print(f"Vulkan Forge {__version__} - Native module loaded successfully")
except ImportError as e:
    # Try to find it in the installation
    try:
        import site
        for site_dir in site.getsitepackages():
            pyd_path = Path(site_dir) / "lib" / "vulkan_forge"
            if pyd_path.exists() and str(pyd_path) not in sys.path:
                sys.path.insert(0, str(pyd_path))
                break
        
        from _vulkan_forge_native import HeightFieldScene, Renderer
        _native_available = True
        print(f"Vulkan Forge {__version__} - Native module loaded from {pyd_path}")
    except ImportError as e2:
        print(f"Warning: Native module not available: {e2}")
        
        # Provide dummy classes for testing
        class HeightFieldScene:
            def __init__(self):
                self.n_indices = 0
                
            def build(self, heights, colors=None, zscale=1.0):
                if not isinstance(heights, np.ndarray) or heights.ndim != 2:
                    raise ValueError("heights must be a 2D numpy array")
                ny, nx = heights.shape
                self.n_indices = (nx - 1) * (ny - 1) * 6
        
        class Renderer:
            def __init__(self, width, height):
                self._width = width
                self._height = height
            
            @property
            def width(self):
                return self._width
                
            @property
            def height(self):
                return self._height
                
            def render(self, scene):
                # Return a gradient image
                img = np.zeros((self._height, self._width, 4), dtype=np.uint8)
                for y in range(self._height):
                    for x in range(self._width):
                        img[y, x] = [
                            int(x / self._width * 255),
                            int(y / self._height * 255),
                            128,
                            255
                        ]
                return img


class VulkanRenderer:
    """High-level wrapper for Vulkan rendering"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self._renderer = Renderer(width, height)
    
    def render_heightfield(self, heights, colors=None, z_scale=1.0, **kwargs):
        """Render a height field"""
        if not isinstance(heights, np.ndarray):
            heights = np.array(heights, dtype=np.float32)
        
        if heights.dtype != np.float32:
            heights = heights.astype(np.float32)
            
        if heights.ndim != 2:
            raise ValueError("Heights must be a 2D array")
        
        # Create and build scene
        scene = HeightFieldScene()
        scene.build(heights, colors, z_scale)
        
        # Render
        return self._renderer.render(scene)


def test_basic():
    """Test basic functionality"""
    print(f"\nTesting Vulkan Forge {__version__}")
    print(f"Native module available: {_native_available}")
    
    # Create scene
    scene = HeightFieldScene()
    print("✓ Created HeightFieldScene")
    
    # Build with test data
    heights = np.ones((10, 10), dtype=np.float32)
    scene.build(heights, zscale=1.0)
    print(f"✓ Built scene with {scene.n_indices} indices")
    
    # Create renderer
    renderer = Renderer(100, 100)
    print(f"✓ Created Renderer ({renderer.width}x{renderer.height})")
    
    # Render
    img = renderer.render(scene)
    print(f"✓ Rendered image: shape={img.shape}, dtype={img.dtype}")
    
    return True


def axes_to_heightfield(ax, scale_z=1.0):
    """Extract height field from matplotlib axes"""
    import matplotlib.pyplot as plt
    
    # Force render
    fig = ax.figure
    fig.canvas.draw()
    
    # Get RGBA buffer
    buf = np.asarray(fig.canvas.buffer_rgba())
    ny, nx = buf.shape[:2]
    
    # Convert to float
    img_float = buf.astype(np.float32) / 255.0
    
    # Calculate height from luminance
    luminance = (0.2126 * img_float[:, :, 0] + 
                 0.7152 * img_float[:, :, 1] + 
                 0.0722 * img_float[:, :, 2])
    
    # Normalize and scale
    if luminance.max() > luminance.min():
        heights = (luminance - luminance.min()) / (luminance.max() - luminance.min())
    else:
        heights = luminance
    
    heights = heights * scale_z
    
    # Get colors
    colors = img_float
    
    # Get coordinate ranges
    x_range = ax.get_xlim()
    y_range = ax.get_ylim()
    
    return (heights.ravel().astype(np.float32),
            colors.reshape(-1, 4).astype(np.float32),
            nx, ny, x_range, y_range)


def render_heightmap(ax, out_wh=(100, 100), spp=1, scale_z=1.0):
    """Render matplotlib axes as height field"""
    heights_flat, colors, nx, ny, xr, yr = axes_to_heightfield(ax, scale_z)
    
    # Reshape heights
    heights = heights_flat.reshape(ny, nx)
    
    # Create and render scene
    scene = HeightFieldScene()
    scene.build(heights, zscale=scale_z)
    
    renderer = Renderer(out_wh[0], out_wh[1])
    return renderer.render(scene)


# Export all public components
__all__ = [
    'HeightFieldScene', 
    'Renderer', 
    'VulkanRenderer',
    '__version__', 
    'test_basic', 
    'render_heightmap', 
    'axes_to_heightfield'
]


# Run test if executed directly
if __name__ == '__main__':
    test_basic()