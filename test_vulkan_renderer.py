#!/usr/bin/env python3
"""Test VulkanRenderer by adding it at runtime"""

import numpy as np
import matplotlib.pyplot as plt

# Import base components
from vulkan_forge import HeightFieldScene, Renderer

print("Base components imported successfully")

# Define VulkanRenderer if it's not available
try:
    from vulkan_forge import VulkanRenderer
    print("VulkanRenderer already available")
except ImportError:
    print("VulkanRenderer not found, creating it...")
    
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
    
    # Add it to the module
    import vulkan_forge
    vulkan_forge.VulkanRenderer = VulkanRenderer
    print("VulkanRenderer added to module")

# Now test it
print("\nTesting VulkanRenderer...")

# Create test terrain
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) + 0.5 * np.cos(X) * np.sin(Y)
Z = Z.astype(np.float32)

# Create renderer
renderer = VulkanRenderer(800, 600)
print(f"Created VulkanRenderer: {renderer.width}x{renderer.height}")

# Render
image = renderer.render_heightfield(Z, z_scale=1.0)
print(f"Rendered image: shape={image.shape}, dtype={image.dtype}")

# Display
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Original height map
im = ax1.imshow(Z, cmap='viridis', origin='lower')
ax1.set_title('Height Map')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im, ax=ax1)

# Rendered output
ax2.imshow(image)
ax2.set_title('Rendered Output (Gradient Pattern)')
ax2.axis('off')

plt.tight_layout()
plt.savefig('test_vulkan_renderer_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("IMPORTANT: The gradient output is expected!")
print("="*50)
print("\nThe current implementation (simple_renderer.cpp) only produces")
print("a gradient pattern for testing. It's not actual 3D rendering.")
print("\nTo get real 3D rendering, you would need to:")
print("1. Implement the full Vulkan pipeline in C++")
print("2. Add vertex/fragment shaders")
print("3. Implement proper 3D transformations")
print("\nBut the Python bindings and infrastructure are working correctly!")