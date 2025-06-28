#!/usr/bin/env python3
"""Simple example using the actual vulkan_forge implementation"""

import numpy as np
import matplotlib.pyplot as plt
from vulkan_forge import HeightFieldScene, Renderer

# Create test terrain - multiple peaks and valleys
print("Creating terrain...")
size = 128
x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)
X, Y = np.meshgrid(x, y)

# Create interesting height field
terrain = np.zeros_like(X)

# Add some peaks
terrain += np.exp(-((X - 3)**2 + (Y - 3)**2) / 5) * 1.5  # Peak 1
terrain += np.exp(-((X + 3)**2 + (Y + 3)**2) / 5) * 1.2  # Peak 2
terrain += np.exp(-((X - 3)**2 + (Y + 3)**2) / 8) * 0.8  # Peak 3

# Add some rolling hills
terrain += 0.3 * np.sin(X * 0.5) * np.cos(Y * 0.5)

# Add noise
terrain += 0.05 * np.random.randn(*X.shape)

# Convert to float32
terrain = terrain.astype(np.float32)

print(f"Terrain shape: {terrain.shape}")
print(f"Height range: [{terrain.min():.2f}, {terrain.max():.2f}]")

# Create and build scene
print("\nBuilding scene...")
scene = HeightFieldScene()
scene.build(terrain, zscale=0.5)
print(f"Scene built with {scene.n_indices} indices")

# Render at different sizes
render_sizes = [(200, 200), (400, 300), (600, 400)]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Show original terrain
im = axes[0].imshow(terrain, cmap='terrain', origin='lower', 
                    extent=[-10, 10, -10, 10])
axes[0].set_title('Original Terrain')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im, ax=axes[0])

# Render at different resolutions
for i, (width, height) in enumerate(render_sizes):
    print(f"\nRendering at {width}x{height}...")
    
    renderer = Renderer(width, height)
    img = renderer.render(scene)
    
    axes[i+1].imshow(img)
    axes[i+1].set_title(f'Rendered {width}x{height}')
    axes[i+1].axis('off')
    
    print(f"  Rendered image: shape={img.shape}")

plt.tight_layout()
plt.savefig('simple_example_output.png', dpi=150, bbox_inches='tight')
print(f"\nSaved output to simple_example_output.png")

plt.show()

# Also test with VulkanRenderer wrapper if available
try:
    from vulkan_forge import VulkanRenderer
    print("\n\nTesting VulkanRenderer wrapper...")
    
    vk_renderer = VulkanRenderer(800, 600)
    img = vk_renderer.render_heightfield(terrain, z_scale=0.5)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title('VulkanRenderer Output (800x600)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('vulkan_renderer_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ VulkanRenderer test successful")
    
except ImportError:
    print("\nVulkanRenderer not available in this version")