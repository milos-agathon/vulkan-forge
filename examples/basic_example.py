#!/usr/bin/env python3
"""Basic example of using vulkan-forge"""

import numpy as np
import matplotlib.pyplot as plt
from vulkan_forge import VulkanRenderer

# Create test terrain
x = np.linspace(-5, 5, 128)
y = np.linspace(-5, 5, 128)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) + 0.5 * np.cos(X) * np.sin(Y)

# Create renderer
renderer = VulkanRenderer(800, 600)

# Render
image = renderer.render_heightfield(Z.astype(np.float32), z_scale=1.0)

# Display
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.title('Vulkan Rendered Terrain')
plt.axis('off')
plt.tight_layout()
plt.show()
