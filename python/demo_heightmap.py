import numpy as np
import matplotlib.pyplot as plt

# ①  Import vulkan-forge
import vulkan_forge as vf

# ②  Build a simple “terrain”
x = np.linspace(-3, 3, 256)
y = np.linspace(-3, 3, 256)
X, Y = np.meshgrid(x, y)
Z = np.sin(X**2 + Y**2) * np.cos(Y)             # 256 × 256 array

# ③  Classic Matplotlib heat-map (for colour only)
fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
hm = ax.imshow(Z, cmap="plasma", origin="lower")
ax.set_axis_off()
plt.tight_layout()

# ④  Convert the Axes object → Vulkan height-field Scene
scene = vf.heightmap_from_ax(ax,      # any AxesImage / imshow
                             zscale=2.0)   # extrude 0…2 units high

# ⑤  Ray-trace with the shaders you just built
img = vf.render(scene,
                width=1024, height=1024,
                spp=128)               # samples-per-pixel

# ⑥  Display the rendered PNG returned as a NumPy array
plt.figure(figsize=(4, 4), dpi=100)
plt.imshow(img)
plt.axis("off")
plt.show()
