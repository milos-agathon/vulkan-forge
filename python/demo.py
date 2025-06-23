import matplotlib.pyplot as plt
import numpy as np
from vulkan_forge import render_heightmap

x = np.linspace(-3,3,300); y = np.linspace(-3,3,300)
X,Y = np.meshgrid(x,y); Z = np.sin(X)*np.cos(Y)

fig, ax = plt.subplots()
im = ax.imshow(Z, cmap="magma", origin="lower", extent=[x.min(),x.max(),y.min(),y.max()])
ax.set_title("Matplotlib view") ; plt.show()

rgba = render_heightmap(ax, out_wh=(800,800), spp=64)
plt.imshow(rgba) ; plt.title("Vulkan trace") ; plt.axis("off")
plt.show()
