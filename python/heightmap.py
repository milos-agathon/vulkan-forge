import matplotlib.pyplot as plt
import numpy as np

def axes_to_heightfield(ax, scale_z=1.0):
    """
    Down-samples the *current* Axes' RGBA buffer, normalises the luminance
    -> heights, and returns per-texel RGBA colours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    scale_z : float
        Max height in Vulkan units.

    Returns
    -------
    heights : (N,) float32
    colors   : (N,4) float32
    nx, ny   : int
    x_range, y_range : tuple(float, float)
    """
    # draw, then grab the RGBA pixel buffer
    fig = ax.figure
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())        # H×W×4, uint8
    ny, nx, _ = img.shape
    img_f = img.astype(np.float32) / 255.0

    # naive luminance -> height
    lum = (0.2126*img_f[...,0] + 0.7152*img_f[...,1] + 0.0722*img_f[...,2])
    heights = (lum / lum.max() * scale_z).astype(np.float32)
    colors  = img_f.reshape(-1,4).astype(np.float32)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    return (heights.ravel(), colors, nx, ny, (x_min,x_max), (y_min,y_max))
