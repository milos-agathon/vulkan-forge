from .core import new_scene, set_heightfield, set_camera, render_to_array as render
from .heightmap import axes_to_heightfield
import numpy as np

class VulkanContext:
    """ RAII helper so users don't need to manage the native handle. """
    def __init__(self):
        self._scene = new_scene()

    # expose methods
    def set_heightfield(self, *args, **kwargs): set_heightfield(self._scene,*args,**kwargs)
    def set_camera(self, *args, **kw): set_camera(self._scene,*args,**kw)
    def render(self,*args,**kw): return render(self._scene,*args,**kw)

def render_heightmap(ax, out_wh=(1024,1024), spp=32, scale_z=1.0):
    heights, colors, nx, ny, xr, yr = axes_to_heightfield(ax, scale_z)
    ctx = VulkanContext()
    ctx.set_heightfield(heights, colors, nx, ny, xr, yr)
    ctx.set_camera((xr[1]*1.1, yr[1]*1.1, scale_z*2.0),
                   (np.mean(xr), np.mean(yr), 0.0))
    w,h = out_wh
    return ctx.render(w,h,spp)
