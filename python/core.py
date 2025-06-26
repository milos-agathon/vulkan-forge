import numpy as np
import importlib
from typing import Tuple

# shared library produced by scikit-build
_native = importlib.import_module("vulkan_forge_native")

def _ensure_float32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)

# -------- public convenience helpers ---------------------------------
def new_scene() -> "_native.HeightFieldScene":
    return _native.new_scene()

def set_heightfield(scene, heights: np.ndarray, colors: np.ndarray,
                    nx: int, ny: int, x_range: Tuple[float, float],
                    y_range: Tuple[float, float]) -> None:
    _native.set_heightfield(scene,
                            _ensure_float32(heights).ravel(),
                            _ensure_float32(colors).ravel(),
                            nx, ny,
                            float(x_range[0]), float(x_range[1]),
                            float(y_range[0]), float(y_range[1]))

def set_camera(scene, pos, target, fov=45.0):
    _native.set_camera(scene,
                       (float(pos[0]), float(pos[1]), float(pos[2])),
                       (float(target[0]), float(target[1]), float(target[2])),
                       float(fov))

def render(scene, width: int, height: int, spp: int = 32) -> np.ndarray:
    raw = _native.render(scene, width, height, int(spp))
    img  = np.frombuffer(raw, np.uint8).copy()
    img.shape = (height, width, 4)
    return img
