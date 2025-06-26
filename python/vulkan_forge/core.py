import numpy as np
from typing import Tuple

def _ensure_float32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)