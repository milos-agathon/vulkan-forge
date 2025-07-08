"""GeoTIFF loading utilities for Vulkan Forge."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
from PIL import Image

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency
    rasterio = None

from ..terrain.coords import GeographicBounds


class InvalidGeoTiffError(Exception):
    """Raised when a GeoTIFF cannot be parsed."""


class GeoTiffLoader:
    """Simplified GeoTIFF loader with optional rasterio backend."""

    def load(self, path: str) -> Tuple[np.ndarray, Dict[str, object]]:
        """Load a GeoTIFF height map.

        Args:
            path: Path to the GeoTIFF file.

        Returns:
            Tuple of height map array and associated metadata.

        Raises:
            FileNotFoundError: If the path does not exist.
            InvalidGeoTiffError: If the file cannot be decoded.
        """

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(path)

        if rasterio is not None:
            try:
                with rasterio.open(file_path) as src:
                    height_map = src.read(1).astype(np.float32)
                    metadata: Dict[str, object] = {
                        "crs": src.crs.to_string() if src.crs else None,
                        "transform": tuple(src.transform),
                        "width": src.width,
                        "height": src.height,
                        "bounds": GeographicBounds(
                            src.bounds.bottom,
                            src.bounds.top,
                            src.bounds.left,
                            src.bounds.right,
                        ),
                    }
                return height_map, metadata
            except Exception as exc:  # pragma: no cover - passthrough
                raise InvalidGeoTiffError(str(exc)) from exc

        # Fallback path using Pillow for dimensions only
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception as exc:  # pragma: no cover - fallback
            raise InvalidGeoTiffError(str(exc)) from exc

        warnings.warn(
            "rasterio not available; returning zero heightmap",
            RuntimeWarning,
        )
        height_map = np.zeros((height, width), dtype=np.float32)
        metadata = {"width": width, "height": height, "crs": None, "transform": None}
        return height_map, metadata
