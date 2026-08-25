"""Small deterministic assets for tests that exercise file-backed behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_hdr(path: Path, *, width: int = 16, height: int = 8) -> Path:
    """Write a valid uncompressed Radiance RGBE image with a color gradient."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend((64 + 8 * x, 72 + 10 * y, 96 + 4 * x, 129))
    header = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n\n"
        + f"-Y {height} +X {width}\n".encode("ascii")
    )
    path.write_bytes(header + pixels)
    return path


def write_geotiff(path: Path, *, width: int = 96, height: int = 72) -> Path:
    """Write a georeferenced float32 DEM with real relief."""
    import rasterio
    from rasterio.transform import from_bounds

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    dem = (
        1200.0
        + 700.0 * np.exp(-5.0 * (xx * xx + yy * yy))
        + 120.0 * xx
        - 80.0 * yy
    ).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_bounds(0.0, 0.0, 9600.0, 7200.0, width, height),
    ) as dst:
        dst.write(dem, 1)
    return path
