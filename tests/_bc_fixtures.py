"""Deterministic, demanding fixtures for the TESSELLA BC fidelity gates.

The fixtures are generated from an integer hash (splitmix64 finaliser) rather
than a NumPy ``Generator``: NumPy does not guarantee bit-stable streams across
releases, and win 4 is a numeric gate that must reproduce exactly on every
machine and every NumPy version.

``hard_albedo`` is deliberately hostile to a single-subset BC7 mode: it mixes
multi-octave detail, saturated chroma patches with hard rectangular edges,
1-pixel and 2-pixel checkerboards, and thin high-contrast lines. ``smooth_albedo``
is the easy baseline kept for comparison.
"""

from __future__ import annotations

import numpy as np

_U64 = np.uint64


def _splitmix(values: np.ndarray) -> np.ndarray:
    """Deterministic uint64 -> [0, 1) float64, stable on every platform."""
    h = values.astype(np.uint64)
    h = h ^ (h >> _U64(33))
    h = h * _U64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> _U64(33))
    h = h * _U64(0xC4CEB9FE1A85EC53)
    h = h ^ (h >> _U64(33))
    return (h >> _U64(11)).astype(np.float64) / float(1 << 53)


def _lattice(cells: int, seed: int, salt: int) -> np.ndarray:
    index = np.arange(cells + 1, dtype=np.uint64)
    yy, xx = np.meshgrid(index, index, indexing="ij")
    mask = (1 << 64) - 1
    seed_term = _U64((seed * 0x165667B19E3779F9) & mask)
    salt_term = _U64((salt * 0x27D4EB2F165667C5) & mask)
    mixed = (
        yy * _U64(0x9E3779B97F4A7C15)
        ^ xx * _U64(0xC2B2AE3D27D4EB4F)
        ^ seed_term
        ^ salt_term
    )
    return _splitmix(mixed)


def value_noise(side: int, cells: int, seed: int, salt: int) -> np.ndarray:
    """Smooth-step bilinear upsample of a hashed coarse lattice."""
    lattice = _lattice(cells, seed, salt)
    coords = np.linspace(0, cells, side, endpoint=False)
    base = np.floor(coords).astype(int)
    frac = coords - base
    frac = frac * frac * (3.0 - 2.0 * frac)
    ty = frac[:, None]
    tx = frac[None, :]
    y0 = base[:, None]
    x0 = base[None, :]
    a = lattice[y0, x0]
    b = lattice[y0, x0 + 1]
    c = lattice[y0 + 1, x0]
    d = lattice[y0 + 1, x0 + 1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def smooth_albedo(side: int = 64) -> np.ndarray:
    """The easy baseline: a pure linear gradient, near-best-case for BC7 mode 6."""
    y, x = np.mgrid[:side, :side].astype(np.float32)
    rgba = np.empty((side, side, 4), dtype=np.uint8)
    rgba[..., 0] = np.round(72.0 + x * 0.45).astype(np.uint8)
    rgba[..., 1] = np.round(96.0 + y * 0.40).astype(np.uint8)
    rgba[..., 2] = np.round(128.0 + (x + y) * 0.20).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def hard_albedo(side: int = 256, seed: int = 19) -> np.ndarray:
    """Multi-octave detail + saturated hard edges + checkerboards + thin lines."""
    image = np.zeros((side, side, 3), dtype=np.float64)
    for salt, (cells, weight) in enumerate(((4, 0.5), (8, 0.25), (16, 0.15), (32, 0.10))):
        for channel in range(3):
            image[..., channel] += weight * value_noise(side, cells, seed, salt * 3 + channel)
    image /= image.max()

    y, x = np.mgrid[:side, :side]
    scale = side / 256.0

    def box(y0, y1, x0, x1):
        return (
            slice(int(y0 * scale), int(y1 * scale)),
            slice(int(x0 * scale), int(x1 * scale)),
        )

    patches = (
        ((10, 70, 10, 70), (0.90, 0.08, 0.06)),
        ((10, 70, 90, 150), (0.05, 0.85, 0.10)),
        ((10, 70, 170, 230), (0.06, 0.10, 0.92)),
        ((90, 150, 10, 70), (0.95, 0.85, 0.05)),
        ((90, 150, 90, 150), (0.90, 0.05, 0.85)),
        ((90, 150, 170, 230), (0.05, 0.88, 0.90)),
    )
    for bounds, colour in patches:
        rows, cols = box(*bounds)
        image[rows, cols] = colour

    rows, cols = box(170, 210, 10, 110)
    checker_1px = ((x + y) % 2 == 0)[rows, cols]
    image[rows, cols] = np.where(checker_1px[..., None], 0.95, 0.05)

    rows, cols = box(170, 210, 130, 230)
    checker_2px = (((x // 2) + (y // 2)) % 2 == 0)[rows, cols]
    image[rows, cols] = np.where(
        checker_2px[..., None], (0.92, 0.15, 0.10), (0.10, 0.20, 0.90)
    )

    rows, _ = box(220, 250, 0, 1)
    image[rows, ::3] = 0.98
    inner, _ = box(225, 245, 0, 1)
    image[inner, 1::7] = 0.02

    rgba = np.empty((side, side, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def flat_normals(side: int = 64) -> np.ndarray:
    """The easy baseline: a gentle dome, tilts under ~12 degrees."""
    y, x = np.mgrid[:side, :side].astype(np.float64)
    nx = (x / (side - 1) - 0.5) * 0.28
    ny = (y / (side - 1) - 0.5) * 0.28
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 0.0))
    return np.stack([nx, ny, nz], axis=-1)


def steep_normals(side: int = 256, seed: int = 23, slope: float = 3.0) -> np.ndarray:
    """A 5-octave surface whose normals tilt up to ~32 degrees."""
    height = np.zeros((side, side), dtype=np.float64)
    for salt, (cells, weight) in enumerate(
        ((4, 1.0), (8, 0.5), (16, 0.25), (32, 0.125), (64, 0.06))
    ):
        height += weight * value_noise(side, cells, seed, salt)
    gradient_y, gradient_x = np.gradient(height * slope * side / 64.0)
    nx, ny, nz = -gradient_x, -gradient_y, np.ones_like(gradient_x)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1)


def normal_tilt_degrees(normals: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(normals[..., 2], -1.0, 1.0)))
