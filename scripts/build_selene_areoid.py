#!/usr/bin/env python3
"""Build the compact SELENE Mars areoid harmonic container from the PDS map."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct

import numpy as np

PDS_A_M = 3_396_000.0
PDS_INVERSE_FLATTENING = 196.877360
IAU_A_M = 3_396_190.0
IAU_INVERSE_FLATTENING = 169.894447223612
NMAX = 179
MMAX = 90


def legendre(latitudes: np.ndarray) -> np.ndarray:
    idx = lambda n, m: n * (n + 1) // 2 + m
    p = np.zeros((len(latitudes), (NMAX + 1) * (NMAX + 2) // 2))
    cos_theta = np.sin(latitudes)
    sin_theta = np.cos(latitudes)
    p[:, idx(0, 0)] = 1.0
    p[:, idx(1, 0)] = np.sqrt(3.0) * cos_theta
    p[:, idx(1, 1)] = np.sqrt(3.0) * sin_theta
    for m in range(2, NMAX + 1):
        p[:, idx(m, m)] = (
            np.sqrt((2 * m + 1) / (2 * m))
            * sin_theta
            * p[:, idx(m - 1, m - 1)]
        )
    for m in range(NMAX):
        p[:, idx(m + 1, m)] = (
            np.sqrt(2 * m + 3) * cos_theta * p[:, idx(m, m)]
        )
    for m in range(NMAX + 1):
        for n in range(m + 2, NMAX + 1):
            a = np.sqrt((2 * n + 1) / ((n + m) * (n - m)))
            b = np.sqrt(2 * n - 1)
            c = np.sqrt((n + m - 1) * (n - m - 1) / (2 * n - 3))
            p[:, idx(n, m)] = a * (
                b * cos_theta * p[:, idx(n - 1, m)]
                - c * p[:, idx(n - 2, m)]
            )
    return p


def ellipsoid_radius(latitude: np.ndarray, a: float, inverse_flattening: float) -> np.ndarray:
    b = a * (1.0 - 1.0 / inverse_flattening)
    return a * b / np.sqrt(
        (b * np.cos(latitude)) ** 2 + (a * np.sin(latitude)) ** 2
    )


def on_iau_ellipsoid(source: np.ndarray) -> np.ndarray:
    latitude = np.deg2rad(89.5 - np.arange(180))
    correction = ellipsoid_radius(
        latitude, PDS_A_M, PDS_INVERSE_FLATTENING
    ) - ellipsoid_radius(latitude, IAU_A_M, IAU_INVERSE_FLATTENING)
    return source + correction[:, None]


def fit(source: np.ndarray) -> tuple[list[tuple[float, float]], np.ndarray]:
    idx = lambda n, m: n * (n + 1) // 2 + m
    latitudes = np.deg2rad(89.5 - np.arange(180))
    longitudes = np.deg2rad(0.5 + np.arange(360))
    p = legendre(latitudes)
    fft = np.fft.rfft(source, axis=1) / source.shape[1]
    coefficients = [(0.0, 0.0)] * ((NMAX + 1) * (NMAX + 2) // 2)
    reconstructed = np.zeros_like(source)
    for m in range(MMAX + 1):
        degrees = range(m, NMAX + 1)
        basis = p[:, [idx(n, m) for n in degrees]]
        harmonic = fft[:, m] * np.exp(-1j * m * np.deg2rad(0.5))
        factor = 1.0 if m == 0 else 2.0
        cosine = np.linalg.lstsq(basis, harmonic.real * factor, rcond=None)[0]
        sine = (
            np.zeros_like(cosine)
            if m == 0
            else np.linalg.lstsq(basis, -harmonic.imag * factor, rcond=None)[0]
        )
        for n, c, s in zip(degrees, cosine, sine):
            coefficients[idx(n, m)] = (c / IAU_A_M, s / IAU_A_M)
        reconstructed += (basis @ cosine)[:, None] * np.cos(
            m * longitudes
        ) + (basis @ sine)[:, None] * np.sin(m * longitudes)
    return coefficients, reconstructed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pds_map", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.pds_map.read_bytes()
    pds_surface = np.frombuffer(raw, dtype="<f4").reshape(180, 360).astype(np.float64)
    source = on_iau_ellipsoid(pds_surface)
    coefficients, reconstructed = fit(source)
    payload = bytearray(b"F3DAREO1")
    payload += struct.pack("<IIII", 1, NMAX, len(coefficients), 0)
    for cosine, sine in coefficients:
        payload += struct.pack("<dd", cosine, sine)
    args.output.write_bytes(payload)
    error = np.abs(reconstructed - source)
    print(f"source_sha256={hashlib.sha256(raw).hexdigest()}")
    print(f"asset_sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"asset_bytes={len(payload)}")
    print(f"max_grid_error_m={error.max():.12f}")
    print(f"rms_grid_error_m={np.sqrt(np.mean(error * error)):.12f}")


if __name__ == "__main__":
    main()
