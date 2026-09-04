#!/usr/bin/env python3
"""Render Swiss land cover as a 4K, 4x4 native off-axis PT poster."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from _import_shim import ensure_repo_import

ensure_repo_import()

from forge3d.path_tracing import render_terrain_poster

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "tif"
FIXTURE_GRID = 1024
POSTER_SIZE = 4096
TILE_SIZE = 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=ASSETS / "switzerland_dem.tif")
    parser.add_argument("--landcover", type=Path, default=ASSETS / "switzerland_land_cover.tif")
    parser.add_argument("--output", type=Path, default=ROOT / "examples" / "out" / "swiss_landcover_pt_oblique.png")
    return parser.parse_args()


def _require_fixture(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required LFS fixture is missing: {path}")
    if path.stat().st_size <= 1024:
        raise RuntimeError(f"Required LFS fixture is pointer-only or invalid: {path}")


def _load_inputs(dem_path: Path, cover_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("swiss_landcover_pt_oblique.py requires rasterio") from exc

    with rasterio.open(dem_path) as source:
        values = source.read(1, out_shape=(FIXTURE_GRID, FIXTURE_GRID), masked=True).astype(np.float32)
        fill = float(values.mean()) if values.count() else 0.0
        dem = np.asarray(values.filled(fill), dtype=np.float32)

    with rasterio.open(cover_path) as source:
        if source.count >= 3:
            rgb = np.moveaxis(
                source.read([1, 2, 3], out_shape=(3, FIXTURE_GRID, FIXTURE_GRID), masked=True).filled(0),
                0,
                -1,
            ).astype(np.float32)
            if rgb.max(initial=0.0) > 1.0:
                rgb /= 255.0
            valid = np.any(rgb > 0.0, axis=2)
        else:
            classes = source.read(1, out_shape=(FIXTURE_GRID, FIXTURE_GRID), masked=True)
            colormap = source.colormap(1)
            if not colormap:
                raise ValueError("single-band Swiss land-cover fixture has no color table")
            palette = np.zeros((max(colormap) + 1, 4), dtype=np.uint8)
            for index, color in colormap.items():
                palette[index] = color
            indices = np.asarray(classes.filled(0), dtype=np.int64)
            rgba8 = palette[np.clip(indices, 0, len(palette) - 1)]
            rgb = rgba8[..., :3].astype(np.float32) / 255.0
            valid = (~np.ma.getmaskarray(classes)) & (rgba8[..., 3] > 0)

    albedo = np.empty((FIXTURE_GRID, FIXTURE_GRID, 4), dtype=np.float32)
    albedo[..., :3] = np.clip(rgb, 0.0, None)
    albedo[..., 3] = valid.astype(np.float32)
    return np.ascontiguousarray(dem), np.ascontiguousarray(albedo)


def _camera(dem: np.ndarray) -> dict:
    tilt = math.radians(35.0)
    span = float(max(dem.shape) - 1)
    relief = float(np.max(dem) - np.min(dem))
    distance = max(8.0 * span, 2.25 * relief)
    half_height = 0.30 * span
    center_y = float(dem[dem.shape[0] // 2, dem.shape[1] // 2])
    return {
        "model": "off_axis",
        "origin": (
            0.0,
            center_y + distance * math.cos(tilt),
            distance * math.sin(tilt),
        ),
        "look_at": (0.0, center_y, 0.0),
        "up": (0.0, math.sin(tilt), -math.cos(tilt)),
        "fov_y": math.degrees(2.0 * math.atan(half_height / distance)),
    }


def main() -> int:
    args = _parse_args()
    _require_fixture(args.dem)
    _require_fixture(args.landcover)
    dem, albedo = _load_inputs(args.dem, args.landcover)
    certificate = args.output.with_suffix(".certificate.json")
    result = render_terrain_poster(
        dem,
        POSTER_SIZE,
        POSTER_SIZE,
        _camera(dem),
        tile=TILE_SIZE,
        albedo_map=albedo,
        albedo_sampling="bilinear",
        sun_azimuth_deg=315.0,
        sun_elevation_deg=28.0,
        sun_intensity=2.5,
        env_intensity=0.15,
        min_frames=32,
        max_frames=32,
        variance_threshold=1e9,
        seed=7,
        spp=1,
        certificate=certificate,
    )
    if len(result["tiles"]) != 16:
        raise RuntimeError(f"expected a 4x4 tile layout, got {len(result['tiles'])} tiles")
    for index, tile in enumerate(result["tiles"]):
        if not tile["converged"]:
            raise RuntimeError(f"tile {index} did not converge")
        if tile["peak_host_visible_bytes"] > 512 * 1024 * 1024:
            raise RuntimeError(f"tile {index} exceeded the 512 MiB host-visible budget")
        if tile["reservoir_valid_count"] <= 0:
            raise RuntimeError(f"tile {index} has no valid sun-facing ReSTIR reservoir")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(result["rgba"], dtype=np.uint8), mode="RGBA").save(args.output)
    artifact_root = Path(os.environ.get("FORGE3D_OBLIQUA_ARTIFACT_DIR", args.output.parent))
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_root / "swiss_landcover_pt_oblique_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "output": str(args.output),
                "certificate": str(certificate),
                "certificate_digest": result["certificate_digest"],
                "tile_layout": "4x4",
                "tiles": result["tiles"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved poster: {args.output}")
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
