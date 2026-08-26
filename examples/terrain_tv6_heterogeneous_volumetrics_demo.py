"""Render real-DEM TV6 heterogeneous-volume snapshots through the viewer."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _import_shim import ensure_repo_import
from _terrain_feature_demo import ROOT, load_dem, save

ensure_repo_import()

import forge3d as f3d
from forge3d.terrain_params import (
    localized_haze_volume,
    plume_volume,
    valley_fog_volume,
)
from forge3d.viewer import open_viewer_async


DEFAULT_DEM = ROOT / "assets" / "tif" / "dem_rainier.tif"


def _write_viewer_dem(path: Path, source: Path, max_size: int) -> tuple[int, int]:
    import rasterio
    from rasterio.transform import from_origin

    heightmap = load_dem(source, max_size) * np.float32(80.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=heightmap.shape[1],
        height=heightmap.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0.0, float(heightmap.shape[0]), 1.0, 1.0),
    ) as dataset:
        dataset.write(heightmap, 1)
    return int(heightmap.shape[1]), int(heightmap.shape[0])


def _volume_payload(name: str, terrain_width: int, terrain_height: int) -> dict:
    center = (terrain_width * 0.5, 22.0, terrain_height * 0.5)
    size = (terrain_width * 0.76, 54.0, terrain_height * 0.76)
    constructors = {
        "valley_fog": lambda: valley_fog_volume(
            center=center,
            size=size,
            resolution=(48, 32, 48),
            density_scale=1.2,
            noise_strength=0.45,
            seed=7,
        ),
        "plume": lambda: plume_volume(
            center=center,
            size=(terrain_width * 0.42, 72.0, terrain_height * 0.42),
            resolution=(40, 64, 40),
            density_scale=1.35,
            noise_strength=0.55,
            seed=11,
        ),
        "localized_haze": lambda: localized_haze_volume(
            center=center,
            size=size,
            resolution=(48, 32, 48),
            density_scale=1.1,
            noise_strength=0.38,
            seed=19,
        ),
    }
    return asdict(constructors[name]())


def _metrics(baseline: np.ndarray, current: np.ndarray) -> tuple[float, int]:
    delta = np.abs(current[..., :3].astype(np.int16) - baseline[..., :3].astype(np.int16))
    return float(delta.mean()), int(np.count_nonzero(np.any(delta > 0, axis=-1)))


def render_demo(
    *,
    dem_path: str | Path = DEFAULT_DEM,
    output_dir: str | Path,
    width: int = 960,
    height: int = 600,
    max_dem_size: int = 768,
    timeout: float = 90.0,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dem = output_dir / "viewer-dem.tif"
    terrain_width, terrain_height = _write_viewer_dem(
        prepared_dem, Path(dem_path), int(max_dem_size)
    )
    started = time.perf_counter()
    viewer = open_viewer_async(
        width=int(width),
        height=int(height),
        terrain_path=prepared_dem,
        timeout=float(timeout),
    )
    try:
        viewer.set_orbit_camera(
            phi_deg=140.0,
            theta_deg=58.0,
            radius=float(max(terrain_width, terrain_height) * 1.25),
            fov_deg=50.0,
            target=(terrain_width * 0.5, 24.0, -terrain_height * 0.5),
        )
        baseline_started = time.perf_counter()
        viewer.send_ipc(
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "volumetrics": {"enabled": False},
            }
        )
        baseline_path = output_dir / "baseline.png"
        viewer.snapshot(baseline_path, width=width, height=height)
        baseline_seconds = time.perf_counter() - baseline_started
        baseline = f3d.png_to_numpy(baseline_path)
        baseline_report = viewer.get_terrain_volumetrics_report()

        scenes = {}
        frames = [baseline]
        for name in ("valley_fog", "plume", "localized_haze"):
            scene_started = time.perf_counter()
            viewer.send_ipc(
                {
                    "cmd": "set_terrain_pbr",
                    "enabled": True,
                    "volumetrics": {
                        "enabled": True,
                        "mode": "heterogeneous",
                        "density": 0.008,
                        "scattering": 0.82,
                        "absorption": 0.12,
                        "steps": 48,
                        "half_res": False,
                        "density_volumes": [
                            _volume_payload(name, terrain_width, terrain_height)
                        ],
                    },
                }
            )
            path = output_dir / f"{name}.png"
            viewer.snapshot(path, width=width, height=height)
            frame = f3d.png_to_numpy(path)
            mean_abs_diff, changed_pixels = _metrics(baseline, frame)
            scenes[name] = {
                "path": str(path),
                "report": viewer.get_terrain_volumetrics_report(),
                "mean_abs_diff": mean_abs_diff,
                "changed_pixels": changed_pixels,
                "render_seconds": time.perf_counter() - scene_started,
            }
            frames.append(frame)
    finally:
        viewer.close()

    gap = np.zeros((int(height), 10, 4), dtype=np.uint8)
    gap[..., 3] = 255
    contact_sheet = np.concatenate(
        [
            item
            for index, frame in enumerate(frames)
            for item in ((gap, frame) if index else (frame,))
        ],
        axis=1,
    )
    contact_sheet_path = save(output_dir / "contact-sheet.png", contact_sheet)
    total_seconds = time.perf_counter() - started
    manifest = {
        "baseline_path": str(baseline_path),
        "baseline_report": baseline_report,
        "baseline_render_seconds": baseline_seconds,
        "scenes": scenes,
        "contact_sheet_path": str(contact_sheet_path),
        "total_render_seconds": total_seconds,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path)}
