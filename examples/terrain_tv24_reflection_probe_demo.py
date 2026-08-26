"""Render a real DEM with TV24 local reflection probes and water."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _terrain_feature_demo import ROOT, load_dem, render, save, side_by_side
from forge3d.terrain_params import ReflectionProbeSettings


DEFAULT_DEM = ROOT / "assets" / "tif" / "dem_rainier.tif"


def render_demo(
    *,
    dem_path: str | Path = DEFAULT_DEM,
    output_dir: str | Path,
    width: int = 960,
    height: int = 600,
    max_dem_size: int = 768,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    dem = load_dem(Path(dem_path), int(max_dem_size))
    water_mask = (dem < float(np.quantile(dem, 0.32))).astype(np.float32)
    settings = ReflectionProbeSettings(
        enabled=True,
        grid_dims=(4, 4),
        resolution=16,
        ray_count=16,
        strength=1.0,
    )
    diffuse, _ = render(dem, width, height, water_mask=water_mask)
    reflection, renderer = render(
        dem,
        width,
        height,
        reflection_probes=settings,
        water_mask=water_mask,
    )
    reflection_debug, _ = render(
        dem,
        width,
        height,
        reflection_probes=settings,
        debug_mode=8,
        water_mask=water_mask,
    )
    reflection_weight, _ = render(
        dem,
        width,
        height,
        reflection_probes=settings,
        debug_mode=53,
        water_mask=water_mask,
    )
    diff = np.abs(
        diffuse[..., :3].astype(np.float32) - reflection[..., :3].astype(np.float32)
    )
    rows = np.minimum(
        (np.arange(int(height)) * water_mask.shape[0] / int(height)).astype(int),
        water_mask.shape[0] - 1,
    )
    columns = np.minimum(
        (np.arange(int(width)) * water_mask.shape[1] / int(width)).astype(int),
        water_mask.shape[1] - 1,
    )
    rendered_water = water_mask[rows[:, None], columns[None, :]] > 0.5
    return {
        "diffuse_path": str(save(output_dir / "diffuse.png", diffuse)),
        "reflection_path": str(save(output_dir / "reflection.png", reflection)),
        "reflection_debug_path": str(
            save(output_dir / "reflection-debug.png", reflection_debug)
        ),
        "reflection_weight_path": str(
            save(output_dir / "reflection-weight.png", reflection_weight)
        ),
        "comparison_path": str(
            save(output_dir / "comparison.png", side_by_side(diffuse, reflection))
        ),
        "mean_abs_diff": float(np.mean(diff)),
        "water_mean_abs_diff": float(np.mean(diff[rendered_water])),
        "water_pixels": int(np.count_nonzero(water_mask > 0.5)),
        "rendered_water_pixels": int(np.count_nonzero(rendered_water)),
        "reflection_probe_memory": renderer.get_reflection_probe_memory_report(),
    }
