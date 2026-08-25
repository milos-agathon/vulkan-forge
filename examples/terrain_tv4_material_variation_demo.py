"""Render a real DEM with TV4 procedural material variation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _terrain_feature_demo import ROOT, load_dem, render, save, side_by_side
from forge3d.terrain_params import MaterialLayerSettings, MaterialNoiseSettings


DEFAULT_DEM = ROOT / "assets" / "tif" / "dem_rainier.tif"


def render_demo(
    *, dem_path=DEFAULT_DEM, output_dir, width=960, height=600, max_dem_size=768
):
    output_dir = Path(output_dir)
    dem = load_dem(Path(dem_path), int(max_dem_size))
    baseline_materials = MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=0.68,
        snow_altitude_blend=0.18,
        rock_enabled=True,
        rock_slope_min=34.0,
        rock_slope_blend=12.0,
        wetness_enabled=True,
        wetness_strength=0.42,
    )
    varied_materials = MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=0.68,
        snow_altitude_blend=0.18,
        rock_enabled=True,
        rock_slope_min=34.0,
        rock_slope_blend=12.0,
        wetness_enabled=True,
        wetness_strength=0.42,
        variation=MaterialNoiseSettings(
            macro_scale=4.2,
            detail_scale=20.0,
            octaves=5,
            snow_macro_amplitude=0.26,
            snow_detail_amplitude=0.12,
            rock_macro_amplitude=0.22,
            rock_detail_amplitude=0.16,
            wetness_macro_amplitude=0.24,
            wetness_detail_amplitude=0.12,
        ),
    )
    baseline, _ = render(dem, width, height, materials=baseline_materials)
    varied, _ = render(dem, width, height, materials=varied_materials)
    mean_abs = float(
        np.mean(np.abs(baseline[..., :3].astype(float) - varied[..., :3].astype(float)))
    )
    return {
        "baseline_path": str(save(output_dir / "baseline.png", baseline)),
        "varied_path": str(save(output_dir / "tv4-varied.png", varied)),
        "comparison_path": str(
            save(output_dir / "comparison.png", side_by_side(baseline, varied))
        ),
        "mean_abs_diff": mean_abs,
    }
