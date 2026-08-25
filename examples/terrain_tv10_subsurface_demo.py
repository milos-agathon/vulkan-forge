"""Render real mountain DEMs with TV10 terrain subsurface response."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _terrain_feature_demo import ROOT, load_dem, render, save, side_by_side
from forge3d.terrain_params import MaterialLayerSettings


DEFAULT_DEMS = (
    ROOT / "assets" / "tif" / "dem_rainier.tif",
    ROOT / "assets" / "tif" / "Gore_Range_Albers_1m.tif",
)


def render_demo(*, output_dir, width=960, height=600, max_dem_size=768):
    output_dir = Path(output_dir)
    baseline_materials = MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=0.78,
        snow_altitude_blend=0.24,
        snow_slope_max=58.0,
        snow_slope_blend=18.0,
        rock_enabled=True,
        rock_slope_min=38.0,
        rock_slope_blend=10.0,
        wetness_enabled=True,
        wetness_strength=0.18,
        wetness_slope_influence=0.45,
    )
    subsurface_materials = MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=0.78,
        snow_altitude_blend=0.24,
        snow_slope_max=58.0,
        snow_slope_blend=18.0,
        rock_enabled=True,
        rock_slope_min=38.0,
        rock_slope_blend=10.0,
        wetness_enabled=True,
        wetness_strength=0.18,
        wetness_slope_influence=0.45,
        snow_subsurface_strength=1.0,
        snow_subsurface_tint=(0.58, 0.80, 1.0),
        rock_subsurface_strength=1.0,
        rock_subsurface_tint=(0.76, 0.46, 0.28),
        wetness_subsurface_strength=1.0,
        wetness_subsurface_tint=(0.32, 0.44, 0.68),
    )
    scenes = []
    comparisons = []
    for index, dem_path in enumerate(DEFAULT_DEMS):
        dem = load_dem(dem_path, int(max_dem_size))
        baseline, _ = render(
            dem,
            width,
            height,
            materials=baseline_materials,
            albedo_mode="mix",
            colormap_strength=0.25,
            terrain_span=2.9,
            z_scale=1.45,
            light_azimuth_deg=132.0,
            light_elevation_deg=11.0,
            sun_intensity=2.6,
            exposure=1.0,
            cam_radius=4.2,
            cam_phi_deg=138.0,
            cam_theta_deg=42.0,
            fov_y_deg=42.0,
            camera_mode="screen",
            ibl_intensity=1.0,
        )
        subsurface, _ = render(
            dem,
            width,
            height,
            materials=subsurface_materials,
            albedo_mode="mix",
            colormap_strength=0.25,
            terrain_span=2.9,
            z_scale=1.45,
            light_azimuth_deg=132.0,
            light_elevation_deg=11.0,
            sun_intensity=2.6,
            exposure=1.0,
            cam_radius=4.2,
            cam_phi_deg=138.0,
            cam_theta_deg=42.0,
            fov_y_deg=42.0,
            camera_mode="screen",
            ibl_intensity=1.0,
        )
        comparison = side_by_side(baseline, subsurface)
        comparisons.append(comparison)
        mean_abs = float(
            np.mean(
                np.abs(
                    baseline[..., :3].astype(np.float32)
                    - subsurface[..., :3].astype(np.float32)
                )
            )
        )
        peak_p99 = float(
            np.percentile(
                np.abs(
                    baseline[..., :3].astype(np.float32)
                    - subsurface[..., :3].astype(np.float32)
                ),
                99.0,
            )
        )
        scene_dir = output_dir / f"scene-{index + 1}"
        scenes.append(
            {
                "baseline_path": str(save(scene_dir / "baseline.png", baseline)),
                "subsurface_path": str(save(scene_dir / "subsurface.png", subsurface)),
                "comparison_path": str(save(scene_dir / "comparison.png", comparison)),
                "mean_abs_diff": mean_abs,
                "peak_p99_diff": peak_p99,
            }
        )
    summary = np.concatenate(comparisons, axis=0)
    return {
        "summary_path": str(save(output_dir / "summary.png", summary)),
        "scenes": scenes,
    }
