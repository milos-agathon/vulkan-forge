#!/usr/bin/env python3
"""Render a real LOLA south-polar DEM in its typed lunar CRS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d import certificate, diagnostics, gis


ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "assets" / "tif" / "moon_south_pole_lola.tif"
DEM_SHA256 = "575e4b57d69d5db1bc16d5cd7edc8c83c570e280717e9e7ca225c060fd0aeb3d"
LUNAR_CRS = "IAU:30110"


def build_scene(snapshot: str | Path) -> f3d.MapScene:
    """Load the committed GeoTIFF and build the normal terrain render recipe."""
    raster = gis.read_raster(DEM_PATH)
    info = raster["info"]
    if info["crs_authority"] != {"name": "IAU", "code": "30110"}:
        raise RuntimeError("LOLA tile must carry the typed lunar CRS IAU:30110")
    body = f3d.crs.body_info("moon")
    heightmap = np.ascontiguousarray(raster["array"][0], dtype=np.float32)
    metadata = {
        "source_id": "nasa-svs-lola-ldem4-south-pole",
        "body": body["name"],
        "body_radius_m": body["semi_major_m"],
        "width": info["width"],
        "height": info["height"],
        "bounds": info["bounds"],
        "resolution": info["resolution"],
        "transform": info["transform"],
        "height_system": info["height_system"],
    }
    terrain_span = float(info["resolution"][0]) * float(info["width"])
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=heightmap,
            crs=LUNAR_CRS,
            metadata=metadata,
            elevation_sampling_available=True,
        ),
        target_crs=LUNAR_CRS,
        camera=f3d.OrbitCamera(
            target=(0.0, 0.0, 0.0),
            distance=terrain_span * 1.15,
            azimuth_deg=225.0,
            elevation_deg=52.0,
            fov_deg=44.0,
        ),
        lighting=f3d.LightingPreset(
            name="lunar_low_sun",
            sun_direction=(-0.72, 0.18, -0.67),
            intensity=2.1,
            settings={
                "resolved_preset": "lunar_low_sun",
                "colormap": "#18191c,#45474c,#85868a,#d8d7d1",
                "exaggeration": 8.0,
                "albedo_mode": "colormap",
                "colormap_strength": 1.0,
            },
        ),
        output=f3d.OutputSpec(
            width=512,
            height=512,
            format="png",
            path=str(snapshot),
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(
            seed=27,
            output_size=(512, 512),
            asset_hashes_or_ids={"lola_dem_sha256": DEM_SHA256},
        ),
    )


def run_example(
    snapshot: str | Path,
    certificate_path: str | Path,
) -> dict[str, Any]:
    snapshot = Path(snapshot)
    certificate_path = Path(certificate_path)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    certificate_path.parent.mkdir(parents=True, exist_ok=True)

    scene = build_scene(snapshot)
    validation = scene.validate()
    result = scene.render()
    unsigned = diagnostics.render_certificate(sign=False)
    unsigned["scene"] = {
        "body": "Moon",
        "crs": LUNAR_CRS,
        "dem_sha256": DEM_SHA256,
        "signing_profile": "development",
    }
    signed = certificate.sign_certificate(unsigned)
    certificate.write_certificate(signed, certificate_path)
    return {
        "validation_status": validation.status,
        "render_status": result.status,
        "render_backend": scene.last_render_backend,
        "snapshot": str(scene.last_render_path),
        "certificate": str(certificate_path),
        "certificate_body": signed["scene"]["body"],
        "certificate_payload_sha256": certificate.payload_sha256(signed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "examples" / "out" / "moon_south_pole" / "moon_south_pole.png",
    )
    parser.add_argument("--certificate", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    certificate_path = args.certificate or args.snapshot.with_suffix(".certificate.json")
    payload = run_example(args.snapshot, certificate_path)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("render:", payload["render_status"], payload["snapshot"])
        print("backend:", payload["render_backend"])
        print("certificate:", payload["certificate"], payload["certificate_body"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
