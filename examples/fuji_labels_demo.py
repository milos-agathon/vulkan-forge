#!/usr/bin/env python3
"""Mount Fuji labels rendered through the public MapScene API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))

import forge3d as f3d


LABELS = (
    ("summit", "Mount Fuji", 48.0, 24.0, 0.34),
    ("yoshida", "Yoshida Trail", 30.0, 42.0, 0.12),
    ("subashiri", "Subashiri", 66.0, 38.0, 0.10),
    ("kawaguchi", "Lake Kawaguchi", 70.0, 65.0, 0.04),
)


def _heightmap(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[-1.0:1.0 : complex(size), -1.0:1.0 : complex(size)]
    cone = np.exp(-(x * x + y * y) * 3.2)
    ridge = 0.16 * np.exp(-((x + 0.35) ** 2 * 18.0 + (y - 0.10) ** 2 * 7.0))
    slope = 0.10 * (1.0 - y)
    data = cone + ridge + slope
    data -= data.min()
    data /= max(float(data.max()), 1.0e-6)
    return data.astype(np.float32)


def build_scene(output_dir: str | Path) -> f3d.MapScene:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        {
            "id": label_id,
            "kind": "point",
            "text": text,
            "geometry": {"type": "Point", "coordinates": (x, y, z)},
            "typography": {
                "color": [1.0, 1.0, 1.0, 1.0],
                "halo_color": [0.02, 0.02, 0.02, 0.88],
                "halo_width_px": 2.0,
            },
        }
        for label_id, text, x, y, z in LABELS
    ]
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=_heightmap(),
            crs="EPSG:32654",
            metadata={"width": 64, "height": 64, "source_id": "synthetic-fuji-dem"},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=850.0, azimuth_deg=38.0, elevation_deg=34.0),
        lighting=f3d.LightingPreset(name="rainier_showcase", intensity=1.15),
        output=f3d.OutputSpec(width=128, height=88, format="png", path=str(output_dir / "fuji_labels.png")),
        layers=[
            f3d.LabelLayer(
                layer_id="fuji.labels",
                labels=labels,
                atlas=f3d.FontAtlas.default_latin(),
                occlusion="none",
                metadata={"source_id": "fuji-labels"},
            )
        ],
        reproducibility_profile=f3d.ReproducibilityProfile(seed=20260703),
    )


def run_example(output_dir: str | Path) -> dict[str, object]:
    scene = build_scene(output_dir)
    validation = scene.validate()
    render_report = scene.render()
    plan = scene.compiled_label_plans["fuji.labels"]
    return {
        "validation_status": validation.status,
        "render_status": render_report.status,
        "render_backend": scene.last_render_backend,
        "png_path": scene.last_render_path,
        "accepted_label_ids": [label.label_id for label in plan.accepted],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("examples/out/fuji_labels/fuji_labels.png"))
    args = parser.parse_args()

    payload = run_example(args.snapshot.parent)
    rendered = Path(str(payload["png_path"]))
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    if rendered != args.snapshot:
        args.snapshot.write_bytes(rendered.read_bytes())
    print(args.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
