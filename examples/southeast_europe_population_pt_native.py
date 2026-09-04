#!/usr/bin/env python3
"""Render the hermetic Southeast-Europe population fixture with native PT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from _import_shim import ensure_repo_import

ensure_repo_import()

from forge3d.path_tracing import render_terrain_poster
import obliqua_se_europe_fixture as fixture

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples" / "out" / "southeast_europe_population_pt_native.png")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--tile", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dem = fixture.make_dem()
    population = fixture.make_population()
    material = fixture.make_albedo_map(population)
    render_args = fixture.make_gpu_lightfield_args(dem)
    certificate = args.output.with_suffix(".certificate.json")
    result = render_terrain_poster(
        dem,
        args.width,
        args.height,
        tile=args.tile,
        albedo_map=material,
        albedo_sampling="bilinear",
        certificate=certificate,
        **render_args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(result["rgba"], dtype=np.uint8), mode="RGBA").save(args.output)

    artifact_root = Path(os.environ.get("FORGE3D_OBLIQUA_ARTIFACT_DIR", args.output.parent))
    artifact_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        "output": str(args.output),
        "certificate": str(certificate),
        "certificate_digest": result["certificate_digest"],
        "fixture_input_hashes": fixture.fixture_input_hashes(),
        "tiles": result["tiles"],
    }
    (artifact_root / "southeast_europe_population_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved poster: {args.output}")
    print(f"Saved metrics: {artifact_root / 'southeast_europe_population_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
