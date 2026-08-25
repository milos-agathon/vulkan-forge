"""P1 MapScene asset and bundle showcase using forge3d datasets.

This example exercises the public Spec 005 surface: data-driven label
ingestion, typography coverage, building and 3D Tiles layer validation,
structured diagnostics, and deterministic bundle round-trip state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))

import forge3d as f3d
from forge3d.datasets import fetch_cityjson, mini_dem_path, sample_boundaries
from forge3d.label_plan import KeepoutRegion, PriorityClass


def _features() -> list[dict[str, Any]]:
    payload = sample_boundaries()
    features: list[dict[str, Any]] = []
    for index, feature in enumerate(payload["features"]):
        item = dict(feature)
        item["id"] = f"p1-boundary-{index}"
        features.append(item)
    return features


def _centroid_point_feature(feature: Mapping[str, Any]) -> dict[str, Any]:
    ring = feature["geometry"]["coordinates"][0]
    x = sum(point[0] for point in ring[:-1]) / max(1, len(ring) - 1)
    y = sum(point[1] for point in ring[:-1]) / max(1, len(ring) - 1)
    return {
        "type": "Feature",
        "id": str(feature["id"]),
        "properties": dict(feature["properties"]),
        "geometry": {"type": "Point", "coordinates": (x * 256.0, (1.0 - y) * 192.0, 0.0)},
    }


def _write_demo_tileset(output_dir: Path) -> Path:
    tileset_dir = output_dir / "p1_tileset"
    tileset_dir.mkdir(parents=True, exist_ok=True)
    tileset_path = tileset_dir / "tileset.json"
    payload = {
        "asset": {"version": "1.0"},
        "geometricError": 0.0,
        "root": {
            "boundingVolume": {"region": [0.0, 0.0, 0.01, 0.01, 0.0, 20.0]},
            "geometricError": 0.0,
            "refine": "ADD",
        },
    }
    tileset_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return tileset_path


def _glyphs_for(labels: f3d.LabelLayer) -> list[str]:
    return sorted(set("".join(str(label["text"]) for label in labels.labels or ())))


def build_scene(output_dir: str | Path) -> f3d.MapScene:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _features()
    label_features = [_centroid_point_feature(feature) for feature in features]
    labels = f3d.LabelLayer.from_features(
        label_features,
        text=["concat", ["upcase", ["get", "name"]], " asset"],
        crs="EPSG:3857",
        target_crs="EPSG:3857",
        terrain_sampling="required",
        terrain_sampler=lambda x, y: 12.5 + float(x) * 0.01 + float(y) * 0.01,
        occlusion="none",
        layer_id="p1_boundary_labels",
        typography=f3d.TypographySettings(
            font_size=14,
            tracking=0.5,
            line_height=24.0,
            multiline=True,
            callout=True,
            callout_offset=(8.0, -6.0),
        ).to_dict(),
        metadata={"source_id": "forge3d.datasets.sample_boundaries.labels", "seed": 505},
    )
    labels.glyph_atlas = f3d.FontAtlas.default_latin(
        fallbacks=[f3d.FontFallbackRange("latin-extended", 128, 255, "Default Latin fallback")]
    ).to_dict()
    labels.priority_rules = [PriorityClass("asset-labels", rank=10)]
    labels.labels = [
        {**label, "priority_class": "asset-labels"}
        for label in (labels.labels or ())
    ]
    labels.labels = [
        *labels.labels,
        {
            "id": "p1-title-collision",
            "kind": "point",
            "text": "P1 Bundle",
            "geometry": {"type": "Point", "coordinates": (60.0, 24.0, 0.0)},
            "priority_class": "asset-labels",
        },
    ]
    labels.glyph_atlas["glyphs"] = sorted(set(labels.glyph_atlas["glyphs"]) | set(_glyphs_for(labels)))

    tileset_path = _write_demo_tileset(output_dir)
    building_path = fetch_cityjson("sample-buildings")

    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=str(mini_dem_path()),
            crs="EPSG:3857",
            metadata={
                "width": 256,
                "height": 256,
                "source_id": "forge3d.datasets.mini_dem",
            },
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.5, 0.5, 0.0), distance=900.0, azimuth_deg=28.0, elevation_deg=50.0),
        lighting=f3d.LightingPreset(name="daylight", intensity=1.1),
        output=f3d.OutputSpec(
            width=256,
            height=192,
            format="png",
            path=str(output_dir / "mapscene_p1_assets_bundle_showcase.png"),
        ),
        map_furniture=f3d.MapFurnitureLayer(
            title="Spec 005 P1 assets",
            keepouts=[KeepoutRegion("title", "title", (0, 0, 112, 48), priority=100)],
            legend={"items": ["sample_boundaries labels", "sample-buildings CityJSON", "synthetic tileset"]},
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(
            seed=1505,
            asset_hashes_or_ids={
                "terrain": "forge3d.datasets.mini_dem",
                "labels": "forge3d.datasets.sample_boundaries",
                "buildings": "forge3d.datasets.sample-buildings",
                "tiles3d": "synthetic.p1.tileset",
            },
        ),
        layers=[
            labels,
            f3d.MapSceneBuildingLayer.from_cityjson(
                building_path,
                layer_id="p1_cityjson_buildings",
                support_level="underdeveloped",
                geometry_count=2,
                bounds=[0.0, 0.0, 0.0, 24.0, 18.0, 12.0],
                material_status="scalar_pbr_underdeveloped",
                metadata={"source_id": "forge3d.datasets.sample-buildings"},
            ),
            f3d.Tiles3DLayer.from_tileset_json(
                tileset_path,
                layer_id="p1_tileset_review",
                lod={"screen_space_error": 16.0, "mode": "review"},
                cache_budget=8 * 1024 * 1024,
                cache_stats={"entry_count": 1, "cache_used": 0, "cache_budget": 8 * 1024 * 1024},
                metadata={
                    "source_id": "synthetic.p1.tileset",
                    "unsupported_features": ["b3dm_batch_table_textures"],
                },
            ),
        ],
    )


def _codes(report: f3d.ValidationReport) -> list[str]:
    return [diagnostic.code for diagnostic in report.diagnostics]


def run_example(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = build_scene(output_dir)
    validation = scene.validate()
    plan = scene.compiled_label_plans["p1_boundary_labels"]

    render_status = "not_requested"
    try:
        render = scene.render()
        render_status = render.status
    except RuntimeError:
        render_status = "blocked_by_diagnostics"

    bundle_path = output_dir / "mapscene_p1_assets_bundle_showcase.forge3d"
    bundle = scene.save_bundle(bundle_path)
    reloaded = f3d.MapScene.load_bundle(bundle_path)
    reloaded_report = reloaded.validate()

    return {
        "validation_status": validation.status,
        "render_status": render_status,
        "bundle_status": bundle.status,
        "roundtrip_status": reloaded_report.status,
        "dataset_names": ["mini_dem", "sample_boundaries", "sample-buildings"],
        "accepted_label_ids": [label.label_id for label in plan.accepted],
        "rejected_label_reasons": {label.label_id: label.reason for label in plan.rejected},
        "diagnostic_codes": _codes(bundle),
        "roundtrip_diagnostic_codes": _codes(reloaded_report),
        "support_levels": dict(bundle.supported_features),
        "unsupported_features": dict(bundle.unsupported_features),
        "png_path": str(output_dir / "mapscene_p1_assets_bundle_showcase.png"),
        "bundle_path": str(bundle_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Spec 005 P1 MapScene asset bundle showcase.")
    parser.add_argument("--output-dir", default="examples/out/mapscene_p1_assets_bundle_showcase")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_example(args.output_dir)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("datasets:", ", ".join(payload["dataset_names"]))
        print("validation:", payload["validation_status"], payload["diagnostic_codes"])
        print("render:", payload["render_status"], payload["png_path"])
        print("bundle:", payload["bundle_status"], payload["bundle_path"])
        print("roundtrip:", payload["roundtrip_status"], payload["roundtrip_diagnostic_codes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
