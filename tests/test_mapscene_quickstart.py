import importlib.util
import sys
import json
import re
from pathlib import Path

import forge3d as f3d
import pytest

from _terrain_runtime import terrain_rendering_available

from _terrain_runtime import (
    terrain_rendering_available as _terrain_rendering_available,
)

_requires_gpu = pytest.mark.skipif(
    not _terrain_rendering_available(),
    reason=(
        "MapScene native terrain render requires a terrain-safe hardware "
        "adapter; hosted runners (GPU-less, WARP, or guarded paravirtual "
        "Metal) either block with a native-render diagnostic or raise by "
        "design (SUTURA), so these render-asserting tests skip honestly"
    ),
)



QUICKSTART = Path("docs/guides/offline_3d_map_rendering.md")
START_QUICKSTART = Path("docs/start/quickstart.md")
VECTOR_EXAMPLE = Path("examples/mapscene_vector_labels.py")
BUILDING_EXAMPLE = Path("examples/mapscene_buildings_labels.py")


def _load_example(path: Path):
    examples_dir = str(path.parent.resolve())
    added_examples_dir = examples_dir not in sys.path
    if added_examples_dir:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def test_mapscene_quickstart_points_to_canonical_examples():
    text = QUICKSTART.read_text(encoding="utf-8")

    assert "examples/mapscene_terrain_raster.py" in text
    assert "examples/mapscene_vector_labels.py" in text
    assert "examples/mapscene_buildings_labels.py" in text
    assert "viewer_ipc" not in text
    assert "raw IPC" not in text


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_quickstart_vector_labels_scenario_is_executable(tmp_path):
    module = _load_example(VECTOR_EXAMPLE)

    first = module.run_example(tmp_path / "first")
    second = module.run_example(tmp_path / "second")

    assert first["validation_status"] == "warning"
    assert first["render_status"] == "warning"
    assert first["render_backend"] in {"gpu_terrain", "placeholder"}
    assert first["accepted_label_ids"] == second["accepted_label_ids"]
    assert first["rejected_label_reasons"] == second["rejected_label_reasons"]
    assert Path(first["png_path"]).exists()


@pytest.mark.apple_metal_physical
def test_start_quickstart_mapscene_snippet_executes(tmp_path, monkeypatch):
    if not terrain_rendering_available():
        pytest.skip("docs/start MapScene snippet requires a terrain-capable GPU runtime")

    text = START_QUICKSTART.read_text(encoding="utf-8")
    section = text.split("## First MapScene Render", 1)[1].split("## First viewer session", 1)[0]
    match = re.search(r"```python\n(.*?)\n```", section, flags=re.S)
    assert match is not None
    snippet = match.group(1)
    assert "allow_placeholder=True" not in snippet

    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    exec(compile(snippet, str(START_QUICKSTART), "exec"), namespace)

    scene = namespace["scene"]
    manifest = namespace["manifest"]
    assert isinstance(scene, f3d.MapScene)
    assert scene.last_render_backend == "gpu_terrain"
    assert Path("mapscene.png").exists()
    assert manifest["kind"] == "mapscene_recipe_manifest"


def test_quickstart_negative_diagnostics_are_available_before_render():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "asset_status": "fixture"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
        layers=[
            f3d.RasterOverlay(
                layer_id="wrong-crs",
                path="fixtures/wgs84.tif",
                crs="EPSG:4326",
                metadata={"asset_status": "fixture"},
            ),
            f3d.VectorOverlay(
                layer_id="bad-style",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                    }
                ],
                style={"version": 8, "layers": [{"id": "heat", "type": "heatmap"}]},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "cafe",
                        "text": "cafe!",
                        "geometry": {"type": "Point", "coordinates": (16.0, 16.0, 0.0)},
                    }
                ],
                glyph_atlas={"glyphs": list("cafe")},
            ),
        ],
    )

    report = scene.validate()
    codes = [diagnostic.code for diagnostic in report.diagnostics]

    assert "crs_mismatch" in codes
    assert "unsupported_style_layer_type" in codes
    assert "missing_glyphs" in codes


def test_quickstart_models_accept_path_objects_in_serialized_recipes(tmp_path):
    terrain_path = tmp_path / "terrain.tif"
    raster_path = tmp_path / "overlay.tif"
    vector_path = tmp_path / "roads.geojson"
    point_path = tmp_path / "points.laz"
    output_path = tmp_path / "scene.png"

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=terrain_path,
            crs="EPSG:32610",
            metadata={"asset_status": "fixture"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png", path=output_path),
        layers=[
            f3d.RasterOverlay(
                layer_id="raster",
                path=raster_path,
                crs="EPSG:32610",
                metadata={"asset_status": "fixture"},
            ),
            f3d.VectorOverlay(
                layer_id="roads",
                path=vector_path,
                crs="EPSG:32610",
                metadata={"asset_status": "fixture"},
            ),
            f3d.PointCloudLayer(
                layer_id="points",
                path=point_path,
                crs="EPSG:32610",
                metadata={"asset_status": "fixture"},
            ),
        ],
    )

    payload = scene.to_dict()

    recipe = payload["recipe"]

    assert recipe["terrain"]["path"] == str(terrain_path)
    assert recipe["output"]["path"] == str(output_path)
    assert [layer["path"] for layer in recipe["layers"]] == [
        str(raster_path),
        str(vector_path),
        str(point_path),
    ]

    report = scene.validate()
    terrain_summary = next(summary for summary in report.layer_summaries if summary.layer_id == "terrain")
    raster_summary = next(summary for summary in report.layer_summaries if summary.layer_id == "raster")
    vector_summary = next(summary for summary in report.layer_summaries if summary.layer_id == "roads")
    point_summary = next(summary for summary in report.layer_summaries if summary.layer_id == "points")
    assert terrain_summary.details["path"] == str(terrain_path)
    assert raster_summary.details["path"] == str(raster_path)
    assert vector_summary.details["path"] == str(vector_path)
    assert point_summary.details["path"] == str(point_path)
    json.dumps(payload)


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_quickstart_building_scenario_renders_native_gpu_buildings(tmp_path):
    module = _load_example(BUILDING_EXAMPLE)

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "ok"
    assert payload["render_status"] == "ok", payload.get("diagnostics") or payload
    assert payload["render_backend"] == "gpu_terrain"
    assert payload["bundle_status"] == "ok"
    assert payload["building_backend"] == "terrain_scatter_instanced_mesh"
    assert payload["building_batch_count"] == 4
    assert payload["building_shadow_model"] == "terrain_csm_mesh_cast_receive"
    assert "pro_gated_path" not in payload["diagnostic_codes"]
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()
