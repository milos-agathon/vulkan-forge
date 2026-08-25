import importlib.util
import sys
from pathlib import Path

import pytest

import forge3d.map_scene as map_scene

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



EXAMPLES = {
    "terrain_raster": Path("examples/mapscene_terrain_raster.py"),
    "vector_labels": Path("examples/mapscene_vector_labels.py"),
    "offline_quality": Path("examples/mapscene_offline_quality.py"),
    "buildings_labels": Path("examples/mapscene_buildings_labels.py"),
    "bundled_datasets_showcase": Path("examples/mapscene_bundled_datasets_showcase.py"),
    "p1_assets_bundle_showcase": Path("examples/mapscene_p1_assets_bundle_showcase.py"),
    "fuji_labels": Path("examples/fuji_labels_demo.py"),
}


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


def _supported_render_backend(payload):
    assert payload["render_backend"] in {"gpu_terrain", "placeholder"}


def test_canonical_mapscene_examples_exist_and_do_not_use_raw_ipc():
    for path in EXAMPLES.values():
        assert path.exists(), f"missing canonical example: {path}"
        text = path.read_text(encoding="utf-8")
        assert "MapScene" in text
        assert "viewer_ipc" not in text
        assert "send_ipc" not in text


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_terrain_raster_example_validates_renders_and_bundles(tmp_path, monkeypatch):
    module = _load_example(EXAMPLES["terrain_raster"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "ok"
    assert payload["render_status"] == "ok", payload.get("diagnostics") or payload
    assert payload["render_backend"] == "gpu_terrain"
    assert payload["bundle_status"] == "ok"
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()
    assert "placeholder_fallback" not in payload["diagnostic_codes"]

    # When the native offscreen backend is unavailable the render blocks with
    # a structured diagnostic instead of writing a CPU placeholder.
    monkeypatch.setattr(map_scene, "_render_native_offscreen_result", lambda _recipe, _compiled, **_kwargs: None)
    blocked_scene = module.build_scene(tmp_path / "blocked")
    with pytest.raises(map_scene.MapSceneNativeUnavailable) as excinfo:
        blocked_scene.render()
    assert excinfo.value.diagnostic["status"] == "diagnostic_block"
    assert blocked_scene.last_render_path is None


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_vector_labels_example_compiles_label_plan_and_bundles(tmp_path):
    module = _load_example(EXAMPLES["vector_labels"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "warning"
    assert payload["render_status"] == "warning", payload.get("diagnostics") or payload
    _supported_render_backend(payload)
    assert payload["bundle_status"] == "warning"
    assert payload["accepted_label_ids"] == ["city", "park"]
    assert payload["rejected_label_reasons"] == {"blocked-title": "keepout_region"}
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_fuji_labels_demo_uses_public_mapscene_gpu_label_path(tmp_path):
    module = _load_example(EXAMPLES["fuji_labels"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "ok"
    assert payload["render_status"] == "ok", payload.get("diagnostics") or payload
    assert payload["render_backend"] == "gpu_terrain"
    assert payload["accepted_label_ids"]
    assert Path(payload["png_path"]).exists()


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_offline_quality_example_uses_native_accumulation_and_aovs(tmp_path):
    module = _load_example(EXAMPLES["offline_quality"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "ok"
    assert payload["render_status"] == "ok", payload.get("diagnostics") or payload
    assert payload["render_backend"] == "gpu_terrain"
    assert payload["bundle_status"] == "ok"
    assert payload["samples_used"] == 4
    assert payload["denoiser_used"] == "atrous"
    assert payload["aa_seed"] == 20260703
    assert set(payload["aov_paths"]) == {"albedo", "normal", "depth"}
    assert Path(payload["png_path"]).exists()
    assert Path(payload["hdr_path"]).exists()
    for path in payload["aov_paths"].values():
        assert Path(path).exists()
    assert Path(payload["bundle_path"]).exists()


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_buildings_labels_example_renders_native_gpu_buildings(tmp_path):
    module = _load_example(EXAMPLES["buildings_labels"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "ok"
    assert payload["render_status"] == "ok", payload.get("diagnostics") or payload
    assert payload["render_backend"] == "gpu_terrain"
    assert payload["bundle_status"] == "ok"
    assert "pro_gated_path" not in payload["diagnostic_codes"]
    assert payload["building_backend"] == "terrain_scatter_instanced_mesh"
    assert payload["building_batch_count"] == 4
    assert payload["building_shadow_model"] == "terrain_csm_mesh_cast_receive"
    assert set(payload["building_batch_ids"]) == {
        "building-flat",
        "building-gabled",
        "building-hipped",
        "building-pyramidal",
    }
    assert set(payload["building_roof_types"].values()) == {"flat", "gabled", "hipped", "pyramidal"}
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_bundled_datasets_showcase_covers_specs_001_to_004(tmp_path):
    module = _load_example(EXAMPLES["bundled_datasets_showcase"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "warning"
    assert payload["render_status"] == "warning", payload.get("diagnostics") or payload
    assert payload["bundle_status"] == "warning"
    assert payload["dataset_names"] == ["mini_dem", "sample_boundaries"]
    assert payload["accepted_label_ids"] == [
        "label-central-basin",
        "label-north-reserve",
        "label-south-ridge",
    ]
    assert payload["diagnostic_codes"] == ["label_rejection_summary"]
    assert payload["support_levels"]["mapscene.validation"] == "supported"
    assert payload["support_levels"]["mapscene.render_png"] == "supported"
    assert payload["support_levels"]["mapscene.save_bundle"] == "supported"
    assert Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_p1_assets_bundle_showcase_covers_spec_005(tmp_path):
    module = _load_example(EXAMPLES["p1_assets_bundle_showcase"])

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "error"
    assert payload["render_status"] == "blocked_by_diagnostics"
    assert payload["bundle_status"] == "error"
    assert payload["roundtrip_status"] == "error"
    assert payload["dataset_names"] == ["mini_dem", "sample_boundaries", "sample-buildings"]
    assert payload["accepted_label_ids"] == [
        "p1-boundary-0",
        "p1-boundary-1",
        "p1-boundary-2",
    ]
    assert payload["rejected_label_reasons"] == {"p1-title-collision": "keepout_region"}
    assert "label_rejection_summary" in payload["diagnostic_codes"]
    assert "unsupported_tile_feature" in payload["diagnostic_codes"]
    assert payload["support_levels"]["mapscene.save_bundle"] == "supported"
    assert payload["support_levels"]["layer.tiles3d_intent"] == "underdeveloped"
    assert payload["unsupported_features"]["tiles3d.feature"] == "unsupported"
    assert not Path(payload["png_path"]).exists()
    assert Path(payload["bundle_path"]).exists()
