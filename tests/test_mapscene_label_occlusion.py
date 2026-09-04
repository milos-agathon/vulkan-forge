from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
from _terrain_runtime import terrain_rendering_available


_DEPTH_CONVENTION = "normalized_device_depth"
_DEPTH_DOMAIN = [0.0, 1.0]


def test_mapscene_label_terrain_occlusion_records_rejection() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.full((4, 4), 20.0, dtype=np.float32),
            crs="EPSG:3857",
            metadata={"source_id": "ridge-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "draped",
                        "text": "Draped",
                        "geometry": {"type": "Point", "coordinates": [10.0, 10.0]},
                    },
                    {
                        "id": "hidden",
                        "text": "Hidden",
                        "geometry": {"type": "Point", "coordinates": [20.0, 20.0, 5.0]},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("DrapedHidden"))},
                occlusion="terrain",
                metadata={"terrain_occlusion_bias": 0.0},
            )
        ],
    )

    report = scene.validate()
    plan = scene.compiled_label_plans["labels"]

    assert report.supported_features["labels.occlusion.terrain"] == "supported"
    assert [label.label_id for label in plan.accepted] == ["draped"]
    assert plan.accepted[0].candidate.anchor[2] == 20.0
    assert [(label.label_id, label.reason) for label in plan.rejected] == [
        ("hidden", "terrain_occluded")
    ]
    assert plan.rejected[0].details["terrain_sample"]["source"] == "mapscene_terrain_heightmap"
    assert plan.rejected[0].details["terrain_sample"]["visible"] is False


def test_label_layer_occlusion_none_disables_terrain_sampling() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.full((4, 4), 20.0, dtype=np.float32),
            metadata={"source_id": "ridge-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "hidden",
                        "text": "Hidden",
                        "geometry": {"type": "Point", "coordinates": [20.0, 20.0, 5.0]},
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("Hidden"))},
                occlusion="none",
            )
        ],
    )

    report = scene.validate()
    plan = scene.compiled_label_plans["labels"]

    assert report.supported_features["labels.occlusion.none"] == "supported"
    assert [label.label_id for label in plan.accepted] == ["hidden"]
    assert plan.rejected == []


def test_mapscene_label_depth_aov_occlusion_rejects_label_behind_scene_depth() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"source_id": "flat-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "front",
                        "text": "Front",
                        "geometry": {"type": "Point", "coordinates": [32.0, 32.0, 0.25]},
                        "projected_anchor": [32.0, 32.0, 0.25],
                        "projected_depth_convention": _DEPTH_CONVENTION,
                        "projected_depth_domain": _DEPTH_DOMAIN,
                    },
                    {
                        "id": "behind",
                        "text": "Behind",
                        "geometry": {"type": "Point", "coordinates": [32.0, 32.0, 0.75]},
                        "projected_anchor": [32.0, 32.0, 0.75],
                        "projected_depth_convention": _DEPTH_CONVENTION,
                        "projected_depth_domain": _DEPTH_DOMAIN,
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("FrontBehind"))},
                occlusion="terrain",
                metadata={
                    "depth_occlusion": {
                        "image": np.full((4, 4), 0.5, dtype=np.float32).tolist(),
                        "source": "unit_test_depth_aov",
                        "bias": 0.0,
                        "depth_convention": _DEPTH_CONVENTION,
                        "depth_domain": _DEPTH_DOMAIN,
                    }
                },
            )
        ],
    )

    report = scene.validate()
    plan = scene.compiled_label_plans["labels"]

    assert report.supported_features["labels.occlusion.depth_aov"] == "supported"
    assert [label.label_id for label in plan.accepted] == ["front"]
    assert [(label.label_id, label.reason) for label in plan.rejected] == [
        ("behind", "terrain_occluded")
    ]
    assert plan.rejected[0].details["terrain_sample"]["source"] == "unit_test_depth_aov"
    assert plan.rejected[0].details["terrain_sample"]["scene_depth"] == 0.5
    assert plan.rejected[0].details["terrain_sample"]["label_depth"] == 0.75


def test_depth_aov_occlusion_tests_projected_depth_without_explicit_z() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"source_id": "flat-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "behind-2d",
                        "text": "Behind",
                        "projected_depth": 0.75,
                        "projected_anchor": [32.0, 32.0, 0.75],
                        "projected_depth_convention": _DEPTH_CONVENTION,
                        "projected_depth_domain": _DEPTH_DOMAIN,
                        "geometry": {"type": "Point", "coordinates": [32.0, 32.0]},
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("Behind"))},
                occlusion="terrain",
                metadata={
                    "depth_occlusion": {
                        "image": np.full((4, 4), 0.5, dtype=np.float32).tolist(),
                        "source": "unit_test_depth_aov",
                        "bias": 0.0,
                        "depth_convention": _DEPTH_CONVENTION,
                        "depth_domain": _DEPTH_DOMAIN,
                    }
                },
            )
        ],
    )

    scene.validate()
    plan = scene.compiled_label_plans["labels"]

    assert plan.accepted == []
    assert [(label.label_id, label.reason) for label in plan.rejected] == [
        ("behind-2d", "terrain_occluded")
    ]
    sample = plan.rejected[0].details["terrain_sample"]
    assert sample["explicit_z"] is False
    assert sample["depth_tested"] is True
    assert sample["label_depth"] == 0.75


def test_curved_label_depth_occlusion_is_documented_unsupported_substitution() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"source_id": "flat-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "curved-ridge",
                        "text": "Ridge",
                        "curved_text": True,
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[8.0, 24.0, 0.2], [32.0, 28.0, 0.2], [56.0, 24.0, 0.2]],
                        },
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("Ridge"))},
                occlusion="terrain",
                metadata={
                    "depth_occlusion": {
                        "image": np.full((4, 4), 0.5, dtype=np.float32).tolist(),
                        "source": "unit_test_depth_aov",
                        "bias": 0.0,
                        "depth_convention": _DEPTH_CONVENTION,
                        "depth_domain": _DEPTH_DOMAIN,
                    }
                },
            )
        ],
    )

    scene.validate()
    plan = scene.compiled_label_plans["labels"]

    assert plan.accepted == []
    assert [(label.label_id, label.reason) for label in plan.rejected] == [
        ("curved-ridge", "missing_geometry_authority")
    ]
    assert plan.rejected[0].details == {"required_authority": "layout_curved_text"}
    assert plan.rejected[0].diagnostic_refs == ("label_geometry_authority_missing",)
    assert any(
        diagnostic.code == "label_geometry_authority_missing"
        and diagnostic.object_id == "curved-ridge"
        for diagnostic in plan.diagnostics
    )


def test_mapscene_render_reads_frozen_compile_phase_label_plan(tmp_path, monkeypatch) -> None:
    """SUTURA: render must draw the frozen compiled plan, never re-plan from a live GPU depth frame."""

    class FakeAovFrame:
        def depth(self):
            # A live GPU depth frame that, if (incorrectly) wired into label
            # occlusion during render, would cull the "behind" label.
            return np.full((4, 4), 0.5, dtype=np.float32)

    def fake_terrain_result(recipe, heightmap):
        rgba = np.zeros((int(recipe.output.height), int(recipe.output.width), 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(
            rgba=rgba,
            aov_frame=FakeAovFrame(),
            hdr_frame=None,
            metadata={"samples_used": 1, "target_samples": 1, "denoiser_used": "none", "adaptive": False},
        )

    observed: dict[str, object] = {}

    def fake_label_composite(base, recipe, plans):
        plan = plans["labels"]
        observed["plan_payload"] = plan.to_dict()
        return base, True

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain_result)
    monkeypatch.setattr(map_scene, "_composite_native_label_layers", fake_label_composite)
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"source_id": "flat-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64, path=str(tmp_path / "depth-label.png")),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "front",
                        "text": "Front",
                        "geometry": {"type": "Point", "coordinates": [32.0, 32.0, 0.25]},
                    },
                    {
                        "id": "behind",
                        "text": "Behind",
                        "geometry": {"type": "Point", "coordinates": [24.0, 48.0, 0.75]},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("FrontBehind"))},
                occlusion="terrain",
            )
        ],
    )

    compiled = scene.compile_plan()
    frozen_payload = compiled.label_plans["labels"].to_dict()

    scene.render()

    # The render phase received exactly the frozen compile-phase plan ...
    assert observed["plan_payload"] == frozen_payload
    # ... and mutated no label state: the plan is unchanged after render.
    assert compiled.label_plans["labels"].to_dict() == frozen_payload
    assert scene.compiled_plan is compiled
    # No label was culled against the live GPU depth frame during render.
    for label in compiled.label_plans["labels"].rejected:
        sample = label.details.get("terrain_sample") or {}
        assert sample.get("source") != "mapscene_depth_aov"


def _real_depth_occlusion_scene(path: Path, *, extra_labels: int = 0) -> f3d.MapScene:
    labels = [
        {
            "id": "front",
            "text": "Front",
            "geometry": {"type": "Point", "coordinates": [32.0, 32.0, 0.0]},
            "projected_anchor": [32.0, 32.0, 0.0],
            "projected_depth_convention": _DEPTH_CONVENTION,
            "projected_depth_domain": _DEPTH_DOMAIN,
        },
        {
            "id": "behind",
            "text": "Behind",
            "geometry": {"type": "Point", "coordinates": [32.0, 32.0, 0.95]},
            "projected_anchor": [32.0, 32.0, 0.95],
            "projected_depth_convention": _DEPTH_CONVENTION,
            "projected_depth_domain": _DEPTH_DOMAIN,
        },
    ]
    # SUTURA: the depth source for occlusion culling is serialized with the
    # recipe (a deterministic compile-phase proxy), never a live GPU frame.
    depth_metadata = {
        "depth_occlusion": {
            "image": np.full((16, 16), 0.5, dtype=np.float32).tolist(),
            "source": "serialized_depth_proxy",
            "bias": 0.0,
            "depth_convention": _DEPTH_CONVENTION,
            "depth_domain": _DEPTH_DOMAIN,
        }
    }
    for index in range(extra_labels):
        labels.append(
            {
                "id": f"dense-{index:02d}",
                "text": f"P{index}",
                "geometry": {"type": "Point", "coordinates": [8.0 + index * 2.0, 12.0 + (index % 5) * 4.0, 0.20]},
                "projected_anchor": [
                    8.0 + index * 2.0,
                    12.0 + (index % 5) * 4.0,
                    0.20,
                ],
                "projected_depth_convention": _DEPTH_CONVENTION,
                "projected_depth_domain": _DEPTH_DOMAIN,
            }
        )
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((16, 16), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"source_id": "flat-dem", "width": 16, "height": 16},
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=120.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=96, height=64, path=str(path)),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=labels,
                glyph_atlas={"glyphs": sorted(set("FrontBehindP0123456789"))},
                occlusion="terrain",
                metadata=depth_metadata,
            )
        ],
        reproducibility_profile=f3d.ReproducibilityProfile(seed=7),
    )


def test_real_render_compile_phase_depth_occludes_and_releases_declutter_slot(tmp_path) -> None:
    if not terrain_rendering_available():
        pytest.skip("real compile-phase label occlusion requires a terrain-capable GPU runtime")

    scene = _real_depth_occlusion_scene(tmp_path / "occlusion.png")
    compiled = scene.compile_plan()
    plan = compiled.label_plans["labels"]

    # Depth occlusion is resolved at compile time from the serialized proxy:
    # "behind" is culled, which releases its declutter slot for "front".
    assert [label.label_id for label in plan.accepted] == ["front"]
    assert [(label.label_id, label.reason) for label in plan.rejected] == [
        ("behind", "terrain_occluded")
    ]
    assert plan.rejected[0].details["terrain_sample"]["source"] == "serialized_depth_proxy"

    frozen_payload = plan.to_dict()
    scene.render()

    # The render phase drew the frozen plan without mutating it.
    assert scene.last_render_backend == "gpu_terrain"
    assert scene.compiled_plan is compiled
    assert compiled.label_plans["labels"].to_dict() == frozen_payload


def test_real_render_depth_occlusion_is_deterministic_for_dense_points(tmp_path) -> None:
    if not terrain_rendering_available():
        pytest.skip("real depth-AOV label occlusion requires a terrain-capable GPU runtime")

    first = _real_depth_occlusion_scene(tmp_path / "dense-a.png", extra_labels=8)
    second = _real_depth_occlusion_scene(tmp_path / "dense-b.png", extra_labels=8)

    first.render()
    second.render()

    assert first.compiled_label_plans["labels"].to_dict() == second.compiled_label_plans["labels"].to_dict()
    assert (tmp_path / "dense-a.png").read_bytes() == (tmp_path / "dense-b.png").read_bytes()
