from __future__ import annotations

import hashlib

import numpy as np
import pytest

import forge3d as f3d
from forge3d import label_plan as lp
from forge3d._map_scene_labels import _DepthOcclusionSampler, _TerrainOcclusionSampler


_NDC_CONVENTION = "normalized_device_depth"
_NDC_DOMAIN = (0.0, 1.0)


def _depth_sampler(depth_image, *, viewport_size, **kwargs):
    return _DepthOcclusionSampler(
        depth_image,
        viewport_size=viewport_size,
        depth_convention=_NDC_CONVENTION,
        depth_domain=_NDC_DOMAIN,
        **kwargs,
    )


class _Rationale:
    def __init__(self, records):
        self._records = records

    def records(self):
        return list(self._records)


class _HorizontalRidge:
    def sample_label(self, coords, *, record, label_id):
        del record, label_id
        scene_depth = 6.0 if float(coords[1]) >= 45.0 else 10.0
        return {
            "scene_depth": scene_depth,
            "label_depth": float(coords[2]),
            "visible": float(coords[2]) <= scene_depth,
            "source": "test_ridge",
            "depth_authority": "pre_supplied_authoritative",
            "depth_convention": "linear_eye_depth",
            "depth_domain": [0.0, 10.0],
        }


def test_compile_bridges_every_candidate_and_reconstructs_native_choice(monkeypatch):
    observed = {}

    def solver(candidates, **kwargs):
        observed["candidates"] = list(candidates)
        observed["kwargs"] = kwargs
        return (
            [(0, 1)],
            0.0,
            _Rationale(
                [
                    {"kind": "placed", "label_id": 0, "candidate_index": 1, "weight": 4.999},
                    {
                        "kind": "solver",
                        "nodes_explored": 1,
                        "certified": True,
                        "gap": 0.0,
                        "gap_tolerance": 0.02,
                    },
                ]
            ),
        )

    monkeypatch.setattr(lp, "_native_declutter_optimal", lambda: solver)
    plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "ridge",
                "text": "Ridge",
                "geometry": {"type": "Point", "coordinates": [50.0, 50.0, 8.0]},
                "projected_anchor": [50.0, 50.0, 8.0],
                "projected_depth_convention": "linear_eye_depth",
                "projected_depth_domain": [0.0, 10.0],
                "priority": 5,
                "requires_terrain": True,
                "candidate_policy": {"radial_count": 0},
            }
        ],
        camera={},
        viewport=(100, 100),
        terrain=_HorizontalRidge(),
    )

    bridged = observed["candidates"]
    assert [(label, candidate) for label, candidate, *_rest in bridged] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    ]
    assert all(x1 > x0 and y1 > y0 for _, _, (x0, y0, x1, y1), _, _ in bridged)
    assert [visible for *_prefix, visible in bridged] == [False, True, False, False, False]
    assert plan.accepted[0].candidate.candidate_id == "ridge:above"
    assert plan.accepted[0].screen_bounds == plan.accepted[0].candidate.bounds
    placed = next(record for record in plan.rationale if record["kind"] == "placed")
    assert placed["candidate_id"] == "ridge:above"


def test_curved_geometry_authority_is_consumed_without_relayout(monkeypatch):
    captured = {}

    def solver(candidates, **_kwargs):
        captured["candidates"] = list(candidates)
        return (
            [(0, 1)],
            0.0,
            _Rationale(
                [
                    {"kind": "placed", "label_id": 0, "candidate_index": 1, "weight": 6.0},
                    {"kind": "solver", "nodes_explored": 1, "certified": True, "gap": 0.0},
                ]
            ),
        )

    monkeypatch.setattr(lp, "_native_declutter_optimal", lambda: solver)
    glyph_geometry = [
        {"font_index": 0, "glyph_id": 7, "origin": [2.0, 3.0], "rotation": 0.4}
    ]
    selected_geometry = [
        {"font_index": 0, "glyph_id": 9, "origin": [4.0, 6.0], "rotation": 0.7}
    ]
    plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "river",
                "text": "River",
                "curved_text": True,
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [20, 10]]},
                "geometry_authority": {
                    "source": "layout_curved_text",
                    "positioned_glyphs": glyph_geometry,
                    "candidates": [
                        {
                            "candidate_id": "river:a",
                            "anchor": [8, 5, 0.4],
                            "bounds": [2, 1, 14, 9],
                            "positioned_glyphs": glyph_geometry,
                        },
                        {
                            "candidate_id": "river:b",
                            "anchor": [16, 7, 0.4],
                            "bounds": [10, 3, 22, 11],
                            "positioned_glyphs": selected_geometry,
                        },
                    ],
                },
            }
        ],
        camera={},
        viewport=(100, 100),
    )

    assert len(captured["candidates"]) == 2
    assert plan.accepted[0].candidate.candidate_id == "river:b"
    assert list(plan.accepted[0].positioned_glyphs) == selected_geometry
    assert not any(d.code == "experimental_feature" for d in plan.diagnostics)


def test_projected_anchor_drives_compile_time_depth_and_canonical_hash(monkeypatch):
    monkeypatch.setattr(lp, "_native_declutter_optimal", lambda: None)

    class Camera:
        projection_authority = "deterministic"

        def project(self, world, *, viewport):
            assert tuple(world) == (1000.0, 2000.0, 3000.0)
            assert viewport == (64, 64)
            return (20.0, 30.0, 0.75)

    depth = _depth_sampler(
        np.full((4, 4), 0.5, dtype=np.float32),
        viewport_size=(64, 64),
        source="serialized_test_depth",
    )
    try:
        plan = lp.LabelPlan.compile(
            labels=[
                {
                    "id": "projected",
                    "text": "Projected",
                    "geometry": {"type": "Point", "coordinates": [1000, 2000, 3000]},
                    "requires_terrain": True,
                    "projected_depth_convention": _NDC_CONVENTION,
                    "projected_depth_domain": list(_NDC_DOMAIN),
                    "candidate_policy": {"radial_count": 0},
                }
            ],
            camera=Camera(),
            viewport=(64, 64),
            terrain=depth,
            declutter="greedy",
        )
    finally:
        depth.close()

    assert not plan.accepted
    sample = plan.rejected[0].details["terrain_sample"]
    assert sample["label_depth"] == 0.75
    assert sample["depth_authority"] == "pre_supplied_authoritative"
    first = plan.canonical_bytes()
    assert first == lp.LabelPlan.from_dict(plan.to_dict()).canonical_bytes()
    assert plan.plan_hash() == hashlib.sha256(first).hexdigest()


def test_depth_allocation_uses_memory_tracker_and_proxy_is_distinct(monkeypatch):
    del monkeypatch
    from forge3d import _forge3d as native

    baseline = dict(native.global_memory_metrics())
    sampler = _depth_sampler([[0.25, 0.5], [0.75, 1.0]], viewport_size=(2, 2))
    first = dict(native.global_memory_metrics())
    concurrent = _depth_sampler(
        [[0.25, 0.5], [0.75, 1.0]], viewport_size=(2, 2)
    )
    second = dict(native.global_memory_metrics())
    assert first["host_visible_bytes"] == baseline["host_visible_bytes"] + 16
    assert second["host_visible_bytes"] == baseline["host_visible_bytes"] + 32

    real_reservation = sampler._native_reservation

    class _CloseOrderProbe:
        def close(self):
            assert sampler._closed is True
            assert sampler.depth_image is None
            return real_reservation.close()

    sampler._native_reservation = _CloseOrderProbe()
    assert sampler.close() is True
    assert sampler.close() is False
    assert sampler.depth_image is None
    with pytest.raises(RuntimeError, match="closed"):
        sampler.sample_label([0.0, 0.0, 0.5], record={}, label_id="closed")
    after_first = dict(native.global_memory_metrics())
    assert after_first["host_visible_bytes"] == baseline["host_visible_bytes"] + 16
    assert concurrent.close() is True
    restored = dict(native.global_memory_metrics())
    assert restored["host_visible_bytes"] == baseline["host_visible_bytes"]

    proxy = _TerrainOcclusionSampler(
        np.zeros((2, 2), dtype=np.float32), viewport_size=(2, 2)
    )
    sample = proxy.sample_label([0, 0, 1], record={"geometry": {"coordinates": [0, 0, 1]}}, label_id="x")
    assert sample["depth_authority"] == "deterministic_terrain_proxy"

    class OversizedDepth:
        shape = (1, int(baseline["limit_bytes"] // 4) + 1)

    with pytest.raises(Exception, match="budget|Budget"):
        _depth_sampler(OversizedDepth(), viewport_size=(2, 2))


def test_duplicate_public_ids_reconstruct_by_internal_solver_identity(monkeypatch):
    def solver(_candidates, **_kwargs):
        return (
            [(0, 0), (1, 1)],
            0.0,
            _Rationale(
                [
                    {"kind": "placed", "label_id": 0, "candidate_index": 0, "weight": 4.0},
                    {"kind": "placed", "label_id": 1, "candidate_index": 1, "weight": 3.0},
                    {"kind": "solver", "nodes_explored": 2, "certified": True, "gap": 0.0},
                ]
            ),
        )

    monkeypatch.setattr(lp, "_native_declutter_optimal", lambda: solver)
    plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "duplicate",
                "source_id": "source-a",
                "text": "A",
                "geometry": {"type": "Point", "coordinates": [20, 20, 0]},
                "candidate_policy": {"radial_count": 0},
            },
            {
                "id": "duplicate",
                "source_id": "source-b",
                "text": "B",
                "geometry": {"type": "Point", "coordinates": [70, 70, 0]},
                "candidate_policy": {"radial_count": 0},
            },
        ],
        camera={},
        viewport=(100, 100),
    )
    assert [(item.source_id, item.candidate.candidate_type) for item in plan.accepted] == [
        ("source-a", "center"),
        ("source-b", "above"),
    ]
    placed = [record for record in plan.rationale if record["kind"] == "placed"]
    assert [(record["source_id"], record["solver_label_index"]) for record in placed] == [
        ("source-a", 0),
        ("source-b", 1),
    ]


def test_viewport_and_keepout_are_candidate_gates(monkeypatch):
    observed = {}

    def solver(candidates, **_kwargs):
        observed["candidates"] = list(candidates)
        return (
            [(0, 1)],
            0.0,
            _Rationale(
                [
                    {"kind": "placed", "label_id": 0, "candidate_index": 1, "weight": 5.0},
                    {"kind": "solver", "nodes_explored": 1, "certified": True, "gap": 0.0},
                ]
            ),
        )

    monkeypatch.setattr(lp, "_native_declutter_optimal", lambda: solver)
    plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "furniture",
                "source_id": "poi-source",
                "text": "POI",
                "geometry": {"type": "Point", "coordinates": [50, 50, 0]},
                "screen_bounds": [45, 45, 55, 55],
                "candidate_policy": {"offset_px": 20, "radial_count": 0},
                "priority": 5,
            }
        ],
        camera={},
        viewport=(100, 100),
        keepouts=[
            {
                "region_id": "legend",
                "kind": "legend",
                "bounds": [44, 44, 56, 56],
            }
        ],
    )
    bridged = observed["candidates"]
    assert bridged[0][-1] is False
    assert bridged[1][-1] is True
    assert plan.accepted[0].candidate.candidate_id == "furniture:above"
    gate = next(
        record
        for record in plan.rationale
        if record.get("kind") == "candidate_ineligible" and record.get("gate") == "keepout"
    )
    assert gate["candidate_id"] == "furniture:center"
    assert "legend" in "\n".join(plan.render_rationale())

    observed.clear()
    viewport_plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "edge",
                "text": "E",
                "geometry": {"type": "Point", "coordinates": [5, 50, 0]},
                "screen_bounds": [0, 45, 10, 55],
                "candidate_policy": {"offset_px": 12, "radial_count": 0},
                "priority": 5,
            }
        ],
        camera={},
        viewport=(100, 100),
    )
    left = next(item for item in observed["candidates"] if item[1] == 3)
    assert left[-1] is False
    assert viewport_plan.accepted[0].candidate.candidate_id == "edge:above"
    assert any(
        record.get("kind") == "candidate_ineligible"
        and record.get("gate") == "viewport"
        and record.get("candidate_id") == "edge:left"
        for record in viewport_plan.rationale
    )


def test_authoritative_depth_fails_closed_without_projection_and_validates_image():
    depth = _depth_sampler([[0.5]], viewport_size=(100, 100))
    try:
        plan = lp.LabelPlan.compile(
            labels=[
                {
                    "id": "world-only",
                    "text": "World",
                    "geometry": {"type": "Point", "coordinates": [20, 30, 8]},
                }
            ],
            camera={},
            viewport=(100, 100),
            terrain=depth,
            declutter="greedy",
        )
    finally:
        depth.close()
    assert not plan.accepted
    assert plan.rejected[0].reason == "missing_projection_authority"
    assert any(d.code == "label_projection_authority_missing" for d in plan.diagnostics)

    from forge3d import _forge3d as native

    baseline = dict(native.global_memory_metrics())["host_visible_bytes"]
    with pytest.raises(ValueError, match="nonempty"):
        _depth_sampler(np.empty((0, 1), dtype=np.float32), viewport_size=(1, 1))
    with pytest.raises(ValueError, match="finite"):
        _depth_sampler([[float("nan")]], viewport_size=(1, 1))
    assert dict(native.global_memory_metrics())["host_visible_bytes"] == baseline


def test_unindexed_terrain_sample_never_replaces_per_candidate_sampling():
    class RecordingSampler:
        def __init__(self):
            self.calls = []

        def sample_label(self, coords, *, record, label_id):
            del record
            self.calls.append((label_id, tuple(float(value) for value in coords)))
            return {"source": "authoritative_candidate_sampler", "visible": True}

    base = {
        "id": "sampled",
        "text": "Sampled",
        "geometry": {"type": "Point", "coordinates": [50, 50, 0]},
        "screen_bounds": [45, 45, 55, 55],
        "candidate_policy": {"radial_count": 0},
        "requires_terrain": True,
    }
    sampler = RecordingSampler()
    global_sample = lp.LabelPlan.compile(
        labels=[
            {
                **base,
                "terrain_sample": {
                    "source": "forbidden_unindexed_sample",
                    "visible": False,
                },
            }
        ],
        camera={},
        viewport=(100, 100),
        terrain=sampler,
        declutter="greedy",
    )
    assert len(sampler.calls) == 5
    assert {
        candidate.terrain_sample["source"]
        for candidate in global_sample.accepted[0].candidates
    } == {"authoritative_candidate_sampler"}

    sampler.calls.clear()
    indexed_sample = lp.LabelPlan.compile(
        labels=[
            {
                **base,
                "terrain_sample": {
                    "candidate_id": "sampled:center",
                    "sample": {"source": "indexed_center", "visible": False},
                },
            }
        ],
        camera={},
        viewport=(100, 100),
        terrain=sampler,
        declutter="greedy",
    )
    assert len(sampler.calls) == 4
    by_id = {
        candidate.candidate_id: candidate
        for candidate in indexed_sample.accepted[0].candidates
    }
    assert by_id["sampled:center"].terrain_sample["source"] == "indexed_center"
    assert indexed_sample.accepted[0].candidate.candidate_id == "sampled:above"


@pytest.mark.parametrize("bias", [float("nan"), float("inf"), float("-inf")])
def test_depth_sampler_rejects_nonfinite_bias_without_leaking_reservation(bias):
    from forge3d import _forge3d as native

    baseline = dict(native.global_memory_metrics())["host_visible_bytes"]
    with pytest.raises(ValueError, match="bias must be finite"):
        _depth_sampler([[0.5]], viewport_size=(1, 1), bias=bias)
    assert dict(native.global_memory_metrics())["host_visible_bytes"] == baseline


def test_depth_sampler_requires_explicit_compatible_convention_and_domain():
    from forge3d import _forge3d as native

    baseline = dict(native.global_memory_metrics())["host_visible_bytes"]
    with pytest.raises(ValueError, match="explicit supported depth_convention"):
        _DepthOcclusionSampler(
            [[0.5]], viewport_size=(1, 1), depth_domain=_NDC_DOMAIN
        )
    with pytest.raises(ValueError, match="explicit depth_domain"):
        _DepthOcclusionSampler(
            [[0.5]],
            viewport_size=(1, 1),
            depth_convention=_NDC_CONVENTION,
        )
    with pytest.raises(ValueError, match="within depth_domain"):
        _DepthOcclusionSampler(
            [[1.5]],
            viewport_size=(1, 1),
            depth_convention=_NDC_CONVENTION,
            depth_domain=_NDC_DOMAIN,
        )
    assert dict(native.global_memory_metrics())["host_visible_bytes"] == baseline

    depth = _depth_sampler([[0.5]], viewport_size=(100, 100))
    try:
        plan = lp.LabelPlan.compile(
            labels=[
                {
                    "id": "mismatched-depth",
                    "text": "Depth",
                    "geometry": {"type": "Point", "coordinates": [20, 30, 8]},
                    "projected_anchor": [20, 30, 0.25],
                    "projected_depth_convention": "reverse_normalized_device_depth",
                    "projected_depth_domain": list(_NDC_DOMAIN),
                    "requires_terrain": True,
                    "candidate_policy": {"radial_count": 0},
                }
            ],
            camera={},
            viewport=(100, 100),
            terrain=depth,
            declutter="greedy",
        )
    finally:
        depth.close()
    assert not plan.accepted
    assert plan.rejected[0].reason == "incompatible_depth_convention"
    assert any(
        diagnostic.code == "label_depth_convention_incompatible"
        for diagnostic in plan.diagnostics
    )


@pytest.mark.parametrize(
    ("convention", "caller_visible", "derived_visible", "comparison"),
    [
        ("normalized_device_depth", True, False, "forward_less_equal"),
        (
            "reverse_normalized_device_depth",
            False,
            True,
            "reverse_greater_equal",
        ),
    ],
)
def test_authoritative_depth_visibility_ignores_contradictory_caller_boolean(
    convention, caller_visible, derived_visible, comparison
):
    class ContradictoryDepth:
        requires_projected_anchor = True
        depth_domain = _NDC_DOMAIN

        def __init__(self):
            self.depth_convention = convention

        def sample_label(self, coords, *, record, label_id):
            del record, label_id
            return {
                "scene_depth": 0.5,
                "label_depth": float(coords[2]),
                "bias": 0.0,
                "visible": caller_visible,
                "source": "contradictory_authoritative_depth",
                "depth_authority": "pre_supplied_authoritative",
                "depth_convention": convention,
                "depth_domain": list(_NDC_DOMAIN),
            }

    plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "contradiction",
                "source_id": "depth-source",
                "text": "Depth",
                "geometry": {"type": "Point", "coordinates": [50, 50, 0.75]},
                "projected_anchor": [50, 50, 0.75],
                "projected_depth_convention": convention,
                "projected_depth_domain": list(_NDC_DOMAIN),
                "requires_terrain": True,
                "label_size": [4, 4],
                "candidate_policy": {"offset_px": 0, "radial_count": 0},
            }
        ],
        camera={},
        viewport=(100, 100),
        terrain=ContradictoryDepth(),
    )

    if derived_visible:
        assert [label.label_id for label in plan.accepted] == ["contradiction"]
        samples = [
            candidate.terrain_sample
            for candidate in plan.accepted[0].candidates
        ]
    else:
        assert not plan.accepted
        assert plan.rejected[0].reason == "terrain_occluded"
        samples = [
            record["terrain_sample"]
            for record in plan.rationale
            if record["kind"] == "occluded_anchor"
        ]
    assert len(samples) == 5
    assert all(sample["visible"] is derived_visible for sample in samples)
    assert all(sample["scene_depth"] == 0.5 for sample in samples)
    assert all(sample["label_depth"] == 0.75 for sample in samples)
    assert all(sample["depth_convention"] == convention for sample in samples)
    assert all(sample["depth_comparison"] == comparison for sample in samples)
    assert all(
        sample["visibility_authority"] == "label_plan.compile"
        for sample in samples
    )


@pytest.mark.parametrize(
    "viewport",
    [
        (0, 100),
        (-1, 100),
        (float("nan"), 100),
        (float("inf"), 100),
        {"width": 100},
    ],
)
def test_compile_rejects_nonfinite_or_nonpositive_viewport(viewport):
    with pytest.raises(ValueError, match="viewport must be finite and positive"):
        lp.LabelPlan.compile(
            labels=[
                {
                    "id": "far-outside",
                    "text": "Outside",
                    "geometry": {"type": "Point", "coordinates": [500, 50, 0]},
                }
            ],
            camera={},
            viewport=viewport,
        )


@pytest.mark.parametrize(
    "invalid_glyph",
    [
        {
            "font_index": 0,
            "glyph_id": 8,
            "origin": [3.0, 4.0],
            "rotation": 0.2,
            "advance": [float("nan"), 0.0],
        },
        {
            "font_index": 0,
            "glyph_id": 8,
            "origin": [3.0, 4.0],
            "rotation": 0.2,
            "scale": 0.0,
        },
        {
            "font_index": 0,
            "glyph_id": 8,
            "origin": [3.0, 4.0],
            "rotation": 0.2,
            "has_outline": "yes",
        },
        {
            "font_index": 0,
            "origin": [3.0, 4.0],
            "rotation": 0.2,
        },
    ],
)
def test_every_candidate_authority_glyph_stream_is_validated_before_solve(
    monkeypatch, invalid_glyph
):
    solver_called = False

    def solver(candidates, **_kwargs):
        nonlocal solver_called
        solver_called = True
        assert list(candidates) == []
        return (
            [],
            0.0,
            _Rationale(
                [{"kind": "solver", "nodes_explored": 0, "certified": True, "gap": 0.0}]
            ),
        )

    monkeypatch.setattr(lp, "_native_declutter_optimal", lambda: solver)
    valid = [
        {"font_index": 0, "glyph_id": 7, "origin": [2.0, 3.0], "rotation": 0.1}
    ]
    plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "road",
                "text": "Road",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [20, 0]],
                },
                "geometry_authority": {
                    "source": "compute_line_label_placement",
                    "positioned_glyphs": valid,
                    "candidates": [
                        {
                            "candidate_id": "road:a",
                            "anchor": [10, 10, 0.5],
                            "bounds": [5, 8, 15, 12],
                            "positioned_glyphs": valid,
                        },
                        {
                            "candidate_id": "road:b",
                            "anchor": [30, 10, 0.5],
                            "bounds": [25, 8, 35, 12],
                            "positioned_glyphs": [invalid_glyph],
                        },
                    ],
                },
            }
        ],
        camera={},
        viewport=(100, 100),
    )
    assert solver_called is True
    assert not plan.accepted
    assert plan.rejected[0].reason == "missing_geometry_authority"
    assert plan.plan_hash() == hashlib.sha256(plan.canonical_bytes()).hexdigest()


def test_visibility_filtered_rationale_translates_identity_and_deduplicates(
    monkeypatch,
):
    def solver(candidates, **_kwargs):
        bridged = list(candidates)
        assert [visible for *_prefix, visible in bridged] == [False, False, True]
        return (
            [(0, 2)],
            0.0,
            _Rationale(
                [
                    {
                        "kind": "visibility_filtered_candidate",
                        "label_id": 0,
                        "candidate_index": 0,
                    },
                    {
                        "kind": "visibility_filtered_candidate",
                        "label_id": 0,
                        "candidate_index": 1,
                    },
                    {
                        "kind": "placed",
                        "label_id": 0,
                        "candidate_index": 2,
                        "weight": 1.0,
                    },
                    {
                        "kind": "solver",
                        "nodes_explored": 1,
                        "certified": True,
                        "gap": 0.0,
                    },
                ]
            ),
        )

    monkeypatch.setattr(lp, "_native_declutter_optimal", lambda: solver)
    glyphs = [
        {"font_index": 0, "glyph_id": 7, "origin": [2.0, 3.0], "rotation": 0.1}
    ]
    plan = lp.LabelPlan.compile(
        labels=[
            {
                "id": "road",
                "source_id": "road-source",
                "text": "Road",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [20, 0]],
                },
                "geometry_authority": {
                    "source": "compute_line_label_placement",
                    "positioned_glyphs": glyphs,
                    "candidates": [
                        {
                            "candidate_id": "road:hidden",
                            "anchor": [20, 20, 0.5],
                            "bounds": [15, 18, 25, 22],
                            "visible": False,
                        },
                        {
                            "candidate_id": "road:outside",
                            "anchor": [120, 20, 0.5],
                            "bounds": [115, 18, 125, 22],
                        },
                        {
                            "candidate_id": "road:chosen",
                            "anchor": [50, 20, 0.5],
                            "bounds": [45, 18, 55, 22],
                        },
                    ],
                },
            }
        ],
        camera={},
        viewport=(100, 100),
    )

    filtered = [
        record
        for record in plan.rationale
        if record["kind"] == "visibility_filtered_candidate"
    ]
    assert filtered == [
        {
            "candidate_bounds": [15.0, 18.0, 25.0, 22.0],
            "candidate_id": "road:hidden",
            "candidate_index": 0,
            "kind": "visibility_filtered_candidate",
            "label_id": "road",
            "source_id": "road-source",
            "visibility_gates": [],
            "visibility_reason": "geometry_authority_visible_false",
        }
    ]
    assert sum(
        record.get("kind") == "candidate_ineligible"
        and record.get("candidate_id") == "road:outside"
        for record in plan.rationale
    ) == 1
    rendered = plan.render_rationale()
    assert any(
        "filtered label 'road' (source 'road-source') candidate 'road:hidden' "
        "at index 0" in line
        for line in rendered
    )
    assert not any("record[visibility_filtered_candidate]" in line for line in rendered)
    assert not any("label 0" in line for line in rendered)
    canonical = plan.canonical_bytes()
    assert canonical == lp.LabelPlan.from_dict(plan.to_dict()).canonical_bytes()
    assert plan.plan_hash() == hashlib.sha256(canonical).hexdigest()


def test_line_authority_requires_complete_positioned_glyph_geometry():
    incomplete = lp.LabelPlan.compile(
        labels=[
            {
                "id": "road",
                "text": "Road",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [20, 0]]},
                "geometry_authority": {
                    "source": "compute_line_label_placement",
                    "candidates": [{"anchor": [10, 0, 0.5], "bounds": [5, -2, 15, 2]}],
                    "positioned_glyphs": [],
                },
            }
        ],
        camera={},
        viewport=(100, 100),
        declutter="greedy",
    )
    assert incomplete.rejected[0].reason == "missing_geometry_authority"
    assert incomplete.diagnostics[0].code == "label_geometry_authority_missing"


def test_mapscene_serialized_projection_compiles_and_releases_native_depth_memory():
    from forge3d import _forge3d as native

    baseline = dict(native.global_memory_metrics())["host_visible_bytes"]
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((2, 2), dtype=np.float32),
            metadata={"source_id": "flat", "width": 2, "height": 2},
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
                        "geometry": {"type": "Point", "coordinates": [1000, 2000, 3000]},
                        "projected_anchor": [32, 32, 0.25],
                        "projected_depth_convention": _NDC_CONVENTION,
                        "projected_depth_domain": list(_NDC_DOMAIN),
                        "candidate_policy": {"radial_count": 0},
                    }
                ],
                glyph_atlas={"glyphs": sorted(set("Front"))},
                metadata={
                    "depth_occlusion": {
                        "image": [[0.5, 0.5], [0.5, 0.5]],
                        "source": "serialized_authoritative_depth",
                        "depth_convention": _NDC_CONVENTION,
                        "depth_domain": list(_NDC_DOMAIN),
                    }
                },
            )
        ],
    )
    compiled = scene.compile_plan()
    assert [item.label_id for item in compiled.label_plans["labels"].accepted] == [
        "front"
    ]
    assert compiled.manifest.depth_cull["depth_proxy"] == "pre_supplied_authoritative"
    assert dict(native.global_memory_metrics())["host_visible_bytes"] == baseline


def test_mapscene_visibility_uses_stable_instance_and_source_identity():
    candidate = lp.LabelCandidate(
        candidate_id="duplicate:center",
        candidate_type="center",
        anchor=(16, 16, 1),
        score=1.0,
        bounds=(12, 12, 20, 20),
        ordering_key="source-a-instance:0000",
    )
    frozen_plan = lp.LabelPlan(
        accepted=[
            lp.AcceptedLabel(
                label_id="duplicate",
                source_id="source-a",
                text="A",
                geometry_type="Point",
                candidate=candidate,
                candidates=(candidate,),
                screen_bounds=(12, 12, 20, 20),
                world_bounds=(16, 16, 1, 16, 16, 1),
                glyphs=("A",),
                ordering_key="source-a-instance:0000",
            )
        ],
        rejected=[
            lp.RejectedLabel(
                label_id="duplicate",
                source_id="source-b",
                reason="terrain_occluded",
                candidate_id="duplicate:center",
                ordering_key="source-b-instance:0000",
            )
        ],
    )
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((2, 2), dtype=np.float32),
            metadata={"source_id": "flat", "width": 2, "height": 2},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                plan=frozen_plan,
                glyph_atlas={"glyphs": ["A"]},
            )
        ],
    )
    compiled = scene.compile_plan()
    visibility = compiled.manifest.depth_cull["layers"]["labels"]["visibility"]
    assert len(visibility) == 2
    assert all(key.startswith("label-instance:") for key in visibility)
    assert all(entry["instance_id"] == key for key, entry in visibility.items())
    assert {entry["label_id"] for entry in visibility.values()} == {"duplicate"}
    assert sorted(entry["source_id"] for entry in visibility.values()) == [
        "source-a",
        "source-b",
    ]
    by_source = {entry["source_id"]: entry for entry in visibility.values()}
    assert by_source["source-a"]["visible"] is True
    assert by_source["source-b"]["visible"] is False
    assert by_source["source-b"]["reason"] == "terrain_occluded"
