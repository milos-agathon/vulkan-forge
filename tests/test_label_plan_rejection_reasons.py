from pathlib import Path

from forge3d.label_plan import REJECTION_REASONS


BASE_REJECTION_REASONS = tuple(
    reason
    for reason in REJECTION_REASONS
    if reason not in {"font_chain_required", "malformed_font", "shaping_failed"}
)


_GLYPHS_WITHOUT_BANG = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
_POINT_CANDIDATE_SUFFIXES = ("center", "above", "below", "left", "right")


def _candidate_samples(label_id, sample):
    return {
        f"{label_id}:{suffix}": dict(sample)
        for suffix in _POINT_CANDIDATE_SUFFIXES
    }


def _point(label_id, text, x, y, *, priority=0, **extra):
    record = {
        "id": label_id,
        "text": text,
        "geometry": {"type": "Point", "coordinates": (x, y, 0.0)},
        "label_size": (4.0, 4.0),
        "candidate_policy": {"offset_px": 0.0, "radial_count": 0},
        "priority": priority,
    }
    record.update(extra)
    return record


def _reason_fixture_labels():
    return [
        _point("empty-text", "   ", 1.0, 1.0),
        _point("missing-glyph", "Bang!", 2.0, 2.0),
        _point("outside-view", "Outside", 200.0, 5.0),
        {
            "id": "invalid-geometry",
            "text": "Invalid",
            "geometry": {"type": "Point", "coordinates": ("bad", 4.0, 0.0)},
        },
        {
            "id": "geometry-authority-missing",
            "text": "Authority",
            "geometry": {"type": "LineString", "coordinates": [(5.0, 5.0), (6.0, 6.0)]},
        },
        {
            "id": "unsupported-geometry",
            "text": "Unsupported",
            "geometry": {"type": "MultiPoint", "coordinates": [(5.0, 5.0), (6.0, 6.0)]},
        },
        _point("keepout-label", "Keepout", 20.0, 20.0),
        _point(
            "terrain-label",
            "Terrain",
            40.0,
            40.0,
            requires_terrain=True,
            candidate_terrain_samples=_candidate_samples(
                "terrain-label",
                {"visible": False, "elevation": 10.0, "source": "fixture"},
            ),
        ),
        _point(
            "missing-projection",
            "Projection",
            80.0,
            20.0,
            requires_terrain=True,
            candidate_terrain_samples=_candidate_samples(
                "missing-projection",
                {
                    "visible": True,
                    "scene_depth": 0.5,
                    "label_depth": 0.25,
                    "depth_authority": "pre_supplied_authoritative",
                    "depth_convention": "normalized_device_depth",
                    "depth_domain": [0.0, 1.0],
                },
            ),
        ),
        _point(
            "incompatible-depth",
            "Depth",
            80.0,
            40.0,
            projected_anchor=[80.0, 40.0, 0.25],
            projected_depth_convention="reverse_normalized_device_depth",
            projected_depth_domain=[0.0, 1.0],
            requires_terrain=True,
            candidate_terrain_samples=_candidate_samples(
                "incompatible-depth",
                {
                    "visible": True,
                    "scene_depth": 0.5,
                    "label_depth": 0.25,
                    "depth_authority": "pre_supplied_authoritative",
                    "depth_convention": "normalized_device_depth",
                    "depth_domain": [0.0, 1.0],
                },
            ),
        ),
        _point(
            "no-eligible",
            "Mixed",
            50.0,
            50.0,
            requires_terrain=True,
            candidate_policy={"offset_px": 100.0, "radial_count": 0},
            candidate_terrain_samples={
                "no-eligible:center": {"visible": False, "source": "fixture"},
                **{
                    f"no-eligible:{suffix}": {"visible": True, "source": "fixture"}
                    for suffix in ("above", "below", "left", "right")
                },
            },
        ),
        _point("collision-a", "One", 50.0, 50.0, priority=5),
        _point("collision-b", "Two", 50.0, 50.0, priority=5),
        _point("priority-high", "High", 60.0, 60.0, priority=20),
        _point("priority-low", "Low", 60.0, 60.0, priority=1),
    ]


def _compile_reason_fixture():
    from forge3d import KeepoutRegion, LabelPlan

    return LabelPlan.compile(
        labels=_reason_fixture_labels(),
        camera={"name": "fixed"},
        viewport={"width": 100, "height": 100},
        keepouts=[
            KeepoutRegion(
                region_id="legend",
                kind="legend",
                bounds=(10.0, 10.0, 30.0, 30.0),
            )
        ],
        glyph_atlas={"glyphs": _GLYPHS_WITHOUT_BANG},
        seed=11,
    )


def test_label_plan_retains_every_required_rejection_reason():
    from forge3d import LabelPlan

    plan = _compile_reason_fixture()
    reasons_by_label = {label.label_id: label.reason for label in plan.rejected}

    assert reasons_by_label == {
        "collision-b": "collision",
        "empty-text": "empty_text",
        "geometry-authority-missing": "missing_geometry_authority",
        "incompatible-depth": "incompatible_depth_convention",
        "invalid-geometry": "invalid_geometry",
        "keepout-label": "keepout_region",
        "missing-projection": "missing_projection_authority",
        "missing-glyph": "missing_glyph",
        "no-eligible": "no_eligible_candidate",
        "outside-view": "outside_view",
        "priority-low": "priority_lost",
        "terrain-label": "terrain_occluded",
        "unsupported-geometry": "unsupported_geometry_type",
    }
    assert set(reasons_by_label.values()) == set(BASE_REJECTION_REASONS)

    diagnostics_by_code = {diagnostic.code: diagnostic for diagnostic in plan.diagnostics}
    assert diagnostics_by_code["missing_glyphs"].object_id == "missing-glyph"
    assert diagnostics_by_code["missing_glyphs"].details["missing_glyphs"] == ["!"]
    assert diagnostics_by_code["label_rejection_summary"].details["rejection_counts"] == {
        reason: 1 for reason in BASE_REJECTION_REASONS
    }

    payload = plan.to_dict()
    assert LabelPlan.from_dict(payload).to_dict() == payload


def test_shaping_rejection_reasons_are_structured(
    tmp_path, monkeypatch
):
    from forge3d import LabelPlan
    from forge3d import text as text_module

    def compile_with(atlas):
        return LabelPlan.compile(
            labels=[_point("arabic", "مرحبا", 10.0, 10.0)],
            camera={},
            viewport=(100, 100),
            glyph_atlas=atlas,
        ).rejected[0]

    missing_chain = compile_with({"glyphs": list("مرحبا")})

    malformed_path = tmp_path / "malformed.ttf"
    malformed_path.write_bytes(b"not a font")
    malformed = compile_with(
        {"glyphs": list("مرحبا"), "font_path": str(malformed_path)}
    )

    packaged = Path(__file__).resolve().parents[1] / "assets/fonts/NotoSansArabic-subset.ttf"
    monkeypatch.setattr(
        text_module,
        "shape",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("synthetic failure")),
    )
    generic = compile_with({"glyphs": list("مرحبا"), "font_path": str(packaged)})

    reasons = {missing_chain.reason, malformed.reason, generic.reason}
    assert reasons == {"font_chain_required", "malformed_font", "shaping_failed"}
    assert reasons | set(BASE_REJECTION_REASONS) == set(REJECTION_REASONS)
    for rejected in (missing_chain, malformed, generic):
        assert rejected.details["diagnostics"] or rejected.reason == "shaping_failed"


def test_rejected_labels_keep_candidate_identity_and_structured_details():
    plan = _compile_reason_fixture()
    rejected_payload = {label["label_id"]: label for label in plan.to_dict()["rejected"]}

    assert rejected_payload["collision-b"]["candidate_id"] == "collision-b:center"
    assert rejected_payload["collision-b"]["details"]["collides_with"] == "collision-a"
    assert rejected_payload["priority-low"]["candidate_id"] == "priority-low:center"
    assert rejected_payload["priority-low"]["details"]["collides_with"] == "priority-high"
    assert rejected_payload["keepout-label"]["candidate_id"] == "keepout-label:center"
    assert rejected_payload["keepout-label"]["details"]["keepout_region_id"] == "legend"
    assert rejected_payload["terrain-label"]["candidate_id"] == "terrain-label:center"
    assert rejected_payload["terrain-label"]["details"]["terrain_sample"]["visible"] is False
    assert plan.to_render_payload()["rejected"] == plan.to_dict()["rejected"]
