from pathlib import Path

from forge3d.diagnostics import Diagnostic
import pytest


def _labels():
    return [
        {
            "id": "city-a",
            "text": "Alpha",
            "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 5.0)},
            "label_size": (1.0, 1.0),
            "priority": 5,
        },
        {
            "id": "city-b",
            "text": "Beta",
            "geometry": {"type": "Point", "coordinates": (30.0, 40.0, 6.0)},
            "label_size": (1.0, 1.0),
            "priority": 3,
        },
    ]


def test_label_plan_public_contract_and_roundtrip():
    from forge3d import AcceptedLabel, KeepoutRegion, LabelCandidate, LabelPlan, PriorityClass

    keepout = KeepoutRegion(
        region_id="legend",
        kind="legend",
        bounds=(700.0, 500.0, 780.0, 560.0),
        priority=10,
    )
    priority = PriorityClass(name="cities", rank=20)

    plan = LabelPlan.compile(
        labels=_labels(),
        camera={"name": "fixed"},
        viewport={"width": 800, "height": 600},
        keepouts=[keepout],
        priority_rules=[priority],
        typography={"family": "default", "size": 14},
        glyph_atlas={"glyphs": list("ABaehlpt")},
        seed=1234,
    )

    assert plan.seed == 1234
    assert len(plan.accepted) == 2
    assert plan.rejected == []
    assert plan.diagnostics == []
    assert plan.bounds["screen"] == [9.5, 19.5, 30.5, 40.5]
    assert isinstance(plan.accepted[0], AcceptedLabel)
    assert isinstance(plan.accepted[0].candidate, LabelCandidate)
    assert plan.accepted[0].candidate.candidate_type == "center"

    payload = plan.to_dict()
    assert payload["payload_version"] == 2
    assert [label["label_id"] for label in payload["accepted"]] == ["city-a", "city-b"]
    assert LabelPlan.from_dict(payload).to_dict() == payload

    render_payload = plan.to_render_payload()
    export_payload = plan.to_export_payload()
    assert render_payload["kind"] == "label_plan_render_payload"
    assert export_payload["kind"] == "label_plan_export_payload"
    assert render_payload["accepted"] == payload["accepted"]
    assert export_payload["accepted"] == payload["accepted"]

    assert isinstance(Diagnostic.from_dict, object)


def test_label_plan_payload_version_migration_and_rejection():
    from forge3d import LabelPlan

    payload = LabelPlan.compile(
        labels=_labels(),
        camera={"name": "fixed"},
        viewport={"width": 800, "height": 600},
        glyph_atlas={"glyphs": list("ABaehlpt")},
        seed=1234,
    ).to_dict()
    v1 = {
        **payload,
        "payload_version": 1,
        "accepted": [
            {key: value for key, value in label.items() if key != "positioned_glyphs"}
            for label in payload["accepted"]
        ],
    }

    migrated = LabelPlan.from_dict(v1).to_dict()
    assert migrated["payload_version"] == 2
    assert all("positioned_glyphs" in label for label in migrated["accepted"])
    assert all(
        label["typography"]["render_mapping"] == "legacy_codepoints_not_renderable"
        for label in migrated["accepted"]
    )

    for version in (0, 999, "bad"):
        with pytest.raises(ValueError, match="unsupported label plan payload_version"):
            LabelPlan.from_dict({**payload, "payload_version": version})


def test_label_plan_serializes_native_line_ranges_and_positioned_lines():
    from forge3d import LabelPlan

    root = Path(__file__).resolve().parents[1]
    font = root / "assets" / "fonts" / "NotoSansHebrew-subset.ttf"
    plan = LabelPlan.compile(
        labels=[
            {
                "id": "hebrew",
                "text": "בד\nשב",
                "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 5.0)},
            }
        ],
        camera={},
        viewport={"width": 800, "height": 600},
        glyph_atlas={"font_path": str(font)},
    )

    payload = plan.to_dict()
    label = payload["accepted"][0]
    assert label["line_ranges"] == [[0, 2], [3, 5]]
    assert sorted({glyph["line_index"] for glyph in label["positioned_glyphs"]}) == [0, 1]
    assert LabelPlan.from_dict(payload).to_dict() == payload


def test_label_plan_uses_explicit_native_line_ranges_without_break_controls():
    from forge3d import LabelPlan

    root = Path(__file__).resolve().parents[1]
    font = root / "assets" / "fonts" / "NotoSansHebrew-subset.ttf"
    plan = LabelPlan.compile(
        labels=[
            {
                "id": "wrapped",
                "text": "בדשב",
                "line_ranges": [[0, 2], [2, 4]],
                "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 5.0)},
            }
        ],
        camera={},
        viewport={"width": 800, "height": 600},
        glyph_atlas={"font_path": str(font)},
    )

    label = plan.to_dict()["accepted"][0]
    assert label["line_ranges"] == [[0, 2], [2, 4]]
    assert sorted({glyph["line_index"] for glyph in label["positioned_glyphs"]}) == [0, 1]
    assert "\n" not in label["glyphs"]


def test_label_plan_rejects_empty_sources_without_placeholder_success():
    from forge3d import LabelPlan

    plan = LabelPlan.compile(
        labels=[
            {
                "id": "empty",
                "text": "   ",
                "geometry": {"type": "Point", "coordinates": (0.0, 0.0, 0.0)},
            }
        ],
        camera={},
        viewport=(100, 100),
        seed=0,
    )

    assert plan.accepted == []
    assert len(plan.rejected) == 1
    assert plan.rejected[0].reason == "empty_text"
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["label_rejection_summary"]
    assert plan.to_render_payload()["accepted"] == []
    assert plan.to_render_payload()["diagnostics"][0]["code"] == "label_rejection_summary"
