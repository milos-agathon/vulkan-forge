from pathlib import Path


QUICKSTART = Path("docs/guides/label_plan_guide.md")


def test_label_plan_quickstart_names_public_api_and_required_scenarios():
    text = QUICKSTART.read_text(encoding="utf-8")

    for term in [
        "LabelPlan.compile",
        "KeepoutRegion",
        "PriorityClass",
        "to_render_payload",
        "to_export_payload",
        "placeholder_fallback",
        "label_rejection_summary",
        "missing_glyphs",
    ]:
        assert term in text

    for reason in [
        "collision",
        "outside_view",
        "missing_glyph",
        "priority_lost",
        "keepout_region",
        "terrain_occluded",
        "invalid_geometry",
        "unsupported_geometry_type",
        "empty_text",
    ]:
        assert f"`{reason}`" in text


def test_label_plan_quickstart_scenario_is_executable_and_reproducible():
    from forge3d import KeepoutRegion, LabelPlan, PriorityClass

    labels = [
        {
            "id": "capital",
            "text": "Capital",
            "geometry": {"type": "Point", "coordinates": (40.0, 40.0, 0.0)},
            "label_size": (10.0, 10.0),
            "candidate_policy": {"offset_px": 0.0, "radial_count": 0},
            "priority_class": "capital",
        },
        {
            "id": "local",
            "text": "Local",
            "geometry": {"type": "Point", "coordinates": (40.0, 40.0, 0.0)},
            "label_size": (10.0, 10.0),
            "candidate_policy": {"offset_px": 0.0, "radial_count": 0},
            "priority_class": "local",
        },
        {
            "id": "legend-hit",
            "text": "Legend",
            "geometry": {"type": "Point", "coordinates": (10.0, 10.0, 0.0)},
            "label_size": (4.0, 4.0),
            "candidate_policy": {"offset_px": 0.0, "radial_count": 0},
        },
        {
            "id": "glyph-gap",
            "text": "Glyph!",
            "geometry": {"type": "Point", "coordinates": (70.0, 10.0, 0.0)},
        },
    ]

    kwargs = dict(
        labels=labels,
        camera={"name": "fixed"},
        viewport=(100, 100),
        keepouts=[KeepoutRegion(region_id="legend", kind="legend", bounds=(0.0, 0.0, 20.0, 20.0))],
        priority_rules=[PriorityClass(name="local", rank=10), PriorityClass(name="capital", rank=20)],
        glyph_atlas={"glyphs": set("CapitalLocalLegendGlyph")},
        seed=17,
    )
    first = LabelPlan.compile(**kwargs)
    second = LabelPlan.compile(**kwargs)

    assert second.to_dict() == first.to_dict()
    assert [label.label_id for label in first.accepted] == ["capital"]
    assert {label.reason for label in first.rejected} == {
        "keepout_region",
        "missing_glyph",
        "priority_lost",
    }
    assert {diagnostic.code for diagnostic in first.diagnostics} == {
        "label_rejection_summary",
        "missing_glyphs",
    }

    render_payload = first.to_render_payload(backend="native-gpu")
    export_payload = first.to_export_payload()
    assert any(diagnostic["code"] == "placeholder_fallback" for diagnostic in render_payload["diagnostics"])
    assert export_payload["accepted"] == first.to_dict()["accepted"]
