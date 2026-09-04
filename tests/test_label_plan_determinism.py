def _city_labels(order):
    records = {
        "b": {
            "id": "city-b",
            "text": "Beta",
            "geometry": {"type": "Point", "coordinates": (30.0, 40.0, 6.0)},
            "label_size": (1.0, 1.0),
            "priority": 3,
        },
        "a": {
            "id": "city-a",
            "text": "Alpha",
            "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 5.0)},
            "label_size": (1.0, 1.0),
            "priority": 5,
        },
        "c": {
            "id": "city-c",
            "text": "Gamma",
            "geometry": {"type": "Point", "coordinates": (20.0, 15.0, 4.0)},
            "label_size": (1.0, 1.0),
            "priority": 4,
        },
    }
    return [records[key] for key in order]


def _compile(labels):
    from forge3d import LabelPlan

    return LabelPlan.compile(
        labels=labels,
        camera={"name": "fixed"},
        viewport={"height": 600, "width": 800},
        keepouts=[],
        priority_rules=[],
        typography={"size": 12, "family": "default"},
        glyph_atlas={"glyphs": list("ABGaaehlmpt")},
        seed=99,
    ).to_dict()


def test_label_plan_compile_is_stable_for_fixed_inputs():
    first = _compile(_city_labels(["b", "a", "c"]))
    second = _compile(_city_labels(["b", "a", "c"]))

    assert second == first


def test_label_plan_normalizes_equivalent_source_order():
    forward = _compile(_city_labels(["a", "b", "c"]))
    reverse = _compile(_city_labels(["c", "b", "a"]))

    assert reverse == forward
    assert [label["label_id"] for label in forward["accepted"]] == [
        "city-a",
        "city-b",
        "city-c",
    ]


def test_label_plan_serialization_is_independent_of_mapping_key_order():
    from forge3d import LabelPlan

    left = LabelPlan.compile(
        labels=[
            {
                "text": "Alpha",
                "priority": 5,
                "geometry": {"coordinates": (10.0, 20.0, 5.0), "type": "Point"},
                "id": "city-a",
            }
        ],
        camera={},
        viewport=(800, 600),
        seed=7,
    ).to_dict()
    right = LabelPlan.compile(
        labels=[
            {
                "id": "city-a",
                "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 5.0)},
                "priority": 5,
                "text": "Alpha",
            }
        ],
        camera={},
        viewport=(800, 600),
        seed=7,
    ).to_dict()

    assert right == left
