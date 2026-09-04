def _point(label_id, priority_class, *, priority=0):
    return {
        "id": label_id,
        "text": label_id.title().replace("-", ""),
        "geometry": {"type": "Point", "coordinates": (40.0, 40.0, 0.0)},
        "label_size": (10.0, 10.0),
        "candidate_policy": {"offset_px": 0.0, "radial_count": 0},
        "priority": priority,
        "priority_class": priority_class,
    }


def test_priority_classes_determine_collision_winner_and_loser():
    from forge3d import LabelPlan, PriorityClass

    plan = LabelPlan.compile(
        labels=[
            _point("local-label", "local", priority=100),
            _point("capital-label", "capital", priority=1),
        ],
        camera={},
        viewport=(100, 100),
        priority_rules=[
            PriorityClass(name="local", rank=10),
            PriorityClass(name="capital", rank=50),
        ],
        seed=4,
    )

    assert [label.label_id for label in plan.accepted] == ["capital-label"]
    assert len(plan.rejected) == 1
    assert plan.rejected[0].label_id == "local-label"
    assert plan.rejected[0].reason == "priority_lost"
    assert plan.rejected[0].details["collides_with"] == "capital-label"
    assert plan.rejected[0].details["winner_priority_class"] == "capital"
    assert plan.rejected[0].details["candidate_priority_class"] == "local"


def test_equal_priority_ties_use_stable_label_ordering():
    from forge3d import LabelPlan, PriorityClass

    plan = LabelPlan.compile(
        labels=[
            _point("zeta-label", "same", priority=5),
            _point("alpha-label", "same", priority=5),
        ],
        camera={},
        viewport=(100, 100),
        priority_rules=[PriorityClass(name="same", rank=20)],
        seed=4,
    )

    assert [label.label_id for label in plan.accepted] == ["alpha-label"]
    assert len(plan.rejected) == 1
    assert plan.rejected[0].label_id == "zeta-label"
    assert plan.rejected[0].reason == "collision"
    assert plan.rejected[0].details["collides_with"] == "alpha-label"
