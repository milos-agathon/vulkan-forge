import pytest


def _label(label_id="label", x=20.0, y=20.0):
    return {
        "id": label_id,
        "text": "Keepout",
        "geometry": {"type": "Point", "coordinates": (x, y, 0.0)},
        "label_size": (4.0, 4.0),
        "candidate_policy": {"offset_px": 0.0, "radial_count": 0},
    }


@pytest.mark.parametrize(
    "kind",
    ["title", "legend", "scale_bar", "north_arrow", "manual_rectangle"],
)
def test_required_keepout_region_kinds_reject_intersecting_candidates(kind):
    from forge3d import KeepoutRegion, LabelPlan

    plan = LabelPlan.compile(
        labels=[_label(label_id=f"{kind}-label")],
        camera={},
        viewport=(100, 100),
        keepouts=[
            KeepoutRegion(
                region_id=f"{kind}-region",
                kind=kind,
                bounds=(10.0, 10.0, 30.0, 30.0),
                priority=5,
            )
        ],
        seed=8,
    )

    assert plan.accepted == []
    assert len(plan.rejected) == 1
    assert plan.rejected[0].reason == "keepout_region"
    assert plan.rejected[0].candidate_id == f"{kind}-label:center"
    assert plan.rejected[0].details["keepout_region_id"] == f"{kind}-region"
    assert plan.rejected[0].details["keepout_kind"] == kind


def test_non_intersecting_keepout_region_is_retained_without_rejection():
    from forge3d import LabelPlan

    plan = LabelPlan.compile(
        labels=[_label()],
        camera={},
        viewport=(100, 100),
        keepouts=[
            {
                "region_id": "manual-away",
                "kind": "manual_rectangle",
                "bounds": (70.0, 70.0, 90.0, 90.0),
            }
        ],
        seed=8,
    )

    assert len(plan.accepted) == 1
    assert plan.rejected == []
    assert plan.bounds["keepouts"] == [
        {
            "region_id": "manual-away",
            "kind": "manual_rectangle",
            "bounds": [70.0, 70.0, 90.0, 90.0],
            "priority": 0,
        }
    ]
