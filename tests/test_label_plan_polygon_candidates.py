import pytest


def _compile(labels):
    from forge3d import LabelPlan

    return LabelPlan.compile(
        labels=labels,
        camera={"name": "fixed"},
        viewport={"width": 200, "height": 200},
        seed=5,
    )


def test_polygon_label_generates_centroid_and_visual_center_candidates():
    plan = _compile(
        [
            {
                "id": "square",
                "text": "Square",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(50.0, 50.0), (70.0, 50.0), (70.0, 60.0), (50.0, 60.0), (50.0, 50.0)]],
                },
            }
        ]
    )

    assert plan.rejected == []
    assert len(plan.accepted) == 1
    accepted = plan.accepted[0]
    candidates = [candidate.to_dict() for candidate in accepted.candidates]

    assert accepted.candidate.candidate_id == "square:centroid"
    assert [candidate["candidate_type"] for candidate in candidates] == [
        "centroid",
        "visual_center",
    ]
    assert candidates[0]["anchor"] == pytest.approx([60.0, 55.0, 0.0])
    assert candidates[0]["details"]["inside_polygon"] is True
    assert candidates[1]["candidate_id"] == "square:visual-center"
    assert candidates[1]["details"]["fallback_for"] == "centroid"


def test_concave_polygon_uses_visual_center_when_centroid_is_unsuitable():
    plan = _compile(
        [
            {
                "id": "concave",
                "text": "Concave",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            (50.0, 50.0),
                            (56.0, 50.0),
                            (56.0, 51.0),
                            (51.0, 51.0),
                            (51.0, 55.0),
                            (56.0, 55.0),
                            (56.0, 56.0),
                            (50.0, 56.0),
                            (50.0, 50.0),
                        ]
                    ],
                },
            }
        ]
    )

    assert plan.rejected == []
    accepted = plan.accepted[0]
    candidates = {candidate.candidate_type: candidate.to_dict() for candidate in accepted.candidates}

    assert accepted.candidate.candidate_type == "visual_center"
    assert candidates["centroid"]["details"]["inside_polygon"] is False
    assert candidates["visual_center"]["details"]["fallback_for"] == "centroid"


def test_invalid_polygon_geometry_is_rejected_with_reason_code():
    plan = _compile(
        [
            {
                "id": "flat-poly",
                "text": "Flat",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (0.0, 0.0)]],
                },
            }
        ]
    )

    assert plan.accepted == []
    assert len(plan.rejected) == 1
    assert plan.rejected[0].label_id == "flat-poly"
    assert plan.rejected[0].reason == "invalid_geometry"
    assert plan.diagnostics[0].code == "label_rejection_summary"
