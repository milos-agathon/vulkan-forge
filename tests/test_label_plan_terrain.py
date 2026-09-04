class _TerrainSampler:
    def __init__(self, *, visible=True, elevation=42.5):
        self.visible = visible
        self.elevation = elevation

    def sample(self, x, y, z=0.0):
        return {
            "elevation": self.elevation + (x * 0.1) + (y * 0.01) + (z * 0.001),
            "source": "fixture-dem",
            "visible": self.visible,
        }


def _terrain_label(**extra):
    label = {
        "id": "terrain-label",
        "text": "Terrain",
        "geometry": {"type": "Point", "coordinates": (10.0, 20.0, 5.0)},
        "label_size": (2.0, 2.0),
        "requires_terrain": True,
    }
    label.update(extra)
    return label


def test_terrain_sampler_updates_point_elevation_and_candidate_samples():
    from forge3d import LabelPlan

    plan = LabelPlan.compile(
        labels=[_terrain_label()],
        camera={"name": "fixed"},
        viewport=(100, 100),
        terrain=_TerrainSampler(elevation=100.0),
        seed=3,
    )

    assert plan.rejected == []
    accepted = plan.accepted[0]
    expected_elevation = 101.205
    assert accepted.candidate.anchor == (10.0, 20.0, expected_elevation)
    assert accepted.world_bounds == (
        10.0,
        20.0,
        expected_elevation,
        10.0,
        20.0,
        expected_elevation,
    )
    assert accepted.candidate.terrain_sample["source"] == "fixture-dem"
    assert accepted.candidate.terrain_sample["elevation"] == expected_elevation


def test_unavailable_required_terrain_sampler_returns_typed_diagnostic():
    from forge3d import LabelPlan

    plan = LabelPlan.compile(
        labels=[_terrain_label(id="missing-terrain")],
        camera={},
        viewport=(100, 100),
        terrain=None,
        seed=3,
    )

    assert plan.accepted == []
    assert len(plan.rejected) == 1
    assert plan.rejected[0].reason == "terrain_occluded"
    assert plan.rejected[0].details["terrain_sample"]["unavailable"] is True
    assert {diagnostic.code for diagnostic in plan.diagnostics} == {
        "label_rejection_summary",
        "placeholder_fallback",
    }


def test_terrain_visibility_rejects_occluded_label_with_sample_details():
    from forge3d import LabelPlan

    plan = LabelPlan.compile(
        labels=[_terrain_label()],
        camera={},
        viewport=(100, 100),
        terrain=_TerrainSampler(visible=False),
        seed=3,
    )

    assert plan.accepted == []
    assert len(plan.rejected) == 1
    assert plan.rejected[0].reason == "terrain_occluded"
    assert plan.rejected[0].candidate_id == "terrain-label:center"
    assert plan.rejected[0].details["terrain_sample"]["visible"] is False
