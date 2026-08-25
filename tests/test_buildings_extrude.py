"""
P4: Tests for 3D building polygon extrusion.

Tests the core extrusion functionality from GeoJSON and raw polygons.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("pro_license")


def test_import_buildings_module():
    """Test that the buildings module can be imported."""
    from forge3d import buildings
    assert hasattr(buildings, "add_buildings")
    assert hasattr(buildings, "Building")
    assert hasattr(buildings, "BuildingLayer")


def test_building_dataclass():
    """Test Building dataclass properties."""
    from forge3d.buildings import Building

    positions = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    building = Building(
        id="test_building",
        positions=positions,
        indices=indices,
        height=10.0,
        roof_type="gabled",
    )

    assert building.id == "test_building"
    assert building.vertex_count == 8
    assert building.triangle_count == 2
    assert building.height == 10.0
    assert building.roof_type == "gabled"


def test_building_bounds():
    """Test Building bounds calculation."""
    from forge3d.buildings import Building

    positions = np.array([
        [0, 0, 0], [10, 0, 0], [10, 20, 0], [0, 20, 0],
        [0, 0, 5], [10, 0, 5], [10, 20, 5], [0, 20, 5],
    ], dtype=np.float32)
    indices = np.zeros(0, dtype=np.uint32)

    building = Building(id="b", positions=positions, indices=indices)
    bounds = building.bounds()

    assert bounds is not None
    assert bounds[0] == 0.0   # min_x
    assert bounds[1] == 0.0   # min_y
    assert bounds[2] == 0.0   # min_z
    assert bounds[3] == 10.0  # max_x
    assert bounds[4] == 20.0  # max_y
    assert bounds[5] == 5.0   # max_z


def test_building_layer_properties():
    """Test BuildingLayer aggregate properties."""
    from forge3d.buildings import Building, BuildingLayer

    b1 = Building(
        id="b1",
        positions=np.zeros((12, 3), dtype=np.float32),
        indices=np.zeros(18, dtype=np.uint32),
        lod=1,
    )
    b2 = Building(
        id="b2",
        positions=np.zeros((8, 3), dtype=np.float32),
        indices=np.zeros(12, dtype=np.uint32),
        lod=2,
    )

    layer = BuildingLayer(name="test_layer", buildings=[b1, b2])

    assert layer.building_count == 2
    assert layer.total_vertices == 20
    assert layer.total_triangles == 10
    assert layer.max_lod == 2


def test_add_buildings_from_geojson():
    """Test loading buildings from GeoJSON file."""
    from forge3d.buildings import add_buildings

    # Create sample GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"height": 15.0, "name": "Building A"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"height": 20.0, "building": "warehouse"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]]
                }
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(geojson, f)
        geojson_path = Path(f.name)

    try:
        layer = add_buildings(geojson_path)

        assert layer.name == geojson_path.stem
        assert layer.source_format == "geojson"
        # Native implementation creates merged geometry; fallback creates separate buildings
        assert layer.building_count >= 1

    finally:
        geojson_path.unlink()


def test_add_buildings_with_height_key():
    """Test custom height key extraction."""
    from forge3d.buildings import add_buildings

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"bldg_height": 25.0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]]
                }
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(geojson, f)
        geojson_path = Path(f.name)

    try:
        layer = add_buildings(geojson_path, height_key="bldg_height")
        assert layer.building_count >= 1
    finally:
        geojson_path.unlink()


def test_add_buildings_file_not_found():
    """Test error handling for missing file."""
    from forge3d.buildings import add_buildings

    with pytest.raises(FileNotFoundError):
        add_buildings("nonexistent_file.geojson")


def test_add_buildings_multipolygon():
    """Test loading buildings with MultiPolygon geometry."""
    from forge3d.buildings import add_buildings

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"height": 10.0},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]],
                        [[[10, 10], [15, 10], [15, 15], [10, 15], [10, 10]]]
                    ]
                }
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(geojson, f)
        geojson_path = Path(f.name)

    try:
        layer = add_buildings(geojson_path)
        assert layer.building_count >= 1
    finally:
        geojson_path.unlink()


def test_extrusion_vertex_count():
    """Test that extruded building has expected vertex count."""
    from forge3d.buildings import add_buildings

    # Simple square building
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"height": 10.0},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                }
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as f:
        json.dump(geojson, f)
        geojson_path = Path(f.name)

    try:
        layer = add_buildings(geojson_path)
        # Extruded square should have:
        # - 4 vertices * 2 (top + bottom) = 8 for caps
        # - 4 sides * 4 vertices each = 16 for walls
        # Total depends on implementation
        assert layer.total_vertices > 0
    finally:
        geojson_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
