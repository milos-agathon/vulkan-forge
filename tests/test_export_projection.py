"""Tests for vector export projection utilities (P5-export).

Tests:
- 3D to 2D projection with view-projection matrix
- 2D to screen coordinate mapping
- Bounds calculation from geometry
- Behind-camera point rejection
"""

import pytest
import numpy as np

# Import Rust export module if available, otherwise use Python fallback
try:
    from forge3d._forge3d import (
        project_3d_to_2d as rust_project_3d,
        project_2d_to_screen as rust_project_2d,
    )
    HAS_RUST_EXPORT = True
except ImportError:
    HAS_RUST_EXPORT = False

from forge3d.export import Bounds


class TestPythonProjection:
    """Test Python-side projection utilities."""

    def test_2d_to_screen_corners(self):
        """Test 2D to screen projection at corners."""
        from forge3d.export import _project_to_screen

        bounds = Bounds(min_x=0, min_y=0, max_x=100, max_y=100)
        width, height = 800, 600

        # Bottom-left corner
        x, y = _project_to_screen(0, 0, bounds, width, height)
        assert abs(x - 0) < 0.01
        assert abs(y - 600) < 0.01  # Y flipped

        # Top-right corner
        x, y = _project_to_screen(100, 100, bounds, width, height)
        assert abs(x - 800) < 0.01
        assert abs(y - 0) < 0.01  # Y flipped

        # Center
        x, y = _project_to_screen(50, 50, bounds, width, height)
        assert abs(x - 400) < 0.01
        assert abs(y - 300) < 0.01

    def test_2d_to_screen_aspect_ratio(self):
        """Test projection preserves aspect ratio mapping."""
        from forge3d.export import _project_to_screen

        bounds = Bounds(min_x=0, min_y=0, max_x=200, max_y=100)  # 2:1 aspect
        width, height = 800, 400  # 2:1 aspect

        # Quarter point
        x, y = _project_to_screen(50, 25, bounds, width, height)
        assert abs(x - 200) < 0.01
        assert abs(y - 300) < 0.01

    def test_2d_to_screen_negative_coords(self):
        """Test projection with negative coordinates."""
        from forge3d.export import _project_to_screen

        bounds = Bounds(min_x=-100, min_y=-100, max_x=100, max_y=100)
        width, height = 400, 400

        # Origin maps to center
        x, y = _project_to_screen(0, 0, bounds, width, height)
        assert abs(x - 200) < 0.01
        assert abs(y - 200) < 0.01

        # Min corner
        x, y = _project_to_screen(-100, -100, bounds, width, height)
        assert abs(x - 0) < 0.01
        assert abs(y - 400) < 0.01

    def test_2d_to_screen_degenerate_bounds(self):
        """Test projection with zero-width or zero-height bounds."""
        from forge3d.export import _project_to_screen

        # Zero-width bounds (vertical line)
        bounds = Bounds(min_x=50, min_y=0, max_x=50, max_y=100)
        width, height = 100, 100

        x, y = _project_to_screen(50, 50, bounds, width, height)
        # Should not crash, x should be normalized somehow
        assert not np.isnan(x)
        assert not np.isnan(y)


class TestBoundsCalculation:
    """Test bounds calculation utilities."""

    def test_bounds_from_polygons_and_lines(self):
        """Test bounds from mixed geometry."""
        from forge3d.export import VectorScene

        scene = VectorScene()
        scene.add_polygon(exterior=[(10, 20), (30, 20), (20, 40)])
        scene.add_polyline(path=[(0, 0), (50, 50)])

        bounds = scene.compute_bounds()

        assert bounds.min_x == 0
        assert bounds.min_y == 0
        assert bounds.max_x == 50
        assert bounds.max_y == 50

    def test_bounds_with_labels(self):
        """Test bounds include label positions."""
        from forge3d.export import VectorScene

        scene = VectorScene()
        scene.add_polygon(exterior=[(0, 0), (10, 0), (5, 10)])
        scene.add_label(text="Far Label", position=(100, 100))

        bounds = scene.compute_bounds()

        assert bounds.max_x == 100
        assert bounds.max_y == 100

    def test_bounds_empty_scene(self):
        """Test bounds of empty scene."""
        from forge3d.export import VectorScene

        scene = VectorScene()
        bounds = scene.compute_bounds()

        # Default bounds
        assert bounds is not None


class TestProjectionIntegration:
    """Integration tests for projection pipeline."""

    def test_round_trip_coordinates(self):
        """Test that projection produces valid screen coordinates."""
        from forge3d.export import VectorScene, generate_svg
        import xml.etree.ElementTree as ET

        # Create scene with known coordinates
        scene = VectorScene()
        scene.add_polygon(exterior=[(0, 0), (100, 0), (50, 100)])

        bounds = Bounds(min_x=0, min_y=0, max_x=100, max_y=100)
        svg = generate_svg(scene, width=100, height=100, bounds=bounds)

        # Parse SVG and extract polygon points
        root = ET.fromstring(svg)
        # Find polygon element
        polygon = root.find('.//{http://www.w3.org/2000/svg}polygon')
        if polygon is None:
            polygon = root.find('.//polygon')

        assert polygon is not None
        points_str = polygon.get('points')

        # Parse points
        points = []
        for pair in points_str.split():
            x, y = pair.split(',')
            points.append((float(x), float(y)))

        # Verify coordinates are within viewport
        for x, y in points:
            assert 0 <= x <= 100, f"x={x} outside viewport"
            assert 0 <= y <= 100, f"y={y} outside viewport"

    def test_coordinate_precision_loss(self):
        """Test that precision setting affects output."""
        from forge3d.export import VectorScene, generate_svg

        scene = VectorScene()
        scene.add_polygon(exterior=[(0.123456789, 0.123456789), (1, 0), (0.5, 1)])

        bounds = Bounds(min_x=0, min_y=0, max_x=1, max_y=1)

        # High precision
        svg_high = generate_svg(scene, precision=6, bounds=bounds, width=1, height=1)
        # Low precision
        svg_low = generate_svg(scene, precision=1, bounds=bounds, width=1, height=1)

        # High precision should have more decimal places
        assert len(svg_high) > len(svg_low)


@pytest.mark.skipif(not HAS_RUST_EXPORT, reason="Rust export module not available")
class TestRustProjection:
    """Test Rust-side projection (when available)."""

    def test_rust_3d_to_2d_center(self):
        """Test 3D to 2D projection of center point."""
        identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        assert rust_project_3d((0.0, 0.0, 0.0), identity, (800, 600)) == pytest.approx(
            (400.0, 300.0)
        )

    def test_rust_3d_behind_camera(self):
        """Test behind-camera rejection and the paired Rust 2D export."""
        behind_camera = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, -1.0,
        ]
        assert rust_project_3d(
            (0.0, 0.0, 0.0), behind_camera, (800, 600)
        ) is None
        assert rust_project_2d(
            (50.0, 25.0), (0.0, 0.0, 100.0, 100.0), (800, 600)
        ) == pytest.approx((400.0, 450.0))


class TestProjectionEdgeCases:
    """Test edge cases in projection."""

    def test_very_small_bounds(self):
        """Test projection with very small coordinate range."""
        from forge3d.export import _project_to_screen

        bounds = Bounds(min_x=0, min_y=0, max_x=1e-10, max_y=1e-10)
        width, height = 100, 100

        # Should not produce NaN or Inf
        x, y = _project_to_screen(0, 0, bounds, width, height)
        assert np.isfinite(x)
        assert np.isfinite(y)

    def test_very_large_bounds(self):
        """Test projection with very large coordinate range."""
        from forge3d.export import _project_to_screen

        bounds = Bounds(min_x=-1e10, min_y=-1e10, max_x=1e10, max_y=1e10)
        width, height = 100, 100

        # Center should project to center
        x, y = _project_to_screen(0, 0, bounds, width, height)
        assert abs(x - 50) < 0.01
        assert abs(y - 50) < 0.01

    def test_viewport_one_pixel(self):
        """Test projection to 1x1 viewport."""
        from forge3d.export import _project_to_screen

        bounds = Bounds(min_x=0, min_y=0, max_x=100, max_y=100)
        width, height = 1, 1

        # All points should map to (0-1, 0-1) range
        x, y = _project_to_screen(50, 50, bounds, width, height)
        assert 0 <= x <= 1
        assert 0 <= y <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
