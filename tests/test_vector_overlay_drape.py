# tests/test_vector_overlay_drape.py
# Unit tests for vector overlay draping logic
# Tests the Python API classes and validation

import pytest
from forge3d.terrain_params import (
    VectorVertex,
    VectorOverlayConfig,
    PrimitiveType,
)


class TestVectorVertex:
    """Test VectorVertex dataclass."""

    def test_create_default_color(self):
        """Default color should be white (1, 1, 1, 1)."""
        v = VectorVertex(x=100.0, y=0.0, z=100.0)
        assert v.r == 1.0
        assert v.g == 1.0
        assert v.b == 1.0
        assert v.a == 1.0

    def test_create_with_color(self):
        """Create vertex with explicit color."""
        v = VectorVertex(x=10.0, y=5.0, z=20.0, r=1.0, g=0.0, b=0.0, a=0.5)
        assert v.x == 10.0
        assert v.y == 5.0
        assert v.z == 20.0
        assert v.r == 1.0
        assert v.g == 0.0
        assert v.b == 0.0
        assert v.a == 0.5

    def test_to_array(self):
        """to_array should return [x, y, z, r, g, b, a, feature_id]."""
        v = VectorVertex(x=1.0, y=2.0, z=3.0, r=0.5, g=0.6, b=0.7, a=0.8)
        arr = v.to_array()
        assert arr == [1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.8, 0]

    def test_color_validation_r_out_of_range(self):
        """Color r must be in [0, 1]."""
        with pytest.raises(ValueError, match="r must be in"):
            VectorVertex(x=0, y=0, z=0, r=1.5)

    def test_color_validation_g_negative(self):
        """Color g must be >= 0."""
        with pytest.raises(ValueError, match="g must be in"):
            VectorVertex(x=0, y=0, z=0, g=-0.1)

    def test_color_validation_a_out_of_range(self):
        """Alpha must be in [0, 1]."""
        with pytest.raises(ValueError, match="a must be in"):
            VectorVertex(x=0, y=0, z=0, a=2.0)


class TestVectorOverlayConfig:
    """Test VectorOverlayConfig dataclass."""

    def test_create_simple_triangle(self):
        """Create a simple triangle overlay."""
        config = VectorOverlayConfig(
            name="marker",
            vertices=[
                VectorVertex(100, 0, 100, r=1, g=0, b=0),
                VectorVertex(200, 0, 100, r=0, g=1, b=0),
                VectorVertex(150, 0, 200, r=0, g=0, b=1),
            ],
            indices=[0, 1, 2],
            primitive=PrimitiveType.TRIANGLES,
        )
        assert config.name == "marker"
        assert config.vertex_count == 3
        assert config.index_count == 3
        assert config.primitive == PrimitiveType.TRIANGLES

    def test_default_values(self):
        """Default values should match plan specification."""
        config = VectorOverlayConfig(
            name="test",
            vertices=[VectorVertex(0, 0, 0)],
            indices=[0],
        )
        assert config.drape is False
        assert config.drape_offset == 0.5
        assert config.opacity == 1.0
        assert config.depth_bias == 0.1
        assert config.line_width == 2.0
        assert config.point_size == 5.0
        assert config.visible is True
        assert config.z_order == 0

    def test_create_with_draping(self):
        """Create overlay with draping enabled."""
        config = VectorOverlayConfig(
            name="draped_lines",
            vertices=[
                VectorVertex(0, 0, 0),
                VectorVertex(100, 0, 100),
            ],
            indices=[0, 1],
            primitive=PrimitiveType.LINES,
            drape=True,
            drape_offset=1.5,
        )
        assert config.drape is True
        assert config.drape_offset == 1.5
        assert config.primitive == PrimitiveType.LINES

    def test_to_ipc_dict(self):
        """to_ipc_dict should produce valid IPC request."""
        config = VectorOverlayConfig(
            name="test_overlay",
            vertices=[
                VectorVertex(10, 0, 20, r=1, g=0, b=0, a=1),
                VectorVertex(30, 0, 40, r=0, g=1, b=0, a=1),
            ],
            indices=[0, 1],
            primitive=PrimitiveType.LINES,
            drape=True,
            opacity=0.8,
        )
        d = config.to_ipc_dict()
        assert d["cmd"] == "add_vector_overlay"
        assert d["name"] == "test_overlay"
        assert d["primitive"] == "lines"
        assert d["drape"] is True
        assert d["opacity"] == 0.8
        assert len(d["vertices"]) == 2
        assert d["vertices"][0] == [10, 0, 20, 1, 0, 0, 1, 0]

    def test_validation_empty_name(self):
        """Name must be non-empty."""
        with pytest.raises(ValueError, match="name must be non-empty"):
            VectorOverlayConfig(
                name="",
                vertices=[VectorVertex(0, 0, 0)],
                indices=[0],
            )

    def test_validation_opacity_out_of_range(self):
        """Opacity must be in [0, 1]."""
        with pytest.raises(ValueError, match="opacity must be in"):
            VectorOverlayConfig(
                name="test",
                vertices=[VectorVertex(0, 0, 0)],
                indices=[0],
                opacity=1.5,
            )

    def test_validation_depth_bias_too_small(self):
        """Depth bias must be >= 0.01."""
        with pytest.raises(ValueError, match="depth_bias must be in"):
            VectorOverlayConfig(
                name="test",
                vertices=[VectorVertex(0, 0, 0)],
                indices=[0],
                depth_bias=0.001,
            )

    def test_validation_line_width_negative(self):
        """Line width must be >= 0.1."""
        with pytest.raises(ValueError, match="line_width must be"):
            VectorOverlayConfig(
                name="test",
                vertices=[VectorVertex(0, 0, 0)],
                indices=[0],
                primitive=PrimitiveType.LINES,
                line_width=0.05,
            )

    def test_validation_point_size_negative(self):
        """Point size must be >= 0.1."""
        with pytest.raises(ValueError, match="point_size must be"):
            VectorOverlayConfig(
                name="test",
                vertices=[VectorVertex(0, 0, 0)],
                indices=[0],
                primitive=PrimitiveType.POINTS,
                point_size=0.0,
            )
    
    def test_large_line_width_allowed(self):
        """Large line widths allowed for world-unit triangle quads."""
        config = VectorOverlayConfig(
            name="wide_lines",
            vertices=[VectorVertex(0, 0, 0)],
            indices=[0],
            primitive=PrimitiveType.TRIANGLES,
            line_width=50.0,
        )
        assert config.line_width == 50.0


class TestPrimitiveType:
    """Test PrimitiveType enum."""

    def test_all_primitive_types(self):
        """All primitive types should have correct string values."""
        assert PrimitiveType.POINTS.value == "points"
        assert PrimitiveType.LINES.value == "lines"
        assert PrimitiveType.LINE_STRIP.value == "line_strip"
        assert PrimitiveType.TRIANGLES.value == "triangles"
        assert PrimitiveType.TRIANGLE_STRIP.value == "triangle_strip"

    def test_use_in_config(self):
        """Primitive types should work in VectorOverlayConfig."""
        for ptype in PrimitiveType:
            config = VectorOverlayConfig(
                name=f"test_{ptype.value}",
                vertices=[VectorVertex(0, 0, 0)],
                indices=[0],
                primitive=ptype,
            )
            assert config.primitive == ptype


class TestDrapeVerticesFlatTerrain:
    """Test draping behavior documentation - draping is done in Rust."""

    def test_drape_vertices_flat_terrain_doc(self):
        """Document expected behavior: on flat terrain, Y = drape_offset.
        
        The actual draping is performed in Rust (src/viewer/terrain/vector_overlay.rs).
        This test documents the expected behavior:
        - When drape=True and terrain is flat (all heights = 0)
        - Vertex Y should be set to drape_offset
        """
        # Config specifies drape intent
        config = VectorOverlayConfig(
            name="flat_test",
            vertices=[
                VectorVertex(100, 0, 100),  # Y=0, will be replaced by drape
                VectorVertex(200, 0, 200),
            ],
            indices=[0, 1],
            primitive=PrimitiveType.LINES,
            drape=True,
            drape_offset=0.5,
        )
        # Draping intent is captured in config
        assert config.drape is True
        assert config.drape_offset == 0.5
