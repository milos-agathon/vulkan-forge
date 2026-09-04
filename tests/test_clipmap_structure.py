"""P2.1/M5: Clipmap structure tests.

Exit criteria from docs/roadmap/roadmap.md:530:
- [ ] Clipmap triangle budget verification (≥40% reduction at distance)
"""

import numpy as np


class TestClipmapConfig:
    """ClipmapConfig creation and validation tests."""

    def test_config_default_values(self):
        """Default config has expected values."""
        from forge3d import ClipmapConfig

        config = ClipmapConfig()
        assert config.ring_count == 4
        assert config.ring_resolution == 64
        assert config.center_resolution == 64
        assert config.skirt_depth == 10.0
        assert abs(config.morph_range - 0.3) < 1e-6  # Float comparison

    def test_config_custom_values(self):
        """Custom config preserves values."""
        from forge3d import ClipmapConfig

        config = ClipmapConfig(
            ring_count=6,
            ring_resolution=32,
            center_resolution=48,
            skirt_depth=20.0,
            morph_range=0.5,
        )
        assert config.ring_count == 6
        assert config.ring_resolution == 32
        assert config.center_resolution == 48
        assert config.skirt_depth == 20.0
        assert config.morph_range == 0.5

    def test_config_morph_range_clamped(self):
        """Morph range is clamped to [0.0, 1.0]."""
        from forge3d import ClipmapConfig

        config = ClipmapConfig(morph_range=1.5)
        assert config.morph_range == 1.0

        config = ClipmapConfig(morph_range=-0.5)
        assert config.morph_range == 0.0

    def test_config_rejects_zero_or_nonfinite_topology_inputs(self):
        """Invalid public inputs raise ValueError instead of panicking in Rust."""
        import pytest
        from forge3d import ClipmapConfig

        with pytest.raises(ValueError, match="must be positive"):
            ClipmapConfig(ring_resolution=0)
        with pytest.raises(ValueError, match="must be positive"):
            ClipmapConfig(center_resolution=0)
        with pytest.raises(ValueError, match="must be finite"):
            ClipmapConfig(morph_range=float("nan"))

    def test_config_repr(self):
        """Config has readable repr."""
        from forge3d import ClipmapConfig

        config = ClipmapConfig(ring_count=4, ring_resolution=64)
        repr_str = repr(config)
        assert "ClipmapConfig" in repr_str
        assert "ring_count=4" in repr_str


class TestClipmapMeshGeneration:
    """ClipmapMesh generation tests."""

    def test_generate_valid_mesh(self):
        """Clipmap generates mesh with correct vertex/index counts."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        assert mesh.vertex_count > 0
        assert mesh.index_count > 0
        assert mesh.index_count % 3 == 0  # All triangles

    def test_generate_rejects_nonfinite_or_empty_world_extent(self):
        """Mesh generation validates world geometry before entering Rust math."""
        import pytest
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig()
        with pytest.raises(ValueError, match="center must be finite"):
            clipmap_generate_py(config, (float("nan"), 0.0), 1000.0)
        with pytest.raises(ValueError, match="terrain_extent"):
            clipmap_generate_py(config, (0.0, 0.0), 0.0)

    def test_generate_at_different_centers(self):
        """Clipmap can be generated at different center positions."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)

        mesh1 = clipmap_generate_py(config, (0.0, 0.0), 1000.0)
        mesh2 = clipmap_generate_py(config, (100.0, 200.0), 1000.0)

        # Both should generate valid meshes
        assert mesh1.vertex_count > 0
        assert mesh2.vertex_count > 0

    def test_generate_different_extents(self):
        """Clipmap scales correctly with terrain extent."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)

        mesh_small = clipmap_generate_py(config, (0.0, 0.0), 100.0)
        mesh_large = clipmap_generate_py(config, (0.0, 0.0), 10000.0)

        # Vertex count should be similar (topology is the same)
        assert mesh_small.vertex_count == mesh_large.vertex_count

    def test_mesh_has_correct_ring_count(self):
        """Mesh reports correct number of rings."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        for ring_count in [2, 4, 6]:
            config = ClipmapConfig(ring_count=ring_count, ring_resolution=32)
            mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)
            assert mesh.rings_count == ring_count


class TestClipmapTriangleReduction:
    """P2.1 exit criteria: ≥40% triangle reduction at distance."""

    def test_triangle_reduction_meets_40_percent(self):
        """Triangle reduction meets P2.1 requirement of ≥40%."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=64)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        reduction = mesh.triangle_reduction_percent

        assert reduction >= 40.0, (
            f"Triangle reduction {reduction:.1f}% should be >= 40% "
            f"(P2.1 exit criteria)"
        )

    def test_more_rings_increases_reduction(self):
        """More LOD rings should generally increase triangle reduction."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config_few = ClipmapConfig(ring_count=2, ring_resolution=64)
        config_many = ClipmapConfig(ring_count=6, ring_resolution=64)

        mesh_few = clipmap_generate_py(config_few, (0.0, 0.0), 1000.0)
        mesh_many = clipmap_generate_py(config_many, (0.0, 0.0), 1000.0)

        # More rings should mean more coarse LOD area, thus more reduction
        # (or at least not significantly worse)
        assert mesh_many.triangle_reduction_percent >= mesh_few.triangle_reduction_percent * 0.8

    def test_calculate_triangle_reduction_function(self):
        """Standalone triangle reduction calculation works."""
        from forge3d import calculate_triangle_reduction_py

        # 1000 full-res, 500 clipmap = 50% reduction
        reduction = calculate_triangle_reduction_py(1000, 500)
        assert abs(reduction - 50.0) < 0.1

        # 1000 full-res, 1000 clipmap = 0% reduction
        reduction = calculate_triangle_reduction_py(1000, 1000)
        assert abs(reduction - 0.0) < 0.1

        # Edge case: 0 full-res
        reduction = calculate_triangle_reduction_py(0, 100)
        assert reduction == 0.0


class TestClipmapVertexData:
    """Clipmap vertex attribute tests."""

    def test_positions_numpy_array(self):
        """Positions returned as numpy array with correct shape."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        positions = mesh.positions()
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (mesh.vertex_count, 2)
        assert positions.dtype == np.float32

    def test_uvs_numpy_array(self):
        """UVs returned as numpy array with correct shape."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        uvs = mesh.uvs()
        assert isinstance(uvs, np.ndarray)
        assert uvs.shape == (mesh.vertex_count, 2)
        assert uvs.dtype == np.float32

    def test_uvs_in_valid_range(self):
        """UV coordinates are in [0, 1] range."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        uvs = mesh.uvs()
        assert uvs.min() >= 0.0
        assert uvs.max() <= 1.0

    def test_morph_data_numpy_array(self):
        """Morph data returned as numpy array with correct shape."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        morph_data = mesh.morph_data()
        assert isinstance(morph_data, np.ndarray)
        assert morph_data.shape == (mesh.vertex_count, 2)
        assert morph_data.dtype == np.float32

    def test_morph_weights_in_valid_range(self):
        """Non-skirt morph weights are in [0, 1] range."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        morph_data = mesh.morph_data()
        morph_weights = morph_data[:, 0]

        # Non-skirt vertices (weight >= 0) should be in [0, 1]
        non_skirt = morph_weights[morph_weights >= 0]
        assert non_skirt.min() >= 0.0
        assert non_skirt.max() <= 1.0

    def test_indices_numpy_array(self):
        """Indices returned as numpy array with correct properties."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        indices = mesh.indices()
        assert isinstance(indices, np.ndarray)
        assert len(indices) == mesh.index_count
        assert indices.dtype == np.uint32

    def test_indices_within_vertex_bounds(self):
        """All indices reference valid vertices."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        indices = mesh.indices()
        assert indices.min() >= 0
        assert indices.max() < mesh.vertex_count


class TestClipmapMeshRepr:
    """Clipmap mesh representation tests."""

    def test_mesh_repr(self):
        """Mesh has readable repr with key stats."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        repr_str = repr(mesh)
        assert "ClipmapMesh" in repr_str
        assert "vertices=" in repr_str
        assert "triangles=" in repr_str
        assert "reduction=" in repr_str
