"""TV13 — Terrain Population LOD Pipeline tests.

Covers:
  - TV13.1: QEM mesh simplification (Rust) and Python wrapper
  - TV13.2: LOD chain generation and auto_lod_levels
  - TV13.3: HLOD clustering, rendering, stats, and memory tracking
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE
from forge3d.terrain_params import make_terrain_params_config
from _terrain_runtime import terrain_rendering_available

if not NATIVE_AVAILABLE:
    pytest.skip(
        "TV13 tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

ts = f3d.terrain_scatter

from forge3d.geometry import MeshBuffers, primitive_mesh, simplify_mesh, generate_lod_chain
from forge3d.terrain_scatter import (
    HLODPolicy,
    TerrainScatterBatch,
    TerrainScatterLevel,
    auto_lod_levels,
)


class TestSimplifyMesh:
    """TV13.1 — Python simplify_mesh wrapper."""

    def test_simplify_cone_reduces_triangles(self):
        cone = primitive_mesh("cone", radial_segments=32)
        simplified = simplify_mesh(cone, 0.5)
        assert simplified.triangle_count < cone.triangle_count
        assert simplified.vertex_count > 0

    def test_simplify_preserves_normals(self):
        sphere = primitive_mesh("sphere", rings=12, radial_segments=24)
        simplified = simplify_mesh(sphere, 0.5)
        assert simplified.normals.shape[0] == simplified.positions.shape[0]

    def test_simplify_ratio_one_unchanged(self):
        box_mesh = primitive_mesh("box")
        result = simplify_mesh(box_mesh, 1.0)
        assert result.triangle_count == box_mesh.triangle_count

    @pytest.mark.parametrize("ratio", [0.5, 1.0])
    def test_simplify_rejects_out_of_bounds_indices(self, ratio):
        mesh = MeshBuffers(
            positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            normals=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32),
            uvs=np.zeros((3, 2), dtype=np.float32),
            indices=np.array([[0, 1, 99]], dtype=np.uint32),
        )
        with pytest.raises(ValueError, match="out of bounds"):
            simplify_mesh(mesh, ratio)


class TestGenerateLodChain:
    """TV13.1 — LOD chain generation from a single mesh."""

    def test_three_level_chain_decreasing_triangles(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        chain = generate_lod_chain(sphere, [1.0, 0.25, 0.07])
        assert len(chain) >= 2  # at least LOD0 + one simplified
        assert chain[0].triangle_count == sphere.triangle_count
        for i in range(1, len(chain)):
            assert chain[i].triangle_count < chain[i - 1].triangle_count

    def test_min_triangles_floor_drops_levels(self):
        box_mesh = primitive_mesh("box")  # 12 tris
        chain = generate_lod_chain(box_mesh, [1.0, 0.01], min_triangles=8)
        assert len(chain) == 1  # only LOD 0 survives

    def test_deduplication_drops_identical_levels(self):
        box_mesh = primitive_mesh("box")  # 12 tris
        chain = generate_lod_chain(box_mesh, [1.0, 0.9, 0.8])
        for i in range(1, len(chain)):
            assert chain[i].triangle_count < chain[i - 1].triangle_count

    def test_ratios_must_start_with_one(self):
        sphere = primitive_mesh("sphere")
        with pytest.raises(ValueError, match="ratios.*1.0"):
            generate_lod_chain(sphere, [0.5, 0.25])

    def test_ratios_must_be_descending(self):
        sphere = primitive_mesh("sphere")
        with pytest.raises(ValueError, match="descending"):
            generate_lod_chain(sphere, [1.0, 0.5, 0.7])


class TestAutoLodLevels:
    """TV13.2 — auto_lod_levels generates scatter LOD levels from one mesh."""

    def test_default_three_levels(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(sphere, lod_count=3, draw_distance=300.0)
        assert len(levels) >= 2
        assert all(isinstance(l, TerrainScatterLevel) for l in levels)
        assert levels[-1].max_distance is None
        if len(levels) >= 2:
            assert levels[0].max_distance is not None

    def test_explicit_distances(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(
            sphere,
            lod_count=3,
            lod_distances=[50.0, 150.0, None],
        )
        assert levels[0].max_distance == 50.0
        assert levels[1].max_distance == 150.0
        assert levels[-1].max_distance is None

    def test_explicit_ratios(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(
            sphere,
            lod_count=3,
            ratios=[1.0, 0.3, 0.05],
            draw_distance=200.0,
        )
        assert levels[0].mesh.triangle_count >= levels[1].mesh.triangle_count

    def test_lod_count_six_works(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(sphere, lod_count=6, draw_distance=600.0)
        assert len(levels) >= 2, f"Expected at least 2 levels, got {len(levels)}"
        # Triangle counts should decrease across levels
        for i in range(1, len(levels)):
            assert levels[i].mesh.triangle_count <= levels[i - 1].mesh.triangle_count
        # Last level should be open-ended
        assert levels[-1].max_distance is None

    def test_feeds_into_scatter_batch(self):
        cone = primitive_mesh("cone", radial_segments=32)
        levels = auto_lod_levels(cone, lod_count=2, draw_distance=100.0)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        batch = TerrainScatterBatch(
            levels=levels,
            transforms=transforms,
            name="auto_lod_test",
        )
        assert batch.instance_count == 1


class TestHLODPolicy:
    """TV13.3 — HLODPolicy dataclass and serialization."""

    def test_hlod_policy_creation(self):
        policy = HLODPolicy(hlod_distance=200.0, cluster_radius=50.0)
        assert policy.hlod_distance == 200.0
        assert policy.cluster_radius == 50.0
        assert policy.simplify_ratio == 0.1

    def test_hlod_in_native_dict(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        policy = HLODPolicy(hlod_distance=100.0, cluster_radius=30.0, simplify_ratio=0.2)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=cone)],
            transforms=transforms,
            hlod=policy,
            max_draw_distance=500.0,
        )
        d = batch.to_native_dict()
        assert "hlod" in d
        assert d["hlod"]["hlod_distance"] == 100.0
        assert d["hlod"]["cluster_radius"] == 30.0
        assert d["hlod"]["simplify_ratio"] == 0.2

    def test_hlod_none_omitted_from_dict(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=cone)],
            transforms=transforms,
        )
        d = batch.to_native_dict()
        assert d.get("hlod") is None

    def test_hlod_in_viewer_payload(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        policy = HLODPolicy(hlod_distance=100.0, cluster_radius=30.0)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=cone)],
            transforms=transforms,
            hlod=policy,
            max_draw_distance=500.0,
        )
        payload = batch.to_viewer_payload()
        assert "hlod" in payload
        assert payload["hlod"]["hlod_distance"] == 100.0

    def test_hlod_validation_rejects_bad_params(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        with pytest.raises(ValueError, match="hlod_distance"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=cone)],
                transforms=transforms,
                hlod=HLODPolicy(hlod_distance=500.0, cluster_radius=30.0),
                max_draw_distance=200.0,
            )

    def test_hlod_rejects_negative_cluster_radius(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        with pytest.raises(ValueError, match="cluster_radius"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=cone)],
                transforms=transforms,
                hlod=HLODPolicy(hlod_distance=50.0, cluster_radius=-10.0),
                max_draw_distance=200.0,
            )

    def test_hlod_rejects_invalid_simplify_ratio(self):
        cone = primitive_mesh("cone", radial_segments=16)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        with pytest.raises(ValueError, match="simplify_ratio"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=cone)],
                transforms=transforms,
                hlod=HLODPolicy(hlod_distance=50.0, cluster_radius=30.0, simplify_ratio=0.0),
                max_draw_distance=200.0,
            )


@pytest.mark.apple_metal_physical
class TestHLODRendering:
    """TV13.3 — HLOD rendering integration tests (require GPU)."""

    @pytest.fixture
    def gpu_session(self):
        # `has_gpu()` is true for hosted WARP/paravirtual adapters that cannot
        # complete this terrain readback.  Require the same terrain-safe probe
        # used by the visual lanes so those environments skip honestly.
        if not terrain_rendering_available():
            pytest.skip("terrain-safe GPU not available")
        return f3d.Session(window=False)

    def _create_test_hdr(self, path):
        with open(path, "wb") as fh:
            fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n")
            fh.write(bytes([128, 128, 180, 128] * 32))

    def _build_render_context(self, gpu_session):
        heightmap = np.sin(np.mgrid[0:96, 0:96][1].astype(np.float32) / 7.0) * 8.0 \
            + np.cos(np.mgrid[0:96, 0:96][0].astype(np.float32) / 9.0) * 6.0 + 25.0
        heightmap = heightmap.astype(np.float32)

        renderer = f3d.TerrainRenderer(gpu_session)
        material_set = f3d.MaterialSet.terrain_default()

        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            hdr_path = tmp.name
        try:
            self._create_test_hdr(hdr_path)
            ibl = f3d.IBL.from_hdr(hdr_path, intensity=1.0)
        finally:
            os.unlink(hdr_path)

        config = make_terrain_params_config(
            size_px=(256, 160),
            render_scale=1.0,
            terrain_span=180.0,
            msaa_samples=4,
            z_scale=1.4,
            exposure=1.0,
            domain=(float(np.min(heightmap)), float(np.max(heightmap))),
            cam_radius=220.0,
            cam_phi_deg=138.0,
            cam_theta_deg=57.0,
            fov_y_deg=48.0,
        )
        params = f3d.TerrainRenderParams(config)
        return renderer, material_set, ibl, params, heightmap

    def _build_dense_scatter(self, hlod=None):
        cone = primitive_mesh("cone", radial_segments=16)
        levels = auto_lod_levels(cone, lod_count=2, draw_distance=200.0)
        transforms = []
        for x in range(10):
            for z in range(10):
                transforms.append([
                    1, 0, 0, float(x * 5),
                    0, 1, 0, 0,
                    0, 0, 1, float(z * 5),
                    0, 0, 0, 1,
                ])
        transforms = np.array(transforms, dtype=np.float32)
        return TerrainScatterBatch(
            levels=levels,
            transforms=transforms,
            name="dense_scatter",
            max_draw_distance=500.0,
            hlod=hlod,
        )

    def test_hlod_none_preserves_baseline(self, gpu_session):
        renderer, material_set, ibl, params, heightmap = self._build_render_context(gpu_session)
        batch = self._build_dense_scatter(hlod=None)
        ts.apply_to_renderer(renderer, [batch])
        renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap)
        stats = renderer.get_scatter_stats()
        assert stats["hlod_cluster_draws"] == 0
        assert stats["hlod_covered_instances"] == 0
        assert stats["effective_draws"] > 0

    def test_hlod_renders_and_reports_stats(self, gpu_session):
        renderer, material_set, ibl, params, heightmap = self._build_render_context(gpu_session)
        policy = HLODPolicy(hlod_distance=50.0, cluster_radius=15.0, simplify_ratio=0.1)
        batch = self._build_dense_scatter(hlod=policy)
        ts.apply_to_renderer(renderer, [batch])
        renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap)
        stats = renderer.get_scatter_stats()
        # Camera at distance 220, most instances beyond hlod_distance 50
        assert stats["hlod_cluster_draws"] > 0
        assert stats["hlod_covered_instances"] > 0
        assert stats["effective_draws"] > 0

    def test_hlod_respects_mesh_extents_and_scale(self, gpu_session):
        """Large-scale instances should have a large effective cluster radius,
        preventing HLOD activation when geometry is still close to camera."""
        renderer, material_set, ibl, params, heightmap = self._build_render_context(gpu_session)
        box_mesh = primitive_mesh("box")
        levels = [TerrainScatterLevel(mesh=box_mesh)]
        # Two boxes at scale 100 placed close together
        transforms = np.array([
            [100, 0, 0, 10, 0, 100, 0, 0, 0, 0, 100, 10, 0, 0, 0, 1],
            [100, 0, 0, 15, 0, 100, 0, 0, 0, 0, 100, 15, 0, 0, 0, 1],
        ], dtype=np.float32)
        # hlod_distance=50, but boxes extend ~86 units from origin at scale 100
        # so the cluster's effective radius should keep HLOD inactive
        policy = HLODPolicy(hlod_distance=50.0, cluster_radius=20.0, simplify_ratio=0.5)
        batch = TerrainScatterBatch(
            levels=levels, transforms=transforms, name="large_scale",
            max_draw_distance=1000.0, hlod=policy,
        )
        ts.apply_to_renderer(renderer, [batch])
        renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap)
        stats = renderer.get_scatter_stats()
        # With camera at ~220 and cluster radius >> hlod_distance,
        # HLOD should NOT activate because geometry extends well past the threshold
        assert stats["hlod_cluster_draws"] == 0, (
            f"HLOD should not activate for 100x-scale instances near camera, "
            f"got {stats['hlod_cluster_draws']} cluster draws"
        )
        assert stats["hlod_covered_instances"] == 0

    def test_hlod_memory_tracked(self, gpu_session):
        renderer, _, _, _, _ = self._build_render_context(gpu_session)
        policy = HLODPolicy(hlod_distance=50.0, cluster_radius=15.0, simplify_ratio=0.1)
        batch = self._build_dense_scatter(hlod=policy)
        ts.apply_to_renderer(renderer, [batch])
        report = renderer.get_scatter_memory_report()
        assert report["hlod_cluster_count"] > 0
        assert report["hlod_buffer_bytes"] > 0
        assert report["total_buffer_bytes"] >= report["hlod_buffer_bytes"]


@pytest.mark.apple_metal_physical
class TestEndToEndImageOutput:
    """TV13 end-to-end: auto-LOD scatter renders to PNG with real content."""

    @pytest.fixture
    def gpu_session(self):
        # `has_gpu()` is true for hosted WARP/paravirtual adapters that cannot
        # complete this terrain readback.  Require the same terrain-safe probe
        # used by the visual lanes so those environments skip honestly.
        if not terrain_rendering_available():
            pytest.skip("terrain-safe GPU not available")
        return f3d.Session(window=False)

    def test_auto_lod_scatter_produces_nonempty_image(self, gpu_session):
        from pathlib import Path

        heightmap = np.random.default_rng(42).uniform(0, 100, (64, 64)).astype(np.float32)
        renderer = f3d.TerrainRenderer(gpu_session)
        material_set = f3d.MaterialSet.terrain_default()

        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            hdr_path = tmp.name
        with open(hdr_path, "wb") as fh:
            fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n")
            fh.write(bytes([128, 128, 180, 128] * 32))
        try:
            ibl = f3d.IBL.from_hdr(hdr_path, intensity=1.0)
        finally:
            os.unlink(hdr_path)

        config = make_terrain_params_config(
            size_px=(256, 160),
            render_scale=1.0,
            terrain_span=64.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(float(heightmap.min()), float(heightmap.max())),
            camera_mode="mesh",
            cam_radius=80.0,
        )
        params = f3d.TerrainRenderParams(config)

        cone = primitive_mesh("cone", radial_segments=16)
        levels = auto_lod_levels(cone, lod_count=2, draw_distance=50.0)
        transforms = np.array([
            [1, 0, 0, 20, 0, 1, 0, 50, 0, 0, 1, 20, 0, 0, 0, 1],
            [1, 0, 0, 40, 0, 1, 0, 50, 0, 0, 1, 40, 0, 0, 0, 1],
        ], dtype=np.float32)

        batch = TerrainScatterBatch(
            levels=levels, transforms=transforms, name="e2e_test",
            color=(0.8, 0.2, 0.2, 1.0), max_draw_distance=100.0,
        )
        ts.apply_to_renderer(renderer, [batch])

        frame = renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            png_path = tmp.name
        frame.save(png_path)
        try:
            assert Path(png_path).stat().st_size > 100, "PNG should be non-trivial"
        finally:
            Path(png_path).unlink(missing_ok=True)
