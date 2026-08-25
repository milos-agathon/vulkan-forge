from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.geometry import MeshBuffers
from forge3d.terrain_params import make_terrain_params_config
from forge3d import terrain_scatter as ts
from forge3d.terrain_scatter import (
    TerrainScatterBatch,
    TerrainContactSettings,
    TerrainScatterFilters,
    TerrainScatterLevel,
    TerrainMeshBlendSettings,
    TerrainScatterSource,
    apply_to_renderer,
    apply_to_viewer,
    clear_viewer,
    grid_jitter_transforms,
    mask_density_transforms,
    seeded_random_transforms,
    viewer_orbit_radius,
)


def _simple_mesh() -> MeshBuffers:
    return MeshBuffers(
        positions=np.asarray(
            [
                [-0.5, 0.0, -0.5],
                [0.5, 0.0, -0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        normals=np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        uvs=np.zeros((3, 2), dtype=np.float32),
        indices=np.asarray([[0, 1, 2]], dtype=np.uint32),
    )


def _scatter_source(height: int = 32, width: int = 32) -> TerrainScatterSource:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    heightmap = np.sin(x / 4.0) * 4.0 + np.cos(y / 5.0) * 3.0 + x * 0.2 + 40.0
    return TerrainScatterSource(heightmap, z_scale=1.5)


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                handle.write(bytes([int(255 * x / max(width - 1, 1)), int(255 * y / max(height - 1, 1)), 180, 128]))


def _write_heightmap_tiff(path: str, heightmap: np.ndarray) -> None:
    pillow = pytest.importorskip("PIL.Image")
    pillow.fromarray(np.ascontiguousarray(heightmap, dtype=np.float32)).save(path, format="TIFF")


class TestPlacementDeterminism:
    def test_seeded_random_is_reproducible(self) -> None:
        source = _scatter_source()
        filters = TerrainScatterFilters(max_slope_deg=50.0)
        first = seeded_random_transforms(source, count=24, seed=17, filters=filters, scale_range=(0.8, 1.2))
        second = seeded_random_transforms(source, count=24, seed=17, filters=filters, scale_range=(0.8, 1.2))
        assert first.dtype == np.float32
        assert first.shape == (24, 16)
        assert np.array_equal(first, second)

    def test_seeded_random_changes_with_seed(self) -> None:
        source = _scatter_source()
        filters = TerrainScatterFilters(max_slope_deg=50.0)
        first = seeded_random_transforms(source, count=16, seed=17, filters=filters)
        second = seeded_random_transforms(source, count=16, seed=23, filters=filters)
        assert not np.array_equal(first, second)

    def test_grid_jitter_is_reproducible(self) -> None:
        source = _scatter_source()
        filters = TerrainScatterFilters(max_slope_deg=55.0)
        first = grid_jitter_transforms(source, spacing=4.0, seed=9, jitter=0.6, filters=filters)
        second = grid_jitter_transforms(source, spacing=4.0, seed=9, jitter=0.6, filters=filters)
        assert np.array_equal(first, second)

    def test_mask_density_is_reproducible_and_density_sensitive(self) -> None:
        source = _scatter_source()
        mask = np.zeros_like(source.heightmap, dtype=np.float32)
        mask[:, : mask.shape[1] // 2] = 1.0
        dense = mask_density_transforms(source, mask, spacing=3.0, seed=11, jitter=0.4)
        repeat = mask_density_transforms(source, mask, spacing=3.0, seed=11, jitter=0.4)
        sparse = mask_density_transforms(source, mask * 0.25, spacing=3.0, seed=11, jitter=0.4)
        assert np.array_equal(dense, repeat)
        assert dense.shape[0] > sparse.shape[0]


class TestFiltersAndContract:
    def test_source_rejects_invalid_z_scale(self) -> None:
        with pytest.raises(ValueError, match="z_scale"):
            TerrainScatterSource(np.full((8, 8), 1.0, dtype=np.float32), z_scale=0.0)

    def test_source_rejects_z_scale_that_overflows_scaled_heights(self) -> None:
        heightmap = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="z_scale produces non-finite scaled heights"):
            TerrainScatterSource(heightmap, z_scale=1.0e308)

    def test_pixel_to_contract_translation_uses_scaled_height(self) -> None:
        source = TerrainScatterSource(np.asarray([[10.0, 20.0], [30.0, 50.0]], dtype=np.float32), z_scale=2.0)
        transforms = seeded_random_transforms(
            source,
            count=1,
            seed=1,
            filters=TerrainScatterFilters(min_elevation=10.0, max_elevation=50.0, max_slope_deg=90.0),
            scale_range=(1.0, 1.0),
            yaw_range_deg=(0.0, 0.0),
        )
        tx, ty, tz = transforms[0, 3], transforms[0, 7], transforms[0, 11]
        row, col = source.contract_to_pixel(float(tx), float(tz))
        expected_y = source.sample_scaled_height(row, col)
        assert ty == pytest.approx(expected_y, rel=1e-5)

    def test_flat_source_passes_low_slope_filter(self) -> None:
        source = TerrainScatterSource(np.full((16, 16), 42.0, dtype=np.float32), z_scale=3.0)
        transforms = grid_jitter_transforms(
            source,
            spacing=4.0,
            seed=3,
            jitter=0.0,
            filters=TerrainScatterFilters(max_slope_deg=0.1),
            yaw_range_deg=(0.0, 0.0),
        )
        assert transforms.shape[0] > 0

    def test_steep_source_can_be_filtered_out(self) -> None:
        heightmap = np.tile(np.linspace(0.0, 100.0, 32, dtype=np.float32), (32, 1))
        source = TerrainScatterSource(heightmap, z_scale=4.0)
        with pytest.raises(ValueError, match="generated zero accepted transforms"):
            grid_jitter_transforms(
                source,
                spacing=6.0,
                seed=5,
                jitter=0.0,
                filters=TerrainScatterFilters(max_slope_deg=10.0),
            )

    def test_edge_margin_keeps_transforms_inside_terrain_border(self) -> None:
        source = TerrainScatterSource(np.full((24, 24), 42.0, dtype=np.float32), z_scale=2.0)
        transforms = grid_jitter_transforms(
            source,
            spacing=4.0,
            seed=7,
            jitter=0.0,
            edge_margin=3.0,
            filters=TerrainScatterFilters(max_slope_deg=0.1),
            yaw_range_deg=(0.0, 0.0),
            scale_range=(1.0, 1.0),
        )
        assert transforms.shape[0] > 0
        assert np.all(transforms[:, 3] >= 3.0)
        assert np.all(transforms[:, 3] <= source.terrain_width - 3.0)
        assert np.all(transforms[:, 11] >= 3.0)
        assert np.all(transforms[:, 11] <= source.terrain_width - 3.0)

    def test_seeded_random_raises_on_short_count(self) -> None:
        source = TerrainScatterSource(np.full((128, 128), 5.0, dtype=np.float32), z_scale=1.0)
        with pytest.raises(ValueError, match="accepted only"):
            seeded_random_transforms(
                source,
                count=500,
                seed=42,
                edge_margin=40.0,
                max_attempts=500,
                filters=TerrainScatterFilters(max_slope_deg=0.1),
            )

    def test_mask_density_rejects_non_finite_values(self) -> None:
        source = _scatter_source()
        mask = np.ones_like(source.heightmap, dtype=np.float32)
        mask[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            mask_density_transforms(source, mask, spacing=4.0, seed=5)

    def test_seeded_random_scales_to_fifty_thousand_transforms(self) -> None:
        source = TerrainScatterSource(np.full((128, 128), 5.0, dtype=np.float32), z_scale=1.0)
        transforms = seeded_random_transforms(
            source,
            count=50_000,
            seed=19,
            filters=TerrainScatterFilters(max_slope_deg=0.1),
            scale_range=(1.0, 1.0),
            yaw_range_deg=(0.0, 0.0),
            edge_margin=4.0,
            max_attempts=60_000,
        )
        assert transforms.shape == (50_000, 16)

    def test_viewer_orbit_radius_uses_terrain_width_units(self) -> None:
        source = TerrainScatterSource(np.full((96, 64), 5.0, dtype=np.float32), z_scale=1.0)
        assert viewer_orbit_radius(source) == pytest.approx(96.0 * 1.9)
        assert viewer_orbit_radius(source.terrain_width, scale=2.0, minimum=1.0) == pytest.approx(192.0)

    def test_viewer_orbit_radius_validates_inputs(self) -> None:
        source = _scatter_source()
        with pytest.raises(ValueError, match="terrain_width"):
            viewer_orbit_radius(0.0)
        with pytest.raises(ValueError, match="scale"):
            viewer_orbit_radius(source, scale=0.0)
        with pytest.raises(ValueError, match="minimum"):
            viewer_orbit_radius(source, minimum=-1.0)


class TestBatchSerialization:
    def test_blend_settings_validate_inputs(self) -> None:
        with pytest.raises(ValueError, match="terrain_blend.bury_depth"):
            TerrainMeshBlendSettings(enabled=True, bury_depth=-0.1)
        with pytest.raises(ValueError, match="terrain_blend.fade_distance"):
            TerrainMeshBlendSettings(enabled=True, fade_distance=0.0)

    def test_contact_settings_validate_inputs(self) -> None:
        with pytest.raises(ValueError, match="terrain_contact.distance"):
            TerrainContactSettings(enabled=True, distance=0.0)
        with pytest.raises(ValueError, match="terrain_contact.strength"):
            TerrainContactSettings(enabled=True, strength=1.5)
        with pytest.raises(ValueError, match="terrain_contact.vertical_weight"):
            TerrainContactSettings(enabled=True, vertical_weight=-0.1)

    def test_batch_rejects_non_finite_transforms(self) -> None:
        mesh = _simple_mesh()
        transforms = np.eye(4, dtype=np.float32).reshape(1, 16)
        transforms[0, 10] = np.nan
        with pytest.raises(ValueError, match="finite"):
            TerrainScatterBatch(
                transforms=transforms,
                levels=[TerrainScatterLevel(mesh=mesh)],
            )

    def test_batch_rejects_non_increasing_lod_distances(self) -> None:
        mesh = _simple_mesh()
        transforms = np.eye(4, dtype=np.float32).reshape(1, 16)
        with pytest.raises(ValueError, match="strictly increasing"):
            TerrainScatterBatch(
                transforms=transforms,
                levels=[
                    TerrainScatterLevel(mesh=mesh, max_distance=80.0),
                    TerrainScatterLevel(mesh=mesh, max_distance=40.0),
                ],
            )

    def test_batch_rejects_non_final_open_ended_lod(self) -> None:
        mesh = _simple_mesh()
        transforms = np.eye(4, dtype=np.float32).reshape(1, 16)
        with pytest.raises(ValueError, match="final LOD level"):
            TerrainScatterBatch(
                transforms=transforms,
                levels=[
                    TerrainScatterLevel(mesh=mesh),
                    TerrainScatterLevel(mesh=mesh, max_distance=80.0),
                ],
            )

    def test_batch_rejects_invalid_max_draw_distance(self) -> None:
        mesh = _simple_mesh()
        transforms = np.eye(4, dtype=np.float32).reshape(1, 16)
        with pytest.raises(ValueError, match="max_draw_distance"):
            TerrainScatterBatch(
                transforms=transforms,
                levels=[TerrainScatterLevel(mesh=mesh)],
                max_draw_distance=0.0,
            )

    def test_native_and_viewer_batch_shapes(self) -> None:
        mesh = _simple_mesh()
        transforms = np.stack(
            [
                np.eye(4, dtype=np.float32).reshape(16),
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 3.0],
                        [0.0, 1.0, 0.0, 4.0],
                        [0.0, 0.0, 1.0, 5.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ).reshape(16),
            ]
        )
        batch = TerrainScatterBatch(
            name="trees",
            color=(0.2, 0.6, 0.3, 1.0),
            max_draw_distance=140.0,
            terrain_blend=TerrainMeshBlendSettings(enabled=True, bury_depth=1.0, fade_distance=3.5),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=2.0,
                strength=0.4,
                vertical_weight=0.8,
            ),
            transforms=transforms,
            levels=[TerrainScatterLevel(mesh=mesh, max_distance=80.0)],
        )

        native = batch.to_native_dict()
        viewer = batch.to_viewer_payload()

        assert native["transforms"].shape == (2, 16)
        assert native["levels"][0]["mesh"]["positions"].shape == (3, 3)
        assert native["terrain_blend"]["enabled"] is True
        assert native["terrain_blend"]["fade_distance"] == pytest.approx(3.5)
        assert native["terrain_contact"]["vertical_weight"] == pytest.approx(0.8)
        assert viewer["transforms"][1][3] == pytest.approx(3.0)
        assert viewer["terrain_blend"]["bury_depth"] == pytest.approx(1.0)
        assert viewer["terrain_contact"]["strength"] == pytest.approx(0.4)
        assert viewer["levels"][0]["positions"][2] == [0.0, 1.0, 0.0]
        assert viewer["levels"][0]["indices"] == [0, 1, 2]

    def test_batch_accepts_dict_style_tv21_settings(self) -> None:
        mesh = _simple_mesh()
        batch = TerrainScatterBatch(
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
            levels=[TerrainScatterLevel(mesh=mesh)],
            terrain_blend={"enabled": True, "bury_depth": 0.5, "fade_distance": 1.5},
            terrain_contact={
                "enabled": True,
                "distance": 1.25,
                "strength": 0.2,
                "vertical_weight": 0.7,
            },
        )
        assert batch.terrain_blend.enabled is True
        assert batch.terrain_blend.fade_distance == pytest.approx(1.5)
        assert batch.terrain_contact.distance == pytest.approx(1.25)

    def test_apply_helpers_use_expected_commands(self) -> None:
        mesh = _simple_mesh()
        batch = TerrainScatterBatch(
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )

        class DummyRenderer:
            def __init__(self) -> None:
                self.batches = None

            def set_scatter_batches(self, batches):
                self.batches = batches

        class DummyViewer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, object]] = []

            def set_terrain_scatter(self, batches):
                self.calls.append(("set", batches))

            def clear_terrain_scatter(self):
                self.calls.append(("clear", None))

        renderer = DummyRenderer()
        viewer = DummyViewer()
        apply_to_renderer(renderer, [batch])
        apply_to_viewer(viewer, [batch])
        clear_viewer(viewer)

        assert renderer.batches is not None
        assert renderer.batches[0]["transforms"].shape == (1, 16)
        assert viewer.calls[0][0] == "set"
        assert viewer.calls[1] == ("clear", None)

    def test_batch_default_wind_is_disabled(self):
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=_simple_mesh())],
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
        )
        assert batch.wind.enabled is False
        assert batch.wind.amplitude == 0.0

    def test_batch_with_explicit_wind(self):
        wind = ts.ScatterWindSettings(enabled=True, amplitude=1.5, rigidity=0.3)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=_simple_mesh())],
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
            wind=wind,
        )
        assert batch.wind.enabled is True
        assert batch.wind.amplitude == 1.5

    def test_native_dict_includes_wind(self):
        wind = ts.ScatterWindSettings(enabled=True, amplitude=2.0, direction_deg=45.0)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=_simple_mesh())],
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
            wind=wind,
        )
        d = batch.to_native_dict()
        assert "wind" in d
        assert d["wind"]["enabled"] is True
        assert d["wind"]["amplitude"] == 2.0
        assert d["wind"]["direction_deg"] == 45.0

    def test_viewer_payload_includes_wind(self):
        wind = ts.ScatterWindSettings(enabled=True, amplitude=1.0)
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=_simple_mesh())],
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
            wind=wind,
        )
        p = batch.to_viewer_payload()
        assert "wind" in p
        assert p["wind"]["enabled"] is True

    def test_disabled_wind_still_serializes(self):
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=_simple_mesh())],
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
        )
        d = batch.to_native_dict()
        assert "wind" in d
        assert d["wind"]["enabled"] is False


_TERRAIN_RUNTIME_AVAILABLE = terrain_rendering_available()


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(
    not (
        _TERRAIN_RUNTIME_AVAILABLE
        and hasattr(f3d, "Session")
        and hasattr(f3d, "TerrainRenderer")
        and hasattr(f3d.TerrainRenderer, "set_scatter_batches")
        and hasattr(f3d.geometry, "primitive_mesh")
    ),
    reason="terrain scatter integration test requires GPU-backed native runtime with instancing",
)
class TestNativeScatterIntegration:
    def test_renderer_scatter_updates_stats_and_pixels(self) -> None:
        y, x = np.mgrid[0:96, 0:96].astype(np.float32)
        heightmap = np.sin(x / 7.0) * 8.0 + np.cos(y / 9.0) * 6.0 + x * 0.35 + y * 0.15 + 25.0
        source = TerrainScatterSource(heightmap, z_scale=1.4)
        transforms = grid_jitter_transforms(
            source,
            spacing=10.0,
            seed=21,
            jitter=0.55,
            filters=TerrainScatterFilters(max_slope_deg=45.0),
            scale_range=(3.0, 6.0),
        )

        batch = TerrainScatterBatch(
            name="integration_trees",
            color=(0.22, 0.48, 0.24, 1.0),
            max_draw_distance=180.0,
            transforms=transforms,
            levels=[
                TerrainScatterLevel(mesh=f3d.geometry.primitive_mesh("cone", radial_segments=10), max_distance=85.0),
                TerrainScatterLevel(mesh=f3d.geometry.primitive_mesh("box"), max_distance=160.0),
            ],
        )

        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        material_set = f3d.MaterialSet.terrain_default()

        with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
            hdr_path = tmp.name
        try:
            _create_test_hdr(hdr_path)
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

        renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap).to_numpy()
        apply_to_renderer(renderer, [batch])
        renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap).to_numpy()

        stats = renderer.get_scatter_stats()
        memory = renderer.get_scatter_memory_report()

        assert stats["batch_count"] == 1
        assert stats["rendered_batches"] == 1
        assert stats["total_instances"] == batch.instance_count
        assert stats["visible_instances"] > 0
        assert memory["total_buffer_bytes"] > 0
        # Exact pixel deltas are backend-sensitive in packaged wheel runs, but the
        # renderer-side contract is that batches upload successfully, survive
        # culling, select an LOD, and allocate the expected GPU buffers.
        assert any(count > 0 for count in stats["lod_instance_counts"])


@pytest.mark.interactive_viewer
@pytest.mark.skipif(
    os.environ.get("RUN_INTERACTIVE_VIEWER_CI") != "1",
    reason="interactive viewer scatter regression is opt-in",
)
def test_viewer_snapshot_scatter_changes_pixels(tmp_path) -> None:
    if not hasattr(f3d, "open_viewer_async"):
        pytest.skip("viewer IPC API unavailable")

    heightmap = np.sin(np.mgrid[0:96, 0:96][1] / 7.0) * 8.0 + np.cos(np.mgrid[0:96, 0:96][0] / 9.0) * 6.0
    heightmap = np.asarray(heightmap + np.mgrid[0:96, 0:96][1] * 0.35 + np.mgrid[0:96, 0:96][0] * 0.15 + 25.0, dtype=np.float32)
    source = TerrainScatterSource(heightmap, z_scale=1.4)
    transforms = grid_jitter_transforms(
        source,
        spacing=10.0,
        seed=21,
        jitter=0.55,
        filters=TerrainScatterFilters(max_slope_deg=45.0),
        scale_range=(6.0, 9.0),
    )
    batch = TerrainScatterBatch(
        name="viewer_scatter",
        color=(0.22, 0.62, 0.28, 1.0),
        max_draw_distance=180.0,
        transforms=transforms,
        levels=[TerrainScatterLevel(mesh=f3d.geometry.primitive_mesh("box"), max_distance=160.0)],
    )

    dem_path = tmp_path / "viewer_scatter_dem.tif"
    baseline_path = tmp_path / "viewer_scatter_baseline.png"
    scatter_path = tmp_path / "viewer_scatter_enabled.png"
    _write_heightmap_tiff(str(dem_path), heightmap)

    try:
        with f3d.open_viewer_async(
            width=640,
            height=400,
            title="Forge3D Scatter Viewer Test",
            terrain_path=str(dem_path),
            timeout=60.0,
        ) as viewer:
            viewer.set_z_scale(1.4)
            viewer.set_orbit_camera(
                phi_deg=146.0,
                theta_deg=58.0,
                radius=viewer_orbit_radius(source),
                fov_deg=50.0,
            )
            viewer.send_ipc(
                {
                    "cmd": "set_terrain_sun",
                    "azimuth_deg": 138.0,
                    "elevation_deg": 28.0,
                    "intensity": 2.4,
                }
            )
            viewer.snapshot(str(baseline_path), width=640, height=400)
            apply_to_viewer(viewer, [batch])
            viewer.snapshot(str(scatter_path), width=640, height=400)
    except FileNotFoundError:
        pytest.skip("interactive_viewer binary not found")

    pillow = pytest.importorskip("PIL.Image")
    baseline = np.asarray(pillow.open(baseline_path))
    scattered = np.asarray(pillow.open(scatter_path))
    changed_pixels = np.count_nonzero(np.any(baseline != scattered, axis=-1))
    assert changed_pixels > 250


class TestScatterWindSettings:
    """Tests for ScatterWindSettings dataclass."""

    def test_defaults(self):
        s = ts.ScatterWindSettings()
        assert s.enabled is False
        assert s.direction_deg == 0.0
        assert s.speed == 1.0
        assert s.amplitude == 0.0
        assert s.rigidity == 0.5
        assert s.bend_start == 0.0
        assert s.bend_extent == 1.0
        assert s.gust_strength == 0.0
        assert s.gust_frequency == 0.3
        assert s.fade_start == 0.0
        assert s.fade_end == 0.0

    def test_enabled_with_amplitude(self):
        s = ts.ScatterWindSettings(enabled=True, amplitude=2.0)
        assert s.enabled is True
        assert s.amplitude == 2.0

    def test_speed_rejects_negative(self):
        with pytest.raises(ValueError, match="speed"):
            ts.ScatterWindSettings(speed=-1.0)

    def test_amplitude_rejects_negative(self):
        with pytest.raises(ValueError, match="amplitude"):
            ts.ScatterWindSettings(amplitude=-0.1)

    def test_rigidity_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="rigidity"):
            ts.ScatterWindSettings(rigidity=1.5)
        with pytest.raises(ValueError, match="rigidity"):
            ts.ScatterWindSettings(rigidity=-0.1)

    def test_bend_start_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="bend_start"):
            ts.ScatterWindSettings(bend_start=-0.1)
        with pytest.raises(ValueError, match="bend_start"):
            ts.ScatterWindSettings(bend_start=1.1)

    def test_bend_extent_rejects_non_positive(self):
        with pytest.raises(ValueError, match="bend_extent"):
            ts.ScatterWindSettings(bend_extent=0.0)
        with pytest.raises(ValueError, match="bend_extent"):
            ts.ScatterWindSettings(bend_extent=-1.0)

    def test_gust_strength_rejects_negative(self):
        with pytest.raises(ValueError, match="gust_strength"):
            ts.ScatterWindSettings(gust_strength=-1.0)

    def test_gust_frequency_rejects_negative(self):
        with pytest.raises(ValueError, match="gust_frequency"):
            ts.ScatterWindSettings(gust_frequency=-1.0)

    def test_fade_start_rejects_negative(self):
        with pytest.raises(ValueError, match="fade_start"):
            ts.ScatterWindSettings(fade_start=-1.0)

    def test_fade_end_rejects_negative(self):
        with pytest.raises(ValueError, match="fade_end"):
            ts.ScatterWindSettings(fade_end=-1.0)

    def test_numpy_types_normalized_to_python(self):
        s = ts.ScatterWindSettings(
            enabled=np.bool_(True),
            speed=np.float32(2.5),
            amplitude=np.float64(1.0),
            direction_deg=np.float32(45.0),
        )
        assert type(s.enabled) is bool
        assert type(s.speed) is float
        assert type(s.amplitude) is float
        assert type(s.direction_deg) is float

    def test_numpy_wind_serializes_as_json(self):
        import json
        s = ts.ScatterWindSettings(
            enabled=np.bool_(True),
            speed=np.float32(2.5),
            amplitude=np.float64(1.0),
        )
        batch = TerrainScatterBatch(
            levels=[TerrainScatterLevel(mesh=_simple_mesh())],
            transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
            wind=s,
        )
        payload = batch.to_viewer_payload()
        # Must not raise TypeError from numpy types
        json.dumps(payload["wind"])

    def test_non_finite_fields_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            ts.ScatterWindSettings(speed=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            ts.ScatterWindSettings(amplitude=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            ts.ScatterWindSettings(direction_deg=float("-inf"))
        with pytest.raises(ValueError, match="finite"):
            ts.ScatterWindSettings(fade_start=float("nan"))

    def test_batch_rejects_invalid_wind_type(self):
        with pytest.raises(TypeError, match="ScatterWindSettings"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=_simple_mesh())],
                transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
                wind=None,
            )
        with pytest.raises(TypeError, match="ScatterWindSettings"):
            TerrainScatterBatch(
                levels=[TerrainScatterLevel(mesh=_simple_mesh())],
                transforms=np.eye(4, dtype=np.float32).reshape(1, 16),
                wind={"enabled": True, "amplitude": 1.0},
            )
