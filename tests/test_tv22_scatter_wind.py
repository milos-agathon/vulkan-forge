"""TV22 Scatter Wind Animation — integration tests.

Verifies that wind settings on scatter batches produce the expected
rendering behaviour:  no-op modes match static baseline, different
time_seconds values produce different frames, same time is deterministic,
and per-batch controls (bend_start, gust, fade) affect output.

RELEVANT FILES: python/forge3d/terrain_scatter.py, src/terrain/scatter.rs,
                src/shaders/mesh_instanced.wgsl, src/terrain/renderer/scatter.rs
"""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest

f3d = pytest.importorskip("forge3d")
from _terrain_runtime import terrain_rendering_available
from forge3d import terrain_scatter as ts
from forge3d.terrain_params import make_terrain_params_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scatter_mesh():
    """Cone with y=0 at base (wind expects Y=0 base convention)."""
    from forge3d.geometry import MeshBuffers
    mesh = f3d.geometry.primitive_mesh("cone", radial_segments=10)
    positions = mesh.positions.copy()
    y_min = positions[:, 1].min()
    positions[:, 1] -= y_min  # shift so base is at y=0
    return MeshBuffers(
        positions=positions, normals=mesh.normals,
        uvs=mesh.uvs, indices=mesh.indices,
    )


def _make_heightmap() -> np.ndarray:
    """Varied heightmap matching the working scatter integration test pattern."""
    y, x = np.mgrid[0:96, 0:96].astype(np.float32)
    return np.sin(x / 7.0) * 8.0 + np.cos(y / 9.0) * 6.0 + x * 0.35 + y * 0.15 + 25.0


def _make_scatter_source() -> ts.TerrainScatterSource:
    heightmap = _make_heightmap()
    return ts.TerrainScatterSource(heightmap, z_scale=1.4)


def _place_instances(source: ts.TerrainScatterSource, count: int = 40, seed: int = 42) -> np.ndarray:
    return ts.seeded_random_transforms(
        source,
        count=count,
        seed=seed,
        scale_range=(3.0, 6.0),
        filters=ts.TerrainScatterFilters(max_slope_deg=45.0),
    )


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                handle.write(
                    bytes(
                        [
                            int(255 * x / max(width - 1, 1)),
                            int(255 * y / max(height - 1, 1)),
                            180,
                            128,
                        ]
                    )
                )


def _write_heightmap_tiff(path: str, heightmap: np.ndarray) -> None:
    pillow = pytest.importorskip("PIL.Image")
    pillow.fromarray(np.ascontiguousarray(heightmap, dtype=np.float32)).save(path, format="TIFF")


def _render_frame(renderer, material_set, ibl, params, heightmap, batches, time_seconds=0.0):
    """Apply scatter batches and render one frame, returning numpy array."""
    ts.apply_to_renderer(renderer, batches)
    frame = renderer.render_terrain_pbr_pom(
        material_set, ibl, params, heightmap, time_seconds=time_seconds,
    )
    return frame.to_numpy()


def _render_frame_with_aov(
    renderer, material_set, ibl, params, heightmap, batches, time_seconds=0.0,
):
    """Apply scatter batches and render one beauty + AOV frame."""
    ts.apply_to_renderer(renderer, batches)
    frame, aov_frame = renderer.render_with_aov(
        material_set, ibl, params, heightmap, time_seconds=time_seconds,
    )
    return frame.to_numpy(), aov_frame


# ---------------------------------------------------------------------------
# Skip guard — identical to TestNativeScatterIntegration
# ---------------------------------------------------------------------------

_TERRAIN_RUNTIME_AVAILABLE = terrain_rendering_available()

_SKIP_REASON = "wind integration tests require GPU-backed native runtime with scatter instancing"

_CAN_RUN = (
    _TERRAIN_RUNTIME_AVAILABLE
    and hasattr(f3d, "Session")
    and hasattr(f3d, "TerrainRenderer")
    and hasattr(f3d.TerrainRenderer, "set_scatter_batches")
    and hasattr(f3d.geometry, "primitive_mesh")
)


def _make_gpu_fixtures():
    """Build Session, TerrainRenderer, MaterialSet, IBL, params, heightmap.

    Uses the same heightmap/camera pattern as TestNativeScatterIntegration
    in test_terrain_scatter.py to ensure scatter instances are visible.
    """
    heightmap = _make_heightmap()

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
        msaa_samples=1,
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


# ---------------------------------------------------------------------------
# TestWindNoOp
# ---------------------------------------------------------------------------


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)
class TestWindNoOp:
    """Wind disabled or zero-amplitude must produce identical output to static baseline."""

    @pytest.fixture(scope="class")
    def gpu(self):
        return _make_gpu_fixtures()

    def test_disabled_wind_matches_static(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)

        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
        )
        batch_disabled = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=False, amplitude=5.0),
        )

        frame_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        frame_disabled = _render_frame(renderer, ms, ibl, params, hm, [batch_disabled])
        np.testing.assert_array_equal(frame_static, frame_disabled)

    def test_zero_amplitude_matches_static(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)

        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
        )
        batch_zero = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=0.0),
        )

        frame_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        frame_zero = _render_frame(renderer, ms, ibl, params, hm, [batch_zero])
        np.testing.assert_array_equal(frame_static, frame_zero)

    def test_rigidity_one_matches_static(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)

        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
        )
        batch_rigid = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=3.0, rigidity=1.0),
        )

        frame_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        frame_rigid = _render_frame(renderer, ms, ibl, params, hm, [batch_rigid])
        np.testing.assert_array_equal(frame_static, frame_rigid)


# ---------------------------------------------------------------------------
# TestWindAnimation
# ---------------------------------------------------------------------------


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)
class TestWindAnimation:
    """Wind enabled must produce visible, deterministic animation."""

    @pytest.fixture(scope="class")
    def gpu(self):
        return _make_gpu_fixtures()

    def test_different_times_differ(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)
        wind = ts.ScatterWindSettings(enabled=True, amplitude=2.0, speed=1.0)
        batch = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=wind,
        )

        f0 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=0.0)
        # Avoid t=1.0 with speed=1.0: temporal_phase wraps to exactly 2π → sin(2π+x)==sin(x)
        f1 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=0.37)
        assert not np.array_equal(f0, f1), "wind at different times should differ"

    def test_same_time_is_deterministic(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)
        wind = ts.ScatterWindSettings(enabled=True, amplitude=2.0, speed=1.0)
        batch = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=wind,
        )

        f1 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=0.5)
        f2 = _render_frame(renderer, ms, ibl, params, hm, [batch], time_seconds=0.5)
        np.testing.assert_array_equal(f1, f2)

    def test_bend_start_affects_output(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)

        batch_low = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, bend_start=0.0),
        )
        batch_high = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, bend_start=0.8),
        )

        f_low = _render_frame(renderer, ms, ibl, params, hm, [batch_low], time_seconds=0.5)
        f_high = _render_frame(renderer, ms, ibl, params, hm, [batch_high], time_seconds=0.5)
        assert not np.array_equal(f_low, f_high), "different bend_start should differ"

    def test_gust_affects_output(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)

        batch_no_gust = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, gust_strength=0.0),
        )
        batch_gust = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(enabled=True, amplitude=2.0, gust_strength=1.5),
        )

        f_no = _render_frame(renderer, ms, ibl, params, hm, [batch_no_gust], time_seconds=0.5)
        f_yes = _render_frame(renderer, ms, ibl, params, hm, [batch_gust], time_seconds=0.5)
        assert not np.array_equal(f_no, f_yes), "gust should affect output"

    def test_render_with_aov_uses_time_seconds(self, gpu) -> None:
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()
        transforms = _place_instances(source)
        batch = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=transforms,
            wind=ts.ScatterWindSettings(
                enabled=True,
                amplitude=2.0,
                speed=1.0,
                gust_strength=0.75,
            ),
        )

        beauty_0, aov_0 = _render_frame_with_aov(
            renderer, ms, ibl, params, hm, [batch], time_seconds=0.0,
        )
        beauty_1, aov_1 = _render_frame_with_aov(
            renderer, ms, ibl, params, hm, [batch], time_seconds=0.37,
        )

        assert not np.array_equal(
            beauty_0,
            beauty_1,
        ), "AOV beauty output should animate with time_seconds"
        assert aov_0.normal().shape == aov_1.normal().shape == (160, 256, 3)


# ---------------------------------------------------------------------------
# TestWindFade
# ---------------------------------------------------------------------------


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)
class TestWindFade:
    """Distance fade must suppress wind at range."""

    @pytest.fixture(scope="class")
    def gpu(self):
        return _make_gpu_fixtures()

    def test_far_instances_match_static_with_fade(self, gpu) -> None:
        """Place instances far from camera with tight fade; verify wind suppressed."""
        renderer, ms, ibl, params, hm = gpu
        source = _make_scatter_source()

        # Place a single instance at the far corner of the terrain
        h = source.sample_scaled_height(
            source.height * 0.95, source.width * 0.95
        )
        far_transform = ts.make_transform_row_major(
            (source.terrain_width * 0.95, h, source.terrain_width * 0.95),
            scale=4.0,
        ).reshape(1, 16)

        batch_static = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=far_transform,
        )
        # Wind with fade that ends well before the instance distance
        batch_faded = ts.TerrainScatterBatch(
            levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh())],
            transforms=far_transform,
            wind=ts.ScatterWindSettings(
                enabled=True,
                amplitude=3.0,
                fade_start=1.0,
                fade_end=2.0,  # fade ends at 2 contract units
            ),
        )

        f_static = _render_frame(renderer, ms, ibl, params, hm, [batch_static])
        f_faded = _render_frame(
            renderer, ms, ibl, params, hm, [batch_faded], time_seconds=1.0,
        )
        np.testing.assert_array_equal(
            f_static,
            f_faded,
            err_msg="wind should be fully suppressed beyond fade_end",
        )


@pytest.mark.interactive_viewer
@pytest.mark.skipif(
    os.environ.get("RUN_INTERACTIVE_VIEWER_CI") != "1",
    reason="interactive viewer wind regression is opt-in",
)
def test_viewer_wind_uses_viewer_time_and_camera_updates(tmp_path) -> None:
    if not hasattr(f3d, "open_viewer_async"):
        pytest.skip("viewer IPC API unavailable")

    heightmap = _make_heightmap()
    source = ts.TerrainScatterSource(heightmap, z_scale=1.4)
    terrain_width = source.terrain_width
    hero_specs = [
        (terrain_width * 0.35, terrain_width * 0.42, 8.0),
        (terrain_width * 0.48, terrain_width * 0.50, 9.0),
        (terrain_width * 0.62, terrain_width * 0.44, 7.5),
    ]
    transforms = []
    for x, z, scale in hero_specs:
        row, col = source.contract_to_pixel(x, z)
        y = source.sample_scaled_height(row, col)
        transforms.append(ts.make_transform_row_major((x, y, z), scale=scale))
    batch = ts.TerrainScatterBatch(
        name="viewer_wind",
        color=(0.22, 0.62, 0.28, 1.0),
        max_draw_distance=220.0,
        transforms=np.ascontiguousarray(np.vstack(transforms).astype(np.float32)),
        levels=[ts.TerrainScatterLevel(mesh=_scatter_mesh(), max_distance=220.0)],
        wind=ts.ScatterWindSettings(
            enabled=True,
            amplitude=5.0,
            speed=1.2,
            rigidity=0.2,
            gust_strength=0.8,
            fade_end=220.0,
        ),
    )

    dem_path = tmp_path / "viewer_wind_dem.tif"
    snap_a = tmp_path / "viewer_wind_t0.png"
    snap_b = tmp_path / "viewer_wind_t1.png"
    snap_c = tmp_path / "viewer_wind_camera.png"
    _write_heightmap_tiff(str(dem_path), heightmap)

    try:
        with f3d.open_viewer_async(
            width=640,
            height=400,
            title="Forge3D Scatter Wind Viewer Test",
            terrain_path=str(dem_path),
            timeout=60.0,
        ) as viewer:
            viewer.set_z_scale(1.4)
            viewer.set_orbit_camera(
                phi_deg=146.0,
                theta_deg=58.0,
                radius=ts.viewer_orbit_radius(source, scale=1.4),
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
            ts.apply_to_viewer(viewer, [batch])
            time.sleep(0.35)
            viewer.snapshot(str(snap_a), width=640, height=400)
            time.sleep(0.45)
            viewer.snapshot(str(snap_b), width=640, height=400)
            viewer.set_orbit_camera(
                phi_deg=162.0,
                theta_deg=52.0,
                radius=ts.viewer_orbit_radius(source, scale=1.25),
                fov_deg=50.0,
            )
            time.sleep(0.15)
            viewer.snapshot(str(snap_c), width=640, height=400)
    except FileNotFoundError:
        pytest.skip("interactive_viewer binary not found")

    pillow = pytest.importorskip("PIL.Image")
    frame_a = np.asarray(pillow.open(snap_a))
    frame_b = np.asarray(pillow.open(snap_b))
    frame_c = np.asarray(pillow.open(snap_c))

    changed_time = np.count_nonzero(np.any(frame_a != frame_b, axis=-1))
    changed_camera = np.count_nonzero(np.any(frame_b != frame_c, axis=-1))

    assert changed_time > 40, "viewer wind should animate over wall-clock time"
    assert changed_camera > 250, "viewer wind path should remain live while camera changes"
