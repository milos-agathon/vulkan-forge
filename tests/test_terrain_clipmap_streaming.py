# tests/test_terrain_clipmap_streaming.py
# BOP-P2-02: clipmap geometry provider + runtime height-tile streaming.
#
# Covers the TerrainRenderer runtime path directly:
# - the indexed clipmap ring/skirt mesh renders on the default backend
#   (no backend-conditional grid fallback),
# - clipmap renders are deterministic,
# - LOD-ring boundaries show no cracks (background bleed-through) at steep
#   relief and a grazing camera,
# - height streaming (ClipmapStreamer + AsyncTileLoader + HeightMosaic)
#   converges during a fly-through with tile loads in flight, never showing
#   holes (coarse-prefill fallback), with GPU-resident height memory bounded.

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d

from _terrain_runtime import (
    _build_overlay,
    _write_test_hdr,
    terrain_rendering_available,
)
from forge3d.diagnostics import render_certificate
from forge3d.terrain_params import PomSettings, make_terrain_params_config

requires_terrain = pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="Terrain rendering runtime unavailable on this adapter",
)

TERRAIN_SPAN_M = 100_000.0
CLIPMAP_MODE = "clipmap:4:32:32:10:0.3"
# Background clear color from the terrain render pass (linear 0.1/0.1/0.15).
BACKGROUND_RGB_LINEAR = np.array([0.1, 0.1, 0.15], dtype=np.float32)


def _steep_dem(size: int = 128) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ridges = 0.5 * np.sin(xx * np.pi * 5.0) * np.cos(yy * np.pi * 4.0)
    peak = 0.8 * np.exp(-(xx**2 + yy**2) * 3.0)
    dem = ridges + peak
    dem -= dem.min()
    dem /= max(float(dem.max()), 1e-6)
    return dem.astype(np.float32)


def _make_params(
    *,
    camera_mode: str = CLIPMAP_MODE,
    size_px: tuple[int, int] = (128, 80),
    theta_deg: float = 49.0,
    phi_deg: float = 28.0,
    cam_radius: float = 1.0,
    z_scale: float = 1.2,
    culling: str = "frustum",
    shading: str = "forward",
    vt=None,
    terrain_span: float = TERRAIN_SPAN_M,
    cam_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "f3d.TerrainRenderParams":
    return f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=size_px,
            render_scale=1.0,
            terrain_span=terrain_span,
            msaa_samples=1,
            z_scale=z_scale,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="colormap",
            colormap_strength=1.0,
            ibl_enabled=True,
            light_azimuth_deg=138.0,
            light_elevation_deg=24.0,
            sun_intensity=2.4,
            cam_radius=cam_radius,
            cam_phi_deg=phi_deg,
            cam_theta_deg=theta_deg,
            cam_target=list(cam_target),
            fov_y_deg=45.0,
            camera_mode=camera_mode,
            culling=culling,
            shading=shading,
            vt=vt,
            clip=(0.1, terrain_span * 1.5),
            overlays=[_build_overlay()],
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        )
    )


@pytest.fixture(scope="module")
def terrain_ibl():
    with tempfile.TemporaryDirectory() as td:
        hdr_path = Path(td) / "probe.hdr"
        _write_test_hdr(hdr_path)
        yield f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)


def _render_rgba(renderer, params, heightmap, terrain_ibl) -> np.ndarray:
    material_set = f3d.MaterialSet.terrain_default()
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=terrain_ibl,
        params=params,
        heightmap=heightmap,
        target=None,
        water_mask=None,
    )
    return np.asarray(frame.to_numpy())


def _background_mask(rgba: np.ndarray, tolerance: int = 6) -> np.ndarray:
    srgb = np.where(
        BACKGROUND_RGB_LINEAR <= 0.0031308,
        BACKGROUND_RGB_LINEAR * 12.92,
        1.055 * np.power(BACKGROUND_RGB_LINEAR, 1.0 / 2.4) - 0.055,
    )
    bg = np.round(srgb * 255.0).astype(np.int32)
    diff = np.abs(rgba[..., :3].astype(np.int32) - bg[None, None, :])
    return (diff <= tolerance).all(axis=-1)


def test_terrain_renderer_exposes_height_streaming_api():
    for name in (
        "enable_height_streaming",
        "disable_height_streaming",
        "stream_height_tiles",
        "height_streaming_stats",
    ):
        assert hasattr(f3d.TerrainRenderer, name), f"TerrainRenderer.{name} missing"


@requires_terrain
class TestClipmapGeometryProvider:
    def test_clipmap_render_uses_gpu_lod_indirect_draws(self, terrain_ibl):
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        _render_rgba(renderer, _make_params(), _steep_dem(64), terrain_ibl)

        certificate = render_certificate(sign=False)
        main_pass = next(p for p in certificate["passes"] if p["label"] == "terrain.main")
        assert main_pass["draw_calls"] >= 1
        assert "clipmap_lod_select" in certificate["engine"]["wgsl_module_hashes"]

    def test_clipmap_render_is_deterministic(self, terrain_ibl):
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        dem = _steep_dem(64)
        params = _make_params()
        first = _render_rgba(renderer, params, dem, terrain_ibl)
        second = _render_rgba(renderer, params, dem, terrain_ibl)
        assert first.shape == second.shape
        np.testing.assert_array_equal(first, second)

    def test_clipmap_render_fills_frame_without_cracks(self, terrain_ibl):
        # Steep relief + grazing camera: cracks between LOD rings would let the
        # clear color bleed through inside the terrain footprint.
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        dem = _steep_dem(128)
        params = _make_params(theta_deg=72.0, z_scale=2.0)
        rgba = _render_rgba(renderer, params, dem, terrain_ibl)
        assert int(rgba[..., 3].max()) == 255
        assert float((rgba[..., :3].sum(axis=-1) > 0).mean()) > 0.99

        h, w = rgba.shape[:2]
        interior = _background_mask(rgba)[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5]
        assert int(interior.sum()) == 0, (
            f"{int(interior.sum())} background-colored pixels inside the "
            "terrain footprint indicate cracks between clipmap LOD rings"
        )

    def test_streaming_before_enable_raises(self):
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        with pytest.raises(RuntimeError, match="height streaming not enabled"):
            renderer.stream_height_tiles((0.0, 500.0, 0.0))
        with pytest.raises(RuntimeError, match="height streaming not enabled"):
            renderer.height_streaming_stats()


@requires_terrain
class TestHeightStreamingFlyThrough:
    LOD = 2
    TILE_RES = 64

    def test_disable_height_streaming_clears_public_family_stats(self):
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        renderer.enable_height_streaming(
            terrain_extent_m=TERRAIN_SPAN_M,
            ring_count=2,
            ring_resolution=16,
            lod=1,
            tile_resolution=32,
            max_in_flight=4,
            pool_size=1,
            dem=_steep_dem(64),
            coarse_prefill=False,
            max_resident_bytes=1024 * 1024,
        )
        renderer.stream_height_tiles((0.0, 500.0, 0.0), max_uploads=1)
        active_stats = f3d.terrain_vt_stats()
        assert active_stats["budget_bytes_height"] > 0
        assert active_stats["height_pending_requests"] > 0

        renderer.disable_height_streaming()
        disabled_stats = f3d.terrain_vt_stats()
        assert disabled_stats["resident_tiles_height"] == 0
        assert disabled_stats["resident_bytes_height"] == 0
        assert disabled_stats["budget_bytes_height"] == 0
        assert disabled_stats["height_pending_requests"] == 0

    def test_fly_through_streams_tiles_without_holes_and_bounded_memory(self, terrain_ibl):
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)

        dem = _steep_dem(256)
        overview = dem[::8, ::8].copy()  # coarse overview fed to each render
        renderer.enable_height_streaming(
            terrain_extent_m=TERRAIN_SPAN_M,
            ring_count=4,
            ring_resolution=32,
            lod=self.LOD,
            tile_resolution=self.TILE_RES,
            max_in_flight=16,
            pool_size=2,
            dem=dem,
            coarse_prefill=True,
            max_resident_bytes=8 * 1024 * 1024,
        )

        tiles_axis = 1 << self.LOD
        expected_resident = tiles_axis * tiles_axis * self.TILE_RES * self.TILE_RES * 4
        stats = renderer.height_streaming_stats()
        assert stats["coarse_prefilled"] == tiles_axis * tiles_axis
        assert stats["resident_height_bytes"] == expected_resident
        assert stats["resident_height_bytes"] <= 8 * 1024 * 1024

        params = _make_params()
        coarse_frame = _render_rgba(renderer, params, overview, terrain_ibl)
        assert float((coarse_frame[..., :3].sum(axis=-1) > 0).mean()) > 0.99

        # Fly across the region: tiles load asynchronously while frames render.
        waypoints = np.linspace(-TERRAIN_SPAN_M * 0.35, TERRAIN_SPAN_M * 0.35, 8)
        last_stats = stats
        for step, wx in enumerate(waypoints):
            last_stats = renderer.stream_height_tiles(
                (float(wx), 500.0, float(wx) * 0.5), max_uploads=4
            )
            rgba = _render_rgba(renderer, params, overview, terrain_ibl)
            h, w = rgba.shape[:2]
            interior = _background_mask(rgba)[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5]
            assert int(interior.sum()) == 0, (
                f"fly-through frame {step} shows {int(interior.sum())} hole pixels "
                "while tile loads are in flight (coarse fallback must cover them)"
            )
            assert last_stats["resident_height_bytes"] == expected_resident

        # Drain until fine residency converges; bounded, real async work.
        for _ in range(200):
            last_stats = renderer.stream_height_tiles(
                (float(waypoints[-1]), 500.0, float(waypoints[-1]) * 0.5), max_uploads=8
            )
            if last_stats["converged"]:
                break
        assert last_stats["converged"], f"streaming never converged: {last_stats}"
        assert last_stats["tiles_uploaded"] >= tiles_axis * tiles_axis
        assert last_stats["resident_fine_tiles"] == tiles_axis * tiles_axis

        # Fine streamed tiles must actually change rendered pixels vs the
        # coarse prefill (the mosaic is the live height source).
        fine_frame = _render_rgba(renderer, params, overview, terrain_ibl)
        assert fine_frame.shape == coarse_frame.shape
        assert not np.array_equal(fine_frame, coarse_frame), (
            "converged fine-tile render is bit-identical to the coarse prefill "
            "render; streamed tiles are not reaching the height texture"
        )

        # The clipmap mesh recenters on the streaming camera.
        assert abs(last_stats["center"][0] - float(waypoints[-1])) < TERRAIN_SPAN_M * 0.05

        renderer.disable_height_streaming()
        disabled_stats = f3d.terrain_vt_stats()
        assert disabled_stats["resident_tiles_height"] == 0
        assert disabled_stats["resident_bytes_height"] == 0
        assert disabled_stats["budget_bytes_height"] == 0
        assert disabled_stats["height_pending_requests"] == 0
        with pytest.raises(RuntimeError, match="height streaming not enabled"):
            renderer.height_streaming_stats()
