from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.geometry import MeshBuffers
from forge3d.terrain_params import PomSettings, make_terrain_params_config
from forge3d.terrain_scatter import (
    TerrainContactSettings,
    TerrainMeshBlendSettings,
    TerrainScatterBatch,
    TerrainScatterLevel,
    TerrainScatterSource,
    apply_to_renderer,
    make_transform_row_major,
)


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int(255 * x / max(width - 1, 1))
                g = int(255 * y / max(height - 1, 1))
                handle.write(bytes([r, g, 180, 128]))


def _scaled_grounded_mesh(mesh: MeshBuffers, scale_xyz: tuple[float, float, float]) -> MeshBuffers:
    positions = np.asarray(mesh.positions, dtype=np.float32).copy()
    positions *= np.asarray(scale_xyz, dtype=np.float32)
    positions[:, 1] -= float(np.min(positions[:, 1]))
    return MeshBuffers(
        positions=positions,
        normals=np.asarray(mesh.normals, dtype=np.float32).copy(),
        uvs=np.asarray(mesh.uvs, dtype=np.float32).copy(),
        indices=np.asarray(mesh.indices, dtype=np.uint32).copy(),
        tangents=None if mesh.tangents is None else np.asarray(mesh.tangents, dtype=np.float32).copy(),
    )


def _heightmap_rock_cluster() -> np.ndarray:
    y, x = np.mgrid[0:160, 0:160].astype(np.float32)
    return np.asarray(
        24.0 + np.sin(x / 10.0) * 8.0 + np.cos(y / 12.0) * 6.0 + x * 0.18 + y * 0.08,
        dtype=np.float32,
    )


def _heightmap_road_edge() -> np.ndarray:
    y, x = np.mgrid[0:160, 0:160].astype(np.float32)
    return np.asarray(
        18.0 + (x * 0.32) + (y * 0.22) + np.sin((x + y) / 18.0) * 2.5,
        dtype=np.float32,
    )


def _heightmap_foundation() -> np.ndarray:
    y, x = np.mgrid[0:160, 0:160].astype(np.float32)
    base = 20.0 + x * 0.12 + y * 0.05 + np.sin(x / 16.0) * 1.4 + np.cos(y / 18.0) * 1.0
    radius = np.sqrt((x - 80.0) ** 2 + (y - 80.0) ** 2)
    plateau = np.clip(1.0 - radius / 42.0, 0.0, 1.0)
    return np.asarray(base - plateau * 2.8, dtype=np.float32)


def _surface_translation(
    source: TerrainScatterSource,
    x: float,
    z: float,
    *,
    bury: float,
) -> tuple[float, float, float]:
    row, col = source.contract_to_pixel(x, z)
    y = source.sample_scaled_height(row, col) - float(bury)
    return (float(x), float(y), float(z))


@dataclass(frozen=True)
class Tv21Case:
    name: str
    heightmap: np.ndarray
    batch: TerrainScatterBatch
    z_scale: float
    cam_radius: float
    cam_phi_deg: float
    cam_theta_deg: float
    min_changed_pixels: int


def _build_case(case_name: str) -> Tv21Case:
    if case_name == "rock_cluster":
        heightmap = _heightmap_rock_cluster()
        source = TerrainScatterSource(heightmap, z_scale=1.35)
        mesh = _scaled_grounded_mesh(
            f3d.geometry.primitive_mesh("cylinder", radial_segments=8),
            (4.0, 6.0, 4.0),
        )
        cx = source.terrain_width * 0.5
        cz = source.terrain_width * 0.5
        transforms = np.asarray(
            [
                make_transform_row_major(
                    _surface_translation(source, cx - 9.0, cz - 4.0, bury=1.0),
                    yaw_deg=18.0,
                    scale=0.85,
                ),
                make_transform_row_major(
                    _surface_translation(source, cx, cz, bury=1.1),
                    yaw_deg=42.0,
                    scale=1.10,
                ),
                make_transform_row_major(
                    _surface_translation(source, cx + 10.0, cz + 5.0, bury=0.9),
                    yaw_deg=11.0,
                    scale=0.95,
                ),
                make_transform_row_major(
                    _surface_translation(source, cx + 4.0, cz - 10.0, bury=0.8),
                    yaw_deg=63.0,
                    scale=0.75,
                ),
            ],
            dtype=np.float32,
        )
        batch = TerrainScatterBatch(
            name="rocks",
            color=(0.55, 0.43, 0.32, 1.0),
            transforms=transforms,
            terrain_blend=TerrainMeshBlendSettings(enabled=True, bury_depth=1.4, fade_distance=3.0),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=2.6,
                strength=0.38,
                vertical_weight=0.55,
            ),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )
        return Tv21Case(case_name, heightmap, batch, source.z_scale, 84.0, 140.0, 58.0, 180)

    if case_name == "road_edge":
        heightmap = _heightmap_road_edge()
        source = TerrainScatterSource(heightmap, z_scale=1.2)
        mesh = _scaled_grounded_mesh(f3d.geometry.primitive_mesh("box"), (28.0, 1.8, 5.0))
        center = source.terrain_width * 0.5
        batch = TerrainScatterBatch(
            name="road_edge",
            color=(0.32, 0.30, 0.28, 1.0),
            transforms=np.asarray(
                [
                    make_transform_row_major(
                        _surface_translation(source, center, center, bury=0.65),
                        yaw_deg=28.0,
                        scale=1.0,
                    )
                ],
                dtype=np.float32,
            ),
            terrain_blend=TerrainMeshBlendSettings(enabled=True, bury_depth=1.0, fade_distance=2.8),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=3.2,
                strength=0.34,
                vertical_weight=0.95,
            ),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )
        return Tv21Case(case_name, heightmap, batch, source.z_scale, 88.0, 134.0, 60.0, 220)

    if case_name == "building_foundation":
        heightmap = _heightmap_foundation()
        source = TerrainScatterSource(heightmap, z_scale=1.15)
        mesh = _scaled_grounded_mesh(f3d.geometry.primitive_mesh("box"), (26.0, 2.5, 26.0))
        center = source.terrain_width * 0.5
        batch = TerrainScatterBatch(
            name="foundation",
            color=(0.70, 0.71, 0.69, 1.0),
            transforms=np.asarray(
                [
                    make_transform_row_major(
                        _surface_translation(source, center, center, bury=1.1),
                        yaw_deg=12.0,
                        scale=1.0,
                    )
                ],
                dtype=np.float32,
            ),
            terrain_blend=TerrainMeshBlendSettings(enabled=True, bury_depth=1.4, fade_distance=3.4),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=3.0,
                strength=0.28,
                vertical_weight=0.85,
            ),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )
        return Tv21Case(case_name, heightmap, batch, source.z_scale, 92.0, 144.0, 57.0, 200)

    raise AssertionError(f"Unknown TV21 case: {case_name}")


def _render_case(case: Tv21Case, batch: TerrainScatterBatch | None) -> tuple[np.ndarray, dict[str, object]]:
    terrain_span = 220.0
    domain = (float(np.min(case.heightmap)), float(np.max(case.heightmap)))

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
        size_px=(420, 300),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=4,
        z_scale=case.z_scale,
        exposure=1.0,
        domain=domain,
        cam_radius=case.cam_radius,
        cam_phi_deg=case.cam_phi_deg,
        cam_theta_deg=case.cam_theta_deg,
        fov_y_deg=46.0,
        camera_mode="mesh",
        clip=(0.1, terrain_span * 4.0),
        light_azimuth_deg=136.0,
        light_elevation_deg=26.0,
        sun_intensity=2.6,
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
    )
    config.cam_target = [0.0, 0.0, 0.0]
    params = f3d.TerrainRenderParams(config)

    if batch is not None:
        apply_to_renderer(renderer, [batch])

    image = renderer.render_terrain_pbr_pom(material_set, ibl, params, case.heightmap).to_numpy()
    return image, renderer.get_scatter_stats()


def _changed_pixels(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(np.any(before != after, axis=-1)))


_TV21_RUNTIME_AVAILABLE = terrain_rendering_available()


def _require_tv21_runtime() -> None:
    if not (
        _TV21_RUNTIME_AVAILABLE
        and hasattr(f3d, "Session")
        and hasattr(f3d, "TerrainRenderer")
        and hasattr(f3d.TerrainRenderer, "set_scatter_batches")
        and hasattr(f3d.geometry, "primitive_mesh")
    ):
        pytest.skip("TV21 renderer regression requires GPU-backed terrain runtime")


@pytest.mark.apple_metal_physical
def test_tv21_disabled_settings_preserve_baseline() -> None:
    _require_tv21_runtime()
    case = _build_case("road_edge")
    baseline_batch = TerrainScatterBatch(
        name=case.batch.name,
        color=case.batch.color,
        transforms=case.batch.transforms.copy(),
        levels=case.batch.levels,
    )
    baseline, baseline_stats = _render_case(case, baseline_batch)
    disabled_batch = TerrainScatterBatch(
        name=case.batch.name,
        color=case.batch.color,
        transforms=case.batch.transforms.copy(),
        terrain_blend=TerrainMeshBlendSettings(enabled=False, bury_depth=2.0, fade_distance=0.5),
        terrain_contact=TerrainContactSettings(
            enabled=False,
            distance=1.5,
            strength=1.0,
            vertical_weight=1.0,
        ),
        levels=case.batch.levels,
    )
    disabled, disabled_stats = _render_case(case, disabled_batch)

    assert baseline_stats["batch_count"] == 1
    assert disabled_stats["batch_count"] == 1
    assert np.array_equal(baseline, disabled)

@pytest.mark.parametrize("case_name", ["rock_cluster", "road_edge", "building_foundation"])
@pytest.mark.apple_metal_physical
def test_tv21_enabled_settings_change_image(case_name: str) -> None:
    _require_tv21_runtime()
    case = _build_case(case_name)
    baseline_batch = TerrainScatterBatch(
        name=case.batch.name,
        color=case.batch.color,
        transforms=case.batch.transforms.copy(),
        levels=case.batch.levels,
    )

    baseline, _ = _render_case(case, baseline_batch)
    enabled, stats = _render_case(case, case.batch)

    changed = _changed_pixels(baseline, enabled)
    mean_delta = float(
        np.mean(np.abs(enabled.astype(np.int16) - baseline.astype(np.int16)))
    )

    assert stats["batch_count"] == 1
    assert stats["visible_instances"] > 0
    assert changed >= case.min_changed_pixels
    assert mean_delta > 0.08
