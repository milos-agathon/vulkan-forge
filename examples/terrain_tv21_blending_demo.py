"""Render three real-DEM TV21 terrain mesh blending cases."""

from __future__ import annotations

import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _terrain_feature_demo import ROOT, diff_image, load_dem, save
from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
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


DEFAULT_DEM = ROOT / "assets" / "tif" / "dem_rainier.tif"


@dataclass(frozen=True)
class _Case:
    name: str
    mesh: MeshBuffers
    transforms: np.ndarray
    color: tuple[float, float, float, float]
    blend: TerrainMeshBlendSettings
    contact: TerrainContactSettings


def _scaled_grounded_mesh(
    mesh: MeshBuffers, scale: tuple[float, float, float]
) -> MeshBuffers:
    positions = np.asarray(mesh.positions, dtype=np.float32).copy()
    positions *= np.asarray(scale, dtype=np.float32)
    positions[:, 1] -= float(positions[:, 1].min())
    return MeshBuffers(
        positions=positions,
        normals=np.asarray(mesh.normals, dtype=np.float32).copy(),
        uvs=np.asarray(mesh.uvs, dtype=np.float32).copy(),
        indices=np.asarray(mesh.indices, dtype=np.uint32).copy(),
        tangents=None
        if mesh.tangents is None
        else np.asarray(mesh.tangents, dtype=np.float32).copy(),
    )


def _transform(
    source: TerrainScatterSource,
    x: float,
    z: float,
    *,
    bury: float,
    yaw: float,
    scale: float = 1.0,
) -> np.ndarray:
    row, column = source.contract_to_pixel(x, z)
    y = source.sample_scaled_height(row, column) - bury
    return make_transform_row_major((x, y, z), yaw_deg=yaw, scale=scale)


def _cases(source: TerrainScatterSource) -> tuple[_Case, ...]:
    center = source.terrain_width * 0.5
    rock = _scaled_grounded_mesh(
        f3d.geometry.primitive_mesh("cylinder", radial_segments=8),
        (4.0, 6.0, 4.0),
    )
    road = _scaled_grounded_mesh(f3d.geometry.primitive_mesh("box"), (28.0, 1.8, 5.0))
    foundation = _scaled_grounded_mesh(
        f3d.geometry.primitive_mesh("box"), (26.0, 2.5, 26.0)
    )
    return (
        _Case(
            "rock_cluster",
            rock,
            np.asarray(
                [
                    _transform(source, center - 9.0, center - 4.0, bury=1.0, yaw=18.0, scale=0.85),
                    _transform(source, center, center, bury=1.1, yaw=42.0, scale=1.10),
                    _transform(source, center + 10.0, center + 5.0, bury=0.9, yaw=11.0, scale=0.95),
                    _transform(source, center + 4.0, center - 10.0, bury=0.8, yaw=63.0, scale=0.75),
                ],
                dtype=np.float32,
            ),
            (0.55, 0.43, 0.32, 1.0),
            TerrainMeshBlendSettings(enabled=True, bury_depth=1.4, fade_distance=3.0),
            TerrainContactSettings(enabled=True, distance=2.6, strength=0.38, vertical_weight=0.55),
        ),
        _Case(
            "road_edge",
            road,
            np.asarray(
                [_transform(source, center, center, bury=0.65, yaw=28.0)],
                dtype=np.float32,
            ),
            (0.32, 0.30, 0.28, 1.0),
            TerrainMeshBlendSettings(enabled=True, bury_depth=1.0, fade_distance=2.8),
            TerrainContactSettings(enabled=True, distance=3.2, strength=0.34, vertical_weight=0.95),
        ),
        _Case(
            "building_foundation",
            foundation,
            np.asarray(
                [_transform(source, center, center, bury=1.1, yaw=12.0)],
                dtype=np.float32,
            ),
            (0.70, 0.71, 0.69, 1.0),
            TerrainMeshBlendSettings(enabled=True, bury_depth=1.4, fade_distance=3.4),
            TerrainContactSettings(enabled=True, distance=3.0, strength=0.28, vertical_weight=0.85),
        ),
    )


def _hdr() -> f3d.IBL:
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as handle:
        path = Path(handle.name)
    try:
        pixels = bytes((96, 112, 176, 128)) * 32
        path.write_bytes(
            b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n" + pixels
        )
        return f3d.IBL.from_hdr(str(path), intensity=1.0)
    finally:
        path.unlink(missing_ok=True)


def _render(
    heightmap: np.ndarray,
    batch: TerrainScatterBatch,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    apply_to_renderer(renderer, [batch])
    config = make_terrain_params_config(
        size_px=(int(width), int(height)),
        render_scale=1.0,
        terrain_span=220.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(float(heightmap.min()), float(heightmap.max())),
        cam_radius=88.0,
        cam_phi_deg=140.0,
        cam_theta_deg=58.0,
        fov_y_deg=46.0,
        camera_mode="mesh",
        clip=(0.1, 880.0),
        light_azimuth_deg=136.0,
        light_elevation_deg=26.0,
        sun_intensity=2.6,
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
    )
    config.cam_target = [0.0, 0.0, 0.0]
    return renderer.render_terrain_pbr_pom(
        f3d.MaterialSet.terrain_default(),
        _hdr(),
        f3d.TerrainRenderParams(config),
        heightmap,
    ).to_numpy()


def render_tv21_demo(
    *,
    dem_path: str | Path = DEFAULT_DEM,
    output_dir: str | Path,
    width: int = 960,
    height: int = 600,
    max_dem_size: int = 768,
    crop_size: int = 160,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    normalized = load_dem(Path(dem_path), int(max_dem_size))
    side = min(int(crop_size), *normalized.shape)
    row = (normalized.shape[0] - side) // 2
    column = (normalized.shape[1] - side) // 2
    heightmap = np.ascontiguousarray(
        20.0 + normalized[row : row + side, column : column + side] * 40.0,
        dtype=np.float32,
    )
    source = TerrainScatterSource(heightmap, z_scale=1.0)
    summaries = []
    contact_rows = []
    for case in _cases(source):
        base = TerrainScatterBatch(
            name=case.name,
            color=case.color,
            transforms=case.transforms.copy(),
            levels=[TerrainScatterLevel(mesh=case.mesh)],
        )
        enabled = TerrainScatterBatch(
            name=case.name,
            color=case.color,
            transforms=case.transforms.copy(),
            terrain_blend=case.blend,
            terrain_contact=case.contact,
            levels=[TerrainScatterLevel(mesh=case.mesh)],
        )
        baseline = _render(heightmap, base, width=width, height=height)
        tv21 = _render(heightmap, enabled, width=width, height=height)
        difference = diff_image(baseline, tv21)
        delta = np.abs(tv21[..., :3].astype(np.int16) - baseline[..., :3].astype(np.int16))
        case_dir = output_dir / case.name
        summaries.append(
            {
                "name": case.name,
                "baseline_path": str(save(case_dir / "baseline.png", baseline)),
                "tv21_path": str(save(case_dir / "tv21.png", tv21)),
                "diff_path": str(save(case_dir / "diff.png", difference)),
                "changed_pixels": int(np.count_nonzero(np.any(delta > 0, axis=-1))),
                "mean_delta": float(delta.mean()),
            }
        )
        contact_rows.append(np.concatenate((baseline, tv21, difference), axis=1))
    contact_sheet = np.concatenate(contact_rows, axis=0)
    return {
        "cases": summaries,
        "contact_sheet_path": str(save(output_dir / "contact-sheet.png", contact_sheet)),
    }
