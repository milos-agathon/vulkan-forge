"""Shared file-backed rendering pieces for the TV terrain examples."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.terrain_params import (
    MaterialLayerSettings,
    PomSettings,
    ReflectionProbeSettings,
    make_terrain_params_config,
)


ROOT = Path(__file__).resolve().parents[1]


def load_dem(path: Path, max_size: int) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as dataset:
        dem = dataset.read(1).astype(np.float32)
        if dataset.nodata is not None:
            dem[dem == dataset.nodata] = np.nan
    if not np.isfinite(dem).any():
        raise ValueError(f"DEM contains no finite heights: {path}")
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    if max_size > 0 and max(dem.shape) > max_size:
        stride = int(np.ceil(max(dem.shape) / max_size))
        dem = dem[::stride, ::stride]
    dem -= float(dem.min())
    dem /= max(float(dem.max()), 1.0e-6)
    return np.ascontiguousarray(dem, dtype=np.float32)


def terrain_overlay() -> f3d.OverlayLayer:
    cmap = f3d.Colormap1D.from_stops(
        [
            (0.0, "#17351b"),
            (0.35, "#526d38"),
            (0.68, "#9a8258"),
            (1.0, "#eef3f8"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _ibl(intensity: float):
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as handle:
        path = Path(handle.name)
    try:
        pixels = bytearray()
        for y in range(4):
            for x in range(8):
                pixels.extend((48 + 24 * x, 64 + 32 * y, 164, 128))
        path.write_bytes(
            b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n" + pixels
        )
        return f3d.IBL.from_hdr(str(path), intensity=float(intensity))
    finally:
        path.unlink(missing_ok=True)


def render(
    dem: np.ndarray,
    width: int,
    height: int,
    *,
    materials: MaterialLayerSettings | None = None,
    reflection_probes: ReflectionProbeSettings | None = None,
    debug_mode: int = 0,
    water_mask: np.ndarray | None = None,
    albedo_mode: str = "colormap",
    colormap_strength: float = 1.0,
    terrain_span: float = 4.0,
    z_scale: float = 1.6,
    light_azimuth_deg: float = 136.0,
    light_elevation_deg: float = 18.0,
    sun_intensity: float = 2.4,
    exposure: float = 1.0,
    cam_radius: float = 5.8,
    cam_phi_deg: float = 140.0,
    cam_theta_deg: float = 58.0,
    fov_y_deg: float = 50.0,
    camera_mode: str = "screen",
    ibl_intensity: float = 1.0,
) -> tuple[np.ndarray, f3d.TerrainRenderer]:
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    config = make_terrain_params_config(
        size_px=(int(width), int(height)),
        render_scale=1.0,
        terrain_span=float(terrain_span),
        msaa_samples=1,
        z_scale=float(z_scale),
        exposure=float(exposure),
        domain=(0.0, 1.0),
        albedo_mode=albedo_mode,
        colormap_strength=float(colormap_strength),
        ibl_enabled=True,
        ibl_intensity=float(ibl_intensity),
        light_azimuth_deg=float(light_azimuth_deg),
        light_elevation_deg=float(light_elevation_deg),
        sun_intensity=float(sun_intensity),
        cam_radius=float(cam_radius),
        cam_phi_deg=float(cam_phi_deg),
        cam_theta_deg=float(cam_theta_deg),
        fov_y_deg=float(fov_y_deg),
        camera_mode=camera_mode,
        debug_mode=int(debug_mode),
        overlays=[terrain_overlay()],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        materials=materials,
        reflection_probes=reflection_probes,
    )
    frame = renderer.render_terrain_pbr_pom(
        f3d.MaterialSet.terrain_default(),
        _ibl(ibl_intensity),
        f3d.TerrainRenderParams(config),
        dem,
        water_mask=water_mask,
    )
    return np.asarray(frame.to_numpy()), renderer


def save(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(path, np.ascontiguousarray(image, dtype=np.uint8))
    return path


def side_by_side(left: np.ndarray, right: np.ndarray, *, gap: int = 12) -> np.ndarray:
    height = max(left.shape[0], right.shape[0])
    out = np.zeros((height, left.shape[1] + gap + right.shape[1], 4), dtype=np.uint8)
    out[..., 3] = 255
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] + gap :] = right
    return out


def diff_image(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.abs(left[..., :3].astype(np.int16) - right[..., :3].astype(np.int16))
    out = np.empty_like(left)
    out[..., :3] = np.clip(delta * 4, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out
