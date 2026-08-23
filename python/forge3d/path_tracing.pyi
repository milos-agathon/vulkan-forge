# python/forge3d/path_tracing.pyi
# Type stubs for path tracing API skeleton (Workstream A).
# This exists to keep the package typed and IDE-friendly.
# RELEVANT FILES:python/forge3d/path_tracing.py,python/forge3d/__init__.pyi,tests/test_path_tracing_api.py

from __future__ import annotations
from typing import Tuple, Iterable, Dict, Mapping, Callable, Any, Optional, Sequence
import numpy as np
from . import AtmosphereLutHandle
from .atmosphere import AtmosphereSettings

class ExperimentalSyntheticOutput(RuntimeError): ...

class PathTracer:
    def __init__(self, width: int, height: int, *, max_bounces: int = ..., seed: int = ..., tile: int | None = ...) -> None: ...
    @property
    def size(self) -> Tuple[int, int]: ...
    def enable_scene_cache(self, enabled: bool = ..., *, capacity: int | None = ...) -> None: ...
    def reset_scene_cache(self) -> None: ...
    def cache_stats(self) -> Dict[str, int]: ...
    def render_rgba(self, *args: Any, spp: int = ..., luminance_clamp: float | None = ..., firefly_clamp: float | None = ..., synthetic_ok: bool = ..., certificate: bool | str | None = ..., cache: str | None = ..., **kwargs: Any) -> np.ndarray: ...
    def add_sphere(self, center: tuple[float, float, float], radius: float, material_or_color) -> None: ...
    def add_triangle(self, v0: tuple[float, float, float], v1: tuple[float, float, float], v2: tuple[float, float, float], material_or_color) -> None: ...
    def render_progressive(
        self,
        *,
        callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = ..., 
        tile_size: int | None = ..., 
        min_updates_per_sec: float = ..., 
        time_source: Callable[[], float] = ..., 
        spp: int = ..., 
        synthetic_ok: bool = ...,
        certificate: bool | str | None = ...,
        cache: str | None = ...,
    ) -> np.ndarray: ...

def create_path_tracer(width: int, height: int, *, max_bounces: int = ..., seed: int = ...) -> PathTracer: ...
def render_rgba(*args: Any, certificate: bool | str | None = ..., cache: str | None = ..., **kwargs: Any) -> np.ndarray: ...
def _fresnel_schlick(cos_theta: np.ndarray, F0: np.ndarray) -> np.ndarray: ...

# AOV API (A14)
def render_aovs(
    width: int,
    height: int,
    scene,
    camera: dict | None = ..., 
    *,
    aovs: Iterable[str] = ..., 
    seed: int = ..., 
    frames: int = ..., 
    use_gpu: bool = ..., 
    synthetic_ok: bool = ...,
    certificate: bool | str | None = ...,
    cache: str | None = ...,
) -> Dict[str, np.ndarray]: ...

def save_aovs(
    aovs_map: Mapping[str, np.ndarray], 
    basename: str, 
    *, 
    output_dir: str | None = ..., 
) -> Dict[str, str]: ...

def iter_tiles(width: int, height: int, tile: int) -> Iterable[Tuple[int, int, int, int]]: ...

# ``earth_model="flat"`` requires ``refraction_model="none"``; unsupported
# model names or pairs raise an exception and never silently fall back.
def hybrid_render_terrain_reference(
    heightmap: np.ndarray,
    width: int,
    height: int,
    camera: dict | None = ...,
    *,
    spacing: Tuple[float, float] = ...,
    exaggeration: float = ...,
    albedo: Tuple[float, float, float] = ...,
    sun_azimuth_deg: float | None = ...,
    sun_elevation_deg: float | None = ...,
    solar_time: object | None = ...,
    sun_intensity: float = ...,
    sun_color: Sequence[float] | np.ndarray = ...,
    env_map: np.ndarray | None = ...,
    env_intensity: float = ...,
    mesh_vertices: np.ndarray | None = ...,
    mesh_indices: np.ndarray | None = ...,
    spp: int = ...,
    max_frames: int = ...,
    min_frames: int = ...,
    variance_threshold: float = ...,
    seed: int = ...,
    certificate: bool | str | PathLikeStr | None = ...,
    cache: str | PathLikeStr | None = ...,
    observer_latitude_deg: float | None = ...,
    observer_longitude_deg: float | None = ...,
    earth_model: str = ...,
    sphere_radius_m: float = ...,
    refraction_model: str = ...,
    refraction_k: float = ...,
    pressure_mbar: float | None = ...,
    temperature_c: float | None = ...,
    atmosphere: AtmosphereSettings | Mapping[str, object] | AtmosphereLutHandle | None = ...,
) -> Dict[str, object]: ...
