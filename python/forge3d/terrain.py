"""Curvature-aware terrain analysis and out-of-core terrain data handles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np

from ._native import get_native_module, native_import_error
from .geo import SolarTime, _coerce_solar_time


def _solar_time_payload(solar_time: SolarTime | Mapping[str, object]) -> dict[str, object]:
    return _coerce_solar_time(solar_time).to_native()


def _terrain_native(name: str) -> Any:
    native = get_native_module()
    if native is None:
        cause = native_import_error()
        detail = f": {cause}" if cause is not None else ""
        raise RuntimeError(f"forge3d native extension is unavailable{detail}")
    if not hasattr(native, name):
        raise RuntimeError(f"forge3d native extension does not provide {name}; rebuild the extension")
    return native


def viewshed(
    dem: np.ndarray,
    observer: tuple[float, float] | tuple[float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    height_system: str,
    observer_height: float = 1.7,
    target_height: float = 0.0,
    max_distance: float | None = None,
    earth_model: str = "ellipsoid",
    sphere_radius_m: float = 6_371_008.8,
    refraction_model: str = "bennett",
    refraction_k: float = 0.13,
    pressure_mbar: float = 1013.25,
    temperature_c: float = 15.0,
    return_diagnostics: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """Compute a GPU visibility raster for an EPSG:4326 north-up DEM.

    ``earth_model`` selects flat, spherical, or WGS84 ellipsoidal curvature;
    ``refraction_model`` selects the atmospheric/effective-radius correction.
    ``earth_model='flat'`` requires ``refraction_model='none'``. Unsupported
    model names or model pairs raise an exception; they never fall back.
    """
    native = _terrain_native("terrain_viewshed")
    if len(observer) == 3:
        observer_height = float(observer[2])
    elif len(observer) != 2:
        raise ValueError("observer must be (lat, lon) or (lat, lon, h_agl)")
    result: dict[str, Any] = native.terrain_viewshed(
        np.ascontiguousarray(dem, dtype=np.float32),
        (float(observer[0]), float(observer[1])),
        bounds,
        height_system,
        observer_height=observer_height,
        target_height=target_height,
        max_distance=max_distance,
        earth_model=earth_model,
        sphere_radius_m=sphere_radius_m,
        refraction_model=refraction_model,
        refraction_k=refraction_k,
        pressure_mbar=pressure_mbar,
        temperature_c=temperature_c,
    )
    arrays = {
        key: np.asarray(value, dtype=np.bool_ if key == "visibility" else np.float32)
        for key, value in result.items()
    }
    return arrays if return_diagnostics else arrays["visibility"]


def shadow_mask(
    dem: np.ndarray,
    solar_time: SolarTime | Mapping[str, object],
    *,
    bounds: tuple[float, float, float, float],
    height_system: str,
    earth_model: str = "ellipsoid",
    sphere_radius_m: float = 6_371_008.8,
    refraction_model: str = "bennett",
    refraction_k: float = 0.13,
) -> np.ndarray:
    """Return DEM-local terrain-to-sun visibility (``True`` means lit).

    Terrain outside ``bounds`` is outside the analysis domain and therefore
    cannot occlude; the mask answers whether this DEM contains a blocker.
    ``earth_model='flat'`` requires ``refraction_model='none'``. Unsupported
    model names or model pairs raise an exception; they never fall back.
    """
    native = _terrain_native("terrain_shadow_mask")
    result = native.terrain_shadow_mask(
        np.ascontiguousarray(dem, dtype=np.float32),
        _solar_time_payload(solar_time),
        bounds,
        height_system,
        earth_model=earth_model,
        sphere_radius_m=sphere_radius_m,
        refraction_model=refraction_model,
        refraction_k=refraction_k,
    )
    return np.asarray(result, dtype=np.bool_)


def shadow_tip(
    dem: np.ndarray,
    peak_lat: float,
    peak_lon: float,
    solar_time: SolarTime | Mapping[str, object],
    *,
    bounds: tuple[float, float, float, float],
    height_system: str,
    earth_model: str = "ellipsoid",
    sphere_radius_m: float = 6_371_008.8,
    refraction_model: str = "bennett",
    refraction_k: float = 0.13,
) -> dict[str, float]:
    """Return the curved-Earth terminus of a peak's direct solar shadow.

    ``earth_model='flat'`` requires ``refraction_model='none'``. Unsupported
    model names or model pairs raise an exception; they never fall back.
    """
    native = _terrain_native("terrain_shadow_tip")
    result = native.terrain_shadow_tip(
        np.ascontiguousarray(dem, dtype=np.float32),
        float(peak_lat),
        float(peak_lon),
        _solar_time_payload(solar_time),
        bounds,
        height_system,
        earth_model=earth_model,
        sphere_radius_m=sphere_radius_m,
        refraction_model=refraction_model,
        refraction_k=refraction_k,
    )
    return {str(key): float(value) for key, value in result.items()}


@dataclass(frozen=True)
class VTStore:
    """Validated handle to a forge3d ``.f3dvt`` packed page store."""

    path: str

    def __fspath__(self) -> str:
        return self.path


def open_vt_store(path: str | PathLike[str]) -> VTStore:
    """Open a disk-backed TESSELLA virtual-texture store.

    The native renderer performs full header, directory, and per-page SHA-256
    validation when the handle is first used. This lightweight boundary check
    catches wrong paths and formats before a GPU render begins.
    """

    resolved = Path(fspath(path)).expanduser().resolve()
    with resolved.open("rb") as stream:
        magic = stream.read(8)
    if magic != b"F3DVT1\0\0":
        raise ValueError(f"{resolved} is not a forge3d VT store")
    return VTStore(str(resolved))


__all__ = ["VTStore", "open_vt_store", "shadow_mask", "shadow_tip", "viewshed"]
