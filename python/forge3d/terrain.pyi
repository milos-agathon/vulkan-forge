from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, overload
import numpy as np
from numpy.typing import NDArray
from .geo import SolarTime

# Model contract for every HELIOS function below: ``earth_model="flat"``
# requires ``refraction_model="none"``; unsupported names or pairs raise
# an exception and never silently fall back.

@overload
def viewshed(
    dem: NDArray[np.floating],
    observer: tuple[float, float] | tuple[float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    height_system: Literal["ellipsoidal", "orthometric_egm96"],
    observer_height: float = ...,
    target_height: float = ...,
    max_distance: float | None = ...,
    earth_model: Literal["flat", "sphere", "ellipsoid", "wgs84"] = ...,
    sphere_radius_m: float = ...,
    refraction_model: Literal["none", "bennett", "saemundsson", "effective_radius"] = ...,
    refraction_k: float = ...,
    pressure_mbar: float = ...,
    temperature_c: float = ...,
    return_diagnostics: Literal[False] = ...,
) -> NDArray[np.bool_]: ...

@overload
def viewshed(
    dem: NDArray[np.floating],
    observer: tuple[float, float] | tuple[float, float, float],
    *,
    bounds: tuple[float, float, float, float],
    height_system: Literal["ellipsoidal", "orthometric_egm96"],
    observer_height: float = ...,
    target_height: float = ...,
    max_distance: float | None = ...,
    earth_model: Literal["flat", "sphere", "ellipsoid", "wgs84"] = ...,
    sphere_radius_m: float = ...,
    refraction_model: Literal["none", "bennett", "saemundsson", "effective_radius"] = ...,
    refraction_k: float = ...,
    pressure_mbar: float = ...,
    temperature_c: float = ...,
    return_diagnostics: Literal[True],
) -> dict[str, NDArray[np.bool_] | NDArray[np.float32]]: ...

def shadow_mask(
    dem: NDArray[np.floating],
    solar_time: SolarTime | Mapping[str, object],
    *,
    bounds: tuple[float, float, float, float],
    height_system: Literal["ellipsoidal", "orthometric_egm96"],
    earth_model: Literal["flat", "sphere", "ellipsoid", "wgs84"] = ...,
    sphere_radius_m: float = ...,
    refraction_model: Literal["none", "bennett", "saemundsson", "effective_radius"] = ...,
    refraction_k: float = ...,
) -> NDArray[np.bool_]: ...

def shadow_tip(
    dem: NDArray[np.floating],
    peak_lat: float,
    peak_lon: float,
    solar_time: SolarTime | Mapping[str, object],
    *,
    bounds: tuple[float, float, float, float],
    height_system: Literal["ellipsoidal", "orthometric_egm96"],
    earth_model: Literal["flat", "sphere", "ellipsoid", "wgs84"] = ...,
    sphere_radius_m: float = ...,
    refraction_model: Literal["none", "bennett", "saemundsson", "effective_radius"] = ...,
    refraction_k: float = ...,
) -> dict[str, float]: ...
