"""Ephemeris-correct positions for SIDERA's declared 2000–2050 window."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple

from ._native import get_native_module

__all__ = [
    "body_position",
    "moon_phase",
    "delta_t_seconds",
    "sidereal_time",
    "refraction_arcminutes",
]


def _utc_components(value: datetime) -> Tuple[int, int, int, int, int, float]:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime_utc must be a timezone-aware datetime")
    value = value.astimezone(timezone.utc)
    return (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second + value.microsecond / 1_000_000.0,
    )


def _native():
    module = get_native_module()
    if module is None:
        raise RuntimeError("forge3d.astro requires the native extension")
    return module


def body_position(
    body: str,
    datetime_utc: datetime,
    lat: float,
    lon: float,
    *,
    height_m: float = 0.0,
    refraction: bool = False,
) -> Tuple[float, float, float]:
    """Return apparent topocentric ``(azimuth_deg, altitude_deg, distance_au)``."""
    return _native().astro_body_position(
        body,
        *_utc_components(datetime_utc),
        float(lat),
        float(lon),
        float(height_m),
        bool(refraction),
    )


def moon_phase(datetime_utc: datetime) -> Tuple[float, float, float]:
    """Return ``(illuminated_fraction, phase_angle_deg, semidiameter_arcsec)``."""
    return _native().astro_moon_phase(*_utc_components(datetime_utc))


def delta_t_seconds(datetime_utc: datetime) -> float:
    """Return ΔT = TT − UT1 in seconds from the committed piecewise-linear fit.

    Beyond the published IERS series this follows JPL Horizons' convention of
    holding UT1−UTC fixed; it is a shared convention, not a prediction of Earth
    rotation. See the module docs of ``src/astro/time.rs``.
    """
    return _native().astro_delta_t_seconds(*_utc_components(datetime_utc))


def sidereal_time(datetime_utc: datetime) -> Tuple[float, float]:
    """Return Greenwich ``(mean, apparent)`` sidereal time in degrees (IAU 2006)."""
    return _native().astro_sidereal_time(*_utc_components(datetime_utc))


def refraction_arcminutes(true_altitude_deg: float) -> float:
    """Return Sæmundsson (1986) refraction in arcminutes for a *true* altitude.

    Declared standard atmosphere: 1010 mbar at 10 °C. Outside ``-1° ≤ h < 89°``
    the fit is not valid and zero is returned rather than an extrapolation;
    non-finite altitudes raise ``ValueError``.
    """
    return _native().astro_refraction_arcminutes(float(true_altitude_deg))
