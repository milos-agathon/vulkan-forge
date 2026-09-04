"""Independent spectral radiance oracle for CPU diagnostics and goldens.

This test-only implementation deliberately imports no forge3d production
module. It owns its spectral constants, spherical geometry, and 64-step
quadrature, but it is not an input to the hard PROMETHEUS closure. That lane
uses native stochastic spectral transport with an explicit black environment.
"""

from __future__ import annotations

import math

import numpy as np


WAVELENGTHS_NM = np.asarray(
    [380.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 660.0, 700.0, 740.0, 780.0],
    dtype=np.float64,
)
CIE_1931_XYZ = np.asarray(
    [
        [0.001368, 0.000039, 0.006450],
        [0.134380, 0.004000, 0.645600],
        [0.290800, 0.060000, 1.669200],
        [0.004900, 0.323000, 0.272000],
        [0.290400, 0.954000, 0.020300],
        [0.916300, 0.870000, 0.001650],
        [0.854450, 0.381000, 0.000190],
        [0.164900, 0.061000, 0.000000],
        [0.011359, 0.004102, 0.000000],
        [0.000690, 0.000249, 0.000000],
        [0.000042, 0.000015, 0.000000],
    ],
    dtype=np.float64,
)
XYZ_TO_LINEAR_SRGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

BOTTOM_RADIUS_M = 6_360_000.0
TOP_RADIUS_M = 6_460_000.0
RAYLEIGH_SCALE_HEIGHT_M = 8_000.0
MIE_SCALE_HEIGHT_M = 1_200.0
GROUND_ALBEDO = 0.3
SEA_LEVEL_NUMBER_DENSITY_M3 = 2.546899e25
RAYLEIGH_CROSS_SECTION_550_M2 = 5.10e-31
MIE_SINGLE_SCATTERING_ALBEDO = 0.9
REFERENCE_STEPS = 64
SUN_TRANSMITTANCE_STEPS = 64

_TRAPEZOID_WEIGHTS = np.ones(WAVELENGTHS_NM.shape, dtype=np.float64)
_TRAPEZOID_WEIGHTS[[0, -1]] = 0.5
_WHITE_XYZ = np.sum(CIE_1931_XYZ * _TRAPEZOID_WEIGHTS[:, None], axis=0)
_WHITE_RGB = XYZ_TO_LINEAR_SRGB @ _WHITE_XYZ


def _density(altitude_m: np.ndarray | float, ozone_du: float) -> np.ndarray:
    altitude = np.maximum(np.asarray(altitude_m, dtype=np.float64), 0.0)
    rayleigh = np.exp(-altitude / RAYLEIGH_SCALE_HEIGHT_M)
    mie = np.exp(-altitude / MIE_SCALE_HEIGHT_M)
    ozone = np.maximum(1.0 - np.abs((altitude - 25_000.0) / 15_000.0), 0.0)
    ozone *= float(ozone_du) / 300.0
    return np.stack((rayleigh, mie, ozone), axis=-1)


def _distance_to_top(altitude_m: float, mu: float) -> float:
    radius = BOTTOM_RADIUS_M + float(np.clip(altitude_m, 0.0, TOP_RADIUS_M - BOTTOM_RADIUS_M))
    radial = radius * mu
    discriminant = radial * radial + (TOP_RADIUS_M - radius) * (TOP_RADIUS_M + radius)
    return max(-radial + math.sqrt(max(discriminant, 0.0)), 0.0)


def _distance_to_ground(altitude_m: float, mu: float) -> float | None:
    if mu >= 0.0:
        return None
    radius = BOTTOM_RADIUS_M + float(np.clip(altitude_m, 0.0, TOP_RADIUS_M - BOTTOM_RADIUS_M))
    radial = radius * mu
    discriminant = radial * radial - (radius - BOTTOM_RADIUS_M) * (
        radius + BOTTOM_RADIUS_M
    )
    if discriminant < 0.0:
        return None
    distance = -radial - math.sqrt(discriminant)
    return distance if distance >= 0.0 else None


def _distance_to_boundary(altitude_m: float, mu: float) -> float:
    ground = _distance_to_ground(altitude_m, mu)
    return _distance_to_top(altitude_m, mu) if ground is None else ground


def _altitude_along(altitude_m: float, mu: float, distance_m: np.ndarray | float) -> np.ndarray:
    radius = BOTTOM_RADIUS_M + float(np.clip(altitude_m, 0.0, TOP_RADIUS_M - BOTTOM_RADIUS_M))
    distance = np.asarray(distance_m, dtype=np.float64)
    radial = np.sqrt(np.maximum(radius * radius + distance * distance + 2.0 * radius * mu * distance, 0.0))
    return radial - BOTTOM_RADIUS_M


def _optical_columns(altitude_m: float, mu: float, distance_m: float, steps: int, ozone_du: float) -> np.ndarray:
    if distance_m <= 0.0:
        return np.zeros(3, dtype=np.float64)
    step_m = distance_m / float(steps)
    distances = (np.arange(steps, dtype=np.float64) + 0.5) * step_m
    return np.sum(_density(_altitude_along(altitude_m, mu, distances), ozone_du), axis=0) * step_m


def _rayleigh_beta() -> np.ndarray:
    return (
        RAYLEIGH_CROSS_SECTION_550_M2
        * SEA_LEVEL_NUMBER_DENSITY_M3
        * np.power(550.0 / WAVELENGTHS_NM, 4.0)
    )


def _mie_extinction(turbidity: float) -> np.ndarray:
    return 1.0e-5 * float(turbidity) * (550.0 / WAVELENGTHS_NM)


def _ozone_absorption() -> np.ndarray:
    return 1.2e-6 * np.exp(-0.5 * np.square((WAVELENGTHS_NM - 600.0) / 85.0))


def _transmittance(columns: np.ndarray, turbidity: float) -> np.ndarray:
    optical_depth = (
        _rayleigh_beta() * columns[0]
        + _mie_extinction(turbidity) * columns[1]
        + _ozone_absorption() * columns[2]
    )
    return np.exp(-np.maximum(optical_depth, 0.0))


def _attenuated_cell_length(
    density: np.ndarray, step_m: float, turbidity: float
) -> np.ndarray:
    extinction = (
        _rayleigh_beta() * density[0]
        + _mie_extinction(turbidity) * density[1]
        + _ozone_absorption() * density[2]
    )
    result = np.full(extinction.shape, float(step_m), dtype=np.float64)
    active = extinction > 1.0e-15
    result[active] = -np.expm1(-extinction[active] * step_m) / extinction[active]
    return result


def _rayleigh_phase(cosine: float) -> float:
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return 3.0 * (1.0 + cosine * cosine) / (16.0 * math.pi)


def _mie_phase(cosine: float, mie_g: float) -> float:
    cosine = float(np.clip(cosine, -1.0, 1.0))
    g = float(np.clip(mie_g, -0.999, 0.999))
    denominator = max(1.0 + g * g - 2.0 * g * cosine, 1.0e-6) ** 1.5
    return 3.0 * (1.0 - g * g) * (1.0 + cosine * cosine) / (
        8.0 * math.pi * (2.0 + g * g) * denominator
    )


def _spectral_to_linear_rgb(spectrum: np.ndarray) -> np.ndarray:
    xyz = np.sum(
        np.asarray(spectrum, dtype=np.float64)[:, None]
        * CIE_1931_XYZ
        * _TRAPEZOID_WEIGHTS[:, None],
        axis=0,
    )
    return (XYZ_TO_LINEAR_SRGB @ xyz) / _WHITE_RGB


def _sky_radiance(
    view_direction: np.ndarray,
    sun_direction: np.ndarray,
    *,
    observer_altitude_m: float,
    turbidity: float,
    ozone_du: float,
    mie_g: float,
) -> np.ndarray:
    view = np.asarray(view_direction, dtype=np.float64)
    sun = np.asarray(sun_direction, dtype=np.float64)
    view /= np.linalg.norm(view)
    sun /= np.linalg.norm(sun)
    mu_view = float(view[1])
    mu_sun = float(sun[1])
    relative_cosine = float(np.dot(view, sun))
    distance_m = _distance_to_boundary(observer_altitude_m, mu_view)
    if distance_m <= 0.0:
        return np.zeros(3, dtype=np.float64)

    step_m = distance_m / REFERENCE_STEPS
    view_columns = np.zeros(3, dtype=np.float64)
    single = np.zeros(WAVELENGTHS_NM.shape, dtype=np.float64)
    rayleigh_scatter = _rayleigh_beta() * _rayleigh_phase(relative_cosine)
    mie_scatter = (
        _mie_extinction(turbidity)
        * MIE_SINGLE_SCATTERING_ALBEDO
        * _mie_phase(relative_cosine, mie_g)
    )
    for index in range(REFERENCE_STEPS):
        sample_distance = (index + 0.5) * step_m
        sample_position = np.asarray(
            [
                view[0] * sample_distance,
                BOTTOM_RADIUS_M + observer_altitude_m + view[1] * sample_distance,
                view[2] * sample_distance,
            ],
            dtype=np.float64,
        )
        sample_radius = float(np.linalg.norm(sample_position))
        sample_altitude = sample_radius - BOTTOM_RADIUS_M
        local_mu_sun = float(np.dot(sun, sample_position / sample_radius))
        density = _density(sample_altitude, ozone_du)
        view_start = _transmittance(view_columns, turbidity)
        if _distance_to_ground(sample_altitude, local_mu_sun) is None:
            sun_distance = _distance_to_top(sample_altitude, local_mu_sun)
            sun_columns = _optical_columns(
                sample_altitude,
                local_mu_sun,
                sun_distance,
                SUN_TRANSMITTANCE_STEPS,
                ozone_du,
            )
            single += (
                view_start
                * _transmittance(sun_columns, turbidity)
                * (rayleigh_scatter * density[0] + mie_scatter * density[1])
                * _attenuated_cell_length(density, step_m, turbidity)
            )
        view_columns += density * step_m

    ground_bounce = np.zeros(WAVELENGTHS_NM.shape, dtype=np.float64)
    if GROUND_ALBEDO > 0.0 and mu_sun > 0.0:
        ground_sun = _transmittance(
            _optical_columns(
                0.0,
                mu_sun,
                _distance_to_top(0.0, mu_sun),
                SUN_TRANSMITTANCE_STEPS,
                ozone_du,
            ),
            turbidity,
        )
        ground_source = GROUND_ALBEDO * mu_sun * ground_sun / math.pi
        view_columns.fill(0.0)
        bounce_cosine = float(np.clip(-mu_view, -1.0, 1.0))
        bounce_rayleigh = _rayleigh_beta() * _rayleigh_phase(bounce_cosine)
        bounce_mie = (
            _mie_extinction(turbidity)
            * MIE_SINGLE_SCATTERING_ALBEDO
            * _mie_phase(bounce_cosine, mie_g)
        )
        for index in range(REFERENCE_STEPS):
            sample_distance = (index + 0.5) * step_m
            sample_altitude = float(
                _altitude_along(observer_altitude_m, mu_view, sample_distance)
            )
            density = _density(sample_altitude, ozone_du)
            view_start = _transmittance(view_columns, turbidity)
            vertical_ground = _transmittance(
                _optical_columns(
                    0.0,
                    1.0,
                    max(sample_altitude, 0.0),
                    SUN_TRANSMITTANCE_STEPS,
                    ozone_du,
                ),
                turbidity,
            )
            ground_bounce += (
                view_start
                * vertical_ground
                * ground_source
                * (bounce_rayleigh * density[0] + bounce_mie * density[1])
                * _attenuated_cell_length(density, step_m, turbidity)
            )
            view_columns += density * step_m

    return np.maximum(_spectral_to_linear_rgb(single + ground_bounce), 0.0)


def independent_reference_environment(
    sun_elevation_deg: float,
    width: int = 2,
    height: int = 3,
    *,
    turbidity: float = 2.0,
    ozone_du: float = 300.0,
    mie_g: float = 0.8,
    observer_altitude_m: float = 1.0,
    sun_azimuth_deg: float = 90.0,
) -> np.ndarray:
    """Return a directional equirectangular spectral oracle field."""

    elevation = math.radians(float(sun_elevation_deg))
    azimuth = math.radians(float(sun_azimuth_deg))
    sun = np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.sin(azimuth),
        ],
        dtype=np.float64,
    )
    result = np.empty((height, width, 3), dtype=np.float32)
    for y in range(height):
        latitude = math.pi * 0.5 - math.pi * (y + 0.5) / height
        cos_latitude = math.cos(latitude)
        for x in range(width):
            longitude = 2.0 * math.pi * (x + 0.5) / width - math.pi
            view = np.asarray(
                [
                    cos_latitude * math.cos(longitude),
                    math.sin(latitude),
                    cos_latitude * math.sin(longitude),
                ],
                dtype=np.float64,
            )
            result[y, x] = _sky_radiance(
                view,
                sun,
                observer_altitude_m=observer_altitude_m,
                turbidity=turbidity,
                ozone_du=ozone_du,
                mie_g=mie_g,
            )
    assert np.all(np.isfinite(result)) and np.all(result >= 0.0)
    return result


def independent_reference_radiance(
    view_direction: np.ndarray,
    sun_elevation_deg: float,
    *,
    turbidity: float = 2.0,
    ozone_du: float = 300.0,
    mie_g: float = 0.8,
    observer_altitude_m: float = 1.0,
    sun_azimuth_deg: float = 90.0,
) -> np.ndarray:
    """Evaluate one test-owned spectral reference ray."""

    elevation = math.radians(float(sun_elevation_deg))
    azimuth = math.radians(float(sun_azimuth_deg))
    sun = np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.sin(azimuth),
        ],
        dtype=np.float64,
    )
    return _sky_radiance(
        np.asarray(view_direction, dtype=np.float64),
        sun,
        observer_altitude_m=observer_altitude_m,
        turbidity=turbidity,
        ozone_du=ozone_du,
        mie_g=mie_g,
    )


__all__ = ["independent_reference_environment", "independent_reference_radiance"]
