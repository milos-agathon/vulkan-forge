"""Canonical participating-media configuration and reference transport."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class MediaError(ValueError):
    """Structured validation/transport failure at the media API boundary."""


def _native_module() -> Any:
    try:
        from . import _forge3d
    except Exception as error:  # pragma: no cover - native import diagnostic
        raise MediaError("forge3d.media requires the native extension") from error
    return _forge3d


class Medium:
    """Validated canonical medium shared by reference and real-time terrain paths."""

    def __init__(
        self,
        sigma_a: Sequence[float],
        sigma_s: Sequence[float],
        density: float = 1.0,
        *,
        phase: str = "isotropic",
        g: float = 0.0,
        density_scale: float = 1.0,
        version: int = 0,
    ) -> None:
        try:
            self._native = _native_module().Medium.homogeneous(
                sigma_a,
                sigma_s,
                float(density),
                phase=phase,
                g=float(g),
                density_scale=float(density_scale),
                version=int(version),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise MediaError(str(error)) from error

    @classmethod
    def homogeneous(cls, *args: Any, **kwargs: Any) -> "Medium":
        return cls(*args, **kwargs)

    @classmethod
    def _from_native(cls, native: Any) -> "Medium":
        """Wrap an existing native medium without reconstructing it."""
        instance = cls.__new__(cls)
        instance._native = native
        return instance

    @classmethod
    def grid3d(
        cls,
        sigma_a: Sequence[float],
        sigma_s: Sequence[float],
        density: Any,
        bounds: tuple[Sequence[float], Sequence[float]],
        **kwargs: Any,
    ) -> "Medium":
        instance = cls.__new__(cls)
        try:
            instance._native = _native_module().Medium.grid3d(
                sigma_a,
                sigma_s,
                np.ascontiguousarray(density, dtype=np.float32),
                bounds,
                **kwargs,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise MediaError(str(error)) from error
        return instance

    @classmethod
    def perlin_worley(
        cls,
        sigma_a: Sequence[float],
        sigma_s: Sequence[float],
        bounds: tuple[Sequence[float], Sequence[float]],
        **kwargs: Any,
    ) -> "Medium":
        instance = cls.__new__(cls)
        try:
            instance._native = _native_module().Medium.perlin_worley(
                sigma_a, sigma_s, bounds, **kwargs
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise MediaError(str(error)) from error
        return instance

    @property
    def sigma_a(self) -> tuple[float, float, float]:
        return tuple(self._native.sigma_a)

    @property
    def sigma_s(self) -> tuple[float, float, float]:
        return tuple(self._native.sigma_s)

    @property
    def sigma_t(self) -> tuple[float, float, float]:
        return tuple(self._native.sigma_t)

    @property
    def version(self) -> int:
        return int(self._native.version)

    @property
    def identity(self) -> str:
        return str(self._native.identity)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._native.to_dict())


def render_volumetric_reference(
    medium: Medium,
    heightmap: Any,
    width: int,
    height: int,
    camera: Mapping[str, Any],
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    exaggeration: float = 1.0,
    albedo: tuple[float, float, float] = (0.6, 0.6, 0.6),
    sun_azimuth_deg: float = 315.0,
    sun_elevation_deg: float = 45.0,
    sun_intensity: float = 2.5,
    sun_color: tuple[float, float, float] = (1.0, 0.97, 0.92),
    environment_intensity: float = 0.35,
    samples_per_pixel: int = 1,
    homogeneous_medium_reach: float,
    clip: tuple[float, float] = (0.1, 6000.0),
    seed: int = 7,
    certificate: bool | str = False,
    cache: Any = None,
) -> Mapping[str, Any]:
    """Render integrated terrain/media beauty and four reference AOVs.

    ``certificate`` follows the other GPU pixel producers: ``True`` retains
    the signed execution certificate in memory, while a path writes it.
    """
    if not isinstance(medium, Medium):
        raise TypeError("medium must be forge3d.media.Medium")
    _ = cache
    try:
        return _native_module()._render_volumetric_reference(
            medium._native,
            np.ascontiguousarray(heightmap, dtype=np.float32),
            int(width),
            int(height),
            dict(camera),
            spacing=spacing,
            exaggeration=float(exaggeration),
            albedo=albedo,
            sun_azimuth_deg=float(sun_azimuth_deg),
            sun_elevation_deg=float(sun_elevation_deg),
            sun_intensity=float(sun_intensity),
            sun_color=sun_color,
            environment_intensity=float(environment_intensity),
            samples_per_pixel=int(samples_per_pixel),
            homogeneous_medium_reach=float(homogeneous_medium_reach),
            clip=clip,
            seed=int(seed),
            certificate=certificate,
            cache=cache,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise MediaError(str(error)) from error


__all__ = ["MediaError", "Medium", "render_volumetric_reference"]
