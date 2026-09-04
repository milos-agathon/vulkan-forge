"""Validated Python helpers for the AETHER spectral atmosphere.

The runtime implementation is native.  This module deliberately provides no
RGB or analytic fallback: if the extension or an AETHER symbol is unavailable,
the call fails with a diagnostic that names the missing native contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from . import AtmosphereLutHandle

from ._native import get_native_module, native_import_error


SUN_ELEVATION_SWEEP_DEG: tuple[float, ...] = (-5.0, 0.0, 5.0, 10.0, 30.0, 60.0, 89.0)
"""Canonical AETHER sunset/reference sweep, in degrees above the horizon."""


@dataclass(frozen=True, slots=True)
class AtmosphereSettings:
    """Physical inputs shared by the shipped LUT and offline bake paths."""

    turbidity: float = 2.0
    ozone_du: float = 300.0
    mie_g: float = 0.8
    ground_albedo: float = 0.3
    scattering_orders: int = 4

    def __post_init__(self) -> None:
        for name in ("turbidity", "ozone_du", "mie_g", "ground_albedo"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 1.0 <= float(self.turbidity) <= 10.0:
            raise ValueError("turbidity must be in [1.0, 10.0]")
        if not 0.0 <= float(self.ozone_du) <= 600.0:
            raise ValueError("ozone_du must be in [0.0, 600.0]")
        if not 0.0 <= float(self.mie_g) <= 0.99:
            raise ValueError("mie_g must be in [0.0, 0.99]")
        if not 0.0 <= float(self.ground_albedo) <= 1.0:
            raise ValueError("ground_albedo must be in [0.0, 1.0]")
        if isinstance(self.scattering_orders, bool) or not isinstance(
            self.scattering_orders, int
        ):
            raise TypeError("scattering_orders must be an integer")
        if not 2 <= self.scattering_orders <= 8:
            raise ValueError("scattering_orders must be in [2, 8]")

    def native_kwargs(self) -> dict[str, float | int]:
        """Return the exact native parameter names used by AETHER entry points."""

        return {
            "turbidity": float(self.turbidity),
            "ozone_du": float(self.ozone_du),
            "mie_g": float(self.mie_g),
            "ground_albedo": float(self.ground_albedo),
            "scattering_orders": self.scattering_orders,
        }


def _native_symbol(name: str) -> Any:
    native = get_native_module()
    if native is None:
        cause = native_import_error()
        detail = f": {cause!r}" if cause is not None else ""
        raise RuntimeError(
            "AETHER requires the compiled forge3d._forge3d extension" + detail
        )
    symbol = getattr(native, name, None)
    if symbol is None:
        raise RuntimeError(
            f"AETHER native contract {name!r} is unavailable in this build; "
            "rebuild the forge3d extension"
        )
    return symbol


def bake_luts(settings: AtmosphereSettings | None = None) -> AtmosphereLutHandle:
    """Resolve or bake LUTs for ``settings`` through the native AETHER path.

    Wheels built with Cargo feature ``atmosphere-bake`` execute the offline
    bake. Minimal runtime builds resolve only shipped, provenance-locked anchor
    tables and reject custom inputs rather than substituting a nearby table.
    The optional ``aerial_perspective_rgba`` froxel stores zero RGB and mean
    finite-segment spectral transmittance in alpha; active in-scatter is always
    derived from the accumulated-scattering LUT.
    """

    resolved = settings if settings is not None else AtmosphereSettings()
    return _native_symbol("atmosphere_bake_luts")(**resolved.native_kwargs())


def spectral_to_linear_rgb(samples: Sequence[float]) -> tuple[float, float, float]:
    """Convert one native wavelength basis sample vector to linear sRGB."""

    values = tuple(float(value) for value in samples)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("samples must be a non-empty finite sequence")
    result = _native_symbol("atmosphere_spectral_to_linear_rgb")(values)
    return (float(result[0]), float(result[1]), float(result[2]))


def generate_environment(
    width: int,
    height: int,
    sun_elevation_deg: float,
    *,
    settings: AtmosphereSettings | None = None,
    mode: str = "lut",
) -> dict[str, Any]:
    """Generate an equirectangular AETHER validation environment.

    ``mode='reference'`` is a CPU transport diagnostic, not a substitute for
    the independent PROMETHEUS reference used by the hard closure gate.
    """

    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError("width and height must be integers")
    if int(width) != width or int(height) != height or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    elevation = float(sun_elevation_deg)
    if not math.isfinite(elevation) or not -90.0 <= elevation <= 90.0:
        raise ValueError("sun_elevation_deg must be finite and in [-90.0, 90.0]")
    if mode not in {"lut", "reference"}:
        raise ValueError("mode must be 'lut' or 'reference'")
    resolved = settings if settings is not None else AtmosphereSettings()
    if resolved.scattering_orders != 4:
        raise ValueError(
            "generate_environment requires scattering_orders=4; custom order counts "
            "are available only through the offline bake handle"
        )
    kwargs = resolved.native_kwargs()
    return _native_symbol("atmosphere_generate_environment")(
        int(width),
        int(height),
        elevation,
        turbidity=kwargs["turbidity"],
        ozone_du=kwargs["ozone_du"],
        mie_g=kwargs["mie_g"],
        ground_albedo=kwargs["ground_albedo"],
        mode=mode,
    )


__all__ = [
    "AtmosphereSettings",
    "SUN_ELEVATION_SWEEP_DEG",
    "bake_luts",
    "generate_environment",
    "spectral_to_linear_rgb",
]
