from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from . import AtmosphereLutHandle

SUN_ELEVATION_SWEEP_DEG: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class AtmosphereSettings:
    turbidity: float
    ozone_du: float
    mie_g: float
    ground_albedo: float
    scattering_orders: int
    def __init__(
        self,
        turbidity: float = ...,
        ozone_du: float = ...,
        mie_g: float = ...,
        ground_albedo: float = ...,
        scattering_orders: int = ...,
    ) -> None: ...
    def native_kwargs(self) -> dict[str, float | int]: ...

def bake_luts(settings: AtmosphereSettings | None = ...) -> AtmosphereLutHandle: ...
def spectral_to_linear_rgb(samples: Sequence[float]) -> tuple[float, float, float]: ...
def generate_environment(
    width: int,
    height: int,
    sun_elevation_deg: float,
    *,
    settings: AtmosphereSettings | None = ...,
    mode: str = ...,
) -> dict[str, Any]: ...
