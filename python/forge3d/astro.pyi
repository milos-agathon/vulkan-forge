from datetime import datetime
from typing import Tuple

def body_position(
    body: str,
    datetime_utc: datetime,
    lat: float,
    lon: float,
    *,
    height_m: float = ...,
    refraction: bool = ...,
) -> Tuple[float, float, float]: ...

def moon_phase(datetime_utc: datetime) -> Tuple[float, float, float]: ...

def delta_t_seconds(datetime_utc: datetime) -> float: ...

def sidereal_time(datetime_utc: datetime) -> Tuple[float, float]: ...

def refraction_arcminutes(true_altitude_deg: float) -> float: ...
