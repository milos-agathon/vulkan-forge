from typing import TypedDict

UtcTuple = tuple[int, int, int, int, int, int | float]

class SolarVector(TypedDict):
    zenith_deg: float
    azimuth_deg: float
    apparent_elevation_deg: float
    true_elevation_deg: float
    distance_au: float
    equation_of_time_min: float

def solar_position(
    utc: UtcTuple,
    lat: float,
    lon: float,
    elev_m: float = ...,
    *,
    tz_offset_hours: float = ...,
    delta_t_seconds: float = ...,
    pressure_mbar: float = ...,
    temperature_c: float = ...,
) -> SolarVector: ...

class SolarTime:
    utc: UtcTuple
    observer_lat: float
    observer_lon: float
    observer_elev_m: float
    tz_offset_hours: float
    delta_t_seconds: float
    pressure_mbar: float
    temperature_c: float
    delta_t: float | None
    def __init__(
        self,
        utc: UtcTuple,
        observer_lat: float,
        observer_lon: float,
        observer_elev_m: float = ...,
        tz_offset_hours: float = ...,
        delta_t_seconds: float = ...,
        pressure_mbar: float = ...,
        temperature_c: float = ...,
        delta_t: float | None = ...,
    ) -> None: ...
    def position(self) -> SolarVector: ...
    def to_native(self) -> dict[str, object]: ...
