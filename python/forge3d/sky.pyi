from datetime import datetime
from typing import Any, Dict

def set_observation(
    datetime_utc: datetime,
    lat: float,
    lon: float,
    *,
    height_m: float = ...,
) -> Dict[str, Any]: ...
