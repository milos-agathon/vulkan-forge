"""Coordinate system helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoordinateSystems(Enum):
    """Supported coordinate systems."""

    WGS84 = "EPSG:4326"
    WebMercator = "EPSG:3857"

    @staticmethod
    def validate(lat: float, lon: float) -> bool:
        """Validate latitude and longitude, raising ``ValueError`` if invalid."""
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("Invalid latitude or longitude")
        return True


@dataclass
class GeographicBounds:
    """Simple geographic bounding box."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


def validate(lat: float, lon: float) -> bool:
    """Validate a latitude/longitude pair."""

    CoordinateSystems.validate(lat, lon)
    return True
