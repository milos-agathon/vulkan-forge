"""Coordinate system helpers."""

from __future__ import annotations


def validate(lat: float, lon: float) -> bool:
    """Validate latitude/longitude pair."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError("Invalid latitude or longitude")
    return True
