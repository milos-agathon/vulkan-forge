"""Observation state shared by SIDERA's CPU ephemeris and sky renderer."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict
import weakref

from ._native import get_native_module
from .astro import _utc_components

__all__ = ["set_observation"]
_active_viewers = []
_last_observation_command = None


def _get_active_viewer():
    while _active_viewers:
        viewer = _active_viewers[-1]()
        if viewer is not None and viewer.is_running:
            return viewer
        _active_viewers.pop()
    return None


def _set_active_viewer(viewer) -> None:
    if viewer is None:
        _active_viewers.clear()
        return
    _active_viewers[:] = [
        reference
        for reference in _active_viewers
        if reference() is not None and reference() is not viewer
    ]
    _active_viewers.append(weakref.ref(viewer))
    if viewer is not None and viewer.is_running and _last_observation_command is not None:
        viewer.send_ipc(_last_observation_command)


def _remove_active_viewer(viewer) -> None:
    # A viewer may have exited before close() reaches this cleanup hook, so use
    # stack ownership rather than ``is_running`` to decide whether its removal
    # exposes an older viewer.  That older viewer must receive the current
    # process-level observation before it becomes the active target again.
    live = []
    for reference in _active_viewers:
        candidate = reference()
        if candidate is not None:
            live.append((reference, candidate))
    was_top = bool(live and live[-1][1] is viewer)
    _active_viewers[:] = [
        reference for reference, candidate in live if candidate is not viewer
    ]
    if not was_top or _last_observation_command is None:
        return
    promoted = _get_active_viewer()
    if promoted is None:
        return
    try:
        promoted.send_ipc(_last_observation_command)
    except Exception:
        # Closing one viewer must remain best-effort even if the older viewer's
        # IPC channel disappeared concurrently.  The replay stays retained, so
        # re-registering that viewer (or opening another) can retry it.
        pass


def _clear_observation_replay() -> None:
    global _last_observation_command
    _last_observation_command = None


def _has_observation_replay() -> bool:
    return _last_observation_command is not None


def set_observation(
    datetime_utc: datetime,
    lat: float,
    lon: float,
    *,
    height_m: float = 0.0,
) -> Dict[str, Any]:
    """Set the active sky observation and return its body-position summary."""
    components = _utc_components(datetime_utc)
    if not 2000 <= components[0] <= 2050:
        raise ValueError("ephemeris date must be within 2000–2050")
    lat, lon, height_m = float(lat), float(lon), float(height_m)
    if (
        not math.isfinite(lat)
        or not -90.0 <= lat <= 90.0
        or not math.isfinite(lon)
        or not -180.0 <= lon <= 180.0
        or not math.isfinite(height_m)
    ):
        raise ValueError("invalid WGS84 observer coordinates")
    native = get_native_module()
    summary = (
        native.sky_set_observation(*components, lat, lon, height_m)
        if native is not None
        else None
    )
    global _last_observation_command
    _last_observation_command = {
        "cmd": "set_observation",
        "year": components[0],
        "month": components[1],
        "day": components[2],
        "hour": components[3],
        "minute": components[4],
        "second": components[5],
        "latitude_deg": lat,
        "longitude_deg": lon,
        "height_m": height_m,
    }
    viewer = _get_active_viewer()
    forwarded = False
    if viewer is not None and viewer.is_running:
        viewer.send_ipc(_last_observation_command)
        forwarded = True
    if summary is None:
        return {
            "revision": None,
            "status": "forwarded_to_viewer" if forwarded else "pending_viewer",
        }
    return summary
