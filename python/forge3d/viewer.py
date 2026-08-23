# python/forge3d/viewer.py
# Viewer control utilities including non-blocking IPC-based viewer workflow.
# Supports Journey 1 (open populated -> interact -> Python updates -> snapshot)
# and Journey 2 (open blank -> build from Python -> snapshot).

from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._viewer_binary import find_viewer_binary as _resolve_viewer_binary
from .diagnostics import (
    Diagnostic,
    experimental_feature_diagnostic,
    missing_glyphs_diagnostic,
    placeholder_fallback_diagnostic,
)
from .viewer_contract import (
    NormalizedExtent,
    VectorOverlayVertex,
    WorldPosition,
    normalized_extent as _normalized_extent,
    vector_overlay_vertices as _vector_overlay_vertices,
    world_position as _world_position,
)

# Import Renderer and MSAA config with fallbacks for standalone testing
try:
    from . import Renderer
    _SUPPORTED_MSAA = {1, 2, 4, 8}  # Standard MSAA sample counts
except ImportError:
    Renderer = None  # type: ignore
    _SUPPORTED_MSAA = {1, 2, 4, 8}


def set_msaa(samples: int) -> int:
    """Set the default MSAA sample count for newly created renderers.

    Sets the class-level default on ``Renderer`` so that future instances
    pick up the requested sample count.  Per-scene MSAA is configured via
    ``Scene.set_msaa_samples`` after construction.
    """
    if samples not in _SUPPORTED_MSAA:
        raise ValueError(f"Unsupported MSAA sample count: {samples} (allowed: {_SUPPORTED_MSAA})")

    if Renderer is not None:
        Renderer._set_default_msaa(samples)

    return samples


# -----------------------------------------------------------------------------
# Non-blocking viewer via IPC (subprocess + TCP + NDJSON)
# -----------------------------------------------------------------------------

_READY_PATTERN = re.compile(r"FORGE3D_VIEWER_READY port=(\d+)")
_DEFAULT_TIMEOUT = 30.0


def _write_temp_tiff_from_array(heightmap: np.ndarray) -> Path:
    """Write a float32 heightmap array to a temporary TIFF file."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Opening .npy terrain files requires Pillow. Install with: pip install pillow"
        ) from exc

    array = np.asarray(heightmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("terrain_path .npy files must contain a 2D array")
    if array.size == 0:
        raise ValueError("terrain_path .npy files must not be empty")

    if not np.isfinite(array).all():
        finite = array[np.isfinite(array)]
        fill_value = float(finite.min()) if finite.size else 0.0
        array = np.where(np.isfinite(array), array, fill_value).astype(np.float32)

    handle, temp_path = tempfile.mkstemp(prefix="forge3d_dem_", suffix=".tif")
    os.close(handle)
    Image.fromarray(np.ascontiguousarray(array), mode="F").save(temp_path, format="TIFF")
    return Path(temp_path)


def _prepare_terrain_path(
    terrain_path: Optional[Union[str, Path]],
) -> Tuple[Optional[str], List[Path]]:
    """Prepare a terrain path for the viewer, converting .npy arrays when needed."""
    if terrain_path is None:
        return None, []

    path = Path(terrain_path)
    if path.suffix.lower() != ".npy":
        return str(path), []

    array = np.load(path)
    temp_tiff = _write_temp_tiff_from_array(array)
    return str(temp_tiff), [temp_tiff]


def _cleanup_paths(paths: List[Path]) -> None:
    """Best-effort cleanup for temporary files created by the viewer wrapper."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _wait_for_snapshot(
    path: Path,
    *,
    timeout: float,
    previous_mtime_ns: Optional[int],
    poll_interval: float = 0.05,
) -> None:
    """Wait for a viewer snapshot file to appear or be updated."""
    deadline = time.time() + max(timeout, poll_interval)
    stable_signature: Optional[Tuple[int, int]] = None
    stable_polls = 0
    while time.time() < deadline:
        try:
            stat = path.stat()
        except FileNotFoundError:
            time.sleep(poll_interval)
            continue

        is_fresh = previous_mtime_ns is None or stat.st_mtime_ns != previous_mtime_ns
        if is_fresh and stat.st_size > 0:
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
            if signature == stable_signature:
                stable_polls += 1
            else:
                stable_signature = signature
                stable_polls = 1
            if stable_polls >= 2:
                return
        else:
            stable_signature = None
            stable_polls = 0

        time.sleep(poll_interval)

    raise ViewerError(f"Timed out waiting for snapshot output: {path}")


class ViewerError(Exception):
    """Error from viewer IPC communication."""


@dataclass(frozen=True)
class LabelBatchResult:
    """Ordered result for a high-level batch label create request."""

    ids: List[Optional[int]]
    diagnostics: List[Diagnostic] = field(default_factory=list)


@dataclass(frozen=True)
class LabelOperationResult:
    """Truthful result for a high-level label state/configuration operation."""

    ok: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)


class ViewerHandle:
    """Handle to a non-blocking interactive viewer subprocess.
    
    The viewer runs as a separate process and communicates via TCP + NDJSON.
    Use `open_viewer_async()` to create a ViewerHandle.
    
    Example:
        >>> from forge3d.viewer import open_viewer_async
        >>> with open_viewer_async(obj_path="model.obj") as v:
        ...     v.set_camera_lookat(eye=[0, 5, 10], target=[0, 0, 0])
        ...     v.snapshot("output.png", width=1920, height=1080)
    """
    
    def __init__(
        self,
        process: subprocess.Popen,
        host: str,
        port: int,
        timeout: float = 10.0,
        cleanup_paths: Optional[List[Path]] = None,
    ):
        self._process = process
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._cleanup_paths = list(cleanup_paths or [])
        self._stdout_drain_thread: Optional[threading.Thread] = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to the viewer's IPC server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._host, self._port))
    
    def _send_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command and wait for response."""
        if self._socket is None:
            raise ViewerError("Not connected to viewer")
        
        # Send NDJSON request
        request = json.dumps(cmd) + "\n"
        self._socket.sendall(request.encode("utf-8"))
        
        # Read response line
        response_data = b""
        while b"\n" not in response_data:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise ViewerError("Connection closed by viewer")
            response_data += chunk
        
        line = response_data.split(b"\n")[0].decode("utf-8")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as e:
            raise ViewerError(f"Invalid JSON response: {e}")
        
        if not response.get("ok", False):
            error_msg = response.get("error", "Unknown error")
            raise ViewerError(f"Viewer command failed: {error_msg}")
        command_name = cmd.get("cmd")
        overrides_observation_sun = command_name in {"lit_sun", "set_terrain_sun"}
        if command_name == "set_terrain":
            overrides_observation_sun = any(
                cmd.get(key) is not None
                for key in ("sun_azimuth", "sun_elevation", "sun_intensity")
            )
        if overrides_observation_sun:
            from . import sky

            active_viewer = sky._get_active_viewer()
            if active_viewer is None or active_viewer is self:
                sky._clear_observation_replay()
        return response

    def send_ipc(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Send a raw IPC command to the viewer and return the decoded response."""
        return self._send_command(cmd)

    def _allocate_label_id(self) -> int:
        return int(getattr(self, "_next_public_label_id", 1))

    def _commit_label_id(self, label_id: int) -> None:
        self._next_public_label_id = max(
            int(getattr(self, "_next_public_label_id", 1)), int(label_id) + 1
        )

    def _allocate_vector_overlay_id(self) -> int:
        return int(getattr(self, "_next_public_vector_overlay_id", 1))

    def _commit_vector_overlay_id(self, overlay_id: int) -> None:
        self._next_public_vector_overlay_id = max(
            int(getattr(self, "_next_public_vector_overlay_id", 1)), int(overlay_id) + 1
        )

    def _ensure_label_api_state(self) -> Dict[str, Any]:
        state = getattr(self, "_label_api_state", None)
        if state is None:
            state = {
                "enabled": True,
                "active_atlas": None,
                "typography": None,
                "declutter_algorithm": None,
                "layout_metrics": self._typography_layout_metrics(None),
                "label_ids": set(),
                "line_label_glyph_instances": {},
            }
            self._label_api_state = state
        return state

    @staticmethod
    def _typography_width(text: str, settings: Optional[Mapping[str, Any]]) -> float:
        font_size = 16.0
        tracking = float((settings or {}).get("tracking", 0.0))
        kerning = bool((settings or {}).get("kerning", True))
        word_spacing = float((settings or {}).get("word_spacing", 1.0))
        width = 0.0
        chars = list(text)
        for index, char in enumerate(chars):
            base = 0.3 * font_size if char == " " else 0.5 * font_size
            if char == " ":
                base *= word_spacing
            width += base + tracking * font_size
            if kerning and index + 1 < len(chars) and (char, chars[index + 1]) == ("A", "V"):
                width -= 0.08 * font_size
        return round(width, 4)

    def _typography_layout_metrics(self, settings: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        sample_text = "AV label"
        line_height = float((settings or {}).get("line_height", 1.2))
        return {
            "sample_text": sample_text,
            "default_width": self._typography_width(sample_text, None),
            "typography_width": self._typography_width(sample_text, settings),
            "line_height_px": round(16.0 * line_height, 4),
        }

    def _label_state_snapshot(self) -> Dict[str, Any]:
        state = self._ensure_label_api_state()
        label_ids = sorted(int(label_id) for label_id in state["label_ids"])
        active_atlas = state["active_atlas"]
        typography = state["typography"]
        line_instances = {
            str(int(label_id)): [dict(glyph) for glyph in glyphs]
            for label_id, glyphs in sorted(
                state["line_label_glyph_instances"].items(),
                key=lambda item: int(item[0]),
            )
        }
        return {
            "enabled": bool(state["enabled"]),
            "active_atlas": None if active_atlas is None else dict(active_atlas),
            "typography": None if typography is None else dict(typography),
            "declutter_algorithm": state["declutter_algorithm"],
            "layout_metrics": dict(state["layout_metrics"]),
            "label_ids": label_ids,
            "label_count": len(label_ids),
            "line_label_glyph_instances": line_instances,
        }

    def label_configuration_state(self) -> Dict[str, Any]:
        """Return the public serializable label configuration/state snapshot."""
        return self._label_state_snapshot()

    def _label_operation_result(
        self,
        ok: bool,
        diagnostics: Optional[Sequence[Diagnostic]] = None,
    ) -> LabelOperationResult:
        return LabelOperationResult(
            ok=bool(ok),
            diagnostics=list(diagnostics or []),
            state=self._label_state_snapshot(),
        )

    def _send_stable_create(self, cmd: Dict[str, Any], object_name: str) -> int:
        response = self._send_command(cmd)
        if "id" not in response:
            raise ViewerError(
                f"{cmd['cmd']} reported success without a stable {object_name} id"
            )
        created_id = int(response["id"])
        if created_id != int(cmd["id"]):
            raise ViewerError(
                f"{cmd['cmd']} returned {object_name} id {created_id}, expected {cmd['id']}"
            )
        return created_id

    def _send_label_create(self, cmd: Dict[str, Any]) -> int:
        return self._send_stable_create(cmd, "label")

    def _text_diagnostic(
        self,
        text: str,
        *,
        feature: str,
        object_id: str = "pending",
    ) -> Optional[Diagnostic]:
        if not str(text):
            return placeholder_fallback_diagnostic(
                feature,
                layer_id="labels",
                object_id=object_id,
            )
        missing = sorted({char for char in str(text) if ord(char) > 0x7F})
        if missing:
            return missing_glyphs_diagnostic(
                missing,
                layer_id="labels",
                object_id=object_id,
            )
        return None

    @staticmethod
    def _normalize_line_path(
        polyline: Sequence[WorldPosition]
    ) -> List[Tuple[float, float, float]]:
        return [tuple(_world_position(point, name="polyline point")) for point in polyline]

    @staticmethod
    def _line_path_length(polyline: Sequence[Tuple[float, float, float]]) -> float:
        length = 0.0
        for start, end in zip(polyline, polyline[1:]):
            dx = end[0] - start[0]
            dz = end[2] - start[2]
            length += math.hypot(dx, dz)
        return length

    @staticmethod
    def _normalize_label_rotation(rotation: float) -> Tuple[float, bool]:
        while rotation <= -math.pi:
            rotation += math.tau
        while rotation > math.pi:
            rotation -= math.tau
        if rotation > math.pi / 2.0:
            return rotation - math.pi, True
        if rotation < -math.pi / 2.0:
            return rotation + math.pi, True
        return rotation, False

    def _line_path_diagnostic(
        self,
        polyline: Sequence[Tuple[float, float, float]],
        *,
        object_id: str = "pending",
    ) -> Optional[Diagnostic]:
        if len(polyline) < 2 or self._line_path_length(polyline) <= 0.0:
            return placeholder_fallback_diagnostic(
                "invalid line label path",
                layer_id="labels",
                object_id=object_id,
            )
        return None

    def _build_line_glyph_instances(
        self,
        label_id: int,
        text: str,
        polyline: Sequence[Tuple[float, float, float]],
    ) -> List[Dict[str, Any]]:
        glyphs = [char for char in str(text) if char != " "]
        if not glyphs:
            return []

        total_length = self._line_path_length(polyline)
        offsets = [
            total_length * ((index + 0.5) / len(glyphs))
            for index in range(len(glyphs))
        ]
        instances: List[Dict[str, Any]] = []
        offset_index = 0
        accumulated = 0.0

        for start, end in zip(polyline, polyline[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dz = end[2] - start[2]
            segment_length = math.hypot(dx, dz)
            if segment_length <= 0.0:
                continue
            while (
                offset_index < len(offsets)
                and offsets[offset_index] <= accumulated + segment_length
            ):
                local_t = (offsets[offset_index] - accumulated) / segment_length
                position = [
                    start[0] + dx * local_t,
                    start[1] + dy * local_t,
                    start[2] + dz * local_t,
                ]
                rotation, adjusted = self._normalize_label_rotation(math.atan2(dz, dx))
                instances.append(
                    {
                        "label_id": int(label_id),
                        "glyph": glyphs[offset_index],
                        "position": [float(value) for value in position],
                        "rotation": float(rotation),
                        "ordering_key": f"{int(label_id)}:{offset_index:04d}",
                        "upside_down_adjusted": bool(adjusted),
                    }
                )
                offset_index += 1
            accumulated += segment_length

        return instances

    def load_label_atlas(
        self,
        atlas_png_path: Union[str, Path],
        metrics_json_path: Union[str, Path],
    ) -> LabelOperationResult:
        """Load a label atlas through the public high-level viewer API."""
        self._send_command(
            {
                "cmd": "load_label_atlas",
                "atlas_png_path": str(atlas_png_path),
                "metrics_json_path": str(metrics_json_path),
            }
        )
        self._ensure_label_api_state()["active_atlas"] = {
            "atlas_png_path": str(atlas_png_path),
            "metrics_json_path": str(metrics_json_path),
        }
        return self._label_operation_result(True)

    def add_label(
        self,
        text: str,
        world_pos: WorldPosition,
        size: Optional[float] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
        halo_color: Optional[Tuple[float, float, float, float]] = None,
        halo_width: Optional[float] = None,
        priority: Optional[int] = None,
        min_zoom: Optional[float] = None,
        max_zoom: Optional[float] = None,
        offset: Optional[Tuple[float, float]] = None,
        rotation: Optional[float] = None,
        underline: Optional[bool] = None,
        small_caps: Optional[bool] = None,
        leader: Optional[bool] = None,
        horizon_fade_angle: Optional[float] = None,
    ) -> Union[int, LabelOperationResult]:
        """Create a label at f64 viewer-world ``(X, display Y, Z)``."""
        diagnostic = self._text_diagnostic(str(text), feature="empty label text")
        if diagnostic is not None:
            return self._label_operation_result(False, [diagnostic])
        label_id = self._allocate_label_id()
        cmd: Dict[str, Any] = {
            "cmd": "add_label",
            "id": label_id,
            "text": str(text),
            "world_pos": _world_position(world_pos, name="world_pos"),
        }
        if size is not None:
            cmd["size"] = float(size)
        if color is not None:
            cmd["color"] = [float(v) for v in color]
        if halo_color is not None:
            cmd["halo_color"] = [float(v) for v in halo_color]
        if halo_width is not None:
            cmd["halo_width"] = float(halo_width)
        if priority is not None:
            cmd["priority"] = int(priority)
        if min_zoom is not None:
            cmd["min_zoom"] = float(min_zoom)
        if max_zoom is not None:
            cmd["max_zoom"] = float(max_zoom)
        if offset is not None:
            cmd["offset"] = [float(offset[0]), float(offset[1])]
        if rotation is not None:
            cmd["rotation"] = float(rotation)
        if underline is not None:
            cmd["underline"] = bool(underline)
        if small_caps is not None:
            cmd["small_caps"] = bool(small_caps)
        if leader is not None:
            cmd["leader"] = bool(leader)
        if horizon_fade_angle is not None:
            cmd["horizon_fade_angle"] = float(horizon_fade_angle)
        created_id = self._send_label_create(cmd)
        self._commit_label_id(created_id)
        self._ensure_label_api_state()["label_ids"].add(created_id)
        return created_id

    def add_labels(self, labels: Sequence[Mapping[str, Any]]) -> LabelBatchResult:
        """Create labels in input order, preserving per-input diagnostics."""
        ids: List[Optional[int]] = []
        diagnostics: List[Diagnostic] = []
        for index, label in enumerate(labels):
            text = str(label.get("text", ""))
            if not text:
                ids.append(None)
                diagnostics.append(
                    placeholder_fallback_diagnostic(
                        "empty label text",
                        layer_id=str(label.get("layer_id", "labels")),
                        object_id=str(label.get("id", index)),
                    )
                )
                continue
            try:
                result = self.add_label(
                    text,
                    tuple(label.get("world_pos", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                    size=label.get("size"),
                    color=label.get("color"),
                    halo_color=label.get("halo_color"),
                    halo_width=label.get("halo_width"),
                    priority=label.get("priority"),
                    min_zoom=label.get("min_zoom"),
                    max_zoom=label.get("max_zoom"),
                    offset=label.get("offset"),
                    rotation=label.get("rotation"),
                    underline=label.get("underline"),
                    small_caps=label.get("small_caps"),
                    leader=label.get("leader"),
                    horizon_fade_angle=label.get("horizon_fade_angle"),
                )
                if isinstance(result, LabelOperationResult):
                    ids.append(None)
                    diagnostics.extend(result.diagnostics)
                else:
                    ids.append(result)
            except ViewerError as exc:
                ids.append(None)
                diagnostics.append(
                    placeholder_fallback_diagnostic(
                        str(exc),
                        layer_id=str(label.get("layer_id", "labels")),
                        object_id=str(label.get("id", index)),
                    )
                )
        return LabelBatchResult(ids=ids, diagnostics=diagnostics)

    def add_line_label(
        self,
        text: str,
        polyline: Sequence[WorldPosition],
        size: Optional[float] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
        halo_color: Optional[Tuple[float, float, float, float]] = None,
        halo_width: Optional[float] = None,
        priority: Optional[int] = None,
        placement: str = "center",
        repeat_distance: Optional[float] = None,
        min_zoom: Optional[float] = None,
        max_zoom: Optional[float] = None,
        terrain_mode: Optional[str] = None,
    ) -> Union[int, LabelOperationResult]:
        """Create a line label along f64 viewer-world points."""
        normalized_polyline = self._normalize_line_path(polyline)
        if terrain_mode not in (None, "none", "flat"):
            diagnostic = experimental_feature_diagnostic(
                "terrain-elevated line labels",
                layer_id="labels",
            )
            return self._label_operation_result(False, [diagnostic])
        diagnostic = self._text_diagnostic(str(text), feature="empty line label text")
        if diagnostic is not None:
            return self._label_operation_result(False, [diagnostic])
        diagnostic = self._line_path_diagnostic(normalized_polyline)
        if diagnostic is not None:
            return self._label_operation_result(False, [diagnostic])

        label_id = self._allocate_label_id()
        cmd: Dict[str, Any] = {
            "cmd": "add_line_label",
            "id": label_id,
            "text": str(text),
            "polyline": [_world_position(p, name="polyline point") for p in normalized_polyline],
            "placement": str(placement),
        }
        if size is not None:
            cmd["size"] = float(size)
        if color is not None:
            cmd["color"] = [float(v) for v in color]
        if halo_color is not None:
            cmd["halo_color"] = [float(v) for v in halo_color]
        if halo_width is not None:
            cmd["halo_width"] = float(halo_width)
        if priority is not None:
            cmd["priority"] = int(priority)
        if repeat_distance is not None:
            cmd["repeat_distance"] = float(repeat_distance)
        if min_zoom is not None:
            cmd["min_zoom"] = float(min_zoom)
        if max_zoom is not None:
            cmd["max_zoom"] = float(max_zoom)
        created_id = self._send_label_create(cmd)
        self._commit_label_id(created_id)
        state = self._ensure_label_api_state()
        state["label_ids"].add(created_id)
        state["line_label_glyph_instances"][created_id] = self._build_line_glyph_instances(
            created_id,
            str(text),
            normalized_polyline,
        )
        return created_id

    def add_curved_label(
        self,
        text: str,
        path: Sequence[WorldPosition],
        *,
        size: Optional[float] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
        halo_color: Optional[Tuple[float, float, float, float]] = None,
        halo_width: Optional[float] = None,
        priority: Optional[int] = None,
        tracking: Optional[float] = None,
        center_on_path: Optional[bool] = None,
    ) -> LabelOperationResult:
        """Return an experimental diagnostic until curved glyph rendering is production-ready."""
        diagnostic = experimental_feature_diagnostic(
            "curved labels",
            layer_id="labels",
        )
        return self._label_operation_result(False, [diagnostic])

    def add_callout(
        self,
        text: str,
        anchor: WorldPosition,
        offset: Tuple[float, float] = (0.0, -50.0),
        background_color: Optional[Tuple[float, float, float, float]] = None,
        border_color: Optional[Tuple[float, float, float, float]] = None,
        border_width: Optional[float] = None,
        corner_radius: Optional[float] = None,
        padding: Optional[float] = None,
        text_size: Optional[float] = None,
        text_color: Optional[Tuple[float, float, float, float]] = None,
    ) -> Union[int, LabelOperationResult]:
        """Create a callout label and return its stable public label ID."""
        diagnostic = self._text_diagnostic(str(text), feature="empty callout text")
        if diagnostic is not None:
            return self._label_operation_result(False, [diagnostic])
        label_id = self._allocate_label_id()
        cmd: Dict[str, Any] = {
            "cmd": "add_callout",
            "id": label_id,
            "text": str(text),
            "anchor": _world_position(anchor, name="anchor"),
            "offset": [float(offset[0]), float(offset[1])],
        }
        if background_color is not None:
            cmd["background_color"] = [float(v) for v in background_color]
        if border_color is not None:
            cmd["border_color"] = [float(v) for v in border_color]
        if border_width is not None:
            cmd["border_width"] = float(border_width)
        if corner_radius is not None:
            cmd["corner_radius"] = float(corner_radius)
        if padding is not None:
            cmd["padding"] = float(padding)
        if text_size is not None:
            cmd["text_size"] = float(text_size)
        if text_color is not None:
            cmd["text_color"] = [float(v) for v in text_color]
        created_id = self._send_label_create(cmd)
        self._commit_label_id(created_id)
        self._ensure_label_api_state()["label_ids"].add(created_id)
        return created_id

    def add_vector_overlay(
        self,
        name: str,
        vertices: Sequence[VectorOverlayVertex],
        indices: Sequence[int],
        primitive: str = "triangles",
        drape: bool = True,
        drape_offset: float = 0.5,
        opacity: float = 1.0,
        depth_bias: float = 0.001,
        line_width: float = 2.0,
        point_size: float = 5.0,
        z_order: int = 0,
    ) -> int:
        """Create an overlay from eight-lane XYZ/RGBA/feature-ID vertices.

        XYZ is absolute viewer world ``(X, display Y, Z)`` in f64; projected
        terrain uses ``Z=-map Y``. RGBA is normalized ``[0, 1]`` and the final
        lane is an exact unsigned 32-bit feature ID.
        """
        validated_vertices = _vector_overlay_vertices(vertices)
        overlay_id = self._allocate_vector_overlay_id()
        cmd: Dict[str, Any] = {
            "cmd": "add_vector_overlay",
            "id": overlay_id,
            "name": str(name),
            "vertices": validated_vertices,
            "indices": [int(index) for index in indices],
            "primitive": str(primitive),
            "drape": bool(drape),
            "drape_offset": float(drape_offset),
            "opacity": float(opacity),
            "depth_bias": float(depth_bias),
            "line_width": float(line_width),
            "point_size": float(point_size),
            "z_order": int(z_order),
        }
        created_id = self._send_stable_create(cmd, "vector overlay")
        self._commit_vector_overlay_id(created_id)
        return created_id

    def set_labels_enabled(self, enabled: bool) -> LabelOperationResult:
        """Set label visibility through the public high-level viewer API."""
        self._send_command({"cmd": "set_labels_enabled", "enabled": bool(enabled)})
        self._ensure_label_api_state()["enabled"] = bool(enabled)
        return self._label_operation_result(True)

    def clear_labels(self) -> LabelOperationResult:
        """Clear labels through the public high-level viewer API."""
        self._send_command({"cmd": "clear_labels"})
        state = self._ensure_label_api_state()
        state["label_ids"].clear()
        state["line_label_glyph_instances"].clear()
        return self._label_operation_result(True)

    def remove_label(self, label_id: int) -> LabelOperationResult:
        """Remove a known public label ID or return a diagnostic for no-op removal."""
        label_id = int(label_id)
        state = self._ensure_label_api_state()
        if label_id not in state["label_ids"]:
            diagnostic = placeholder_fallback_diagnostic(
                "remove unknown label",
                layer_id="labels",
                object_id=str(label_id),
            )
            return self._label_operation_result(False, [diagnostic])
        self._send_command({"cmd": "remove_label", "id": label_id})
        state["label_ids"].remove(label_id)
        state["line_label_glyph_instances"].pop(label_id, None)
        return self._label_operation_result(True)

    def set_label_typography(
        self,
        *,
        tracking: Optional[float] = None,
        kerning: Optional[bool] = None,
        line_height: Optional[float] = None,
        word_spacing: Optional[float] = None,
    ) -> LabelOperationResult:
        """Set label typography and expose deterministic layout metrics."""
        cmd: Dict[str, Any] = {"cmd": "set_label_typography"}
        state = self._ensure_label_api_state()
        previous = dict(state["typography"] or {})
        typography = {
            "tracking": float(previous.get("tracking", 0.0) if tracking is None else tracking),
            "kerning": bool(previous.get("kerning", True) if kerning is None else kerning),
            "line_height": float(previous.get("line_height", 1.2) if line_height is None else line_height),
            "word_spacing": float(previous.get("word_spacing", 1.0) if word_spacing is None else word_spacing),
        }
        if tracking is not None:
            cmd["tracking"] = typography["tracking"]
        if kerning is not None:
            cmd["kerning"] = typography["kerning"]
        if line_height is not None:
            cmd["line_height"] = typography["line_height"]
        if word_spacing is not None:
            cmd["word_spacing"] = typography["word_spacing"]
        self._send_command(cmd)
        state["typography"] = typography
        state["layout_metrics"] = self._typography_layout_metrics(typography)
        return self._label_operation_result(True)

    def set_declutter_algorithm(
        self,
        algorithm: str,
        *,
        seed: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> LabelOperationResult:
        """Set deterministic label declutter policy state."""
        normalized_algorithm = str(algorithm).lower()
        if normalized_algorithm not in {"greedy", "annealing"}:
            diagnostic = placeholder_fallback_diagnostic(
                "unsupported label declutter algorithm",
                layer_id="labels",
                object_id=normalized_algorithm,
            )
            return self._label_operation_result(False, [diagnostic])
        cmd: Dict[str, Any] = {
            "cmd": "set_declutter_algorithm",
            "algorithm": normalized_algorithm,
        }
        if seed is not None:
            cmd["seed"] = int(seed)
        if max_iterations is not None:
            cmd["max_iterations"] = int(max_iterations)
        self._send_command(cmd)
        self._ensure_label_api_state()["declutter_algorithm"] = {
            "algorithm": normalized_algorithm,
            "seed": None if seed is None else int(seed),
            "max_iterations": None if max_iterations is None else int(max_iterations),
            "placement_order": (
                "priority_then_energy"
                if normalized_algorithm == "annealing"
                else "priority_then_collision"
            ),
        }
        return self._label_operation_result(True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get viewer stats (geometry readiness, vertex/index counts).

        Returns a dictionary containing ``vb_ready``, ``vertex_count``,
        ``index_count``, and ``scene_has_mesh``.

        Raises:
            ViewerError: if stats unavailable or command fails
        """
        response = self._send_command({"cmd": "get_stats"})
        stats = response.get("stats")
        if stats is None:
            raise ViewerError("get_stats returned no stats data")
        return stats

    def poll_pick_events(self) -> List[Dict[str, Any]]:
        """Drain and return completed screen-pick events.

        Every result ``world_pos`` is an absolute f64 viewer-world position in
        ``(X, display Y, Z)`` coordinates; it is never anchor/render relative.
        """
        response = self._send_command({"cmd": "poll_pick_events"})
        return list(response.get("pick_events") or [])

    def pick_at(
        self,
        x: int,
        y: int,
        *,
        shift: bool = False,
        ctrl: bool = False,
    ) -> List[Dict[str, Any]]:
        """Pick the frozen rendered frame at a screen pixel.

        The method drains older events, waits for execution of this exact
        command, and returns its pick results.  Result positions are absolute
        f64 viewer-world ``(X, display Y, Z)`` values.
        """
        self.poll_pick_events()
        self._send_command(
            {
                "cmd": "pick_at",
                "x": int(x),
                "y": int(y),
                "shift": bool(shift),
                "ctrl": bool(ctrl),
            }
        )
        events = self.poll_pick_events()
        correlated = [
            event
            for event in events
            if event.get("screen_pos") == [int(x), int(y)]
        ]
        return [
            dict(result)
            for event in correlated
            for result in (event.get("results") or [])
        ]

    def update_labels(self) -> None:
        """Update labels against the same frozen frame used by rendering."""
        self._send_command({"cmd": "update_labels"})

    def _wait_for_rendered_revision(self, revision: int, *, timeout: float) -> None:
        """Fence evidence capture on both command publication and frame submit."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            stats = self.get_stats()
            if int(stats.get("rendered_frame_revision", 0)) >= revision:
                return
            time.sleep(0.01)
        raise ViewerError(
            f"Timeout waiting for rendered frame revision {revision}"
        )

    def get_terrain_volumetrics_report(self) -> Dict[str, Any]:
        """Get the latest terrain heterogeneous-volumetrics memory report."""
        response = self._send_command({"cmd": "get_terrain_volumetrics_report"})
        report = response.get("terrain_volumetrics_report")
        if report is None:
            raise ViewerError("get_terrain_volumetrics_report returned no report data")
        return report
    
    def load_obj(self, path: Union[str, Path]) -> None:
        """Load an OBJ file into the viewer."""
        self._send_command({"cmd": "load_obj", "path": str(path)})
    
    def load_gltf(self, path: Union[str, Path]) -> None:
        """Load a glTF/GLB file into the viewer."""
        self._send_command({"cmd": "load_gltf", "path": str(path)})

    def load_terrain(self, path: Union[str, Path]) -> None:
        """Load a terrain heightmap file into the viewer.

        ``.npy`` heightmaps are converted to a temporary TIFF automatically so they
        can be consumed by the current viewer binary.
        """
        actual_path, cleanup_paths = _prepare_terrain_path(path)
        self._cleanup_paths.extend(cleanup_paths)
        self._send_command({"cmd": "load_terrain", "path": actual_path})

    def load_bundle(
        self,
        path_or_bundle: Union[str, Path, "LoadedBundle"],
        variant_id: Optional[str] = None,
    ) -> "LoadedBundle":
        """Load a bundle's terrain and TV16 review state into the viewer."""
        from .bundle import LoadedBundle, load_bundle as load_scene_bundle

        bundle = (
            path_or_bundle
            if isinstance(path_or_bundle, LoadedBundle)
            else load_scene_bundle(path_or_bundle)
        )
        if variant_id is not None:
            bundle.apply_variant(variant_id)

        if bundle.dem_path is not None:
            self.load_terrain(bundle.dem_path)
        self._send_command({"cmd": "set_scene_review_state", "state": bundle.scene_state.to_dict()})
        return bundle

    def load_overlay(
        self,
        name: str,
        path: Union[str, Path],
        extent: Optional[NormalizedExtent] = None,
        opacity: Optional[float] = None,
        z_order: Optional[int] = None,
        preserve_colors: Optional[bool] = None,
    ) -> None:
        """Load an image overlay using normalized UV ``(u0, v0, u1, v1)``.

        ``preserve_colors`` switches raster overlays into a viewer mode that
        composites their source colors after terrain lighting so categorical
        palettes survive unchanged.
        """
        cmd: Dict[str, Any] = {
            "cmd": "load_overlay",
            "name": str(name),
            "path": str(path),
        }
        if extent is not None:
            cmd["extent"] = _normalized_extent(extent)
        if opacity is not None:
            cmd["opacity"] = float(opacity)
        if z_order is not None:
            cmd["z_order"] = int(z_order)
        self._send_command(cmd)
        if preserve_colors is not None:
            self._send_command(
                {
                    "cmd": "set_overlay_preserve_colors",
                    "preserve_colors": bool(preserve_colors),
                }
            )

    def load_point_cloud(
        self,
        path: Union[str, Path],
        point_size: float = 2.0,
        max_points: int = 500_000,
        color_mode: Optional[str] = None,
    ) -> None:
        """Load a LAZ/LAS point cloud into the viewer."""
        cmd: Dict[str, Any] = {
            "cmd": "load_point_cloud",
            "path": str(path),
            "point_size": float(point_size),
            "max_points": int(max_points),
        }
        if color_mode is not None:
            cmd["color_mode"] = str(color_mode)
        self._send_command(cmd)

    def set_point_cloud_params(
        self,
        point_size: Optional[float] = None,
        visible: Optional[bool] = None,
        color_mode: Optional[str] = None,
    ) -> None:
        """Update point cloud rendering parameters."""
        cmd: Dict[str, Any] = {"cmd": "set_point_cloud_params"}
        if point_size is not None:
            cmd["point_size"] = float(point_size)
        if visible is not None:
            cmd["visible"] = bool(visible)
        if color_mode is not None:
            cmd["color_mode"] = str(color_mode)
        self._send_command(cmd)
    
    def set_transform(
        self,
        translation: Optional[Tuple[float, float, float]] = None,
        rotation_quat: Optional[Tuple[float, float, float, float]] = None,
        scale: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Set object transform (translation, rotation quaternion, scale)."""
        cmd: Dict[str, Any] = {"cmd": "set_transform"}
        if translation is not None:
            cmd["translation"] = _world_position(translation, name="translation")
        if rotation_quat is not None:
            cmd["rotation_quat"] = list(rotation_quat)
        if scale is not None:
            cmd["scale"] = list(scale)
        self._send_command(cmd)
    
    def set_camera_lookat(
        self,
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        """Set f64 viewer-world ``(X, display Y, Z)`` look-at coordinates."""
        self._send_command({
            "cmd": "cam_lookat",
            "eye": _world_position(eye, name="eye"),
            "target": _world_position(target, name="target"),
            "up": _world_position(up, name="up"),
        })
    
    def set_fov(self, deg: float) -> None:
        """Set camera field of view in degrees."""
        self._send_command({"cmd": "set_fov", "deg": float(deg)})
    
    def set_sun(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Set sun direction (azimuth and elevation in degrees)."""
        command = {
            "cmd": "lit_sun",
            "azimuth_deg": float(azimuth_deg),
            "elevation_deg": float(elevation_deg),
        }
        self._send_command(command)
        self._send_command({
            "cmd": "set_terrain_sun",
            "azimuth_deg": command["azimuth_deg"],
            "elevation_deg": command["elevation_deg"],
            "intensity": 1.0,
            "source": "manual_angles",
        })

    def set_sun_time(self, solar_time: "Any", intensity: float = 1.0) -> None:
        """Set the terrain sun from a :class:`forge3d.geo.SolarTime`."""
        from .geo import _coerce_solar_time

        solar_time = _coerce_solar_time(solar_time)
        position = solar_time.position()
        self._send_command({
            "cmd": "lit_sun",
            "azimuth_deg": float(position["azimuth_deg"]),
            "elevation_deg": float(position["apparent_elevation_deg"]),
        })
        self._send_command({
            "cmd": "set_terrain_sun",
            "azimuth_deg": float(position["azimuth_deg"]),
            "elevation_deg": float(position["apparent_elevation_deg"]),
            "intensity": float(intensity),
            "source": "solar_time",
        })
    
    def set_ibl(self, path: Union[str, Path], intensity: float = 1.0) -> None:
        """Set IBL (environment map) from HDR file."""
        self._send_command({
            "cmd": "lit_ibl",
            "path": str(path),
            "intensity": float(intensity),
        })
    
    def set_z_scale(self, value: float) -> None:
        """Set terrain z-scale (height exaggeration). Only applies to terrain scenes."""
        self._send_command({"cmd": "set_terrain", "zscale": float(value)})

    def set_terrain_scatter(self, batches: List[Dict[str, Any]]) -> None:
        """Upload terrain scatter batches through the viewer IPC surface."""
        self._send_command({"cmd": "set_terrain_scatter", "batches": batches})

    def clear_terrain_scatter(self) -> None:
        """Clear all terrain scatter batches from the active terrain scene."""
        self._send_command({"cmd": "clear_terrain_scatter"})

    def list_scene_variants(self) -> List[Dict[str, Any]]:
        """Return structured summaries for installed scene variants."""
        response = self._send_command({"cmd": "list_scene_variants"})
        variants = response.get("scene_variants")
        if variants is None:
            raise ViewerError("list_scene_variants returned no variant data")
        return variants

    def list_review_layers(self) -> List[Dict[str, Any]]:
        """Return structured summaries for installed review layers."""
        response = self._send_command({"cmd": "list_review_layers"})
        layers = response.get("review_layers")
        if layers is None:
            raise ViewerError("list_review_layers returned no layer data")
        return layers

    def get_active_scene_variant(self) -> Optional[str]:
        """Return the active scene variant ID, if any."""
        response = self._send_command({"cmd": "get_active_scene_variant"})
        if "active_scene_variant" not in response:
            raise ViewerError("get_active_scene_variant returned no variant data")
        return response.get("active_scene_variant")

    def apply_scene_variant(self, variant_id: str) -> None:
        """Apply a named scene variant and clear manual layer overrides."""
        self._send_command({"cmd": "apply_scene_variant", "variant_id": str(variant_id)})

    def set_review_layer_visible(self, layer_id: str, visible: bool) -> None:
        """Apply a manual review-layer visibility override."""
        self._send_command(
            {
                "cmd": "set_review_layer_visible",
                "layer_id": str(layer_id),
                "visible": bool(visible),
            }
        )
    
    def set_orbit_camera(
        self,
        phi_deg: float,
        theta_deg: float,
        radius: float,
        fov_deg: Optional[float] = None,
        target: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Set orbit camera parameters (phi/theta/radius).
        
        Args:
            phi_deg: Azimuth angle in degrees (horizontal rotation)
            theta_deg: Polar angle in degrees measured down from the vertical axis.
                `0` looks straight down, `90` is level with the horizon.
            radius: Distance from target/center. For terrain scenes this is expressed in the
                viewer's terrain-width world units, not DEM resolution meters.
            fov_deg: Optional field of view in degrees
            target: Optional explicit orbit target in terrain-viewer world coordinates
        """
        cmd: Dict[str, Any] = {
            "cmd": "set_terrain_camera",
            "phi_deg": float(phi_deg),
            "theta_deg": float(theta_deg),
            "radius": float(radius),
        }
        if fov_deg is not None:
            cmd["fov_deg"] = float(fov_deg)
        if target is not None:
            cmd["target"] = [float(target[0]), float(target[1]), float(target[2])]
        self._send_command(cmd)
    
    def snapshot(
        self,
        path: Union[str, Path],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Take an interactive-viewer snapshot and save it to file.

        Outside CENSOR's render-certificate scope: the interactive viewer is
        controlled through its subprocess; CENSOR certifies offscreen renders.
        """
        output_path = Path(path)
        previous_mtime_ns: Optional[int]
        try:
            previous_mtime_ns = output_path.stat().st_mtime_ns
        except FileNotFoundError:
            previous_mtime_ns = None

        cmd: Dict[str, Any] = {"cmd": "snapshot", "path": str(output_path)}
        if width is not None:
            cmd["width"] = int(width)
        if height is not None:
            cmd["height"] = int(height)
        self._send_command(cmd)
        target_revision = int(self.get_stats().get("applied_command_revision", 0))
        _wait_for_snapshot(
            output_path,
            timeout=max(self._timeout, 2.0),
            previous_mtime_ns=previous_mtime_ns,
        )
        self._wait_for_rendered_revision(
            target_revision,
            timeout=max(self._timeout, 2.0),
        )
    
    def render_animation(
        self,
        animation: "CameraAnimation",
        output_dir: Union[str, Path],
        fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Render animation frames to disk.

        Outside CENSOR's render-certificate scope: this is the interactive
        viewer subprocess path, while CENSOR certifies offscreen renders.
        
        Iterates through the animation, setting camera state for each frame,
        rendering, and saving to PNG files.
        
        Args:
            animation: CameraAnimation with keyframes
            output_dir: Directory to save frames (created if doesn't exist)
            fps: Frames per second (determines total frame count)
            width: Output width in pixels (optional, uses viewer size if None)
            height: Output height in pixels (optional, uses viewer size if None)
            progress_callback: Optional callback(frame, total_frames) for progress
        
        Example:
            >>> anim = CameraAnimation()
            >>> anim.add_keyframe(time=0.0, phi=0, theta=45, radius=5000, fov=60)
            >>> anim.add_keyframe(time=5.0, phi=180, theta=30, radius=3000, fov=60)
            >>> viewer.render_animation(anim, "./frames", fps=30)
        """
        from pathlib import Path as P
        
        output_path = P(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        total_frames = animation.get_frame_count(fps)
        
        for frame in range(total_frames):
            # Calculate time for this frame
            t = frame / fps
            
            # Get interpolated camera state
            state = animation.evaluate(t)
            if state is None:
                continue
            
            # Set camera state
            self.set_orbit_camera(
                phi_deg=state.phi_deg,
                theta_deg=state.theta_deg,
                radius=state.radius,
                fov_deg=state.fov_deg,
                target=getattr(state, "target", None),
            )
            
            # Generate frame path
            frame_path = output_path / f"frame_{frame:04d}.png"
            
            # Render and save
            self.snapshot(str(frame_path), width=width, height=height)
            
            # Progress callback
            if progress_callback is not None:
                progress_callback(frame, total_frames)
    
    def close(self) -> None:
        """Close the viewer window and terminate the subprocess."""
        try:
            if self._socket is not None:
                self._send_command({"cmd": "close"})
        except Exception:
            pass
        finally:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
            _cleanup_paths(self._cleanup_paths)
            self._cleanup_paths.clear()
            from . import sky

            try:
                sky._remove_active_viewer(self)
            except Exception:
                pass
    
    def __enter__(self) -> "ViewerHandle":
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()
    
    @property
    def port(self) -> int:
        """The port the viewer IPC server is listening on."""
        return self._port
    
    @property
    def is_running(self) -> bool:
        """Check if the viewer process is still running."""
        return self._process is not None and self._process.poll() is None


def _find_viewer_binary() -> str:
    """Find the interactive_viewer binary."""
    return _resolve_viewer_binary(__file__)


def open_viewer_async(
    width: int = 1280,
    height: int = 720,
    title: str = "forge3d Interactive Viewer",
    obj_path: Optional[Union[str, Path]] = None,
    gltf_path: Optional[Union[str, Path]] = None,
    terrain_path: Optional[Union[str, Path]] = None,
    fov_deg: float = 60.0,
    timeout: float = _DEFAULT_TIMEOUT,
    ipc_host: str = "127.0.0.1",
    ipc_port: int = 0,
) -> ViewerHandle:
    """Open an interactive viewer in a subprocess with IPC control.
    
    Returns immediately with a ViewerHandle that can be used to send commands
    to the viewer while keeping the Python REPL available.
    
    Args:
        width: Window width in pixels
        height: Window height in pixels
        title: Window title
        obj_path: Optional OBJ file to load on startup
        gltf_path: Optional glTF file to load on startup (mutually exclusive)
        terrain_path: Optional DEM file to load as terrain (mutually exclusive)
        fov_deg: Initial field of view in degrees
        timeout: Timeout for startup and commands in seconds
        ipc_host: Host for IPC server (default: 127.0.0.1)
        ipc_port: Port for IPC server (0 = auto-select)
    
    Returns:
        ViewerHandle for controlling the viewer
    
    Example:
        >>> from forge3d.viewer import open_viewer_async
        >>> v = open_viewer_async(obj_path="model.obj")
        >>> v.set_camera_lookat(eye=[0, 5, 10], target=[0, 0, 0])
        >>> v.snapshot("output.png", width=3840, height=2160)
        >>> v.close()
    """
    del title
    if sum(x is not None for x in (obj_path, gltf_path, terrain_path)) > 1:
        raise ValueError("obj_path, gltf_path, and terrain_path are mutually exclusive")
    
    binary = _find_viewer_binary()
    prepared_terrain_path, cleanup_paths = _prepare_terrain_path(terrain_path)
    
    try:
        # Build command line
        cmd = [
            binary,
            "--ipc-host", ipc_host,
            "--ipc-port", str(ipc_port),
            "--size", f"{width}x{height}",
            "--fov", str(fov_deg),
        ]
        
        if obj_path is not None:
            cmd.extend(["--obj", str(obj_path)])
        elif gltf_path is not None:
            cmd.extend(["--gltf", str(gltf_path)])
        elif prepared_terrain_path is not None:
            cmd.extend(["--terrain", prepared_terrain_path])
        
        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Wait for READY line
        start_time = time.time()
        actual_port: Optional[int] = None
        
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                # Process exited
                output = process.stdout.read() if process.stdout else ""
                raise ViewerError(f"Viewer process exited unexpectedly: {output}")
            
            line = process.stdout.readline() if process.stdout else ""
            if not line:
                time.sleep(0.01)
                continue
            
            match = _READY_PATTERN.search(line)
            if match:
                actual_port = int(match.group(1))
                break
        
        if actual_port is None:
            process.terminate()
            raise ViewerError(f"Timed out waiting for viewer READY line after {timeout}s")

        stdout_thread: Optional[threading.Thread] = None
        if process.stdout is not None:
            def _drain_stdout() -> None:
                try:
                    for _ in process.stdout:
                        pass
                except Exception:
                    pass

            stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
            stdout_thread.start()

        handle = ViewerHandle(
            process,
            ipc_host,
            actual_port,
            timeout=timeout,
            cleanup_paths=cleanup_paths,
        )
        handle._stdout_drain_thread = stdout_thread
        from . import sky

        try:
            sky._set_active_viewer(handle)
        except Exception:
            handle.close()
            raise
        return handle
    except Exception:
        _cleanup_paths(cleanup_paths)
        raise


def open_viewer(
    width: int = 1280,
    height: int = 720,
    title: str = "forge3d Interactive Viewer",
    obj_path: Optional[Union[str, Path]] = None,
    gltf_path: Optional[Union[str, Path]] = None,
    fov_deg: float = 60.0,
    snapshot_path: Optional[str] = None,
    snapshot_width: Optional[int] = None,
    snapshot_height: Optional[int] = None,
    initial_commands: Optional[list[str]] = None,
) -> int:
    """Open an interactive viewer and block until window is closed.
    
    This is a blocking function that launches the viewer as a subprocess
    and waits for it to exit. For non-blocking control, use open_viewer_async().
    
    Args:
        width: Window width in pixels
        height: Window height in pixels
        title: Window title
        obj_path: Optional OBJ file to load on startup
        gltf_path: Optional glTF file to load on startup
        fov_deg: Initial field of view in degrees
        snapshot_path: Optional path for automatic snapshot after startup
        snapshot_width: Snapshot width (requires snapshot_path)
        snapshot_height: Snapshot height (requires snapshot_path)
        initial_commands: List of initial commands to run (e.g., [":gi gtao on"])
    
    Returns:
        Exit code from viewer process
    
    Example:
        >>> import forge3d as f3d
        >>> f3d.open_viewer(obj_path="model.obj", width=1280, height=720)
    """
    if obj_path is not None and gltf_path is not None:
        raise ValueError("obj_path and gltf_path are mutually exclusive")
    from . import sky

    if not initial_commands and not sky._has_observation_replay():
        cmd = [
            _find_viewer_binary(),
            "--size",
            f"{width}x{height}",
            "--fov",
            str(fov_deg),
        ]
        if obj_path is not None:
            cmd.extend(["--obj", str(obj_path)])
        if gltf_path is not None:
            cmd.extend(["--gltf", str(gltf_path)])
        if snapshot_path is not None:
            cmd.extend(["--snapshot", snapshot_path])
            if snapshot_width is not None and snapshot_height is not None:
                cmd.extend(
                    ["--snapshot-size", f"{snapshot_width}x{snapshot_height}"]
                )
        return subprocess.Popen(cmd).wait()

    handle = open_viewer_async(
        width=width,
        height=height,
        title=title,
        obj_path=obj_path,
        gltf_path=gltf_path,
        fov_deg=fov_deg,
    )
    try:
        for command in initial_commands or ():
            print(f"[init] {command.removeprefix(':')}")
        if snapshot_path is not None:
            handle.snapshot(
                snapshot_path,
                width=snapshot_width or width,
                height=snapshot_height or height,
            )
            handle.close()
            return 0
        return handle._process.wait()
    finally:
        handle.close()
