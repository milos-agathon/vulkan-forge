"""Required live acceptance for M-06 Option-2 full geospatial viewer.

The required CI lane sets RUN_M06_VIEWER_CI=1, builds a fresh release viewer,
and rejects every skip through scripts/assert_junit_zero_skips.py.  These tests
use only public NDJSON IPC and emitted artifacts so the evidence represents the
shipped executable rather than an in-process test double.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from PIL import Image
from forge3d import gis
from tests._ssim import ssim


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.environ.get("FORGE3D_M06_ARTIFACT_DIR", ROOT / "tests/artifacts/m06"))
RUN_REQUIRED = os.environ.get("RUN_M06_VIEWER_CI") == "1"
pytestmark = [
    pytest.mark.interactive_viewer,
    pytest.mark.skipif(not RUN_REQUIRED, reason="required M-06 hardware lane only"),
]


def _record_input(path: Path) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, ARTIFACT_DIR / f"input-{path.name}")


def _validated_adapter_pair(probe: dict[str, Any], viewer: dict[str, Any]) -> dict[str, Any]:
    """Require the probe and actual viewer to select the same physical adapter."""
    probe = dict(probe.get("probe", probe))
    pairs = {
        "name": (probe.get("name"), viewer.get("adapter_name")),
        "vendor": (int(probe.get("vendor", -1)), int(viewer.get("adapter_vendor", -2))),
        "device": (int(probe.get("device", -1)), int(viewer.get("adapter_device", -2))),
        "backend": (
            str(probe.get("backend", "")).lower(),
            str(viewer.get("adapter_backend", "")).lower(),
        ),
        "device_type": (
            str(probe.get("device_type", "")).lower(),
            str(viewer.get("adapter_device_type", "")).lower(),
        ),
        "driver": (probe.get("driver"), viewer.get("adapter_driver")),
        "driver_info": (probe.get("driver_info"), viewer.get("adapter_driver_info")),
    }
    mismatches = {key: value for key, value in pairs.items() if value[0] != value[1]}
    if mismatches:
        raise AssertionError(f"probe/viewer adapter mismatch: {mismatches}")
    if pairs["vendor"][1] != 0x10DE:
        raise AssertionError(f"viewer adapter is not NVIDIA: {pairs['vendor'][1]:#x}")
    if pairs["backend"][1] != "vulkan":
        raise AssertionError(f"viewer backend is not Vulkan: {pairs['backend'][1]}")
    if pairs["device_type"][1] != "discretegpu":
        raise AssertionError(f"viewer adapter is not a physical discrete GPU: {pairs['device_type'][1]}")
    return {key: value[1] for key, value in pairs.items()}


def _viewer_binary() -> Path:
    override = os.environ.get("FORGE3D_VIEWER_BINARY")
    if override is not None:
        assert override.strip(), "FORGE3D_VIEWER_BINARY must not be empty"
        path = Path(override)
        assert path.is_file(), f"FORGE3D_VIEWER_BINARY does not exist: {path}"
        return path
    suffix = ".exe" if os.name == "nt" else ""
    path = ROOT / "target" / "release" / f"interactive_viewer{suffix}"
    assert path.is_file(), f"fresh release viewer is required: {path}"
    return path


class ViewerProcess:
    def __init__(
        self,
        *,
        effects: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        args = [str(_viewer_binary()), "--ipc-port", "0", "--size", "640x400"]
        if effects:
            args += [
                "--gi",
                "ssgi:on",
                "--gi",
                "ssr:on",
                "--ssgi-temporal-enable",
                "true",
                "--ssgi-temporal-alpha",
                "0.88",
                "--ssr-enable",
                "true",
                "--fog",
                "on",
                "--fog-temporal",
                "0.8",
            ]
        env = os.environ.copy()
        env.setdefault("WGPU_BACKEND", "vulkan")
        if extra_env:
            env.update(extra_env)
        self.process = subprocess.Popen(
            args,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stdout is not None
        self.lines: list[str] = []
        self._line_queue: queue.Queue[str] = queue.Queue()

        def drain() -> None:
            assert self.process.stdout is not None
            for line in iter(self.process.stdout.readline, ""):
                self.lines.append(line.rstrip())
                self._line_queue.put(line)

        threading.Thread(target=drain, daemon=True).start()
        ready = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
        deadline = time.monotonic() + 45.0
        port: int | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"viewer exited before READY\n{self.log_tail()}")
            try:
                line = self._line_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            match = ready.search(line)
            if match:
                port = int(match.group(1))
                break
        assert port is not None, f"viewer READY timeout\n{self.log_tail()}"
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=20.0)
        self._recv_buffer = b""
        self.wait_frames(2)

    def log_tail(self, count: int = 100) -> str:
        return "\n".join(self.lines[-count:])

    def send(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        assert self.process.poll() is None, f"viewer is not running\n{self.log_tail()}"
        self.socket.settimeout(timeout)
        self.socket.sendall((json.dumps(payload, allow_nan=False) + "\n").encode("utf-8"))
        while b"\n" not in self._recv_buffer:
            chunk = self.socket.recv(65536)
            assert chunk, f"viewer IPC closed\n{self.log_tail()}"
            self._recv_buffer += chunk
        line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
        response = json.loads(line.decode("utf-8"))
        return response

    def ok(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        response = self.send(payload, timeout)
        assert response.get("ok"), f"IPC rejected {payload.get('cmd')}: {response}\n{self.log_tail()}"
        return response

    def stats(self) -> dict[str, Any]:
        return self.ok({"cmd": "get_stats"})["stats"]

    def pick(self, x: int, y: int) -> list[dict[str, Any]]:
        # Drain prior GUI/IPC events so the returned event is correlated with
        # this exact execution-completed command.
        self.ok({"cmd": "poll_pick_events"})
        self.ok({"cmd": "pick_at", "x": x, "y": y})
        events = self.ok({"cmd": "poll_pick_events"}).get("pick_events") or []
        assert all(event["screen_pos"] == [x, y] for event in events)
        return [result for event in events for result in event.get("results", [])]

    def wait_until(
        self, predicate: Callable[[], Any], *, timeout: float = 45.0, description: str
    ) -> Any:
        deadline = time.monotonic() + timeout
        last: Any = None
        while time.monotonic() < deadline:
            assert self.process.poll() is None, f"viewer crashed while waiting for {description}\n{self.log_tail()}"
            last = predicate()
            if last is not None:
                return last
            time.sleep(0.08)
        raise AssertionError(f"timeout waiting for {description}; last={last!r}\n{self.log_tail()}")

    def wait_frames(self, count: int = 2, timeout: float = 30.0) -> dict[str, Any]:
        start = self.stats()["frame_count"] if hasattr(self, "socket") else 0
        return self.wait_until(
            lambda: _matching_stats(self, lambda stats: stats["frame_count"] >= start + count),
            timeout=timeout,
            description=f"{count} new frames",
        )

    def snapshot(self, path: Path, *, width: int = 512, height: int = 320) -> np.ndarray:
        path.unlink(missing_ok=True)
        self.ok({"cmd": "snapshot", "path": str(path), "width": width, "height": height})
        target_revision = self.stats()["applied_command_revision"]

        def loaded() -> np.ndarray | None:
            if not path.is_file() or path.stat().st_size < 1000:
                return None
            try:
                with Image.open(path) as image:
                    image.load()
                    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            except (OSError, ValueError):
                return None

        image = self.wait_until(loaded, timeout=45.0, description=f"snapshot {path.name}")
        rendered = self.wait_until(
            lambda: _matching_stats(
                self,
                lambda stats: stats
                if stats["rendered_frame_revision"] >= target_revision
                else None,
            ),
            timeout=45.0,
            description=f"rendered revision {target_revision}",
        )
        assert rendered["applied_command_revision"] >= target_revision
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, ARTIFACT_DIR / path.name)
        return image

    def close(self, artifact_name: str) -> None:
        try:
            if self.process.poll() is None:
                self.ok({"cmd": "close"}, timeout=5.0)
        except (OSError, AssertionError):
            pass
        try:
            self.socket.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=8.0)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / artifact_name).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _heightfield(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    field = (
        50.0
        + 20.0 * np.sin(xx / max(width, 1) * np.pi * 3.0)
        + 14.0 * np.cos(yy / max(height, 1) * np.pi * 2.0)
        + xx * 0.08
    )
    return np.asarray(field, dtype=np.float32)


def _write_local_tiff(path: Path, values: np.ndarray) -> None:
    gis.write_raster(path, values, overwrite=True)
    _record_input(path)


def _write_geotiff(
    path: Path,
    values: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    span_x: float,
    span_y: float,
    crs: str | None = "EPSG:32633",
    mirrored_x: bool = False,
    south_up: bool = False,
) -> None:
    height, width = values.shape
    pixel_x = span_x / width
    pixel_y = span_y / height
    transform = (
        -pixel_x if mirrored_x else pixel_x,
        0.0,
        origin_x,
        0.0,
        pixel_y if south_up else -pixel_y,
        origin_y,
    )
    gis.write_raster(path, values, transform=transform, crs=crs, overwrite=True)
    _record_input(path)


def _rasterio_oracle(path: Path) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "transform": tuple(dataset.transform)[:6],
            "crs": None if dataset.crs is None else dataset.crs.to_string(),
        }


def _hide_tiff_tag(source: Path, destination: Path, tag: int) -> None:
    """Derive a malformed partial-tag adversary from a native-written TIFF."""
    shutil.copyfile(source, destination)
    data = bytearray(destination.read_bytes())
    endian = bytes(data[:2])
    assert endian in {b"II", b"MM"}
    prefix = "<" if endian == b"II" else ">"
    ifd_offset = struct.unpack_from(f"{prefix}I", data, 4)[0]
    count = struct.unpack_from(f"{prefix}H", data, ifd_offset)[0]
    found = False
    for index in range(count):
        offset = ifd_offset + 2 + index * 12
        if struct.unpack_from(f"{prefix}H", data, offset)[0] == tag:
            struct.pack_into(f"{prefix}H", data, offset, 65000)
            found = True
            break
    assert found, f"native fixture did not contain TIFF tag {tag}"
    destination.write_bytes(data)
    _record_input(destination)


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    assert a.shape == b.shape
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "ssim": ssim(a, b, data_range=1.0),
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
    }


def _parity_passes(metrics: dict[str, float]) -> bool:
    return metrics["ssim"] >= 0.999 and metrics["mae"] <= 0.5 / 255.0


def _write_json(name: str, payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_diff(name: str, left: np.ndarray, right: np.ndarray) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    difference = np.clip(np.abs(left - right) * 4.0, 0.0, 1.0)
    Image.fromarray(np.asarray(difference * 255.0, dtype=np.uint8)).save(
        ARTIFACT_DIR / name
    )


def _matching_stats(
    viewer: ViewerProcess, predicate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any] | None:
    stats = viewer.stats()
    return stats if predicate(stats) else None


def _active_volumetrics(viewer: ViewerProcess) -> dict[str, Any] | None:
    report = viewer.ok({"cmd": "get_terrain_volumetrics_report"})[
        "terrain_volumetrics_report"
    ]
    return report if report["active_volume_count"] > 0 else None


def _load_terrain(viewer: ViewerProcess, path: Path) -> dict[str, Any]:
    before = viewer.stats()["frame_count"]
    viewer.ok({"cmd": "load_terrain", "path": str(path)}, timeout=45.0)
    return viewer.wait_until(
        lambda: _matching_stats(
            viewer,
            lambda stats: stats["frame_count"] >= before + 2
            and stats["active_camera"] == "terrain",
        ),
        timeout=45.0,
        description=f"terrain {path.name}",
    )


def _camera(viewer: ViewerProcess, *, target: list[float], phi: float, radius: float) -> None:
    viewer.ok(
        {
            "cmd": "set_terrain_camera",
            "phi_deg": phi,
            "theta_deg": 52.0,
            "radius": radius,
            "fov_deg": 50.0,
            "target": target,
        }
    )
    viewer.wait_frames(2)


def _write_obj(path: Path) -> None:
    path.write_text(
        """v -1 0 -1
v 1 0 -1
v 1 0 1
v -1 0 1
v -1 2 -1
v 1 2 -1
v 1 2 1
v -1 2 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
""",
        encoding="utf-8",
    )
    _record_input(path)


def _write_las(path: Path, *, count: int, center_x: float, center_y: float) -> None:
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([center_x, center_y, 0.0])
    las = laspy.LasData(header)
    side = int(np.ceil(np.sqrt(count)))
    idx = np.arange(count, dtype=np.int64)
    gx = idx % side
    gy = idx // side
    las.x = center_x + (gx / max(side - 1, 1) - 0.5) * 900.0
    las.y = center_y + (gy / max(side - 1, 1) - 0.5) * 600.0
    las.z = 90.0 + 30.0 * np.sin(gx / 21.0) + 20.0 * np.cos(gy / 17.0)
    las.red = np.full(count, 65535, dtype=np.uint16)
    las.green = np.asarray((gy % 256) * 257, dtype=np.uint16)
    las.blue = np.asarray((gx % 256) * 257, dtype=np.uint16)
    las.intensity = np.asarray(idx % 65535, dtype=np.uint16)
    las.write(path)
    _record_input(path)


def _write_isolated_las(path: Path, *, center_x: float) -> None:
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([center_x, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x = np.full(3, center_x)
    las.y = np.array([-10_000.0, 0.0, 10_000.0])
    las.z = np.zeros(3)
    las.red = np.full(3, 65535, dtype=np.uint16)
    las.green = np.zeros(3, dtype=np.uint16)
    las.blue = np.zeros(3, dtype=np.uint16)
    las.write(path)
    _record_input(path)


def _pick_center(viewer: ViewerProcess) -> dict[str, Any]:
    results = viewer.pick(320, 200)
    assert results
    return results[0]


def _scatter_payload(width: int, height: int) -> dict[str, Any]:
    positions = [[-0.6, 0.0, -0.6], [0.6, 0.0, -0.6], [0.0, 2.2, 0.0], [0.6, 0.0, 0.6], [-0.6, 0.0, 0.6]]
    normals = [[0.0, 1.0, 0.0]] * len(positions)
    indices = [0, 1, 2, 1, 3, 2, 3, 4, 2, 4, 0, 2, 0, 4, 3, 0, 3, 1]
    transforms = []
    for x, z, scale in [(0.28, 0.3, 9.0), (0.5, 0.5, 12.0), (0.72, 0.64, 8.0)]:
        transforms.append(
            [scale, 0.0, 0.0, width * x, 0.0, scale, 0.0, 80.0, 0.0, 0.0, scale, height * z, 0.0, 0.0, 0.0, 1.0]
        )
    return {
        "cmd": "set_terrain_scatter",
        "batches": [
            {
                "name": "m06-nonsquare",
                "color": [0.05, 0.95, 0.1, 1.0],
                "max_draw_distance": 10000.0,
                "transforms": transforms,
                "levels": [{"positions": positions, "normals": normals, "indices": indices, "max_distance": 10000.0}],
            }
        ],
    }


def test_m06_metric_and_precision_negative_controls() -> None:
    yy, xx = np.mgrid[0:64, 0:96]
    asymmetric = np.stack(
        [
            (xx / 95.0) ** 2,
            (yy / 63.0) * (xx < 37),
            ((xx - 71) ** 2 + (yy - 19) ** 2 < 81).astype(np.float64),
        ],
        axis=-1,
    ).astype(np.float32)
    warped = 0.5 * asymmetric + 0.5 * np.roll(asymmetric, 1, axis=1)
    warp_metrics = _metrics(asymmetric, warped)
    assert not _parity_passes(warp_metrics), warp_metrics

    earth_x = 6_378_137.0
    unanchored_delta = float(np.float32(earth_x + 0.001) - np.float32(earth_x))
    anchored_delta = float(np.float32((earth_x + 0.001) - earth_x))
    assert unanchored_delta == 0.0
    assert anchored_delta == pytest.approx(0.001, abs=1.0e-7)
    co_located = np.asarray([[20.0, 30.0], [20.0, 30.0]])
    assert float(np.linalg.norm(np.diff(co_located, axis=0))) < 1.0
    compatible = {
        "name": "NVIDIA RTX",
        "vendor": 0x10DE,
        "device": 123,
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "driver": "driver",
        "driver_info": "info",
    }
    actual = {f"adapter_{key}": value for key, value in compatible.items()}
    assert _validated_adapter_pair(compatible, actual)["backend"] == "vulkan"
    for field, incompatible in (
        ("vendor", 0x1002),
        ("backend", "Dx12"),
        ("device_type", "Cpu"),
        ("device", 456),
    ):
        mutated = dict(actual)
        mutated[f"adapter_{field}"] = incompatible
        with pytest.raises(AssertionError, match="adapter"):
            _validated_adapter_pair(compatible, mutated)
    _write_json(
        "negative-controls.json",
        {
            "backend": os.environ.get("WGPU_BACKEND"),
            "half_pixel_warp": warp_metrics,
            "unanchored_earth_mm_delta": unanchored_delta,
            "anchored_earth_mm_delta": anchored_delta,
            "zero_distance_px": 0.0,
        },
    )


def test_m06_actual_viewer_adapter_matches_required_probe() -> None:
    probe_path = ARTIFACT_DIR / "adapter-probe.json"
    assert probe_path.is_file(), f"required adapter probe evidence is missing: {probe_path}"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    viewer = ViewerProcess()
    try:
        stats = viewer.stats()
        identity = _validated_adapter_pair(probe, stats)
        _write_json(
            "viewer-adapter.json",
            {"probe": probe, "viewer_stats": stats, "validated_identity": identity},
        )
    finally:
        viewer.close("viewer-adapter.log")


def test_m06_georeferencing_fail_closed_and_local_pixel_lock(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        values = _heightfield(160, 96)
        local = tmp_path / "local-no-transform.tif"
        translated = tmp_path / "translated-arbitrary-crs.tif"
        mirrored = tmp_path / "invalid-mirrored.tif"
        south_up = tmp_path / "invalid-south-up.tif"
        partial = tmp_path / "invalid-scale-only.tif"
        _write_local_tiff(local, values)
        _write_geotiff(translated, values, origin_x=6_378_000.0, origin_y=1_000_000.0, span_x=160.0, span_y=96.0, crs="EPSG:32633")
        _write_geotiff(mirrored, values, origin_x=6_378_000.0, origin_y=1_000_000.0, span_x=160.0, span_y=96.0, mirrored_x=True)
        _write_geotiff(south_up, values, origin_x=6_378_000.0, origin_y=1_000_000.0, span_x=160.0, span_y=96.0, south_up=True)
        _hide_tiff_tag(translated, partial, 33922)
        oracles = {
            path.name: _rasterio_oracle(path)
            for path in (local, translated, mirrored, south_up, partial)
        }
        assert oracles[translated.name]["crs"] == "EPSG:32633"
        assert oracles[local.name]["crs"] is None

        _load_terrain(viewer, local)
        _camera(viewer, target=[80.0, 45.0, 48.0], phi=137.0, radius=280.0)
        local_frame = viewer.snapshot(tmp_path / "local.png")

        _load_terrain(viewer, translated)
        _camera(viewer, target=[6_378_080.0, 45.0, -999_952.0], phi=137.0, radius=280.0)
        translated_frame = viewer.snapshot(tmp_path / "translated.png")
        parity = _metrics(local_frame, translated_frame)
        _write_diff("local-vs-translated-diff.png", local_frame, translated_frame)
        assert _parity_passes(parity), parity

        rejections: dict[str, Any] = {}
        invariant_keys = (
            "active_camera",
            "camera_anchor_origin",
            "camera_rebase_count",
            "history_invalidation_count",
            "applied_command_revision",
            "tracked_buffer_count",
            "tracked_texture_count",
            "tracked_total_bytes",
            "host_visible_bytes",
        )
        for invalid, diagnostic in (
            (mirrored, "unsupported_axis_orientation"),
            (south_up, "unsupported_axis_orientation"),
            (partial, "ModelTiepointTag is missing"),
        ):
            before = viewer.stats()
            rejected = viewer.send({"cmd": "load_terrain", "path": str(invalid)})
            after = viewer.stats()
            assert rejected.get("ok") is False, rejected
            assert diagnostic in rejected.get("error", ""), rejected
            assert {key: after[key] for key in invariant_keys} == {
                key: before[key] for key in invariant_keys
            }
            rejections[invalid.name] = {
                "response": rejected,
                "before": before,
                "after": after,
            }
        _write_json("georeferencing.json", {"backend": os.environ.get("WGPU_BACKEND"), "oracles": oracles, "parity": parity, "rejections": rejections})
    finally:
        viewer.close("georeferencing-viewer.log")


def test_m06_nonzero_origin_effects_orbit_no_flash_and_scatter(tmp_path: Path) -> None:
    viewer = ViewerProcess(effects=True)
    try:
        values = _heightfield(192, 96)
        terrain = tmp_path / "non-square-effects.tif"
        # Pixel aspect is 2:1 while the physical footprint is 4:3; this catches
        # any path that reconstructs depth from raster dimensions.
        _write_geotiff(terrain, values, origin_x=6_378_000.0, origin_y=1_000_000.0, span_x=4000.0, span_y=3000.0)
        _load_terrain(viewer, terrain)
        target = [6_380_000.0, 50.0, -998_500.0]
        _camera(viewer, target=target, phi=0.0, radius=6000.0)
        viewer.ok({"cmd": "set_taa_enabled", "enabled": True})
        viewer.ok(
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "shadow_technique": "csm",
                "shadow_map_res": 1024,
                "dof": {"enabled": True, "f_stop": 8.0, "focus_distance": 6000.0, "quality": "medium"},
                "volumetrics": {
                    "enabled": True,
                    "mode": "heterogeneous",
                    "density": 0.006,
                    "steps": 24,
                    "half_res": True,
                    "density_volumes": [
                        {"preset": "fog_bank", "center": [96.0, 16.0, 48.0], "size": [60.0, 28.0, 28.0], "resolution": [32, 16, 16], "density_scale": 0.35, "seed": 7}
                    ],
                },
            }
        )
        enabled = viewer.wait_until(
            lambda: _matching_stats(
                viewer,
                lambda stats: all(
                    stats[key]
                    for key in (
                        "taa_enabled",
                        "ssgi_enabled",
                        "ssgi_temporal_enabled",
                        "ssr_enabled",
                        "fog_enabled",
                    )
                ),
            ),
            timeout=45.0,
            description="all temporal/effect telemetry flags",
        )
        report = viewer.wait_until(
            lambda: _active_volumetrics(viewer),
            timeout=45.0,
            description="heterogeneous density volume",
        )

        viewer.wait_frames(8)
        no_overlay = viewer.snapshot(tmp_path / "no-overlay.png")
        overlay = tmp_path / "terrain-overlay.png"
        Image.new("RGBA", (64, 64), (238, 44, 166, 255)).save(overlay)
        _record_input(overlay)
        viewer.ok(
            {
                "cmd": "load_overlay",
                "name": "m06-overlay",
                "path": str(overlay),
                "extent": [0.2, 0.2, 0.8, 0.8],
                "opacity": 0.9,
                "z_order": 0,
            }
        )
        viewer.ok({"cmd": "set_overlays_enabled", "enabled": True})
        viewer.wait_frames(4)
        with_overlay = viewer.snapshot(tmp_path / "with-overlay.png")
        overlay_delta = _metrics(no_overlay, with_overlay)
        _write_diff("overlay-diff.png", no_overlay, with_overlay)
        assert overlay_delta["mae"] > 0.00005, overlay_delta

        viewer.ok(
            {
                "cmd": "set_scene_review_state",
                "state": {
                    "review_layers": [
                        {
                            "id": "effects-label",
                            "name": "Effects label",
                            "labels": [
                                {
                                    "kind": "point",
                                    "text": "M06",
                                    "world_pos": [target[0], 80.0, target[2]],
                                }
                            ],
                        }
                    ],
                    "variants": [
                        {
                            "id": "effects",
                            "active_layer_ids": ["effects-label"],
                        }
                    ],
                    "active_variant_id": "effects",
                },
            }
        )
        labels = viewer.wait_until(
            lambda: viewer.ok({"cmd": "list_review_layers"})["review_layers"]
            or None,
            timeout=45.0,
            description="non-zero-origin label layer",
        )

        no_scatter = viewer.snapshot(tmp_path / "no-scatter.png")
        viewer.ok(_scatter_payload(192, 96))
        viewer.wait_frames(5)
        with_scatter = viewer.snapshot(tmp_path / "with-scatter.png")
        scatter_delta = _metrics(no_scatter, with_scatter)
        _write_diff("scatter-diff.png", no_scatter, with_scatter)
        assert scatter_delta["mae"] > 0.00005, scatter_delta

        viewer.wait_frames(10)
        rebase_start = viewer.stats()
        stable_resources = (
            rebase_start["tracked_buffer_count"],
            rebase_start["tracked_texture_count"],
            rebase_start["tracked_total_bytes"],
            rebase_start["host_visible_bytes"],
        )
        atlas_id = viewer.ok({"cmd": "get_terrain_volumetrics_report"})[
            "terrain_volumetrics_report"
        ]["atlas_allocation_id"]
        assert atlas_id != 0
        rebase_evidence = []
        reference_frames: dict[str, np.ndarray] = {}
        for index in range(10):
            key = "east" if index % 2 == 0 else "center"
            shifted_target = (
                [target[0] + 1100.0, target[1], target[2]]
                if key == "east"
                else target
            )
            viewer.ok(
                {
                    "cmd": "set_terrain_camera",
                    "phi_deg": 0.0,
                    "theta_deg": 52.0,
                    "radius": 5400.0,
                    "fov_deg": 50.0,
                    "target": shifted_target,
                }
            )
            transition = viewer.wait_until(
                lambda expected=rebase_start["camera_rebase_count"] + index + 1: _matching_stats(
                    viewer,
                    lambda stats: stats["camera_rebase_count"] >= expected,
                ),
                timeout=45.0,
                description=f"rebase {index + 1}",
            )
            settled = viewer.wait_frames(3)
            current_report = viewer.ok({"cmd": "get_terrain_volumetrics_report"})[
                "terrain_volumetrics_report"
            ]
            current_resources = (
                settled["tracked_buffer_count"],
                settled["tracked_texture_count"],
                settled["tracked_total_bytes"],
                settled["host_visible_bytes"],
            )
            assert current_resources == stable_resources
            assert current_report["atlas_allocation_id"] == atlas_id
            assert all(
                settled[name]
                for name in (
                    "taa_history_valid",
                    "ssgi_history_valid",
                    "ssr_history_valid",
                    "fog_history_valid",
                )
            ), settled
            frame = viewer.snapshot(tmp_path / f"rebase-{index:02d}.png")
            repeat_metrics = None
            if key in reference_frames:
                repeat_metrics = _metrics(reference_frames[key], frame)
                assert repeat_metrics["ssim"] >= 0.97, repeat_metrics
                assert repeat_metrics["mae"] <= 0.03, repeat_metrics
            else:
                reference_frames[key] = frame
            rebase_evidence.append(
                {
                    "index": index,
                    "target": shifted_target,
                    "transition": transition,
                    "settled": settled,
                    "resources": current_resources,
                    "atlas_allocation_id": current_report["atlas_allocation_id"],
                    "repeat_metrics": repeat_metrics,
                }
            )
        assert viewer.stats()["camera_rebase_count"] == rebase_start["camera_rebase_count"] + 10
        assert viewer.stats()["history_invalidation_count"] == rebase_start["history_invalidation_count"] + 10

        start = viewer.stats()
        orbit_frames = []
        for phi in range(0, 361, 30):
            _camera(viewer, target=target, phi=float(phi), radius=5400.0)
            orbit_frames.append(viewer.stats())
        end = viewer.stats()
        assert end["camera_rebase_count"] == start["camera_rebase_count"]
        assert end["history_invalidation_count"] == start["history_invalidation_count"]
        resource_counts = {
            (stats["tracked_buffer_count"], stats["tracked_texture_count"])
            for stats in orbit_frames
        }
        assert len(resource_counts) == 1, resource_counts

        frames = []
        for index in range(3):
            viewer.wait_frames(3)
            frames.append(viewer.snapshot(tmp_path / f"stationary-{index}.png"))
        no_flash = [_metrics(frames[index], frames[index + 1]) for index in range(2)]
        _write_diff("effects-orbit-0-1-diff.png", frames[0], frames[1])
        _write_diff("effects-orbit-1-2-diff.png", frames[1], frames[2])
        assert min(metric["ssim"] for metric in no_flash) >= 0.97, no_flash
        assert max(metric["mae"] for metric in no_flash) <= 0.03, no_flash
        assert enabled["within_host_visible_budget"] is True
        _write_json("effects-orbit.json", {"backend": os.environ.get("WGPU_BACKEND"), "enabled": enabled, "volumetrics": report, "labels": labels, "overlay_delta": overlay_delta, "scatter_delta": scatter_delta, "rebase_start": rebase_start, "rebases": rebase_evidence, "start": start, "end": end, "orbit": orbit_frames, "no_flash": no_flash})
    finally:
        viewer.close("effects-orbit-viewer.log")


def test_m06_terrain_replacement_rebinds_shadow_resources_atomically(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        yy_a, xx_a = np.mgrid[0:64, 0:96]
        dem_a = np.where(xx_a < 48, 15.0, 240.0).astype(np.float32)
        yy_b, xx_b = np.mgrid[0:80, 0:128]
        dem_b = (
            40.0
            + 180.0 * np.exp(-((xx_b - 88.0) ** 2 + (yy_b - 27.0) ** 2) / 180.0)
        ).astype(np.float32)
        path_a = tmp_path / "shadow-a.tif"
        path_b = tmp_path / "shadow-b.tif"
        _write_geotiff(
            path_a,
            dem_a,
            origin_x=500_000.0,
            origin_y=5_500_000.0,
            span_x=960.0,
            span_y=640.0,
        )
        _write_geotiff(
            path_b,
            dem_b,
            origin_x=500_000.0,
            origin_y=5_500_000.0,
            span_x=1280.0,
            span_y=800.0,
        )
        _load_terrain(viewer, path_a)
        viewer.ok(
            {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "shadow_technique": "csm",
                "shadow_map_res": 1024,
            }
        )
        viewer.ok(
            {
                "cmd": "set_terrain_sun",
                "azimuth_deg": 80.0,
                "elevation_deg": 18.0,
                "intensity": 1.2,
            }
        )
        _camera(
            viewer,
            target=[500_480.0, 100.0, -5_499_680.0],
            phi=135.0,
            radius=1500.0,
        )
        viewer.wait_frames(5)
        frame_a = viewer.snapshot(tmp_path / "shadow-a.png")
        stats_a = viewer.stats()
        assert stats_a["terrain_revision"] > 0
        assert stats_a["terrain_shadow_binding_revision"] == stats_a["terrain_revision"]

        _load_terrain(viewer, path_b)
        _camera(
            viewer,
            target=[500_640.0, 100.0, -5_499_600.0],
            phi=135.0,
            radius=1800.0,
        )
        viewer.wait_frames(5)
        frame_b = viewer.snapshot(tmp_path / "shadow-b.png")
        stats_b = viewer.stats()
        assert stats_b["terrain_revision"] > stats_a["terrain_revision"]
        assert (
            stats_b["terrain_heightmap_allocation_id"]
            != stats_a["terrain_heightmap_allocation_id"]
        )
        assert stats_b["terrain_heightmap_bytes"] == 128 * 80 * 4
        assert stats_b["terrain_heightmap_bytes"] != stats_a["terrain_heightmap_bytes"]
        assert stats_b["terrain_shadow_binding_revision"] == stats_b["terrain_revision"]
        assert stats_b["tracked_texture_count"] == stats_a["tracked_texture_count"]

        delta = _metrics(frame_a, frame_b)
        assert delta["mae"] > 0.005, delta
        _write_diff("terrain-shadow-a-b-diff.png", frame_a, frame_b)
        _write_json(
            "terrain-shadow-replacement.json",
            {"terrain_a": stats_a, "terrain_b": stats_b, "image_metrics": delta},
        )
    finally:
        viewer.close("terrain-shadow-replacement-viewer.log")


def test_m06_rigid_rebases_reset_all_temporal_history_without_reallocation(
    tmp_path: Path,
) -> None:
    viewer = ViewerProcess(effects=True)
    try:
        obj = tmp_path / "temporal-rigid-box.obj"
        _write_obj(obj)
        viewer.ok({"cmd": "load_obj", "path": str(obj)})
        viewer.ok({"cmd": "set_taa_enabled", "enabled": True})
        base = np.asarray([6_378_137.0, 100.0, -250_000.0], dtype=np.float64)

        def place(offset_x: float) -> None:
            center = base + np.asarray([offset_x, 0.0, 0.0])
            viewer.ok(
                {
                    "cmd": "set_transform",
                    "translation": center.tolist(),
                    "scale": [2.0, 2.0, 2.0],
                }
            )
            viewer.ok(
                {
                    "cmd": "cam_lookat",
                    "eye": (center + np.asarray([0.0, 3.0, 14.0])).tolist(),
                    "target": (center + np.asarray([0.0, 1.0, 0.0])).tolist(),
                    "up": [0.0, 1.0, 0.0],
                }
            )

        place(0.0)
        viewer.wait_frames(8)
        reference = viewer.snapshot(tmp_path / "temporal-rigid-reference.png")
        baseline = viewer.stats()
        identities = baseline["temporal_history_allocation_ids"]
        assert all(int(value) > 0 for value in identities), baseline

        records = []
        for index, offset in enumerate((1500.0, 0.0, 1500.0, 0.0)):
            place(offset)
            # Queue the snapshot immediately after publication: the captured
            # image is the first rendered frame in the new anchor frame.
            image = viewer.snapshot(tmp_path / f"temporal-rigid-{index}.png")
            stats = viewer.stats()
            metrics = _metrics(reference, image)
            _write_diff(f"temporal-rigid-{index}-diff.png", reference, image)
            assert stats["temporal_history_allocation_ids"] == identities
            assert stats["tracked_buffer_count"] == baseline["tracked_buffer_count"]
            assert stats["tracked_texture_count"] == baseline["tracked_texture_count"]
            assert stats["history_invalidation_count"] >= (
                baseline["history_invalidation_count"] + index + 1
            )
            assert stats["rendered_frame_revision"] >= stats["applied_command_revision"]
            assert metrics["ssim"] >= 0.97 and metrics["mae"] <= 0.03, metrics
            records.append({"offset_x": offset, "stats": stats, "metrics": metrics})

        _write_json(
            "temporal-rigid-rebases.json",
            {"baseline": baseline, "resource_ids": identities, "records": records},
        )
    finally:
        viewer.close("temporal-rigid-rebases-viewer.log")


def test_m06_object_pointcloud_coexistence_and_allocation_free_rebases(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        values = _heightfield(192, 128)
        terrain = tmp_path / "coexistence.tif"
        origin_x, origin_y = 6_378_000.0, 1_000_000.0
        span_x, span_y = 4000.0, 2600.0
        center_x, center_z = origin_x + span_x / 2.0, -origin_y + span_y / 2.0
        _write_geotiff(terrain, values, origin_x=origin_x, origin_y=origin_y, span_x=span_x, span_y=span_y)
        _load_terrain(viewer, terrain)
        _camera(viewer, target=[center_x, 60.0, center_z], phi=145.0, radius=5000.0)
        terrain_only = viewer.snapshot(tmp_path / "terrain-before-object.png")

        obj = tmp_path / "anchored-box.obj"
        _write_obj(obj)
        viewer.ok({"cmd": "load_obj", "path": str(obj)})
        angle = np.deg2rad(37.0) * 0.5
        viewer.ok({"cmd": "set_transform", "translation": [center_x, 170.0, center_z], "rotation_quat": [0.0, float(np.sin(angle)), 0.0, float(np.cos(angle))], "scale": [120.0, 180.0, 80.0]})
        object_stats = viewer.wait_until(
            lambda: _matching_stats(
                viewer,
                lambda stats: stats["scene_has_mesh"]
                and not stats["transform_is_identity"],
            ),
            timeout=45.0,
            description="anchored object transform",
        )
        assert object_stats["active_camera"] == "terrain"
        object_frame = viewer.snapshot(tmp_path / "translated-rotated-scaled-object.png")
        object_difference = np.max(np.abs(object_frame - terrain_only), axis=2)
        object_pixels = np.argwhere(object_difference > 3.0 / 255.0)
        assert object_pixels.shape[0] > 20
        object_centroid = object_pixels.mean(axis=0)
        object_pick_x = int(round(object_centroid[1] * 640.0 / object_frame.shape[1]))
        object_pick_y = int(round(object_centroid[0] * 400.0 / object_frame.shape[0]))
        object_results = viewer.pick(object_pick_x, object_pick_y)
        object_pick = next(
            result for result in object_results if result["layer_name"] == "Object"
        )
        assert abs(object_pick["world_pos"][0] - center_x) < 300.0
        assert abs(object_pick["world_pos"][2] - center_z) < 300.0
        assert 169.999 <= object_pick["world_pos"][1] <= 530.001
        _write_diff("object-placement-diff.png", terrain_only, object_frame)

        point_count = 500_000
        las = tmp_path / "earth-scale-500k.las"
        _write_las(las, count=point_count, center_x=center_x, center_y=center_z)
        viewer.ok({"cmd": "load_point_cloud", "path": str(las), "point_size": 5.0, "max_points": point_count, "color_mode": "rgb"}, timeout=90.0)
        loaded = viewer.wait_until(
            lambda: _matching_stats(
                viewer, lambda stats: stats["point_cloud_point_count"] == point_count
            ),
            timeout=120.0,
            description="500k point cloud",
        )
        assert loaded["active_camera"] == "terrain"
        assert loaded["point_cloud_source_bytes"] == point_count * 48
        assert loaded["point_cloud_render_cache_bytes"] == point_count * 48
        assert loaded["point_cloud_gpu_instance_bytes"] == point_count * 48
        assert loaded["within_host_visible_budget"] is True

        viewer.ok({"cmd": "set_point_cloud_params", "visible": False})
        hidden = viewer.snapshot(tmp_path / "point-hidden.png")
        viewer.ok({"cmd": "set_point_cloud_params", "visible": True, "point_size": 6.0})
        visible = viewer.snapshot(tmp_path / "point-visible.png")
        point_delta = _metrics(hidden, visible)
        _write_diff("pointcloud-visibility-diff.png", hidden, visible)
        assert point_delta["mae"] > 0.0001, point_delta

        stable = viewer.stats()
        counts = []
        point_buffer_ids = []
        anchors = []
        for index in range(10):
            target_x = center_x + (1500.0 if index % 2 == 0 else 0.0)
            _camera(viewer, target=[target_x, 60.0, center_z], phi=145.0, radius=5000.0)
            stats = viewer.stats()
            counts.append([stats["tracked_buffer_count"], stats["tracked_texture_count"]])
            point_buffer_ids.append(stats["point_cloud_gpu_instance_id"])
            anchors.append(stats["camera_anchor_origin"])
        assert all(count == counts[0] for count in counts), counts
        assert counts[0] == [stable["tracked_buffer_count"], stable["tracked_texture_count"]]
        assert point_buffer_ids == [loaded["point_cloud_gpu_instance_id"]] * 10
        assert viewer.stats()["camera_rebase_count"] >= stable["camera_rebase_count"] + 9
        viewer.ok({"cmd": "set_point_cloud_params", "visible": False})
        object_after_rebases = viewer.snapshot(tmp_path / "object-after-rebases.png")
        object_rebase_metrics = _metrics(object_frame, object_after_rebases)
        assert _parity_passes(object_rebase_metrics), object_rebase_metrics
        object_pick_after = next(
            result
            for result in viewer.pick(object_pick_x, object_pick_y)
            if result["layer_name"] == "Object"
        )
        assert object_pick_after["world_pos"] == pytest.approx(
            object_pick["world_pos"], abs=0.0005
        )
        _write_json("coexistence-rebases.json", {"backend": os.environ.get("WGPU_BACKEND"), "object": object_stats, "object_centroid_yx": object_centroid.tolist(), "object_pick": object_pick, "object_pick_after_rebases": object_pick_after, "object_rebase_metrics": object_rebase_metrics, "loaded": loaded, "point_delta": point_delta, "counts": counts, "point_buffer_ids": point_buffer_ids, "anchors": anchors})
    finally:
        viewer.close("coexistence-viewer.log")


def test_m06_pointcloud_visibility_culling_picking_and_hidden_precedence(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        obj = tmp_path / "local-visible-mesh.obj"
        _write_obj(obj)
        viewer.ok({"cmd": "load_obj", "path": str(obj)})

        center_x = 6_378_137.0
        las = tmp_path / "isolated-earth-points.las"
        _write_isolated_las(las, center_x=center_x)
        viewer.ok(
            {
                "cmd": "load_point_cloud",
                "path": str(las),
                "point_size": 12.0,
                "max_points": 3,
                "color_mode": "rgb",
            }
        )
        viewer.wait_frames(3)
        assert viewer.stats()["active_camera"] == "point_cloud"

        viewer.ok({"cmd": "set_point_cloud_params", "visible": False})
        hidden = viewer.wait_until(
            lambda: _matching_stats(
                viewer,
                lambda stats: stats if stats["active_camera"] == "general" else None,
            ),
            description="hidden point cloud relinquishes camera ownership",
        )
        assert hidden["point_cloud_point_count"] == 0

        viewer.ok(
            {
                "cmd": "set_point_cloud_params",
                "visible": True,
                "radius": 0.1,
                "phi": 0.0,
                "theta": 0.5,
            }
        )
        first_image = viewer.snapshot(tmp_path / "isolated-points-before-rebase.png")
        first = viewer.wait_until(
            lambda: _matching_stats(
                viewer,
                lambda stats: stats
                if stats["point_cloud_visible_point_count"] == 1
                else None,
            ),
            description="isolated point frustum cull",
        )
        first_pick = _pick_center(viewer)
        assert first_pick["layer_name"] == "PointCloud"
        assert first_pick["world_pos"] == pytest.approx([center_x, 0.0, 0.0], abs=1e-9)

        viewer.ok({"cmd": "set_point_cloud_params", "phi": float(np.pi)})
        second_image = viewer.snapshot(tmp_path / "isolated-points-after-rebase.png")
        second = viewer.wait_until(
            lambda: _matching_stats(
                viewer,
                lambda stats: stats
                if stats["camera_rebase_count"] > first["camera_rebase_count"]
                and stats["point_cloud_visible_point_count"] == 1
                else None,
            ),
            description="rebased isolated point cull",
        )
        second_pick = _pick_center(viewer)
        assert second_pick["layer_name"] == "PointCloud"
        assert second_pick["world_pos"] == pytest.approx(first_pick["world_pos"], abs=1e-9)
        assert second["point_cloud_gpu_instance_id"] == first["point_cloud_gpu_instance_id"]
        assert second["point_cloud_gpu_instance_bytes"] == first["point_cloud_gpu_instance_bytes"]

        parity = _metrics(first_image, second_image)
        _write_diff("isolated-point-rebase-diff.png", first_image, second_image)
        _write_json(
            "pointcloud-cull-pick.json",
            {
                "hidden": hidden,
                "before": first,
                "after": second,
                "before_pick": first_pick,
                "after_pick": second_pick,
                "image_metrics": parity,
            },
        )
    finally:
        viewer.close("pointcloud-cull-pick-viewer.log")


def test_m06_pointcloud_repack_failpoint_rejects_before_anchor_publication(
    tmp_path: Path,
) -> None:
    viewer = ViewerProcess(
        extra_env={"FORGE3D_M06_POINT_REPACK_FAILPOINT": "before_anchor_publish"}
    )
    try:
        cloud = tmp_path / "point-repack-failpoint.las"
        _write_isolated_las(cloud, center_x=6_378_137.0)
        viewer.ok(
            {
                "cmd": "load_point_cloud",
                "path": str(cloud),
                "point_size": 10.0,
                "max_points": 3,
                "color_mode": "rgb",
            }
        )
        viewer.ok({"cmd": "set_point_cloud_params", "visible": False})
        viewer.ok(
            {
                "cmd": "cam_lookat",
                "eye": [0.0, 0.0, 20.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
            }
        )
        viewer.wait_frames(3)
        before_image = viewer.snapshot(tmp_path / "point-repack-before.png")
        before_stats = viewer.stats()
        before_pick = viewer.pick(320, 200)

        rejected = viewer.send({"cmd": "set_point_cloud_params", "visible": True})
        assert rejected.get("ok") is False, rejected
        assert "point_cloud_repack_failpoint" in rejected.get("error", ""), rejected
        viewer.wait_frames(3)
        after_image = viewer.snapshot(tmp_path / "point-repack-after.png")
        after_stats = viewer.stats()
        after_pick = viewer.pick(320, 200)

        invariant_keys = (
            "active_camera",
            "camera_anchor_origin",
            "camera_rebase_count",
            "point_cloud_point_count",
            "point_cloud_visible_point_count",
            "point_cloud_source_bytes",
            "point_cloud_render_cache_bytes",
            "point_cloud_gpu_instance_bytes",
            "point_cloud_gpu_instance_id",
            "tracked_buffer_count",
            "tracked_texture_count",
            "tracked_total_bytes",
            "host_visible_bytes",
        )
        assert {key: after_stats[key] for key in invariant_keys} == {
            key: before_stats[key] for key in invariant_keys
        }
        assert after_pick == before_pick
        parity = _metrics(before_image, after_image)
        assert parity["ssim"] >= 0.99 and parity["mae"] <= 0.01, parity
        _write_diff("point-repack-rollback-diff.png", before_image, after_image)
        _write_json(
            "point-repack-rollback.json",
            {
                "before_stats": before_stats,
                "after_stats": after_stats,
                "before_pick": before_pick,
                "after_pick": after_pick,
                "rejection": rejected,
                "image_metrics": parity,
            },
        )
    finally:
        viewer.close("point-repack-failpoint-viewer.log")


def _color_centroid(frame: np.ndarray, channel: int) -> tuple[int, np.ndarray | None]:
    other = [index for index in range(3) if index != channel]
    mask = (frame[..., channel] > 0.75) & (frame[..., other[0]] < 0.35) & (frame[..., other[1]] < 0.35)
    coords = np.argwhere(mask)
    return int(coords.shape[0]), None if coords.size == 0 else coords.mean(axis=0)


def test_m06_live_millimetre_vectors_and_scene_review_transaction(tmp_path: Path) -> None:
    viewer = ViewerProcess()
    try:
        values = np.zeros((128, 256), dtype=np.float32)
        terrain = tmp_path / "millimetre.tif"
        origin_x, origin_y = 6_378_137.0, 1_000_000.0
        span_x, span_y = 0.02, 0.01
        _write_geotiff(terrain, values, origin_x=origin_x, origin_y=origin_y, span_x=span_x, span_y=span_y, crs=None)
        oracle = _rasterio_oracle(terrain)
        assert oracle["crs"] is None
        assert oracle["transform"][:3] == pytest.approx(
            (span_x / values.shape[1], 0.0, origin_x)
        )
        _load_terrain(viewer, terrain)
        # This is a vector-precision oracle, not a sky-composition test. Keep
        # SIDERA's opt-in sky from covering color-only overlays over clear depth.
        viewer.ok({"cmd": "set_terrain_pbr", "sky": {"enabled": False}})
        center = [origin_x + span_x / 2.0, 0.002, -origin_y + span_y / 2.0]
        _camera(viewer, target=center, phi=90.0, radius=0.03)
        separated = {
            "cmd": "add_vector_overlay",
            "id": 700,
            "name": "one-millimetre",
            "vertices": [
                [center[0] - 0.0005, center[1], center[2], 1.0, 0.0, 0.0, 1.0, 71],
                [center[0] + 0.0005, center[1], center[2], 0.0, 0.0, 1.0, 1.0, 72],
            ],
            "indices": [0, 1],
            "primitive": "points",
            "point_size": 12.0,
            "drape": False,
        }
        viewer.ok(separated)
        viewer.wait_frames(3)
        packed = viewer.stats()
        assert packed["last_vector_source_delta"][0] == pytest.approx(0.001, abs=1e-9)
        assert packed["last_vector_packed_delta"][0] == pytest.approx(0.001, abs=1e-7)
        precise = viewer.snapshot(tmp_path / "one-mm.png", width=640, height=400)
        red_count, red_centroid = _color_centroid(precise, 0)
        blue_count, blue_centroid = _color_centroid(precise, 2)
        assert red_count > 0 and blue_count > 0, (red_count, blue_count)
        assert red_centroid is not None and blue_centroid is not None
        separation_px = float(np.linalg.norm(red_centroid - blue_centroid))
        assert separation_px >= 1.0, separation_px

        viewer.ok({"cmd": "remove_vector_overlay", "id": 700})
        viewer.ok(
            {
                **separated,
                "id": 701,
                "name": "zero-millimetre-control",
                "vertices": [
                    [center[0], center[1], center[2], 1.0, 0.0, 0.0, 1.0, 73],
                    [center[0], center[1], center[2], 0.0, 0.0, 1.0, 1.0, 74],
                ],
            }
        )
        viewer.wait_frames(3)
        control = viewer.snapshot(tmp_path / "zero-mm.png", width=640, height=400)
        control_red, _ = _color_centroid(control, 0)
        control_blue, _ = _color_centroid(control, 2)
        assert min(control_red, control_blue) == 0, (control_red, control_blue)
        viewer.ok({"cmd": "remove_vector_overlay", "id": 701})
        viewer.ok(
            {
                "cmd": "add_vector_overlay",
                "id": 702,
                "name": "persistent-triangle-bvh",
                "vertices": [
                    [center[0] - 0.004, center[1] - 0.0005, center[2], 0.2, 0.8, 0.2, 0.5, 91],
                    [center[0] + 0.004, center[1] - 0.0005, center[2], 0.2, 0.8, 0.2, 0.5, 91],
                    [center[0], center[1] + 0.003, center[2], 0.2, 0.8, 0.2, 0.5, 91],
                ],
                "indices": [0, 1, 2],
                "primitive": "triangles",
                "drape": False,
                "opacity": 0.5,
            }
        )

        label_commands = [
            {"cmd": "add_label", "id": 801, "text": "point", "world_pos": center, "size": 18.0},
            {"cmd": "add_line_label", "id": 802, "text": "line", "polyline": [[center[0] - 0.004, center[1], center[2]], [center[0] + 0.004, center[1], center[2]]], "size": 16.0},
            {"cmd": "add_curved_label", "id": 803, "text": "curve", "polyline": [[center[0] - 0.004, center[1], center[2] - 0.002], center, [center[0] + 0.004, center[1], center[2] + 0.002]], "size": 16.0},
            {"cmd": "add_callout", "id": 804, "text": "callout", "anchor": center, "offset": [48.0, -32.0]},
        ]
        for command in label_commands:
            viewer.ok(command)
        viewer.wait_frames(3)
        automatic_labels = viewer.snapshot(tmp_path / "labels-automatic.png", width=640, height=400)
        viewer.ok({"cmd": "update_labels"})
        viewer.wait_frames(1)
        manual_labels = viewer.snapshot(tmp_path / "labels-manual.png", width=640, height=400)
        manual_parity = _metrics(automatic_labels, manual_labels)
        assert _parity_passes(manual_parity), manual_parity
        label_pick = _pick_center(viewer)
        assert label_pick["layer_name"] == "Labels"
        assert label_pick["world_pos"] == pytest.approx(center, abs=1e-9)

        label_resources = viewer.stats()
        assert label_resources["vector_source_bytes"] > 0
        assert label_resources["vector_render_cache_bytes"] > 0
        assert label_resources["vector_gpu_bytes"] > 0
        assert len(label_resources["vector_gpu_allocation_ids"]) == 2
        assert label_resources["vector_bvh_cpu_bytes"] > 0
        _camera(
            viewer,
            target=[center[0] + 1100.0, center[1], center[2]],
            phi=90.0,
            radius=0.03,
        )
        _camera(viewer, target=center, phi=90.0, radius=0.03)
        viewer.ok({"cmd": "update_labels"})
        viewer.wait_frames(2)
        rebased_labels = viewer.snapshot(tmp_path / "labels-after-rebase.png", width=640, height=400)
        label_rebase_parity = _metrics(manual_labels, rebased_labels)
        assert _parity_passes(label_rebase_parity), label_rebase_parity
        after_label_rebase = viewer.stats()
        assert after_label_rebase["camera_rebase_count"] >= label_resources["camera_rebase_count"] + 2
        assert (
            after_label_rebase["tracked_buffer_count"],
            after_label_rebase["tracked_texture_count"],
            after_label_rebase["tracked_total_bytes"],
        ) == (
            label_resources["tracked_buffer_count"],
            label_resources["tracked_texture_count"],
            label_resources["tracked_total_bytes"],
        )
        assert after_label_rebase["vector_gpu_allocation_ids"] == label_resources[
            "vector_gpu_allocation_ids"
        ]
        assert after_label_rebase["vector_gpu_bytes"] == label_resources["vector_gpu_bytes"]
        assert after_label_rebase["vector_bvh_cpu_bytes"] == label_resources[
            "vector_bvh_cpu_bytes"
        ]
        label_pick_after = _pick_center(viewer)
        assert label_pick_after["layer_name"] == "Labels"
        assert label_pick_after["world_pos"] == pytest.approx(center, abs=1e-9)

        baseline_state = {
            "cmd": "set_scene_review_state",
            "state": {
                "review_layers": [{"id": "accepted", "name": "Accepted", "labels": [{"kind": "point", "text": "valid", "world_pos": center}]}],
                "variants": [{"id": "baseline", "active_layer_ids": ["accepted"]}],
                "active_variant_id": "baseline",
            },
        }
        viewer.ok(baseline_state)
        baseline = viewer.wait_until(
            lambda: {
                "layers": viewer.ok({"cmd": "list_review_layers"})["review_layers"],
                "variants": viewer.ok({"cmd": "list_scene_variants"})["scene_variants"],
                "active": viewer.ok({"cmd": "get_active_scene_variant"})["active_scene_variant"],
            }
            if viewer.ok({"cmd": "get_active_scene_variant"})["active_scene_variant"] == "baseline"
            else None,
            timeout=45.0,
            description="baseline scene-review transaction",
        )
        rejected_state = {
            "cmd": "set_scene_review_state",
            "state": {
                "review_layers": [{"id": "poison", "labels": [{"kind": "point", "text": "too far", "world_pos": [center[0] + 2_000_000.0, center[1], center[2]]}]}],
                "variants": [{"id": "rejected", "active_layer_ids": ["poison"]}],
                "active_variant_id": "rejected",
            },
        }
        rejection = viewer.send(rejected_state)
        assert rejection.get("ok") is False, rejection
        assert "scene_review_transaction_failed" in rejection.get("error", ""), rejection
        viewer.wait_frames(5)
        after = {
            "layers": viewer.ok({"cmd": "list_review_layers"})["review_layers"],
            "variants": viewer.ok({"cmd": "list_scene_variants"})["scene_variants"],
            "active": viewer.ok({"cmd": "get_active_scene_variant"})["active_scene_variant"],
        }
        assert after == baseline
        _write_json("millimetre-transaction.json", {"backend": os.environ.get("WGPU_BACKEND"), "rasterio_oracle": oracle, "packed": packed, "red_pixels": red_count, "blue_pixels": blue_count, "separation_px": separation_px, "control_red_pixels": control_red, "control_blue_pixels": control_blue, "manual_label_parity": manual_parity, "label_rebase_parity": label_rebase_parity, "label_pick_before": label_pick, "label_pick_after": label_pick_after, "baseline": baseline, "rejection": rejection, "after_rejected": after})
    finally:
        viewer.close("millimetre-viewer.log")


@pytest.mark.parametrize(
    "failure_point",
    ("after_label_staging", "after_raster_composite", "after_vector_allocation"),
)
def test_m06_scene_review_nth_failure_preserves_live_frame(
    tmp_path: Path, failure_point: str
) -> None:
    viewer = ViewerProcess(
        extra_env={"FORGE3D_M06_SCENE_REVIEW_FAILPOINT": f"{failure_point}:2"}
    )
    try:
        terrain = tmp_path / f"transaction-{failure_point}.tif"
        _write_local_tiff(terrain, _heightfield(96, 64))
        _load_terrain(viewer, terrain)
        _camera(viewer, target=[48.0, 20.0, -32.0], phi=130.0, radius=180.0)
        overlay = tmp_path / f"transaction-{failure_point}.png"
        Image.new("RGBA", (8, 8), (80, 140, 220, 255)).save(overlay)
        _record_input(overlay)

        def state(identifier: str, text: str) -> dict[str, Any]:
            return {
                "cmd": "set_scene_review_state",
                "state": {
                    "review_layers": [
                        {
                            "id": identifier,
                            "name": identifier,
                            "labels": [
                                {
                                    "kind": "point",
                                    "text": text,
                                    "world_pos": [48.0, 30.0, -32.0],
                                }
                            ],
                            "raster_overlays": [
                                {
                                    "name": f"{identifier}-raster",
                                    "path": str(overlay),
                                    "extent": [0.1, 0.1, 0.9, 0.9],
                                }
                            ],
                            "vector_overlays": [
                                {
                                    "name": f"{identifier}-vector",
                                    "vertices": [
                                        [30.0, 25.0, -25.0, 1.0, 0.0, 0.0, 1.0, 101],
                                        [66.0, 25.0, -25.0, 0.0, 1.0, 0.0, 1.0, 102],
                                        [48.0, 25.0, -44.0, 0.0, 0.0, 1.0, 1.0, 103],
                                    ],
                                    "indices": [0, 1, 2],
                                    "primitive": "triangles",
                                }
                            ],
                        }
                    ],
                    "variants": [
                        {"id": identifier, "active_layer_ids": [identifier]}
                    ],
                    "active_variant_id": identifier,
                },
            }

        viewer.ok(state("accepted", "before"))
        before_registry = {
            "layers": viewer.ok({"cmd": "list_review_layers"})["review_layers"],
            "variants": viewer.ok({"cmd": "list_scene_variants"})["scene_variants"],
            "active": viewer.ok({"cmd": "get_active_scene_variant"})[
                "active_scene_variant"
            ],
        }
        before_image = viewer.snapshot(tmp_path / f"{failure_point}-before.png")
        before_stats = viewer.stats()

        rejected = viewer.send(state("rejected", "after"))
        assert rejected.get("ok") is False, rejected
        assert f"point={failure_point}" in rejected.get("error", ""), rejected
        viewer.wait_frames(3)
        after_registry = {
            "layers": viewer.ok({"cmd": "list_review_layers"})["review_layers"],
            "variants": viewer.ok({"cmd": "list_scene_variants"})["scene_variants"],
            "active": viewer.ok({"cmd": "get_active_scene_variant"})[
                "active_scene_variant"
            ],
        }
        after_image = viewer.snapshot(tmp_path / f"{failure_point}-after.png")
        after_stats = viewer.stats()

        assert after_registry == before_registry
        invariant_keys = (
            "camera_anchor_origin",
            "camera_rebase_count",
            "tracked_buffer_count",
            "tracked_texture_count",
            "tracked_total_bytes",
            "host_visible_bytes",
        )
        assert {key: after_stats[key] for key in invariant_keys} == {
            key: before_stats[key] for key in invariant_keys
        }
        parity = _metrics(before_image, after_image)
        assert parity["ssim"] >= 0.99 and parity["mae"] <= 0.01, parity
        _write_diff(
            f"scene-review-{failure_point}-rollback-diff.png",
            before_image,
            after_image,
        )
        _write_json(
            f"scene-review-{failure_point}-rollback.json",
            {
                "before_registry": before_registry,
                "after_registry": after_registry,
                "before_stats": before_stats,
                "after_stats": after_stats,
                "rejection": rejected,
                "image_metrics": parity,
            },
        )
    finally:
        viewer.close(f"scene-review-{failure_point}-viewer.log")
