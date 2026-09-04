"""Tests for PBR terrain viewer integration.

Verifies:
1. Legacy mode produces consistent output (regression)
2. PBR mode produces visually different output from legacy
3. PBR config parameters are honored via IPC
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Find project root
PROJECT_ROOT = Path(__file__).parent.parent

pytestmark = pytest.mark.interactive_viewer


def find_viewer_binary() -> Path:
    """Find the interactive_viewer binary."""
    override = os.environ.get("FORGE3D_VIEWER_BINARY")
    if override is not None:
        assert override.strip(), "FORGE3D_VIEWER_BINARY must not be empty"
        binary = Path(override)
        assert binary.is_file(), f"FORGE3D_VIEWER_BINARY does not exist: {binary}"
        return binary
    ext = ".exe" if os.name == "nt" else ""
    candidates = [
        PROJECT_ROOT / "target" / "release" / f"interactive_viewer{ext}",
        PROJECT_ROOT / "target" / "debug" / f"interactive_viewer{ext}",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("interactive_viewer binary not found - run: cargo build --release --bin interactive_viewer")


def find_test_dem() -> Path:
    """Find a test DEM file."""
    candidates = [
        PROJECT_ROOT / "assets" / "Gore_Range_Albers_1m.tif",
        PROJECT_ROOT / "assets" / "dem_rainier.tif",
        PROJECT_ROOT / "assets" / "tif" / "switzerland_dem.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("No test DEM found in assets/")


def find_test_hdri() -> Path:
    """Find an HDRI fixture for terrain PBR tests."""
    candidate = PROJECT_ROOT / "assets" / "hdri" / "brown_photostudio_02_4k.hdr"
    if candidate.exists():
        return candidate
    pytest.skip("No HDRI fixture found in assets/hdri/")


def send_ipc(sock: socket.socket, cmd: dict, timeout: float = 10.0) -> dict:
    """Send an IPC command and receive response."""
    sock.settimeout(timeout)
    msg = json.dumps(cmd) + "\n"
    sock.sendall(msg.encode())
    
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    
    response_str = data.decode().strip()
    if response_str:
        return json.loads(response_str)
    return {"ok": False, "error": "Empty response"}


def _drain_viewer_stdout(stream) -> None:
    """Keep the viewer pipe flowing while retaining post-READY crash evidence."""
    for line in iter(stream.readline, ""):
        print(f"[forge3d-viewer] {line}", end="", flush=True)


def start_viewer_with_ipc(binary: Path, width: int = 640, height: int = 480) -> tuple:
    """Start viewer with IPC and return (process, port)."""
    cmd = [str(binary), "--ipc-port", "0", "--size", f"{width}x{height}"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    
    ready_pattern = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
    port = None
    start = time.time()
    
    while time.time() - start < 30.0:
        if process.poll() is not None:
            raise RuntimeError("Viewer exited unexpectedly")
        line = process.stdout.readline()
        if line:
            match = ready_pattern.search(line)
            if match:
                port = int(match.group(1))
                break
    
    if port is None:
        process.terminate()
        raise RuntimeError("Timeout waiting for viewer READY signal")

    threading.Thread(
        target=_drain_viewer_stdout, args=(process.stdout,), daemon=True
    ).start()
    
    return process, port


def image_hash(path: Path) -> str:
    """Compute hash of image file."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def wait_for_snapshot(path: Path, timeout_s: float = 10.0) -> bool:
    """Wait for an async viewer snapshot to exist and stabilize on disk."""
    deadline = time.time() + timeout_s
    last_size = -1
    stable_reads = 0
    while time.time() < deadline:
        if path.exists():
            try:
                size = path.stat().st_size
            except OSError:
                time.sleep(0.2)
                continue
            if size > 1000:
                stable_reads = stable_reads + 1 if size == last_size else 0
                last_size = size
                if stable_reads >= 2:
                    return True
        time.sleep(0.2)
    return False


def wait_for_frame_advance(sock: socket.socket, frames: int = 2, timeout_s: float = 15.0) -> None:
    """Wait until queued IPC state has crossed at least ``frames`` renders.

    IPC acknowledgements confirm enqueueing, not event-loop application.  A
    snapshot sent immediately after a PBR/HDR command can therefore capture the
    prior state and make visual comparisons intermittently identical.
    """
    start_response = send_ipc(sock, {"cmd": "get_stats"})
    assert start_response.get("ok"), start_response
    start = int(start_response["stats"]["frame_count"])
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = send_ipc(sock, {"cmd": "get_stats"})
        assert response.get("ok"), response
        if int(response["stats"]["frame_count"]) >= start + frames:
            return
        time.sleep(0.05)
    raise AssertionError(f"viewer did not advance {frames} frames after IPC command")


def write_solid_overlay(path: Path, rgba: tuple[int, int, int, int]) -> None:
    """Write a flat RGBA overlay image for preserve_colors testing."""
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to write overlay fixtures")
    Image.new("RGBA", (64, 64), rgba).save(path)


def write_ridge_heightmap_tiff(path: Path, size: int = 256) -> None:
    """Write a synthetic ridge TIFF that produces strong lee-side shadows."""
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to write terrain fixtures")
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ridge = np.exp(-xx**2 * 10.0) * 1.0
    ridge2 = np.exp(-(xx - 0.35) ** 2 * 30.0) * 0.65
    ridge3 = np.exp(-(xx + 0.45) ** 2 * 24.0) * 0.55
    ramp = np.clip(0.2 + 0.3 * yy, 0.0, 1.0)
    height = ridge + ridge2 + ridge3 + ramp
    height = (height - height.min()) / max(float(height.max() - height.min()), 1e-6)
    Image.fromarray(np.round(height * 65535.0).astype(np.uint16)).save(path)


def write_asymmetric_test_hdr(path: Path, width: int = 16, height: int = 8) -> None:
    """Write an asymmetric Radiance HDR fixture for rotation tests."""
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            vertical_scale = 1.0 if y < height // 2 else 0.4
            for x in range(width):
                if x < max(width // 8, 1):
                    base = (255, 255, 255)
                elif x < width // 2:
                    base = (250, 48, 24)
                else:
                    base = (36, 88, 250)
                rgb = [int(channel * vertical_scale) for channel in base]
                handle.write(bytes([rgb[0], rgb[1], rgb[2], 128]))


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float RGB in [0, 1]."""
    if not HAS_PIL:
        raise RuntimeError("Pillow is required to inspect snapshots")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB."""
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def mean_normalized_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute mean RGB normalized by total intensity."""
    mean = np.mean(rgb[mask], axis=0)
    return mean / max(float(np.sum(mean)), 1e-6)


def terrain_mask(rgb: np.ndarray) -> np.ndarray:
    """Mask out the white background around the terrain snapshot."""
    return np.any(rgb < 0.97, axis=2) & (np.sum(rgb, axis=2) > 0.05)


@pytest.fixture
def viewer_context():
    """Fixture that starts viewer and provides IPC connection."""
    binary = find_viewer_binary()
    terrain_tmp = tempfile.TemporaryDirectory()
    dem = Path(terrain_tmp.name) / "local-pbr-ridge.tif"
    # Keep this suite independent of geospatial unit metadata.  The former
    # fallback fixture was a WGS84-degree GeoTIFF with metre elevations; M-06
    # correctly preserves that 4.5-degree physical span, which is not a valid
    # local PBR lighting test domain.
    write_ridge_heightmap_tiff(dem)
    
    process, port = start_viewer_with_ipc(binary)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)
    
    # Load terrain
    resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(dem)})
    assert resp.get("ok"), f"Failed to load terrain: {resp.get('error')}"
    
    # Set consistent camera for reproducible output
    send_ipc(sock, {
        "cmd": "set_terrain",
        "phi": 45.0, "theta": 45.0, "radius": 220.0, "fov": 55.0,
        "zscale": 1.0, "sun_azimuth": 135.0, "sun_elevation": 45.0,
        "sun_intensity": 1.0, "ambient": 0.3
    })
    wait_for_frame_advance(sock)
    
    yield {
        "process": process,
        "sock": sock,
        "port": port,
        "dem": dem,
    }
    
    # Cleanup
    try:
        send_ipc(sock, {"cmd": "close"})
    except Exception:
        pass
    sock.close()
    process.terminate()
    process.wait(timeout=5)
    terrain_tmp.cleanup()


class TestTerrainViewerPbr:
    """Test suite for PBR terrain viewer."""

    def test_legacy_mode_renders(self, viewer_context):
        """Test that legacy mode renders without PBR enabled."""
        sock = viewer_context["sock"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "legacy.png"
            
            # Wait for render to settle
            time.sleep(0.3)
            
            # Take snapshot in legacy mode
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(snap_path),
                "width": 640,
                "height": 480,
            })
            
            # Give time for snapshot to be written
            time.sleep(0.5)
            
            assert snap_path.exists(), "Legacy snapshot not created"
            assert snap_path.stat().st_size > 1000, "Legacy snapshot too small"
            print(f"[test] Legacy snapshot: {snap_path.stat().st_size} bytes")

    def test_pbr_mode_enables(self, viewer_context):
        """Test that PBR mode can be enabled via IPC."""
        sock = viewer_context["sock"]
        
        # Enable PBR mode
        resp = send_ipc(sock, {
            "cmd": "set_terrain_pbr",
            "enabled": True,
            "exposure": 1.5,
            "normal_strength": 1.2,
        })
        
        assert resp.get("ok"), f"PBR enable failed: {resp.get('error')}"
        print("[test] PBR mode enabled successfully")

    def test_pbr_produces_different_output(self, viewer_context):
        """Test that PBR mode produces visually different output than legacy."""
        sock = viewer_context["sock"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "legacy.png"
            pbr_path = Path(tmpdir) / "pbr.png"
            
            # Capture legacy mode
            time.sleep(0.3)
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(legacy_path),
                "width": 640,
                "height": 480,
            })
            assert wait_for_snapshot(legacy_path), "Legacy snapshot not created"
            
            # Enable PBR and capture
            send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.0,
            })
            time.sleep(0.3)
            
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(pbr_path),
                "width": 640,
                "height": 480,
            })
            assert wait_for_snapshot(pbr_path), "PBR snapshot not created"

            # Both should exist
            assert legacy_path.exists(), "Legacy snapshot not created"
            assert pbr_path.exists(), "PBR snapshot not created"
            
            # Compute hashes
            legacy_hash = image_hash(legacy_path)
            pbr_hash = image_hash(pbr_path)
            
            print(f"[test] Legacy hash: {legacy_hash}")
            print(f"[test] PBR hash:    {pbr_hash}")
            
            assert legacy_hash != pbr_hash, (
                "PBR output is identical to legacy output; the PBR pipeline "
                "did not produce an observable result"
            )

    def test_pbr_exposure_affects_output(self, viewer_context):
        """Test that changing exposure produces different output."""
        sock = viewer_context["sock"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp1_path = Path(tmpdir) / "exp1.png"
            exp2_path = Path(tmpdir) / "exp2.png"
            
            # Enable PBR with exposure 0.5
            send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 0.5,
            })
            time.sleep(0.3)
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(exp1_path),
                "width": 640,
                "height": 480,
            })
            time.sleep(0.5)
            
            # Change to exposure 2.0
            send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "exposure": 2.0,
            })
            time.sleep(0.3)
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(exp2_path),
                "width": 640,
                "height": 480,
            })
            time.sleep(0.5)
            
            exp1_hash = image_hash(exp1_path)
            exp2_hash = image_hash(exp2_path)
            
            print(f"[test] Exposure 0.5 hash: {exp1_hash}")
            print(f"[test] Exposure 2.0 hash: {exp2_hash}")
            
            assert exp1_hash != exp2_hash, (
                "PBR exposure 0.5 and 2.0 produced identical output"
            )

    @pytest.mark.skipif(not HAS_PIL, reason="Requires Pillow")
    def test_hdr_path_changes_indirect_lighting(self, viewer_context):
        """Loading an HDRI should change terrain shading relative to the analytic fallback."""
        sock = viewer_context["sock"]
        hdri_path = find_test_hdri()

        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "fallback.png"
            hdri_snapshot_path = Path(tmpdir) / "hdri.png"

            resp = send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.0,
                "ibl_intensity": 1.0,
                "normal_strength": 1.2,
                "height_ao": {"enabled": False},
                "sun_visibility": {"enabled": False},
            })
            assert resp.get("ok"), f"Fallback PBR config failed: {resp.get('error')}"
            wait_for_frame_advance(sock)

            assert send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(fallback_path),
                "width": 640,
                "height": 480,
            }, timeout=60.0).get("ok")
            assert wait_for_snapshot(fallback_path, timeout_s=60.0), "Fallback snapshot not created"

            resp = send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "hdr_path": str(hdri_path),
                "ibl_intensity": 1.0,
            })
            assert resp.get("ok"), f"HDRI PBR config failed: {resp.get('error')}"
            wait_for_frame_advance(sock)

            assert send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(hdri_snapshot_path),
                "width": 640,
                "height": 480,
            }, timeout=60.0).get("ok")
            assert wait_for_snapshot(hdri_snapshot_path, timeout_s=60.0), "HDRI snapshot not created"

            fallback_rgb = load_rgb(fallback_path)
            hdri_rgb = load_rgb(hdri_snapshot_path)
            mask = terrain_mask(fallback_rgb) | terrain_mask(hdri_rgb)
            assert np.any(mask), "Terrain mask was empty for HDRI comparison"

            mean_abs_delta = float(np.mean(np.abs(hdri_rgb[mask] - fallback_rgb[mask])))
            mean_luma_delta = float(
                abs(np.mean(luminance(hdri_rgb)[mask]) - np.mean(luminance(fallback_rgb)[mask]))
            )

            assert image_hash(fallback_path) != image_hash(hdri_snapshot_path)
            assert mean_abs_delta > 0.005, f"HDRI delta too small: {mean_abs_delta:.4f}"
            assert mean_luma_delta > 0.005, f"HDRI luminance delta too small: {mean_luma_delta:.4f}"

    @pytest.mark.skipif(not HAS_PIL, reason="Requires Pillow")
    def test_hdr_rotation_changes_environment_sampling(self):
        """Rotating an asymmetric HDRI should change terrain shading."""
        binary = find_viewer_binary()
        process, port = start_viewer_with_ipc(binary)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        sock.settimeout(30.0)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                terrain_path = Path(tmpdir) / "ridge.tif"
                hdr_path = Path(tmpdir) / "asymmetric.hdr"
                rotation_0_path = Path(tmpdir) / "rot_0.png"
                rotation_90_path = Path(tmpdir) / "rot_90.png"

                write_ridge_heightmap_tiff(terrain_path)
                write_asymmetric_test_hdr(hdr_path)

                resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(terrain_path)})
                assert resp.get("ok"), f"Terrain load failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "set_terrain",
                    "phi": 72.0,
                    "theta": 28.0,
                    "radius": 220.0,
                    "fov": 35.0,
                    "zscale": 1.8,
                    "sun_intensity": 0.0,
                    "ambient": 1.0,
                    "shadow": 0.0,
                    "background": [1.0, 1.0, 1.0],
                })
                assert resp.get("ok"), f"Terrain lighting override failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "set_terrain_pbr",
                    "enabled": True,
                    "hdr_path": str(hdr_path),
                    "ibl_intensity": 4.0,
                    "hdr_rotate_deg": 0.0,
                    "exposure": 1.0,
                    "normal_strength": 4.0,
                    "height_ao": {"enabled": False},
                    "sun_visibility": {"enabled": False},
                })
                assert resp.get("ok"), f"Rotation-0 PBR config failed: {resp.get('error')}"
                wait_for_frame_advance(sock)

                assert send_ipc(sock, {
                    "cmd": "snapshot",
                    "path": str(rotation_0_path),
                    "width": 640,
                    "height": 480,
                }, timeout=60.0).get("ok")
                assert wait_for_snapshot(rotation_0_path, timeout_s=60.0), "Rotation-0 snapshot not created"

                resp = send_ipc(sock, {
                    "cmd": "set_terrain_pbr",
                    "hdr_rotate_deg": 90.0,
                })
                assert resp.get("ok"), f"Rotation-90 PBR config failed: {resp.get('error')}"
                wait_for_frame_advance(sock)

                assert send_ipc(sock, {
                    "cmd": "snapshot",
                    "path": str(rotation_90_path),
                    "width": 640,
                    "height": 480,
                }, timeout=60.0).get("ok")
                assert wait_for_snapshot(rotation_90_path, timeout_s=60.0), "Rotation-90 snapshot not created"

                rotation_0_rgb = load_rgb(rotation_0_path)
                rotation_90_rgb = load_rgb(rotation_90_path)
                mask = terrain_mask(rotation_0_rgb) | terrain_mask(rotation_90_rgb)
                assert np.any(mask), "Terrain mask was empty for HDR rotation comparison"

                mean_abs_delta = float(np.mean(np.abs(rotation_90_rgb[mask] - rotation_0_rgb[mask])))

                assert image_hash(rotation_0_path) != image_hash(rotation_90_path)
                assert mean_abs_delta > 0.005, f"HDR rotation delta too small: {mean_abs_delta:.4f}"
        finally:
            try:
                send_ipc(sock, {"cmd": "close"})
            except Exception:
                pass
            sock.close()
            process.terminate()
            process.wait(timeout=5)

    def test_dof_enabled_does_not_crash(self, viewer_context):
        """Regression test: enabling DoF via IPC should not crash viewer."""
        sock = viewer_context["sock"]

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "dof.png"

            resp = send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.0,
                "dof": {
                    "enabled": True,
                    "f_stop": 2.8,
                    "focus_distance": 500.0,
                    "focal_length": 50.0,
                    "quality": "medium",
                },
            })
            assert resp.get("ok"), f"DoF enable failed: {resp.get('error')}"

            time.sleep(0.5)

            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(snap_path),
                "width": 640,
                "height": 480,
            })
            assert wait_for_snapshot(snap_path), "DoF snapshot not created (viewer crash?)"

            assert snap_path.exists(), "DoF snapshot not created (viewer crash?)"
            assert snap_path.stat().st_size > 1000, "DoF snapshot too small"
            print(f"[test] DoF snapshot: {snap_path.stat().st_size} bytes")

    def test_pbr_can_be_disabled(self, viewer_context):
        """Test that PBR mode can be disabled to return to legacy."""
        sock = viewer_context["sock"]
        
        # Enable PBR
        resp = send_ipc(sock, {
            "cmd": "set_terrain_pbr",
            "enabled": True,
        })
        assert resp.get("ok")
        
        # Disable PBR
        resp = send_ipc(sock, {
            "cmd": "set_terrain_pbr",
            "enabled": False,
        })
        assert resp.get("ok"), f"PBR disable failed: {resp.get('error')}"
        print("[test] PBR mode disabled successfully")

    def test_height_ao_compute_pipeline_metal_compat(self, viewer_context):
        """Regression test: height_ao compute pipeline with R32Float texture.
        
        Tests fix for Metal/R32Float compatibility issue where filterable: true
        in bind group layout caused crash on macOS because R32Float doesn't
        support filtering. The fix changed to filterable: false and NonFiltering
        sampler since the shader uses textureLoad (not textureSample).
        """
        sock = viewer_context["sock"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "height_ao.png"
            
            # Enable PBR with height_ao - this would crash before the fix
            resp = send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.0,
                "height_ao": {
                    "enabled": True,
                    "directions": 6,
                    "steps": 16,
                    "max_distance": 200.0,
                    "strength": 1.0,
                    "resolution_scale": 0.5,
                },
            })
            assert resp.get("ok"), f"Height AO enable failed: {resp.get('error')}"
            
            # Wait for compute pass to run
            time.sleep(0.5)
            
            # Take snapshot - crash would occur here during render
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(snap_path),
                "width": 640,
                "height": 480,
            })
            assert wait_for_snapshot(snap_path), "Height AO snapshot not created (compute pipeline crash?)"

            assert snap_path.exists(), "Height AO snapshot not created (compute pipeline crash?)"
            assert snap_path.stat().st_size > 1000, "Height AO snapshot too small"
            print(f"[test] Height AO snapshot: {snap_path.stat().st_size} bytes - Metal/R32Float compat OK")

    def test_sun_visibility_compute_pipeline_metal_compat(self, viewer_context):
        """Regression test: sun_visibility compute pipeline with R32Float texture.
        
        Same Metal/R32Float compatibility fix as height_ao.
        """
        sock = viewer_context["sock"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "sun_vis.png"
            
            # Enable PBR with sun_visibility - this would crash before the fix
            resp = send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.0,
                "sun_visibility": {
                    "enabled": True,
                    "mode": "soft",
                    "samples": 4,
                    "steps": 24,
                    "max_distance": 400.0,
                    "softness": 1.0,
                    "bias": 0.01,
                    "resolution_scale": 0.5,
                },
            })
            assert resp.get("ok"), f"Sun visibility enable failed: {resp.get('error')}"
            
            time.sleep(0.5)
            
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(snap_path),
                "width": 640,
                "height": 480,
            })
            assert wait_for_snapshot(snap_path), "Sun vis snapshot not created (compute pipeline crash?)"

            assert snap_path.exists(), "Sun vis snapshot not created (compute pipeline crash?)"
            assert snap_path.stat().st_size > 1000, "Sun vis snapshot too small"
            print(f"[test] Sun visibility snapshot: {snap_path.stat().st_size} bytes - Metal/R32Float compat OK")

    def test_both_heightfield_effects_combined(self, viewer_context):
        """Regression test: both height_ao and sun_visibility enabled together."""
        sock = viewer_context["sock"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "combined.png"
            
            # Enable both effects - stress test for compute pipelines
            resp = send_ipc(sock, {
                "cmd": "set_terrain_pbr",
                "enabled": True,
                "exposure": 1.2,
                "height_ao": {
                    "enabled": True,
                    "strength": 1.0,
                },
                "sun_visibility": {
                    "enabled": True,
                    "mode": "soft",
                },
            })
            assert resp.get("ok"), f"Combined effects enable failed: {resp.get('error')}"
            
            time.sleep(0.5)
            
            send_ipc(sock, {
                "cmd": "snapshot",
                "path": str(snap_path),
                "width": 640,
                "height": 480,
            })
            assert wait_for_snapshot(snap_path), "Combined effects snapshot not created"

            assert snap_path.exists(), "Combined effects snapshot not created"
            assert snap_path.stat().st_size > 1000, "Combined effects snapshot too small"
            print(f"[test] Combined effects snapshot: {snap_path.stat().st_size} bytes")

    @pytest.mark.skipif(not HAS_PIL, reason="Requires Pillow")
    def test_preserve_colors_toggle_restores_regular_output(self):
        """Toggling preserve-colors should not perturb normal overlay output."""
        binary = find_viewer_binary()
        process, port = start_viewer_with_ipc(binary)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        sock.settimeout(30.0)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                terrain_path = Path(tmpdir) / "ridge.tif"
                overlay_path = Path(tmpdir) / "overlay.png"
                regular_path = Path(tmpdir) / "regular.png"
                preserve_path = Path(tmpdir) / "preserve.png"
                regular_again_path = Path(tmpdir) / "regular_again.png"
                source_rgba = (214, 82, 38, 255)

                write_ridge_heightmap_tiff(terrain_path)
                write_solid_overlay(overlay_path, source_rgba)

                resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(terrain_path)})
                assert resp.get("ok"), f"Terrain load failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "set_terrain",
                    "phi": 90.0,
                    "theta": 30.0,
                    "radius": 220.0,
                    "fov": 35.0,
                    "zscale": 1.4,
                    "sun_azimuth": 315.0,
                    "sun_elevation": 10.0,
                    "sun_intensity": 4.0,
                    "ambient": 0.12,
                    "shadow": 1.0,
                    "background": [1.0, 1.0, 1.0],
                })
                assert resp.get("ok"), f"Terrain config failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "set_terrain_pbr",
                    "enabled": True,
                    "shadow_technique": "pcss",
                    "shadow_map_res": 4096,
                    "exposure": 1.1,
                    "msaa": 4,
                    "ibl_intensity": 0.15,
                    "normal_strength": 3.8,
                    "height_ao": {
                        "enabled": False,
                    },
                    "sun_visibility": {
                        "enabled": True,
                        "mode": "hard",
                        "samples": 1,
                        "steps": 64,
                        "max_distance": 400.0,
                        "softness": 0.0,
                        "bias": 0.005,
                        "resolution_scale": 1.0,
                    },
                })
                assert resp.get("ok"), f"PBR config failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "load_overlay",
                    "name": "flat",
                    "path": str(overlay_path),
                    "extent": [0.0, 0.0, 1.0, 1.0],
                    "opacity": 1.0,
                    "z_order": 0,
                })
                assert resp.get("ok"), f"Overlay load failed: {resp.get('error')}"
                assert send_ipc(sock, {"cmd": "set_overlays_enabled", "enabled": True}).get("ok")
                assert send_ipc(sock, {"cmd": "set_overlay_solid", "solid": True}).get("ok")
                assert send_ipc(sock, {"cmd": "set_overlay_preserve_colors", "preserve_colors": False}).get("ok")

                wait_for_frame_advance(sock)
                assert send_ipc(sock, {
                    "cmd": "snapshot",
                    "path": str(regular_path),
                    "width": 640,
                    "height": 480,
                }, timeout=60.0).get("ok")
                assert wait_for_snapshot(regular_path, timeout_s=60.0), "regular snapshot not written"

                assert send_ipc(sock, {
                    "cmd": "set_overlay_preserve_colors",
                    "preserve_colors": True,
                }).get("ok")
                wait_for_frame_advance(sock)
                assert send_ipc(sock, {
                    "cmd": "snapshot",
                    "path": str(preserve_path),
                    "width": 640,
                    "height": 480,
                }, timeout=60.0).get("ok")
                assert wait_for_snapshot(preserve_path, timeout_s=60.0), "preserve_colors snapshot not written"

                assert send_ipc(sock, {
                    "cmd": "set_overlay_preserve_colors",
                    "preserve_colors": False,
                }).get("ok")
                wait_for_frame_advance(sock)
                assert send_ipc(sock, {
                    "cmd": "snapshot",
                    "path": str(regular_again_path),
                    "width": 640,
                    "height": 480,
                }, timeout=60.0).get("ok")
                assert wait_for_snapshot(regular_again_path, timeout_s=60.0), "regular-again snapshot not written"

                regular_hash = image_hash(regular_path)
                preserve_hash = image_hash(preserve_path)
                regular_again_hash = image_hash(regular_again_path)

                assert regular_hash == regular_again_hash, "preserve_colors toggle should not alter normal overlay output"
                assert preserve_hash != regular_hash, "preserve_colors should produce a distinct shaded result"
        finally:
            try:
                send_ipc(sock, {"cmd": "close"})
            except Exception:
                pass
            sock.close()
            process.terminate()
            process.wait(timeout=5)

    @pytest.mark.skipif(not HAS_PIL, reason="Requires Pillow")
    @pytest.mark.parametrize(
        ("source_rgba", "min_lit_mean"),
        [
            ((214, 82, 38, 255), 0.0),
            ((57, 125, 73, 255), 0.35),
        ],
    )
    def test_preserve_colors_ridge_regression(self, source_rgba, min_lit_mean):
        """Preserve-colors mode should preserve hue without collapsing to a dark mask."""
        binary = find_viewer_binary()
        process, port = start_viewer_with_ipc(binary)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        sock.settimeout(30.0)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                terrain_path = Path(tmpdir) / "ridge.tif"
                overlay_path = Path(tmpdir) / "overlay.png"
                preserve_path = Path(tmpdir) / "preserve.png"
                source_rgb = np.asarray(source_rgba[:3], dtype=np.float32) / 255.0
                source_norm = source_rgb / np.sum(source_rgb)

                write_ridge_heightmap_tiff(terrain_path)
                write_solid_overlay(overlay_path, source_rgba)

                resp = send_ipc(sock, {"cmd": "load_terrain", "path": str(terrain_path)})
                assert resp.get("ok"), f"Terrain load failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "set_terrain",
                    "phi": 90.0,
                    "theta": 30.0,
                    "radius": 220.0,
                    "fov": 35.0,
                    "zscale": 1.4,
                    "sun_azimuth": 315.0,
                    "sun_elevation": 10.0,
                    "sun_intensity": 4.0,
                    "ambient": 0.12,
                    "shadow": 1.0,
                    "background": [1.0, 1.0, 1.0],
                })
                assert resp.get("ok"), f"Terrain config failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "set_terrain_pbr",
                    "enabled": True,
                    "shadow_technique": "pcss",
                    "shadow_map_res": 4096,
                    "exposure": 1.1,
                    "msaa": 4,
                    "ibl_intensity": 0.15,
                    "normal_strength": 3.8,
                    "sky": {"enabled": False},
                    "height_ao": {"enabled": False},
                    "sun_visibility": {
                        "enabled": True,
                        "mode": "hard",
                        "samples": 1,
                        "steps": 64,
                        "max_distance": 400.0,
                        "softness": 0.0,
                        "bias": 0.005,
                        "resolution_scale": 1.0,
                    },
                })
                assert resp.get("ok"), f"PBR config failed: {resp.get('error')}"

                resp = send_ipc(sock, {
                    "cmd": "load_overlay",
                    "name": "flat",
                    "path": str(overlay_path),
                    "extent": [0.0, 0.0, 1.0, 1.0],
                    "opacity": 1.0,
                    "z_order": 0,
                })
                assert resp.get("ok"), f"Overlay load failed: {resp.get('error')}"
                assert send_ipc(sock, {"cmd": "set_overlays_enabled", "enabled": True}).get("ok")
                assert send_ipc(sock, {"cmd": "set_overlay_solid", "solid": True}).get("ok")
                assert send_ipc(sock, {"cmd": "set_overlay_preserve_colors", "preserve_colors": True}).get("ok")

                time.sleep(2.0)
                assert send_ipc(sock, {
                    "cmd": "snapshot",
                    "path": str(preserve_path),
                    "width": 640,
                    "height": 480,
                }, timeout=60.0).get("ok")
                assert wait_for_snapshot(preserve_path), "preserve_colors snapshot not written"

                preserve_rgb = load_rgb(preserve_path)
                mask = terrain_mask(preserve_rgb)
                assert np.count_nonzero(mask) > 1000, "terrain mask should cover a meaningful ROI"

                lum = luminance(preserve_rgb)
                roi_lum = lum[mask]
                shadow_threshold = float(np.quantile(roi_lum, 0.10))
                lit_threshold = float(np.quantile(roi_lum, 0.90))
                shadow_mask = mask & (lum <= shadow_threshold)
                lit_mask = mask & (lum >= lit_threshold)

                shadow_mean = float(np.mean(lum[shadow_mask]))
                lit_mean = float(np.mean(lum[lit_mask]))
                shadow_ratio = shadow_mean / max(lit_mean, 1e-6)

                lit_norm = mean_normalized_rgb(preserve_rgb, lit_mask)
                shadow_norm = mean_normalized_rgb(preserve_rgb, shadow_mask)
                lit_error = float(np.max(np.abs(lit_norm - source_norm)))
                shadow_error = float(np.max(np.abs(shadow_norm - source_norm)))

                assert shadow_ratio <= 0.35, f"preserve_colors shadows are too bright: ratio={shadow_ratio:.3f}"
                if min_lit_mean > 0.0:
                    assert lit_mean >= min_lit_mean, (
                        f"preserve_colors lit regions are too dark: mean={lit_mean:.3f}"
                    )
                assert lit_error <= 0.10, f"lit overlay hue drifted too far from source: err={lit_error:.3f}"
                assert shadow_error <= 0.12, f"shadow overlay hue drifted too far from source: err={shadow_error:.3f}"
        finally:
            try:
                send_ipc(sock, {"cmd": "close"})
            except Exception:
                pass
            sock.close()
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
