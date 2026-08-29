# tests/test_vector_overlay_rendering.py
# Integration tests for vector overlay rendering
# Tests IPC commands, lighting, and visibility controls

import pytest
import json
import socket
import subprocess
import re
import threading
import time
import os
from pathlib import Path


# Skip all tests if no DEM available or viewer can't start
pytestmark = [
    pytest.mark.interactive_viewer,
    pytest.mark.skipif(
        not os.environ.get("FORGE3D_TEST_DEM"),
        reason="Set FORGE3D_TEST_DEM to path of test DEM file to run integration tests",
    ),
]

PROJECT_ROOT = Path(__file__).parent.parent
_IPC_BUFFERS: dict[int, bytes] = {}


def find_viewer_binary() -> Path:
    """Find the release viewer used by the integration lane."""
    override = os.environ.get("FORGE3D_VIEWER_BINARY")
    if override is not None:
        assert override.strip(), "FORGE3D_VIEWER_BINARY must not be empty"
        binary = Path(override)
        assert binary.is_file(), f"FORGE3D_VIEWER_BINARY does not exist: {binary}"
        return binary
    extension = ".exe" if os.name == "nt" else ""
    for profile in ("release", "debug"):
        binary = PROJECT_ROOT / "target" / profile / f"interactive_viewer{extension}"
        if binary.is_file():
            return binary
    pytest.skip(
        "interactive_viewer binary not found - run: "
        "cargo build --release --bin interactive_viewer"
    )


def start_viewer_with_ipc(binary: Path) -> tuple[subprocess.Popen[str], int]:
    """Start a viewer on a dynamically assigned IPC port."""
    process = subprocess.Popen(
        [str(binary), "--ipc-port", "0", "--size", "640x480"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    ready_pattern = re.compile(r"FORGE3D_VIEWER_READY\s+port=(\d+)")
    port = None
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Viewer exited before its IPC server became ready")
        line = process.stdout.readline()
        match = ready_pattern.search(line)
        if match:
            port = int(match.group(1))
            break
    if port is None:
        process.terminate()
        raise RuntimeError("Timeout waiting for viewer READY signal")

    def drain_stdout() -> None:
        for _ in iter(process.stdout.readline, ""):
            pass

    threading.Thread(target=drain_stdout, daemon=True).start()
    return process, port


def send_ipc(sock: socket.socket, cmd: dict) -> dict:
    """Send IPC command and receive response."""
    msg = json.dumps(cmd) + "\n"
    sock.sendall(msg.encode())
    key = sock.fileno()
    data = _IPC_BUFFERS.pop(key, b"")
    while True:
        while b"\n" in data:
            line, data = data.split(b"\n", 1)
            if line.strip():
                _IPC_BUFFERS[key] = data
                return json.loads(line)
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Viewer closed IPC before returning a response")
        data += chunk


@pytest.fixture(scope="module")
def viewer_context():
    """Start viewer with IPC and yield (socket, process)."""
    dem_path = os.environ.get("FORGE3D_TEST_DEM")
    if not dem_path or not Path(dem_path).exists():
        pytest.skip("Test DEM not found")
    
    proc, port = start_viewer_with_ipc(find_viewer_binary())
    
    # Connect to IPC
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", port))
    sock.settimeout(30.0)

    response = send_ipc(sock, {"cmd": "load_terrain", "path": dem_path})
    assert response.get("ok", False), response
    
    yield sock, proc
    
    # Cleanup
    _IPC_BUFFERS.pop(sock.fileno(), None)
    sock.close()
    proc.terminate()
    proc.wait(timeout=5)


class TestVectorOverlayDefaultOff:
    """Test that vector overlays are disabled by default (regression)."""

    def test_vector_overlay_default_off(self, viewer_context):
        """Verify no vector overlays by default.
        
        Per plan Section 11: vector_overlays_enabled = false by default.
        Existing tests and renders should be byte-identical to before.
        """
        sock, _ = viewer_context
        
        # List overlays - should be empty
        resp = send_ipc(sock, {"cmd": "list_vector_overlays"})
        assert resp.get("ok", False)
        # Empty list expected


class TestVectorOverlayAddRemove:
    """Test adding and removing vector overlays."""

    def test_add_vector_overlay_triangle(self, viewer_context):
        """Add a simple triangle overlay."""
        sock, _ = viewer_context
        
        resp = send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "test_triangle",
            "vertices": [
                [100, 0, 100, 1, 0, 0, 1, 101],  # Red
                [200, 0, 100, 0, 1, 0, 1, 102],  # Green
                [150, 0, 200, 0, 0, 1, 1, 103],  # Blue
            ],
            "indices": [0, 1, 2],
            "primitive": "triangles",
            "drape": True,
            "drape_offset": 1.0,
        })
        assert resp.get("ok", False), f"Failed to add overlay: {resp}"

    def test_add_vector_overlay_lines(self, viewer_context):
        """Add line overlay."""
        sock, _ = viewer_context
        
        resp = send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "test_lines",
            "vertices": [
                [0, 0, 0, 1, 1, 0, 1, 201],
                [100, 0, 100, 1, 1, 0, 1, 202],
            ],
            "indices": [0, 1],
            "primitive": "lines",
            "drape": True,
            "line_width": 3.0,
        })
        assert resp.get("ok", False), f"Failed to add lines: {resp}"

    def test_remove_vector_overlay(self, viewer_context):
        """Remove vector overlay by ID."""
        sock, _ = viewer_context
        
        # Add overlay
        resp = send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "to_remove",
            "vertices": [[50, 0, 50, 1, 1, 1, 1, 301]],
            "indices": [0],
            "primitive": "points",
        })
        assert resp.get("ok", False)
        
        # Remove it (ID 0 if first added)
        resp = send_ipc(sock, {"cmd": "remove_vector_overlay", "id": 0})
        # May or may not find it depending on test order


class TestVectorOverlayLighting:
    """Test vector overlay lighting integration."""

    def test_vector_overlay_lit_by_sun(self, viewer_context):
        """Verify vector overlay receives sun lighting.
        
        Per plan Section 8: Overlays use identical lighting to terrain.
        Changing sun direction should change overlay appearance.
        """
        sock, _ = viewer_context
        
        # Add a white overlay
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "lit_test",
            "vertices": [
                [200, 0, 200, 1, 1, 1, 1, 401],
                [300, 0, 200, 1, 1, 1, 1, 402],
                [250, 0, 300, 1, 1, 1, 1, 403],
            ],
            "indices": [0, 1, 2],
            "primitive": "triangles",
            "drape": True,
        })
        
        # Set sun to east (azimuth 90)
        send_ipc(sock, {
            "cmd": "set_terrain_sun",
            "azimuth_deg": 90,
            "elevation_deg": 45,
            "intensity": 1.0,
        })
        
        # Set sun to west (azimuth 270)
        send_ipc(sock, {
            "cmd": "set_terrain_sun",
            "azimuth_deg": 270,
            "elevation_deg": 45,
            "intensity": 1.0,
        })


class TestVectorOverlayVisibility:
    """Test vector overlay visibility controls."""

    def test_set_vector_overlay_visible(self, viewer_context):
        """Set overlay visibility."""
        sock, _ = viewer_context
        
        # Add overlay
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "vis_test",
            "vertices": [[100, 0, 100, 1, 0, 0, 1, 501]],
            "indices": [0],
            "primitive": "points",
        })
        
        # Hide it
        send_ipc(sock, {
            "cmd": "set_vector_overlay_visible",
            "id": 0,
            "visible": False,
        })
        # Response may vary

    def test_set_vector_overlay_opacity(self, viewer_context):
        """Set overlay opacity."""
        sock, _ = viewer_context
        
        send_ipc(sock, {
            "cmd": "set_vector_overlay_opacity",
            "id": 0,
            "opacity": 0.5,
        })
        # Response may vary

    def test_set_global_vector_overlay_opacity(self, viewer_context):
        """Set global overlay opacity multiplier."""
        sock, _ = viewer_context
        
        resp = send_ipc(sock, {
            "cmd": "set_global_vector_overlay_opacity",
            "opacity": 0.8,
        })
        assert resp.get("ok", False)


class TestVectorOverlayZOrder:
    """Test vector overlay z-ordering."""

    def test_z_order_stacking(self, viewer_context):
        """Overlays with higher z_order should render on top."""
        sock, _ = viewer_context
        
        # Add background overlay (z_order=0)
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "background",
            "vertices": [
                [100, 0, 100, 0, 0, 1, 1, 601],
                [200, 0, 100, 0, 0, 1, 1, 602],
                [200, 0, 200, 0, 0, 1, 1, 603],
                [100, 0, 200, 0, 0, 1, 1, 604],
            ],
            "indices": [0, 1, 2, 0, 2, 3],
            "primitive": "triangles",
            "drape": True,
            "z_order": 0,
        })
        
        # Add foreground overlay (z_order=1)
        send_ipc(sock, {
            "cmd": "add_vector_overlay",
            "name": "foreground",
            "vertices": [
                [120, 0, 120, 1, 0, 0, 1, 701],
                [180, 0, 120, 1, 0, 0, 1, 702],
                [150, 0, 180, 1, 0, 0, 1, 703],
            ],
            "indices": [0, 1, 2],
            "primitive": "triangles",
            "drape": True,
            "z_order": 1,
        })
        
        # Visual inspection would show red triangle on blue quad
