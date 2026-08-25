"""Headless tests for the optional Phase 2 notebook widget surface."""

import pytest

import numpy as np

import ipywidgets

from forge3d import widgets as widgets_module
from forge3d.datasets import mini_dem
from forge3d.widgets import ViewerWidget, widgets_available


class _DummyHandle:
    def __init__(self) -> None:
        self.commands = []
        self.closed = False
        self.port = 4242

    def send_ipc(self, cmd):
        self.commands.append(("send_ipc", cmd))
        return {"ok": True, "cmd": cmd.get("cmd")}

    def set_orbit_camera(self, phi_deg, theta_deg, radius, fov_deg=None, target=None):
        self.commands.append(
            ("set_orbit_camera", phi_deg, theta_deg, radius, fov_deg, target)
        )

    def set_sun(self, azimuth_deg, elevation_deg):
        self.commands.append(("set_sun", azimuth_deg, elevation_deg))

    def snapshot(self, path, width=None, height=None):
        self.commands.append(("snapshot", path, width, height))
        try:
            from PIL import Image
            Image.new("RGBA", (max(int(width or 64), 1), max(int(height or 64), 1)), (32, 64, 96, 255)).save(path)
        except ImportError:
            # Fallback: write a minimal valid PNG without Pillow
            import struct, zlib
            w, h = max(int(width or 64), 1), max(int(height or 64), 1)
            raw = b"".join(b"\x00" + b"\x20\x40\x60\xff" * w for _ in range(h))
            with open(str(path), "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")
                def _chunk(ctype, data):
                    f.write(struct.pack(">I", len(data)))
                    f.write(ctype)
                    f.write(data)
                    f.write(struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))
                _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                _chunk(b"IDAT", zlib.compress(raw))
                _chunk(b"IEND", b"")

    def close(self):
        self.closed = True
        self.commands.append(("close",))


def test_widgets_available_reflects_optional_dependency():
    """The widgets module should detect the installed ipywidgets dependency."""
    assert widgets_available() is True


def test_inline_preview_can_render_png_bytes():
    """The internal inline fallback should generate PNG bytes from the bundled DEM."""
    view = widgets_module._InlinePreview(src=mini_dem(), width=320, height=220, auto_render=False)

    png_bytes = view.render_png_bytes()

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert view.wait_for_idle() is True
    assert view._last_image is not None
    assert view._last_image.shape == (220, 320, 4)


def test_inline_preview_update_params_triggers_fresh_render():
    """The inline fallback should re-render when persistent parameters are updated."""
    view = widgets_module._InlinePreview(src=mini_dem(), width=240, height=180, auto_render=False)
    first = view.render_png_bytes()

    view.update_params(camera_phi=120.0, lighting_azimuth=180.0)
    second = view.render_png_bytes()

    assert first != second


def test_inline_hillshade_uses_geographic_east_west_direction():
    east_rising_heightmap = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))

    shade_from_east = widgets_module._hillshade(east_rising_heightmap, azimuth_deg=90.0, elevation_deg=30.0)
    shade_from_west = widgets_module._hillshade(east_rising_heightmap, azimuth_deg=270.0, elevation_deg=30.0)

    assert float(shade_from_west.mean()) > float(shade_from_east.mean())


def test_viewer_widget_delegates_to_handle_methods(tmp_path):
    """ViewerWidget should expose the ViewerHandle control surface in notebooks."""
    widget = ViewerWidget(auto_launch=False, fallback_to_render=False)
    dummy = _DummyHandle()
    widget._handle = dummy

    response = widget.send_ipc({"cmd": "ping"})
    widget.set_camera(phi_deg=45.0, theta_deg=35.0, radius=1500.0, fov_deg=40.0)
    widget.set_sun(azimuth_deg=135.0, elevation_deg=30.0)
    output_path = widget.snapshot(path=tmp_path / "widget.png", width=200, height=120)

    assert response == {"ok": True, "cmd": "ping"}
    assert output_path == tmp_path / "widget.png"
    assert output_path.exists()
    assert ("set_orbit_camera", 45.0, 35.0, 1500.0, 40.0, None) in dummy.commands
    assert ("set_sun", 135.0, 30.0) in dummy.commands

    widget.close()

    assert dummy.closed is True


def test_viewer_widget_snapshot_uses_inline_fallback(tmp_path):
    """ViewerWidget should be able to snapshot from the inline fallback preview."""
    widget = ViewerWidget(auto_launch=False, fallback_to_render=True)
    widget._fallback = widgets_module._InlinePreview(src=mini_dem(), width=200, height=140, auto_render=False)

    output_path = widget.snapshot(path=tmp_path / "fallback.png", width=200, height=140)

    assert output_path == tmp_path / "fallback.png"
    assert output_path.exists()


def test_widgets_module_only_exports_public_widget_surface():
    """The module-level public surface should hide the old RenderView API."""
    assert widgets_module.__all__ == ["ViewerWidget", "widgets_available"]
    assert not hasattr(widgets_module, "RenderView")
