"""CLI-level perspective projection checks using terrain_demo."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_DEM = Path(__file__).parent.parent / "python" / "forge3d" / "data" / "mini_dem.npy"
pytestmark = pytest.mark.offscreen


def _render_with_probe(
    camera_mode: str,
    fov: float,
    theta: float,
    output_name: str,
    output_dir: Path,
    hdr_path: Path,
    *,
    phi: float = 135.0,
    debug_mode: int = 41,
) -> str:
    """Render terrain with projection probe and return MD5 hash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "examples" / "terrain_demo.py"),
        "--dem",
        str(DEMO_DEM),
        "--size",
        "128",
        "128",
        "--hdr",
        str(hdr_path),
        "--camera-mode",
        camera_mode,
        "--cam-fov",
        str(fov),
        "--cam-theta",
        str(theta),
        "--cam-phi",
        str(phi),
        "--debug-mode",
        str(debug_mode),
        "--shadows",
        "none",
        "--msaa",
        "1",
        "--output",
        str(output_path),
        "--overwrite",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        pytest.fail(f"Render failed: {result.stderr[:500]}")

    return hashlib.md5(output_path.read_bytes()).hexdigest()


@pytest.mark.apple_metal_physical
class TestPerspectiveProjectionCli:
    """Ensure CLI plumbing preserves perspective controls."""

    def test_mesh_mode_fov_changes_probe(self, tmp_path):
        """FOV variation should change NDC-depth probe in mesh mode."""
        from _generated_assets import write_hdr

        hdr = write_hdr(tmp_path / "probe.hdr")
        h1 = _render_with_probe(
            "mesh", fov=30, theta=45, output_name="cli_probe_fov30.png",
            output_dir=tmp_path, hdr_path=hdr,
        )
        h2 = _render_with_probe(
            "mesh", fov=90, theta=45, output_name="cli_probe_fov90.png",
            output_dir=tmp_path, hdr_path=hdr,
        )
        assert h1 != h2

    def test_mesh_mode_theta_changes_probe(self, tmp_path):
        """Theta variation should change probe."""
        from _generated_assets import write_hdr

        hdr = write_hdr(tmp_path / "probe.hdr")
        h1 = _render_with_probe(
            "mesh", fov=55, theta=25, output_name="cli_probe_theta25.png",
            output_dir=tmp_path, hdr_path=hdr,
        )
        h2 = _render_with_probe(
            "mesh", fov=55, theta=75, output_name="cli_probe_theta75.png",
            output_dir=tmp_path, hdr_path=hdr,
        )
        assert h1 != h2

    def test_mesh_mode_phi_rotates_probe(self, tmp_path):
        """Phi variation should rotate probe output."""
        from _generated_assets import write_hdr

        hdr = write_hdr(tmp_path / "probe.hdr")
        h1 = _render_with_probe(
            "mesh", fov=55, theta=45, phi=0.0,
            output_name="cli_probe_phi0.png", output_dir=tmp_path, hdr_path=hdr,
        )
        h2 = _render_with_probe(
            "mesh", fov=55, theta=45, phi=90.0,
            output_name="cli_probe_phi90.png", output_dir=tmp_path, hdr_path=hdr,
        )
        assert h1 != h2

    def test_screen_vs_mesh_differ(self, tmp_path):
        """Legacy screen mode should differ from mesh mode for same angles."""
        from _generated_assets import write_hdr

        hdr = write_hdr(tmp_path / "probe.hdr")
        h_screen = _render_with_probe(
            "screen", fov=55, theta=45, output_name="cli_probe_screen.png",
            output_dir=tmp_path, hdr_path=hdr, debug_mode=41,
        )
        h_mesh = _render_with_probe(
            "mesh", fov=55, theta=45, output_name="cli_probe_mesh.png",
            output_dir=tmp_path, hdr_path=hdr, debug_mode=41,
        )
        assert h_screen != h_mesh
