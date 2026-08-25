"""Tests for camera-sun alignment warning and rainier_showcase preset.

These tests verify that:
1. The alignment warning triggers when camera and sun are nearly aligned (dot > 0.7)
2. The warning does NOT trigger when sufficiently offset
3. The rainier_showcase preset loads correctly
4. The check_camera_sun_alignment function computes correct dot products
"""

import pytest
from forge3d.terrain_demo import check_camera_sun_alignment
from forge3d import presets


class TestCameraSunAlignment:
    """Test the camera-sun alignment detection function."""

    def test_aligned_high_dot(self):
        """When camera and sun point in same direction, dot should be high."""
        # Camera phi and sun azimuth use different conventions
        # cam_phi=0 looks along +X, sun_azimuth=90 comes from +X direction
        # So cam_phi=0 + sun_azimuth=90 are aligned (both reference +X)
        dot = check_camera_sun_alignment(
            cam_phi_deg=0.0,
            cam_theta_deg=35.0,
            sun_azimuth_deg=90.0,
            sun_elevation_deg=35.0,
        )
        # Should be very high (close to 1.0)
        assert dot > 0.9, f"Expected dot > 0.9 for aligned angles, got {dot}"

    def test_offset_90_low_dot(self):
        """Sun offset by 90° azimuth should produce low/negative dot."""
        # Camera at phi=0°, sun at azimuth=180° (cross-lighting)
        dot = check_camera_sun_alignment(
            cam_phi_deg=0.0,
            cam_theta_deg=35.0,
            sun_azimuth_deg=180.0,
            sun_elevation_deg=35.0,
        )
        # Cross-lighting should produce moderate dot
        assert dot < 0.7, f"Expected dot < 0.7 for cross-lighting, got {dot}"

    def test_opposite_negative_dot(self):
        """Sun opposite to camera should produce negative dot."""
        # Camera at phi=0°, sun at azimuth=270° (opposite direction)
        dot = check_camera_sun_alignment(
            cam_phi_deg=0.0,
            cam_theta_deg=30.0,
            sun_azimuth_deg=270.0,
            sun_elevation_deg=30.0,
        )
        # Opposite direction should be negative
        assert dot < 0.0, f"Expected dot < 0 for opposite angles, got {dot}"

    def test_warning_threshold_aligned(self):
        """Verify that the warning threshold (0.7) is exceeded for near-aligned configs."""
        # This is the "flat Rainier" scenario: cam_phi similar to sun_azimuth
        dot = check_camera_sun_alignment(
            cam_phi_deg=30.0,
            cam_theta_deg=35.0,
            sun_azimuth_deg=30.0,  # Same as camera phi
            sun_elevation_deg=35.0,
        )
        # Same azimuth should trigger warning
        assert dot > 0.7, f"Expected dot > 0.7 for same azimuth, got {dot}"

    def test_warning_threshold_offset(self):
        """Verify that 90° offset does NOT trigger warning."""
        dot = check_camera_sun_alignment(
            cam_phi_deg=30.0,
            cam_theta_deg=35.0,
            sun_azimuth_deg=120.0,  # 90° offset from camera
            sun_elevation_deg=35.0,
        )
        # 90° offset should NOT trigger warning
        assert dot < 0.7, f"Expected dot < 0.7 for 90° offset, got {dot}"

    def test_user_reported_case(self):
        """Test the actual user-reported flat Rainier configuration."""
        # User config: cam_phi=30, sun_azimuth=135
        dot = check_camera_sun_alignment(
            cam_phi_deg=30.0,
            cam_theta_deg=35.0,
            sun_azimuth_deg=135.0,
            sun_elevation_deg=35.0,
        )
        # This is 105° offset in azimuth, but elevation is same
        # The reported dot was 0.503, which is below warning threshold
        assert 0.4 < dot < 0.6, f"Expected dot ~0.5 for this config, got {dot}"
        # Should NOT trigger warning (below 0.7 threshold)
        assert dot < 0.7, f"This config should not trigger warning"

    def test_dot_product_range(self):
        """Verify dot product is always in valid range [-1, 1]."""
        test_cases = [
            (0, 0, 0, 0),
            (90, 45, 90, 45),
            (180, 60, 0, 30),
            (270, 15, 45, 75),
            (45, 45, 225, 45),
        ]
        for cam_phi, cam_theta, sun_az, sun_el in test_cases:
            dot = check_camera_sun_alignment(cam_phi, cam_theta, sun_az, sun_el)
            assert -1.0 <= dot <= 1.0, f"Dot product {dot} out of range for angles {(cam_phi, cam_theta, sun_az, sun_el)}"


class TestRainierShowcasePreset:
    """Test the rainier_showcase preset."""

    def test_preset_exists(self):
        """Verify rainier_showcase is in available presets."""
        available = presets.available()
        assert "rainiershowcase" in available, f"rainier_showcase not in presets: {available}"

    def test_preset_loads(self):
        """Verify rainier_showcase loads without error."""
        cfg = presets.get("rainier_showcase")
        assert isinstance(cfg, dict)
        assert "lighting" in cfg
        assert "shadows" in cfg
        assert "gi" in cfg

    def test_preset_has_pcss_shadows(self):
        """Verify preset uses PCSS with 4 cascades."""
        cfg = presets.get("rainier_showcase")
        assert cfg["shadows"]["technique"] == "pcss"
        assert cfg["shadows"]["cascades"] == 4
        assert cfg["shadows"]["map_size"] == 4096

    def test_preset_enables_ibl(self):
        """Verify preset enables IBL for fill lighting."""
        cfg = presets.get("rainier_showcase")
        assert "ibl" in cfg["gi"]["modes"]

    def test_preset_aliases(self):
        """Verify preset aliases work."""
        for alias in ["rainier", "showcase", "terrain"]:
            cfg = presets.get(alias)
            assert cfg["shadows"]["technique"] == "pcss"
            assert cfg["shadows"]["cascades"] == 4


class TestCLIIntegration:
    """Test CLI integration (requires subprocess)."""

    def test_warning_printed_when_aligned(self, tmp_path):
        """Verify warning is printed when camera-sun are aligned."""
        import subprocess
        import sys
        
        from _generated_assets import write_geotiff, write_hdr

        dem = write_geotiff(tmp_path / "aligned.tif")
        hdr = write_hdr(tmp_path / "aligned.hdr")
        output = tmp_path / "aligned.png"
        result = subprocess.run(
            [
                sys.executable, "-B", "examples/terrain_demo.py",
                "--dem", str(dem),
                "--hdr", str(hdr),
                "--size", "64", "64",
                "--cam-phi", "0",
                "--sun-azimuth", "90",  # Both use +X after convention conversion
                "--output", str(output),
                "--overwrite",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0, result.stderr
        assert output.is_file()
        # Warning should be in output
        assert "[WARNING]" in result.stdout or "[WARNING]" in result.stderr
        assert (
            "nearly aligned" in result.stdout.lower()
            or "nearly aligned" in result.stderr.lower()
        )

    def test_no_warning_when_offset(self, tmp_path):
        """Verify no warning when camera-sun are offset."""
        import subprocess
        import sys

        from _generated_assets import write_geotiff, write_hdr

        dem = write_geotiff(tmp_path / "offset.tif")
        hdr = write_hdr(tmp_path / "offset.hdr")
        output = tmp_path / "offset.png"
        result = subprocess.run(
            [
                sys.executable, "-B", "examples/terrain_demo.py",
                "--dem", str(dem),
                "--hdr", str(hdr),
                "--size", "64", "64",
                "--cam-phi", "0",
                "--sun-azimuth", "135",  # 135° offset - should NOT trigger warning
                "--output", str(output),
                "--overwrite",
            ],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0, result.stderr
        assert output.is_file()
        # Warning should NOT be in output
        assert "[WARNING]" not in result.stdout
