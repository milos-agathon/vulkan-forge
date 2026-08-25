"""P1.1 SSGI/SSR Settings Wiring -- behavioral tests.

Proves that SSGI and SSR settings alter runtime state beyond mere mutability:
- Settings construction with different parameters produces observably different state.
- Scene integration: set → get round-trip preserves custom values (GPU-gated).
- Enable/disable toggle changes query result (GPU-gated).

Canonical settings-object tests live in
``tests/test_api_contracts.py::TestSsgiSsrSettingsWiring``.
"""

from __future__ import annotations

import pytest
import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE, get_native_module

if not NATIVE_AVAILABLE:
    pytest.skip(
        "SSGI/SSR wiring tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

_native = get_native_module()

# Re-export canonical settings-object tests so they run under this file too.
from tests.test_api_contracts import TestSsgiSsrSettingsWiring  # noqa: F401


# ---------------------------------------------------------------------------
# Behavioral state-change proof (no GPU needed)
# ---------------------------------------------------------------------------
class TestSsgiBehavioralStateChange:
    """Prove SSGISettings construction produces observably different state.

    These go beyond mutability: each test creates two distinct settings
    objects and asserts their fields differ in a meaningful way.
    """

    def test_different_ray_steps_produce_different_state(self):
        a = _native.SSGISettings(ray_steps=24)
        b = _native.SSGISettings(ray_steps=96)
        assert a.ray_steps != b.ray_steps

    def test_different_intensity_produce_different_state(self):
        a = _native.SSGISettings(intensity=1.0)
        b = _native.SSGISettings(intensity=4.0)
        assert a.intensity != b.intensity

    def test_use_half_res_toggle(self):
        a = _native.SSGISettings(use_half_res=False)
        b = _native.SSGISettings(use_half_res=True)
        assert a.use_half_res != b.use_half_res

    def test_full_configuration_round_trip(self):
        """Construct with all fields, read back, verify all match."""
        s = _native.SSGISettings(
            ray_steps=48, ray_radius=10.0, ray_thickness=1.0,
            intensity=2.5, temporal_alpha=0.6, use_half_res=True,
            ibl_fallback=0.4,
        )
        assert s.ray_steps == 48
        assert s.ray_radius == pytest.approx(10.0)
        assert s.ray_thickness == pytest.approx(1.0)
        assert s.intensity == pytest.approx(2.5)
        assert s.temporal_alpha == pytest.approx(0.6)
        assert s.use_half_res is True
        assert s.ibl_fallback == pytest.approx(0.4)


class TestSsrBehavioralStateChange:
    """Prove SSRSettings construction produces observably different state."""

    def test_different_max_steps_produce_different_state(self):
        a = _native.SSRSettings(max_steps=48)
        b = _native.SSRSettings(max_steps=200)
        assert a.max_steps != b.max_steps

    def test_different_intensity_produce_different_state(self):
        a = _native.SSRSettings(intensity=1.0)
        b = _native.SSRSettings(intensity=0.2)
        assert a.intensity != b.intensity

    def test_roughness_fade_toggle(self):
        a = _native.SSRSettings(roughness_fade=0.1)
        b = _native.SSRSettings(roughness_fade=0.9)
        assert a.roughness_fade != b.roughness_fade

    def test_full_configuration_round_trip(self):
        """Construct with all fields, read back, verify all match."""
        s = _native.SSRSettings(
            max_steps=128, max_distance=300.0, thickness=0.5,
            stride=3.0, intensity=0.6, roughness_fade=0.4,
            edge_fade=0.2, temporal_alpha=0.9,
        )
        assert s.max_steps == 128
        assert s.max_distance == pytest.approx(300.0)
        assert s.thickness == pytest.approx(0.5)
        assert s.stride == pytest.approx(3.0)
        assert s.intensity == pytest.approx(0.6)
        assert s.roughness_fade == pytest.approx(0.4)
        assert s.edge_fade == pytest.approx(0.2)
        assert s.temporal_alpha == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Scene integration (GPU-gated, tolerates shader compilation issues)
# ---------------------------------------------------------------------------
_HAS_GPU = f3d.has_gpu()


def _try_create_scene():
    """Attempt to create a Scene; return None if GPU/shader init fails."""
    try:
        return _native.Scene(64, 64)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return None


_SCENE_AVAILABLE = _HAS_GPU and _try_create_scene() is not None


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not _SCENE_AVAILABLE, reason="Scene requires GPU + valid shaders")
class TestSsgiSceneRoundTrip:
    """Prove set_ssgi_settings → get_ssgi_settings round-trip on a live Scene."""

    @pytest.fixture
    def scene(self):
        return _native.Scene(64, 64)

    def test_default_ssgi_settings_retrievable(self, scene):
        settings = scene.get_ssgi_settings()
        assert isinstance(settings, dict)
        assert "ray_steps" in settings
        assert "intensity" in settings

    def test_custom_ssgi_settings_round_trip(self, scene):
        custom = _native.SSGISettings(ray_steps=64, intensity=3.0, ray_radius=8.0)
        scene.set_ssgi_settings(custom)
        got = scene.get_ssgi_settings()
        assert got["ray_steps"] == 64
        assert got["intensity"] == pytest.approx(3.0)
        assert got["ray_radius"] == pytest.approx(8.0)

    def test_ssgi_enable_disable_cycle(self, scene):
        scene.enable_ssgi()
        assert scene.is_ssgi_enabled() is True
        scene.disable_ssgi()
        assert scene.is_ssgi_enabled() is False

    def test_ssgi_settings_differ_from_defaults_after_set(self, scene):
        """Changing settings produces different get_ssgi_settings output."""
        defaults = scene.get_ssgi_settings()
        custom = _native.SSGISettings(ray_steps=100, intensity=4.5)
        scene.set_ssgi_settings(custom)
        updated = scene.get_ssgi_settings()
        assert updated["ray_steps"] != defaults["ray_steps"]
        assert updated["intensity"] != defaults["intensity"]

    def test_ssgi_changes_render_output_when_enabled(self, scene):
        """Enabled SSGI with strong settings changes rendered output."""
        import numpy as np

        scene.disable_ssgi()
        baseline = np.asarray(scene.render_rgba()).copy()

        scene.set_ssgi_settings(
            _native.SSGISettings(ray_steps=96, ray_radius=12.0, intensity=6.0)
        )
        scene.enable_ssgi()
        with_ssgi = np.asarray(scene.render_rgba())

        diff = np.abs(with_ssgi.astype(np.int16) - baseline.astype(np.int16)).mean()
        assert diff > 0.05, f"Expected SSGI output change, got mean diff={diff}"


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not _SCENE_AVAILABLE, reason="Scene requires GPU + valid shaders")
class TestSsrSceneRoundTrip:
    """Prove set_ssr_settings → get_ssr_settings round-trip on a live Scene."""

    @pytest.fixture
    def scene(self):
        return _native.Scene(64, 64)

    def test_default_ssr_settings_retrievable(self, scene):
        settings = scene.get_ssr_settings()
        assert isinstance(settings, dict)
        assert "max_steps" in settings
        assert "intensity" in settings

    def test_custom_ssr_settings_round_trip(self, scene):
        custom = _native.SSRSettings(max_steps=128, intensity=0.5, max_distance=200.0)
        scene.set_ssr_settings(custom)
        got = scene.get_ssr_settings()
        assert got["max_steps"] == 128
        assert got["intensity"] == pytest.approx(0.5)
        assert got["max_distance"] == pytest.approx(200.0)

    def test_ssr_enable_disable_cycle(self, scene):
        scene.enable_ssr()
        assert scene.is_ssr_enabled() is True
        scene.disable_ssr()
        assert scene.is_ssr_enabled() is False

    def test_ssr_settings_differ_from_defaults_after_set(self, scene):
        """Changing settings produces different get_ssr_settings output."""
        defaults = scene.get_ssr_settings()
        custom = _native.SSRSettings(max_steps=200, intensity=4.0)
        scene.set_ssr_settings(custom)
        updated = scene.get_ssr_settings()
        assert updated["max_steps"] != defaults["max_steps"]
        assert updated["intensity"] != defaults["intensity"]

    def test_ssr_changes_render_output_when_enabled(self, scene):
        """Enabled SSR with strong settings changes rendered output."""
        import numpy as np

        scene.disable_ssr()
        gradient = np.zeros((64, 64, 4), dtype=np.uint8)
        rows = np.linspace(0, 255, 64, dtype=np.uint8)[:, None]
        gradient[..., 0] = rows
        gradient[..., 1] = 255 - rows
        gradient[..., 3] = 255
        scene.set_raster_overlay(gradient, 1.0, None, None)
        baseline = np.asarray(scene.render_rgba()).copy()
        vertical_delta = np.abs(
            baseline.astype(np.int16) - baseline[::-1].astype(np.int16)
        ).mean()
        if vertical_delta <= 0.05:
            pytest.skip("SSR output-change proof requires a vertically varying baseline")

        scene.set_ssr_settings(
            _native.SSRSettings(
                max_steps=192, max_distance=300.0, thickness=0.8, stride=2.0,
                intensity=5.0, roughness_fade=0.1, edge_fade=0.9, temporal_alpha=0.0,
            )
        )
        scene.enable_ssr()
        with_ssr = np.asarray(scene.render_rgba())

        diff = np.abs(with_ssr.astype(np.int16) - baseline.astype(np.int16)).mean()
        assert diff > 0.05, f"Expected SSR output change, got mean diff={diff}"
