"""P1.2 Bloom Execute Behavior -- behavioral tests.

Proves bloom settings affect the live Scene runtime state:
- BloomSettings Python-level validation (disabled passthrough semantics).
- Scene enable/disable/set/get round-trip (GPU-gated).
- No remaining no-op marker text in the retained bloom config surface.

Canonical BloomSettings validation tests live in
``tests/test_api_contracts.py::TestBloomSettingsWiring``.
"""

from __future__ import annotations

import pytest
import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE, get_native_module

if not NATIVE_AVAILABLE:
    pytest.skip(
        "Bloom behavior tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

_native = get_native_module()

# ---------------------------------------------------------------------------
# Bloom disabled-passthrough semantics (no GPU needed)
# ---------------------------------------------------------------------------
class TestBloomDisabledPassthrough:
    """Verify that disabled bloom is a semantic no-op at the config level."""

    def test_bloom_settings_default_disabled(self):
        """Default BloomSettings has enabled=False."""
        from forge3d.terrain_params import BloomSettings
        s = BloomSettings()
        assert s.enabled is False

    def test_bloom_disabled_does_not_alter_params(self):
        """Passing disabled BloomSettings to terrain params leaves bloom off."""
        from forge3d.terrain_params import BloomSettings, make_terrain_params_config
        bloom = BloomSettings()  # default: disabled
        params = make_terrain_params_config(
            size_px=(64, 64),
            render_scale=1.0,
            terrain_span=100.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            bloom=bloom,
        )
        assert params.bloom is not None
        assert params.bloom.enabled is False

    def test_bloom_enabled_propagates_to_params(self):
        """Enabled BloomSettings with custom values propagates correctly."""
        from forge3d.terrain_params import BloomSettings, make_terrain_params_config
        bloom = BloomSettings(enabled=True, threshold=2.0, intensity=0.8, radius=1.5)
        params = make_terrain_params_config(
            size_px=(64, 64),
            render_scale=1.0,
            terrain_span=100.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(0.0, 100.0),
            bloom=bloom,
        )
        assert params.bloom.enabled is True
        assert params.bloom.threshold == 2.0
        assert params.bloom.intensity == 0.8
        assert params.bloom.radius == 1.5


# ---------------------------------------------------------------------------
# Bloom validation edge cases (no GPU needed)
# ---------------------------------------------------------------------------
class TestBloomValidation:
    """Verify BloomSettings rejects invalid configurations."""

    def test_negative_threshold_rejected(self):
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="threshold"):
            BloomSettings(threshold=-0.5)

    def test_softness_out_of_range_rejected(self):
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="softness"):
            BloomSettings(softness=2.0)

    def test_negative_intensity_rejected(self):
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="intensity"):
            BloomSettings(intensity=-0.1)

    def test_zero_radius_rejected(self):
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="radius"):
            BloomSettings(radius=0.0)


# ---------------------------------------------------------------------------
# No-op marker absence
# ---------------------------------------------------------------------------
class TestBloomNoOpMarkerAbsent:
    """Verify no remaining no-op marker text in the retained bloom module.

    The checklist requires: ``rg -n "Bloom is a no-op" src/core/bloom.rs``
    returns exit code 1 (no matches).  We verify from Python by scanning
    the file for the marker string.
    """

    def test_no_noop_marker_in_bloom_rs(self):
        import pathlib
        bloom_rs = (
            pathlib.Path(__file__).resolve().parent.parent
            / "src" / "core" / "bloom.rs"
        )
        assert bloom_rs.exists(), f"bloom.rs not found at {bloom_rs}"
        text = bloom_rs.read_text(encoding="utf-8")
        assert "Bloom is a no-op" not in text, (
            "Found no-op marker 'Bloom is a no-op' in src/core/bloom.rs -- "
            "this should have been removed when execute() was implemented."
        )


# ---------------------------------------------------------------------------
# Scene integration (GPU-gated)
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


@pytest.mark.skipif(not _SCENE_AVAILABLE, reason="Scene requires GPU + valid shaders")
class TestBloomSceneRoundTrip:
    """Prove bloom enable/disable and set/get round-trip on a live Scene."""

    @pytest.fixture
    def scene(self):
        return _native.Scene(64, 64)

    def test_bloom_default_disabled(self, scene):
        """Scene starts with bloom disabled."""
        assert scene.is_bloom_enabled() is False

    def test_bloom_enable_disable_cycle(self, scene):
        scene.enable_bloom()
        assert scene.is_bloom_enabled() is True
        scene.disable_bloom()
        assert scene.is_bloom_enabled() is False

    def test_bloom_settings_round_trip(self, scene):
        """set_bloom_settings → get_bloom_settings preserves custom values."""
        scene.set_bloom_settings(threshold=2.0, softness=0.7,
                                  strength=0.8, radius=1.5)
        got = scene.get_bloom_settings()
        assert isinstance(got, dict)
        assert got["threshold"] == pytest.approx(2.0)
        assert got["softness"] == pytest.approx(0.7)
        assert got["strength"] == pytest.approx(0.8)
        assert got["radius"] == pytest.approx(1.5)

    def test_bloom_settings_differ_after_update(self, scene):
        """Changing bloom settings produces different get output."""
        defaults = scene.get_bloom_settings()
        scene.set_bloom_settings(threshold=5.0, strength=1.0)
        updated = scene.get_bloom_settings()
        assert updated["threshold"] != defaults["threshold"]
        assert updated["strength"] != defaults["strength"]

    def test_bloom_disabled_is_render_passthrough(self, scene):
        """Changing bloom params while disabled leaves rendered output unchanged."""
        import numpy as np

        scene.disable_bloom()
        baseline = np.asarray(scene.render_rgba()).copy()

        scene.set_bloom_settings(threshold=0.2, softness=1.0, strength=3.0, radius=4.0)
        still_disabled = np.asarray(scene.render_rgba())

        diff = np.abs(still_disabled.astype(np.int16) - baseline.astype(np.int16)).mean()
        assert diff <= 0.01, f"Disabled bloom should be passthrough, got mean diff={diff}"

    def test_bloom_enabled_changes_render_output(self, scene):
        """Enabled bloom with aggressive params changes rendered output."""
        import numpy as np

        scene.disable_bloom()
        scene.set_ssgi_settings(
            _native.SSGISettings(ray_steps=96, ray_radius=12.0, intensity=6.0)
        )
        scene.enable_ssgi()
        baseline = np.asarray(scene.render_rgba()).copy()

        # The default Scene can render only the low-luminance clear color on
        # hosted/software adapters. A zero threshold still proves enabled bloom
        # mutates rendered pixels without depending on terrain brightness.
        scene.set_bloom_settings(threshold=0.0, softness=1.0, strength=2.5, radius=4.0)
        scene.enable_bloom()
        bloomed = np.asarray(scene.render_rgba())

        diff = np.abs(bloomed.astype(np.int16) - baseline.astype(np.int16)).mean()
        assert diff > 0.05, f"Expected bloom output change, got mean diff={diff}"
