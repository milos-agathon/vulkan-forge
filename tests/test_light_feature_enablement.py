"""Regression coverage for light feature enablement on a live Scene.

These tests mirror the Windows notebook failure mode from March 19, 2026:
construct a Scene, upload a DEM, then call the enable methods that previously
panicked during shader/pipeline validation.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE, get_native_module

if not NATIVE_AVAILABLE:
    pytest.skip(
        "Light enablement tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

_native = get_native_module()
_HAS_GPU = f3d.has_gpu()


def _try_create_scene():
    try:
        return _native.Scene(64, 64)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return None


_SCENE_AVAILABLE = _HAS_GPU and _try_create_scene() is not None


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not _SCENE_AVAILABLE, reason="Scene requires GPU + valid shaders")
class TestLightFeatureEnablement:
    @pytest.fixture
    def scene(self):
        scene = _native.Scene(64, 64)
        scene.set_height_from_r32f(f3d.mini_dem().astype("float32"))
        return scene

    def test_enable_soft_light_radius_matches_notebook_repro(self, scene):
        scene.enable_soft_light_radius()
        assert scene.is_soft_light_radius_enabled() is True

        frame = np.asarray(scene.render_rgba())
        assert frame.shape == (64, 64, 4)
        assert frame.dtype == np.uint8

    def test_enable_point_spot_lights_matches_notebook_repro(self, scene):
        scene.enable_point_spot_lights()
        assert scene.is_point_spot_lights_enabled() is True

        light_id = scene.add_point_light(0.0, 10.0, 0.0, 1.0, 0.9, 0.8, 2.0, 25.0)
        assert isinstance(light_id, int)
        assert scene.get_light_count() == 1

        frame = np.asarray(scene.render_rgba())
        assert frame.shape == (64, 64, 4)
        assert frame.dtype == np.uint8

    def test_clear_all_lights_no_longer_hits_enablement_catch_22(self, scene):
        scene.enable_point_spot_lights()
        scene.add_point_light(0.0, 8.0, 0.0, 1.0, 1.0, 1.0, 1.5, 20.0)
        assert scene.get_light_count() == 1

        scene.clear_all_lights()
        assert scene.get_light_count() == 0
