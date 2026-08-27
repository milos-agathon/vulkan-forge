# tests/test_terrain_demo_preset_integration.py
# Acceptance one-liner: simulate `examples/terrain_demo.py --preset outdoor_sun` + overrides
# by constructing a RendererConfig via load_renderer_config(preset_map, overrides).
# This mirrors the merge order in examples/terrain_demo.py::_build_renderer_config.
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import forge3d as f3d
from forge3d import presets
from forge3d.config import load_renderer_config
from forge3d.terrain_demo import (
    DEFAULT_CAMERA_MODE,
    DEFAULT_CAM_FOV,
    DEFAULT_CAM_PHI,
    DEFAULT_CAM_RADIUS,
    DEFAULT_CAM_THETA,
    _apply_preset_dem_defaults,
    _apply_preset_cli_defaults,
)
from forge3d.terrain_params import SamplingSettings, make_terrain_params_config


def test_acceptance_one_liner_outdoor_sun_presets_merge() -> None:
    # Equivalent of:
    #   python examples/terrain_demo.py --preset outdoor_sun \
    #       --brdf cooktorrance-ggx --shadows pcf --cascades 4 --hdr assets/sky.hdr
    preset_map = presets.get("outdoor_sun")
    cfg = load_renderer_config(
        preset_map,
        overrides={
            "brdf": "cooktorrance-ggx",
            "shadows": "pcf",
            "cascades": 4,
            "hdr": "assets/sky.hdr",
        },
    )

    # Expectations: overrides win; preset-provided atmosphere sky is preserved
    assert cfg.shading.brdf == "cooktorrance-ggx"
    assert cfg.shadows.technique == "pcf"
    assert cfg.shadows.cascades == 4
    assert cfg.atmosphere.sky == "hosek-wilkie"
    assert cfg.atmosphere.hdr_path == "assets/sky.hdr"


def _demo_args(preset: str, *, explicit: set[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        preset=preset,
        camera_mode=DEFAULT_CAMERA_MODE,
        cam_radius=DEFAULT_CAM_RADIUS,
        cam_phi=DEFAULT_CAM_PHI,
        cam_theta=DEFAULT_CAM_THETA,
        cam_fov=DEFAULT_CAM_FOV,
        sun_azimuth=None,
        sun_elevation=None,
        sun_intensity=None,
        sun_color=None,
        ibl_intensity=1.0,
        z_scale=2.0,
        _explicit_cli_args=explicit or set(),
    )


def test_terrain_demo_preset_defaults_come_from_preset_payload() -> None:
    args = _demo_args("rainier_showcase")

    _apply_preset_cli_defaults(args)
    _apply_preset_dem_defaults(args, terrain_span=1000.0)

    assert args.cam_radius == 2400.0
    assert args.cam_phi == 135.0
    assert args.cam_theta == 45.0
    assert args.cam_fov == 55.0
    assert args.sun_azimuth == 135.0
    assert args.sun_elevation == 25.0
    assert args.sun_intensity == 4.0
    assert args.sun_color == [1.0, 0.95, 0.90]
    assert args.ibl_intensity == 0.3
    assert Path(args.hdr).exists()
    assert args.z_scale == 1.35
    assert args.height_filter == "Nearest"


def test_height_filter_defaults_to_linear_and_validates_choices() -> None:
    sampling = SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat")
    assert sampling.height_filter == "Linear"
    nearest = SamplingSettings(
        "Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat", "Nearest"
    )
    assert nearest.height_filter == "Nearest"
    for native_sampling in (sampling, nearest):
        f3d.TerrainRenderParams(
            make_terrain_params_config(
                size_px=(64, 64),
                render_scale=1.0,
                terrain_span=10.0,
                msaa_samples=1,
                z_scale=1.0,
                exposure=1.0,
                domain=(0.0, 1.0),
                sampling=native_sampling,
            )
        )
    with pytest.raises(ValueError, match="height_filter"):
        SamplingSettings(
            "Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat", "Cubic"
        )


def test_terrain_demo_preserves_explicit_cli_overrides() -> None:
    args = _demo_args("rainier_relief", explicit={"cam_radius", "cam_theta", "sun_elevation", "ibl_intensity", "z_scale"})
    args.cam_radius = 777.0
    args.cam_theta = 12.0
    args.sun_elevation = 42.0
    args.ibl_intensity = 0.9
    args.z_scale = 3.0

    _apply_preset_cli_defaults(args)
    _apply_preset_dem_defaults(args, terrain_span=1000.0)

    assert args.camera_mode == "mesh"
    assert args.cam_radius == 777.0
    assert args.cam_theta == 12.0
    assert args.cam_phi == 45.0
    assert args.sun_azimuth == 225.0
    assert args.sun_elevation == 42.0
    assert args.ibl_intensity == 0.9
    assert args.z_scale == 3.0


def test_mapscene_and_terrain_demo_premium_preset_fields_match() -> None:
    terrain = f3d.TerrainSource(
        data=np.zeros((12, 16), dtype=np.float32),
        crs="EPSG:32610",
        metadata={"width": 16, "height": 12, "source_id": "inline-dem"},
        elevation_sampling_available=True,
    )

    for name in ("rainier_showcase", "rainier_relief"):
        args = _demo_args(name)
        _apply_preset_cli_defaults(args)
        _apply_preset_dem_defaults(args, terrain_span=16.0)
        scene = f3d.MapScene(
            terrain=terrain,
            lighting=f3d.LightingPreset(name=name),
            output=f3d.OutputSpec(width=64, height=64, path="out.png"),
        )

        assert scene.recipe.camera.distance == args.cam_radius
        assert scene.recipe.camera.azimuth_deg == args.cam_phi
        assert scene.recipe.camera.elevation_deg == args.cam_theta
        assert scene.recipe.camera.fov_deg == args.cam_fov
        assert scene.recipe.lighting.intensity == args.sun_intensity
        assert scene.recipe.lighting.settings["ibl"]["intensity"] == args.ibl_intensity
        assert scene.recipe.lighting.settings["exaggeration"] == args.z_scale
        expected_height_filter = "Nearest" if name == "rainier_showcase" else "Linear"
        assert scene.recipe.lighting.settings["height_sampling"] == expected_height_filter
        assert args.height_filter == expected_height_filter
