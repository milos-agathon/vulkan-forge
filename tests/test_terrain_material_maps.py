from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import (
    AovSettings,
    MaterialLayerSettings,
    PomSettings,
    make_terrain_params_config,
)

from _terrain_runtime import (
    terrain_rendering_available as _terrain_rendering_available,
)

_requires_gpu = pytest.mark.skipif(
    not _terrain_rendering_available(),
    reason=(
        "MapScene native terrain render requires a terrain-safe hardware "
        "adapter; hosted runners (GPU-less, WARP, or guarded paravirtual "
        "Metal) either block with a native-render diagnostic or raise by "
        "design (SUTURA), so these render-asserting tests skip honestly"
    ),
)



GPU_AVAILABLE = terrain_rendering_available()


def test_material_layer_settings_accepts_texture_map_paths() -> None:
    settings = MaterialLayerSettings(
        normal_path="rock-normal.png",
        roughness_path="rock-roughness.png",
        mask_path="rock-mask.png",
    )

    assert settings.normal_path == "rock-normal.png"
    assert settings.roughness_path == "rock-roughness.png"
    assert settings.mask_path == "rock-mask.png"


def test_material_layer_settings_rejects_empty_texture_map_path() -> None:
    with pytest.raises(ValueError, match="normal_path"):
        MaterialLayerSettings(normal_path="")


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8), mode="RGBA").save(path)


def _build_params(*, materials: MaterialLayerSettings | None, size_px: tuple[int, int] = (96, 72)):
    config = make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=6.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="material",
        colormap_strength=0.0,
        ibl_enabled=True,
        ibl_intensity=1.0,
        light_azimuth_deg=120.0,
        light_elevation_deg=28.0,
        sun_intensity=2.0,
        cam_radius=4.0,
        cam_phi_deg=140.0,
        cam_theta_deg=55.0,
        fov_y_deg=50.0,
        camera_mode="mesh",
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aov=AovSettings(enabled=True, albedo=False, normal=True, depth=False),
        materials=materials,
    )
    return f3d.TerrainRenderParams(config)


def test_native_render_params_decode_material_map_paths(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.png"
    roughness_path = tmp_path / "roughness.png"
    mask_path = tmp_path / "mask.png"
    flat = np.full((2, 2, 4), 255, dtype=np.uint8)
    _write_rgba_png(normal_path, flat)
    _write_rgba_png(roughness_path, flat)
    _write_rgba_png(mask_path, flat)

    params = _build_params(
        materials=MaterialLayerSettings(
            normal_path=str(normal_path),
            roughness_path=str(roughness_path),
            mask_path=str(mask_path),
        )
    )

    assert params.material_map_paths == {
        "normal": str(normal_path),
        "roughness": str(roughness_path),
        "mask": str(mask_path),
    }


@pytest.mark.apple_metal_physical
@_requires_gpu
def test_mapscene_terrain_metadata_wires_material_maps_to_native_params(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.png"
    roughness_path = tmp_path / "roughness.png"
    mask_path = tmp_path / "mask.png"
    flat = np.full((2, 2, 4), 255, dtype=np.uint8)
    _write_rgba_png(normal_path, flat)
    _write_rgba_png(roughness_path, flat)
    _write_rgba_png(mask_path, flat)

    heightmap = np.zeros((8, 8), dtype=np.float32)
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=heightmap,
            crs="EPSG:32610",
            metadata={
                "width": 8,
                "height": 8,
                "material_maps": {
                    "normal_path": str(normal_path),
                    "roughness_path": str(roughness_path),
                    "mask_path": str(mask_path),
                },
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(
            name="daylight",
            settings={"albedo_mode": "material", "colormap_strength": 0.0},
        ),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )

    params = map_scene._build_mapscene_terrain_params(scene.recipe, heightmap, (64, 64))

    assert params is not None
    assert params.material_map_paths == {
        "normal": str(normal_path),
        "roughness": str(roughness_path),
        "mask": str(mask_path),
    }


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not GPU_AVAILABLE, reason="terrain material map render test requires GPU-backed forge3d module")
def test_material_normal_map_changes_native_normal_aov(tmp_path: Path) -> None:
    normal_path = tmp_path / "tilted-normal.png"
    normal_map = np.zeros((16, 16, 4), dtype=np.uint8)
    normal_map[..., 0] = 230
    normal_map[..., 1] = 128
    normal_map[..., 2] = 180
    normal_map[..., 3] = 255
    _write_rgba_png(normal_path, normal_map)

    heightmap = np.full((64, 64), 0.35, dtype=np.float32)
    material_set = f3d.MaterialSet.terrain_default(normal_strength=1.0)
    with (tmp_path / "env.hdr").open("wb") as handle:
        handle.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n")
        handle.write(bytes([128, 128, 196, 128]) * 4)
    ibl = f3d.IBL.from_hdr(str(tmp_path / "env.hdr"), intensity=1.0)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))

    _, baseline_aov = renderer.render_with_aov(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(materials=None),
        heightmap=heightmap,
    )
    _, mapped_aov = renderer.render_with_aov(
        material_set=material_set,
        env_maps=ibl,
        params=_build_params(materials=MaterialLayerSettings(normal_path=str(normal_path))),
        heightmap=heightmap,
    )

    baseline = np.asarray(baseline_aov.normal(), dtype=np.float32)
    mapped = np.asarray(mapped_aov.normal(), dtype=np.float32)
    assert float(np.mean(np.abs(mapped - baseline))) > 0.02
