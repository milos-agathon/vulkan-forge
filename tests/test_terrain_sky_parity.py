from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available
from forge3d.terrain_params import PomSettings, SkySettings, make_terrain_params_config


pytestmark = pytest.mark.apple_metal_physical
def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                b = 128
                e = 128
                f.write(bytes([r, g, b, e]))


def _build_heightmap(size: int = 96) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ridge = 0.65 * np.exp(-((xx + 0.25) ** 2 * 7.5 + (yy - 0.05) ** 2 * 14.0))
    spur = 0.30 * np.exp(-((xx - 0.38) ** 2 * 22.0 + (yy + 0.18) ** 2 * 24.0))
    valley = 0.18 * (1.0 - yy)
    heightmap = ridge + spur + valley
    heightmap -= heightmap.min()
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_overlay():
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#0b2f18"),
            (0.45, "#4b7a2d"),
            (0.75, "#9e8f53"),
            (1.0, "#f0f2f6"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _build_params(
    overlay,
    *,
    sky: SkySettings | None,
    light_azimuth_deg: float = 135.0,
    light_elevation_deg: float = 26.0,
):
    return make_terrain_params_config(
        size_px=(192, 128),
        render_scale=1.0,
        terrain_span=2.5,
        msaa_samples=1,
        z_scale=1.4,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="colormap",
        colormap_strength=1.0,
        ibl_enabled=False,
        light_azimuth_deg=light_azimuth_deg,
        light_elevation_deg=light_elevation_deg,
        sun_intensity=2.5,
        cam_radius=5.0,
        cam_phi_deg=135.0,
        cam_theta_deg=64.0,
        fov_y_deg=55.0,
        camera_mode="screen",
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        sky=sky,
    )


def _render_scene(
    renderer,
    material_set,
    ibl,
    heightmap,
    overlay,
    *,
    sky,
    light_azimuth_deg=135.0,
    light_elevation_deg=26.0,
):
    params = _build_params(
        overlay,
        sky=sky,
        light_azimuth_deg=light_azimuth_deg,
        light_elevation_deg=light_elevation_deg,
    )
    native_params = f3d.TerrainRenderParams(params)
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=native_params,
        heightmap=heightmap,
        target=None,
    )
    return frame.to_numpy()


def _image_hash(image: np.ndarray) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def _mean_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(
        np.mean(np.abs(lhs[..., :3].astype(np.float32) - rhs[..., :3].astype(np.float32)))
    )


def _full_rgb(image: np.ndarray) -> np.ndarray:
    return image[..., :3].astype(np.float32) / 255.0


def _mean_saturation(rgb: np.ndarray) -> float:
    max_rgb = np.max(rgb, axis=-1)
    min_rgb = np.min(rgb, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(max_rgb > 0.0, (max_rgb - min_rgb) / max_rgb, 0.0)
    return float(np.mean(sat))


@pytest.fixture(scope="module")
def terrain_sky_env():
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    overlay = _build_overlay()
    heightmap = _build_heightmap()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)

    return renderer, material_set, ibl, heightmap, overlay


def test_sky_disabled_is_pixel_identical_to_baseline(terrain_sky_env):
    renderer, material_set, ibl, heightmap, overlay = terrain_sky_env

    baseline = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=None,
    )
    disabled = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(
            enabled=False,
            turbidity=8.0,
            ground_albedo=0.95,
            sun_intensity=4.0,
            sun_size=3.5,
            aerial_density=2.5,
            sky_exposure=2.0,
        ),
    )

    assert np.array_equal(baseline, disabled), "Disabled sky must preserve the pre-TV1 baseline exactly"


def test_sky_enabled_changes_output(terrain_sky_env):
    renderer, material_set, ibl, heightmap, overlay = terrain_sky_env

    baseline = _render_scene(renderer, material_set, ibl, heightmap, overlay, sky=None)
    enabled = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(
            enabled=True,
            turbidity=2.0,
            ground_albedo=0.3,
            sun_intensity=1.5,
            sun_size=1.25,
            aerial_density=2.5,
            sky_exposure=1.0,
        ),
    )

    assert _image_hash(baseline) != _image_hash(enabled), "Sky-enabled terrain output must differ from the baseline"
    assert _mean_abs_diff(baseline, enabled) > 0.25


@pytest.mark.parametrize(
    ("base_kwargs", "variant_kwargs", "label"),
    [
        (
            dict(turbidity=2.0),
            dict(turbidity=8.0),
            "turbidity",
        ),
        (
            dict(ground_albedo=0.1),
            dict(ground_albedo=0.9),
            "ground_albedo",
        ),
        (
            dict(sun_intensity=0.5),
            dict(sun_intensity=4.0),
            "sun_intensity",
        ),
        (
            dict(sun_size=0.5),
            dict(sun_size=4.0),
            "sun_size",
        ),
        (
            dict(sky_exposure=0.6),
            dict(sky_exposure=2.0),
            "sky_exposure",
        ),
    ],
)
def test_sky_parameters_change_output(terrain_sky_env, base_kwargs, variant_kwargs, label):
    renderer, material_set, ibl, heightmap, overlay = terrain_sky_env

    base = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(enabled=True, aerial_density=2.5, **base_kwargs),
    )
    variant = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(enabled=True, aerial_density=2.5, **variant_kwargs),
    )

    assert _image_hash(base) != _image_hash(variant), f"{label} is still dead plumbing in the terrain renderer"
    assert _mean_abs_diff(base, variant) > 0.15, f"{label} change is not observable enough to be trustworthy"


def test_sky_aerial_density_changes_output(terrain_sky_env):
    renderer, material_set, ibl, heightmap, overlay = terrain_sky_env

    low_haze = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(enabled=True, turbidity=3.0, aerial_density=0.25, sun_intensity=1.5),
        light_elevation_deg=12.0,
    )
    high_haze = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(enabled=True, turbidity=3.0, aerial_density=4.0, sun_intensity=1.5),
        light_elevation_deg=12.0,
    )

    assert _image_hash(low_haze) != _image_hash(high_haze), "Sky aerial density is still dead plumbing in the terrain renderer"
    assert _mean_abs_diff(low_haze, high_haze) > 0.12


def test_clear_hazy_low_sun_regression_scenes(terrain_sky_env):
    renderer, material_set, ibl, heightmap, overlay = terrain_sky_env

    clear = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(enabled=True, turbidity=1.5, aerial_density=1.0, sun_intensity=1.2),
        light_elevation_deg=28.0,
    )
    hazy = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(enabled=True, turbidity=8.0, aerial_density=3.0, sun_intensity=1.2),
        light_elevation_deg=28.0,
    )
    low_sun = _render_scene(
        renderer,
        material_set,
        ibl,
        heightmap,
        overlay,
        sky=SkySettings(enabled=True, turbidity=3.0, aerial_density=2.0, sun_intensity=2.5, sun_size=2.0),
        light_elevation_deg=5.0,
    )

    clear_rgb = _full_rgb(clear)
    hazy_rgb = _full_rgb(hazy)
    low_sun_rgb = _full_rgb(low_sun)

    assert _image_hash(clear) != _image_hash(hazy)
    assert _image_hash(clear) != _image_hash(low_sun)
    assert _mean_saturation(clear_rgb) > _mean_saturation(hazy_rgb), "Clear scene should retain more saturation than hazy atmosphere"

    clear_warmth = float(np.mean(clear_rgb[..., 0] - clear_rgb[..., 2]))
    low_sun_warmth = float(np.mean(low_sun_rgb[..., 0] - low_sun_rgb[..., 2]))
    assert low_sun_warmth > clear_warmth, "Low-sun scene should warm the terrain image relative to the clear midday scene"
