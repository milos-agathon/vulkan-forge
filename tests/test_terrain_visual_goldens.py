from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.terrain_params import (
    PomSettings,
    ReflectionSettings,
    SkySettings,
    make_terrain_params_config,
)

from tests._golden_variants import (
    assert_nvidia_vulkan_golden_adapter,
    nvidia_vulkan_golden_selected,
    selected_golden_path,
)
from tests._ssim import ssim


if os.environ.get("FORGE3D_RUN_TERRAIN_GOLDENS") != "1":
    pytest.skip("Terrain golden tests run only in the dedicated GPU CI lane", allow_module_level=True)

if not f3d.has_gpu() or not all(
    hasattr(f3d, name)
    for name in ("TerrainRenderer", "TerrainRenderParams", "OverlayLayer", "MaterialSet", "IBL")
):
    pytest.skip("Terrain golden tests require GPU-backed native module", allow_module_level=True)


GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "terrain"
UPDATE_GOLDENS = os.environ.get("FORGE3D_UPDATE_TERRAIN_GOLDENS") == "1"
ARTIFACT_DIR = (
    Path(os.environ["FORGE3D_TERRAIN_GOLDEN_ARTIFACT_DIR"])
    if os.environ.get("FORGE3D_TERRAIN_GOLDEN_ARTIFACT_DIR")
    else None
)


def _create_test_hdr(path: str, width: int = 8, height: int = 4) -> None:
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                f.write(bytes([r, g, 128, 128]))


def _build_heightmap(size: int = 96) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ridge = 0.52 * np.exp(-((xx + 0.25) ** 2 * 6.5 + (yy - 0.12) ** 2 * 10.0))
    basin = -0.18 * np.exp(-((xx - 0.05) ** 2 * 20.0 + (yy + 0.05) ** 2 * 24.0))
    spur = 0.22 * np.exp(-((xx - 0.42) ** 2 * 28.0 + (yy + 0.22) ** 2 * 18.0))
    slope = 0.25 * (1.0 - yy) + 0.10 * xx
    heightmap = ridge + basin + spur + slope
    heightmap -= heightmap.min()
    heightmap /= max(float(heightmap.max()), 1e-6)
    return heightmap.astype(np.float32)


def _build_overlay() -> object:
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (0.0, "#18391f"),
            (0.38, "#4e7c35"),
            (0.65, "#8f7a4a"),
            (0.82, "#b8ac88"),
            (1.0, "#f2f4f7"),
        ],
        domain=(0.0, 1.0),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _build_water_mask(size: int = 96) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    lake = ((xx + 0.02) / 0.55) ** 2 + ((yy + 0.18) / 0.28) ** 2 <= 1.0
    inlet = ((xx - 0.34) / 0.22) ** 2 + ((yy + 0.10) / 0.18) ** 2 <= 1.0
    return np.where(lake | inlet, 1.0, 0.0).astype(np.float32)


def _render_scene(
    renderer,
    material_set,
    ibl,
    heightmap,
    overlay,
    *,
    reflection: ReflectionSettings | None = None,
    sky: SkySettings | None = None,
    water_mask: np.ndarray | None = None,
    light_azimuth_deg: float = 135.0,
    light_elevation_deg: float = 24.0,
    sun_intensity: float = 2.4,
    size_px: tuple[int, int] = (192, 128),
    render_scale: float = 1.0,
    msaa_samples: int = 1,
    cam_radius: float = 5.0,
    cam_phi_deg: float = 138.0,
    cam_theta_deg: float = 63.0,
    albedo_mode: str = "colormap",
    colormap_strength: float = 1.0,
    pom: PomSettings | None = None,
) -> np.ndarray:
    pom_settings = pom or PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False)
    params = make_terrain_params_config(
        size_px=size_px,
        render_scale=render_scale,
        terrain_span=2.8,
        msaa_samples=msaa_samples,
        z_scale=1.45,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode=albedo_mode,
        colormap_strength=colormap_strength,
        ibl_enabled=True,
        light_azimuth_deg=light_azimuth_deg,
        light_elevation_deg=light_elevation_deg,
        sun_intensity=sun_intensity,
        cam_radius=cam_radius,
        cam_phi_deg=cam_phi_deg,
        cam_theta_deg=cam_theta_deg,
        fov_y_deg=54.0,
        camera_mode="screen",
        overlays=[overlay],
        pom=pom_settings,
        reflection=reflection,
        sky=sky,
    )
    native_params = f3d.TerrainRenderParams(params)
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=native_params,
        heightmap=heightmap,
        target=None,
        water_mask=water_mask,
    )
    return frame.to_numpy()


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(path, image)


def _golden_path(scene_name: str) -> Path:
    """Select an explicitly bound physical backend baseline.

    The required NVIDIA lane and optional Metal diagnostic use separate
    references because backend output differs measurably. There is no
    cross-backend reference fallback.
    """
    return selected_golden_path(
        GOLDEN_DIR,
        scene_name,
        "FORGE3D_TERRAIN_GOLDEN_VARIANT",
        implicit_metal=True,
    )


def _write_failure_artifacts(scene_name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    diff = np.abs(actual[..., :3].astype(np.int16) - expected[..., :3].astype(np.int16)).astype(
        np.uint8
    )
    diff_rgba = np.concatenate(
        [diff, np.full(diff.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1
    )
    _save_png(ARTIFACT_DIR / f"{scene_name}_actual.png", actual)
    _save_png(ARTIFACT_DIR / f"{scene_name}_expected.png", expected)
    _save_png(ARTIFACT_DIR / f"{scene_name}_diff.png", diff_rgba)


def _assert_matches_golden(scene_name: str, actual: np.ndarray) -> None:
    golden_path = _golden_path(scene_name)
    if UPDATE_GOLDENS:
        _save_png(golden_path, actual)
        return

    assert golden_path.exists(), (
        f"Missing terrain golden {golden_path}. "
        "Regenerate with FORGE3D_UPDATE_TERRAIN_GOLDENS=1."
    )

    expected = f3d.png_to_numpy(golden_path)
    assert actual.shape == expected.shape

    mean_abs = float(
        np.mean(np.abs(actual[..., :3].astype(np.float32) - expected[..., :3].astype(np.float32)))
    )
    score = ssim(actual[..., :3], expected[..., :3], data_range=255.0)

    if score < 0.995 or mean_abs > 2.0:
        _write_failure_artifacts(scene_name, actual, expected)

    assert score >= 0.995, f"{scene_name} SSIM too low: {score:.6f}"
    assert mean_abs <= 2.0, f"{scene_name} mean absolute difference too high: {mean_abs:.4f}"


def _assert_sky_scene_is_meaningful(scene_name: str, actual: np.ndarray) -> None:
    if scene_name not in {"terrain_atmosphere", "terrain_low_sun_sky"}:
        return
    rgb = np.asarray(actual[..., :3], dtype=np.float64)
    if rgb.size and float(rgb.max()) <= 1.5:
        rgb = rgb * 255.0
    # Both fixed fixtures reserve the upper eighth for unobstructed sky. Test
    # that region rather than the whole frame so lit terrain cannot hide a
    # missing/black atmosphere during an intentional baseline refresh.
    sky_band = rgb[: max(rgb.shape[0] // 8, 1)]
    luminance = (
        0.2126 * sky_band[..., 0]
        + 0.7152 * sky_band[..., 1]
        + 0.0722 * sky_band[..., 2]
    )
    assert float(luminance.mean()) > 20.0, f"{scene_name} collapsed to a black sky"
    assert float((luminance > 10.0).mean()) > 0.9, (
        f"{scene_name} has too little visible sky coverage"
    )
    assert float(luminance.std()) > 1.0, f"{scene_name} lost visible sky variation"


@pytest.fixture(scope="module")
def terrain_golden_env():
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    if nvidia_vulkan_golden_selected("FORGE3D_TERRAIN_GOLDEN_VARIANT"):
        adapter_probe = f3d.device_probe("vulkan")
        assert_nvidia_vulkan_golden_adapter(
            "FORGE3D_TERRAIN_GOLDEN_VARIANT", adapter_probe
        )
        if ARTIFACT_DIR is not None:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            (ARTIFACT_DIR / "terrain-render-adapter.json").write_text(
                json.dumps(adapter_probe, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    material_set = f3d.MaterialSet.terrain_default()
    overlay = _build_overlay()
    heightmap = _build_heightmap()
    water_mask = _build_water_mask()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        tmp.close()
        _create_test_hdr(tmp.name)
        ibl = f3d.IBL.from_hdr(tmp.name, intensity=1.0)
    os.unlink(tmp.name)

    return renderer, material_set, ibl, heightmap, overlay, water_mask


@pytest.mark.parametrize(
    ("scene_name", "scene_kwargs"),
    [
        (
            "terrain_pbr",
            dict(),
        ),
        (
            "terrain_water",
            dict(
                water_mask=True,
                light_elevation_deg=18.0,
            ),
        ),
        (
            "terrain_atmosphere",
            dict(
                sky=SkySettings(
                    enabled=True,
                    turbidity=5.5,
                    ground_albedo=0.35,
                    sun_intensity=1.8,
                    sun_size=1.6,
                    aerial_density=2.8,
                    sky_exposure=1.1,
                ),
                light_elevation_deg=12.0,
            ),
        ),
        (
            "terrain_low_sun_sky",
            dict(
                sky=SkySettings(
                    enabled=True,
                    model="hosek-wilkie",
                    turbidity=7.0,
                    ground_albedo=0.42,
                    sun_intensity=2.2,
                    sun_size=1.8,
                    aerial_density=3.2,
                    sky_exposure=1.15,
                ),
                light_elevation_deg=5.0,
                light_azimuth_deg=118.0,
                cam_radius=5.1,
                cam_phi_deg=138.0,
                cam_theta_deg=68.0,
            ),
        ),
        (
            "terrain_pom",
            dict(
                size_px=(256, 160),
                render_scale=1.25,
                msaa_samples=4,
                albedo_mode="material",
                colormap_strength=0.0,
                cam_radius=4.2,
                cam_phi_deg=142.0,
                cam_theta_deg=38.0,
                light_elevation_deg=22.0,
                pom=PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
            ),
        ),
        (
            "terrain_water_reflection",
            dict(
                size_px=(256, 160),
                msaa_samples=4,
                albedo_mode="mix",
                colormap_strength=0.35,
                water_mask=True,
                light_elevation_deg=15.0,
                sun_intensity=2.8,
                cam_radius=4.3,
                cam_phi_deg=142.0,
                cam_theta_deg=42.0,
                reflection=ReflectionSettings(
                    enabled=True,
                    intensity=1.0,
                    fresnel_power=3.0,
                    wave_strength=0.05,
                    shore_atten_width=0.12,
                    water_plane_height=0.0,
                ),
            ),
        ),
    ],
)
def test_terrain_visual_goldens(terrain_golden_env, scene_name: str, scene_kwargs: dict) -> None:
    renderer, material_set, ibl, heightmap, overlay, water_mask = terrain_golden_env

    kwargs = dict(scene_kwargs)
    if kwargs.pop("water_mask", False):
        kwargs["water_mask"] = water_mask

    actual = _render_scene(renderer, material_set, ibl, heightmap, overlay, **kwargs)
    _assert_sky_scene_is_meaningful(scene_name, actual)
    _assert_matches_golden(scene_name, actual)
