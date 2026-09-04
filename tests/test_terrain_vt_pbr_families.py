"""SUBSTRATIA: full-PBR terrain VT normal + mask family gates.

Covers the moonshot definition-of-done:

- ``test_normal_family_changes_lighting_ssim`` — the gated measurable win: the
  normal VT family must change *beauty lighting* under grazing light by an
  SSIM difference > 0.05 (beyond the existing normal-AOV coverage in
  ``test_tv20_virtual_texturing.py``).
- ``test_all_families_page_within_budget`` — albedo, normal, and mask all page
  with per-family resident bytes > 0 whose sum stays within the VT residency
  budget and the 512 MiB host-visible ceiling.
- ``test_missing_family_is_fatal`` — a requested family with no registered
  source raises a fatal diagnostic instead of degrading silently.
- ``test_partial_normal_residency_degrades_gracefully`` — non-resident normal
  tiles fall back to the geometric surface normal (no corrupted/black-normal
  lighting) while the render completes.
- ``test_partial_mask_residency_uses_neutral_fallback`` — non-resident mask
  tiles preserve neutral roughness/AO instead of sampling stale atlas data.
- ``test_gpu_shader_feedback_preserves_family_coordinates`` — raw forward and
  visibility feedback proves all family/material pairs, shared Grid UVs for
  normal/mask, and distinct triplanar albedo coordinates.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _substratia_evidence import (
    image_sha256,
    load_golden,
    record_substratia_image,
    record_substratia_result,
)
from _terrain_runtime import _build_heightmap, _build_overlay, terrain_rendering_available
from forge3d.diagnostics import visibility_stats
from forge3d.helpers.offscreen import save_png_deterministic
from forge3d.terrain_params import (
    AovSettings,
    PomSettings,
    TerrainVTSettings,
    VTLayerFamily,
    make_terrain_params_config,
)

GPU_AVAILABLE = terrain_rendering_available()
VT_MATERIAL_COUNT = 4
MIB = 1024.0 * 1024.0
MEMORY_BUDGET_LIMIT_BYTES = 512 * 1024 * 1024
GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "terrain"
GOLDEN_VARIANT = os.environ.get(
    "FORGE3D_SUBSTRATIA_GOLDEN_VARIANT",
    "metal" if sys.platform == "darwin" else "nvidia-vulkan",
)
if GOLDEN_VARIANT not in {"metal", "nvidia-vulkan"}:
    raise RuntimeError(
        "FORGE3D_SUBSTRATIA_GOLDEN_VARIANT must be 'metal' or 'nvidia-vulkan'"
    )
BASELINE_GOLDEN = GOLDEN_DIR / f"substratia_grazing_baseline.{GOLDEN_VARIANT}.png"
NORMAL_GOLDEN = GOLDEN_DIR / f"substratia_grazing_normal.{GOLDEN_VARIANT}.png"

# Labeled grazing-light detail region (fractions of image height/width) used
# by the SSIM gate: central band where the low-sun normal shading dominates.
GRAZING_REGION = (0.18, 0.85, 0.12, 0.88)


# ---------------------------------------------------------------------------
# Local SSIM harness (no external dependency)
# ---------------------------------------------------------------------------

def _box_mean(img: np.ndarray, radius: int) -> np.ndarray:
    """Edge-clamped box-filter mean with window (2*radius+1)^2."""
    size = 2 * radius + 1
    padded = np.pad(img, radius, mode="edge").astype(np.float64)
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)))
    h, w = img.shape
    total = (
        csum[size : size + h, size : size + w]
        - csum[0:h, size : size + w]
        - csum[size : size + h, 0:w]
        + csum[0:h, 0:w]
    )
    return total / float(size * size)


def _ssim(a: np.ndarray, b: np.ndarray, radius: int = 3) -> float:
    """Mean structural similarity of two grayscale images in [0, 1]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = _box_mean(a, radius)
    mu_b = _box_mean(b, radius)
    sigma_a = _box_mean(a * a, radius) - mu_a * mu_a
    sigma_b = _box_mean(b * b, radius) - mu_b * mu_b
    sigma_ab = _box_mean(a * b, radius) - mu_a * mu_b
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / denominator))


def _luminance(rgba: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgba, dtype=np.float64)[..., :3]
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _region_slices(shape: tuple[int, ...]) -> tuple[slice, slice]:
    top, bottom, left, right = GRAZING_REGION
    h, w = shape[0], shape[1]
    return (
        slice(int(h * top), int(h * bottom)),
        slice(int(w * left), int(w * right)),
    )


def _bind_render_process_to_expected_adapter() -> None:
    expected_path = os.environ.get("FORGE3D_EXPECTED_ADAPTER_PROBE")
    if not expected_path:
        return
    envelope = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    expected = envelope.get("probe")
    requested_backend = str(envelope.get("requested_backend", "")).strip() or None
    actual = f3d.device_probe(requested_backend)
    assert isinstance(expected, dict) and isinstance(actual, dict)
    name = str(actual.get("name", ""))
    assert actual.get("status") == "ok"
    assert str(actual.get("device_type", "")).lower() in {"discretegpu", "integratedgpu"}
    assert actual.get("software_fallback") is False
    for token in ("software", "virtual", "paravirtual", "warp", "llvmpipe"):
        assert token not in name.lower()
    for field in (
        "backend",
        "device_type",
        "name",
        "vendor",
        "device",
        "software_fallback",
    ):
        assert str(actual.get(field, "")).lower() == str(expected.get(field, "")).lower()
    artifact_dir = os.environ.get("FORGE3D_SUBSTRATIA_ARTIFACT_DIR")
    if artifact_dir:
        output = Path(artifact_dir) / "render-process-adapter.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Procedural VT sources
# ---------------------------------------------------------------------------

def _build_albedo_source(size: int, material_index: int) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    checker = ((np.floor(xx * 12) + np.floor(yy * 12)) % 2.0).astype(np.float32)
    palette = np.array(
        [
            [0.80, 0.25, 0.15],
            [0.20, 0.65, 0.25],
            [0.20, 0.35, 0.85],
            [0.90, 0.80, 0.20],
        ],
        dtype=np.float32,
    )
    base = palette[material_index % len(palette)]
    rgb = np.clip(base * (0.4 + 0.6 * checker[..., None]), 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((size, size, 1), dtype=np.float32)], axis=-1)
    return np.ascontiguousarray((rgba * 255.0).round().astype(np.uint8))


def _build_bumpy_normal_source(
    size: int,
    material_index: int,
    frequency: float = 22.0,
    amplitude: float = 6.0,
) -> np.ndarray:
    """Tangent-space normal map with strong sinusoidal bumps.

    Encoded [0,1] RGB; decodes to normals with pronounced slopes so grazing
    light produces long shading variation across the tile.
    """
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    two_pi_f = 2.0 * np.pi * (frequency + material_index * 3.0)
    dzdx = amplitude * np.cos(two_pi_f * xx) * np.sin(two_pi_f * yy * 0.71 + 1.3)
    dzdy = amplitude * np.sin(two_pi_f * xx * 0.63 + 0.4) * np.cos(two_pi_f * yy)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba)


def _build_mask_source(size: int, material_index: int) -> np.ndarray:
    """Mask family source: r = gate (on), g = roughness pattern, b = AO."""
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    rough = 0.25 + 0.5 * (
        0.5 + 0.5 * np.sin(xx * (18.0 + material_index)) * np.cos(yy * 15.0)
    )
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.clip(rough * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba)


def _register_family_sources(
    renderer: "f3d.TerrainRenderer",
    virtual_size: int,
    families: tuple[str, ...],
) -> None:
    builders = {
        "albedo": _build_albedo_source,
        "normal": _build_bumpy_normal_source,
        "mask": _build_mask_source,
    }
    fallbacks = {
        "albedo": [0.5, 0.5, 0.5, 1.0],
        "normal": [0.5, 0.5, 1.0, 1.0],
        "mask": [1.0, 1.0, 1.0, 1.0],
    }
    for material_index in range(VT_MATERIAL_COUNT):
        for family in families:
            renderer.register_material_vt_source(
                material_index,
                family,
                builders[family](virtual_size, material_index),
                (virtual_size, virtual_size),
                fallbacks[family],
            )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _write_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 164, 128]))


def _build_test_ibl():
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_test_hdr(hdr_path)
        return f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)


def _build_render_params(
    *,
    vt_settings: TerrainVTSettings | None,
    size_px: tuple[int, int] = (256, 192),
    cam_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cam_radius: float = 4.0,
    light_elevation_deg: float = 24.0,
    sun_intensity: float = 2.2,
    ibl_intensity: float = 1.8,
    normal_aov: bool = False,
    shading: str = "forward",
    camera_mode: str = "mesh",
    culling: str = "frustum",
    prefetch_horizon_ms: float = 100.0,
    vt_upload_budget_bytes: int = 16 * 1024 * 1024,
) -> "f3d.TerrainRenderParams":
    config = make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=8.0,
        msaa_samples=1,
        z_scale=1.6,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="material",
        colormap_strength=0.0,
        ibl_enabled=True,
        ibl_intensity=ibl_intensity,
        light_azimuth_deg=136.0,
        light_elevation_deg=light_elevation_deg,
        sun_intensity=sun_intensity,
        cam_radius=cam_radius,
        cam_phi_deg=142.0,
        cam_theta_deg=58.0,
        fov_y_deg=50.0,
        camera_mode=camera_mode,
        culling=culling,
        shading=shading,
        prefetch_horizon_ms=prefetch_horizon_ms,
        vt_upload_budget_bytes=vt_upload_budget_bytes,
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aov=AovSettings(enabled=True, albedo=False, normal=normal_aov, depth=False),
    )
    config.cam_target = [float(cam_target[0]), float(cam_target[1]), float(cam_target[2])]
    config.vt = vt_settings
    return f3d.TerrainRenderParams(config)


def _is_meaningful_render(image: np.ndarray) -> bool:
    rgba = _as_rgba8(image)
    rgb = rgba[..., :3]
    magenta = (rgb[..., 0] > 220) & (rgb[..., 1] < 80) & (rgb[..., 2] > 220)
    return (
        rgba.ndim == 3
        and rgba.shape[-1] == 4
        and float(magenta.mean()) < 0.01
        and float(rgb.astype(np.float64).std(axis=(0, 1)).max()) > 2.0
        and int(rgba[..., 3].max()) > 0
    )


def _render_beauty(env, params) -> np.ndarray:
    # The DoD needs a beauty image, not a particular one-shot graph. Use the
    # renderer's attachment-explicit path: it returns the same beauty target
    # while keeping the golden gate independent of the optional ANAMNESIS
    # one-shot graph (whose validation is outside SUBSTRATIA's non-goals).
    renderer, material_set, ibl, heightmap = env
    frame, _ = renderer.render_with_aov(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
    )
    image = np.asarray(frame.to_numpy())
    assert _is_meaningful_render(image), "terrain beauty render returned a magenta/empty marker"
    return image


def _render_beauty_and_normal_aov(env, params) -> tuple[np.ndarray, np.ndarray]:
    renderer, material_set, ibl, heightmap = env
    beauty = None
    aov_frame = None
    for _ in range(4):
        frame, aov_frame = renderer.render_with_aov(
            material_set=material_set,
            env_maps=ibl,
            params=params,
            heightmap=heightmap,
        )
        beauty = np.asarray(frame.to_numpy())
        if _is_meaningful_render(beauty):
            break
    assert beauty is not None and aov_frame is not None
    assert _is_meaningful_render(beauty), "AOV terrain render returned a magenta/empty marker"
    return (
        beauty,
        np.asarray(aov_frame.normal(), dtype=np.float32),
    )


def _as_rgba8(image: np.ndarray) -> np.ndarray:
    rgba = np.asarray(image)
    if rgba.dtype == np.uint8:
        return np.ascontiguousarray(rgba)
    scaled = np.asarray(rgba, dtype=np.float64)
    if scaled.size and float(np.nanmax(scaled)) <= 1.5:
        scaled = scaled * 255.0
    return np.ascontiguousarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))


def _golden(path: Path, actual: np.ndarray) -> np.ndarray:
    if os.environ.get("FORGE3D_UPDATE_SUBSTRATIA_GOLDENS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        save_png_deterministic(path, _as_rgba8(actual))
    return load_golden(path)


@pytest.fixture(scope="module")
def vt_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("terrain VT PBR tests require a terrain-capable GPU runtime")

    session = f3d.Session(window=False)
    _bind_render_process_to_expected_adapter()
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    heightmap = _build_heightmap(160)
    ibl = _build_test_ibl()
    renderer.clear_material_vt_sources()
    try:
        yield renderer, material_set, ibl, heightmap
    finally:
        try:
            renderer.clear_material_vt_sources()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Static contracts (no GPU required)
# ---------------------------------------------------------------------------

def test_vt_settings_families_property() -> None:
    settings = TerrainVTSettings(
        layers=[
            VTLayerFamily(family="albedo"),
            VTLayerFamily(family="normal"),
            VTLayerFamily(family="mask"),
        ]
    )
    assert settings.families == ("albedo", "normal", "mask")


def test_vt_family_defaults_are_pbr_safe() -> None:
    assert VTLayerFamily(family="albedo").fallback == (0.5, 0.5, 0.5, 1.0)
    assert VTLayerFamily(family="normal").fallback == (0.5, 0.5, 1.0, 1.0)
    assert VTLayerFamily(family="mask").fallback == (1.0, 1.0, 1.0, 1.0)


def test_vt_settings_enforce_shared_geometry_and_host_budget() -> None:
    with pytest.raises(ValueError, match="must share"):
        TerrainVTSettings(
            enabled=True,
            layers=[
                VTLayerFamily(family="albedo", virtual_size_px=(1024, 1024)),
                VTLayerFamily(family="normal", virtual_size_px=(2048, 2048)),
            ],
        )
    with pytest.raises(ValueError, match="512 MiB"):
        TerrainVTSettings(residency_budget_mb=512.01)
    with pytest.raises(ValueError, match="one logical tile"):
        TerrainVTSettings(
            enabled=True,
            residency_budget_mb=0.49,
            layers=[
                VTLayerFamily(family="albedo"),
                VTLayerFamily(family="normal"),
            ],
        )
    with pytest.raises(ValueError, match="one physical slot per enabled family"):
        TerrainVTSettings(
            enabled=True,
            atlas_size=256,
            residency_budget_mb=0.75,
            layers=[
                VTLayerFamily(family="albedo"),
                VTLayerFamily(family="normal"),
                VTLayerFamily(family="mask"),
            ],
        )


def test_mapscene_procedural_sources_and_defaults_are_family_safe() -> None:
    from forge3d.map_scene import _mapscene_register_vt_sources, _procedural_vt_source

    normal = _procedural_vt_source(16, 0, "normal", "checker")
    mask = _procedural_vt_source(16, 0, "mask", "checker")
    np.testing.assert_array_equal(normal[0, 0], [128, 128, 255, 255])
    np.testing.assert_array_equal(mask[0, 0], [255, 255, 255, 255])
    assert _procedural_vt_source((24, 16), 0, "albedo", "checker").shape == (
        16,
        24,
        4,
    )

    class CaptureRenderer:
        def __init__(self) -> None:
            self.sources = []

        def clear_material_vt_sources(self) -> None:
            self.sources.clear()

        def register_material_vt_source(
            self, material_index, family, image, virtual_size_px, fallback
        ) -> None:
            self.sources.append(
                (material_index, family, image, virtual_size_px, tuple(fallback))
            )

    recipe = type(
        "Recipe",
        (),
        {
            "terrain": type(
                "Terrain",
                (),
                {
                    "metadata": {
                        "virtual_texture": {
                            "enabled": True,
                            "families": [
                                {
                                    "family": family,
                                    "virtual_size_px": [256, 512],
                                    "tile_size": 248,
                                    "tile_border": 4,
                                }
                                for family in ("albedo", "normal", "mask")
                            ],
                            "procedural_sources": True,
                            "source_count": 2,
                            "source_size": [256, 512],
                        }
                    }
                },
            )()
        },
    )()
    renderer = CaptureRenderer()
    _mapscene_register_vt_sources(renderer, recipe)

    assert len(renderer.sources) == 6
    assert {(source[0], source[1]) for source in renderer.sources} == {
        (material_index, family)
        for family in ("albedo", "normal", "mask")
        for material_index in range(2)
    }
    assert all(source[2].shape == (512, 256, 4) for source in renderer.sources)
    assert all(source[3] == (256, 512) for source in renderer.sources)
    albedo = next(source for source in renderer.sources if source[1] == "albedo")
    albedo_float = np.asarray(albedo[2], dtype=np.float32)
    expected_albedo_fallback = tuple(
        float(value) for value in albedo_float[..., :3].mean(axis=(0, 1)) / 255.0
    ) + (1.0,)
    assert albedo[4] == pytest.approx(expected_albedo_fallback)
    assert {source[4] for source in renderer.sources if source[1] == "normal"} == {
        (0.5, 0.5, 1.0, 1.0)
    }
    assert {source[4] for source in renderer.sources if source[1] == "mask"} == {
        (1.0, 1.0, 1.0, 1.0)
    }

    default_recipe = type(
        "DefaultRecipe",
        (),
        {
            "terrain": type(
                "Terrain",
                (),
                {
                    "metadata": {
                        "virtual_texture": {
                            "enabled": True,
                            "procedural_sources": True,
                            "source_count": 1,
                        }
                    }
                },
            )()
        },
    )()
    _mapscene_register_vt_sources(renderer, default_recipe)
    assert len(renderer.sources) == 1
    assert renderer.sources[0][2].shape == (512, 512, 4)
    assert renderer.sources[0][3] == (512, 512)


def test_native_stub_exposes_raw_shader_feedback_records() -> None:
    stub = (Path(f3d.__file__).resolve().parent / "__init__.pyi").read_text(
        encoding="utf-8"
    )
    assert "def read_latest_vt_shader_feedback(" in stub


def test_shader_carries_family_info_and_residency_gate() -> None:
    shader_dir = Path(__file__).resolve().parents[1] / "src" / "shaders"
    shader = (shader_dir / "terrain_pbr_pom.wgsl").read_text(encoding="utf-8")
    visibility_shader = (shader_dir / "terrain_visibility_fullscreen.wgsl").read_text(
        encoding="utf-8"
    )
    for token in (
        "struct TerrainVtFamilyInfo",
        "family_info: array<TerrainVtFamilyInfo, 3>",
        "fn terrain_vt_triplanar_feedback_uvs(",
        "fn terrain_vt_triplanar_feedback_uvs_from_gradients(",
        "fn terrain_vt_write_surface_feedback(",
        "fn terrain_vt_resolve_family_uv(",
        "fn terrain_vt_sample_family_data(",
        "return vec4<f32>(0.5, 0.5, 1.0, 0.0);",
        "return vec4<f32>(1.0, 1.0, 1.0, 0.0);",
        "fn terrain_vt_normalize_family_uv(",
        "let normal_sample = terrain_vt_sample_family_data(",
        "let mask_sample = terrain_vt_sample_family_data(",
        "vt_mask_resident_roughness = vt_mask_resident_roughness",
        "roughness * (1.0 - clamp(vt_mask_residency, 0.0, 1.0))",
    ):
        assert token in shader, f"missing hardened VT shader token: {token}"

    feedback_fn = shader.split("fn terrain_vt_write_surface_feedback(", 1)[1].split(
        "// Shared residency-gated page walk.", 1
    )[0]
    for derivative_token in (
        "terrain_screen_ddx",
        "terrain_screen_ddy",
        "dpdx",
        "dpdy",
        "fwidth",
    ):
        assert derivative_token not in feedback_fn
    assert "world_feedback_ddx: vec3<f32>" in feedback_fn
    assert "world_feedback_ddy: vec3<f32>" in feedback_fn
    assert feedback_fn.count("grid_feedback_ddx,") == 2
    assert feedback_fn.count("grid_feedback_ddy,") == 2

    triplanar_from_gradients = shader.split(
        "fn terrain_vt_triplanar_feedback_uvs_from_gradients(", 1
    )[1].split("fn terrain_vt_triplanar_feedback_uvs(", 1)[0]
    for derivative_token in (
        "terrain_screen_ddx",
        "terrain_screen_ddy",
        "dpdx",
        "dpdy",
        "fwidth",
    ):
        assert derivative_token not in triplanar_from_gradients
    assert "let scaled_ddx_world = ddx_world * scale;" in triplanar_from_gradients
    assert "let scaled_ddy_world = ddy_world * scale;" in triplanar_from_gradients

    forward_fn = shader.split("fn fs_main(input : VertexOutput)", 1)[1].split(
        "fn fs_aov_main(input : VertexOutput)", 1
    )[0]
    shade_offset = forward_fn.index("let out = shade_main(input);")
    for declaration in (
        "let feedback_ddx_uv = terrain_screen_ddx_uv(input.tex_coord);",
        "let feedback_ddy_uv = terrain_screen_ddy_uv(input.tex_coord);",
        "let feedback_ddx_world = terrain_screen_ddx_world(input.world_position);",
        "let feedback_ddy_world = terrain_screen_ddy_world(input.world_position);",
    ):
        assert forward_fn.index(declaration) < shade_offset
    for argument in (
        "feedback_ddx_uv,",
        "feedback_ddy_uv,",
        "feedback_ddx_world,",
        "feedback_ddy_world,",
    ):
        assert argument in forward_fn

    fullscreen_fn = visibility_shader.split(
        "fn fs_visibility_resolve_fullscreen(", 1
    )[1].split("fn fs_visibility_geometry(", 1)[0]
    feedback_call_offset = fullscreen_fn.index("terrain_vt_write_surface_feedback(")
    for declaration in (
        "let feedback_ddx_uv = anchor10.uv - anchor00.uv;",
        "let feedback_ddy_uv = anchor01.uv - anchor00.uv;",
        "let feedback_ddx_world = anchor10.world - anchor00.world;",
        "let feedback_ddy_world = anchor01.world - anchor00.world;",
    ):
        assert fullscreen_fn.index(declaration) < feedback_call_offset

    geometry_fn = visibility_shader.split("fn fs_visibility_geometry(", 1)[1]
    discard_offset = geometry_fn.index("if (encoded != expected)")
    for declaration in (
        "let resolve_ddx_uv = dpdx(input.tex_coord);",
        "let resolve_ddy_uv = dpdy(input.tex_coord);",
        "let resolve_ddx_world = dpdx(input.world_position);",
        "let resolve_ddy_world = dpdy(input.world_position);",
    ):
        assert geometry_fn.index(declaration) < discard_offset
    geometry_feedback = geometry_fn.split("terrain_vt_write_surface_feedback(", 1)[1]
    for argument in (
        "resolve_ddx_uv,",
        "resolve_ddy_uv,",
        "resolve_ddx_world,",
        "resolve_ddy_world,",
    ):
        assert argument in geometry_feedback

    # Mixed layer lock: a resident zero-roughness half and a missing half keep
    # the missing layer's weighted base roughness instead of double-blending 1.
    base_roughness = 0.8
    resident_roughness_sum = 0.5 * 0.0
    resident_coverage = 0.5
    resolved = base_roughness * (1.0 - resident_coverage) + resident_roughness_sum
    assert resolved == pytest.approx(0.4)

    # GridVertex UVs cover the closed [0,1] domain: uv=1 must address the last
    # texel/page, while repeating albedo alone wraps to zero.
    virtual_size = 1024.0
    normalized_grid_edge = min(1.0, 1.0 - 0.5 / virtual_size)
    assert int(normalized_grid_edge * virtual_size) == 1023
    assert int((1.0 % 1.0) * virtual_size) == 0


# ---------------------------------------------------------------------------
# GPU-gated DoD tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GPU_AVAILABLE, reason="requires GPU-backed forge3d runtime")
class TestTerrainVTPbrFamilies:
    @pytest.fixture(autouse=True)
    def _reset_vt_sources(self, vt_render_env):
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        yield
        renderer.clear_material_vt_sources()

    def test_normal_family_changes_lighting_ssim(self, vt_render_env) -> None:
        """Gated measurable win: the normal family must change grazing-light
        beauty output by SSIM difference > 0.05."""
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 2048

        grazing = dict(
            light_elevation_deg=7.0,
            sun_intensity=3.5,
            ibl_intensity=0.25,
            cam_radius=3.0,
        )

        baseline = _render_beauty(
            vt_render_env,
            _build_render_params(vt_settings=None, **grazing),
        )

        _register_family_sources(renderer, virtual_size, ("normal",))
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=2048,
            residency_budget_mb=192.0,
            max_mip_levels=6,
            layers=[
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(0.5, 0.5, 1.0, 1.0),
                )
            ],
        )
        with_normal = _render_beauty(
            vt_render_env,
            _build_render_params(vt_settings=vt_settings, **grazing),
        )
        stats = renderer.get_material_vt_stats()

        rows, cols = _region_slices(baseline.shape)
        ssim_value = _ssim(
            _luminance(baseline)[rows, cols],
            _luminance(with_normal)[rows, cols],
        )
        ssim_delta = 1.0 - ssim_value

        # Attribution: the only difference between the renders is the resident
        # normal family.
        assert stats["resident_tiles_normal"] > 0.0
        assert stats["resident_tiles_albedo"] == pytest.approx(0.0)
        assert stats["resident_tiles_mask"] == pytest.approx(0.0)
        assert ssim_delta > 0.05, (
            f"normal family must change grazing-light beauty output: "
            f"SSIM delta {ssim_delta:.4f} <= 0.05 (SSIM {ssim_value:.4f})"
        )

        actual_baseline = _as_rgba8(baseline)
        actual_normal = _as_rgba8(with_normal)
        golden_baseline = _golden(BASELINE_GOLDEN, actual_baseline)
        golden_normal = _golden(NORMAL_GOLDEN, actual_normal)
        assert actual_baseline.shape == golden_baseline.shape
        assert actual_normal.shape == golden_normal.shape
        golden_ssim_baseline = _ssim(
            _luminance(actual_baseline), _luminance(golden_baseline)
        )
        golden_ssim_normal = _ssim(
            _luminance(actual_normal), _luminance(golden_normal)
        )
        golden_error_baseline = float(
            np.mean(np.abs(actual_baseline.astype(np.float64) - golden_baseline)) / 255.0
        )
        golden_error_normal = float(
            np.mean(np.abs(actual_normal.astype(np.float64) - golden_normal)) / 255.0
        )
        assert golden_ssim_baseline >= 0.99
        assert golden_ssim_normal >= 0.99
        assert golden_error_baseline <= 0.01
        assert golden_error_normal <= 0.01

        record_substratia_image("actual_baseline.png", actual_baseline)
        record_substratia_image("actual_normal.png", actual_normal)
        record_substratia_image("golden_baseline.png", golden_baseline)
        record_substratia_image("golden_normal.png", golden_normal)
        record_substratia_result(
            "normal_lighting_ssim",
            {
                "status": "PASS",
                "ssim": ssim_value,
                "ssim_delta": ssim_delta,
                "threshold": 0.05,
                "region": list(GRAZING_REGION),
                "actual_baseline": "actual_baseline.png",
                "actual_normal": "actual_normal.png",
                "golden_baseline": "golden_baseline.png",
                "golden_normal": "golden_normal.png",
                "golden_ssim_baseline": golden_ssim_baseline,
                "golden_ssim_normal": golden_ssim_normal,
                "golden_mean_error_baseline": golden_error_baseline,
                "golden_mean_error_normal": golden_error_normal,
                "actual_baseline_rgba_sha256": image_sha256(actual_baseline),
                "actual_normal_rgba_sha256": image_sha256(actual_normal),
            },
        )

    def test_all_families_page_within_budget(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 2048
        budget_mb = 96.0

        _register_family_sources(renderer, virtual_size, ("albedo", "normal", "mask"))
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=2048,
            residency_budget_mb=budget_mb,
            max_mip_levels=6,
            layers=[
                VTLayerFamily(family="albedo", virtual_size_px=(virtual_size, virtual_size)),
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(0.5, 0.5, 1.0, 1.0),
                ),
                VTLayerFamily(
                    family="mask",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(1.0, 1.0, 1.0, 1.0),
                ),
            ],
        )

        # Capture two camera-dependent GPU demand sets without allowing CPU
        # prefetch or uploads to turn the proof into a prefilled/count-only
        # check. The blocking accessor contains only shader-emitted records.
        demand_sets = []
        demand_frame_stats = []
        for cam_target in ((-2.0, -2.0, 0.0), (2.0, 2.0, 0.0)):
            _render_beauty(
                vt_render_env,
                _build_render_params(
                    vt_settings=vt_settings,
                    cam_target=cam_target,
                    cam_radius=1.5,
                    prefetch_horizon_ms=0.0,
                    vt_upload_budget_bytes=1,
                ),
            )
            demand = {
                tuple(int(value) for value in record)
                for record in renderer.read_latest_vt_shader_feedback()
            }
            assert demand, f"camera target {cam_target} emitted no VT shader feedback"
            assert {record[0] for record in demand} == {0, 1, 2}
            assert all(len(record) == 5 for record in demand)
            demand_sets.append(demand)
            frame_stats = renderer.get_material_vt_stats()
            assert frame_stats["tiles_streamed"] == 0.0
            assert frame_stats["uploaded_bytes"] == 0.0
            for family in ("albedo", "normal", "mask"):
                assert frame_stats[f"resident_tiles_{family}"] == 0.0
                assert frame_stats[f"resident_bytes_{family}"] == 0.0
                assert frame_stats[f"feedback_tiles_streamed_{family}"] == 0.0
            demand_frame_stats.append(frame_stats)

        assert demand_sets[0] != demand_sets[1], (
            "camera sweep must change shader-feedback page identity"
        )
        assert any(
            {
                record[2:]
                for record in demand_sets[0]
                if record[0] == family_slot
            }
            != {
                record[2:]
                for record in demand_sets[1]
                if record[0] == family_slot
            }
            for family_slot in range(3)
        ), "at least one family must demand different pages after camera movement"

        retained_after_sweep = {
            tuple(int(value) for value in record)
            for record in renderer.read_retained_vt_requests()
        }
        assert demand_sets[1] <= retained_after_sweep, (
            "second-camera shader demand must enter the retained request set"
        )
        stats_before_upload = renderer.get_material_vt_stats()
        assert stats_before_upload["feedback_overflow"] == 0.0

        # Feedback emitted by a frame is consumed on the next frame. The native
        # counters below tag only exact keys admitted from retained feedback,
        # so overlapping CPU visible-rect requests cannot impersonate this
        # path. A 64 MiB frame permits at least 256 page attempts even on the
        # raw-atlas compatibility path.
        _render_beauty(
            vt_render_env,
            _build_render_params(
                vt_settings=vt_settings,
                cam_target=(2.0, 2.0, 0.0),
                cam_radius=1.5,
                prefetch_horizon_ms=0.0,
                vt_upload_budget_bytes=64 * MIB,
            ),
        )
        retained_after_upload = {
            tuple(int(value) for value in record)
            for record in renderer.read_retained_vt_requests()
        }
        stats = renderer.get_material_vt_stats()
        assert stats["feedback_overflow"] == 0.0
        assert stats["tiles_streamed"] > 0.0
        assert stats["uploaded_bytes"] > 0.0
        removed_demand_by_family = {}
        for family_slot, family in enumerate(("albedo", "normal", "mask")):
            family_demand = {
                record for record in demand_sets[1] if record[0] == family_slot
            }
            removed_demand = family_demand - retained_after_upload
            assert removed_demand, (
                f"family '{family}' processed no exact retained page; "
                f"removed={sorted(removed_demand)}"
            )
            assert stats[f"feedback_tiles_streamed_{family}"] > 0.0, (
                f"family '{family}' streamed no feedback-admitted page"
            )
            assert (
                stats[f"resident_tiles_{family}"]
                > stats_before_upload[f"resident_tiles_{family}"]
            ), f"family '{family}' shader demand did not advance residency"
            removed_demand_by_family[family] = len(removed_demand)

        budget_bytes = budget_mb * MIB
        resident_sum = 0.0
        budget_sum = 0.0
        for family in ("albedo", "normal", "mask"):
            assert stats[f"resident_bytes_{family}"] > 0.0, (
                f"family '{family}' paged no tiles during the sweep"
            )
            assert stats[f"resident_bytes_{family}"] <= stats[f"budget_bytes_{family}"]
            resident_sum += stats[f"resident_bytes_{family}"]
            budget_sum += stats[f"budget_bytes_{family}"]

        assert resident_sum == pytest.approx(stats["resident_bytes_total"])
        assert resident_sum <= budget_bytes
        assert budget_sum <= budget_bytes
        assert resident_sum <= MEMORY_BUDGET_LIMIT_BYTES
        assert budget_bytes <= MEMORY_BUDGET_LIMIT_BYTES
        registry_metrics = f3d.memory_metrics()
        for family in ("albedo", "normal", "mask"):
            assert registry_metrics[f"resident_bytes_{family}"] >= stats[
                f"resident_bytes_{family}"
            ]
        record_substratia_result(
            "family_residency_budget",
            {
                "status": "PASS",
                "resident_bytes": {
                    family: int(stats[f"resident_bytes_{family}"])
                    for family in ("albedo", "normal", "mask")
                },
                "family_budget_bytes": {
                    family: int(stats[f"budget_bytes_{family}"])
                    for family in ("albedo", "normal", "mask")
                },
                "total_resident_bytes": int(resident_sum),
                "configured_budget_bytes": int(budget_bytes),
                "memory_limit_bytes": MEMORY_BUDGET_LIMIT_BYTES,
                "camera_feedback_record_counts": [
                    len(demand) for demand in demand_sets
                ],
                "camera_feedback_sets_distinct": True,
                "retained_second_camera_requests": len(demand_sets[1]),
                "retained_after_upload": len(retained_after_upload),
                "removed_shader_demand_by_family": removed_demand_by_family,
                "feedback_tiles_streamed_by_family": {
                    family: int(stats[f"feedback_tiles_streamed_{family}"])
                    for family in ("albedo", "normal", "mask")
                },
                "tiles_streamed_demand_frames": [
                    int(frame_stats["tiles_streamed"])
                    for frame_stats in demand_frame_stats
                ],
                "tiles_streamed_upload_frame": int(stats["tiles_streamed"]),
            },
        )

    def test_missing_family_is_fatal(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 1024

        # Register albedo only, then request albedo + normal.
        _register_family_sources(renderer, virtual_size, ("albedo",))
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=2048,
            residency_budget_mb=32.0,
            max_mip_levels=4,
            layers=[
                VTLayerFamily(family="albedo", virtual_size_px=(virtual_size, virtual_size)),
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                    fallback=(0.5, 0.5, 1.0, 1.0),
                ),
            ],
        )

        with pytest.raises(
            RuntimeError,
            match=r"family 'normal' requested but no source registered",
        ) as raised:
            _render_beauty(
                vt_render_env,
                _build_render_params(vt_settings=vt_settings),
            )
        record_substratia_result(
            "missing_family_fatal",
            {"status": "PASS", "message": str(raised.value)},
        )

    def test_missing_family_offline_preflight_leaves_no_active_session(
        self, vt_render_env
    ) -> None:
        renderer, material_set, ibl, heightmap = vt_render_env
        virtual_size = 1024
        _register_family_sources(renderer, virtual_size, ("albedo",))
        missing_normal = TerrainVTSettings(
            enabled=True,
            atlas_size=2048,
            residency_budget_mb=32.0,
            max_mip_levels=4,
            layers=[
                VTLayerFamily(family="albedo", virtual_size_px=(virtual_size, virtual_size)),
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                ),
            ],
        )

        # begin_offline_accumulation is intentionally a no-render negative
        # control: rejection must happen before offline GPU state is built or
        # installed, so a valid session can begin immediately afterwards.
        with pytest.raises(
            RuntimeError,
            match=r"family 'normal' requested but no source registered",
        ):
            renderer.begin_offline_accumulation(
                material_set=material_set,
                env_maps=ibl,
                params=_build_render_params(vt_settings=missing_normal),
                heightmap=heightmap,
                water_mask=None,
            )

        renderer.begin_offline_accumulation(
            material_set=material_set,
            env_maps=ibl,
            params=_build_render_params(vt_settings=None),
            heightmap=heightmap,
            water_mask=None,
        )
        try:
            # Reaching an active valid session proves the rejected request did
            # not leave an installed state behind.
            with pytest.raises(RuntimeError, match="already active"):
                renderer.begin_offline_accumulation(
                    material_set=material_set,
                    env_maps=ibl,
                    params=_build_render_params(vt_settings=None),
                    heightmap=heightmap,
                    water_mask=None,
                )
        finally:
            renderer.end_offline_accumulation()

    def test_partial_normal_residency_degrades_gracefully(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 1024

        def normal_only_settings(budget_mb: float) -> TerrainVTSettings:
            # max_mip_levels=1 removes the coarse-mip rescue path so
            # non-resident tiles must fall back to the geometric normal;
            # use_feedback=False keeps the resident subset deterministic.
            return TerrainVTSettings(
                enabled=True,
                atlas_size=2048,
                residency_budget_mb=budget_mb,
                max_mip_levels=1,
                use_feedback=False,
                layers=[
                    VTLayerFamily(
                        family="normal",
                        virtual_size_px=(virtual_size, virtual_size),
                        # Hostile caller fallback: the shader must still use a
                        # flat tangent normal for nonresident pages.
                        fallback=(0.0, 0.0, 0.0, 1.0),
                    )
                ],
            )

        params_kwargs = dict(cam_radius=3.0, normal_aov=True)
        baseline_beauty, baseline_aov = _render_beauty_and_normal_aov(
            vt_render_env,
            _build_render_params(vt_settings=None, **params_kwargs),
        )

        _register_family_sources(renderer, virtual_size, ("normal",))
        full_beauty, full_aov = _render_beauty_and_normal_aov(
            vt_render_env,
            _build_render_params(vt_settings=normal_only_settings(64.0), **params_kwargs),
        )
        full_stats = renderer.get_material_vt_stats()

        partial_beauty, partial_aov = _render_beauty_and_normal_aov(
            vt_render_env,
            _build_render_params(vt_settings=normal_only_settings(1.0), **params_kwargs),
        )
        partial_stats = renderer.get_material_vt_stats()

        # The render completed with finite output.
        for image in (partial_beauty, partial_aov):
            assert np.all(np.isfinite(np.asarray(image, dtype=np.float64)))

        # Partial run holds strictly fewer normal tiles than the full run.
        assert partial_stats["resident_tiles_normal"] > 0.0
        assert (
            partial_stats["resident_tiles_normal"]
            < full_stats["resident_tiles_normal"]
        )

        # Identify fragments whose normal tiles fell back in the partial run:
        # their normal AOV matches the geometric baseline although the fully
        # resident render disagrees with it.
        aov_delta_partial = np.max(np.abs(partial_aov - baseline_aov), axis=-1)
        aov_delta_full = np.max(np.abs(full_aov - baseline_aov), axis=-1)
        fallback_region = (aov_delta_partial < 0.01) & (aov_delta_full > 0.03)
        assert fallback_region.mean() > 0.02, (
            "expected a visible region of non-resident normal tiles "
            f"(got {fallback_region.mean():.4f} coverage)"
        )

        # Graceful degradation: in the fallback region, beauty lighting matches
        # the geometric-normal baseline (no corrupted/black-normal lighting).
        baseline_lum = _luminance(baseline_beauty)
        partial_lum = _luminance(partial_beauty)
        region_diff = np.abs(partial_lum - baseline_lum)[fallback_region]
        assert float(region_diff.mean()) < 0.02, (
            f"fallback region deviates from geometric baseline: "
            f"mean {float(region_diff.mean()):.4f}"
        )
        # And nothing collapsed to black in the fallback region.
        assert float(partial_lum[fallback_region].min()) >= 0.0
        assert float(partial_lum[fallback_region].mean()) > 0.01
        record_substratia_result(
            "partial_normal_residency",
            {
                "status": "PASS",
                "fallback_coverage": float(fallback_region.mean()),
                "mean_luminance_error": float(region_diff.mean()),
                "error_threshold": 0.02,
            },
        )

    def test_unusable_family_source_is_fatal(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 512
        # Material 99 is outside the terrain material set and therefore cannot
        # satisfy material 0's requested normal family even though a normal
        # source exists elsewhere in the registry.
        renderer.register_material_vt_source(
            99,
            "normal",
            _build_bumpy_normal_source(virtual_size, 0),
            (virtual_size, virtual_size),
            [0.5, 0.5, 1.0, 1.0],
        )
        settings = TerrainVTSettings(
            enabled=True,
            atlas_size=1024,
            residency_budget_mb=8.0,
            layers=[
                VTLayerFamily(
                    family="normal",
                    virtual_size_px=(virtual_size, virtual_size),
                )
            ],
        )
        with pytest.raises(
            RuntimeError,
            match=r"family 'normal' requested but no source registered for material 0",
        ):
            _render_beauty(
                vt_render_env,
                _build_render_params(vt_settings=settings),
            )

    def test_partial_mask_residency_uses_neutral_fallback(self, vt_render_env) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 1024
        baseline = _render_beauty(
            vt_render_env,
            _build_render_params(vt_settings=None, cam_radius=3.0),
        )

        destructive_mask = np.full(
            (virtual_size, virtual_size, 4),
            [255, 0, 0, 255],
            dtype=np.uint8,
        )
        for material_index in range(VT_MATERIAL_COUNT):
            renderer.register_material_vt_source(
                material_index,
                "mask",
                destructive_mask,
                (virtual_size, virtual_size),
                [1.0, 1.0, 1.0, 1.0],
            )

        def settings(atlas_size: int, budget_mb: float) -> TerrainVTSettings:
            return TerrainVTSettings(
                enabled=True,
                atlas_size=atlas_size,
                residency_budget_mb=budget_mb,
                max_mip_levels=1,
                use_feedback=False,
                layers=[
                    VTLayerFamily(
                        family="mask",
                        virtual_size_px=(virtual_size, virtual_size),
                        tile_size=120,
                        tile_border=4,
                        # Hostile caller fallback: missing mask pages must
                        # preserve current material defaults unconditionally.
                        fallback=(0.0, 0.0, 0.0, 1.0),
                    )
                ],
            )

        partial = _render_beauty(
            vt_render_env,
            _build_render_params(
                vt_settings=settings(256, 0.25),
                cam_radius=3.0,
                vt_upload_budget_bytes=256 * 1024,
            ),
        )
        partial_stats = renderer.get_material_vt_stats()
        full = _render_beauty(
            vt_render_env,
            _build_render_params(
                vt_settings=settings(2048, 64.0),
                cam_radius=3.0,
                vt_upload_budget_bytes=64 * 1024 * 1024,
            ),
        )
        full_stats = renderer.get_material_vt_stats()

        assert 0.0 < partial_stats["resident_tiles_mask"] < full_stats["resident_tiles_mask"]
        baseline_lum = _luminance(baseline)
        partial_lum = _luminance(partial)
        full_lum = _luminance(full)
        partial_delta = np.abs(partial_lum - baseline_lum)
        full_delta = np.abs(full_lum - baseline_lum)
        fallback_region = (partial_delta < (2.0 / 255.0)) & (full_delta > (5.0 / 255.0))
        fallback_pixels = int(np.count_nonzero(fallback_region))
        assert fallback_pixels >= 64, (
            "partial residency must expose a meaningful absolute region that "
            "uses the neutral mask fallback"
        )
        assert float(partial_delta[fallback_region].max()) < (2.0 / 255.0)
        assert float(full_delta[fallback_region].min()) > (5.0 / 255.0)
        assert float(partial_delta.mean()) <= float(full_delta.mean())
        assert float(partial_lum.mean()) > 0.01
        record_substratia_result(
            "partial_mask_residency",
            {
                "status": "PASS",
                "fallback_coverage": float(fallback_region.mean()),
                "fallback_pixels": fallback_pixels,
                "partial_mean_luminance_error": float(partial_delta.mean()),
                "full_mean_luminance_error": float(full_delta.mean()),
                "partial_resident_tiles": int(partial_stats["resident_tiles_mask"]),
                "full_resident_tiles": int(full_stats["resident_tiles_mask"]),
            },
        )

    @pytest.mark.parametrize("shading", ["forward", "visibility"])
    def test_gpu_shader_feedback_preserves_family_coordinates(
        self,
        vt_render_env,
        shading: str,
    ) -> None:
        renderer = vt_render_env[0]
        renderer.clear_material_vt_sources()
        virtual_size = 1024
        _register_family_sources(renderer, virtual_size, ("albedo", "normal", "mask"))
        vt_settings = TerrainVTSettings(
            enabled=True,
            atlas_size=512,
            residency_budget_mb=0.75,
            max_mip_levels=4,
            use_feedback=True,
            layers=[
                VTLayerFamily(
                    family=family,
                    virtual_size_px=(virtual_size, virtual_size),
                    tile_size=120,
                    tile_border=4,
                )
                for family in ("albedo", "normal", "mask")
            ],
        )
        if shading == "visibility":
            # Reuse the repository's physically proven clipmap camera contract;
            # visibility shading is only active for clipmap geometry. The
            # one-byte upload cap guarantees that every shader-visible request
            # is a genuine miss rather than a CPU-prefetch record.
            from test_terrain_clipmap_streaming import _steep_dem

            visibility_size = (160, 90)
            visibility_span = 100_000.0
            visibility_radius = 50_000.0
            visibility_fov_y_deg = 45.0
            # At the target plane this view covers more than two virtual texels
            # per pixel in both axes. That makes a nonzero Grid-UV mip an
            # intentional fixture precondition rather than an assumption about
            # backend derivative behaviour.
            grid_texels_per_pixel = (
                2.0
                * visibility_radius
                * math.tan(math.radians(visibility_fov_y_deg) * 0.5)
                / visibility_size[1]
                * virtual_size
                / visibility_span
            )
            assert grid_texels_per_pixel > 2.0

            render_config = make_terrain_params_config(
                size_px=visibility_size,
                render_scale=1.0,
                terrain_span=visibility_span,
                msaa_samples=1,
                z_scale=1.2,
                exposure=1.0,
                domain=(0.0, 1.0),
                albedo_mode="colormap",
                colormap_strength=1.0,
                ibl_enabled=True,
                light_azimuth_deg=138.0,
                light_elevation_deg=24.0,
                sun_intensity=2.4,
                cam_radius=visibility_radius,
                cam_phi_deg=28.0,
                cam_theta_deg=10.0,
                cam_target=(0.0, 0.0, 0.0),
                fov_y_deg=visibility_fov_y_deg,
                camera_mode="clipmap:4:32:32:10:0.3",
                culling="frustum",
                shading="visibility",
                vt=vt_settings,
                vt_upload_budget_bytes=1,
                clip=(0.1, 150_000.0),
                overlays=[_build_overlay()],
                pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            )
            render_params = f3d.TerrainRenderParams(render_config)
            render_env = (*vt_render_env[:3], _steep_dem(96))
        else:
            render_params = _build_render_params(
                vt_settings=vt_settings,
                size_px=(96, 72),
                shading="forward",
                vt_upload_budget_bytes=1024 * 1024,
            )
            render_env = vt_render_env
        if shading == "visibility":
            material_set, ibl, heightmap = render_env[1:]
            with pytest.raises(
                RuntimeError, match="does not support shading='visibility'"
            ):
                renderer.render_with_aov(
                    material_set=material_set,
                    env_maps=ibl,
                    params=render_params,
                    heightmap=heightmap,
                )
            frame = renderer.render_terrain_pbr_pom(
                material_set=material_set,
                env_maps=ibl,
                params=render_params,
                heightmap=heightmap,
                target=None,
                water_mask=None,
            )
            image = np.asarray(frame.to_numpy())
            assert _is_meaningful_render(image), (
                "visibility terrain render returned a magenta/empty marker"
            )
            resolve_stats = visibility_stats()
            assert resolve_stats["visible_pixels"] > 0
            assert (
                resolve_stats["visible_pixels"]
                + resolve_stats["background_pixels"]
                == visibility_size[0] * visibility_size[1]
            )
            assert (
                resolve_stats["visibility_feedback_records"]
                == resolve_stats["visible_pixels"]
            )
            assert (
                resolve_stats["material_invocations"]
                == resolve_stats["visible_pixels"]
            )
        else:
            _render_beauty(render_env, render_params)
        records = [tuple(int(value) for value in record) for record in renderer.read_latest_vt_shader_feedback()]
        assert records, f"{shading} produced no raw VT shader feedback"
        assert all(len(record) == 5 for record in records)
        pair_set = {(family, material) for family, material, _, _, _ in records}
        assert pair_set == {
            (family, material)
            for family in range(3)
            for material in range(VT_MATERIAL_COUNT)
        }
        by_family = {
            family: {
                (material, mip, tile_x, tile_y)
                for record_family, material, mip, tile_x, tile_y in records
                if record_family == family
            }
            for family in range(3)
        }
        assert by_family[1] and by_family[2]
        assert by_family[0] != by_family[1], "albedo feedback must use triplanar coordinates"
        assert all(0 <= mip < 4 for _, _, mip, _, _ in records)
        base_pages = math.ceil(virtual_size / 120)
        for _, _, mip, tile_x, tile_y in records:
            divisor = 1 << mip
            pages_at_mip = max((base_pages + divisor - 1) // divisor, 1)
            assert 0 <= tile_x < pages_at_mip
            assert 0 <= tile_y < pages_at_mip
        mip_sets = {
            family: {mip for _, mip, _, _ in family_records}
            for family, family_records in by_family.items()
        }
        assert all(any(mip > 0 for mip in mips) for mips in mip_sets.values()), (
            "every family must preserve a nonzero shader-selected mip; "
            f"observed {mip_sets}"
        )
        record_substratia_result(
            f"shader_feedback_{shading}",
            {
                "status": "PASS",
                "record_count": len(records),
                "family_material_pairs": len(pair_set),
                "normal_mask_shared_grid_mapping_source_locked": True,
                "albedo_triplanar_coordinates_distinct": True,
                "mip_sets": {str(family): sorted(mips) for family, mips in mip_sets.items()},
            },
        )
