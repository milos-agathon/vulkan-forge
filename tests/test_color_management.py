from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available


ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "src" / "shaders"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _linear_to_srgb(channel: float) -> float:
    channel = min(1.0, max(0.0, float(channel)))
    if channel <= 0.0031308:
        return channel * 12.92
    return 1.055 * channel ** (1.0 / 2.4) - 0.055


def _quantize_unorm8(channel: float) -> int:
    return int(round(min(1.0, max(0.0, float(channel))) * 255.0))


def _tonemap_reinhard(channel: float) -> float:
    return channel / (1.0 + channel)


def _tonemap_reinhard_extended(channel: float, white_point: float) -> float:
    white_sq = max(white_point * white_point, 1.0e-6)
    return channel * (1.0 + channel / white_sq) / (1.0 + channel)


def _tonemap_aces(channel: float) -> float:
    channel = max(channel, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return min(1.0, max(0.0, (channel * (channel * a + b)) / (channel * (channel * c + d) + e)))


def _tonemap_uncharted2_partial(channel: float) -> float:
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((channel * (channel * a + c * b) + d * e) / (channel * (channel * a + b) + d * f)) - e / f


def _tonemap_uncharted2(channel: float, white_point: float) -> float:
    current = _tonemap_uncharted2_partial(max(channel, 0.0))
    white_scale = 1.0 / max(_tonemap_uncharted2_partial(max(white_point, 1.0e-3)), 1.0e-6)
    return min(1.0, max(0.0, current * white_scale))


def _tonemap_exposure(channel: float) -> float:
    return 1.0 - math.exp(-max(channel, 0.0))


def _tonemap_filmic_terrain(channel: float) -> float:
    a, b, c, d, e, f, w = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30, 11.2
    x = max(channel, 0.0)
    curve = ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
    white_curve = ((w * (a * w + c * b) + d * e) / (w * (a * w + b) + d * f)) - e / f
    return min(1.0, max(0.0, curve / max(white_curve, 1.0e-6)))


def _tonemap_apply_operator(channel: float, operator_index: int, white_point: float) -> float:
    if operator_index == 0:
        return _tonemap_reinhard(channel)
    if operator_index == 1:
        return _tonemap_reinhard_extended(channel, white_point)
    if operator_index == 2:
        return _tonemap_aces(channel)
    if operator_index == 3:
        return _tonemap_uncharted2(channel, white_point)
    if operator_index == 4:
        return _tonemap_exposure(channel)
    if operator_index == 5:
        return _tonemap_filmic_terrain(channel)
    return _tonemap_reinhard(channel)


def test_tonemap_common_locks_operator_ids() -> None:
    source = _read(SHADERS / "includes" / "tonemap_common.wgsl")
    expected = {
        "TONEMAP_OPERATOR_REINHARD: u32 = 0u",
        "TONEMAP_OPERATOR_REINHARD_EXTENDED: u32 = 1u",
        "TONEMAP_OPERATOR_ACES: u32 = 2u",
        "TONEMAP_OPERATOR_UNCHARTED2: u32 = 3u",
        "TONEMAP_OPERATOR_EXPOSURE: u32 = 4u",
        "TONEMAP_OPERATOR_FILMIC_TERRAIN: u32 = 5u",
    }
    for token in expected:
        assert token in source
    assert "fn tonemap_apply_operator" in source
    assert "fn linear_to_srgb" in source


def test_gpu_tonemap_loaders_use_common_source() -> None:
    common = "includes/tonemap_common.wgsl"
    loaders = {
        ROOT / "src" / "pipeline" / "hdr_offscreen" / "pipeline.rs": common,
        ROOT / "src" / "terrain" / "renderer" / "offline.rs": common,
        ROOT / "src" / "terrain" / "renderer" / "pipeline_cache.rs": "shader_sources::terrain",
        ROOT / "src" / "pipeline" / "pbr" / "rendering.rs": "shader_sources::pbr",
        ROOT / "src" / "pipeline" / "pbr" / "tone_mapping.rs": common,
    }
    for path, needle in loaders.items():
        assert needle in _read(path), f"{path} does not load shared tonemap WGSL"

    centralized = _read(ROOT / "src" / "shader_sources.rs")
    pbr_loader = centralized.split("fn pbr()", 1)[1].split("#[cfg(test)]", 1)[0]
    assert common in pbr_loader, "shader_sources::pbr omits shared tonemap WGSL"


def test_cpu_adjudication_resolver_has_an_explicit_matching_contract() -> None:
    source = _read(ROOT / "src" / "core" / "tonemap.rs")

    assert "includes/tonemap_common.wgsl" not in source
    assert "resolve_reference_hdr_to_rgba8" in source
    assert "let t = x / (1.0 + x);" in source
    assert "12.92 * c" in source
    assert "1.055 * c.powf(1.0 / 2.4) - 0.055" in source


def test_duplicate_tonemap_bodies_were_removed() -> None:
    assert not (SHADERS / "tonemap.wgsl").exists()
    for shader_name in ("postprocess_tonemap.wgsl", "tonemap_terrain_offline.wgsl"):
        source = _read(SHADERS / shader_name)
        assert "fn aces_tonemap" not in source
        assert "fn reinhard_tonemap" not in source
        assert "fn uncharted2_tonemap" not in source
        assert "fn exposure_tonemap" not in source
    assert "fn tonemap_filmic_terrain" not in _read(SHADERS / "terrain_pbr_pom.wgsl")


def test_pbr_srgb_target_has_no_manual_output_gamma() -> None:
    source = _read(SHADERS / "pbr.wgsl")
    assert "RGBA8UnormSrgb" in source
    assert "pow(color, vec3<f32>(1.0 / lighting.gamma))" not in source
    assert "color = tonemap_reinhard(color);" in source


def test_linear_half_gray_is_encoded_once_not_twice() -> None:
    encoded_once = _linear_to_srgb(0.5)
    encoded_twice = _linear_to_srgb(encoded_once)

    assert encoded_once == pytest.approx(0.7353569, abs=1.0e-7)
    assert _quantize_unorm8(encoded_once) == 188
    assert _quantize_unorm8(encoded_twice) == 223


def test_offline_and_postfx_color_contract_match_within_one_lsb() -> None:
    if not terrain_rendering_available():
        pytest.skip("native offline tonemap comparison requires a terrain-capable GPU runtime")

    samples = np.array([0.0, 0.0031308, 0.18, 0.5, 1.0, 2.0, 8.0], dtype=np.float32)
    rgb = np.stack([samples, samples * 0.5, samples * 1.25], axis=1)
    hdr = np.ones((1, len(samples), 4), dtype=np.float32)
    hdr[0, :, :3] = rgb

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    hdr_frame = renderer.upload_hdr_frame(hdr, (len(samples), 1))
    offline_frame = renderer.tonemap_offline_hdr(hdr_frame)
    offline_bytes = np.asarray(offline_frame.to_numpy(), dtype=np.uint8)[0, :, :3]

    expected = np.zeros_like(offline_bytes)
    for pixel_index in range(rgb.shape[0]):
        for channel_index in range(3):
            tonemapped = _tonemap_apply_operator(float(rgb[pixel_index, channel_index]), 5, white_point=4.0)
            expected[pixel_index, channel_index] = _quantize_unorm8(_linear_to_srgb(tonemapped))

    assert np.max(np.abs(offline_bytes.astype(np.int16) - expected.astype(np.int16))) <= 1


def test_core_tonemap_has_no_inline_compute_tonemapper() -> None:
    source = _read(ROOT / "src" / "core" / "tonemap.rs")

    assert "create_compute_effect" not in source
    assert "Simple Reinhard tone mapping" not in source
    assert "Gamma correction (sRGB approximation)" not in source


def test_tonemap_operator_ids_are_unified_in_rust() -> None:
    pbr = _read(ROOT / "src" / "pipeline" / "pbr" / "tone_mapping.rs")
    assert "ToneMappingMode::Aces => 2" in pbr
    assert "ToneMappingMode::Reinhard => 0" in pbr
    assert "ToneMappingMode::Hable => 3" in pbr

    terrain = _read(ROOT / "src" / "terrain" / "render_params" / "native_postfx" / "tonemap.rs")
    assert '"aces" => 2' in terrain
    assert '"reinhard" => 0' in terrain
    assert '"uncharted2" => 3' in terrain
