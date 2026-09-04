"""AETHER's exact typed LUT handoff into active renderer consumers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE, get_native_module
from forge3d.atmosphere import AtmosphereSettings
from forge3d.terrain_params import (
    PomSettings,
    ShadowSettings,
    SkySettings,
    make_terrain_params_config,
)


ROOT = Path(__file__).resolve().parents[1]


class _HandleLike:
    turbidity = 4.0
    ozone_du = 321.0
    mie_g = 0.71
    ground_albedo = 0.42


def _terrain_config(sky: object):
    return make_terrain_params_config(
        size_px=(64, 64),
        render_scale=1.0,
        terrain_span=10.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        sky=sky,  # type: ignore[arg-type]
    )


def test_atmosphere_settings_carries_ground_albedo_to_native_kwargs() -> None:
    settings = AtmosphereSettings(ground_albedo=0.42)
    assert settings.native_kwargs()["ground_albedo"] == pytest.approx(0.42)
    with pytest.raises(ValueError, match="ground_albedo"):
        AtmosphereSettings(ground_albedo=float("nan"))


def test_sky_settings_omitted_physical_values_adopt_handle_values() -> None:
    settings = SkySettings(model="aether", lut_handle=_HandleLike())
    assert settings.turbidity == 4.0
    assert settings.ozone_du == 321.0
    assert settings.mie_g == pytest.approx(0.71)
    assert settings.ground_albedo == pytest.approx(0.42)
    assert all(
        type(getattr(settings, name)) is float
        for name in ("turbidity", "ozone_du", "mie_g", "ground_albedo")
    )
    serialized = asdict(settings)
    assert serialized["turbidity"] == 4.0
    assert serialized["lut_handle"] is not None


def test_sky_settings_rejects_explicit_handle_scalar_conflict() -> None:
    with pytest.raises(ValueError, match="turbidity=.*does not match"):
        SkySettings(model="aether", lut_handle=_HandleLike(), turbidity=3.0)
    with pytest.raises(ValueError, match="requires model='aether'"):
        SkySettings(lut_handle=_HandleLike())


def test_sky_settings_compares_explicit_values_at_native_f32_precision() -> None:
    matching = _HandleLike()
    matching.turbidity = float(np.float32(2.1))
    settings = SkySettings(model="aether", lut_handle=matching, turbidity=2.1)
    assert settings.turbidity == float(np.float32(2.1))


def test_active_consumers_are_typed_to_the_exact_handle() -> None:
    terrain = (ROOT / "src/terrain/renderer/atmosphere.rs").read_text()
    terrain_luts = (ROOT / "src/terrain/renderer/atmosphere/luts.rs").read_text()
    prometheus = (ROOT / "src/path_tracing/hybrid_compute/aether_post.rs").read_text()
    prometheus_shader = (
        ROOT / "src/shaders/atmosphere/prometheus_aerial.wgsl"
    ).read_text()
    evaluation_core = (
        ROOT / "src/shaders/atmosphere/evaluation_core.wgsl"
    ).read_text()
    desc = (ROOT / "src/path_tracing/hybrid_compute/render_terrain.rs").read_text()
    assert "lut_handle.deterministic_sha256()" in terrain
    assert "lut_handle.luts()" in prometheus
    assert "Option<crate::core::atmosphere::AtmosphereLutHandle>" in desc
    assert "load_precomputed_atmosphere_luts" not in prometheus
    assert "tracked_lut_upload_bytes" in terrain_luts
    assert "tracked_lut_upload_bytes" in prometheus
    assert "prometheus_reference_sky" not in prometheus_shader
    miss_branch = prometheus_shader.split("if (visibility < 0.5)", 1)[1].split(
        "return;", 1
    )[0]
    assert (
        "let miss_scattering = prometheus_load_endpoint_scattering(\n"
        "            camera_height_unit,\n"
        "            sun_dir.y,\n"
        "            ray.y,\n"
        "            dot(ray, sun_dir),\n"
        "        ) * sun_intensity;"
        in miss_branch
    )
    assert (
        "aether_eval_clamp_hdr_radiance(miss_scattering) * atmosphere_exposure"
        in miss_branch
    )
    assert "let sun_intensity = aether_eval_clamp_radiometric_scale(" in prometheus_shader
    assert "let atmosphere_exposure = aether_eval_clamp_radiometric_scale(" in prometheus_shader
    assert prometheus.count(".clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX)") >= 2
    assert "aether_eval_sample_accumulated_scattering(" in prometheus_shader
    assert "aether_eval_segment_transmittance(" in prometheus_shader
    assert "fn aether_eval_load_scattering_texel" in evaluation_core
    assert "let fraction = fract(coordinates);" in evaluation_core
    assert "height_side < 2" in evaluation_core
    assert "nu_side < 2" in evaluation_core
    assert "sun_side < 2" in evaluation_core
    assert "view_side < 2" in evaluation_core
    assert "fn prometheus_load_scattering_texel" not in prometheus_shader
    assert "AETHER_PT_CIE_XYZ" not in prometheus_shader
    assert "prometheus_load_boundary_mean_transmittance" not in prometheus_shader
    assert "scatter_fraction" not in prometheus_shader
    assert (
        "camera_scattering - transmittance * endpoint_scattering,\n"
        "        vec3<f32>(0.0),"
        in prometheus_shader
    )


def test_lut_payload_and_physical_metadata_are_immutable_and_origin_sealed() -> None:
    bake = (ROOT / "src/core/atmosphere/bake.rs").read_text(encoding="utf-8")
    runtime = (ROOT / "src/core/atmosphere/runtime.rs").read_text(encoding="utf-8")
    assert "pub(crate) metadata: AtmosphereLutMetadata" in bake
    assert "pub(crate) texels: Vec<f16>" in bake
    assert "pub metadata: AtmosphereLutMetadata" not in bake
    assert "pub texels: Vec<f16>" not in bake
    assert "AETHER-LUT-v2\\0" in bake
    assert "_sealed_deterministic_sha256: [u8; 32]" in bake
    assert "self._sealed_deterministic_sha256 = self.deterministic_sha256();" in bake
    assert "deterministic_sha256 != luts.sealed_deterministic_sha256()" in runtime
    assert "refusing relabeled provenance" in runtime


@pytest.fixture(scope="module")
def native():
    if not NATIVE_AVAILABLE:
        pytest.skip("typed handoff tests require the compiled native extension")
    return get_native_module()


@pytest.fixture(scope="module")
def shipped_handle(native):
    handle = native.atmosphere_bake_luts()
    assert isinstance(handle, native.AtmosphereLutHandle)
    return handle


@pytest.fixture(scope="module")
def interpolated_handle(native):
    return native.atmosphere_bake_luts(turbidity=2.1)


@pytest.fixture(scope="module")
def custom_handle(native):
    try:
        return native.atmosphere_bake_luts(
            turbidity=4.0,
            ozone_du=321.0,
            mie_g=0.71,
            ground_albedo=0.42,
            scattering_orders=2,
        )
    except RuntimeError as error:
        if "atmosphere-bake" in str(error):
            pytest.skip("custom handoff requires the atmosphere-bake feature")
        raise


def test_handle_preserves_report_compatibility_and_copying(shipped_handle) -> None:
    assert shipped_handle["precomputed"] is True
    assert shipped_handle["ground_albedo"] == pytest.approx(0.3)
    assert shipped_handle.deterministic_sha256 == shipped_handle["deterministic_sha256"]
    assert len(shipped_handle.deterministic_sha256) == 64
    report = shipped_handle.as_dict()
    assert report["byte_size"] == shipped_handle.byte_size

    settings = SkySettings(model="aether", lut_handle=shipped_handle)
    serialized = asdict(settings)
    assert isinstance(serialized["lut_handle"], type(shipped_handle))
    assert 42 not in shipped_handle


def test_handle_is_factory_only(native) -> None:
    with pytest.raises(RuntimeError, match="returned by atmosphere_bake_luts"):
        native.AtmosphereLutHandle()


def test_terrain_decoder_accepts_exact_handle(shipped_handle) -> None:
    settings = SkySettings(enabled=True, model="aether", lut_handle=shipped_handle)
    f3d.TerrainRenderParams(_terrain_config(settings))


def test_actual_handle_sky_normalization_and_mismatch(interpolated_handle) -> None:
    settings = SkySettings(
        model="aether",
        lut_handle=interpolated_handle,
        turbidity=2.1,
    )
    assert settings.turbidity == interpolated_handle.turbidity
    with pytest.raises(ValueError, match="turbidity=.*does not match"):
        SkySettings(
            model="aether",
            lut_handle=interpolated_handle,
            turbidity=2.2,
        )


def test_custom_bake_handle_reaches_terrain_decoder(custom_handle) -> None:
    assert custom_handle.precomputed is False
    settings = SkySettings(model="aether", lut_handle=custom_handle)
    assert settings.ozone_du == custom_handle.ozone_du
    assert settings.mie_g == custom_handle.mie_g
    assert settings.ground_albedo == custom_handle.ground_albedo
    f3d.TerrainRenderParams(_terrain_config(settings))
    # Native callers are not required to materialize duplicate scalar fields:
    # omission adopts the exact physical configuration owned by the handle.
    handle_only = SimpleNamespace(model="aether", lut_handle=custom_handle)
    f3d.TerrainRenderParams(_terrain_config(handle_only))


def test_terrain_decoder_rejects_wrong_type_and_getter_failures(native) -> None:
    wrong_type = SimpleNamespace(model="hosek-wilkie", turbidity="not-a-number")
    with pytest.raises(TypeError):
        native.TerrainRenderParams(_terrain_config(wrong_type))

    class GetterFailure:
        model = "hosek-wilkie"

        @property
        def turbidity(self):
            raise RuntimeError("authoritative turbidity getter failed")

    with pytest.raises(RuntimeError, match="authoritative turbidity getter failed"):
        native.TerrainRenderParams(_terrain_config(GetterFailure()))


def test_terrain_decoder_rejects_handle_scalar_conflict(shipped_handle, native) -> None:
    conflicting = SimpleNamespace(
        model="aether",
        lut_handle=shipped_handle,
        turbidity=3.0,
    )
    with pytest.raises(ValueError, match="does not match the exact LUT handle"):
        native.TerrainRenderParams(_terrain_config(conflicting))


def test_custom_scalar_without_handle_fails_closed_in_both_consumers(native) -> None:
    custom = SimpleNamespace(model="aether", ozone_du=301.0)
    with pytest.raises(RuntimeError, match="no nearby or legacy LUT was substituted"):
        native.TerrainRenderParams(_terrain_config(custom))

    with pytest.raises(RuntimeError, match="no nearby or default LUT was substituted"):
        native.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            1,
            1,
            {},
            atmosphere={"ozone_du": 301.0},
        )


def test_prometheus_rejects_handle_scalar_conflict_before_gpu(shipped_handle, native) -> None:
    with pytest.raises(ValueError, match="does not match the exact LUT handle"):
        native.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            1,
            1,
            {},
            atmosphere={"lut_handle": shipped_handle, "turbidity": 3.0},
        )


@pytest.mark.parametrize("unsupported", [object(), 42])
def test_prometheus_rejects_unrecognized_atmosphere_shape(native, unsupported) -> None:
    with pytest.raises(TypeError, match="recognized AETHER settings"):
        native.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            1,
            1,
            {},
            atmosphere=unsupported,
        )


@pytest.mark.parametrize("unknown", ["turbidty", "ozone", "lut_handles"])
def test_prometheus_rejects_unknown_mapping_keys_before_gpu(native, unknown) -> None:
    with pytest.raises(ValueError, match="unknown atmosphere setting"):
        native.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            1,
            1,
            {},
            atmosphere={unknown: 2.0},
        )


def test_handle_reaches_live_terrain_renderer(custom_handle, tmp_path) -> None:
    from _terrain_runtime import _write_test_hdr, terrain_rendering_available

    if not terrain_rendering_available():
        pytest.skip("live TerrainRenderer handoff requires a terrain-safe GPU")

    hdr_path = tmp_path / "handoff.hdr"
    _write_test_hdr(hdr_path)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=0.2)
    shadows = ShadowSettings(
        enabled=False,
        technique="NONE",
        resolution=512,
        cascades=1,
        max_distance=40.0,
        softness=0.0,
        intensity=0.0,
        slope_scale_bias=0.0,
        depth_bias=0.0,
        normal_bias=0.0,
        min_variance=0.0,
        light_bleed_reduction=0.0,
        evsm_exponent=1.0,
        fade_start=1.0,
    )
    params = f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=(64, 64),
            render_scale=1.0,
            terrain_span=3.0,
            msaa_samples=1,
            z_scale=0.1,
            exposure=1.0,
            domain=(0.0, 1.0),
            ibl_enabled=False,
            camera_mode="mesh:zup",
            shadows=shadows,
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            sky=SkySettings(
                enabled=True,
                model="aether",
                lut_handle=custom_handle,
            ),
        )
    )
    frame = renderer.render_terrain_pbr_pom(
        material,
        ibl,
        params,
        np.zeros((8, 8), dtype=np.float32),
    )
    assert np.asarray(frame.to_numpy()).shape == (64, 64, 4)
