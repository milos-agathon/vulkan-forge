"""AETHER spectral-basis, public API, and fail-closed Scene contracts."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d import atmosphere
from forge3d._native import NATIVE_AVAILABLE
from forge3d.terrain_params import SkySettings


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def default_lut_report() -> dict:
    if not NATIVE_AVAILABLE:
        pytest.skip("native AETHER contract requires extension")
    return f3d.atmosphere_bake_luts()


def test_python_helper_exposes_validated_settings_and_canonical_sweep() -> None:
    settings = atmosphere.AtmosphereSettings()
    assert settings.native_kwargs() == {
        "turbidity": 2.0,
        "ozone_du": 300.0,
        "mie_g": 0.8,
        "ground_albedo": 0.3,
        "scattering_orders": 4,
    }
    assert atmosphere.SUN_ELEVATION_SWEEP_DEG == (
        -5.0,
        0.0,
        5.0,
        10.0,
        30.0,
        60.0,
        89.0,
    )
    assert f3d.SUN_ELEVATION_SWEEP_DEG == atmosphere.SUN_ELEVATION_SWEEP_DEG


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"turbidity": 0.99}, "turbidity"),
        ({"ozone_du": 601.0}, "ozone_du"),
        ({"mie_g": 1.0}, "mie_g"),
        ({"scattering_orders": 1}, "scattering_orders"),
    ],
)
def test_python_helper_rejects_invalid_settings(kwargs: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        atmosphere.AtmosphereSettings(**kwargs)


def test_validation_environment_rejects_unrepresented_scattering_order() -> None:
    settings = atmosphere.AtmosphereSettings(scattering_orders=2)
    with pytest.raises(ValueError, match="requires scattering_orders=4"):
        atmosphere.generate_environment(8, 4, 10.0, settings=settings)


def test_auxiliary_cpu_oracle_is_directional_and_production_independent() -> None:
    oracle = (ROOT / "tests" / "_aether_pt_oracle.py").read_text(encoding="utf-8")
    assert "import forge3d" not in oracle
    assert "atmosphere_generate_environment" not in oracle
    assert "atmosphere_bake_luts" not in oracle
    assert "REFERENCE_STEPS = 64" in oracle
    assert "local_mu_sun = float(np.dot(sun, sample_position / sample_radius))" in oracle
    assert "def _attenuated_cell_length(" in oracle
    assert "-np.expm1(-extinction[active] * step_m)" in oracle
    assert "view_start = _transmittance(view_columns, turbidity)" in oracle
    assert "* _attenuated_cell_length(density, step_m, turbidity)" in oracle

    from _aether_pt_oracle import independent_reference_environment

    environment = independent_reference_environment(10.0, width=2, height=3)
    assert environment.shape == (3, 2, 3)
    assert np.all(np.isfinite(environment)) and np.all(environment >= 0.0)
    assert np.unique(environment.reshape(-1, 3), axis=0).shape[0] > 1


def test_terrain_sky_settings_exposes_aether_parameters() -> None:
    settings = SkySettings(
        enabled=True,
        model="aether",
        turbidity=4.0,
        ozone_du=275.0,
        mie_g=0.72,
    )
    assert settings.model == "aether"
    assert settings.turbidity == 4.0
    assert settings.ozone_du == 275.0
    assert settings.mie_g == pytest.approx(0.72)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "spectral-ish"}, "model"),
        ({"model": "aether", "turbidity": float("nan")}, "turbidity"),
        ({"model": "aether", "ground_albedo": float("nan")}, "ground_albedo"),
        ({"model": "aether", "ozone_du": float("nan")}, "ozone_du"),
        ({"model": "aether", "mie_g": -0.01}, "mie_g"),
        ({"model": "aether", "aerial_density": float("nan")}, "aerial_density"),
        ({"model": "aether", "aerial_density": 10.01}, "aerial_density"),
        ({"model": "aether", "sun_intensity": float("inf")}, "sun_intensity"),
        ({"model": "aether", "sun_size": float("nan")}, "sun_size"),
        ({"model": "aether", "sky_exposure": float("inf")}, "sky_exposure"),
    ],
)
def test_terrain_sky_settings_rejects_invalid_aether_inputs(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SkySettings(**kwargs)


def test_public_sources_and_stubs_lock_aether_surface() -> None:
    expected_functions = {
        "atmosphere_bake_luts",
        "atmosphere_spectral_to_linear_rgb",
        "atmosphere_generate_environment",
        "atmosphere_reference_aerial",
    }
    init_source = (ROOT / "python" / "forge3d" / "__init__.py").read_text(
        encoding="utf-8"
    )
    stub = (ROOT / "python" / "forge3d" / "__init__.pyi").read_text(
        encoding="utf-8"
    )
    for name in expected_functions:
        assert f'"{name}"' in init_source
        assert f"def {name}(" in stub
    for name in ("set_atmosphere", "clear_atmosphere", "get_atmosphere_settings"):
        assert f"def {name}(" in stub
    assert "class SkySettings:" not in stub
    assert "from .terrain_params import (" in stub and "    SkySettings," in stub
    atmosphere_stub = (ROOT / "python" / "forge3d" / "atmosphere.pyi").read_text(
        encoding="utf-8"
    )
    assert "@dataclass(frozen=True, slots=True)" in atmosphere_stub
    terrain_params = (ROOT / "python" / "forge3d" / "terrain_params.py").read_text(
        encoding="utf-8"
    )
    assert "lut_handle: AtmosphereLutHandle | None = None" in terrain_params


def test_aether_terrain_segment_contract_locks_actual_distance_and_domain_guards(
) -> None:
    shader = (ROOT / "src" / "shaders" / "terrain_pbr_pom.wgsl").read_text(
        encoding="utf-8"
    )
    evaluation_core = (
        ROOT / "src" / "shaders" / "atmosphere" / "evaluation_core.wgsl"
    ).read_text(encoding="utf-8")
    decoder = (
        ROOT / "src" / "terrain" / "render_params" / "decode_atmosphere.rs"
    ).read_text(encoding="utf-8")
    contract = (
        ROOT / "shaders" / "contracts" / "terrain_pbr_pom.toml"
    ).read_text(encoding="utf-8")
    assert "aether_terrain_segment_transmittance(\n        distance_m," in shader
    assert (
        "aether_terrain_apply_segment(\n"
        "            fog_result,\n"
        "            view_distance,"
    ) in shader
    assert "return aether_eval_segment_transmittance(" in shader
    assert "aether_eval_sample_accumulated_scattering(" in shader
    assert "clamp(distance_m, 0.0, 20000000.0)" in evaluation_core
    assert "clamp(camera_height_m, 0.0, 100000.0)" in evaluation_core
    assert "fn aether_eval_spherical_altitude(" in evaluation_core
    assert "fn aether_eval_spherical_endpoint_mus(" in evaluation_core
    assert "2.0 * radius_m * bounded_distance_m * clamp(view_mu" in evaluation_core
    assert "let h00 = aether_eval_spherical_altitude(" in evaluation_core
    assert "bounded_camera_height_m, view_mu," in evaluation_core
    assert "bounded_distance_m * 0.03125, bottom_radius_m," in evaluation_core
    assert "bounded_distance_m * 0.96875, bottom_radius_m," in evaluation_core
    assert "mix(bounded_camera_height_m, bounded_surface_height_m" not in evaluation_core
    assert "let endpoint_mus = aether_eval_spherical_endpoint_mus(" in shader
    assert "endpoint_mus.y,\n        endpoint_mus.x," in shader
    assert "let rayleigh_density_sum = det_exp(-h00 / 8000.0)" in evaluation_core
    assert "let mie_density_sum = det_exp(-h00 / 1200.0)" in evaluation_core
    assert (
        "let ozone_density_sum = max(1.0 - abs((h00 - 25000.0)"
        in evaluation_core
    )
    assert (
        "let path_per_sample = bounded_distance_m * density_scale * 0.0625;"
        in evaluation_core
    )
    assert "path * det_exp(-mean_height / 8000.0)" not in evaluation_core
    assert "path * det_exp(-mean_height / 1200.0)" not in evaluation_core
    assert "AETHER_TERRAIN_WAVELENGTHS_NM" not in shader
    assert "AETHER_TERRAIN_CIE_XYZ" not in shader
    assert "fn aether_terrain_mu_to_unit" not in shader
    assert "fn aether_terrain_nu_to_unit" not in shader
    assert "fn aether_terrain_load_scattering" not in shader
    assert "fn aether_terrain_finite_normalize(direction: vec3<f32>)" in shader
    assert "direction / max(largest_component, 1.0)" in shader
    assert "inverseSqrt(max(length_squared, 1.0e-12))" in shader
    assert "let view = aether_terrain_finite_normalize(view_direction);" in shader
    assert "let sun = aether_terrain_finite_normalize(sun_direction);" in shader
    assert "let view = normalize(view_direction);" not in shader
    assert "let sun = normalize(sun_direction);" not in shader
    assert "atmosphere_segment_transmittance_spectral" not in shader
    assert "aether_accumulated_scattering_tex" in shader
    assert "aether_terrain_sample_inscatter" in shader
    assert (
        "camera_scattering - base_transmittance * endpoint_scattering" in shader
    )
    assert "let sun_intensity = aether_eval_clamp_radiometric_scale(" in shader
    assert "fog_uniforms.sky_params0.w);" in shader
    assert "let atmosphere_exposure = aether_eval_clamp_radiometric_scale(" in shader
    assert "fog_uniforms.sky_params1.w);" in shader
    assert (
        "base_finite_inscatter * density_adjustment * atmosphere_exposure" in shader
    )
    assert "fn aether_eval_clamp_radiometric_scale(value: f32) -> f32" in evaluation_core
    assert "return min(max(value, 0.0), 65504.0);" in evaluation_core
    assert "fn aether_eval_clamp_hdr_radiance(radiance: vec3<f32>)" in evaluation_core
    assert "vec3<f32>(65504.0)" in evaluation_core
    background = (
        ROOT / "src" / "shaders" / "atmosphere" / "scattering.wgsl"
    ).read_text(encoding="utf-8")
    assert "let sun_intensity = aether_eval_clamp_radiometric_scale(" in background
    assert "let atmosphere_exposure = aether_eval_clamp_radiometric_scale(" in background
    assert "aether_eval_clamp_hdr_radiance(radiance*atmosphere_exposure)" in background
    assert "bounded_sky * (vec3<f32>(1.0) - transmittance)" not in shader
    assert "bounded_surface * transmittance + finite_inscatter" in shader
    assert "fog_enabled && !(aether_enabled && sky_aerial_enabled)" in shader
    assert "sky.aerial_density must be finite and in [0.0, 10.0]" in decoder
    apply_contract = contract.split('name = "aether_terrain_apply_segment"', 1)[1]
    assert '"value:camera_height_m:0:10000000"' in apply_contract
    assert '"value:surface_height_m:0:10000000"' not in apply_contract
    assert '"value:view_direction:-20000000:20000000"' in apply_contract
    assert '"value:sun_direction:-1:1"' in apply_contract
    assert '"uniform:fog_uniforms.sky_params0:0:3.4028235e38"' in apply_contract
    assert '"uniform:fog_uniforms.sky_params1:0:3.4028235e38"' in apply_contract
    atmosphere_host = (
        ROOT / "src" / "terrain" / "renderer" / "atmosphere.rs"
    ).read_text(encoding="utf-8")
    atmosphere_core = (
        ROOT / "src" / "core" / "atmosphere" / "mod.rs"
    ).read_text(encoding="utf-8")
    terrain_host = (
        ROOT / "src" / "terrain" / "renderer" / "bind_groups" / "terrain_pass.rs"
    ).read_text(encoding="utf-8")
    assert "pub(crate) const AETHER_RADIOMETRIC_SCALE_MAX: f32 = 65_504.0;" in atmosphere_core
    assert atmosphere_host.count(".clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX)") >= 2
    assert terrain_host.count(".clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX)") >= 2
    assert "norm_ge:view_direction" not in apply_contract
    assert "norm_ge:sun_direction" not in apply_contract
    assert "invariants = []" in apply_contract

    blit = (ROOT / "src" / "shaders" / "terrain_aether_blit.wgsl").read_text(
        encoding="utf-8"
    )
    assert "vec2<f32>(input.uv.x, 1.0 - input.uv.y)" in blit
    assert "out.uv = uv;" in blit
    assert "out.uv = uv * 0.5;" not in blit


def test_terrain_group_four_is_the_dedicated_shared_atmosphere_group() -> None:
    shader = (ROOT / "src" / "shaders" / "terrain_pbr_pom.wgsl").read_text(
        encoding="utf-8"
    )
    layouts = (
        ROOT / "src" / "terrain" / "renderer" / "bind_groups" / "layouts.rs"
    ).read_text(encoding="utf-8")
    pipeline = (
        ROOT / "src" / "terrain" / "renderer" / "pipeline_cache.rs"
    ).read_text(encoding="utf-8")
    terrain_pass = (
        ROOT
        / "src"
        / "terrain"
        / "renderer"
        / "bind_groups"
        / "terrain_pass.rs"
    ).read_text(encoding="utf-8")

    assert "Shared atmosphere uniforms (@group(4))" in shader
    assert "@group(4) @binding(0)\nvar<uniform> fog_uniforms" in shader
    assert "@group(4) @binding(1)\nvar sky_atmosphere_tex" in shader
    assert (
        "@group(4) @binding(2)\nvar aether_accumulated_scattering_tex" in shader
    )
    assert shader.count("@group(4)") == 4  # one ownership comment + three resources
    assert "terrain_pbr_pom.atmosphere_bind_group_layout" in layouts
    assert "group exclusively owns fog, sky, and AETHER LUT resources" in layouts
    assert "@group(4): dedicated shared atmosphere (bindings 0-2)" in pipeline
    assert "terrain.atmosphere.bind_group" in terrain_pass
    assert "terrain_pbr_pom.fog_bind_group_layout" not in layouts
    assert "terrain.fog.bind_group" not in terrain_pass


def test_oblique_zup_sky_uses_authoritative_terrain_camera() -> None:
    """Lock the non-polar camera case that exposed shadow-camera reuse."""

    forward = (
        ROOT / "src" / "terrain" / "renderer" / "draw" / "execute.rs"
    ).read_text(encoding="utf-8")
    aov = (ROOT / "src" / "terrain" / "renderer" / "aov.rs").read_text(
        encoding="utf-8"
    )
    offline = (ROOT / "src" / "terrain" / "renderer" / "offline.rs").read_text(
        encoding="utf-8"
    )

    for source in (forward, aov):
        assert (
            "let (camera_eye, camera_view, camera_proj) = "
            "Self::build_camera_matrices(params);"
        ) in source
        sky_call = source.split("let sky_texture = self.render_sky_texture(", 1)[
            1
        ].split(")?;", 1)[0]
        assert all(
            camera in sky_call
            for camera in ("camera_view", "camera_proj", "camera_eye")
        )
        assert "shadow_setup." not in sky_call
        assert "camera_eye.z" in source
        assert "camera_eye.y" in source

        water_call = source.split(
            "let water_reflection_bind_group = "
            "self.prepare_water_reflection_bind_group(",
            1,
        )[1].split(");", 1)[0]
        assert all(
            camera in water_call
            for camera in ("camera_eye", "camera_view", "camera_proj")
        )
        assert "shadow_setup.view_matrix" not in water_call
        assert "shadow_setup.proj_matrix" not in water_call
        assert "shadow_setup.eye" not in water_call

    assert "let (eye, view, proj) = Self::build_camera_matrices(&state.params);" in offline
    assert "if is_zup_camera_mode(&state.params.camera_mode)" in offline
    assert "eye.z" in offline

    # theta=90 degrees made the Z-up and legacy Y-up camera heights coincide
    # at the target. At an oblique theta they diverge materially, so this case
    # cannot pass if the sky path is wired back to ShadowSetup's Y-up eye.
    radius = 40_000.0
    theta = np.deg2rad(70.0)
    phi = np.deg2rad(180.0)
    target = np.array([100.0, 200.0, 300.0])
    authoritative_zup_eye = target + radius * np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    legacy_shadow_eye = target + radius * np.array(
        [np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)]
    )
    assert authoritative_zup_eye[2] == pytest.approx(
        target[2] + radius * np.cos(theta)
    )
    assert abs(authoritative_zup_eye[2] - legacy_shadow_eye[2]) > 10_000.0


def test_terrain_segment_midpoint_columns_cover_vertical_and_curved_paths() -> None:
    camera_height = 13_681.0
    surface_height = 0.0
    height_delta = surface_height - camera_height

    def exponential_mean(scale_height: float) -> float:
        return scale_height * (
            np.exp(-camera_height / scale_height)
            - np.exp(-surface_height / scale_height)
        ) / height_delta

    exact_rayleigh_mean = exponential_mean(8_000.0)
    exact_mie_mean = exponential_mean(1_200.0)
    midpoint_fractions = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    midpoint_heights = camera_height + height_delta * midpoint_fractions
    rayleigh_mean = float(np.mean(np.exp(-midpoint_heights / 8_000.0)))
    mie_mean = float(np.mean(np.exp(-midpoint_heights / 1_200.0)))
    ozone_mean = float(
        np.mean(np.maximum(1.0 - np.abs((midpoint_heights - 25_000.0) / 15_000.0), 0.0))
    )
    heights = np.linspace(camera_height, surface_height, 200_001, dtype=np.float64)
    numeric_rayleigh = float(np.trapz(np.exp(-heights / 8_000.0), heights) / height_delta)
    numeric_mie = float(np.trapz(np.exp(-heights / 1_200.0), heights) / height_delta)
    ozone = np.maximum(1.0 - np.abs((heights - 25_000.0) / 15_000.0), 0.0)
    numeric_ozone = float(np.trapz(ozone, heights) / height_delta)

    assert exact_rayleigh_mean == pytest.approx(numeric_rayleigh, rel=1.0e-10)
    assert exact_mie_mean == pytest.approx(numeric_mie, rel=1.0e-9)
    assert numeric_ozone > 0.0
    assert rayleigh_mean == pytest.approx(exact_rayleigh_mean, rel=0.01)
    assert mie_mean == pytest.approx(exact_mie_mean, rel=0.03)
    assert ozone_mean == pytest.approx(numeric_ozone, rel=0.03)

    midpoint = 0.5 * (camera_height + surface_height)
    midpoint_rayleigh = float(np.exp(-midpoint / 8_000.0))
    midpoint_mie = float(np.exp(-midpoint / 1_200.0))
    midpoint_ozone = max(1.0 - abs((midpoint - 25_000.0) / 15_000.0), 0.0)
    assert abs(midpoint_rayleigh / exact_rayleigh_mean - 1.0) > 0.10
    assert midpoint_mie / exact_mie_mean < 0.05
    assert midpoint_ozone == 0.0

    bottom_radius = 6_360_000.0
    horizontal_distance = 200_000.0
    midpoint_distances = horizontal_distance * midpoint_fractions
    spherical_heights = np.sqrt(
        (bottom_radius + surface_height) ** 2 + midpoint_distances**2
    ) - bottom_radius
    flat_heights = np.zeros_like(spherical_heights)
    assert spherical_heights[-1] > 2_500.0
    spherical_rayleigh = float(np.mean(np.exp(-spherical_heights / 8_000.0)))
    flat_rayleigh = float(np.mean(np.exp(-flat_heights / 8_000.0)))
    assert spherical_rayleigh < flat_rayleigh * 0.90

    def endpoint_mus(
        camera_height_m: float,
        view_mu: float,
        sun_mu: float,
        view_sun_nu: float,
        distance_m: float,
    ) -> tuple[float, float]:
        radius_m = bottom_radius + camera_height_m
        endpoint_radius_m = np.sqrt(
            radius_m**2
            + distance_m**2
            + 2.0 * radius_m * distance_m * view_mu
        )
        return (
            (radius_m * view_mu + distance_m) / endpoint_radius_m,
            (radius_m * sun_mu + distance_m * view_sun_nu)
            / endpoint_radius_m,
        )

    horizon_view_mu, horizon_sun_mu = endpoint_mus(
        0.0, 0.0, 0.5, 0.25, horizontal_distance
    )
    assert horizon_view_mu == pytest.approx(0.031431, rel=1.0e-5)
    assert horizon_sun_mu > 0.5

    # A descending ray must rotate both endpoint cosines into the endpoint's
    # radial frame, not reuse camera-frame mu values.
    start_height = 10_000.0
    descending_view_mu = -0.02
    descending_sun_mu = 0.4
    view_2d = np.array(
        [np.sqrt(1.0 - descending_view_mu**2), descending_view_mu]
    )
    sun_2d = np.array(
        [np.sqrt(1.0 - descending_sun_mu**2), descending_sun_mu]
    )
    descending_nu = float(np.dot(view_2d, sun_2d))
    descending_distance = 50_000.0
    computed_view_mu, computed_sun_mu = endpoint_mus(
        start_height,
        descending_view_mu,
        descending_sun_mu,
        descending_nu,
        descending_distance,
    )
    endpoint_position = np.array([0.0, bottom_radius + start_height]) + (
        descending_distance * view_2d
    )
    endpoint_up = endpoint_position / np.linalg.norm(endpoint_position)
    assert computed_view_mu == pytest.approx(float(np.dot(view_2d, endpoint_up)))
    assert computed_sun_mu == pytest.approx(float(np.dot(sun_2d, endpoint_up)))
    assert computed_view_mu != pytest.approx(descending_view_mu)
    assert computed_sun_mu != pytest.approx(descending_sun_mu)


def test_python_helper_contains_no_silent_atmosphere_fallback() -> None:
    source = inspect.getsource(atmosphere)
    assert "_native_symbol" in source
    assert "analytic fallback" in source
    assert "np.zeros" not in source
    assert "hosek" not in source.lower()


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native AETHER contract requires extension")
def test_native_flat_spectrum_converts_to_neutral_linear_rgb(
    default_lut_report: dict,
) -> None:
    report = default_lut_report
    wavelength_count = int(report["wavelength_count"])
    assert wavelength_count >= 8
    rgb = np.asarray(
        f3d.atmosphere_spectral_to_linear_rgb([1.0] * wavelength_count),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(rgb))
    assert np.allclose(rgb, 1.0, atol=1.0e-5)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native AETHER contract requires extension")
def test_native_default_lut_report_is_finite_tracked_and_convergent(
    default_lut_report: dict,
) -> None:
    report = default_lut_report
    assert report["storage_format"] == "rgba16float"
    assert report["scattering_lut_semantics"] == (
        "accumulated-single-plus-higher-orders-density-height-u-squared"
    )
    assert report["aerial_lut_semantics"] == (
        "rgb-zero-alpha-mean-segment-transmittance"
    )
    assert report.aerial_lut_semantics == report["aerial_lut_semantics"]
    assert isinstance(report["precomputed"], bool)
    assert int(report["byte_size"]) > 0
    assert len(report["deterministic_sha256"]) == 64
    assert int(report["wavelength_count"]) >= 8
    for name in (
        "transmittance_rgba",
        "single_scattering_rgba",
        "multiple_scattering_rgba",
        "aerial_perspective_rgba",
    ):
        payload = np.asarray(report[name], dtype=np.float32)
        assert payload.size > 0, name
        assert np.all(np.isfinite(payload)), name
    aerial = np.asarray(report["aerial_perspective_rgba"], dtype=np.float32).reshape(
        -1, 4
    )
    assert np.array_equal(aerial[:, :3], np.zeros_like(aerial[:, :3]))
    assert np.all((0.0 <= aerial[:, 3]) & (aerial[:, 3] <= 1.0))
    deltas = np.asarray(report["order_deltas"], dtype=np.float64)
    assert deltas.size >= 3
    assert np.all(deltas > 0.0)
    assert np.all(np.diff(deltas) < 0.0)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native AETHER contract requires extension")
def test_validation_environment_is_linear_hdr_and_certificate_excluded_honestly() -> None:
    report = f3d.atmosphere_generate_environment(8, 4, 10.0, mode="lut")
    rgb = np.asarray(report["rgb_linear"], dtype=np.float32)
    assert rgb.shape == (4, 8, 3)
    assert report["linear_hdr"] is True
    assert np.all(np.isfinite(rgb))
    assert "Outside CENSOR's render-certificate scope" in (
        f3d.atmosphere_generate_environment.__doc__ or ""
    )


@pytest.mark.skipif(
    not NATIVE_AVAILABLE or not f3d.has_gpu(),
    reason="live Scene AETHER semantics require the native GPU extension",
)
def test_scene_aether_state_is_configured_not_active_and_render_fails_closed() -> None:
    scene = f3d.Scene(16, 16, grid=2)
    scene.set_atmosphere(turbidity=2.0, ozone_du=300.0, mie_g=0.8)
    settings = scene.get_atmosphere_settings()
    assert settings["configured"] is True
    assert settings["active"] is False
    assert settings["precomputed"] is True
    assert len(settings["deterministic_sha256"]) == 64
    assert "TerrainRenderer" in settings["active_render_path"]
    with pytest.raises(RuntimeError, match="no spectral-atmosphere render consumer"):
        scene.render_rgba()
    scene.clear_atmosphere()
    assert scene.get_atmosphere_settings()["configured"] is False


@pytest.mark.skipif(
    not NATIVE_AVAILABLE or not f3d.has_gpu(),
    reason="live Scene AETHER semantics require the native GPU extension",
)
def test_scene_custom_unshipped_lut_inputs_fail_without_mutating_state() -> None:
    scene = f3d.Scene(16, 16, grid=2)
    with pytest.raises(RuntimeError, match="no nearby or legacy LUT was substituted"):
        scene.set_atmosphere(turbidity=2.0, ozone_du=301.0, mie_g=0.8)
    assert scene.get_atmosphere_settings()["configured"] is False
