"""Contracts for AETHER's genuine PROMETHEUS stochastic reference lane."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / "src/shaders/atmosphere/prometheus_spectral_reference.wgsl"
RUST_DRIVER = ROOT / "src/path_tracing/hybrid_compute/aether_reference.rs"


def test_reference_source_cannot_be_replaced_by_lut_or_cpu_injection() -> None:
    source = SHADER.read_text()
    driver = RUST_DRIVER.read_text()
    assert "AETHER_REF_WAVELENGTH_COUNT: u32 = 11u" in source
    assert "AETHER_REF_MAX_DEPTH: u32 = 6u" in source
    assert "free_flight" in source
    assert "aether_ref_sample_rayleigh" in source
    assert "aether_ref_sample_mie" in source
    assert "AETHER_REF_RR_START_DEPTH" in source
    assert "AETHER_REF_PLANET_RAY_OFFSET_M: f32 = 2.0" in source
    assert "fn aether_ref_surface_ray_origin" in source
    assert "normal * (AETHER_REF_BOTTOM_RADIUS_M + AETHER_REF_PLANET_RAY_OFFSET_M)" in source
    assert "Ray(surface_ray_origin, 1e-3, bounce_direction, 1e30)" in source
    assert "intersect_hybrid(camera_ray)" in source
    assert "intersect_shadow_ray(shadow_ray, top_t)" in source
    assert "let step_count = 64u" in source
    assert "let step_count = 24u" not in source
    assert "let enabled = uniforms.aov_flags != 0u" in source
    assert "terrain.mips.y & 2u" not in source
    assert "explicit black environment" in source
    assert "textureSample" not in source
    assert "sample_inscatter" not in source
    assert "terrain_env_radiance" not in source
    assert "var sum_xyz = vec3<f32>(0.0)" in source
    assert "sum_xyz = sum_xyz + xyz" in source
    assert "let sample_y = xyz.y" in source
    assert "sample_rgb = max" not in source
    assert "aether_ref_xyz_to_rgb" not in source
    assert "render_aether_spectral_reference" in driver
    assert "pipeline_aether_reference" in driver
    assert "let final_rgb = aether_finalize_xyz_sum(sum_xyz, desc.spp)" in driver
    assert "aether_xyz_to_signed_linear_rgb(mean_xyz)" in driver
    assert "pub mean_xyz: Vec<f32>" in driver
    assert 'result.set_item("mean_xyz", mean_xyz)' in (
        ROOT / "src/py_functions/path_tracing/aether_reference.rs"
    ).read_text()


def test_final_conversion_commutes_with_split_xyz_accumulation() -> None:
    matrix = np.asarray(
        [
            [3.2404542 / 3.2613921, -1.5371385 / 3.2613921, -0.4985314 / 3.2613921],
            [-0.9692660 / 2.5069624, 1.8760108 / 2.5069624, 0.0415560 / 2.5069624],
            [0.0556434 / 2.3679786, -0.2040259 / 2.3679786, 1.0572252 / 2.3679786],
        ],
        dtype=np.float64,
    )
    samples = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    def finalize(sum_xyz: np.ndarray, count: int) -> np.ndarray:
        return np.maximum(matrix @ (sum_xyz / float(count)), 0.0)

    whole = finalize(samples.sum(axis=0), len(samples))
    split_sum = samples[:1].sum(axis=0) + samples[1:].sum(axis=0)
    np.testing.assert_array_equal(whole, finalize(split_sum, len(samples)))

    prematurely_clipped = np.stack([finalize(sample, 1) for sample in samples]).mean(
        axis=0
    )
    assert np.max(np.abs(whole - prematurely_clipped)) > 1.0e-3


@pytest.fixture(scope="module")
def native():
    try:
        import forge3d
        from forge3d import _forge3d
    except Exception as error:  # pragma: no cover - environment diagnostic
        pytest.skip(f"native forge3d extension unavailable: {error}")
    if not hasattr(_forge3d, "hybrid_render_aether_spectral_reference"):
        pytest.skip("native extension predates the AETHER stochastic reference")
    return _forge3d


def _render(native, *, spp: int, seed: int, enabled: bool = True) -> dict:
    heightmap = np.zeros((8, 8), dtype=np.float32)
    camera = {
        "origin": (0.0, 2.0, 0.0),
        "look_at": (1.0, 2.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": 8.0,
    }
    return native.hybrid_render_aether_spectral_reference(
        heightmap,
        1,
        1,
        camera,
        spacing=(1000.0, 1000.0),
        sun_azimuth_deg=60.0,
        sun_elevation_deg=10.0,
        sun_intensity=20.0,
        spp=spp,
        seed=seed,
        enabled=enabled,
        variance_threshold=5.0e-3,
    )


def test_disabled_reference_is_explicit_black(native) -> None:
    output = _render(native, spp=2, seed=11, enabled=False)
    np.testing.assert_array_equal(output["mean_xyz"], np.zeros((1, 1, 3), np.float32))
    np.testing.assert_array_equal(output["linear_rgb"], np.zeros((1, 1, 3), np.float32))
    assert output["environment"] == "black"
    assert output["variance"] == 0.0
    assert output["converged"] is True


def test_non_emitted_capture_marks_disabled_timing_as_a_degradation(native) -> None:
    from forge3d.diagnostics import render_certificate

    _render(native, spp=2, seed=13)
    certificate = render_certificate(sign=False)
    assert certificate["passes"] == [
        {
            "label": "hybrid_pt.aether_spectral_reference",
            "gpu_ms": 0.0,
            "draw_calls": 1,
        }
    ]
    assert any(
        item["kind"] == "timing_unavailable"
        and item["name"] == "hybrid_pt.aether_spectral_reference"
        for item in certificate["degradations"]
    )


def test_low_spp_changes_with_seed(native) -> None:
    first = _render(native, spp=2, seed=11)
    repeated = _render(native, spp=2, seed=11)
    second = _render(native, spp=2, seed=97)
    assert first["seed"] == 11 and second["seed"] == 97
    assert first["spp"] == second["spp"] == 2
    assert first["wavelength_count"] == 11
    assert first["max_depth"] >= 4
    np.testing.assert_array_equal(first["mean_xyz"], repeated["mean_xyz"])
    np.testing.assert_array_equal(first["linear_rgb"], repeated["linear_rgb"])
    assert not np.array_equal(first["mean_xyz"], second["mean_xyz"])
    assert first["variance"] == repeated["variance"]
    assert not np.array_equal(first["linear_rgb"], second["linear_rgb"])


def test_public_rgb_is_finalized_once_from_unclipped_mean_xyz(native) -> None:
    output = _render(native, spp=16, seed=23)
    matrix = np.asarray(
        [
            [3.2404542 / 3.2613921, -1.5371385 / 3.2613921, -0.4985314 / 3.2613921],
            [-0.9692660 / 2.5069624, 1.8760108 / 2.5069624, 0.0415560 / 2.5069624],
            [0.0556434 / 2.3679786, -0.2040259 / 2.3679786, 1.0572252 / 2.3679786],
        ],
        dtype=np.float64,
    )
    expected = np.maximum(
        np.asarray(output["mean_xyz"], dtype=np.float64) @ matrix.T, 0.0
    )
    np.testing.assert_allclose(output["linear_rgb"], expected, rtol=2e-6, atol=1e-8)


def test_more_samples_improve_reported_mean_variance(native) -> None:
    # The high-SPP estimator extends the exact same deterministic path stream.
    # Seed 7 is locked because its four-sample prefix is intentionally noisy,
    # making the expected reduction large and stable across GPU backends.
    low = _render(native, spp=4, seed=7)
    high = _render(native, spp=64, seed=7)
    assert np.isfinite(low["linear_rgb"]).all()
    assert np.isfinite(high["linear_rgb"]).all()
    assert np.isfinite(low["variance"])
    assert np.isfinite(high["variance"])
    assert high["variance"] < low["variance"]
    assert high["gpu_resource_bytes"] > 0


def test_primary_rays_report_real_terrain_classification(native) -> None:
    heightmap = np.zeros((8, 8), dtype=np.float32)
    camera = {
        "origin": (0.0, 20.0, 25.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": 5.0,
    }
    output = native.hybrid_render_aether_spectral_reference(
        heightmap,
        1,
        1,
        camera,
        spacing=(10.0, 10.0),
        spp=4,
        seed=5,
    )
    assert output["terrain_primary_hits"] == 4


def test_reference_emits_live_pass_and_exact_shader_provenance(native) -> None:
    from forge3d.diagnostics import render_certificate

    heightmap = np.zeros((4, 4), dtype=np.float32)
    camera = {
        "origin": (0.0, 2.0, 0.0),
        "look_at": (1.0, 2.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": 8.0,
    }
    native.hybrid_render_aether_spectral_reference(
        heightmap,
        1,
        1,
        camera,
        spacing=(1000.0, 1000.0),
        spp=2,
        seed=17,
        certificate=True,
        cache=None,
    )
    certificate = render_certificate(sign=False)
    assert [entry["label"] for entry in certificate["passes"]] == [
        "hybrid_pt.aether_spectral_reference"
    ]
    assert set(certificate["engine"]["wgsl_module_hashes"]) == {"hybrid-pt-kernel"}
    timing = certificate["passes"][0]
    if "timestamp_query" not in certificate["capabilities"]["granted"]:
        assert timing["gpu_ms"] == 0.0
    elif timing["gpu_ms"] == 0.0:
        assert any(
            item["kind"] == "timing_unavailable"
            and item["name"] == "hybrid_pt.aether_spectral_reference"
            for item in certificate["degradations"]
        )
    else:
        assert timing["gpu_ms"] > 0.0
