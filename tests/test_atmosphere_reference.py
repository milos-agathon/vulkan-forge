"""AETHER hard closure gates against the PROMETHEUS terrain reference."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import forge3d as f3d
from _aether_quadrature import (
    SUN_ELEVATIONS_DEG,
    filmic_terrain_srgb,
    physical_metal_probe,
    saturation,
    write_constant_hdr,
)
from _deltae import delta_e_2000, srgb_to_lab, srgb_to_linear
from _terrain_runtime import terrain_rendering_available
from forge3d.path_tracing import hybrid_render_terrain_reference
from forge3d.terrain_params import (
    OfflineQualitySettings,
    PomSettings,
    ShadowSettings,
    SkySettings,
    make_terrain_params_config,
)


ROOT = Path(__file__).resolve().parents[1]
SIZE = 65
DELTA_E_LIMIT = 2.0
SATURATION_RELATIVE_ERROR_LIMIT = 0.10
TERRAIN_SIZE = 64
TERRAIN_SPAN_M = 60_000.0
TERRAIN_CAMERA_RADIUS_M = 40_000.0
TERRAIN_CLIP_FAR_M = 200_000.0
SKY_OBSERVER_ALTITUDE_M = 1.0
# Deterministic stratification over the visible upper hemisphere and relative
# sun-angle axis. Solar azimuths remain outside the 20-degree camera frustum,
# so the hard closure scores atmospheric radiance rather than disk rasterization.
SKY_SAMPLE_COORDS = tuple((x, y) for y in (8, 20, 28) for x in (8, SIZE // 2, 56))
SKY_CASES = tuple(
    (f"az{sun_azimuth:g}_x{x}_y{y}", sun_azimuth, x, y)
    for sun_azimuth in (20.0, 90.0, 160.0)
    for x, y in SKY_SAMPLE_COORDS
)
REFERENCE_SEEDS = (17, 23, 41, 97)
REFERENCE_SPP_PER_SEED = 4096
REFERENCE_BATCH_VARIANCE_LIMIT = 1.0e-3
# With equal independent batches, Var(mean) = sum(Var_i) / B^2. This is the
# strict bound implied by every batch satisfying the canonical convergence
# threshold; the observed aggregate remains part of the acceptance evidence.
REFERENCE_MEAN_VARIANCE_LIMIT = REFERENCE_BATCH_VARIANCE_LIMIT / len(REFERENCE_SEEDS)

XYZ_TO_NORMALIZED_LINEAR_SRGB = np.asarray(
    [
        [3.2404542 / 3.2613921, -1.5371385 / 3.2613921, -0.4985314 / 3.2613921],
        [-0.9692660 / 2.5069624, 1.8760108 / 2.5069624, 0.0415560 / 2.5069624],
        [0.0556434 / 2.3679786, -0.2040259 / 2.3679786, 1.0572252 / 2.3679786],
    ],
    dtype=np.float64,
)


def _record_acceptance_metric(name: str, value: object) -> None:
    raw_path = os.environ.get("FORGE3D_AETHER_METRICS_PATH")
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"schema_version": 1}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise AssertionError("AETHER metrics artifact must contain a JSON object")
        payload.update(loaded)
    payload[name] = value
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _mean_xyz_to_signed_linear_rgb(mean_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(mean_xyz, dtype=np.float64) @ XYZ_TO_NORMALIZED_LINEAR_SRGB.T


def _disabled_shadows() -> ShadowSettings:
    return ShadowSettings(
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


def _require_physical_metal() -> dict:
    physical, probe = physical_metal_probe()
    required = os.environ.get("FORGE3D_AETHER_PHYSICAL_METAL") == "1"
    if not physical:
        message = f"AETHER physical Metal reference unavailable: {probe}"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    if not terrain_rendering_available():
        message = f"AETHER terrain runtime unavailable on physical Metal: {probe}"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    return probe


def _make_metal_runtime(hdr_path: Path) -> tuple[object, object, object, dict]:
    probe = dict(f3d.device_probe("metal"))
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material = f3d.MaterialSet.custom((0.78, 0.24, 0.08), 0.0, 0.75, 1.0, 0.0, 4.0)
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    return renderer, material, ibl, probe


def _run_aether_physical_process(mode: str, elevation: float | None = None) -> dict:
    command = [
        sys.executable,
        str(ROOT / "tests" / "_aether_physical_probe.py"),
        mode,
    ]
    if elevation is not None:
        command.append(str(float(elevation)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, {
        "mode": mode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    prefix = "AETHER_PHYSICAL_RESULT="
    payloads = [
        line.removeprefix(prefix)
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(payloads) == 1, {
        "mode": mode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    return json.loads(payloads[0])


def _sky_params(
    elevation_deg: float,
    sun_azimuth_deg: float,
    *,
    sun_intensity: float = 1.0,
    sky_exposure: float = 1.0,
) -> object:
    return f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=(SIZE, SIZE),
            render_scale=1.0,
            terrain_span=3.0,
            msaa_samples=1,
            z_scale=0.1,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="material",
            colormap_strength=0.0,
            ibl_enabled=False,
            light_azimuth_deg=float(sun_azimuth_deg),
            light_elevation_deg=float(elevation_deg),
            sun_intensity=1.0,
            cam_radius=10.0,
            cam_phi_deg=180.0,
            cam_theta_deg=90.0,
            cam_target=(0.0, 0.0, SKY_OBSERVER_ALTITUDE_M),
            fov_y_deg=20.0,
            camera_mode="mesh:zup",
            # The hard radiometric closure must use the renderer's documented
            # pixel-correctness oracle, not visibility/indirect submission.
            culling="none",
            clip=(0.1, 40.0),
            shadows=_disabled_shadows(),
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            sky=SkySettings(
                enabled=True,
                model="aether",
                turbidity=2.0,
                ozone_du=300.0,
                mie_g=0.8,
                ground_albedo=0.3,
                sun_intensity=sun_intensity,
                aerial_perspective=True,
                aerial_density=1.0,
                sky_exposure=sky_exposure,
            ),
            aa_samples=1,
            aa_seed=17,
        )
    )


def _render_lut_sky_samples(
    runtime: tuple[object, object, object, dict], elevation: float
) -> np.ndarray:
    renderer, material, ibl, _ = runtime
    samples: list[np.ndarray] = []
    rendered: dict[float, np.ndarray] = {}
    # Every scored pixel is above the optical axis.  With the Z-up horizon
    # camera and an observer above the flat z=0 fixture, those rays have a
    # positive z component and cannot intersect terrain.  Use the production
    # beauty path directly; the MRT AOV path is not part of the sky contract.
    assert all(y < SIZE // 2 for _, _, _, y in SKY_CASES)
    for _, sun_azimuth, x, y in SKY_CASES:
        if sun_azimuth not in rendered:
            frame = renderer.render_terrain_pbr_pom(
                material,
                ibl,
                _sky_params(elevation, sun_azimuth),
                np.zeros((8, 8), dtype=np.float32),
            )
            rendered[sun_azimuth] = np.asarray(frame.to_numpy(), dtype=np.uint8)
        rgba = rendered[sun_azimuth]
        samples.append(rgba[y, x, :3])
    return np.stack(samples)


def _render_prometheus_reference_samples(
    elevation: float,
) -> tuple[np.ndarray, dict[str, object]]:
    # Each case dispatches PROMETHEUS's genuine stochastic spectral transport
    # on a 1x1 pinhole aligned to the LUT pixel-center ray.  The tiny FOV makes
    # its mandatory subpixel jitter a matched center footprint instead of
    # mixing terrain below the geometric horizon into a center-only sky pixel.
    # Escaped paths see the kernel's explicit black environment; no expected
    # color, LUT handle, or CPU oracle enters this reference path.
    linear_samples: list[np.ndarray] = []
    variances: list[float] = []
    assert len(set(REFERENCE_SEEDS)) == len(REFERENCE_SEEDS)
    tan_half_fov = math.tan(math.radians(20.0) * 0.5)
    # Average independent maximum-SPP batches before display conversion. A
    # single 4096-sample seed can still move a dark channel by one byte and
    # straddle the strict DeltaE=2 boundary; the aggregate estimates the PT
    # mean instead of accepting or rejecting on that stochastic coin flip.
    for _, sun_azimuth, x, y in SKY_CASES:
        ndc_x = ((x + 0.5) / SIZE) * 2.0 - 1.0
        ndc_y = 1.0 - ((y + 0.5) / SIZE) * 2.0
        ray = np.asarray(
            [1.0, ndc_y * tan_half_fov, -ndc_x * tan_half_fov], dtype=np.float64
        )
        ray /= np.linalg.norm(ray)
        origin = np.asarray((-10.0, SKY_OBSERVER_ALTITUDE_M, 0.0), dtype=np.float64)
        batch_xyz: list[np.ndarray] = []
        batch_variances: list[float] = []
        for reference_seed in REFERENCE_SEEDS:
            output = f3d.hybrid_render_aether_spectral_reference(
                np.zeros((8, 8), dtype=np.float32),
                1,
                1,
                {
                    "origin": tuple(origin),
                    "look_at": tuple(origin + ray),
                    "up": (0.0, 1.0, 0.0),
                    "fov_y": 0.001,
                },
                spacing=(3.0 / 7.0, 3.0 / 7.0),
                exaggeration=0.1,
                sun_azimuth_deg=sun_azimuth,
                sun_elevation_deg=float(elevation),
                sun_intensity=1.0,
                turbidity=2.0,
                ozone_du=300.0,
                mie_g=0.8,
                ground_albedo=0.3,
                spp=REFERENCE_SPP_PER_SEED,
                seed=reference_seed,
                enabled=True,
                variance_threshold=REFERENCE_BATCH_VARIANCE_LIMIT,
                cache=None,
            )
            assert output["environment"] == "black"
            assert int(output["wavelength_count"]) >= 8
            assert int(output["max_depth"]) >= 4
            assert int(output["seed"]) == reference_seed
            assert int(output["spp"]) == REFERENCE_SPP_PER_SEED
            assert bool(output["converged"]), output
            assert float(output["variance"]) <= REFERENCE_BATCH_VARIANCE_LIMIT
            assert int(output["terrain_primary_hits"]) == 0
            batch_mean_xyz = np.asarray(output["mean_xyz"][0, 0], dtype=np.float64)
            batch_rgb = np.asarray(output["linear_rgb"][0, 0], dtype=np.float64)
            np.testing.assert_allclose(
                batch_rgb,
                np.maximum(_mean_xyz_to_signed_linear_rgb(batch_mean_xyz), 0.0),
                rtol=2e-6,
                atol=1e-8,
            )
            batch_xyz.append(batch_mean_xyz)
            batch_variances.append(float(output["variance"]))
        # Combine the unclipped estimator in XYZ. Converting and clipping once
        # after the equal-SPP mean preserves signed-channel cancellation.
        combined_xyz = np.mean(np.stack(batch_xyz), axis=0)
        linear_samples.append(
            np.maximum(_mean_xyz_to_signed_linear_rgb(combined_xyz), 0.0)
        )
        # Independent equal-size means: Var(mean(batches)) = sum(Var_i)/B^2.
        variances.append(sum(batch_variances) / len(batch_variances) ** 2)
    reference_linear = np.stack(linear_samples)
    samples = (
        np.rint(filmic_terrain_srgb(reference_linear) * 255.0)
        .clip(0.0, 255.0)
        .astype(np.uint8)
    )
    evidence: dict[str, object] = {
        "seeds": list(REFERENCE_SEEDS),
        "batch_count": len(REFERENCE_SEEDS),
        "spp_per_seed": REFERENCE_SPP_PER_SEED,
        "total_spp": REFERENCE_SPP_PER_SEED * len(REFERENCE_SEEDS),
        "max_variance": max(variances),
        "converged": True,
        "environment": "black",
        "wavelength_count": 11,
        "max_depth": 6,
        "case_count": len(SKY_CASES),
    }
    return samples, evidence


def test_sky_delta_e2000_under_two_for_full_sun_elevation_sweep() -> None:
    _require_physical_metal()
    scores: dict[str, float] = {}
    comparisons: dict[str, dict] = {}
    for elevation in SUN_ELEVATIONS_DEG:
        lut_sample = _run_aether_physical_process("sky-lut", elevation)
        pt_sample = _run_aether_physical_process("sky-prometheus", elevation)
        assert lut_sample["mode"] == "sky-lut"
        assert pt_sample["mode"] == "sky-prometheus"
        assert float(lut_sample["elevation"]) == float(elevation)
        assert float(pt_sample["elevation"]) == float(elevation)
        evidence = pt_sample["reference"]
        assert evidence["environment"] == "black"
        assert bool(evidence["converged"])
        assert evidence["seeds"] == list(REFERENCE_SEEDS)
        assert int(evidence["batch_count"]) == len(REFERENCE_SEEDS)
        assert len(set(evidence["seeds"])) == len(REFERENCE_SEEDS)
        assert int(evidence["spp_per_seed"]) == REFERENCE_SPP_PER_SEED
        assert int(evidence["total_spp"]) == REFERENCE_SPP_PER_SEED * len(
            REFERENCE_SEEDS
        )
        assert int(evidence["case_count"]) == len(SKY_CASES)
        assert float(evidence["max_variance"]) <= REFERENCE_MEAN_VARIANCE_LIMIT
        lut_rgb = np.asarray(lut_sample["samples"], dtype=np.uint8)
        pt_rgb = np.asarray(pt_sample["samples"], dtype=np.uint8)
        assert lut_rgb.shape == pt_rgb.shape == (len(SKY_CASES), 3)
        for case, lut_pixel, pt_pixel in zip(SKY_CASES, lut_rgb, pt_rgb):
            # Pass normalized floats explicitly: AEQUITAS accepts either 0..1
            # floats or 0..255 bytes, but an all-dark byte triplet whose maximum
            # is exactly one is otherwise ambiguous at the conversion boundary.
            score = float(
                delta_e_2000(
                    srgb_to_lab(lut_pixel.astype(np.float64) / 255.0),
                    srgb_to_lab(pt_pixel.astype(np.float64) / 255.0),
                )
            )
            label, sun_azimuth, x, y = case
            sample_key = f"{elevation:g}:{label}"
            scores[sample_key] = score
            comparisons[sample_key] = {
                "elevation_deg": elevation,
                "case": label,
                "sun_azimuth_deg": sun_azimuth,
                "pixel": [x, y],
                "delta_e_2000": score,
                "lut_rgb": lut_pixel.tolist(),
                "prometheus_rgb": pt_pixel.tolist(),
            }
    print("AETHER_DELTA_E_SWEEP=" + json.dumps(scores, sort_keys=True))
    _record_acceptance_metric("delta_e_sweep", scores)
    assert max(scores.values()) < DELTA_E_LIMIT, {
        "limit": DELTA_E_LIMIT,
        "comparisons": comparisons,
    }


def _terrain_params(*, aether: bool, sky_exposure: float = 1.0) -> object:
    # Keep the spectral sky/background identical and toggle only the terrain
    # transport under test. This is the paired control for surface*T+inscatter.
    sky = SkySettings(
        enabled=True,
        model="aether",
        turbidity=2.0,
        ozone_du=300.0,
        mie_g=0.8,
        ground_albedo=0.3,
        sun_intensity=1.0,
        aerial_perspective=aether,
        aerial_density=1.0,
        sky_exposure=sky_exposure,
    )
    return f3d.TerrainRenderParams(
        make_terrain_params_config(
            size_px=(TERRAIN_SIZE, TERRAIN_SIZE),
            render_scale=1.0,
            terrain_span=TERRAIN_SPAN_M,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(-0.5, 0.5),
            albedo_mode="material",
            colormap_strength=0.0,
            ibl_enabled=False,
            light_azimuth_deg=90.0,
            light_elevation_deg=10.0,
            sun_intensity=1.0,
            cam_radius=TERRAIN_CAMERA_RADIUS_M,
            cam_phi_deg=180.0,
            cam_theta_deg=70.0,
            cam_target=(0.0, 0.0, 0.0),
            fov_y_deg=52.0,
            camera_mode="mesh:zup",
            culling="none",
            clip=(100.0, TERRAIN_CLIP_FAR_M),
            shadows=_disabled_shadows(),
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            sky=sky,
            aa_samples=1,
            aa_seed=23,
        )
    )


def _camera_geometry(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    radius = TERRAIN_CAMERA_RADIUS_M
    phi = math.radians(180.0)
    theta = math.radians(70.0)
    eye = np.array(
        [
            radius * math.sin(theta) * math.cos(phi),
            radius * math.sin(theta) * math.sin(phi),
            radius * math.cos(theta),
        ],
        dtype=np.float64,
    )
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half_fov = math.tan(math.radians(52.0) * 0.5)
    xs = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0)
    ys = (1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0)
    xx, yy = np.meshgrid(xs, ys)
    rays = (
        forward
        + xx[..., None] * half_fov * (width / height) * right
        + yy[..., None] * half_fov * up
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    cosine = np.maximum(rays @ forward, 1.0e-6)
    return eye, forward, rays, cosine


def _flat_terrain_hit_geometry(
    width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eye, _, rays_zup, _ = _camera_geometry(width, height)
    distances = np.full(rays_zup.shape[:2], np.nan, dtype=np.float64)
    downward = rays_zup[..., 2] < -1.0e-8
    distances[downward] = -eye[2] / rays_zup[..., 2][downward]
    intersections = eye + rays_zup * distances[..., None]
    half_span = TERRAIN_SPAN_M * 0.5 * 0.98
    hit = (
        np.isfinite(distances)
        & (distances > 0.0)
        & (np.abs(intersections[..., 0]) < half_span)
        & (np.abs(intersections[..., 1]) < half_span)
    )
    return eye, rays_zup, distances, hit


def _filmic_terrain_curve(value: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    a, b, c, d, e, f = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


def _terrain_display_from_linear(linear_rgb: np.ndarray) -> np.ndarray:
    return filmic_terrain_srgb(np.asarray(linear_rgb, dtype=np.float64))


def _terrain_linear_from_display(display_rgb: np.ndarray) -> np.ndarray:
    # Invert the exact IEC sRGB transfer used by `linear_to_srgb` in
    # tonemap_common.wgsl before inverting the filmic curve. A gamma-2.2
    # approximation is already off by far more than the 10% acceptance bound
    # in the dark terrain range.
    target = srgb_to_linear(
        np.clip(np.asarray(display_rgb, dtype=np.float64), 0.0, 1.0)
    )
    low = np.zeros_like(target)
    high = np.full_like(target, 128.0)
    white = float(_filmic_terrain_curve(np.array(11.2)))
    for _ in range(56):
        middle = 0.5 * (low + high)
        mapped = _filmic_terrain_curve(middle) / white
        low = np.where(mapped < target, middle, low)
        high = np.where(mapped >= target, middle, high)
    return 0.5 * (low + high)


def test_terrain_display_roundtrip_uses_exact_iec_srgb_transfer() -> None:
    linear = np.asarray([0.0, 0.001, 0.01, 0.1, 1.0, 4.0], dtype=np.float64)
    display = _terrain_display_from_linear(linear)
    np.testing.assert_allclose(display, filmic_terrain_srgb(linear), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        _terrain_linear_from_display(display), linear, atol=2.0e-12, rtol=2.0e-12
    )

    upload = (
        ROOT / "src" / "terrain" / "renderer" / "upload.rs"
    ).read_text(encoding="utf-8")
    assert "params.output_srgb_eotf" in upload
    assert "decoded.sky.enabled && decoded.sky.model == 3" in upload
    shader = (ROOT / "src" / "shaders" / "terrain_pbr_pom.wgsl").read_text(
        encoding="utf-8"
    )
    assert "active AETHER so sky and terrain share one exact IEC sRGB transfer" in shader


def _measure_terrain_saturation(
    metal_runtime: tuple[object, object, object, dict],
) -> dict:
    renderer, material, ibl, _ = metal_runtime
    heightmap = np.zeros((32, 32), dtype=np.float32)
    baseline_frame = renderer.render_terrain_pbr_pom(
        material, ibl, _terrain_params(aether=False), heightmap
    )
    actual_frame = renderer.render_terrain_pbr_pom(
        material, ibl, _terrain_params(aether=True), heightmap
    )
    surface_display = (
        np.asarray(baseline_frame.to_numpy(), dtype=np.float64)[..., :3] / 255.0
    )
    transported_display = (
        np.asarray(actual_frame.to_numpy(), dtype=np.float64)[..., :3] / 255.0
    )
    surface = _terrain_linear_from_display(surface_display)
    eye, rays_zup, distances, hit = _flat_terrain_hit_geometry(
        transported_display.shape[1], transported_display.shape[0]
    )

    # The fixture is exactly the bounded z=0 terrain plane. Derive its pixel
    # mask from the same pinned camera instead of depending on an auxiliary MRT
    # path: this gate is specifically for the production beauty renderer.
    assert int(hit.sum()) > 64, "flat terrain did not provide enough near/far samples"
    assert np.isfinite(surface[hit]).all() and (surface[hit] >= 0.0).all()
    assert np.isfinite(transported_display[hit]).all()

    assert np.isfinite(distances[hit]).all() and (distances[hit] > 0.0).all()
    hit_indices = np.argwhere(hit)
    hit_distances = distances[hit]
    order = np.argsort(hit_distances)
    chosen = [hit_indices[order[len(order) // 5]], hit_indices[order[4 * len(order) // 5]]]
    sun = np.array([0.0, math.sin(math.radians(10.0)), math.cos(math.radians(10.0))])

    measured_saturation: list[float] = []
    predicted_saturation: list[float] = []
    selected_distances: list[float] = []
    for y, x in chosen:
        ray = rays_zup[y, x]
        ray_yup = (float(ray[0]), float(ray[2]), float(ray[1]))
        distance = float(distances[y, x])
        assert math.isfinite(distance) and distance >= 0.0
        predicted = f3d.atmosphere_reference_aerial(
            tuple(float(v) for v in surface[y, x]),
            float(eye[2]),
            distance,
            ray_yup,
            tuple(float(v) for v in sun),
            turbidity=2.0,
            ozone_du=300.0,
            mie_g=0.8,
        )
        measured_saturation.append(saturation(transported_display[y, x]))
        predicted_saturation.append(
            saturation(_terrain_display_from_linear(np.asarray(predicted)))
        )
        selected_distances.append(distance)

    return {
        "hit_count": int(hit.sum()),
        "distances_m": selected_distances,
        "measured_saturation": measured_saturation,
        "predicted_saturation": predicted_saturation,
    }


def _measure_terrain_exposure_scaling(
    metal_runtime: tuple[object, object, object, dict],
) -> dict:
    renderer, material, ibl, _ = metal_runtime
    heightmap = np.zeros((32, 32), dtype=np.float32)
    displays: list[np.ndarray] = []
    for exposure in (0.0, 1.0, 2.0):
        frame = renderer.render_terrain_pbr_pom(
            material,
            ibl,
            _terrain_params(aether=True, sky_exposure=exposure),
            heightmap,
        )
        displays.append(
            np.asarray(frame.to_numpy(), dtype=np.float64)[..., :3] / 255.0
        )

    linear = [_terrain_linear_from_display(display) for display in displays]
    _, _, _, hit = _flat_terrain_hit_geometry(
        displays[0].shape[1], displays[0].shape[0]
    )
    assert int(hit.sum()) > 64
    delta_one = np.maximum(linear[1] - linear[0], 0.0)
    delta_two = np.maximum(linear[2] - linear[0], 0.0)
    signal = hit & (np.max(delta_one, axis=-1) > 1.0e-3)
    one_energy = float(delta_one[signal].sum())
    two_energy = float(delta_two[signal].sum())
    changed_fraction = float(
        np.any(np.asarray(displays[2]) != np.asarray(displays[0]), axis=-1)[hit].mean()
    )
    linear_scale_ratio = two_energy / max(one_energy, 1.0e-12)
    midpoint_relative_error = float(
        np.abs(delta_two[signal] - 2.0 * delta_one[signal]).sum()
        / max(two_energy, 1.0e-12)
    )
    return {
        "hit_count": int(hit.sum()),
        "signal_count": int(signal.sum()),
        "changed_hit_fraction": changed_fraction,
        "zero_to_one_energy": one_energy,
        "zero_to_two_energy": two_energy,
        "linear_scale_ratio": linear_scale_ratio,
        "midpoint_relative_error": midpoint_relative_error,
    }


def _measure_high_exposure_sky_hdr(
    metal_runtime: tuple[object, object, object, dict],
) -> dict:
    renderer, material, ibl, _ = metal_runtime
    # These are valid finite f32 inputs far above the old proof assumption.
    # The sun is centered inside the camera frustum so the solar-disc term is
    # exercised, not merely the diffuse sky background.
    params = _sky_params(
        5.0,
        0.0,
        sun_intensity=1.0e35,
        sky_exposure=1.0e35,
    )
    result = f3d.render_offline(
        renderer,
        material,
        ibl,
        params,
        np.zeros((8, 8), dtype=np.float32),
        settings=OfflineQualitySettings(
            enabled=True,
            adaptive=False,
            max_samples=1,
            min_samples=1,
            batch_size=1,
        ),
    )
    hdr = np.asarray(result.hdr_frame.to_numpy_f32(), dtype=np.float32)[..., :3]
    upper_sky = hdr[: SIZE // 2]
    return {
        "component_count": int(upper_sky.size),
        "finite_component_count": int(np.isfinite(upper_sky).sum()),
        "minimum": float(upper_sky.min()),
        "maximum": float(upper_sky.max()),
        "near_f16_max_component_count": int((upper_sky >= 65_500.0).sum()),
    }


def test_high_exposure_sun_aligned_sky_hdr_stays_finite() -> None:
    _require_physical_metal()
    physical = _run_aether_physical_process("high-exposure")
    assert physical["mode"] == "high-exposure"
    measurement = physical["measurement"]
    assert int(measurement["component_count"]) > 0
    assert int(measurement["finite_component_count"]) == int(
        measurement["component_count"]
    )
    assert 0.0 <= float(measurement["minimum"])
    assert 60_000.0 <= float(measurement["maximum"]) <= 65_504.0
    assert int(measurement["near_f16_max_component_count"]) > 0
    print("AETHER_HIGH_EXPOSURE_HDR=" + json.dumps(measurement, sort_keys=True))


def test_terrain_aether_inscatter_scales_with_sky_exposure() -> None:
    _require_physical_metal()
    physical = _run_aether_physical_process("exposure")
    assert physical["mode"] == "exposure"
    measurement = physical["measurement"]
    assert int(measurement["hit_count"]) > 64
    assert int(measurement["signal_count"]) > 64
    assert float(measurement["changed_hit_fraction"]) > 0.50
    assert float(measurement["zero_to_one_energy"]) > 0.0
    assert float(measurement["zero_to_two_energy"]) > float(
        measurement["zero_to_one_energy"]
    )
    assert 1.80 <= float(measurement["linear_scale_ratio"]) <= 2.20
    assert float(measurement["midpoint_relative_error"]) <= 0.20
    print("AETHER_EXPOSURE_SCALING=" + json.dumps(measurement, sort_keys=True))


def test_terrain_saturation_falloff_matches_scattering_law_within_ten_percent() -> None:
    _require_physical_metal()
    physical = _run_aether_physical_process("saturation")
    assert physical["mode"] == "saturation"
    measurement = physical["measurement"]
    assert int(measurement["hit_count"]) > 64
    measured_saturation = measurement["measured_saturation"]
    predicted_saturation = measurement["predicted_saturation"]
    selected_distances = measurement["distances_m"]

    assert selected_distances[1] > selected_distances[0] * 1.10
    assert measured_saturation[0] > 0.05
    assert predicted_saturation[0] > 0.05
    assert 0.0 <= measured_saturation[1] < measured_saturation[0]
    assert 0.0 <= predicted_saturation[1] < predicted_saturation[0]
    assert measured_saturation[0] - measured_saturation[1] > 0.005
    assert predicted_saturation[0] - predicted_saturation[1] > 0.005

    measured_ratio = measured_saturation[1] / max(measured_saturation[0], 1.0e-8)
    predicted_ratio = predicted_saturation[1] / max(predicted_saturation[0], 1.0e-8)
    relative_error = abs(measured_ratio - predicted_ratio) / max(abs(predicted_ratio), 1.0e-8)
    print(
        "AETHER_SATURATION_FALLOFF="
        + json.dumps(
            {
                "distances_m": selected_distances,
                "measured_saturation": measured_saturation,
                "predicted_saturation": predicted_saturation,
                "measured_far_near_ratio": measured_ratio,
                "predicted_far_near_ratio": predicted_ratio,
                "relative_error": relative_error,
            },
            sort_keys=True,
        )
    )
    _record_acceptance_metric(
        "saturation_falloff",
        {
            "distances_m": selected_distances,
            "measured_saturation": measured_saturation,
            "predicted_saturation": predicted_saturation,
            "measured_far_near_ratio": measured_ratio,
            "predicted_far_near_ratio": predicted_ratio,
            "relative_error": relative_error,
        },
    )
    assert relative_error <= SATURATION_RELATIVE_ERROR_LIMIT


def _run_prometheus_aerial_process(
    output_path: Path,
    *,
    enabled: bool,
    exposure: float = 1.0,
    sun_intensity: float = 2.5,
    size: int = 64,
) -> dict[str, np.ndarray]:
    code = r"""
import sys
import json
import numpy as np
from forge3d.datasets import mini_dem
from forge3d.path_tracing import hybrid_render_terrain_reference

dem = mini_dem()[::8, ::8].astype(np.float32)
dem -= dem.min()
dem /= max(float(dem.max()), 1.0e-6)
camera = {
    "origin": (0.0, 35_000.0, 90_000.0),
    "look_at": (0.0, 5_000.0, 0.0),
    "up": (0.0, 1.0, 0.0),
    "fov_y": 45.0,
    "exposure": float(sys.argv[3]),
}
atmosphere = None
if sys.argv[2] == "enabled":
    atmosphere = {"turbidity": 10.0, "ozone_du": 300.0, "mie_g": 0.8}
result = hybrid_render_terrain_reference(
    dem,
    int(sys.argv[5]),
    int(sys.argv[5]),
    camera,
    spacing=(100_000.0 / (dem.shape[1] - 1), 100_000.0 / (dem.shape[0] - 1)),
    exaggeration=20_000.0,
    albedo=(0.55, 0.52, 0.48),
    sun_azimuth_deg=225.0,
    sun_elevation_deg=35.0,
    sun_intensity=float(sys.argv[4]),
    env_intensity=0.35,
    spp=1,
    min_frames=2,
    max_frames=2,
    variance_threshold=1.0e30,
    seed=7,
    atmosphere=atmosphere,
    certificate=sys.argv[1] + ".certificate.json",
)
with open(sys.argv[1] + ".certificate.json", encoding="utf-8") as stream:
    certificate = json.load(stream)
np.savez(
    sys.argv[1],
    rgba=result["rgba"],
    albedo=result["albedo"],
    normal=result["normal"],
    depth=result["depth"],
    gpu_resource_bytes=np.asarray(result["gpu_resource_bytes"], dtype=np.uint64),
    pass_labels=np.asarray([entry["label"] for entry in certificate["passes"]], dtype="U64"),
)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(output_path),
            "enabled" if enabled else "baseline",
            str(exposure),
            str(sun_intensity),
            str(size),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, {
        "mode": "enabled" if enabled else "baseline",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    with np.load(output_path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def test_prometheus_aerial_post_preserves_aovs_and_transports_hits_and_misses(
    tmp_path: Path,
) -> None:
    _require_physical_metal()
    # PROMETHEUS's native GPU context is process-global. Isolated processes
    # keep this paired evidence independent of any prior reference render.
    baseline = _run_prometheus_aerial_process(tmp_path / "baseline.npz", enabled=False)
    actual = _run_prometheus_aerial_process(tmp_path / "aether.npz", enabled=True)

    baseline_hit = np.isfinite(baseline["depth"]) & (baseline["depth"] > 0.0)
    actual_hit = np.isfinite(actual["depth"]) & (actual["depth"] > 0.0)
    assert np.array_equal(baseline_hit, actual_hit)
    assert int(actual_hit.sum()) > 1_000
    np.testing.assert_array_equal(baseline["depth"], actual["depth"])
    np.testing.assert_array_equal(baseline["normal"], actual["normal"])
    np.testing.assert_array_equal(baseline["albedo"], actual["albedo"])
    miss_delta = np.abs(
        baseline["rgba"][..., :3].astype(np.int16)
        - actual["rgba"][..., :3].astype(np.int16)
    )[~actual_hit]
    assert miss_delta.size > 0
    changed_miss_fraction = float((miss_delta.max(axis=-1) > 0).mean())
    assert changed_miss_fraction > 0.50
    assert np.any(actual["rgba"][..., :3][~actual_hit] > 0)

    hit_delta = np.abs(
        baseline["rgba"][..., :3].astype(np.int16)
        - actual["rgba"][..., :3].astype(np.int16)
    )[actual_hit]
    changed_fraction = float((hit_delta.max(axis=-1) > 0).mean())
    mean_delta = float(hit_delta.mean())
    assert changed_fraction > 0.50
    assert mean_delta > 1.0

    pass_labels = actual["pass_labels"].tolist()
    assert pass_labels[-1] == "hybrid_pt.aether_aerial", pass_labels
    assert all(label != "hybrid_pt.aether_aerial" for label in pass_labels[:-1])

    baseline_bytes = int(baseline["gpu_resource_bytes"])
    actual_bytes = int(actual["gpu_resource_bytes"])
    assert baseline_bytes < actual_bytes <= 512 * 1024 * 1024
    print(
        "AETHER_PROMETHEUS_POST="
        + json.dumps(
            {
                "terrain_hits": int(actual_hit.sum()),
                "changed_hit_fraction": changed_fraction,
                "mean_abs_hit_rgb_delta": mean_delta,
                "changed_miss_fraction": changed_miss_fraction,
                "certificate_pass_labels": pass_labels,
                "baseline_gpu_resource_bytes": baseline_bytes,
                "aether_gpu_resource_bytes": actual_bytes,
            },
            sort_keys=True,
        )
    )


def test_prometheus_aerial_extreme_radiometric_inputs_do_not_blacken_hits_or_misses(
    tmp_path: Path,
) -> None:
    _require_physical_metal()
    extreme = _run_prometheus_aerial_process(
        tmp_path / "aether-extreme.npz",
        enabled=True,
        exposure=1.0e35,
        sun_intensity=1.0e35,
        size=32,
    )
    hit = np.isfinite(extreme["depth"]) & (extreme["depth"] > 0.0)
    miss = ~hit
    assert int(hit.sum()) > 100
    assert int(miss.sum()) > 100
    rgb = extreme["rgba"][..., :3]
    assert np.isfinite(rgb.astype(np.float32)).all()
    hit_nonblack = float((rgb[hit].max(axis=-1) > 0).mean())
    miss_nonblack = float((rgb[miss].max(axis=-1) > 0).mean())
    assert hit_nonblack > 0.99
    assert miss_nonblack > 0.99
    assert int(rgb.max()) >= 254
    assert extreme["pass_labels"].tolist()[-1] == "hybrid_pt.aether_aerial"
    print(
        "AETHER_PROMETHEUS_EXTREME="
        + json.dumps(
            {
                "terrain_hits": int(hit.sum()),
                "sky_misses": int(miss.sum()),
                "hit_nonblack_fraction": hit_nonblack,
                "miss_nonblack_fraction": miss_nonblack,
                "max_rgb8": int(rgb.max()),
            },
            sort_keys=True,
        )
    )


def test_aether_shader_and_depth_source_contracts_are_locked() -> None:
    core = (ROOT / "src/shaders/atmosphere/evaluation_core.wgsl").read_text(
        encoding="utf-8"
    )
    shared = (ROOT / "src/shaders/atmosphere/scattering.wgsl").read_text(encoding="utf-8")
    prometheus = (ROOT / "src/shaders/atmosphere/prometheus_aerial.wgsl").read_text(
        encoding="utf-8"
    )
    source_registry = (ROOT / "src/shader_sources.rs").read_text(encoding="utf-8")
    offline = (ROOT / "src/terrain/renderer/offline.rs").read_text(encoding="utf-8")
    pt_driver = (ROOT / "src/path_tracing/hybrid_compute/render_terrain.rs").read_text(
        encoding="utf-8"
    )

    assert "view_ws.xzy" in shared
    assert "sign(c)" not in core and "sign(c)" not in shared and "sign(c)" not in prometheus
    assert "fn atmosphere_ray_hits_ground" in shared
    assert "sun_visible && alignment>=cos(sun_radius)" in shared
    assert "textureLoad(prometheus_depth_aov" in prometheus
    assert "textureLoad(prometheus_visibility_aov" in prometheus
    assert "depth != depth" not in prometheus
    assert "prometheus_reference_sky" not in prometheus
    assert "prometheus_reference_" not in prometheus
    miss_branch = prometheus.split("if (visibility < 0.5)", 1)[1].split("return;", 1)[0]
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
    assert "let sun_intensity = aether_eval_clamp_radiometric_scale(" in prometheus
    assert "let atmosphere_exposure = aether_eval_clamp_radiometric_scale(" in prometheus
    assert "aether_eval_clamp_hdr_radiance(\n        surface_or_environment" in prometheus
    assert "aether_eval_sample_accumulated_scattering(" in shared
    assert "aether_eval_sample_accumulated_scattering(" in prometheus
    assert "aether_eval_segment_transmittance(" in prometheus
    assert "fn aether_eval_load_scattering_texel" in core
    assert "let fraction = fract(coordinates);" in core
    assert "height_side < 2" in core
    assert "nu_side < 2" in core
    assert "sun_side < 2" in core
    assert "view_side < 2" in core
    assert "let h00 = aether_eval_spherical_altitude(" in core
    assert "let h15 = aether_eval_spherical_altitude(" in core
    assert "fn aether_eval_spherical_endpoint_mus(" in core
    assert "mix(bounded_camera_height_m, bounded_surface_height_m" not in core
    assert "AETHER_PT_CIE_XYZ" not in prometheus
    assert "fn prometheus_mu_to_unit" not in prometheus
    assert "fn prometheus_load_scattering_texel" not in prometheus
    assert "fn atmosphere_mu_to_unit" not in shared
    assert "fn atmosphere_load_scattering" not in shared
    assert "prometheus_load_boundary_mean_transmittance" not in prometheus
    assert "scatter_fraction" not in prometheus
    assert (
        "camera_scattering - transmittance * endpoint_scattering,\n"
        "        vec3<f32>(0.0),"
        in prometheus
    )
    assert "surface_or_environment * transmittance + finite_inscatter" in prometheus
    assert "let endpoint_mus = aether_eval_spherical_endpoint_mus(" in prometheus
    assert "endpoint_mus.y,\n        endpoint_mus.x," in prometheus
    aerial_loader = prometheus.split("fn prometheus_load_aerial_transmittance", 1)[1].split(
        "@compute", 1
    )[0]
    assert "0.5 * (clamp(mu_view, -1.0, 1.0) + 1.0)" in aerial_loader
    assert "prometheus_mu_to_unit(mu_view)" not in aerial_loader
    assert ".rgb" not in aerial_loader
    assert "-> f32" in prometheus
    assert "aerial_mean_transmittance" in prometheus
    assert "aerial_transport" not in prometheus
    assert "tonemap_apply_operator" in prometheus
    assert "TONEMAP_OPERATOR_REINHARD" in prometheus
    assert "linear_to_srgb" not in prometheus
    assert 'include_str!("shaders/includes/tonemap_common.wgsl")' in source_registry
    production_registry = source_registry.split("#[cfg(test)]", 1)[0]
    assert production_registry.count(
        'include_str!("shaders/atmosphere/evaluation_core.wgsl")'
    ) == 3
    assert "copy_texture_to_texture" in offline
    assert "LoadOp::Clear" in offline
    assert "aov_frames.get_texture(AovKind::Depth)" in pt_driver
    assert "aov_frames.get_texture(AovKind::Visibility)" in pt_driver
    assert not (ROOT / "src/shaders/hybrid_aether_post.wgsl").exists()
