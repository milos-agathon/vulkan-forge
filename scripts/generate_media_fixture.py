#!/usr/bin/env python3
"""Generate the tracked NEPHELE cloud-over-terrain reference fixture."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/nephele/fixture"
MANIFEST = ROOT / "tests/nephele/fixture-manifest.json"
DEM_WIDTH = DEM_HEIGHT = 16
WIDTH = HEIGHT = 64  # TerrainRenderParams' physical minimum and tracked full viewport.
VIEWPORT_WIDTH = VIEWPORT_HEIGHT = 64
CROP_X = CROP_Y = 0
REFERENCE_SEED = 0x4E455048
SCENE_INPUT_FILES = {
    "camera": "camera.json",
    "terrain_dem": "terrain-dem.npy",
    "terrain": "terrain.json",
    "medium": "medium.json",
    "sun": "sun.json",
    "atmosphere": "atmosphere.json",
    "material": "material.json",
    "exposure": "exposure.json",
    "tonemap": "tonemap.json",
    "crop": "crop.json",
}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_tracked_revision() -> None:
    required = [
        "scripts/generate_media_fixture.py",
        "scripts/nephele_fixture_masks.py",
        "scripts/record_media_reference_convergence.py",
        *_reference_source_inputs().keys(),
        *[f"tests/nephele/fixture/{name}" for name in SCENE_INPUT_FILES.values()],
    ]
    subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", *required],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("acceptance reference generation requires a clean tracked revision")


def _reference_source_inputs() -> dict[str, str]:
    paths = [
        Path("scripts/generate_media_fixture.py"),
        Path("python/forge3d/media.py"),
        Path("src/media_py.rs"),
        Path("src/path_tracing/hybrid_compute/media_reference.rs"),
        Path("src/path_tracing/hybrid_compute/terrain_heightfield.rs"),
        Path("src/path_tracing/hybrid_compute/render_terrain.rs"),
        Path("src/terrain/camera.rs"),
        Path("src/terrain/mod.rs"),
        Path("src/shader_sources.rs"),
        Path("src/shaders/hybrid_terrain_traversal.wgsl"),
        Path("src/shaders/nephele_terrain_trace_adapter.wgsl"),
        *sorted(path.relative_to(ROOT) for path in (ROOT / "src/media").glob("*.rs")),
    ]
    return {path.as_posix(): _sha256(ROOT / path) for path in paths}


def _terrain_media_reference_module_source() -> bytes:
    inputs = (
        ("src/shaders/sdf_primitives.wgsl", False),
        ("src/shaders/sdf_operations.wgsl", True),
        ("src/shaders/hybrid_traversal.wgsl", True),
        ("src/shaders/hybrid_terrain_traversal.wgsl", True),
        ("src/shaders/atmosphere/prometheus_spectral_reference.wgsl", False),
        ("src/shaders/hybrid_kernel.wgsl", True),
    )
    assembled = []
    for relative, strip in inputs:
        source = (ROOT / relative).read_text(encoding="utf-8")
        if strip:
            source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith("#include")
            )
        assembled.append(source)
    hybrid_kernel = "\n".join(assembled).encode("utf-8")
    query_shader = (ROOT / "src/shaders/nephele_terrain_trace_adapter.wgsl").read_bytes()
    return hybrid_kernel + b"\n" + query_shader


def _scene_input_records() -> dict[str, dict[str, str]]:
    return {role: _record(name) for role, name in SCENE_INPUT_FILES.items()}


def _scene_input_hashes() -> dict[str, str]:
    return {role: record["sha256"] for role, record in _scene_input_records().items()}


def _color_pipeline(scene: dict[str, dict[str, str]]) -> dict[str, str]:
    return {
        "linear_input": "linear-sRGB",
        "tonemap": "aces-fitted",
        "output_encoding": "IEC 61966-2-1 sRGB",
        "quantization": "round-to-nearest uint8",
        "exposure": scene["exposure"]["sha256"],
        "tonemap_contract": scene["tonemap"]["sha256"],
        "crop": scene["crop"]["sha256"],
    }


def _fixture_manifest_record(
    *,
    converged: bool,
    provenance: dict[str, Any],
    scene: dict[str, dict[str, str]],
    files: dict[str, dict[str, str]],
    mask_generator: dict[str, str],
    mask_rules: dict[str, str],
    fixture_id: str = "nephele-cloud-over-terrain-v2",
    revision: int = 2,
) -> dict[str, Any]:
    return {
        "schema": "forge3d.nephele.fixture_manifest/3",
        "status": "APPROVED" if converged else "UNRESOLVED",
        "fixture_id": fixture_id,
        "revision": revision,
        "source_revision": provenance["source_revision"],
        "color_pipeline": _color_pipeline(scene),
        "scene_inputs": scene,
        "files": files,
        "mask_generator": mask_generator,
        "mask_rules": mask_rules,
    }


def _scene_inputs() -> tuple[np.ndarray, np.ndarray]:
    mask_contract = __import__("scripts.nephele_fixture_masks", fromlist=["EXPECTED_RULES"])
    FIXTURE.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:DEM_HEIGHT, 0:DEM_WIDTH].astype(np.float32)
    terrain = (5.0 + 0.45 * x + 8.0 * np.exp(-((x - 8.0) ** 2) / 5.0)).astype(np.float32)
    _save(FIXTURE / "terrain-dem.npy", terrain)
    density = np.array(
        [
            0.05, 0.25, 0.05, 0.20, 0.80, 0.25, 0.05, 0.30, 0.05,
            0.10, 0.55, 0.10, 0.45, 1.00, 0.55, 0.10, 0.50, 0.10,
            0.02, 0.20, 0.02, 0.15, 0.65, 0.20, 0.02, 0.25, 0.02,
        ],
        dtype=np.float32,
    ).reshape(3, 3, 3)
    r16 = np.rint(np.clip(density, 0.0, 1.0) * 65535.0).astype(np.uint16)
    transported = (r16.astype(np.float32) / np.float32(65535.0)).astype(np.float16)
    transported_f32 = transported.astype(np.float32)
    transported_sha256 = hashlib.sha256(
        transported.view(np.uint16).astype("<u2", copy=False).tobytes()
    ).hexdigest()
    sigma_a = [0.01, 0.012, 0.015]
    sigma_s = [0.055, 0.05, 0.045]
    sigma_t = [a + s for a, s in zip(sigma_a, sigma_s)]
    sigma_t_f32 = max(
        float(np.float32(a) + np.float32(s)) for a, s in zip(sigma_a, sigma_s)
    )

    def next_up(value: np.float32) -> np.float32:
        if value == 0.0 or not np.isfinite(value):
            return value
        return np.asarray(value.view(np.uint32) + np.uint32(1), dtype=np.uint32).view(np.float32)

    def outward_mul(left: np.float32, right: np.float32) -> np.float32:
        exact = float(left) * float(right)
        rounded = np.float32(left * right)
        return next_up(rounded) if float(rounded) < exact else rounded

    def trilinear_upper(maximum: np.float32) -> np.float32:
        if maximum == 0.0:
            return maximum
        operations = 13.0
        unit_roundoff = 1.0 / float(1 << 24)
        half_min_subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0))) * 0.5
        denominator = 1.0 - operations * unit_roundoff
        gamma = operations * unit_roundoff / denominator
        exact = float(maximum) * (1.0 + gamma) + operations * half_min_subnormal / denominator
        rounded = np.float32(exact)
        return next_up(rounded) if float(rounded) < exact else rounded

    majorant_cells: list[float] = []
    for z in range(transported_f32.shape[0]):
        for y in range(transported_f32.shape[1]):
            for x in range(transported_f32.shape[2]):
                neighborhood = transported_f32[
                    max(0, z - 1) : min(transported_f32.shape[0], z + 2),
                    max(0, y - 1) : min(transported_f32.shape[1], y + 2),
                    max(0, x - 1) : min(transported_f32.shape[2], x + 2),
                ]
                authored = trilinear_upper(np.float32(neighborhood.max()))
                physical = outward_mul(authored, np.float32(1.0))
                majorant_cells.append(float(outward_mul(physical, np.float32(sigma_t_f32))))
    _json(FIXTURE / "camera.json", {
        "schema": "forge3d.nephele.camera/1", "fov_y": 45.0,
        "terrain_camera": {
            "mode": "mesh:yup", "radius": 50.990195, "phi_deg": 90.0,
            "theta_deg": 78.690068, "target": [0.0, 15.0, 0.0],
        },
    })
    _json(FIXTURE / "terrain.json", {
        "schema": "forge3d.nephele.terrain/1", "dem": "terrain-dem.npy",
        "dem_sha256": _sha256(FIXTURE / "terrain-dem.npy"), "dimensions": [DEM_WIDTH, DEM_HEIGHT],
        "spacing": [4.0, 4.0], "exaggeration": 1.0,
    })
    _json(FIXTURE / "medium.json", {
        "schema": "forge3d.nephele.heterogeneous_medium/2",
        "domain": {"bounds_min": [-30.0, 4.0, -30.0], "bounds_max": [30.0, 44.0, 30.0], "grid_shape": [3, 3, 3]},
        "density_r16": r16.reshape(-1).astype(int).tolist(),
        "density_transport": {
            "schema": "forge3d.nephele.density_transport/1",
            "decode": "unorm16-div-65535-as-f32-then-ieee-f16-rne",
            "storage": "ieee-f16-bits-little-endian",
            "sampling": "normalized-clamp-to-edge-linear-texel-center-uN-minus-0.5",
            "f16_sha256": transported_sha256,
        },
        "majorant_cells": majorant_cells,
        "majorant_transport": {
            "schema": "forge3d.nephele.majorant_transport/1",
            "grid_shape": [3, 3, 3],
            "query": "floor(clamp(unit,0,1)*N)-clamped-to-N-minus-1",
            "construction": "3x3x3-clamped-neighborhood-trilinear-outward-then-f32-extinction-outward",
        },
        "transport": {
            "sigma_t_spectrum": sigma_t,
            "sigma_t_max_channel": max(sigma_t),
            "extinction_channel": sigma_t.index(max(sigma_t)),
            "slab_axis": 2,
        },
        "sigma_a": sigma_a, "sigma_s": sigma_s,
        "phase": {"kind": "henyey_greenstein", "g": 0.45}, "density_scale": 1.0,
    })
    _json(FIXTURE / "sun.json", {
        "schema": "forge3d.nephele.sun/1", "azimuth_deg": 315.0, "elevation_deg": 18.0,
        "intensity": 3.0, "color": [1.0, 0.97, 0.92],
    })
    _json(FIXTURE / "atmosphere.json", {
        "schema": "forge3d.nephele.atmosphere_input/1", "kind": "analytic_clear_sky",
        "environment_intensity": 0.25,
    })
    _json(FIXTURE / "material.json", {
        "schema": "forge3d.nephele.material/1",
        "albedo": [0.6, 0.6, 0.6],
        "metallic": 0.0,
        "roughness": 1.0,
        "triplanar_scale": 1.0,
        "normal_strength": 0.0,
        "blend_sharpness": 1.0,
        "colormap_strength": 0.0,
        "albedo_mode": "material",
    })
    _json(FIXTURE / "exposure.json", {"schema": "forge3d.nephele.exposure/1", "value": 1.0})
    _json(FIXTURE / "tonemap.json", {
        "schema": "forge3d.nephele.tonemap/1",
        "operator": "aces-fitted",
        "formula": "clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0,1)",
        "encoding": "IEC 61966-2-1 sRGB",
        "quantization": "round-to-nearest uint8",
    })
    _json(FIXTURE / "crop.json", {
        "schema": "forge3d.nephele.crop/1",
        "x": CROP_X, "y": CROP_Y, "width": WIDTH, "height": HEIGHT,
        "full_viewport": [VIEWPORT_WIDTH, VIEWPORT_HEIGHT],
    })
    _json(FIXTURE / "mask-rules.json", {
        "schema": "forge3d.nephele.mask_rules/3",
        "rules": mask_contract.EXPECTED_RULES,
        "technical_contracts": mask_contract.TECHNICAL_CONTRACTS,
    })
    return terrain, r16.astype(np.float32) / np.float32(65535.0)


def _aces_srgb8(linear: np.ndarray) -> np.ndarray:
    value = np.maximum(linear.astype(np.float32), 0.0)
    value = np.clip(value * (2.51 * value + 0.03) / (value * (2.43 * value + 0.59) + 0.14), 0.0, 1.0)
    value = np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)
    return np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def _spatial_tiles() -> list[dict[str, int]]:
    return [
        {"x": CROP_X, "y": CROP_Y + y, "width": WIDTH, "height": 1}
        for y in range(HEIGHT)
    ]


def _validated_grid_density(
    density: np.ndarray, medium_data: dict[str, Any]
) -> tuple[np.ndarray, float]:
    tracked_shape = medium_data["domain"]["grid_shape"]
    if (
        not isinstance(tracked_shape, list)
        or len(tracked_shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in tracked_shape)
    ):
        raise ValueError("medium domain.grid_shape must contain exactly three positive integers")
    native_shape = tuple(int(value) for value in reversed(density.shape))
    if native_shape != tuple(tracked_shape):
        raise ValueError(
            f"loaded density shape {density.shape} maps to native grid shape {native_shape}, "
            f"not tracked grid_shape {tuple(tracked_shape)}"
        )
    density_scale = medium_data.get("density_scale")
    if (
        isinstance(density_scale, bool)
        or not isinstance(density_scale, (int, float))
        or not np.isfinite(density_scale)
        or density_scale <= 0.0
    ):
        raise ValueError("medium density_scale must be finite and positive")
    return np.ascontiguousarray(density, dtype=np.float32), float(density_scale)


def _render_batch(
    payload: tuple[np.ndarray, np.ndarray, dict[str, int], int, int]
) -> dict[str, Any]:
    from forge3d.media import Medium, _native_module

    terrain, density, tile, samples_per_pixel, seed = payload
    camera = json.loads((FIXTURE / "camera.json").read_text(encoding="utf-8"))
    camera["terrain_camera"]["target"] = tuple(camera["terrain_camera"]["target"])
    terrain_data = json.loads((FIXTURE / "terrain.json").read_text(encoding="utf-8"))
    medium_data = json.loads((FIXTURE / "medium.json").read_text(encoding="utf-8"))
    material = json.loads((FIXTURE / "material.json").read_text(encoding="utf-8"))
    sun = json.loads((FIXTURE / "sun.json").read_text(encoding="utf-8"))
    atmosphere = json.loads((FIXTURE / "atmosphere.json").read_text(encoding="utf-8"))
    exposure = json.loads((FIXTURE / "exposure.json").read_text(encoding="utf-8"))
    density, density_scale = _validated_grid_density(density, medium_data)
    medium = Medium.grid3d(
        medium_data["sigma_a"], medium_data["sigma_s"], density,
        (medium_data["domain"]["bounds_min"], medium_data["domain"]["bounds_max"]),
        phase="henyey_greenstein", g=medium_data["phase"]["g"],
        density_scale=density_scale, version=1,
    )
    return _native_module()._render_volumetric_reference(
        medium._native, np.ascontiguousarray(terrain, dtype=np.float32),
        tile["width"], tile["height"], camera,
        spacing=tuple(terrain_data["spacing"]), exaggeration=terrain_data["exaggeration"],
        albedo=tuple(material["albedo"]),
        sun_azimuth_deg=sun["azimuth_deg"], sun_elevation_deg=sun["elevation_deg"],
        sun_intensity=sun["intensity"], sun_color=tuple(sun["color"]),
        environment_intensity=atmosphere["environment_intensity"], exposure=exposure["value"],
        samples_per_pixel=samples_per_pixel,
        homogeneous_medium_reach=120.0, seed=seed,
        full_viewport=(VIEWPORT_WIDTH, VIEWPORT_HEIGHT),
        crop=(tile["x"], tile["y"], tile["width"], tile["height"]),
    )


DIAGNOSTIC_FIELDS = {
    "adapter", "backend", "driver", "source_revision", "sample_count", "step_count",
    "majorant_proof", "majorant_valid", "executed_multi_scatter", "host_visible_bytes",
    "density_device_local_bytes", "majorant_device_local_bytes", "froxel_device_local_bytes",
    "staging_readback_bytes", "temporal_history_decision", "temporal_history_reason",
    "single_scatter_luminance", "multiple_scatter_luminance",
    "energy_accounting_residual",
}


def _native_runtime_record(wheel: Path | None, source_revision: str) -> dict[str, str] | None:
    if wheel is None:
        return None
    from forge3d.media import _native_module

    wheel = wheel.resolve()
    if not wheel.is_file():
        raise ValueError(f"native wheel does not exist: {wheel}")
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name for name in archive.namelist()
            if name.startswith("forge3d/_forge3d")
            and Path(name).suffix.lower() in {".so", ".dylib", ".pyd"}
        ]
        if len(members) != 1:
            raise ValueError("native wheel must contain exactly one forge3d native member")
        member = members[0]
        member_bytes = archive.read(member)
    native_path = Path(_native_module().__file__).resolve()
    native_sha256 = _sha256(native_path)
    if hashlib.sha256(member_bytes).hexdigest() != native_sha256:
        raise ValueError("imported forge3d native module does not match the supplied wheel")
    return {
        "source_revision": source_revision,
        "wheel_filename": wheel.name,
        "wheel_sha256": _sha256(wheel),
        "wheel_native_member": member,
        "native_sha256": native_sha256,
    }


def _render_reference(
    terrain: np.ndarray,
    density: np.ndarray,
    samples_per_pixel: int,
    native_wheel: Path | None,
    diagnostic: bool,
    requested_workers: int | None,
) -> None:
    tiles = _spatial_tiles()
    worker_count = min(requested_workers or (os.cpu_count() or 1), len(tiles))
    if worker_count < 1:
        raise ValueError("worker count must be positive")
    payloads = [
        (terrain, density, tile, samples_per_pixel, REFERENCE_SEED)
        for tile in tiles
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(_render_batch, payloads))

    def assembled(key: str, dtype: np.dtype[Any]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(result[key], dtype=dtype) for result in results], axis=0
        )

    beauty = assembled("beauty", np.dtype(np.float32))
    transmittance = assembled("transmittance", np.dtype(np.float32))
    reference_slice = assembled("terrain_slice", np.dtype(np.float32))
    _save(FIXTURE / "reference-rgb.npy", _aces_srgb8(beauty))
    _save(FIXTURE / "reference-transmittance.npy", transmittance)
    _save(FIXTURE / "reference-in-scatter.npy", assembled("in_scatter", np.dtype(np.float32)))
    _save(FIXTURE / "reference-cloud-shadow-aov.npy", assembled("cloud_shadow", np.dtype(np.float32)))
    _save(FIXTURE / "reference-optical-depth.npy", -np.log(np.maximum(transmittance, np.finfo(np.float32).tiny)))
    _save(FIXTURE / "reference-terrain-slice.npy", reference_slice)
    terrain_hit = assembled("terrain_hit", np.dtype(np.uint8))
    lighting = assembled("media_lighting_visibility", np.dtype(np.uint8))
    if not np.isin(terrain_hit, [0, 1]).all() or not np.isin(lighting, [0, 1, 2]).all():
        raise RuntimeError("reference classification returned an unknown code")
    _save(FIXTURE / "reference-terrain-hit.npy", terrain_hit.astype(np.bool_))
    _save(FIXTURE / "reference-media-lighting-visibility.npy", lighting)
    camera_contract = {
        key: np.asarray(results[0]["camera_contract"][key], dtype=np.float32).tolist()
        for key in ("origin", "look_at", "up", "right", "forward")
    }
    camera_contract["fov_y"] = float(np.float32(results[0]["camera_contract"]["fov_y"]))
    for result in results[1:]:
        for key, expected in camera_contract.items():
            current = np.asarray(result["camera_contract"][key], dtype=np.float32)
            if not np.array_equal(current, np.asarray(expected, dtype=np.float32)):
                raise RuntimeError(f"reference camera contract changed across spatial tiles: {key}")
    diagnostics = dict(results[0]["diagnostics"])
    if set(diagnostics) != DIAGNOSTIC_FIELDS:
        raise RuntimeError("reference diagnostics fields differ from the provenance contract")
    for result in results[1:]:
        current = dict(result["diagnostics"])
        if set(current) != DIAGNOSTIC_FIELDS:
            raise RuntimeError("reference diagnostics fields changed across spatial tiles")
        for key in (
            "adapter", "backend", "driver", "source_revision", "majorant_proof",
            "majorant_valid", "single_scatter_luminance",
            "multiple_scatter_luminance", "energy_accounting_residual",
        ):
            if current[key] != diagnostics[key]:
                raise RuntimeError(f"reference diagnostic {key} changed across spatial tiles")
    diagnostics["sample_count"] = sum(int(result["diagnostics"]["sample_count"]) for result in results)
    diagnostics["step_count"] = sum(int(result["diagnostics"]["step_count"]) for result in results)
    diagnostics["executed_multi_scatter"] = any(bool(result["diagnostics"]["executed_multi_scatter"]) for result in results)
    diagnostics["host_visible_bytes"] = sum(int(result["diagnostics"]["host_visible_bytes"]) for result in results)
    for key in ("density_device_local_bytes", "majorant_device_local_bytes", "froxel_device_local_bytes", "staging_readback_bytes"):
        diagnostics[key] = sum(int(result["diagnostics"][key]) for result in results)
    if (
        diagnostics["sample_count"] != WIDTH * HEIGHT * samples_per_pixel
        or diagnostics["step_count"] <= 0
        or not isinstance(diagnostics["adapter"], str)
        or not diagnostics["adapter"]
        or not isinstance(diagnostics["backend"], str)
        or not diagnostics["backend"]
        or not isinstance(diagnostics["driver"], str)
        or not isinstance(diagnostics["source_revision"], str)
        or len(diagnostics["source_revision"]) != 40
        or diagnostics["majorant_proof"] != "TrilinearConvexHull"
        or diagnostics["majorant_valid"] is not True
        or diagnostics["executed_multi_scatter"] is not True
        or diagnostics["host_visible_bytes"] != WIDTH * HEIGHT * (5 * 3 * 4 + 4 + 2)
        or any(diagnostics[key] != 0 for key in (
            "density_device_local_bytes", "majorant_device_local_bytes",
            "froxel_device_local_bytes", "staging_readback_bytes",
        ))
        or diagnostics["temporal_history_decision"] != "not_applicable"
        or diagnostics["temporal_history_reason"] != "independent reference samples do not reuse temporal history"
        or diagnostics["single_scatter_luminance"] is not None
        or diagnostics["multiple_scatter_luminance"] is not None
        or diagnostics["energy_accounting_residual"] is not None
    ):
        raise RuntimeError("reference diagnostics violate the exact runtime invariants")
    source_revision = _source_revision()
    native_runtime = _native_runtime_record(native_wheel, str(diagnostics["source_revision"]))
    acceptance_eligible = not diagnostic and native_runtime is not None
    if acceptance_eligible and diagnostics["source_revision"] != source_revision:
        raise RuntimeError("reference native runtime source revision differs from the clean checkout")
    _json(FIXTURE / "reference-provenance.json", {
        "schema": "forge3d.nephele.reference_provenance/2",
        "acceptance_eligible": acceptance_eligible,
        "diagnostic_reason": None if acceptance_eligible else "explicit diagnostic generation; cannot seed acceptance convergence",
        "algorithm": "integrated-hybrid-terrain-ratio-delta-tracking-reference",
        "samples_per_pixel": samples_per_pixel,
        "seed": REFERENCE_SEED,
        "sample_identity": {
            "algorithm": "per-pixel-absolute-sample-index-v1",
            "seed": REFERENCE_SEED,
            "range": [0, samples_per_pixel],
        },
        "spatial_tiles": tiles,
        "runtime_partition": {
            "kind": "process_pool_max_workers",
            "value": worker_count,
        },
        "source_revision": source_revision,
        "source_inputs": _reference_source_inputs(),
        "scene_inputs": _scene_input_hashes(),
        "assembled_sources": {
            "terrain_media_reference_module_sha256": hashlib.sha256(
                _terrain_media_reference_module_source()
            ).hexdigest(),
        },
        "native_runtime": native_runtime,
        "camera_contract": camera_contract,
        "full_viewport": [VIEWPORT_WIDTH, VIEWPORT_HEIGHT],
        "crop": [CROP_X, CROP_Y, WIDTH, HEIGHT],
        "diagnostics": diagnostics,
    })


def _manifest() -> None:
    from scripts.nephele_fixture_masks import generate_masks

    generate_masks(FIXTURE, FIXTURE / "mask-rules.json", FIXTURE)
    scene = _scene_input_records()
    files = {role: _record(name) for role, name in {
        "reference_rgb": "reference-rgb.npy", "reference_transmittance": "reference-transmittance.npy",
        "reference_in_scatter": "reference-in-scatter.npy", "reference_cloud_shadow_aov": "reference-cloud-shadow-aov.npy",
        "reference_optical_depth": "reference-optical-depth.npy", "sky_cloud_mask": "sky-cloud-mask.npy",
        "godray_roi_mask": "godray-roi-mask.npy", "terrain_mask": "terrain-mask.npy",
        "cloud_shadow_mask": "cloud-shadow-mask.npy", "shaft_mask": "shaft-mask.npy",
        "reference_terrain_slice": "reference-terrain-slice.npy",
        "reference_terrain_hit": "reference-terrain-hit.npy",
        "reference_media_lighting_visibility": "reference-media-lighting-visibility.npy",
    }.items()}
    generator = ROOT / "scripts/nephele_fixture_masks.py"
    convergence_path = FIXTURE / "reference-convergence.json"
    convergence = json.loads(convergence_path.read_text(encoding="utf-8")) if convergence_path.is_file() else {}
    provenance = json.loads((FIXTURE / "reference-provenance.json").read_text(encoding="utf-8"))
    converged = (
        provenance.get("acceptance_eligible") is True
        and
        convergence.get("status") == "CONVERGED"
        and convergence.get("final", {}).get("rgb_sha256") == _sha256(FIXTURE / "reference-rgb.npy")
        and convergence.get("generator_sha256") == _sha256(ROOT / "scripts/record_media_reference_convergence.py")
    )
    if not converged:
        _json(convergence_path, {
            "schema": "forge3d.nephele.reference_convergence/2",
            "status": "UNRESOLVED",
            "criterion": "actual downstream Gate 3 and approved Gate 4 metrics over exact identical reference-derived masks",
            "nested_prefix": False,
            "final": {
                "samples_per_pixel": provenance["samples_per_pixel"],
                "rgb_path": "tests/nephele/fixture/reference-rgb.npy",
                "rgb_sha256": _sha256(FIXTURE / "reference-rgb.npy"),
                "provenance_path": "tests/nephele/fixture/reference-provenance.json",
                "provenance_sha256": _sha256(FIXTURE / "reference-provenance.json"),
            },
            "next_samples_per_pixel_if_unresolved": (
                provenance["samples_per_pixel"] * 2
                if provenance.get("acceptance_eligible") is True else None
            ),
            "generator": "scripts/record_media_reference_convergence.py",
            "generator_sha256": _sha256(ROOT / "scripts/record_media_reference_convergence.py"),
        })
    _json(MANIFEST, _fixture_manifest_record(
        converged=converged,
        provenance=provenance,
        scene=scene,
        files=files,
        mask_generator={
            "path": generator.relative_to(ROOT).as_posix(),
            "sha256": _sha256(generator),
        },
        mask_rules={
            "path": (FIXTURE / "mask-rules.json").relative_to(ROOT).as_posix(),
            "sha256": _sha256(FIXTURE / "mask-rules.json"),
        },
    ))


def _record(name: str) -> dict[str, str]:
    path = FIXTURE / name
    return {"path": path.relative_to(ROOT).as_posix(), "artifact": name, "sha256": _sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-pixel", type=int, required=True)
    parser.add_argument("--native-wheel", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.samples_per_pixel < 1:
        parser.error("--samples-per-pixel must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if not args.diagnostic and args.native_wheel is None:
        parser.error("acceptance reference generation requires --native-wheel")
    if not args.diagnostic:
        _require_clean_tracked_revision()
    terrain, density = _scene_inputs()
    if not args.diagnostic:
        _require_clean_tracked_revision()
    _render_reference(
        terrain,
        density,
        args.samples_per_pixel,
        args.native_wheel,
        args.diagnostic,
        args.workers,
    )
    _manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
