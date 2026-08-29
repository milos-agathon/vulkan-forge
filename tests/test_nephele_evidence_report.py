"""Malicious-regression tests for the NEPHELE physical evidence verifier."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import scripts.nephele_majorant_domain_probe as majorant_probe_module
import scripts.record_media_reference_convergence as convergence_module
from scripts.generate_media_fixture import _color_pipeline, _fixture_manifest_record

from scripts.nephele_evidence_report import (
    EvidenceError,
    RAW_FILES,
    REALTIME_DIAGNOSTIC_KEYS,
    REQUIRED_JUNIT_CASES,
    _accumulator_from_array,
    _derived_cache_key,
    _policy_evaluator,
    _reference_module_sha256,
    _validate_realtime_diagnostics,
    _verify_transported_medium,
    _gate2,
    _verify_reference_provenance,
    _verify_fixture_manifest,
    build_report,
    main,
)
from scripts.nephele_fixture_masks import EXPECTED_RULES, TECHNICAL_CONTRACTS, generate_masks
from scripts.nephele_heterogeneous_comparator import (
    SEED as COMPARATOR_SEED,
    produce as produce_comparator,
    produce_rr_evidence,
)
from scripts.nephele_majorant_domain_probe import (
    DENSITY_SAMPLING,
    MAJORANT_QUERY,
    PROBE_MAPPING,
    canonical_majorant_cells,
    domain_coverage_sha256,
    produce as produce_majorant_probe,
    represented_density,
)
from scripts.nephele_shader_analyzer import analyze_shaders
from scripts.run_media_physical_capture import (
    DIAGNOSTIC_KEYS,
    _retain_reference_wheel,
    _validate_candidate_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _remove_large_synthetic_evidence(request: pytest.FixtureRequest):
    """Release each synthetic raw-evidence tree before the next test."""
    yield
    temporary = request.node.funcargs.get("tmp_path")
    if isinstance(temporary, Path):
        shutil.rmtree(temporary, ignore_errors=True)


def _identity(algorithm: str, **parameters: object) -> dict:
    payload = {"algorithm": algorithm, "parameters": parameters}
    return {
        **payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


SEED = _identity("splitmix64", seed=7)
HOMOGENEOUS_MAPPING = _identity(
    "canonical-sample-identity-v1",
    frame_u64=0x4E4550481001,
    fields=["frame", "pixel", "sample", "bounce", "dimension"],
)
HETEROGENEOUS_MAPPING = _identity(
    "canonical-grid3d-z-ray-v1",
    coordinate_frame_u64=0x4E4550481101,
    tracking_frame_u64=0x4E4550481102,
    fields=["sample", "dimension"],
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)


def _acc(count: int, value: float) -> dict[str, float | int]:
    return {"count": count, "sum": count * value, "sum_squares": count * value * value}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    repo, artifact = tmp_path / "repo", tmp_path / "evidence"
    fixture = repo / "tests/nephele/fixture"
    artifact.mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / "src/shaders").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_media_physical_capture.py", repo / "scripts/run_media_physical_capture.py")
    shutil.copy2(ROOT / "src/media_py.rs", repo / "src/media_py.rs")
    shutil.copy2(ROOT / "scripts/nephele_fixture_masks.py", repo / "scripts/nephele_fixture_masks.py")
    shutil.copy2(ROOT / "scripts/nephele_shader_analyzer.py", repo / "scripts/nephele_shader_analyzer.py")
    shutil.copy2(ROOT / "scripts/nephele_heterogeneous_comparator.py", repo / "scripts/nephele_heterogeneous_comparator.py")
    shutil.copy2(ROOT / "scripts/nephele_majorant_domain_probe.py", repo / "scripts/nephele_majorant_domain_probe.py")
    shutil.copy2(ROOT / "scripts/record_media_reference_convergence.py", repo / "scripts/record_media_reference_convergence.py")
    shutil.copy2(ROOT / "scripts/generate_media_fixture.py", repo / "scripts/generate_media_fixture.py")
    (repo / "python/forge3d").mkdir(parents=True)
    shutil.copy2(ROOT / "python/forge3d/media.py", repo / "python/forge3d/media.py")
    (repo / "src/shader_sources.rs").write_text("// synthetic source-assembly owner\n", encoding="utf-8")
    for relative in (
        "src/path_tracing/hybrid_compute/media_reference.rs",
        "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
        "src/path_tracing/hybrid_compute/render_terrain.rs",
        "src/terrain/camera.rs",
        "src/terrain/mod.rs",
        "src/shaders/hybrid_terrain_traversal.wgsl",
        "src/shaders/nephele_terrain_trace_adapter.wgsl",
        "src/shaders/sdf_primitives.wgsl",
        "src/shaders/sdf_operations.wgsl",
        "src/shaders/hybrid_traversal.wgsl",
        "src/shaders/atmosphere/prometheus_spectral_reference.wgsl",
        "src/shaders/hybrid_kernel.wgsl",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".rs":
            target.write_text("// synthetic tracked reference dependency\n", encoding="utf-8")
        else:
            target.write_text("// synthetic tracked shader dependency\n", encoding="utf-8")
    (repo / "src/shaders/test.wgsl").write_text(
        "fn shadow_ok() -> f32 { return 1.0; }\n@compute @workgroup_size(1) fn main() { let x = shadow_ok(); }\n",
        encoding="utf-8",
    )
    (repo / "src/shader_loader.rs").write_text(
        'const SHADER: &str = include_str!("shaders/test.wgsl");\nfn load() { let _ = wgpu::ShaderSource::Wgsl(SHADER.into()); }\n',
        encoding="utf-8",
    )

    shape = (12, 12)
    reference = np.full((*shape, 3), 128, dtype=np.uint8)
    dark = np.zeros_like(reference)
    scalar = np.ones(shape, dtype=np.float32)
    vector = np.full((*shape, 3), 0.5, dtype=np.float32)
    depth = np.full(shape, 3.0, dtype=np.float32)
    terrain_hit = np.indices(shape).sum(axis=0) % 2 == 0
    lighting = np.zeros(shape, dtype=np.uint8)
    lighting[:11, :11] = 1
    fixture_arrays = {
        "reference-rgb.npy": reference,
        "reference-transmittance.npy": vector,
        "reference-in-scatter.npy": vector,
        "reference-cloud-shadow-aov.npy": vector,
        "reference-optical-depth.npy": scalar,
        "reference-terrain-slice.npy": depth,
        "reference-terrain-hit.npy": terrain_hit,
        "reference-media-lighting-visibility.npy": lighting,
    }
    for name, value in fixture_arrays.items():
        _save(fixture / name, value)
    rules = {"schema": "forge3d.nephele.mask_rules/3", "rules": EXPECTED_RULES, "technical_contracts": TECHNICAL_CONTRACTS}
    rules_path = fixture / "mask-rules.json"
    _json(rules_path, rules)
    generate_masks(fixture, rules_path, fixture)
    slab_path = repo / "tests/nephele/homogeneous-slab.json"
    _json(slab_path, {
        "schema": "forge3d.nephele.homogeneous_slab/1",
        "analytic_comparator": "exp(-sigma_t_max_channel * density * distance)",
        "density": 1.0, "distance": 1.0,
        "sigma_a": [0.0, 0.0, 0.0],
        "sigma_s": [float(-np.log(0.5)), float(-np.log(0.5)), float(-np.log(0.5))],
        "sigma_t_max_channel": float(-np.log(0.5)),
    })
    slab_record = {"path": slab_path.relative_to(repo).as_posix(), "sha256": _hash(slab_path)}

    scene_records = {}
    for role in ("camera", "terrain_dem", "terrain", "medium", "sun", "atmosphere", "exposure", "tonemap", "crop", "material"):
        path = fixture / f"{role}.json"
        value = {"schema": f"test.{role}/1", "value": role}
        if role == "terrain_dem":
            path = fixture / "terrain-dem.npy"
            _save(path, np.zeros((2, 2), dtype=np.float32))
        if role == "crop":
            value = {"schema": "forge3d.nephele.crop/1", "x": 0, "y": 0, "width": 12, "height": 12, "full_viewport": [12, 12]}
        if role == "medium":
            density_r16 = [32768] * 8
            represented, density_f16_sha256 = represented_density(density_r16, (2, 2, 2))
            runtime_sigma_t = float(np.float32(0.25) + np.float32(0.75))
            value = {
                "schema": "forge3d.nephele.heterogeneous_medium/2",
                "domain": {"bounds_min": [0.0, 0.0, 0.0], "bounds_max": [1.0, 1.0, 1.0], "grid_shape": [2, 2, 2]},
                "density_r16": density_r16,
                "density_transport": {
                    "schema": "forge3d.nephele.density_transport/1",
                    "decode": "unorm16-div-65535-as-f32-then-ieee-f16-rne",
                    "storage": "ieee-f16-bits-little-endian",
                    "sampling": DENSITY_SAMPLING,
                    "f16_sha256": density_f16_sha256,
                },
                "majorant_cells": canonical_majorant_cells(
                    represented, 1.0, runtime_sigma_t
                ),
                "majorant_transport": {
                    "schema": "forge3d.nephele.majorant_transport/1",
                    "grid_shape": [2, 2, 2],
                    "query": MAJORANT_QUERY,
                    "construction": "3x3x3-clamped-neighborhood-trilinear-outward-then-f32-extinction-outward",
                },
                "sigma_a": [0.25, 0.25, 0.25],
                "sigma_s": [0.75, 0.75, 0.75],
                "phase": {"kind": "henyey_greenstein", "g": 0.0},
                "density_scale": 1.0,
                "transport": {
                    "sigma_t_spectrum": [1.0, 1.0, 1.0],
                    "sigma_t_max_channel": 1.0,
                    "extinction_channel": 0,
                    "slab_axis": 2,
                },
            }
        if role == "material":
            value = {
                "schema": "forge3d.nephele.material/1", "albedo": [0.6, 0.6, 0.6],
                "metallic": 0.0, "roughness": 1.0, "triplanar_scale": 1.0,
                "normal_strength": 0.0, "blend_sharpness": 1.0,
                "colormap_strength": 0.0, "albedo_mode": "material",
            }
        if role != "terrain_dem":
            _json(path, value)
        scene_records[role] = {
            "path": path.relative_to(repo).as_posix(),
            "artifact": path.name,
            "sha256": _hash(path),
        }
    files = {}
    names = {
        "reference_rgb": "reference-rgb.npy",
        "reference_transmittance": "reference-transmittance.npy",
        "reference_in_scatter": "reference-in-scatter.npy",
        "reference_cloud_shadow_aov": "reference-cloud-shadow-aov.npy",
        "reference_optical_depth": "reference-optical-depth.npy",
        "sky_cloud_mask": "sky-cloud-mask.npy",
        "godray_roi_mask": "godray-roi-mask.npy",
        "terrain_mask": "terrain-mask.npy",
        "cloud_shadow_mask": "cloud-shadow-mask.npy",
        "shaft_mask": "shaft-mask.npy",
        "reference_terrain_slice": "reference-terrain-slice.npy",
        "reference_terrain_hit": "reference-terrain-hit.npy",
        "reference_media_lighting_visibility": "reference-media-lighting-visibility.npy",
    }
    for role, name in names.items():
        files[role] = {
            "path": (fixture / name).relative_to(repo).as_posix(),
            "artifact": name,
            "sha256": _hash(fixture / name),
        }
    manifest = _fixture_manifest_record(
        converged=True,
        provenance={"source_revision": "c" * 40},
        scene=scene_records,
        files=files,
        mask_generator={
            "path": "scripts/nephele_fixture_masks.py",
            "sha256": _hash(repo / "scripts/nephele_fixture_masks.py"),
        },
        mask_rules={
            "path": rules_path.relative_to(repo).as_posix(),
            "sha256": _hash(rules_path),
        },
        fixture_id="synthetic-verifier-test-only",
        revision=1,
    )
    manifest_path = repo / "tests/nephele/fixture-manifest.json"
    _json(manifest_path, manifest)
    policy = {
        "schema": "forge3d.nephele.gate4_policy/1",
        "status": "APPROVED",
        "policy_id": "synthetic-verifier-maximum",
        "approved_by": "test-only",
        "approved_revision": "e" * 40,
        "aggregation": {"kind": "maximum"},
        "definition": "Synthetic verifier branch coverage only.",
    }
    policy_path = repo / "tests/nephele/gate4-policy.json"
    _json(policy_path, policy)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "NEPHELE Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "tracked verifier fixture")
    source_head = _git(repo, "rev-parse", "HEAD")

    manifest["source_revision"] = source_head
    _json(manifest_path, manifest)
    source_paths = (
        "scripts/generate_media_fixture.py", "python/forge3d/media.py", "src/media_py.rs",
        "src/path_tracing/hybrid_compute/media_reference.rs",
        "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
        "src/path_tracing/hybrid_compute/render_terrain.rs",
        "src/terrain/camera.rs", "src/terrain/mod.rs",
        "src/shaders/hybrid_terrain_traversal.wgsl",
        "src/shaders/nephele_terrain_trace_adapter.wgsl", "src/shader_sources.rs",
    )
    source_inputs = {name: _hash(repo / name) for name in source_paths}
    prior_spp, final_spp = 2, 4
    spatial_tiles = [{"x": 0, "y": y, "width": 12, "height": 1} for y in range(12)]
    camera_contract = {
        "origin": [1.0, 2.0, 3.0], "look_at": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0], "right": [1.0, 0.0, 0.0],
        "forward": [0.0, 0.0, -1.0], "fov_y": 45.0,
    }
    reference_native = b"reference-native:" + source_head.encode("ascii")
    reference_wheel_path = artifact / "forge3d-reference.whl"
    reference_native_member = "forge3d/_forge3d.test.so"
    with zipfile.ZipFile(reference_wheel_path, "w") as wheel:
        wheel.writestr(reference_native_member, reference_native)
    reference_native_runtime = {
        "source_revision": source_head, "wheel_filename": reference_wheel_path.name,
        "wheel_sha256": _hash(reference_wheel_path),
        "wheel_native_member": reference_native_member,
        "native_sha256": hashlib.sha256(reference_native).hexdigest(),
    }
    provenance_base = {
        "schema": "forge3d.nephele.reference_provenance/2",
        "algorithm": "integrated-hybrid-terrain-ratio-delta-tracking-reference",
        "seed": 0x4E455048,
        "source_revision": source_head,
        "source_inputs": source_inputs,
        "scene_inputs": {role: scene_records[role]["sha256"] for role in sorted(scene_records)},
        "assembled_sources": {"terrain_media_reference_module_sha256": _reference_module_sha256(repo, source_head)},
        "spatial_tiles": spatial_tiles,
        "camera_contract": camera_contract,
        "native_runtime": reference_native_runtime,
        "acceptance_eligible": True, "diagnostic_reason": None,
        "full_viewport": [12, 12], "crop": [0, 0, 12, 12],
    }
    diagnostics = lambda spp: {
        "majorant_proof": "TrilinearConvexHull", "majorant_valid": True,
        "sample_count": 12 * 12 * spp, "step_count": 1,
        "temporal_history_decision": "not_applicable",
        "temporal_history_reason": "independent reference samples do not reuse temporal history",
        "host_visible_bytes": 12 * 12 * (5 * 3 * 4 + 4 + 2), "froxel_device_local_bytes": 0,
        "density_device_local_bytes": 0, "majorant_device_local_bytes": 0,
        "staging_readback_bytes": 0, "adapter": "Apple Test GPU", "backend": "Metal",
        "driver": "", "source_revision": source_head, "executed_multi_scatter": True,
        "single_scatter_luminance": None, "multiple_scatter_luminance": None,
        "energy_accounting_residual": None,
    }
    final_provenance = {
        **provenance_base, "samples_per_pixel": final_spp,
        "sample_identity": {"algorithm": "per-pixel-absolute-sample-index-v1", "seed": 0x4E455048, "range": [0, final_spp]},
        "runtime_partition": {"kind": "process_pool_max_workers", "value": 3},
        "diagnostics": diagnostics(final_spp),
    }
    prior_provenance = {
        **provenance_base, "samples_per_pixel": prior_spp,
        "sample_identity": {"algorithm": "per-pixel-absolute-sample-index-v1", "seed": 0x4E455048, "range": [0, prior_spp]},
        "runtime_partition": {"kind": "process_pool_max_workers", "value": 1},
        "diagnostics": diagnostics(prior_spp),
    }
    final_provenance_path = fixture / "reference-provenance.json"
    prior_provenance_path = fixture / f"reference-provenance-spp{prior_spp}.json"
    _json(final_provenance_path, final_provenance)
    _json(prior_provenance_path, prior_provenance)
    previous_dir = repo / "previous-reference"
    previous_dir.mkdir()
    for name in convergence_module.PREFIX_ARTIFACTS:
        shutil.copy2(fixture / name, previous_dir / name)
    shutil.copy2(prior_provenance_path, previous_dir / "reference-provenance.json")
    original_root = convergence_module.ROOT
    original_fixture = convergence_module.FIXTURE
    original_file = convergence_module.__file__
    original_argv = sys.argv
    try:
        convergence_module.ROOT = repo
        convergence_module.FIXTURE = fixture
        convergence_module.__file__ = str(repo / "scripts/record_media_reference_convergence.py")
        sys.argv = [convergence_module.__file__, "--previous-dir", str(previous_dir)]
        assert convergence_module.main() == 0
    finally:
        convergence_module.ROOT = original_root
        convergence_module.FIXTURE = original_fixture
        convergence_module.__file__ = original_file
        sys.argv = original_argv
        shutil.rmtree(previous_dir)
    convergence = _load(fixture / "reference-convergence.json")
    prior_records = convergence["previous"]["artifacts"]
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "bind converged reference evidence")
    head = _git(repo, "rev-parse", "HEAD")

    for record in [*scene_records.values(), *files.values()]:
        shutil.copy2(repo / record["path"], artifact / record["artifact"])
    shutil.copy2(manifest_path, artifact / "fixture-manifest.json")
    shutil.copy2(policy_path, artifact / "gate4-policy.json")
    shutil.copy2(slab_path, artifact / slab_path.name)
    for source in (
        final_provenance_path, prior_provenance_path, fixture / "reference-convergence.json",
        repo / "scripts/record_media_reference_convergence.py",
        *(repo / record["path"] for record in prior_records.values()),
    ):
        shutil.copy2(source, artifact / source.name)
    for name, value in {
        "realtime-rgb.npy": reference,
        "medium-disabled-rgb.npy": dark,
        "terrain-occlusion-disabled-rgb.npy": dark,
        "realtime-termination-slice.npy": depth,
        "froxel-frame-run-a.npy": reference,
        "froxel-frame-run-b.npy": reference,
        "transmittance.npy": scalar,
        "in-scatter.npy": vector,
        "cloud-shadow-aov.npy": scalar,
        "optical-depth.npy": scalar,
    }.items():
        _save(artifact / name, value)

    comparator_raw = artifact / "heterogeneous-comparator.samples.bin"
    comparator_raw.write_bytes(bytes([0x55]) * 12_500_000)
    comparator_seed = _identity("splitmix64", seed_u64=COMPARATOR_SEED)
    comparator_mapping = _identity("indexed-ray-bernoulli-v1", dimensions=["u", "v", "survival"], bit_order="lsb0")
    synthetic_medium = _load(fixture / "medium.json")
    density_transport = synthetic_medium["density_transport"]
    implementation_transport = {
        "density_f16_sha256": density_transport["f16_sha256"],
        "grid_shape": [2, 2, 2],
        "majorant_grid_shape": [2, 2, 2],
        "density_scale": 1.0,
        "density_sampling": DENSITY_SAMPLING,
        "majorant_query": MAJORANT_QUERY,
        "sigma_t_max": 1.0,
    }
    comparator = {
        "schema": "forge3d.nephele.heterogeneous_comparator/4",
        "algorithm": "independent-f16-texel-center-column-bernoulli-v3",
        "transport_representation": {
            "density_f16_sha256": density_transport["f16_sha256"],
            "grid_shape": [2, 2, 2],
            "density_scale": 1.0,
            "sampling": DENSITY_SAMPLING,
            "slab_integration": "exact-piecewise-linear-texel-center-column-mean",
        },
        "extinction_channel": 0,
        "sigma_t": 1.0,
        "seed_identity": comparator_seed,
        "sample_mapping": comparator_mapping,
        "source_revision": head,
        "command": "test-only",
        "generated_at_utc": "2000-01-01T00:00:00Z",
        "producer_tool": {"path": "scripts/nephele_heterogeneous_comparator.py", "sha256": _hash(repo / "scripts/nephele_heterogeneous_comparator.py")},
        "medium": {"path": scene_records["medium"]["path"], "sha256": scene_records["medium"]["sha256"]},
        "cache_key": "",
        "samples": {"count": 100_000_000, "sum": 50_000_000, "sum_squares": 50_000_000},
        "statistics": {"mean": 0.5, "sample_variance": 0.25000000250000004, "standard_error": 5.000000025e-05},
        "raw_output": {"path": comparator_raw.name, "sha256": _hash(comparator_raw), "encoding": "bit-packed-lsb0-bernoulli", "samples": 100_000_000},
        "cache_provenance": {"status": "freshly_executed", "cache_key_scope": "inputs-only"},
    }
    comparator["cache_key"] = _derived_cache_key(comparator)
    _json(artifact / "heterogeneous-comparator.json", comparator)
    domain = {
        "schema": "forge3d.nephele.majorant_domain_coverage/3",
        "medium_sha256": scene_records["medium"]["sha256"],
        "density_f16_sha256": density_transport["f16_sha256"],
        "bounds_min": [0.0, 0.0, 0.0], "bounds_max": [1.0, 1.0, 1.0], "grid_shape": [2, 2, 2],
        "majorant_grid_shape": [2, 2, 2], "exact_domain": True,
        "domain_boundary_points": 8, "domain_face_interiors": 24,
        "texel_center_extrema": 8,
        "majorant_cell_centers": 8, "texel_centers_are_majorant_cell_centers": True,
        "majorant_boundary_sides": 24, "majorant_query": MAJORANT_QUERY,
        "probe_mapping": PROBE_MAPPING, "sha256": "",
    }
    domain["sha256"] = domain_coverage_sha256(domain)
    majorant_raw = artifact / "majorant-domain-probe.pairs.bin"
    represented_bound = float(np.float32(synthetic_medium["majorant_cells"][0]))
    majorant_raw.write_bytes(struct.pack("<ff", 0.5, represented_bound) * 1_000_000)
    represented_bound_sum = 0.0
    for _ in range(1_000_000):
        represented_bound_sum += represented_bound
    majorant_probe = {
        "schema": "forge3d.nephele.majorant_probe/3", "probe_count": 1_000_000,
        "violation_count": 0, "max_represented_extinction_minus_bound": 0.5 - represented_bound,
        "measured_representation_error": 0.0, "source_revision": head,
        "producer_tool": {"path": "scripts/nephele_majorant_domain_probe.py", "sha256": _hash(repo / "scripts/nephele_majorant_domain_probe.py")},
        "medium": {"path": scene_records["medium"]["path"], "sha256": scene_records["medium"]["sha256"]},
        "transport_representation": {
            "density_f16_sha256": density_transport["f16_sha256"],
            "grid_shape": [2, 2, 2], "majorant_grid_shape": [2, 2, 2],
            "density_sampling": DENSITY_SAMPLING, "majorant_query": MAJORANT_QUERY,
            "sigma_t_max": 1.0,
        },
        "sample_mapping": {"algorithm": PROBE_MAPPING, "count": 1_000_000},
        "domain": domain,
        "raw_output": {"path": majorant_raw.name, "sha256": _hash(majorant_raw), "encoding": "little-endian-f32-extinction-bound-pairs", "pairs": 1_000_000},
        "summary": {"extinction": {"minimum": 0.5, "maximum": 0.5, "sum": 500_000.0}, "bound": {"minimum": represented_bound, "maximum": represented_bound, "sum": represented_bound_sum}},
    }
    _json(artifact / "majorant-domain-probe.json", majorant_probe)
    rr_evidence = produce_rr_evidence(np.full(1_000_000, 0.5), np.full(1_000_000, 0.5), head, artifact / "rr-contributions.bin", 0.5)
    _json(artifact / "rr-evidence.json", rr_evidence)
    producer = {
        "implementation": "canonical-ratio-delta-roulette-and-analog-sphere-v1",
        "source_revision": head,
        "native_source": {"path": "src/media_py.rs", "sha256": _hash(repo / "src/media_py.rs")},
        "producer_tool": {"path": "scripts/run_media_physical_capture.py", "sha256": _hash(repo / "scripts/run_media_physical_capture.py")},
    }
    ratio = np.full(1_000_000, 0.5, dtype="<f8")
    for name in ("homogeneous-ratio-samples.bin", "heterogeneous-ratio-samples.bin"):
        (artifact / name).write_bytes(ratio.tobytes())
    ratio_raw = lambda name: {"path": name, "sha256": _hash(artifact / name), "encoding": "little-endian-f64", "samples": 1_000_000}
    _json(artifact / "gate1-statistics.json", {
        "schema": "forge3d.nephele.gate1_raw/1",
        "producer": producer, "homogeneous_slab": slab_record,
        "homogeneous": {"samples": _acc(1_000_000, 0.5), "sigma_t": float(-np.log(0.5)), "density": 1.0, "distance": 1.0, "sample_mapping": HOMOGENEOUS_MAPPING, "raw_output": ratio_raw("homogeneous-ratio-samples.bin")},
        "heterogeneous": {"samples": _acc(1_000_000, 0.5), "sample_mapping": HETEROGENEOUS_MAPPING, "transport_representation": implementation_transport, "raw_output": ratio_raw("heterogeneous-ratio-samples.bin"), "comparator_sha256": _hash(artifact / "heterogeneous-comparator.json")},
        "majorant": {"evidence_sha256": _hash(artifact / "majorant-domain-probe.json")},
        "russian_roulette": {"evidence_sha256": _hash(artifact / "rr-evidence.json")},
    })
    transmitted = np.zeros(1_000_000); transmitted[:250_000] = 1
    scattered = np.zeros(1_000_000); scattered[:500_000] = 1
    absorbed = np.zeros(1_000_000); absorbed[750_000:] = 1
    incident = np.ones(1_000_000)
    closure = transmitted + scattered + absorbed - incident
    energy = np.column_stack((transmitted, scattered, absorbed, incident, closure)).astype("<f8")
    energy_path = artifact / "gate2-energy-samples.bin"; energy_path.write_bytes(energy.tobytes())
    _json(artifact / "gate2-energy.json", {
        "schema": "forge3d.nephele.gate2_raw/2", "producer": producer,
        "homogeneous_slab": slab_record,
        "sample_mapping": _identity("independent-analog-closed-sphere-v1", streams={"transmitted": 0x4E4550482001, "scattered_out": 0x4E4550482002, "absorbed": 0x4E4550482003}, normalization="one incident energy unit per estimator sample"),
        "normalization": "three independent analog-transport ensembles share one incident energy unit per sample", "raw_output": {"path": energy_path.name, "sha256": _hash(energy_path), "encoding": "little-endian-f64", "samples": 1_000_000, "columns": ["transmitted", "scattered_out", "absorbed", "incident", "closure_residual"]},
        "transmitted": {"count": 1_000_000, "sum": float(transmitted.sum()), "sum_squares": float(np.square(transmitted).sum())},
        "scattered_out": {"count": 1_000_000, "sum": float(scattered.sum()), "sum_squares": float(np.square(scattered).sum())},
        "absorbed": {"count": 1_000_000, "sum": float(absorbed.sum()), "sum_squares": float(np.square(absorbed).sum())},
        "incident": _acc(1_000_000, 1.0),
        "closure_residual": {"count": 1_000_000, "sum": float(closure.sum()), "sum_squares": float(np.square(closure).sum())},
    })
    static = analyze_shaders(repo)
    static["tool_sha256"] = _hash(repo / "scripts/nephele_shader_analyzer.py")
    _json(artifact / "gate5-static.json", static)
    _json(artifact / "memory.json", {
        "schema": "forge3d.nephele.memory/2", "output_dimensions": [12, 12],
        "internal_dimensions": [12, 12], "include_no_medium": True,
        "froxel_depth": 64, "termination_integration_steps": 4 * 12 * 12,
        "peak_host_visible_bytes": 100, "froxel_device_local_bytes": 200,
        "density_majorant_device_local_bytes": 300, "staging_bytes": 40,
        "readback_bytes": (40 + 24) * 12 * 12,
    })
    cases = "".join(f'<testcase classname="{classname}" name="{name}"/>' for classname, name in sorted(REQUIRED_JUNIT_CASES))
    (artifact / "junit.xml").write_text(f'<testsuite tests="6" failures="0" errors="0" skipped="0">{cases}</testsuite>', encoding="utf-8")

    native = b"native-extension:" + head.encode("ascii")
    wheel_name = "forge3d-test.whl"
    with zipfile.ZipFile(artifact / wheel_name, "w") as wheel:
        wheel.writestr("forge3d/_forge3d.pyd", native)
    installed = tmp_path / "installed/_forge3d.pyd"
    installed.parent.mkdir()
    installed.write_bytes(native)
    (artifact / "native-extension.bin").write_bytes(native)
    _json(artifact / "installed-wheel-runtime.json", {
        "schema": "forge3d.nephele.installed_runtime/1", "source_revision": head,
        "package_version": "test-only", "wheel_filename": wheel_name, "wheel_sha256": _hash(artifact / wheel_name),
        "wheel_native_member": "forge3d/_forge3d.pyd", "native_sha256": hashlib.sha256(native).hexdigest(),
        "installed_native_path": str(installed),
    })
    probe = {"status": "ok", "name": "NVIDIA Physical Test GPU", "vendor": 0x10DE, "device": 2, "backend": "Vulkan", "device_type": "DiscreteGpu", "driver": "test-driver", "driver_info": "test-driver-info", "software_fallback": False}
    _json(artifact / "adapter-probe.json", {"schema": "forge3d.nephele.adapter_probe/1", "status": "passed", "requested_backend": "vulkan", "probe": probe})
    context = {
        "schema": "forge3d.nephele.run_context/1", "status": "captured", "head_sha": head,
        "checked_out_head": head, "tracked_worktree_clean": True, "required_backend": "vulkan",
        "command": "pytest tests/test_nephele_physical.py", "fixture_manifest_sha256": _hash(manifest_path),
        "gate4_policy_sha256": _hash(policy_path), "fixture_commit": head, "runner_os": "Windows",
        "runner_arch": "X64", "lane": "windows-nvidia-vulkan",
    }
    _json(artifact / "run-context.json", context)
    bundle = {"fixture_manifest_sha256": _hash(manifest_path), "scene_inputs": {role: scene_records[role]["sha256"] for role in sorted(scene_records)}, "files": {role: files[role]["sha256"] for role in sorted(files)}, "homogeneous_slab_sha256": slab_record["sha256"]}
    bundle_hash = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    adapter_hash = hashlib.sha256(json.dumps(probe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    candidate_diagnostics = {
        "majorant_proof": "TrilinearConvexHull", "majorant_valid": True,
        "sample_count": 144, "step_count": 12345,
        "temporal_history_decision": "reset",
        "temporal_history_reason": "new acceptance renderer has no prior temporal history",
        "host_visible_bytes": 100, "froxel_device_local_bytes": 200,
        "density_device_local_bytes": 64, "majorant_device_local_bytes": 64,
        "staging_readback_bytes": 40, "adapter": probe["name"], "backend": "Vulkan",
        "driver": f'{probe["driver"]} {probe["driver_info"]}', "source_revision": head,
        "executed_multi_scatter": True, "single_scatter_dispatches": 1,
        "multiple_scatter_dispatches": 1, "terrain_trace_queries": 3,
        "single_scatter_luminance": 1.0, "multiple_scatter_luminance": 0.5,
        "energy_accounting_residual": 0.0,
        "sun_transmittance_method": "bounded_nested_midpoint",
        "sun_transmittance_bias": "fine_midpoint_with_coarse_fine_abs_rgb_error",
        "sun_transmittance_max_segment_length": 60.0,
        "sun_transmittance_executed_steps": 8192,
        "sun_transmittance_max_abs_error": 0.0005,
    }
    _validate_candidate_diagnostics(candidate_diagnostics, head, probe)
    _validate_realtime_diagnostics(
        candidate_diagnostics, "synthetic capture diagnostics", head, probe
    )
    for label, pid, nonce in (("a", 101, "1" * 64), ("b", 202, "2" * 64)):
        frame = f"froxel-frame-run-{label}.npy"
        _json(artifact / f"run-{label}.json", {
            "schema": "forge3d.nephele.clean_run/1", "process_id": pid, "run_nonce": nonce,
            "head_sha": head, "tracked_worktree_clean": True,
            "fixture_manifest_sha256": _hash(manifest_path), "input_bundle_sha256": bundle_hash,
            "adapter_identity_sha256": adapter_hash, "backend": "vulkan",
            "frame_file": frame, "frame_sha256": _hash(artifact / frame),
            "camera_contract": camera_contract,
            "diagnostics": candidate_diagnostics,
            "presentation_identity": {
                "capture_api": "TerrainRenderer._capture_nephele_acceptance",
                "color_pipeline": ["white_balance", "exposure", "aces", "lut", "iec_61966_2_1_srgb8"],
                "no_medium_same_call": label == "a",
            },
            "medium_disabled_rgb_sha256": _hash(artifact / "medium-disabled-rgb.npy") if label == "a" else None,
        })
    return repo, artifact, head


def _report(repo: Path, artifact: Path, head: str, observed_arch: str = "X64") -> dict:
    runtime = _load(artifact / "installed-wheel-runtime.json")
    return build_report(
        artifact,
        head_sha=head,
        repo_root=repo,
        imported_native_path=Path(runtime["installed_native_path"]),
        observed_host=("Windows", observed_arch, "windows-nvidia-vulkan"),
    )


def _rebind_native_head(artifact: Path, head: str) -> None:
    runtime = _load(artifact / "installed-wheel-runtime.json")
    native = b"native-extension:" + head.encode("ascii")
    with zipfile.ZipFile(artifact / runtime["wheel_filename"], "w") as wheel:
        wheel.writestr(runtime["wheel_native_member"], native)
    (artifact / "native-extension.bin").write_bytes(native)
    Path(runtime["installed_native_path"]).write_bytes(native)
    runtime["source_revision"] = head
    runtime["wheel_sha256"] = _hash(artifact / runtime["wheel_filename"])
    runtime["native_sha256"] = hashlib.sha256(native).hexdigest()
    _json(artifact / "installed-wheel-runtime.json", runtime)


def test_report_recomputes_all_six_gates_from_bound_evidence(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    report = _report(repo, artifact, head)
    assert report["status"] == "PASS"
    assert set(report["gates"]) == {f"gate{i}" for i in range(1, 7)}
    assert report["identity"]["junit"]["tests"] == 6
    assert report["gates"]["gate6"]["frame_hashes"][0] == report["gates"]["gate6"]["frame_hashes"][1]
    assert report["gates"]["gate6"]["sun_transmittance"] == {
        "method": "bounded_nested_midpoint",
        "bias": "fine_midpoint_with_coarse_fine_abs_rgb_error",
        "max_segment_length": 60.0,
        "executed_steps": 8192,
        "max_abs_error": 0.0005,
    }


def test_capture_shaped_sun_diagnostics_interoperate_and_fail_closed(
    tmp_path: Path,
) -> None:
    repo, artifact, head = _fixture(tmp_path)
    base = _load(artifact / "run-a.json")["diagnostics"]
    probe = _load(artifact / "adapter-probe.json")["probe"]
    assert DIAGNOSTIC_KEYS == REALTIME_DIAGNOSTIC_KEYS == set(base)
    _validate_candidate_diagnostics(base, head, probe)
    assert _validate_realtime_diagnostics(
        base, "synthetic capture diagnostics", head, probe
    ) == {
        "method": "bounded_nested_midpoint",
        "bias": "fine_midpoint_with_coarse_fine_abs_rgb_error",
        "max_segment_length": 60.0,
        "executed_steps": 8192,
        "max_abs_error": 0.0005,
    }
    mutations = (
        ("missing", "sun_transmittance_method", None),
        ("extra", "untracked_sun_claim", 1),
        ("type", "sun_transmittance_method", 1),
        ("method", "sun_transmittance_method", "not_executed"),
        ("bias", "sun_transmittance_bias", "none_exact"),
        ("segment-type", "sun_transmittance_max_segment_length", 60),
        ("segment-invalid", "sun_transmittance_max_segment_length", 0.0),
        ("steps-type", "sun_transmittance_executed_steps", True),
        ("steps-invalid", "sun_transmittance_executed_steps", 0),
        ("error-type", "sun_transmittance_max_abs_error", 0),
        ("error-nonfinite", "sun_transmittance_max_abs_error", float("nan")),
        ("error-invalid", "sun_transmittance_max_abs_error", 1.1),
    )
    for mutation, field, value in mutations:
        diagnostics = dict(base)
        if mutation == "missing":
            diagnostics.pop(field)
        else:
            diagnostics[field] = value
        with pytest.raises(ValueError, match="diagnostic|schema|sun-transmittance"):
            _validate_candidate_diagnostics(diagnostics, head, probe)
        with pytest.raises(EvidenceError, match="diagnostic|keys|sun-transmittance"):
            _validate_realtime_diagnostics(
                diagnostics, "synthetic capture diagnostics", head, probe
            )


def test_verifier_accepts_actual_generator_manifest_and_recorder_prefix_schema(
    tmp_path: Path,
) -> None:
    repo, artifact, head = _fixture(tmp_path)
    manifest = _load(artifact / "fixture-manifest.json")
    assert manifest["color_pipeline"] == _color_pipeline(manifest["scene_inputs"])
    _verify_fixture_manifest(artifact, manifest, repo, head)

    convergence = _load(artifact / "reference-convergence.json")
    assert set(convergence["previous"]["artifacts"]) == {
        "reference-rgb.npy",
        "reference-transmittance.npy",
        "reference-in-scatter.npy",
        "reference-cloud-shadow-aov.npy",
        "reference-optical-depth.npy",
        "reference-terrain-hit.npy",
        "reference-media-lighting-visibility.npy",
        "reference-terrain-slice.npy",
    }
    _verify_reference_provenance(artifact, manifest, repo, head)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", "color pipeline"),
        ("extra", "color pipeline"),
        ("literal", "color pipeline"),
        ("provenance", "not bound"),
    ],
)
def test_generator_color_pipeline_schema_fails_closed(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    repo, artifact, head = _fixture(tmp_path)
    manifest = _load(artifact / "fixture-manifest.json")
    if mutation == "missing":
        manifest["color_pipeline"].pop("linear_input")
    elif mutation == "extra":
        manifest["color_pipeline"]["encoding"] = "sRGB8"
    elif mutation == "literal":
        manifest["color_pipeline"]["tonemap"] = "legacy"
    else:
        manifest["color_pipeline"]["tonemap_contract"] = "0" * 64
    with pytest.raises(EvidenceError, match=expected):
        _verify_fixture_manifest(artifact, manifest, repo, head)


def test_windows_runner_arch_x64_matches_unmasked_amd64_platform(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    assert _report(repo, artifact, head, observed_arch="AMD64")["status"] == "PASS"


def test_reference_prefix_identity_is_independent_of_worker_count(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    convergence = _load(artifact / "reference-convergence.json")
    prior = _load(artifact / Path(convergence["previous"]["provenance_path"]).name)
    final = _load(artifact / Path(convergence["final"]["provenance_path"]).name)
    assert prior["runtime_partition"]["value"] != final["runtime_partition"]["value"]
    assert prior["spatial_tiles"] == final["spatial_tiles"]
    assert prior["sample_identity"]["range"] == [0, prior["samples_per_pixel"]]
    assert final["sample_identity"]["range"] == [0, final["samples_per_pixel"]]
    assert _report(repo, artifact, head)["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["camera", "presentation", "no-medium-hash"])
def test_gate6_rejects_camera_or_shared_presentation_tampering(tmp_path: Path, mutation: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    run = _load(artifact / "run-a.json")
    if mutation == "camera":
        run["camera_contract"]["origin"][0] += 1.0
    elif mutation == "presentation":
        run["presentation_identity"]["color_pipeline"] = ["legacy-gamma"]
    else:
        run["medium_disabled_rgb_sha256"] = "0" * 64
    _json(artifact / "run-a.json", run)
    with pytest.raises(EvidenceError, match="camera|presentation"):
        _report(repo, artifact, head)


@pytest.mark.parametrize(
    "mutation",
    [
        "step-mismatch", "fractional-step", "readback-bytes", "conditional-readback",
        "integration-steps", "fractional-termination", "energy-missing",
        "energy-nonfinite", "energy-negative", "energy-string", "energy-bool",
        "single-string", "multiple-bool",
    ],
)
def test_gate6_rejects_step_and_readback_accounting_tampering(
    tmp_path: Path, mutation: str
) -> None:
    repo, artifact, head = _fixture(tmp_path)
    if mutation in {
        "step-mismatch", "fractional-step", "energy-missing", "energy-nonfinite",
        "energy-negative", "energy-string", "energy-bool", "single-string",
        "multiple-bool",
    }:
        run = _load(artifact / "run-b.json")
        if mutation == "step-mismatch":
            run["diagnostics"]["step_count"] += 1
        elif mutation == "fractional-step":
            run["diagnostics"]["step_count"] = 1.5
        elif mutation == "energy-missing":
            run["diagnostics"].pop("energy_accounting_residual")
        elif mutation == "energy-nonfinite":
            run["diagnostics"]["energy_accounting_residual"] = float("nan")
        elif mutation == "energy-negative":
            run["diagnostics"]["energy_accounting_residual"] = -1.0
        elif mutation == "energy-string":
            run["diagnostics"]["energy_accounting_residual"] = "0.0"
        elif mutation == "energy-bool":
            run["diagnostics"]["energy_accounting_residual"] = False
        elif mutation == "single-string":
            run["diagnostics"]["single_scatter_luminance"] = "1.0"
        else:
            run["diagnostics"]["multiple_scatter_luminance"] = True
        _json(artifact / "run-b.json", run)
        if mutation == "energy-string":
            run_a = _load(artifact / "run-a.json")
            run_a["diagnostics"]["energy_accounting_residual"] = "0.0"
            _json(artifact / "run-a.json", run_a)
    elif mutation in {"readback-bytes", "conditional-readback", "integration-steps"}:
        memory = _load(artifact / "memory.json")
        if mutation == "conditional-readback":
            memory["include_no_medium"] = False
        else:
            key = "readback_bytes" if mutation == "readback-bytes" else "termination_integration_steps"
            memory[key] += 1
        _json(artifact / "memory.json", memory)
    else:
        path = artifact / "realtime-termination-slice.npy"
        termination = np.load(path, allow_pickle=False)
        termination[0, 0] = 3.5
        _save(path, termination)
    with pytest.raises(EvidenceError, match="step_count|readback|integration|termination|integer|no-medium|diagnostic|finite|negative"):
        _report(repo, artifact, head)


def test_capture_uses_same_acceptance_call_for_medium_disabled_presentation() -> None:
    source = (ROOT / "scripts/run_media_physical_capture.py").read_text(encoding="utf-8")
    render_source = source[source.index("def render("):source.index("\ndef main(")]
    assert "include_no_medium=capture_primary" in render_source
    assert 'capture["no_medium_beauty"]' in render_source
    assert "render_terrain_pbr_pom" not in render_source
    assert "media=None" not in render_source
    assert 'grid_shape = medium_data["domain"].get("grid_shape")' in render_source
    assert 'density_scale = float(medium_data["density_scale"])' in render_source
    assert "density_scale=density_scale" in render_source
    assert "reshape(3, 3, 3)" not in render_source


def test_reference_binding_exposes_exact_terrain_classification_arrays() -> None:
    source = (ROOT / "src/media_py.rs").read_text(encoding="utf-8")
    assert '"terrain_hit"' in source
    assert 'output.terrain_hit' in source
    assert '"media_lighting_visibility"' in source
    assert 'output.media_lighting_visibility' in source


@pytest.mark.parametrize("mutation", [
    "assembled", "metric", "prefix", "sample-identity", "spatial-tiles", "runtime-partition",
    "classification", "scene", "camera", "native-runtime", "eligibility", "diagnostics", "unresolved", "arbitrary",
    "diagnostics-host", "diagnostics-temporal", "diagnostics-device", "diagnostics-adapter",
    "diagnostics-driver", "diagnostics-proof", "diagnostics-step", "diagnostics-multi",
    "diagnostics-luminance", "diagnostics-energy",
])
def test_reference_provenance_and_convergence_tampering_fails_closed(tmp_path: Path, mutation: str) -> None:
    repo, artifact, _ = _fixture(tmp_path)
    convergence_path = repo / "tests/nephele/fixture/reference-convergence.json"
    convergence = _load(convergence_path)
    if mutation == "assembled":
        provenance_path = repo / convergence["final"]["provenance_path"]
        provenance = _load(provenance_path)
        provenance["assembled_sources"]["terrain_media_reference_module_sha256"] = "0" * 64
        _json(provenance_path, provenance)
        shutil.copy2(provenance_path, artifact / provenance_path.name)
        convergence["final"]["provenance_sha256"] = _hash(provenance_path)
    elif mutation == "metric":
        convergence["metrics"]["gate3_godray_roi_ssim"] = 0.999
    elif mutation == "prefix":
        convergence["spatial_tiles_identity"] = False
    elif mutation in {"sample-identity", "spatial-tiles", "runtime-partition", "camera", "native-runtime", "eligibility"}:
        provenance_path = repo / convergence["final"]["provenance_path"]
        provenance = _load(provenance_path)
        if mutation == "sample-identity":
            provenance["sample_identity"]["range"] = [1, provenance["samples_per_pixel"] + 1]
        elif mutation == "spatial-tiles":
            provenance["spatial_tiles"][0]["width"] -= 1
        elif mutation == "runtime-partition":
            provenance["runtime_partition"]["value"] = len(provenance["spatial_tiles"]) + 1
        elif mutation == "camera":
            provenance["camera_contract"]["origin"][0] += 1.0
        elif mutation == "eligibility":
            provenance["acceptance_eligible"] = False
            provenance["diagnostic_reason"] = "dirty reference runtime"
        else:
            provenance["native_runtime"]["native_sha256"] = "0" * 64
        _json(provenance_path, provenance)
        shutil.copy2(provenance_path, artifact / provenance_path.name)
        convergence["final"]["provenance_sha256"] = _hash(provenance_path)
    elif mutation == "classification":
        convergence["classification_identity"]["reference-terrain-hit.npy"] = False
    elif mutation == "scene":
        provenance_path = repo / convergence["final"]["provenance_path"]
        provenance = _load(provenance_path)
        provenance["scene_inputs"]["material"] = "0" * 64
        _json(provenance_path, provenance)
        shutil.copy2(provenance_path, artifact / provenance_path.name)
        convergence["final"]["provenance_sha256"] = _hash(provenance_path)
    elif mutation.startswith("diagnostics"):
        provenance_path = repo / convergence["final"]["provenance_path"]
        provenance = _load(provenance_path)
        if mutation == "diagnostics":
            provenance["diagnostics"]["sample_count"] += 1
        elif mutation == "diagnostics-host":
            provenance["diagnostics"]["host_visible_bytes"] = 0
        elif mutation == "diagnostics-temporal":
            provenance["diagnostics"]["temporal_history_reason"] = "arbitrary"
        elif mutation == "diagnostics-device":
            provenance["diagnostics"]["froxel_device_local_bytes"] = 1
        elif mutation == "diagnostics-adapter":
            provenance["diagnostics"]["adapter"] = "Different Physical GPU"
        elif mutation == "diagnostics-driver":
            provenance["diagnostics"]["driver"] = "different-driver"
        elif mutation == "diagnostics-proof":
            provenance["diagnostics"]["majorant_proof"] = "self-attested"
        elif mutation == "diagnostics-step":
            provenance["diagnostics"]["step_count"] = 0
        elif mutation == "diagnostics-luminance":
            provenance["diagnostics"]["single_scatter_luminance"] = 0.0
        elif mutation == "diagnostics-energy":
            provenance["diagnostics"]["energy_accounting_residual"] = 0.0
        else:
            provenance["diagnostics"]["executed_multi_scatter"] = False
        _json(provenance_path, provenance)
        shutil.copy2(provenance_path, artifact / provenance_path.name)
        convergence["final"]["provenance_sha256"] = _hash(provenance_path)
    elif mutation == "unresolved":
        convergence["status"] = "UNRESOLVED"
        convergence["next_samples_per_pixel_if_unresolved"] = convergence["final"]["samples_per_pixel"] * 2
    else:
        convergence["criterion"] = "looks stable"
    _json(convergence_path, convergence)
    shutil.copy2(convergence_path, artifact / convergence_path.name)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", f"malicious reference {mutation}")
    fixture_commit = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(EvidenceError, match="assembled|metrics|prefix|sample|spatial|runtime|classification|scene|camera|native|eligibility|diagnostic|unresolved|arbitrary"):
        _verify_reference_provenance(
            artifact, _load(repo / "tests/nephele/fixture-manifest.json"), repo, fixture_commit
        )


def test_reference_module_hash_binds_query_shader_bytes(tmp_path: Path) -> None:
    repo, _, _ = _fixture(tmp_path)
    source_revision = _load(repo / "tests/nephele/fixture-manifest.json")["source_revision"]
    before = _reference_module_sha256(repo, source_revision)
    query = repo / "src/shaders/nephele_terrain_trace_adapter.wgsl"
    query.write_text(query.read_text(encoding="utf-8") + "// tampered query\n", encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "tamper reference query")
    after = _reference_module_sha256(repo, _git(repo, "rev-parse", "HEAD"))
    assert after != before


def test_gate2_rejects_tautological_per_sample_closure(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    count = 1_000_000
    transmitted = np.zeros(count); transmitted[:250_000] = 1.0
    scattered = np.zeros(count); scattered[250_000:750_000] = 1.0
    absorbed = np.zeros(count); absorbed[750_000:] = 1.0
    incident = np.ones(count)
    closure = transmitted + scattered + absorbed - incident
    raw = np.column_stack((transmitted, scattered, absorbed, incident, closure)).astype("<f8")
    path = artifact / "gate2-energy-samples.bin"
    path.write_bytes(raw.tobytes())
    record = _load(artifact / "gate2-energy.json")
    record["raw_output"]["sha256"] = _hash(path)
    for index, name in enumerate(("transmitted", "scattered_out", "absorbed", "incident", "closure_residual")):
        record[name] = _accumulator_from_array(raw[:, index])
    _json(artifact / "gate2-energy.json", record)
    with pytest.raises(EvidenceError, match="tautological"):
        _gate2(artifact, head, repo)


def test_gate2_rejects_fractional_non_analog_outcomes(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    count = 1_000_000
    transmitted = np.full(count, 0.25)
    scattered = np.full(count, 0.5)
    absorbed = np.full(count, 0.25)
    incident = np.ones(count)
    closure = np.zeros(count)
    raw = np.column_stack((transmitted, scattered, absorbed, incident, closure)).astype("<f8")
    path = artifact / "gate2-energy-samples.bin"
    path.write_bytes(raw.tobytes())
    record = _load(artifact / "gate2-energy.json")
    record["raw_output"]["sha256"] = _hash(path)
    for index, name in enumerate(("transmitted", "scattered_out", "absorbed", "incident", "closure_residual")):
        record[name] = _accumulator_from_array(raw[:, index])
    _json(artifact / "gate2-energy.json", record)
    with pytest.raises(EvidenceError, match="invalid support"):
        _gate2(artifact, head, repo)


def test_gate2_rejects_unbound_homogeneous_medium_identity(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    record = _load(artifact / "gate2-energy.json")
    record["homogeneous_slab"]["sha256"] = "0" * 64
    _json(artifact / "gate2-energy.json", record)
    with pytest.raises(EvidenceError, match="homogeneous slab"):
        _gate2(artifact, head, repo)


@pytest.mark.parametrize("mutation", ["head", "dirty", "os", "lane"])
def test_rejects_self_attested_repository_and_lane_identity(tmp_path: Path, mutation: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    context = _load(artifact / "run-context.json")
    if mutation == "head": context["head_sha"] = "f" * 40; context["checked_out_head"] = "f" * 40
    elif mutation == "dirty": (repo / "untracked.txt").write_text("dirty", encoding="utf-8")
    elif mutation == "os": context["runner_os"] = "Linux"
    else: context["lane"] = "hosted-software"
    _json(artifact / "run-context.json", context)
    with pytest.raises(EvidenceError, match="identity|clean|lane|source|HEAD"):
        _report(repo, artifact, head if mutation != "head" else "f" * 40)


@pytest.mark.parametrize("mutation", ["fake-wheel", "wrong-member", "wrong-installed"])
def test_rejects_fake_wheel_or_unrelated_native_bytes(tmp_path: Path, mutation: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    runtime = _load(artifact / "installed-wheel-runtime.json")
    if mutation == "fake-wheel":
        (artifact / runtime["wheel_filename"]).write_bytes(b"not-a-wheel")
        runtime["wheel_sha256"] = _hash(artifact / runtime["wheel_filename"])
    elif mutation == "wrong-member": runtime["wheel_native_member"] = "forge3d/unrelated.pyd"
    else: Path(runtime["installed_native_path"]).write_bytes(b"unrelated")
    _json(artifact / "installed-wheel-runtime.json", runtime)
    with pytest.raises(EvidenceError, match="wheel|native"):
        _report(repo, artifact, head)


@pytest.mark.parametrize("mutation", ["wheel", "member", "native"])
def test_reference_runtime_requires_retained_wheel_and_native_bytes(tmp_path: Path, mutation: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    manifest = _load(repo / "tests/nephele/fixture-manifest.json")
    convergence_path = repo / "tests/nephele/fixture/reference-convergence.json"
    convergence = _load(convergence_path)
    final_path = repo / convergence["final"]["provenance_path"]
    prior_path = repo / convergence["previous"]["provenance_path"]
    provenance = _load(final_path)
    runtime = provenance["native_runtime"]
    wheel_path = artifact / runtime["wheel_filename"]
    if mutation == "wheel":
        wheel_path.write_bytes(b"invented-wheel")
    elif mutation == "member":
        with zipfile.ZipFile(wheel_path, "w") as wheel:
            wheel.writestr("forge3d/unrelated.so", b"unrelated:" + head.encode("ascii"))
        runtime["wheel_sha256"] = _hash(wheel_path)
    else:
        native = b"invented-native:" + head.encode("ascii")
        with zipfile.ZipFile(wheel_path, "w") as wheel:
            wheel.writestr(runtime["wheel_native_member"], native)
    fixture_commit = head
    if mutation != "wheel":
        for path, endpoint in ((final_path, "final"), (prior_path, "previous")):
            value = _load(path)
            value["native_runtime"]["wheel_sha256"] = _hash(wheel_path)
            _json(path, value)
            shutil.copy2(path, artifact / path.name)
            convergence[endpoint]["provenance_sha256"] = _hash(path)
        _json(convergence_path, convergence)
        shutil.copy2(convergence_path, artifact / convergence_path.name)
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", f"tamper retained reference {mutation}")
        fixture_commit = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(EvidenceError, match="reference wheel|reference native|native member"):
        _verify_reference_provenance(artifact, manifest, repo, fixture_commit)


def test_capture_reference_wheel_seam_recomputes_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifact = tmp_path / "artifact"
    source.mkdir(); artifact.mkdir()
    native = b"reference:" + ("a" * 40).encode("ascii")
    wheel_path = source / "forge3d-reference.whl"
    member = "forge3d/_forge3d.abi3.so"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(member, native)
    provenance = {
        "acceptance_eligible": True,
        "native_runtime": {
            "source_revision": "a" * 40, "wheel_filename": wheel_path.name,
            "wheel_sha256": _hash(wheel_path), "wheel_native_member": member,
            "native_sha256": hashlib.sha256(native).hexdigest(),
        },
    }
    _retain_reference_wheel(artifact, provenance, source)
    assert (artifact / wheel_path.name).read_bytes() == wheel_path.read_bytes()
    provenance["native_runtime"]["native_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="native bytes"):
        _retain_reference_wheel(artifact, provenance, source)


@pytest.mark.parametrize("field,value", [("vendor", 1), ("device_type", "VirtualGpu"), ("driver", ""), ("name", "NVIDIA Paravirtual GPU")])
def test_rejects_unmeaningful_or_nonphysical_adapter(tmp_path: Path, field: str, value: object) -> None:
    repo, artifact, head = _fixture(tmp_path)
    adapter = _load(artifact / "adapter-probe.json")
    adapter["probe"][field] = value
    _json(artifact / "adapter-probe.json", adapter)
    with pytest.raises(EvidenceError) as caught:
        _report(repo, artifact, head)
    assert caught.value.code == "adapter_error"


@pytest.mark.parametrize("surface", ["context", "requested", "probe", "run-a", "run-b"])
def test_windows_nvidia_vulkan_lane_rejects_dx12_everywhere(tmp_path: Path, surface: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    if surface == "context":
        path = artifact / "run-context.json"; value = _load(path); value["required_backend"] = "DX12"
    elif surface in {"requested", "probe"}:
        path = artifact / "adapter-probe.json"; value = _load(path)
        if surface == "requested": value["requested_backend"] = "DX12"
        else: value["probe"]["backend"] = "DX12"
    else:
        path = artifact / f"{surface}.json"; value = _load(path); value["backend"] = "DX12"
    _json(path, value)
    with pytest.raises(EvidenceError, match="Vulkan|backend|render"):
        _report(repo, artifact, head)


@pytest.mark.parametrize("role", sorted(EXPECTED_RULES))
def test_mask_generator_rejects_nonreference_role_sources(tmp_path: Path, role: str) -> None:
    repo, _, _ = _fixture(tmp_path)
    fixture = repo / "tests/nephele/fixture"
    rules_path = fixture / "mask-rules.json"
    rules = _load(rules_path)
    rules["rules"][role] = "candidate-derived classification"
    _json(rules_path, rules)
    with pytest.raises(ValueError, match="allowed reference|semantics|reviewed"):
        generate_masks(fixture, rules_path, tmp_path / "masks")


def test_rejects_untracked_fixture_blob_and_regenerated_mask_mismatch(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    _save(artifact / "sky-cloud-mask.npy", np.eye(12, dtype=np.bool_))
    manifest = _load(repo / "tests/nephele/fixture-manifest.json")
    manifest["files"]["sky_cloud_mask"]["sha256"] = _hash(artifact / "sky-cloud-mask.npy")
    _json(artifact / "fixture-manifest.json", manifest)
    context = _load(artifact / "run-context.json")
    context["fixture_manifest_sha256"] = _hash(artifact / "fixture-manifest.json")
    _json(artifact / "run-context.json", context)
    with pytest.raises(EvidenceError, match="tracked|regeneration|differ"):
        _report(repo, artifact, head)


def test_rejects_nonexistent_fixture_commit_even_when_json_claims_it(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    context = _load(artifact / "run-context.json")
    context["fixture_commit"] = "f" * 40
    _json(artifact / "run-context.json", context)
    with pytest.raises(EvidenceError, match="cat-file|tracked blob|commit"):
        _report(repo, artifact, head)


def test_rejects_self_selected_installed_path_even_with_matching_bytes(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    runtime = _load(artifact / "installed-wheel-runtime.json")
    actual_imported = Path(runtime["installed_native_path"])
    decoy = tmp_path / "decoy/_forge3d.pyd"
    decoy.parent.mkdir()
    shutil.copy2(actual_imported, decoy)
    runtime["installed_native_path"] = str(decoy)
    _json(artifact / "installed-wheel-runtime.json", runtime)
    with pytest.raises(EvidenceError, match="imported native extension"):
        build_report(
            artifact,
            head_sha=head,
            repo_root=repo,
            imported_native_path=actual_imported,
            observed_host=("Windows", "X64", "windows-nvidia-vulkan"),
        )


@pytest.mark.parametrize("mutation", ["support", "seed", "cache"])
def test_rejects_impossible_accumulator_seed_or_cached_comparator(tmp_path: Path, mutation: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    if mutation == "cache":
        path = artifact / "heterogeneous-comparator.json"; data = _load(path); data["cache_key"] = "0" * 64; _json(path, data)
        stats = _load(artifact / "gate1-statistics.json"); stats["heterogeneous"]["comparator_sha256"] = _hash(path); _json(artifact / "gate1-statistics.json", stats)
    else:
        path = artifact / "gate1-statistics.json"; data = _load(path)
        if mutation == "support": data["homogeneous"]["samples"] = {"count": 1_000_000, "sum": 500_000.0, "sum_squares": 900_000.0}
        else: data["homogeneous"]["sample_mapping"] = {"algorithm": "fake", "parameters": {}, "sha256": "0" * 64}
        _json(path, data)
    with pytest.raises(EvidenceError, match="support|accumulator|SHA-256|provenance"):
        _report(repo, artifact, head)


@pytest.mark.parametrize("mutation", ["comparator-tool", "medium", "comparator-representation", "domain", "boundary-count", "domain-tool"])
def test_rejects_unbound_comparator_or_incomplete_majorant_domain(tmp_path: Path, mutation: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    if mutation in {"comparator-tool", "medium", "comparator-representation"}:
        path = artifact / "heterogeneous-comparator.json"; value = _load(path)
        if mutation == "comparator-tool": value["producer_tool"]["sha256"] = "0" * 64
        elif mutation == "medium": value["medium"]["sha256"] = "0" * 64
        else: value["transport_representation"]["sampling"] = "endpoint-linear"
        value["cache_key"] = _derived_cache_key(value)
        _json(path, value)
        gate1 = _load(artifact / "gate1-statistics.json")
        gate1["heterogeneous"]["comparator_sha256"] = _hash(path)
        _json(artifact / "gate1-statistics.json", gate1)
    else:
        path = artifact / "majorant-domain-probe.json"; value = _load(path)
        if mutation == "domain": value["domain"]["exact_domain"] = False
        elif mutation == "boundary-count": value["domain"]["majorant_boundary_sides"] -= 1
        else: value["producer_tool"]["sha256"] = "0" * 64
        if mutation != "domain-tool":
            value["domain"]["sha256"] = domain_coverage_sha256(value["domain"])
        _json(path, value)
        gate1 = _load(artifact / "gate1-statistics.json")
        gate1["majorant"]["evidence_sha256"] = _hash(path)
        _json(artifact / "gate1-statistics.json", gate1)
    with pytest.raises(EvidenceError, match="producer|medium|domain|coverage|boundary|provenance|transported"):
        _report(repo, artifact, head)


@pytest.mark.parametrize("mutation", ["f16-hash", "endpoint-sampling", "n-minus-one-cells", "sigma-metadata"])
def test_transported_medium_tampering_fails_closed(mutation: str) -> None:
    medium = _load(ROOT / "tests/nephele/fixture/medium.json")
    if mutation == "f16-hash":
        medium["density_transport"]["f16_sha256"] = "0" * 64
    elif mutation == "endpoint-sampling":
        medium["density_transport"]["sampling"] = "normalized-endpoint-linear-u-times-N-minus-one"
    elif mutation == "n-minus-one-cells":
        medium["majorant_cells"] = medium["majorant_cells"][:8]
    else:
        medium["transport"]["sigma_t_max_channel"] = 0.08
    with pytest.raises(EvidenceError, match=r"transported representation|canonical N\^3"):
        _verify_transported_medium(medium)


def test_rr_allows_contributions_above_one_but_rejects_negative_samples(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    path = artifact / "rr-evidence.json"
    contributions = np.zeros(1_000_000)
    contributions[:333_333] = 1.5
    value = produce_rr_evidence(contributions, contributions, head, artifact / "rr-contributions.bin", 0.5)
    _json(path, value)
    gate1 = _load(artifact / "gate1-statistics.json"); gate1["russian_roulette"]["evidence_sha256"] = _hash(path); _json(artifact / "gate1-statistics.json", gate1)
    assert _report(repo, artifact, head)["gates"]["gate1"]["russian_roulette"]["on"]["mean"] == pytest.approx(0.4999995)
    raw_path = artifact / "rr-contributions.bin"
    payload = bytearray(raw_path.read_bytes())
    struct.pack_into("<d", payload, 0, -1.0)
    raw_path.write_bytes(payload)
    value["raw_output"]["sha256"] = _hash(raw_path)
    _json(path, value); gate1["russian_roulette"]["evidence_sha256"] = _hash(path); _json(artifact / "gate1-statistics.json", gate1)
    with pytest.raises(EvidenceError, match="nonnegative"):
        _report(repo, artifact, head)


def test_rejects_lying_shader_summary_and_compute_reachable_compare(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    shader = repo / "src/shaders/test.wgsl"
    shader.write_text("fn bad(t: texture_depth_2d, s: sampler_comparison) { let x = textureSampleCompare(t, s, vec2f(0.0), 0.0); }\n@compute @workgroup_size(1) fn main() { bad(); }\n", encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "malicious shader")
    new_head = _git(repo, "rev-parse", "HEAD")
    context = _load(artifact / "run-context.json"); context["head_sha"] = new_head; context["checked_out_head"] = new_head; _json(artifact / "run-context.json", context)
    _rebind_native_head(artifact, new_head)
    comparator = _load(artifact / "heterogeneous-comparator.json"); comparator["source_revision"] = new_head; comparator["cache_key"] = _derived_cache_key(comparator); _json(artifact / "heterogeneous-comparator.json", comparator)
    majorant = _load(artifact / "majorant-domain-probe.json"); majorant["source_revision"] = new_head; _json(artifact / "majorant-domain-probe.json", majorant)
    rr = _load(artifact / "rr-evidence.json"); rr["source_revision"] = new_head; _json(artifact / "rr-evidence.json", rr)
    gate1 = _load(artifact / "gate1-statistics.json"); gate1["producer"]["source_revision"] = new_head; gate1["heterogeneous"]["comparator_sha256"] = _hash(artifact / "heterogeneous-comparator.json"); gate1["majorant"]["evidence_sha256"] = _hash(artifact / "majorant-domain-probe.json"); gate1["russian_roulette"]["evidence_sha256"] = _hash(artifact / "rr-evidence.json"); _json(artifact / "gate1-statistics.json", gate1)
    gate2 = _load(artifact / "gate2-energy.json"); gate2["producer"]["source_revision"] = new_head; _json(artifact / "gate2-energy.json", gate2)
    for label in ("a", "b"):
        run = _load(artifact / f"run-{label}.json"); run["head_sha"] = new_head; _json(artifact / f"run-{label}.json", run)
    with pytest.raises(EvidenceError, match="shader analysis"):
        _report(repo, artifact, new_head)


@pytest.mark.parametrize("kind", ["outside-shader-directory", "embedded-rust"])
def test_shader_analyzer_covers_all_tracked_wgsl_and_embedded_rust(tmp_path: Path, kind: str) -> None:
    repo, _, _ = _fixture(tmp_path)
    source = "@compute @workgroup_size(1) fn exploit(t: texture_depth_2d, s: sampler_comparison) { let x = textureSampleCompare(t, s, vec2f(0.0), 0.0); }\n"
    if kind == "outside-shader-directory":
        path = repo / "src/viewer/exploit.wgsl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    else:
        path = repo / "src/embedded.rs"
        path.write_text(f'const EXPLOIT: &str = r#"{source}"#;\nfn load() {{ let _ = wgpu::ShaderSource::Wgsl(EXPLOIT.into()); }}\n', encoding="utf-8")
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "tracked shader exploit")
    result = analyze_shaders(repo)
    assert result["compute_entry_texture_sample_compare_calls"] == 1
    if kind == "outside-shader-directory": assert path.relative_to(repo).as_posix() in result["tracked_wgsl_files"]
    else: assert result["embedded_rust_wgsl_sources"]


def test_shader_analyzer_rejects_unsafe_source_routed_through_wrapper(tmp_path: Path) -> None:
    repo, _, _ = _fixture(tmp_path)
    path = repo / "src/wrapper_bypass.rs"
    path.write_text(
        'const UNSAFE: &str = "@compute @workgroup_size(1) fn exploit(t: texture_depth_2d, s: sampler_comparison) { let x = textureSampleCompare(t, s, vec2f(0.0), 0.0); }\\n";\n'
        'fn wrapper(device: &wgpu::Device, source: &str) { crate::core::shader_registry::create_labeled_shader_module(device, "wrapped", source); }\n'
        'fn live(device: &wgpu::Device) { wrapper(device, UNSAFE); }\n',
        encoding="utf-8",
    )
    _git(repo, "add", "."); _git(repo, "commit", "-qm", "unsafe wrapper bypass")
    result = analyze_shaders(repo)
    assert result["rust_shader_construction_wrappers"]
    assert result["rust_shader_wrapper_invocations"]
    assert result["compute_entry_texture_sample_compare_calls"] == 1


@pytest.mark.slow
def test_production_shader_constructions_resolve_and_naga_validate() -> None:
    result = analyze_shaders(ROOT)
    assert result["unresolved_source_expressions"] == []
    assert result["rust_self_naga_validated_constructions"] == []
    assert result["naga_validated_assemblies"] == len(result["resolved_source_sha256"])
    assert result["compute_entry_texture_sample_compare_calls"] == 0


@pytest.mark.parametrize("mutation", ["same-process", "wrong-input", "wrong-adapter", "wrong-frame"])
def test_rejects_copyable_or_unbound_clean_runs(tmp_path: Path, mutation: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    run_a, run_b = _load(artifact / "run-a.json"), _load(artifact / "run-b.json")
    if mutation == "same-process": run_b["process_id"] = run_a["process_id"]
    elif mutation == "wrong-input": run_b["input_bundle_sha256"] = "0" * 64
    elif mutation == "wrong-adapter": run_b["adapter_identity_sha256"] = "0" * 64
    else: run_b["frame_file"] = run_a["frame_file"]
    _json(artifact / "run-b.json", run_b)
    with pytest.raises(EvidenceError, match="run|process"):
        _report(repo, artifact, head)


@pytest.mark.parametrize("kind", ["duplicate", "extra"])
def test_junit_is_exact_six_case_multiset(tmp_path: Path, kind: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    root = (artifact / "junit.xml").read_text(encoding="utf-8")
    extra = '<testcase classname="tests.test_nephele_physical" name="gate1_estimator_majorant_rr"/>' if kind == "duplicate" else '<testcase classname="other" name="extra"/>'
    (artifact / "junit.xml").write_text(root.replace("</testsuite>", extra + "</testsuite>"), encoding="utf-8")
    with pytest.raises(EvidenceError) as caught:
        _report(repo, artifact, head)
    assert caught.value.code == "junit_error"


def test_junit_accepts_namespaces_and_rejects_false_suite_aggregates(tmp_path: Path) -> None:
    repo, artifact, head = _fixture(tmp_path)
    cases = "".join(
        f'<testcase classname="{classname}" name="{name}"/>'
        for classname, name in sorted(REQUIRED_JUNIT_CASES)
    )
    path = artifact / "junit.xml"
    path.write_text(
        f'<testsuites xmlns="urn:junit"><testsuite tests="6" failures="0" errors="0" skipped="0">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    assert _report(repo, artifact, head)["identity"]["junit"]["skipped"] == 0
    path.write_text(
        f'<testsuites xmlns="urn:junit"><testsuite tests="6" failures="0" errors="0" skipped="6">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(EvidenceError, match="aggregate"):
        _report(repo, artifact, head)


@pytest.mark.parametrize("xml", [
    '<root><testcase classname="tests.test_nephele_physical" name="gate1_estimator_majorant_rr"/></root>',
    '<testsuites><testcase classname="tests.test_nephele_physical" name="gate1_estimator_majorant_rr"/></testsuites>',
])
def test_junit_rejects_cases_outside_validated_suites(tmp_path: Path, xml: str) -> None:
    repo, artifact, head = _fixture(tmp_path)
    (artifact / "junit.xml").write_text(xml, encoding="utf-8")
    with pytest.raises(EvidenceError) as caught:
        _report(repo, artifact, head)
    assert caught.value.code == "junit_error"


def test_producers_execute_samples_and_bind_raw_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    medium = tmp_path / "medium.json"
    raw = [16384, 32768, 32768, 49152, 16384, 32768, 32768, 49152]
    represented, f16_sha256 = represented_density(raw, (2, 2, 2))
    majorants = canonical_majorant_cells(represented, 1.0, 1.0)
    _json(medium, {
        "schema": "forge3d.nephele.heterogeneous_medium/2",
        "domain": {"bounds_min": [0.0, 0.0, 0.0], "bounds_max": [1.0, 1.0, 1.0], "grid_shape": [2, 2, 2]},
        "density_r16": raw,
        "density_transport": {
            "schema": "forge3d.nephele.density_transport/1",
            "decode": "unorm16-div-65535-as-f32-then-ieee-f16-rne",
            "storage": "ieee-f16-bits-little-endian",
            "sampling": DENSITY_SAMPLING,
            "f16_sha256": f16_sha256,
        },
        "majorant_cells": majorants,
        "majorant_transport": {
            "schema": "forge3d.nephele.majorant_transport/1", "grid_shape": [2, 2, 2],
            "query": MAJORANT_QUERY,
            "construction": "3x3x3-clamped-neighborhood-trilinear-outward-then-f32-extinction-outward",
        },
        "sigma_a": [0.25, 0.25, 0.25],
        "sigma_s": [0.75, 0.75, 0.75],
        "phase": {"kind": "henyey_greenstein", "g": 0.0},
        "density_scale": 1.0,
        "transport": {"sigma_t_spectrum": [1.0, 1.0, 1.0], "sigma_t_max_channel": 1.0, "extinction_channel": 0, "slab_axis": 2},
    })
    comparator = produce_comparator(medium, "a" * 40, tmp_path / "comparator.bin", samples=200)
    assert comparator["samples"]["count"] == 200
    assert sum(byte.bit_count() for byte in (tmp_path / "comparator.bin").read_bytes()) == comparator["samples"]["sum"]
    assert comparator["statistics"]["standard_error"] >= 0.0
    monkeypatch.setattr(majorant_probe_module, "PROBES", 1_000)
    probe = produce_majorant_probe(medium, "a" * 40, tmp_path / "majorant.bin", probes=1_000)
    assert probe["probe_count"] == 1_000
    assert probe["violation_count"] == 0
    assert probe["raw_output"]["pairs"] == 1_000


def test_synthetic_unresolved_fixture_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError) as caught:
        _verify_fixture_manifest(tmp_path, {"status": "UNRESOLVED"}, tmp_path, "a" * 40)
    assert caught.value.code == "fixture_unresolved"


def test_synthetic_unresolved_gate4_policy_fails_closed() -> None:
    with pytest.raises(EvidenceError) as caught:
        _policy_evaluator({"status": "UNRESOLVED"})
    assert caught.value.code == "policy_unresolved"


@pytest.mark.parametrize("aggregation", [
    {"kind": "percentile", "percentile": 99.0},
    {"kind": "per_pixel_pass_fraction", "minimum_fraction": 1.0},
])
def test_gate4_rejects_nonmaximum_aggregation(aggregation: dict[str, object]) -> None:
    policy = {
        "schema": "forge3d.nephele.gate4_policy/1", "status": "APPROVED",
        "policy_id": "test", "approved_by": "test", "approved_revision": "a" * 40,
        "aggregation": aggregation, "definition": "test",
    }
    with pytest.raises(EvidenceError) as caught:
        _policy_evaluator(policy)
    assert caught.value.code == "policy_unresolved"


def test_cli_has_no_arbitrary_fixture_override_and_writes_structured_error(tmp_path: Path) -> None:
    artifact = tmp_path / "missing"
    with pytest.raises(SystemExit):
        main([str(artifact), "--head-sha", "a" * 40, "--fixture-manifest", "/tmp/fake"])
    assert main([str(artifact), "--head-sha", "a" * 40]) == 2
    error = _load(artifact / "verification-error.json")
    assert error["schema"] == "forge3d.nephele.verification_error/1"
    assert error["status"] == "FAIL"


def test_tracked_unresolved_fixture_fails_closed_and_inventory_is_exact() -> None:
    inventory = _load(ROOT / "tests/nephele/evidence-inventory.json")
    expected = RAW_FILES | {
        "run-context.json", "adapter-probe.json", "installed-wheel-runtime.json",
        "native-extension.bin", "fixture-manifest.json", "gate4-policy.json", "junit.xml",
            "camera.json", "terrain.json", "medium.json", "sun.json", "atmosphere.json",
            "exposure.json", "tonemap.json", "crop.json", "material.json", "terrain-dem.npy",
    }
    actual = set(inventory["required_raw_artifacts"])
    assert actual == expected
    force_tracked = {
        "scripts/generate_media_blue_noise.py",
        "scripts/generate_media_fixture.py",
        "scripts/nephele_evidence_report.py",
        "scripts/nephele_fixture_masks.py",
        "scripts/nephele_heterogeneous_comparator.py",
        "scripts/nephele_majorant_domain_probe.py",
        "scripts/nephele_shader_analyzer.py",
        "scripts/record_media_reference_convergence.py",
        "scripts/run_media_physical_capture.py",
    }
    assert set(inventory["force_tracked_sources"]) == force_tracked
    assert all((ROOT / relative).is_file() for relative in force_tracked)
    assert inventory["transport_representation"] == {
        "comparator_algorithm": "independent-f16-texel-center-column-bernoulli-v3",
        "comparator_schema": "forge3d.nephele.heterogeneous_comparator/4",
        "density_sampling": DENSITY_SAMPLING,
        "density_storage": "ieee-f16-bits-little-endian",
        "majorant_cells_topology": "N^3",
        "majorant_probe_mapping": PROBE_MAPPING,
        "majorant_probe_schema": "forge3d.nephele.majorant_probe/3",
        "majorant_query": MAJORANT_QUERY,
    }
    fixture = _load(ROOT / "tests/nephele/fixture-manifest.json")
    policy = _load(ROOT / "tests/nephele/gate4-policy.json")
    assert fixture["status"] == "UNRESOLVED"
    with pytest.raises(EvidenceError) as unresolved:
        _verify_fixture_manifest(ROOT, fixture, ROOT, _git(ROOT, "rev-parse", "HEAD"))
    assert unresolved.value.code == "fixture_unresolved"
    assert policy["status"] == "APPROVED"
    assert policy["aggregation"] == {"kind": "maximum"}
    assert inventory["gate4_aggregation"] == "maximum"
