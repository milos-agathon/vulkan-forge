#!/usr/bin/env python3
"""Produce exact-head raw inputs for the NEPHELE physical evidence verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.nephele_heterogeneous_comparator import produce as produce_comparator
from scripts.nephele_heterogeneous_comparator import produce_rr_evidence
from scripts.nephele_majorant_domain_probe import (
    DENSITY_SAMPLING,
    MAJORANT_CONSTRUCTION,
    MAJORANT_QUERY,
    canonical_majorant_cells,
    produce as produce_majorant,
    represented_density,
)
from scripts.nephele_shader_analyzer import analyze_shaders


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/nephele/fixture"
GATE_SAMPLE_COUNT = 1_000_000
HOMOGENEOUS_SLAB = ROOT / "tests/nephele/homogeneous-slab.json"
ADAPTER_KEYS = {
    "status", "name", "vendor", "device", "backend", "device_type",
    "driver", "driver_info", "software_fallback",
}
DIAGNOSTIC_KEYS = {
    "majorant_proof", "majorant_valid", "sample_count", "step_count",
    "temporal_history_decision", "temporal_history_reason", "host_visible_bytes",
    "froxel_device_local_bytes", "density_device_local_bytes",
    "majorant_device_local_bytes", "staging_readback_bytes", "adapter", "backend",
    "driver", "source_revision", "executed_multi_scatter",
    "single_scatter_dispatches", "multiple_scatter_dispatches", "terrain_trace_queries",
    "single_scatter_luminance", "multiple_scatter_luminance",
    "energy_accounting_residual",
    "sun_transmittance_method", "sun_transmittance_bias",
    "sun_transmittance_max_segment_length", "sun_transmittance_executed_steps",
    "sun_transmittance_max_abs_error",
}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ from the tracked NEPHELE schema")


def _validate_candidate_diagnostics(
    diagnostics: dict[str, Any], head: str, probe: dict[str, Any]
) -> None:
    _keys(diagnostics, DIAGNOSTIC_KEYS, "candidate diagnostics")
    expected_driver = f'{probe["driver"]} {probe["driver_info"]}'
    segment_length = diagnostics["sun_transmittance_max_segment_length"]
    max_abs_error = diagnostics["sun_transmittance_max_abs_error"]
    if (
        diagnostics.get("source_revision") != head
        or str(diagnostics.get("backend", "")).lower() != "vulkan"
        or diagnostics.get("adapter") != probe["name"]
        or diagnostics.get("driver") != expected_driver
        or type(diagnostics.get("step_count")) is not int
        or diagnostics["step_count"] <= 0
        or isinstance(diagnostics["energy_accounting_residual"], bool)
        or not isinstance(diagnostics["energy_accounting_residual"], (int, float))
        or not math.isfinite(diagnostics["energy_accounting_residual"])
        or diagnostics["energy_accounting_residual"] < 0
        or diagnostics["sun_transmittance_method"] != "bounded_nested_midpoint"
        or diagnostics["sun_transmittance_bias"]
        != "fine_midpoint_with_coarse_fine_abs_rgb_error"
        or type(segment_length) is not float
        or not math.isfinite(segment_length)
        or segment_length <= 0.0
        or type(diagnostics["sun_transmittance_executed_steps"]) is not int
        or diagnostics["sun_transmittance_executed_steps"] <= 0
        or type(max_abs_error) is not float
        or not math.isfinite(max_abs_error)
        or not 0.0 <= max_abs_error <= 1.0
    ):
        raise ValueError(
            "live media diagnostics differ from exact source, Vulkan adapter, "
            "driver identity, or executed sun-transmittance measurement"
        )


def _scene_input_path(manifest: dict[str, Any], role: str) -> Path:
    record = manifest.get("scene_inputs", {}).get(role)
    if not isinstance(record, dict):
        raise ValueError(f"fixture manifest has no tracked {role} scene input")
    _keys(record, {"path", "artifact", "sha256"}, f"manifest scene input {role}")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"manifest scene input {role} path is invalid")
    path = ROOT / relative
    if path.name != record["artifact"] or _sha256(path) != record["sha256"]:
        raise ValueError(f"tracked {role} scene input differs from its manifest hash")
    return path


def _scene_input(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    return _object(_scene_input_path(manifest, role))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(algorithm: str, **parameters: Any) -> dict[str, Any]:
    record = {"algorithm": algorithm, "parameters": parameters}
    return {**record, "sha256": hashlib.sha256(_canonical(record)).hexdigest()}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_head(head: str) -> None:
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ValueError("--head-sha must be one full lowercase Git SHA")
    if _git("rev-parse", "HEAD") != head or _git("status", "--porcelain"):
        raise ValueError("NEPHELE capture requires the exact clean candidate HEAD")


def _fixture_bundle(manifest: dict[str, Any]) -> str:
    value = {
        "fixture_manifest_sha256": _sha256(ROOT / "tests/nephele/fixture-manifest.json"),
        "scene_inputs": {role: manifest["scene_inputs"][role]["sha256"] for role in sorted(manifest["scene_inputs"])},
        "files": {role: manifest["files"][role]["sha256"] for role in sorted(manifest["files"])},
        "homogeneous_slab_sha256": _sha256(HOMOGENEOUS_SLAB),
    }
    return hashlib.sha256(_canonical(value)).hexdigest()


def _retain_reference_wheel(
    artifact: Path,
    provenance: dict[str, Any],
    reference_wheel_dir: Path,
) -> None:
    runtime = provenance.get("native_runtime")
    if provenance.get("acceptance_eligible") is not True or not isinstance(runtime, dict):
        raise ValueError("reference provenance is not eligible or has no native runtime")
    _keys(
        runtime,
        {"source_revision", "wheel_filename", "wheel_sha256", "wheel_native_member", "native_sha256"},
        "reference native runtime",
    )
    wheel_name = runtime["wheel_filename"]
    if not isinstance(wheel_name, str) or Path(wheel_name).name != wheel_name:
        raise ValueError("reference wheel filename is invalid")
    source = reference_wheel_dir / wheel_name
    if not source.is_file() or _sha256(source) != runtime["wheel_sha256"]:
        raise ValueError("supplied reference wheel bytes differ from tracked provenance")
    try:
        with zipfile.ZipFile(source) as wheel:
            members = [
                name for name in wheel.namelist()
                if name.startswith("forge3d/_forge3d")
                and Path(name).suffix.lower() in {".so", ".dylib", ".pyd"}
            ]
            if members != [runtime["wheel_native_member"]]:
                raise ValueError("supplied reference wheel native member is missing or ambiguous")
            native_bytes = wheel.read(members[0])
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"supplied reference wheel is invalid: {exc}") from exc
    if (
        hashlib.sha256(native_bytes).hexdigest() != runtime["native_sha256"]
        or str(runtime["source_revision"]).encode("ascii") not in native_bytes
    ):
        raise ValueError("supplied reference native bytes differ from tracked provenance or source")
    destination = artifact / wheel_name
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError("reference and candidate wheels collide by filename with different bytes")
    shutil.copy2(source, destination)


def prepare(
    artifact: Path,
    head: str,
    wheel_dir: Path,
    reference_wheel_dir: Path,
) -> None:
    _require_head(head)
    manifest_path = ROOT / "tests/nephele/fixture-manifest.json"
    policy_path = ROOT / "tests/nephele/gate4-policy.json"
    manifest = _object(manifest_path)
    if manifest.get("status") != "APPROVED":
        raise ValueError("tracked NEPHELE fixture is not approved")
    artifact.mkdir(parents=True, exist_ok=True)
    reference_provenance = _object(FIXTURE / "reference-provenance.json")
    _retain_reference_wheel(artifact, reference_provenance, reference_wheel_dir)
    for record in [*manifest["scene_inputs"].values(), *manifest["files"].values()]:
        source = ROOT / record["path"]
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"tracked fixture hash mismatch: {source}")
        shutil.copy2(source, artifact / record["artifact"])
    convergence_path = FIXTURE / "reference-convergence.json"
    convergence = _object(convergence_path)
    prefix_sources = []
    def prefix_source(record: dict[str, Any], label: str) -> Path:
        relative = Path(str(record.get("path", "")))
        digest = record.get("sha256")
        if relative.is_absolute() or ".." in relative.parts or len(str(digest)) != 64:
            raise ValueError(f"{label} has an invalid repository path or hash")
        source = ROOT / relative
        if _sha256(source) != digest:
            raise ValueError(f"{label} differs from its convergence hash")
        return source
    previous = convergence.get("previous")
    if isinstance(previous, dict):
        if "provenance_path" in previous:
            prefix_sources.append(prefix_source(
                {"path": previous.get("provenance_path"), "sha256": previous.get("provenance_sha256")},
                "prior reference provenance",
            ))
        artifacts = previous.get("artifacts")
        if isinstance(artifacts, dict):
            for name, record in artifacts.items():
                if not isinstance(record, dict):
                    raise ValueError(f"prior reference {name} has no provenance record")
                prefix_sources.append(prefix_source(record, f"prior reference {name}"))
    for source in (
        manifest_path,
        policy_path,
        FIXTURE / "terrain-dem.npy",
        HOMOGENEOUS_SLAB,
        FIXTURE / "reference-provenance.json",
        convergence_path,
        ROOT / "scripts/record_media_reference_convergence.py",
        ROOT / "assets/media/nephele_blue_noise_8x8.txt",
        ROOT / "assets/media/nephele_blue_noise_provenance.json",
        *prefix_sources,
    ):
        shutil.copy2(source, artifact / source.name)

    raw_probe = _object(artifact / "adapter-probe-raw.json")
    probe = raw_probe.get("probe")
    if raw_probe.get("status") != "passed" or not isinstance(probe, dict):
        raise ValueError("physical adapter probe did not pass")
    missing = ADAPTER_KEYS - set(probe)
    if missing:
        raise ValueError(f"physical adapter probe is missing fields: {sorted(missing)}")
    probe = {key: probe[key] for key in ADAPTER_KEYS}
    _json(artifact / "adapter-probe.json", {
        "schema": "forge3d.nephele.adapter_probe/1", "status": "passed",
        "requested_backend": "vulkan", "probe": probe,
    })

    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one downloaded Windows wheel, found {len(wheels)}")
    wheel = wheels[0]
    retained_wheel = artifact / wheel.name
    if retained_wheel.exists() and retained_wheel.read_bytes() != wheel.read_bytes():
        raise ValueError("candidate and reference wheels collide by filename with different bytes")
    shutil.copy2(wheel, retained_wheel)
    with zipfile.ZipFile(retained_wheel) as archive:
        members = [name for name in archive.namelist() if name.startswith("forge3d/_forge3d") and name.endswith(".pyd")]
        if len(members) != 1:
            raise ValueError("wheel does not contain exactly one Forge3D native extension")
        native_bytes = archive.read(members[0])
    if head.encode("ascii") not in native_bytes:
        raise ValueError("wheel native bytes do not contain the exact source revision")
    import forge3d
    import forge3d._forge3d as native

    installed = Path(native.__file__).resolve()
    if installed.read_bytes() != native_bytes:
        raise ValueError("imported native extension differs from the downloaded wheel")
    (artifact / "native-extension.bin").write_bytes(native_bytes)
    _json(artifact / "installed-wheel-runtime.json", {
        "schema": "forge3d.nephele.installed_runtime/1", "source_revision": head,
        "package_version": forge3d.__version__, "wheel_filename": wheel.name,
        "wheel_sha256": _sha256(retained_wheel), "wheel_native_member": members[0],
        "native_sha256": hashlib.sha256(native_bytes).hexdigest(), "installed_native_path": str(installed),
    })
    _json(artifact / "run-context.json", {
        "schema": "forge3d.nephele.run_context/1", "status": "captured", "head_sha": head,
        "checked_out_head": head, "tracked_worktree_clean": True, "required_backend": "vulkan",
        "command": "python -m pytest tests/test_nephele_physical.py -o python_functions=gate*",
        "fixture_manifest_sha256": _sha256(manifest_path), "gate4_policy_sha256": _sha256(policy_path),
        "fixture_commit": head, "runner_os": "Windows", "runner_arch": os.environ.get("RUNNER_ARCH", ""),
        "lane": "windows-nvidia-vulkan",
    })


def _accumulator(values: np.ndarray) -> dict[str, float | int]:
    values = values.astype(np.float64, copy=False)
    return {"count": int(values.size), "sum": float(values.sum(dtype=np.float64)), "sum_squares": float(np.square(values).sum(dtype=np.float64))}


def _raw_f64(path: Path, values: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(values, dtype="<f8")
    path.write_bytes(values.tobytes())
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "encoding": "little-endian-f64",
        "samples": int(values.size),
    }


def _producer_identity(head: str, implementation: str) -> dict[str, Any]:
    native_path = ROOT / "src/media_py.rs"
    tool_path = ROOT / "scripts/run_media_physical_capture.py"
    return {
        "implementation": implementation,
        "source_revision": head,
        "native_source": {"path": "src/media_py.rs", "sha256": _sha256(native_path)},
        "producer_tool": {"path": "scripts/run_media_physical_capture.py", "sha256": _sha256(tool_path)},
    }


def _transport_identity(medium: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema", "domain", "density_r16", "density_transport", "majorant_cells",
        "majorant_transport", "transport", "sigma_a", "sigma_s", "phase",
        "density_scale",
    }
    _keys(medium, expected_keys, "heterogeneous medium")
    shape = tuple(medium["domain"]["grid_shape"])
    represented, f16_sha256 = represented_density(medium["density_r16"], shape)
    density_transport = {
        "schema": "forge3d.nephele.density_transport/1",
        "decode": "unorm16-div-65535-as-f32-then-ieee-f16-rne",
        "storage": "ieee-f16-bits-little-endian",
        "sampling": DENSITY_SAMPLING,
        "f16_sha256": f16_sha256,
    }
    majorant_transport = {
        "schema": "forge3d.nephele.majorant_transport/1",
        "grid_shape": list(shape),
        "query": MAJORANT_QUERY,
        "construction": MAJORANT_CONSTRUCTION,
    }
    sigma_t = [float(a) + float(s) for a, s in zip(medium["sigma_a"], medium["sigma_s"])]
    channel = max(range(3), key=sigma_t.__getitem__)
    transport = {
        "sigma_t_spectrum": sigma_t,
        "sigma_t_max_channel": sigma_t[channel],
        "extinction_channel": channel,
        "slab_axis": 2,
    }
    runtime_sigma_t = max(
        float(np.float32(a) + np.float32(s))
        for a, s in zip(medium["sigma_a"], medium["sigma_s"])
    )
    majorants = canonical_majorant_cells(
        represented, float(medium["density_scale"]), runtime_sigma_t
    )
    if (
        medium["density_transport"] != density_transport
        or medium["majorant_transport"] != majorant_transport
        or medium["transport"] != transport
        or medium["majorant_cells"] != majorants
    ):
        raise ValueError("heterogeneous medium transported representation differs")
    return {
        "density_f16_sha256": f16_sha256,
        "grid_shape": list(shape),
        "majorant_grid_shape": list(shape),
        "density_scale": float(medium["density_scale"]),
        "density_sampling": DENSITY_SAMPLING,
        "majorant_query": MAJORANT_QUERY,
        "sigma_t_max": sigma_t[channel],
    }


def _implementation_samples(medium_data: dict[str, Any], slab: dict[str, Any]) -> dict[str, Any]:
    from forge3d.media import Medium
    import forge3d._forge3d as native

    shape = medium_data["domain"]["grid_shape"]
    density = np.asarray(medium_data["density_r16"], dtype=np.float32).reshape(
        shape[2], shape[1], shape[0]
    ) / 65535.0
    homogeneous = Medium.homogeneous(
        slab["sigma_a"], slab["sigma_s"], slab["density"], version=1
    )
    heterogeneous = Medium.grid3d(
        medium_data["sigma_a"],
        medium_data["sigma_s"],
        density,
        (medium_data["domain"]["bounds_min"], medium_data["domain"]["bounds_max"]),
        phase="henyey_greenstein",
        g=medium_data["phase"]["g"],
        density_scale=medium_data["density_scale"],
        version=1,
    )
    return dict(
        native._nephele_physical_samples(
            homogeneous._native,
            heterogeneous._native,
            float(slab["distance"]),
            GATE_SAMPLE_COUNT,
        )
    )


def produce(artifact: Path, head: str) -> None:
    _require_head(head)
    manifest = _object(ROOT / "tests/nephele/fixture-manifest.json")
    medium_path = ROOT / manifest["scene_inputs"]["medium"]["path"]
    medium = _object(medium_path)
    transport_identity = _transport_identity(medium)
    comparator_path = artifact / "heterogeneous-comparator.json"
    comparator = produce_comparator(medium_path.relative_to(ROOT), head, artifact / "heterogeneous-comparator.samples.bin")
    _json(comparator_path, comparator)
    majorant_path = artifact / "majorant-domain-probe.json"
    _json(majorant_path, produce_majorant(medium_path.relative_to(ROOT), head, artifact / "majorant-domain-probe.pairs.bin"))

    homogeneous_fixture = _object(HOMOGENEOUS_SLAB)
    homogeneous_record = {
        "path": HOMOGENEOUS_SLAB.relative_to(ROOT).as_posix(),
        "sha256": _sha256(HOMOGENEOUS_SLAB),
    }
    samples = _implementation_samples(medium, homogeneous_fixture)
    if samples.get("source_revision") != head:
        raise ValueError("native physical producer does not match the exact source revision")
    implementation = samples.get("implementation")
    if implementation != "canonical-ratio-delta-roulette-and-analog-sphere-v1":
        raise ValueError("native physical producer identity is unknown")
    homogeneous_samples = np.asarray(samples["homogeneous_ratio"], dtype=np.float64)
    heterogeneous_samples = np.asarray(samples["heterogeneous_ratio"], dtype=np.float64)
    rr_on = np.asarray(samples["rr_on"], dtype=np.float64)
    rr_off = np.asarray(samples["rr_off"], dtype=np.float64)
    for name, values in (
        ("homogeneous", homogeneous_samples), ("heterogeneous", heterogeneous_samples),
        ("rr_on", rr_on), ("rr_off", rr_off),
    ):
        if values.shape != (GATE_SAMPLE_COUNT,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"native physical producer returned invalid {name} samples")
    producer = _producer_identity(head, implementation)
    rr_path = artifact / "rr-evidence.json"
    tau = homogeneous_fixture["sigma_t_max_channel"] * homogeneous_fixture["density"] * homogeneous_fixture["distance"]
    albedo = homogeneous_fixture["sigma_s"][0] / homogeneous_fixture["sigma_t_max_channel"]
    rr_oracle = (1.0 - math.exp(-tau)) * albedo
    _json(rr_path, produce_rr_evidence(rr_on, rr_off, head, artifact / "rr-contributions.bin", rr_oracle))
    homogeneous_raw = _raw_f64(artifact / "homogeneous-ratio-samples.bin", homogeneous_samples)
    heterogeneous_raw = _raw_f64(artifact / "heterogeneous-ratio-samples.bin", heterogeneous_samples)
    homogeneous_mapping = _identity("canonical-sample-identity-v1", frame_u64=0x4E4550481001, fields=["frame", "pixel", "sample", "bounce", "dimension"])
    heterogeneous_mapping = _identity("canonical-grid3d-z-ray-v1", coordinate_frame_u64=0x4E4550481101, tracking_frame_u64=0x4E4550481102, fields=["sample", "dimension"])
    _json(artifact / "gate1-statistics.json", {
        "schema": "forge3d.nephele.gate1_raw/1",
        "producer": producer, "homogeneous_slab": homogeneous_record,
        "homogeneous": {"samples": _accumulator(homogeneous_samples), "sigma_t": homogeneous_fixture["sigma_t_max_channel"], "density": homogeneous_fixture["density"], "distance": homogeneous_fixture["distance"], "sample_mapping": homogeneous_mapping, "raw_output": homogeneous_raw},
        "heterogeneous": {"samples": _accumulator(heterogeneous_samples), "sample_mapping": heterogeneous_mapping, "transport_representation": transport_identity, "raw_output": heterogeneous_raw, "comparator_sha256": _sha256(comparator_path)},
        "majorant": {"evidence_sha256": _sha256(majorant_path)},
        "russian_roulette": {"evidence_sha256": _sha256(rr_path)},
    })

    transmitted = np.asarray(samples["transmitted"], dtype=np.float64)
    scattered = np.asarray(samples["scattered_out"], dtype=np.float64)
    absorbed = np.asarray(samples["absorbed"], dtype=np.float64)
    if any(values.shape != (GATE_SAMPLE_COUNT,) or not np.isin(values, (0.0, 1.0)).all() for values in (transmitted, scattered, absorbed)):
        raise ValueError("native closed-sphere producer returned invalid independent outcomes")
    incident = np.ones(GATE_SAMPLE_COUNT, dtype=np.float64)
    closure = transmitted + scattered + absorbed - incident
    energy_matrix = np.column_stack((transmitted, scattered, absorbed, incident, closure))
    energy_raw = _raw_f64(artifact / "gate2-energy-samples.bin", energy_matrix)
    energy_raw["samples"] = GATE_SAMPLE_COUNT
    energy_raw["columns"] = ["transmitted", "scattered_out", "absorbed", "incident", "closure_residual"]
    _json(artifact / "gate2-energy.json", {
        "schema": "forge3d.nephele.gate2_raw/2", "producer": producer,
        "homogeneous_slab": homogeneous_record,
        "sample_mapping": _identity("independent-analog-closed-sphere-v1", streams={"transmitted": 0x4E4550482001, "scattered_out": 0x4E4550482002, "absorbed": 0x4E4550482003}, normalization="one incident energy unit per estimator sample"),
        "normalization": "three independent analog-transport ensembles share one incident energy unit per sample", "raw_output": energy_raw, "transmitted": _accumulator(transmitted),
        "scattered_out": _accumulator(scattered), "absorbed": _accumulator(absorbed),
        "incident": _accumulator(incident), "closure_residual": _accumulator(closure),
    })
    static = analyze_shaders(ROOT)
    static["tool_sha256"] = _sha256(ROOT / "scripts/nephele_shader_analyzer.py")
    _json(artifact / "gate5-static.json", static)


def _write_hdr(path: Path) -> None:
    # RGBE (128,128,128,129) decodes to unit linear radiance; IBL intensity
    # therefore maps the tracked 0.25 atmosphere input without another scale.
    path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n" + bytes([128, 128, 128, 129]))


def _crop(value: Any, crop: dict[str, Any]) -> np.ndarray:
    array = np.asarray(value)
    y, x = int(crop["y"]), int(crop["x"])
    height, width = int(crop["height"]), int(crop["width"])
    full_width, full_height = map(int, crop["full_viewport"])
    if array.shape[:2] != (full_height, full_width) or x + width > full_width or y + height > full_height:
        raise ValueError("live capture shape or tracked crop is inconsistent with the full viewport")
    return array[y : y + height, x : x + width]


def _camera_contract(value: Any) -> dict[str, Any]:
    vector_keys = {"origin", "look_at", "up", "right", "forward"}
    if not isinstance(value, dict) or set(value) != vector_keys | {"fov_y"}:
        raise ValueError("NEPHELE camera contract is incomplete")
    result: dict[str, Any] = {}
    for key in vector_keys:
        raw = value[key]
        vector = np.asarray(raw, dtype=np.float32)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"NEPHELE camera contract {key} is invalid")
        result[key] = vector.tolist()
    fov_y = np.float32(value["fov_y"])
    if not np.isfinite(fov_y):
        raise ValueError("NEPHELE camera contract fov_y is invalid")
    result["fov_y"] = float(fov_y)
    return result


def render(artifact: Path, head: str, label: str, capture_primary: bool) -> None:
    _require_head(head)
    import forge3d as f3d
    from forge3d.media import Medium
    from forge3d.terrain_params import AovSettings, TonemapSettings, make_terrain_params_config

    manifest = _object(ROOT / "tests/nephele/fixture-manifest.json")
    medium_data = _scene_input(manifest, "medium")
    camera_input = _scene_input(manifest, "camera")
    camera = camera_input["terrain_camera"]
    crop = _scene_input(manifest, "crop")
    sun = _scene_input(manifest, "sun")
    terrain_data = _scene_input(manifest, "terrain")
    atmosphere = _scene_input(manifest, "atmosphere")
    exposure = _scene_input(manifest, "exposure")
    tonemap = _scene_input(manifest, "tonemap")
    material_input = _scene_input(manifest, "material")
    reference_provenance = _object(FIXTURE / "reference-provenance.json")
    _keys(material_input, {"schema", "albedo", "metallic", "roughness", "triplanar_scale", "normal_strength", "blend_sharpness", "colormap_strength", "albedo_mode"}, "material")
    _keys(tonemap, {"schema", "operator", "formula", "encoding", "quantization"}, "tonemap")
    _keys(exposure, {"schema", "value"}, "exposure")
    _keys(atmosphere, {"schema", "kind", "environment_intensity"}, "atmosphere")
    _keys(camera_input, {"schema", "fov_y", "terrain_camera"}, "camera")
    _keys(camera, {"target", "radius", "phi_deg", "theta_deg", "mode"}, "camera.terrain_camera")
    _keys(terrain_data, {"schema", "dem", "dem_sha256", "dimensions", "spacing", "exaggeration"}, "terrain")
    _keys(sun, {"schema", "azimuth_deg", "elevation_deg", "intensity", "color"}, "sun")
    _keys(crop, {"schema", "x", "y", "width", "height", "full_viewport"}, "crop")
    _transport_identity(medium_data)
    if (
        material_input["schema"] != "forge3d.nephele.material/1"
        or material_input["albedo_mode"] != "material"
        or tonemap != {
            "schema": "forge3d.nephele.tonemap/1",
            "operator": "aces-fitted",
            "formula": "clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0,1)",
            "encoding": "IEC 61966-2-1 sRGB",
            "quantization": "round-to-nearest uint8",
        }
        or exposure["schema"] != "forge3d.nephele.exposure/1"
        or atmosphere["schema"] != "forge3d.nephele.atmosphere_input/1"
        or atmosphere["kind"] != "analytic_clear_sky"
        or camera_input["schema"] != "forge3d.nephele.camera/1"
        or camera["mode"] != "mesh:yup"
        or terrain_data["schema"] != "forge3d.nephele.terrain/1"
        or sun["schema"] != "forge3d.nephele.sun/1"
        or crop["schema"] != "forge3d.nephele.crop/1"
        or medium_data["schema"] != "forge3d.nephele.heterogeneous_medium/2"
        or medium_data["phase"].get("kind") != "henyey_greenstein"
    ):
        raise ValueError("tracked scene or color-pipeline input is incomplete")
    dimensions = terrain_data["dimensions"]
    spacing = terrain_data["spacing"]
    terrain_spans = [float(spacing[index]) * (int(dimensions[index]) - 1) for index in range(2)]
    if terrain_spans[0] != terrain_spans[1]:
        raise ValueError("NEPHELE terrain capture requires one square tracked terrain span")
    terrain_span = terrain_spans[0]
    grid_shape = medium_data["domain"].get("grid_shape")
    density_scale = float(medium_data["density_scale"])
    if (
        not isinstance(grid_shape, list)
        or len(grid_shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in grid_shape)
        or len(medium_data["density_r16"]) != math.prod(grid_shape)
        or not math.isfinite(density_scale)
        or density_scale <= 0.0
    ):
        raise ValueError("tracked medium density grid shape or sample count is invalid")
    density = np.asarray(medium_data["density_r16"], dtype=np.float32).reshape(
        grid_shape[2], grid_shape[1], grid_shape[0]
    ) / 65535.0
    terrain_path = _scene_input_path(manifest, "terrain_dem")
    if terrain_path.name != terrain_data["dem"] or _sha256(terrain_path) != terrain_data["dem_sha256"]:
        raise ValueError("tracked terrain DEM differs from terrain input hash")
    terrain = np.load(terrain_path, allow_pickle=False)
    if terrain.shape != (int(dimensions[1]), int(dimensions[0])):
        raise ValueError("tracked terrain DEM dimensions differ from terrain input")
    medium = Medium.grid3d(
        medium_data["sigma_a"], medium_data["sigma_s"], density,
        (medium_data["domain"]["bounds_min"], medium_data["domain"]["bounds_max"]),
        phase="henyey_greenstein", g=medium_data["phase"]["g"],
        density_scale=density_scale, version=1,
    )
    config = make_terrain_params_config(
        size_px=tuple(crop["full_viewport"]), render_scale=1.0, terrain_span=terrain_span, msaa_samples=1, z_scale=float(terrain_data["exaggeration"]),
        exposure=float(exposure["value"]), domain=(float(terrain.min()), float(terrain.max())), light_azimuth_deg=sun["azimuth_deg"],
        light_elevation_deg=sun["elevation_deg"], sun_intensity=sun["intensity"], sun_color=sun["color"],
        albedo_mode=material_input["albedo_mode"], colormap_strength=float(material_input["colormap_strength"]),
        cam_radius=camera["radius"], cam_phi_deg=camera["phi_deg"], cam_theta_deg=camera["theta_deg"],
        cam_target=camera["target"], fov_y_deg=float(camera_input["fov_y"]), camera_mode=camera["mode"], aa_samples=1, aa_seed=0x4E455048,
        tonemap=TonemapSettings(operator="aces"), aov=AovSettings(enabled=True, transmittance=True, in_scatter=True, cloud_shadow=True, optical_depth=True), media=medium,
    )
    params = f3d.TerrainRenderParams(config)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material = f3d.MaterialSet.custom(
        tuple(material_input["albedo"]), float(material_input["metallic"]), float(material_input["roughness"]),
        triplanar_scale=float(material_input["triplanar_scale"]), normal_strength=float(material_input["normal_strength"]), blend_sharpness=float(material_input["blend_sharpness"]),
    )
    with tempfile.TemporaryDirectory(prefix="nephele-ibl-") as temporary:
        hdr = Path(temporary) / "fixture.hdr"
        _write_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=float(atmosphere["environment_intensity"]))
        capture = renderer._capture_nephele_acceptance(
            material, ibl, params, terrain,
            terrain_occlusion_in_media=True,
            include_no_medium=capture_primary,
        )
        probe = _object(artifact / "adapter-probe.json")["probe"]
        diagnostics = dict(capture["diagnostics"])
        _validate_candidate_diagnostics(diagnostics, head, probe)
        camera_contract = _camera_contract(dict(capture["camera_contract"]))
        reference_camera_contract = _camera_contract(reference_provenance["camera_contract"])
        if any(
            not np.array_equal(
                np.asarray(camera_contract[key], dtype=np.float32),
                np.asarray(reference_camera_contract[key], dtype=np.float32),
            )
            for key in camera_contract
        ):
            raise ValueError("candidate and reference camera contracts differ bitwise")
        frame = np.asarray(capture["beauty"], dtype=np.uint8)
        np.save(artifact / f"froxel-frame-run-{label}.npy", frame, allow_pickle=False)
        if capture_primary:
            for filename, key in (("realtime-rgb.npy", "beauty"), ("transmittance.npy", "transmittance"), ("in-scatter.npy", "in_scatter"), ("cloud-shadow-aov.npy", "cloud_shadow"), ("optical-depth.npy", "optical_depth"), ("realtime-termination-slice.npy", "termination_slice")):
                np.save(artifact / filename, _crop(capture[key], crop), allow_pickle=False)
            np.save(
                artifact / "medium-disabled-rgb.npy",
                _crop(np.asarray(capture["no_medium_beauty"], dtype=np.uint8), crop),
                allow_pickle=False,
            )
            ablation = renderer._capture_nephele_acceptance(material, ibl, params, terrain, terrain_occlusion_in_media=False)
            np.save(artifact / "terrain-occlusion-disabled-rgb.npy", _crop(np.asarray(ablation["beauty"], dtype=np.uint8), crop), allow_pickle=False)
            output_height, output_width = frame.shape[:2]
            termination = np.asarray(capture["termination_slice"])
            froxel_depth = capture.get("froxel_depth")
            if termination.shape[:2] != (output_height, output_width):
                raise ValueError("render-scale-one termination dimensions differ from output")
            if (
                type(froxel_depth) is not int
                or froxel_depth < 1
                or not np.isfinite(termination).all()
                or np.any(termination < 0)
                or np.any(termination > froxel_depth - 1)
                or np.any(termination != np.floor(termination))
            ):
                raise ValueError("termination slices do not define exact integration work")
            termination_integration_steps = int(
                np.asarray(termination, dtype=np.uint64).sum(dtype=np.uint64)
                + termination.size
            )
            _json(artifact / "memory.json", {
                "schema": "forge3d.nephele.memory/2",
                "output_dimensions": [output_width, output_height],
                "internal_dimensions": [output_width, output_height],
                "include_no_medium": True,
                "froxel_depth": froxel_depth,
                "termination_integration_steps": termination_integration_steps,
                **dict(capture["memory"]),
            })
    input_hash = _fixture_bundle(manifest)
    adapter_hash = hashlib.sha256(_canonical(probe)).hexdigest()
    frame_path = artifact / f"froxel-frame-run-{label}.npy"
    nonce = hashlib.sha256(f"{head}:{label}:{os.getpid()}".encode()).hexdigest()
    _json(artifact / f"run-{label}.json", {
        "schema": "forge3d.nephele.clean_run/1", "process_id": os.getpid(), "run_nonce": nonce,
        "head_sha": head, "tracked_worktree_clean": True, "fixture_manifest_sha256": _sha256(ROOT / "tests/nephele/fixture-manifest.json"),
        "input_bundle_sha256": input_hash, "adapter_identity_sha256": adapter_hash, "backend": "vulkan",
        "frame_file": frame_path.name, "frame_sha256": _sha256(frame_path),
        "camera_contract": camera_contract,
        "diagnostics": diagnostics,
        "presentation_identity": {
            "capture_api": "TerrainRenderer._capture_nephele_acceptance",
            "color_pipeline": ["white_balance", "exposure", "aces", "lut", "iec_61966_2_1_srgb8"],
            "no_medium_same_call": capture_primary,
        },
        "medium_disabled_rgb_sha256": _sha256(artifact / "medium-disabled-rgb.npy") if capture_primary else None,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "produce", "render"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--reference-wheel-dir", type=Path)
    parser.add_argument("--run-label", choices=("a", "b"))
    parser.add_argument("--capture-primary", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            if args.wheel_dir is None or args.reference_wheel_dir is None:
                raise ValueError("prepare requires --wheel-dir and --reference-wheel-dir")
            prepare(
                args.artifact_dir, args.head_sha, args.wheel_dir,
                args.reference_wheel_dir,
            )
        elif args.action == "produce":
            produce(args.artifact_dir, args.head_sha)
        else:
            if args.run_label is None:
                raise ValueError("render requires --run-label")
            render(args.artifact_dir, args.head_sha, args.run_label, args.capture_primary)
    except Exception as error:
        print(f"NEPHELE physical capture failed: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
