#!/usr/bin/env python3
"""Verify and summarize exact-head HELIOS physical acceptance evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_TESTS = {
    "test_spa_worked_example_matches_nrel",
    "test_spa_matches_official_reference_rows",
    "test_real_dem_curvature_and_refraction_are_load_bearing",
    "test_viewshed_matches_committed_curved_whitebox_reference",
    "test_shadow_tip_bearing_and_curved_length_contract",
    "test_curved_shadow_memory_matches_flat_baseline",
    "test_shadow_mask_is_boolean_and_bitwise_deterministic",
    "test_shadow_mask_golden",
    "test_shadow_mask_is_identical_on_dx12_and_vulkan",
    "test_viewshed_and_shadow_mask_report_exact_tracked_host_memory",
}


class EvidenceError(ValueError):
    """Raised when a required HELIOS artifact is absent or untrustworthy."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"missing or unreadable evidence: {path.name}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read(path))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON evidence: {path.name}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path.name} must contain a JSON object")
    return value


def _one(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise EvidenceError(f"{label} metric must appear exactly once")
    return matches[0]


def _floats(match: re.Match[str]) -> list[float]:
    return [float(value) for value in match.groups()]


def _verify_adapter(record: dict[str, Any]) -> dict[str, Any]:
    probe = record.get("probe")
    if not isinstance(probe, dict):
        raise EvidenceError("adapter-probe.json probe must be an object")
    if record.get("status") != "passed":
        raise EvidenceError("adapter probe envelope status must equal 'passed'")
    valid = (
        str(record.get("requested_backend", "")).lower() == "vulkan"
        and probe.get("status") == "ok"
        and probe.get("vendor") == 0x10DE
        and "nvidia" in str(probe.get("name", "")).lower()
        and str(probe.get("backend", "")).lower() == "vulkan"
        and str(probe.get("device_type", "")).lower() == "discretegpu"
        and probe.get("software_fallback") is False
    )
    if not valid:
        raise EvidenceError("adapter probe is not physical NVIDIA Vulkan evidence")
    return probe


def _verify_junit(path: Path) -> int:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise EvidenceError("pytest-junit.xml is missing or malformed") from exc
    cases = list(root.iter("testcase"))
    names = {case.get("name", "") for case in cases}
    missing = sorted(REQUIRED_TESTS - names)
    if missing:
        raise EvidenceError(f"JUnit is missing required HELIOS tests: {missing}")
    for case in cases:
        if any(child.tag in {"failure", "error", "skipped"} for child in case):
            raise EvidenceError("JUnit contains a failure, error, or skip")
    return len(cases)


def _metrics(
    pytest_log: str,
    rust_log: str,
    production_gpu_log: str,
    acceptance_adapter: dict[str, Any],
    candidate_sha: str,
) -> dict[str, Any]:
    worked = _floats(
        _one(
            r"HELIOS SPA worked example: delta_zenith_deg=([0-9.eE+-]+), "
            r"delta_azimuth_deg=([0-9.eE+-]+)$",
            pytest_log,
            "SPA worked-example",
        )
    )
    reference_match = _one(
        r"HELIOS SPA reference rows: rows=(\d+), "
        r"max_zenith_error_deg=([0-9.eE+-]+), "
        r"max_azimuth_error_deg=([0-9.eE+-]+), "
        r"max_true_elevation_error_deg=([0-9.eE+-]+)$",
        pytest_log,
        "SPA reference-row",
    )
    reference = [int(reference_match.group(1))] + [
        float(value) for value in reference_match.groups()[1:]
    ]
    descent_match = _one(
        r"HELIOS 2D 256x256 conservative descent: rays=(\d+), "
        r"false_misses=(\d+), false_hit_rate=([0-9.eE+-]+), "
        r"shadow_mask_agreement=([0-9.eE+-]+), mask_false_misses=(\d+), "
        r"mask_false_hits=(\d+)$",
        rust_log,
        "conservative-descent",
    )
    descent = (
        int(descent_match.group(1)),
        int(descent_match.group(2)),
        float(descent_match.group(3)),
        float(descent_match.group(4)),
        int(descent_match.group(5)),
        int(descent_match.group(6)),
    )
    if not re.search(
        r"curvature_descent_is_conservative \.\.\. ok\s*$", rust_log, re.MULTILINE
    ):
        raise EvidenceError("conservative-descent Rust test did not report PASS")
    production_gpu_match = _one(
        r"HELIOS_PRODUCTION_TERRAIN_TRACE_GPU_JSON (\{.*\})$",
        production_gpu_log,
        "production terrain_trace GPU",
    )
    try:
        production_gpu = json.loads(production_gpu_match.group(1))
    except json.JSONDecodeError as exc:
        raise EvidenceError("production terrain_trace GPU record is invalid JSON") from exc
    if not re.search(
        r"curvature_descent_production_gpu_is_conservative \.\.\. ok\s*$",
        production_gpu_log,
        re.MULTILINE,
    ):
        raise EvidenceError("production terrain_trace GPU Rust test did not report PASS")
    load_bearing_match = _one(
        r"HELIOS load-bearing: IoU=([0-9.eE+-]+), flipped=(\d+), "
        r"extent=([0-9.eE+-]+)x([0-9.eE+-]+) m$",
        pytest_log,
        "load-bearing viewshed",
    )
    load_bearing = (
        float(load_bearing_match.group(1)),
        int(load_bearing_match.group(2)),
        float(load_bearing_match.group(3)),
        float(load_bearing_match.group(4)),
    )
    independent_match = _one(
        r"HELIOS curved Whitebox reference: IoU=([0-9.eE+-]+), flipped=(\d+), "
        r"flat_negative_iou=([0-9.eE+-]+), flat_negative_flipped=(\d+)$",
        pytest_log,
        "independent viewshed",
    )
    independent = (
        float(independent_match.group(1)),
        int(independent_match.group(2)),
        float(independent_match.group(3)),
        int(independent_match.group(4)),
    )
    shadow = _floats(
        _one(
            r"HELIOS shadow-tip: bearing_deg=([0-9.eE+-]+), "
            r"expected_bearing_deg=([0-9.eE+-]+), "
            r"bearing_error_deg=([0-9.eE+-]+), length_m=([0-9.eE+-]+), "
            r"curved_prediction_m=([0-9.eE+-]+), relative_error=([0-9.eE+-]+), "
            r"flat_length_m=([0-9.eE+-]+), flat_delta_fraction=([0-9.eE+-]+)$",
            pytest_log,
            "shadow-tip",
        )
    )
    memory_match = _one(r"HELIOS memory: (\{.*\})$", pytest_log, "memory")
    try:
        memory = json.loads(memory_match.group(1))
    except json.JSONDecodeError as exc:
        raise EvidenceError("HELIOS memory metric is not valid JSON") from exc
    path_memory_match = _one(
        r"HELIOS path memory: (\{.*\})$", pytest_log, "path memory"
    )
    try:
        path_memory = json.loads(path_memory_match.group(1))
    except json.JSONDecodeError as exc:
        raise EvidenceError("HELIOS path memory metric is not valid JSON") from exc

    if max(worked) > 0.0003 or reference[0] < 20 or max(reference[1:]) > 0.0003:
        raise EvidenceError("SPA accuracy gate failed")
    if descent[0] != 10_000 or descent[1] != 0 or descent[2] >= 0.001:
        raise EvidenceError("conservative-descent ray gate failed")
    if descent[3] < 0.999 or descent[4] != 0:
        raise EvidenceError("conservative-descent shadow-mask gate failed")
    expected_production_keys = {
        "schema",
        "status",
        "candidate_sha",
        "strict_physical_required",
        "adapter",
        "metrics",
    }
    if not isinstance(production_gpu, dict) or set(production_gpu) != expected_production_keys:
        raise EvidenceError("production terrain_trace GPU record is incomplete")
    if (
        production_gpu["schema"] != "forge3d.helios_production_terrain_trace_gpu/1"
        or production_gpu["status"] != "PASS"
        or production_gpu["candidate_sha"] != candidate_sha
        or production_gpu["strict_physical_required"] is not True
    ):
        raise EvidenceError("production terrain_trace GPU proof is not bound to candidate and strict lane")
    producer_adapter = production_gpu["adapter"]
    producer_identity = {
        "status",
        "name",
        "vendor",
        "device",
        "device_type",
        "backend",
        "driver",
        "driver_info",
        "software_fallback",
        "software_classification",
    }
    if not isinstance(producer_adapter, dict) or set(producer_adapter) != producer_identity:
        raise EvidenceError("production terrain_trace GPU adapter identity is incomplete")
    expected_adapter = {
        key: acceptance_adapter.get(key)
        for key in producer_identity - {"software_classification"}
    }
    expected_adapter["software_classification"] = (
        "software" if acceptance_adapter.get("software_fallback") else "hardware"
    )
    if producer_adapter != expected_adapter:
        raise EvidenceError(
            "production terrain_trace GPU proof is not bound to the acceptance adapter"
        )
    production_metrics = production_gpu["metrics"]
    if not isinstance(production_metrics, dict) or set(production_metrics) != {
        "rays",
        "false_misses",
        "false_hit_rate",
        "shadow_mask_pixels",
        "shadow_mask_agreement",
        "mask_false_misses",
        "mask_false_hits",
    }:
        raise EvidenceError("production terrain_trace GPU metrics are incomplete")
    if (
        production_metrics["rays"] != 10_000
        or production_metrics["false_misses"] != 0
        or production_metrics["false_hit_rate"] >= 0.001
    ):
        raise EvidenceError("production terrain_trace GPU ray gate failed")
    if (
        production_metrics["shadow_mask_pixels"] != 255 * 255
        or production_metrics["shadow_mask_agreement"] < 0.999
        or production_metrics["mask_false_misses"] != 0
    ):
        raise EvidenceError("production terrain_trace GPU shadow-mask gate failed")
    if load_bearing[0] > 0.96 or load_bearing[1] <= 0 or min(load_bearing[2:]) < 60_000:
        raise EvidenceError("load-bearing viewshed gate failed")
    if independent[0] < 0.98 or independent[2] >= 0.98:
        raise EvidenceError("independent viewshed gate failed")
    if shadow[2] > 0.05 or shadow[5] > 0.005 or shadow[7] <= 0.02:
        raise EvidenceError("shadow-tip gate failed")
    try:
        if set(memory) != {"schema", "before", "after", "delta"} or memory["schema"] != (
            "forge3d.helios_memory_comparison/1"
        ):
            raise EvidenceError("HELIOS memory metric has an unsupported schema")
        before = memory["before"]
        after = memory["after"]
        required_measurement = {
            "mode",
            "returncode",
            "package_path",
            "native_path",
            "native_sha256",
            "adapter",
            "workload",
            "metrics",
        }
        for label, measurement, expected_mode in (
            ("before", before, "flat_baseline"),
            ("after", after, "helios"),
        ):
            if not isinstance(measurement, dict) or set(measurement) != required_measurement:
                raise EvidenceError(f"HELIOS memory {label} measurement is incomplete")
            if measurement["mode"] != expected_mode or measurement["returncode"] != 0:
                raise EvidenceError(f"HELIOS memory {label} subprocess did not succeed")
            if not all(
                isinstance(measurement[key], str) and measurement[key]
                for key in ("package_path", "native_path")
            ):
                raise EvidenceError(f"HELIOS memory {label} import paths are invalid")
            if SHA256_RE.fullmatch(str(measurement["native_sha256"])) is None:
                raise EvidenceError(f"HELIOS memory {label} native SHA-256 is invalid")
            if not isinstance(measurement["adapter"], dict) or not measurement["adapter"]:
                raise EvidenceError(f"HELIOS memory {label} adapter is invalid")
            if not isinstance(measurement["workload"], dict) or not measurement["workload"]:
                raise EvidenceError(f"HELIOS memory {label} workload is invalid")
        if before["native_sha256"] != after["native_sha256"]:
            raise EvidenceError("HELIOS memory native SHA-256 mismatch")
        if before["native_path"] != after["native_path"]:
            raise EvidenceError("HELIOS memory native import path mismatch")
        if before["package_path"] != after["package_path"]:
            raise EvidenceError("HELIOS memory package import path mismatch")
        if before["adapter"] != after["adapter"]:
            raise EvidenceError("HELIOS memory adapter mismatch")
        identity_fields = (
            "status",
            "name",
            "vendor",
            "backend",
            "device_type",
            "software_fallback",
        )
        if any(
            before["adapter"].get(key) != acceptance_adapter.get(key)
            for key in identity_fields
        ):
            raise EvidenceError("HELIOS memory adapter differs from acceptance adapter")
        if before["workload"] != after["workload"]:
            raise EvidenceError("HELIOS memory workload mismatch")
        required_metrics = {
            "peak_host_visible_bytes",
            "gpu_resource_bytes",
            "minmax_pyramid_bytes",
        }
        for label, measurement in (("before", before), ("after", after)):
            values = measurement["metrics"]
            if not isinstance(values, dict) or set(values) != required_metrics:
                raise EvidenceError(f"HELIOS memory {label} metrics are incomplete")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values.values()):
                raise EvidenceError(f"HELIOS memory {label} metrics must be positive integers")
        if not isinstance(memory["delta"], dict) or set(memory["delta"]) != required_metrics:
            raise EvidenceError("HELIOS memory delta is incomplete")
        for key in required_metrics:
            expected_delta = int(after["metrics"][key]) - int(before["metrics"][key])
            if memory["delta"][key] != expected_delta:
                raise EvidenceError(f"HELIOS memory delta is invalid for {key}")
        for key in ("peak_host_visible_bytes", "gpu_resource_bytes"):
            if int(after["metrics"][key]) > math.ceil(int(before["metrics"][key]) * 1.05):
                raise EvidenceError(f"HELIOS memory gate failed for {key}")
        if int(after["metrics"]["minmax_pyramid_bytes"]) != int(
            before["metrics"]["minmax_pyramid_bytes"]
        ):
            raise EvidenceError("HELIOS min-max pyramid allocation changed")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError("HELIOS memory metric is incomplete") from exc
    try:
        if not isinstance(path_memory, dict) or set(path_memory) != {
            "viewshed",
            "shadow_mask",
        }:
            raise EvidenceError("HELIOS path memory metric is incomplete")
        required_memory_fields = {
            "current_host_visible_bytes",
            "expected_readback_bytes",
            "peak_host_visible_bytes",
        }
        for section in ("viewshed", "shadow_mask"):
            values = path_memory[section]
            if not isinstance(values, dict) or set(values) != required_memory_fields:
                raise EvidenceError(f"HELIOS {section} memory metric is incomplete")
            current = values["current_host_visible_bytes"]
            expected = values["expected_readback_bytes"]
            peak = values["peak_host_visible_bytes"]
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values.values()
            ):
                raise EvidenceError(f"HELIOS {section} memory values must be integers")
            if current != 0 or expected <= 0 or peak != expected:
                raise EvidenceError(f"HELIOS {section} memory accounting gate failed")
            if peak >= 512 * 1024 * 1024:
                raise EvidenceError(f"HELIOS {section} memory exceeds the 512 MiB budget")
    except (KeyError, TypeError) as exc:
        raise EvidenceError("HELIOS path memory metric is incomplete") from exc

    return {
        "sky": {
            "worked_delta_zenith_deg": worked[0],
            "worked_delta_azimuth_deg": worked[1],
            "reference_rows": reference[0],
            "max_reference_zenith_error_deg": reference[1],
            "max_reference_azimuth_error_deg": reference[2],
            "max_reference_true_elevation_error_deg": reference[3],
        },
        "conservative_descent": {
            "rays": descent[0],
            "false_misses": descent[1],
            "false_hit_rate": descent[2],
            "shadow_mask_agreement": descent[3],
            "mask_false_misses": descent[4],
            "mask_false_hits": descent[5],
        },
        "production_gpu_conservative_descent": production_gpu,
        "load_bearing_viewshed": {
            "iou": load_bearing[0],
            "flipped_pixels": load_bearing[1],
            "extent_east_west_m": load_bearing[2],
            "extent_north_south_m": load_bearing[3],
        },
        "independent_viewshed": {
            "iou": independent[0],
            "flipped_pixels": independent[1],
            "flat_negative_iou": independent[2],
            "flat_negative_flipped_pixels": independent[3],
        },
        "shadow_tip": {
            "bearing_deg": shadow[0],
            "expected_bearing_deg": shadow[1],
            "bearing_error_deg": shadow[2],
            "length_m": shadow[3],
            "curved_prediction_m": shadow[4],
            "relative_error": shadow[5],
            "flat_length_m": shadow[6],
            "flat_delta_fraction": shadow[7],
        },
        "memory": memory,
        "path_memory": path_memory,
    }


def verify(artifact_dir: Path, candidate_sha: str) -> dict[str, Any]:
    if SHA_RE.fullmatch(candidate_sha) is None:
        raise EvidenceError("candidate SHA must be a full lowercase Git SHA")
    checked_out = _read(artifact_dir / "checked-out-head.txt").strip()
    if checked_out != candidate_sha:
        raise EvidenceError("checked-out head does not match candidate SHA")
    rust_checked_out = _read(artifact_dir / "rust-checked-out-head.txt").strip()
    if rust_checked_out != candidate_sha:
        raise EvidenceError("Rust proof head does not match candidate SHA")
    adapter = _verify_adapter(_read_object(artifact_dir / "adapter-probe.json"))
    test_count = _verify_junit(artifact_dir / "pytest-junit.xml")
    metrics = _metrics(
        _read(artifact_dir / "pytest.log"),
        _read(artifact_dir / "rust-conservative-descent.log"),
        _read(artifact_dir / "production-gpu-conservative-descent.log"),
        adapter,
        candidate_sha,
    )
    report = {
        "status": "PASS",
        "candidate_sha": candidate_sha,
        "adapter": adapter,
        "pytest_test_count": test_count,
        "metrics": metrics,
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / "lane-ran.json").write_text(
        json.dumps(
            {"status": "RAN", "candidate_sha": candidate_sha},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha", required=True)
    args = parser.parse_args()
    try:
        report = verify(args.artifact_dir, args.candidate_sha)
    except EvidenceError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
