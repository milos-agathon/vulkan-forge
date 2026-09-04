"""Fail-closed exact-head and physical-lane validation for TESSELLA evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

if __package__:
    from .assert_junit_zero_skips import JUnitValidationError, verify_junit
    from .tessella_evidence_contract import non_finite_paths
else:
    from assert_junit_zero_skips import JUnitValidationError, verify_junit
    from tessella_evidence_contract import non_finite_paths


NVIDIA_VENDOR_ID = 0x10DE
LOD_TEST_NAME = (
    "terrain::clipmap::gpu_lod::tests::"
    "gpu_and_cpu_select_identical_tile_sets_for_1000_cameras"
)
HZB_TEST_NAME = (
    "terrain::culling::two_phase::tests::"
    "hzb_cull_shader_matches_the_cpu_occlusion_predicate"
)
PROVENANCE_FILES = (
    "run-context.json",
    "checked-out-head.txt",
    "adapter-probe.json",
    "gpu-cpu-lod-differential.log",
    "hzb-conservativeness-differential.log",
    "junit.xml",
)
REQUIRED_LABELS = {
    "self-hosted",
    "Windows",
    "X64",
    "forge3d-gpu",
    "gpu-nvidia",
}


def _load_object(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, [f"missing provenance file: {path.name} ({label})"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [f"{path.name}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{path.name}: JSON root must be an object"]
    paths = non_finite_paths(data)
    if paths:
        return None, [
            f"{path.name}: non-finite numeric value at {field_path}"
            for field_path in paths
        ]
    return data, []


def _run_context(artifact_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    context, load_errors = _load_object(
        artifact_dir / "run-context.json", "exact-head run context"
    )
    errors.extend(load_errors)
    checked_path = artifact_dir / "checked-out-head.txt"
    checked: str | None = None
    if not checked_path.is_file():
        errors.append(
            "missing provenance file: checked-out-head.txt (exact checked-out head)"
        )
    else:
        try:
            checked = checked_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            errors.append(f"checked-out-head.txt: unreadable exact head: {exc}")
    if checked is not None and not re.fullmatch(r"[0-9a-f]{40}", checked):
        errors.append("checked-out-head.txt: must contain one full lowercase Git SHA")

    if context is not None:
        string_fields = (
            "repository",
            "head_sha",
            "checked_out_head",
            "runner_name",
            "runner_os",
            "runner_arch",
            "required_backend",
        )
        for field in string_fields:
            if not isinstance(context.get(field), str) or not context[field]:
                errors.append(f"run-context.json: missing non-empty string '{field}'")
        labels = context.get("required_labels")
        if not isinstance(labels, list) or any(
            not isinstance(label, str) or not label for label in labels
        ):
            errors.append(
                "run-context.json: required_labels must be a list of non-empty strings"
            )
        elif not REQUIRED_LABELS.issubset(labels):
            errors.append(
                "run-context.json: required_labels do not identify the TESSELLA "
                "Windows NVIDIA GPU lane"
            )
        head = context.get("head_sha")
        if isinstance(head, str) and not re.fullmatch(r"[0-9a-f]{40}", head):
            errors.append("run-context.json: head_sha must be a full lowercase Git SHA")
        if context.get("required_backend") != "vulkan":
            errors.append("run-context.json: required_backend must equal 'vulkan'")
        if checked is not None and head != checked:
            errors.append(
                "exact-head mismatch: run-context.json head_sha does not equal "
                "checked-out-head.txt"
            )
        if checked is not None and context.get("checked_out_head") != checked:
            errors.append(
                "exact-head mismatch: run-context.json checked_out_head does not equal "
                "checked-out-head.txt"
            )

    exact = bool(
        context
        and checked
        and context.get("head_sha") == checked
        and context.get("checked_out_head") == checked
    )
    return {
        "run_context": context,
        "checked_out_head": checked,
        "exact_head": exact,
    }, errors


def _adapter(
    artifact_dir: Path,
    required_backend: str | None,
    capability_evidence: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    record, errors = _load_object(
        artifact_dir / "adapter-probe.json", "physical adapter probe"
    )
    if record is None:
        return None, errors
    probe = record.get("probe")
    if not isinstance(probe, dict):
        return record, errors + ["adapter-probe.json: probe must be an object"]
    if (
        record.get("requested_backend") != required_backend
        or required_backend != "vulkan"
    ):
        errors.append(
            "adapter-probe.json: requested_backend must match the required 'vulkan' backend"
        )
    name = probe.get("name")
    vendor = probe.get("vendor")
    backend = probe.get("backend")
    device_type = probe.get("device_type")
    if probe.get("status") != "ok":
        errors.append("adapter-probe.json: probe.status must equal 'ok'")
    if not isinstance(name, str) or "nvidia" not in name.lower():
        errors.append("adapter-probe.json: probe.name must identify an NVIDIA adapter")
    if (
        isinstance(vendor, bool)
        or not isinstance(vendor, int)
        or vendor != NVIDIA_VENDOR_ID
    ):
        errors.append(
            "adapter-probe.json: probe.vendor must equal NVIDIA vendor 0x10de"
        )
    if not isinstance(backend, str) or backend.lower() != "vulkan":
        errors.append("adapter-probe.json: probe.backend must equal 'vulkan'")
    if not isinstance(device_type, str) or device_type.lower() != "discretegpu":
        errors.append("adapter-probe.json: probe.device_type must equal 'DiscreteGpu'")
    if probe.get("software_fallback") is not False:
        errors.append("adapter-probe.json: software_fallback must be false")
    if (
        capability_evidence is not None
        and isinstance(name, str)
        and isinstance(backend, str)
    ):
        if capability_evidence.get("adapter") != name:
            errors.append(
                "capability_degradations.adapter does not match adapter-probe.json"
            )
        capability_backend = capability_evidence.get("backend")
        if (
            not isinstance(capability_backend, str)
            or capability_backend.lower() != backend.lower()
        ):
            errors.append(
                "capability_degradations.backend does not match adapter-probe.json"
            )
    return record, errors


def _lod_log(artifact_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = artifact_dir / "gpu-cpu-lod-differential.log"
    if not path.is_file():
        return None, [
            "missing provenance file: gpu-cpu-lod-differential.log "
            "(1,000-camera GPU/CPU LOD differential)"
        ]
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [f"gpu-cpu-lod-differential.log: unreadable test log: {exc}"]
    log = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    result_pattern = rf"(?m)^test {re.escape(LOD_TEST_NAME)} \.\.\. ok\s*$"
    summary_pattern = (
        r"(?m)^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
        r"\d+ filtered out; finished in "
    )
    errors: list[str] = []
    if len(re.findall(result_pattern, log)) != 1:
        errors.append(
            "gpu-cpu-lod-differential.log: exact 1,000-camera differential "
            "must report PASS exactly once"
        )
    if len(re.findall(summary_pattern, log)) != 1 or "test result: FAILED" in log:
        errors.append(
            "gpu-cpu-lod-differential.log: cargo summary must report exactly one "
            "passed, zero failed, and zero ignored tests"
        )
    return {
        "test": LOD_TEST_NAME,
        "camera_count": 1_000,
        "status": "pass" if not errors else "fail",
    }, errors


def _hzb_log(artifact_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = artifact_dir / "hzb-conservativeness-differential.log"
    if not path.is_file():
        return None, [
            "missing provenance file: hzb-conservativeness-differential.log "
            "(real-shader HZB conservativeness differential)"
        ]
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [
            "hzb-conservativeness-differential.log: unreadable test log: " f"{exc}"
        ]
    log = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    result_pattern = rf"(?m)^test {re.escape(HZB_TEST_NAME)} \.\.\. ok\s*$"
    summary_pattern = (
        r"(?m)^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
        r"\d+ filtered out; finished in "
    )
    errors: list[str] = []
    if len(re.findall(result_pattern, log)) != 1:
        errors.append(
            "hzb-conservativeness-differential.log: exact real-shader HZB "
            "differential must report PASS exactly once"
        )
    if len(re.findall(summary_pattern, log)) != 1 or "test result: FAILED" in log:
        errors.append(
            "hzb-conservativeness-differential.log: cargo summary must report "
            "exactly one passed, zero failed, and zero ignored tests"
        )
    return {
        "test": HZB_TEST_NAME,
        "status": "pass" if not errors else "fail",
    }, errors


def _junit(artifact_dir: Path) -> tuple[dict[str, int] | None, list[str]]:
    try:
        counts = verify_junit(artifact_dir / "junit.xml")
    except JUnitValidationError as exc:
        return None, [f"junit.xml: {exc}"]
    return counts.as_dict(), []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_provenance(
    artifact_dir: Path,
    input_names: set[str],
    capability_evidence: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    context, context_errors = _run_context(artifact_dir)
    run_context = context.get("run_context")
    required_backend = (
        run_context.get("required_backend") if isinstance(run_context, dict) else None
    )
    adapter, adapter_errors = _adapter(
        artifact_dir, required_backend, capability_evidence
    )
    lod, lod_errors = _lod_log(artifact_dir)
    hzb, hzb_errors = _hzb_log(artifact_dir)
    junit, junit_errors = _junit(artifact_dir)
    errors = context_errors + adapter_errors + lod_errors + hzb_errors + junit_errors
    hashes: dict[str, str] = {}
    for name in sorted(input_names):
        path = artifact_dir / name
        if path.is_file():
            try:
                hashes[name] = _sha256(path)
            except OSError as exc:
                errors.append(f"{name}: could not hash evidence input: {exc}")
    return {
        **context,
        "adapter_probe": adapter,
        "gpu_cpu_lod_differential": lod,
        "hzb_conservativeness_differential": hzb,
        "junit": junit,
        "input_sha256": hashes,
    }, errors
