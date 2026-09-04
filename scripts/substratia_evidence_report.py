#!/usr/bin/env python3
"""Verify SUBSTRATIA physical-GPU evidence without trusting its ledger.

The test ledger is useful context, but it is not proof by itself.  This verifier
binds every claim to an explicit clean candidate commit and physical adapter,
loads the candidate's golden bytes directly from Git, decodes the PNGs (including
CRC and filter validation), recomputes the image metrics, and checks the exact
four moonshot acceptance tests in JUnit.  A PASS report and lane marker are only
written after every independent check succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA = "forge3d.substratia.results.v1"
REPORT_SCHEMA = "forge3d.substratia.verification.v1"
LANE_SCHEMA = "forge3d.substratia.lane.v1"
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
SSIM_DELTA_MIN = 0.05
GOLDEN_SSIM_MIN = 0.99
GOLDEN_MEAN_ERROR_MAX = 0.01
PARTIAL_COVERAGE_MIN = 0.02
PARTIAL_ERROR_MAX = 0.02
GRAZING_REGION = (0.18, 0.85, 0.12, 0.88)
CORE_TESTS = (
    "test_normal_family_changes_lighting_ssim",
    "test_all_families_page_within_budget",
    "test_missing_family_is_fatal",
    "test_partial_normal_residency_degrades_gracefully",
)
PNG_NAMES = (
    "actual_baseline.png",
    "actual_normal.png",
    "golden_baseline.png",
    "golden_normal.png",
)
UNSAFE_ADAPTER_TOKENS = (
    "basic render driver",
    "lavapipe",
    "llvmpipe",
    "swiftshader",
    "software",
    "virtual",
    "paravirtual",
    "virtio",
    "warp",
)


class EvidenceError(RuntimeError):
    """The supplied evidence does not prove the SUBSTRATIA claim."""


@dataclass(frozen=True)
class DecodedPng:
    pixels: np.ndarray
    width: int
    height: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON evidence {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON evidence is not an object: {path}")
    return value


def _git(repository: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=text,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr if text else exc.stderr.decode("utf-8", "replace")
        raise EvidenceError(f"git {' '.join(args)} failed: {detail.strip()}") from exc
    return result.stdout


def _validate_candidate(repository: Path, candidate_sha: str, results: dict) -> None:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", candidate_sha) is not None,
        "candidate SHA must be an explicit full 40-character lowercase Git SHA",
    )
    head = str(_git(repository, "rev-parse", "HEAD")).strip()
    _require(head == candidate_sha, f"candidate SHA {candidate_sha} != checked-out HEAD {head}")
    _git(repository, "cat-file", "-e", f"{candidate_sha}^{{commit}}")
    dirty = str(_git(repository, "status", "--porcelain", "--untracked-files=no")).strip()
    _require(not dirty, f"candidate has tracked working-tree changes:\n{dirty}")
    _require(
        results.get("candidate_sha") == candidate_sha,
        "results.json is not bound to the explicit candidate SHA",
    )


def _validate_probe(probe: object, expected_backend: str) -> dict:
    _require(isinstance(probe, dict), "adapter probe is not an object")
    backend = str(probe.get("backend", "")).lower()
    device_type = str(probe.get("device_type", "")).lower()
    name = str(probe.get("name", ""))
    name_lower = name.lower()
    vendor = int(probe.get("vendor", 0))
    device = int(probe.get("device", 0))
    _require(probe.get("status") == "ok", "adapter probe did not return status=ok")
    _require(backend == expected_backend, "selected adapter backend differs from the required backend")
    _require(
        probe.get("software_fallback") is False,
        "adapter probe does not explicitly prove software_fallback=false",
    )
    _require(
        device_type in {"discretegpu", "integratedgpu"},
        f"adapter device type is not physical GPU evidence: {device_type!r}",
    )
    _require(
        not any(token in name_lower for token in UNSAFE_ADAPTER_TOKENS),
        f"software/virtual/paravirtual adapter is forbidden: {name!r}",
    )
    _require(name.strip() != "", "adapter name is empty")
    if expected_backend == "vulkan":
        _require(
            device_type == "discretegpu",
            "NVIDIA Vulkan evidence requires a discrete GPU",
        )
        _require(vendor == 0x10DE, "NVIDIA Vulkan evidence has a non-NVIDIA vendor ID")
        _require("nvidia" in name_lower, "NVIDIA Vulkan evidence has a non-NVIDIA adapter name")
    return {
        "backend": backend,
        "device_type": device_type,
        "name": name,
        "vendor": vendor,
        "device": device,
    }


def _validate_adapter(path: Path, expected_backend: str) -> dict:
    envelope = _read_json(path)
    requested = str(envelope.get("requested_backend", "")).lower()
    _require(requested == expected_backend, "adapter probe requested the wrong backend")
    return _validate_probe(envelope.get("probe"), expected_backend)


def _validate_render_adapter(path: Path, expected_backend: str, expected: dict) -> dict:
    actual = _validate_probe(_read_json(path), expected_backend)
    for field in ("backend", "device_type", "name", "vendor", "device"):
        _require(
            str(actual.get(field, "")).lower() == str(expected.get(field, "")).lower(),
            f"render-process adapter {field} differs from the workflow probe",
        )
    return actual


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(path: Path) -> DecodedPng:
    """Decode a non-interlaced 8-bit PNG and validate every chunk CRC."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read PNG {path}: {exc}") from exc
    _require(data.startswith(b"\x89PNG\r\n\x1a\n"), f"invalid PNG signature: {path}")
    offset = 8
    width = height = color_type = bit_depth = interlace = None
    idat = bytearray()
    seen_iend = False
    while offset < len(data):
        _require(offset + 12 <= len(data), f"truncated PNG chunk in {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        _require(end <= len(data), f"truncated {kind!r} PNG chunk in {path}")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = zlib.crc32(kind)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        _require(actual_crc == expected_crc, f"PNG CRC mismatch in {path} chunk {kind!r}")
        if kind == b"IHDR":
            _require(length == 13 and width is None, f"invalid duplicate IHDR in {path}")
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            _require(width > 0 and height > 0, f"invalid PNG dimensions in {path}")
            _require(compression == 0 and filter_method == 0, f"unsupported PNG encoding in {path}")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            _require(length == 0, f"invalid IEND in {path}")
            seen_iend = True
            offset = end
            break
        offset = end
    _require(seen_iend and offset == len(data), f"invalid trailing or missing IEND in {path}")
    _require(bit_depth == 8, f"only 8-bit PNG evidence is accepted: {path}")
    _require(interlace == 0, f"interlaced PNG evidence is not accepted: {path}")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    _require(channels is not None, f"unsupported PNG color type {color_type} in {path}")
    assert width is not None and height is not None
    stride = width * channels
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise EvidenceError(f"invalid PNG compressed stream in {path}: {exc}") from exc
    _require(len(raw) == height * (stride + 1), f"unexpected PNG scanline size in {path}")
    rows = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for y in range(height):
        filter_type = raw[cursor]
        cursor += 1
        source = raw[cursor : cursor + stride]
        cursor += stride
        _require(filter_type <= 4, f"unknown PNG filter {filter_type} in {path}")
        reconstructed = np.empty(stride, dtype=np.uint8)
        for x, byte in enumerate(source):
            left = int(reconstructed[x - channels]) if x >= channels else 0
            above = int(previous[x])
            upper_left = int(previous[x - channels]) if x >= channels else 0
            predictor = 0
            if filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth(left, above, upper_left)
            reconstructed[x] = (int(byte) + predictor) & 0xFF
        rows[y] = reconstructed
        previous = reconstructed
    return DecodedPng(rows.reshape(height, width, channels), width, height)


def _box_mean(image: np.ndarray, radius: int = 3) -> np.ndarray:
    size = 2 * radius + 1
    padded = np.pad(image, radius, mode="edge").astype(np.float64)
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)))
    height, width = image.shape
    total = (
        csum[size : size + height, size : size + width]
        - csum[:height, size : size + width]
        - csum[size : size + height, :width]
        + csum[:height, :width]
    )
    return total / float(size * size)


def _luminance(pixels: np.ndarray) -> np.ndarray:
    if pixels.shape[2] == 1:
        return pixels[..., 0].astype(np.float64) / 255.0
    rgb = pixels[..., :3].astype(np.float64) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    _require(a.shape == b.shape and a.ndim == 2, "SSIM inputs have mismatched shapes")
    mu_a, mu_b = _box_mean(a), _box_mean(b)
    sigma_a = _box_mean(a * a) - mu_a * mu_a
    sigma_b = _box_mean(b * b) - mu_b * mu_b
    sigma_ab = _box_mean(a * b) - mu_a * mu_b
    numerator = (2.0 * mu_a * mu_b + 0.01**2) * (2.0 * sigma_ab + 0.03**2)
    denominator = (mu_a * mu_a + mu_b * mu_b + 0.01**2) * (
        sigma_a + sigma_b + 0.03**2
    )
    return float(np.mean(numerator / denominator))


def _metric_equal(declared: object, actual: float, label: str) -> None:
    _require(isinstance(declared, (int, float)), f"missing numeric metric {label}")
    _require(np.isfinite(float(declared)), f"non-finite metric {label}")
    _require(abs(float(declared) - actual) <= 1e-6, f"declared {label} does not match recomputation")


def _candidate_golden_bytes(repository: Path, candidate_sha: str, relative: str) -> bytes:
    value = _git(repository, "show", f"{candidate_sha}:{relative}", text=False)
    assert isinstance(value, bytes)
    return value


def _validate_images(
    artifact_dir: Path,
    repository: Path,
    candidate_sha: str,
    adapter: dict,
    gate: dict,
) -> dict:
    for name in PNG_NAMES:
        _require((artifact_dir / name).is_file(), f"missing image artifact {name}")
    expected_names = {
        "actual_baseline": "actual_baseline.png",
        "actual_normal": "actual_normal.png",
        "golden_baseline": "golden_baseline.png",
        "golden_normal": "golden_normal.png",
    }
    for key, expected in expected_names.items():
        _require(gate.get(key) == expected, f"non-canonical {key} selection")

    backend = adapter["backend"]
    if backend == "metal":
        variant = "metal"
    elif backend == "vulkan":
        variant = "nvidia-vulkan"
    else:
        raise EvidenceError(f"no canonical SUBSTRATIA golden for adapter {adapter}")
    tracked_paths = {
        "golden_baseline.png": f"tests/golden/terrain/substratia_grazing_baseline.{variant}.png",
        "golden_normal.png": f"tests/golden/terrain/substratia_grazing_normal.{variant}.png",
    }
    hashes: dict[str, str] = {}
    for artifact_name, tracked_path in tracked_paths.items():
        tracked = _candidate_golden_bytes(repository, candidate_sha, tracked_path)
        working_path = repository / tracked_path
        _require(working_path.read_bytes() == tracked, f"working golden differs from candidate: {tracked_path}")
        artifact = (artifact_dir / artifact_name).read_bytes()
        _require(artifact == tracked, f"artifact golden is not candidate-tracked bytes: {artifact_name}")
        hashes[artifact_name] = hashlib.sha256(artifact).hexdigest()

    decoded = {name: decode_png(artifact_dir / name) for name in PNG_NAMES}
    shapes = {(value.height, value.width) for value in decoded.values()}
    _require(len(shapes) == 1, "actual and golden image dimensions differ")
    for label in ("baseline", "normal"):
        pixels = np.ascontiguousarray(decoded[f"actual_{label}.png"].pixels)
        rgba_sha256 = hashlib.sha256(pixels.tobytes()).hexdigest()
        _require(
            gate.get(f"actual_{label}_rgba_sha256") == rgba_sha256,
            f"declared actual_{label}_rgba_sha256 does not match decoded RGBA",
        )
    luminance = {name: _luminance(value.pixels) for name, value in decoded.items()}
    height, width = next(iter(shapes))
    region = gate.get("region")
    _require(
        isinstance(region, list) and tuple(float(v) for v in region) == GRAZING_REGION,
        "normal-lighting metric uses a non-canonical grazing region",
    )
    top, bottom, left, right = GRAZING_REGION
    rows = slice(int(height * top), int(height * bottom))
    cols = slice(int(width * left), int(width * right))
    ssim = _ssim(
        luminance["actual_baseline.png"][rows, cols],
        luminance["actual_normal.png"][rows, cols],
    )
    delta = 1.0 - ssim
    _require(delta > SSIM_DELTA_MIN, f"normal-lighting SSIM delta {delta:.6f} <= {SSIM_DELTA_MIN}")
    _require(float(gate.get("threshold", -1)) == SSIM_DELTA_MIN, "SSIM threshold drift")
    _metric_equal(gate.get("ssim"), ssim, "normal_lighting_ssim.ssim")
    _metric_equal(gate.get("ssim_delta"), delta, "normal_lighting_ssim.ssim_delta")

    golden_metrics: dict[str, float] = {}
    for label in ("baseline", "normal"):
        actual = luminance[f"actual_{label}.png"]
        golden = luminance[f"golden_{label}.png"]
        golden_ssim = _ssim(actual, golden)
        actual_rgba = decoded[f"actual_{label}.png"].pixels.astype(np.float64)
        golden_rgba = decoded[f"golden_{label}.png"].pixels.astype(np.float64)
        mean_error = float(np.mean(np.abs(actual_rgba - golden_rgba)) / 255.0)
        _require(golden_ssim >= GOLDEN_SSIM_MIN, f"{label} golden SSIM {golden_ssim:.6f} < {GOLDEN_SSIM_MIN}")
        _require(mean_error <= GOLDEN_MEAN_ERROR_MAX, f"{label} golden mean error {mean_error:.6f} > {GOLDEN_MEAN_ERROR_MAX}")
        _metric_equal(gate.get(f"golden_ssim_{label}"), golden_ssim, f"golden_ssim_{label}")
        _metric_equal(gate.get(f"golden_mean_error_{label}"), mean_error, f"golden_mean_error_{label}")
        golden_metrics[f"golden_ssim_{label}"] = golden_ssim
        golden_metrics[f"golden_mean_error_{label}"] = mean_error
    for name in ("actual_baseline.png", "actual_normal.png"):
        hashes[name] = hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
    return {"ssim": ssim, "ssim_delta": delta, **golden_metrics, "sha256": hashes}


def _pass_gate(gates: dict, name: str) -> dict:
    gate = gates.get(name)
    _require(isinstance(gate, dict), f"missing results gate {name}")
    _require(gate.get("status") == "PASS", f"results gate {name} is not PASS")
    return gate


def _validate_non_image_gates(gates: dict) -> dict:
    budget = _pass_gate(gates, "family_residency_budget")
    resident = budget.get("resident_bytes")
    family_budget = budget.get("family_budget_bytes")
    _require(isinstance(resident, dict) and isinstance(family_budget, dict), "invalid family budget maps")
    families = ("albedo", "normal", "mask")
    resident_values = []
    family_budget_values = []
    for family in families:
        value = resident.get(family)
        limit = family_budget.get(family)
        _require(isinstance(value, int) and value > 0, f"{family} has no resident bytes")
        _require(isinstance(limit, int) and limit > 0, f"{family} has no family budget")
        _require(value <= limit, f"{family} exceeds its family budget")
        resident_values.append(value)
        family_budget_values.append(limit)
    total = sum(resident_values)
    configured = budget.get("configured_budget_bytes")
    _require(budget.get("total_resident_bytes") == total, "resident total is inconsistent")
    _require(isinstance(configured, int) and configured > 0, "invalid configured VT budget")
    _require(sum(family_budget_values) <= configured, "family budgets exceed configured VT budget")
    _require(total <= configured <= MEMORY_LIMIT_BYTES, "VT residency exceeds the 512 MiB contract")
    _require(budget.get("memory_limit_bytes") == MEMORY_LIMIT_BYTES, "memory ceiling drift")

    missing = _pass_gate(gates, "missing_family_fatal")
    message = str(missing.get("message", ""))
    _require(
        "family 'normal' requested but no source registered" in message
        and "refusing to render with corrupted PBR" in message,
        "missing-family evidence is not the required fatal diagnostic",
    )

    partial = _pass_gate(gates, "partial_normal_residency")
    coverage = float(partial.get("fallback_coverage", -1.0))
    error = float(partial.get("mean_luminance_error", float("inf")))
    threshold = float(partial.get("error_threshold", -1.0))
    _require(np.isfinite(coverage) and coverage > PARTIAL_COVERAGE_MIN, "partial fallback coverage is insufficient")
    _require(threshold == PARTIAL_ERROR_MAX, "partial-residency error threshold drift")
    _require(np.isfinite(error) and 0.0 <= error < threshold, "partial-residency fallback is corrupted")
    return {
        "resident_bytes": dict(zip(families, resident_values)),
        "total_resident_bytes": total,
        "configured_budget_bytes": configured,
        "partial_fallback_coverage": coverage,
        "partial_mean_luminance_error": error,
    }


def _validate_junit(path: Path) -> dict:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise EvidenceError(f"cannot parse JUnit evidence {path}: {exc}") from exc
    cases: dict[str, list[ET.Element]] = {name: [] for name in CORE_TESTS}
    for case in root.iter("testcase"):
        name = case.get("name", "")
        if name in cases:
            cases[name].append(case)
    for name, matches in cases.items():
        _require(len(matches) == 1, f"JUnit must contain exactly one {name}, found {len(matches)}")
        case = matches[0]
        bad = [child.tag for child in case if child.tag in {"failure", "error", "skipped"}]
        _require(not bad, f"JUnit core test {name} did not pass: {bad}")
    return {"passed_core_tests": list(CORE_TESTS), "junit_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(args: argparse.Namespace) -> dict:
    artifact_dir = args.artifact_dir.resolve()
    repository = args.repository.resolve()
    results_path = artifact_dir / "results.json"
    results = _read_json(results_path)
    _require(results.get("schema") == SCHEMA, "SUBSTRATIA results schema mismatch")
    _validate_candidate(repository, args.candidate_sha, results)
    adapter = _validate_adapter(args.adapter_probe.resolve(), args.expected_backend)
    render_adapter = _validate_render_adapter(
        args.render_adapter.resolve(), args.expected_backend, adapter
    )
    gates = results.get("gates")
    _require(isinstance(gates, dict), "results.json has no gates object")
    normal = _pass_gate(gates, "normal_lighting_ssim")
    image_metrics = _validate_images(artifact_dir, repository, args.candidate_sha, adapter, normal)
    gate_metrics = _validate_non_image_gates(gates)
    junit = _validate_junit(args.junit.resolve())
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "candidate_sha": args.candidate_sha,
        "adapter": adapter,
        "render_adapter": render_adapter,
        "metrics": {**image_metrics, **gate_metrics},
        "junit": junit,
        "inputs": {
            "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
            "adapter_probe_sha256": hashlib.sha256(args.adapter_probe.read_bytes()).hexdigest(),
            "render_adapter_sha256": hashlib.sha256(args.render_adapter.read_bytes()).hexdigest(),
        },
    }
    report_path = artifact_dir / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lane_marker = {
        "schema": LANE_SCHEMA,
        "status": "RAN",
        "verifier_status": "PASS",
        "candidate_sha": args.candidate_sha,
        "adapter": adapter,
        "verification_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    (artifact_dir / "lane-ran.json").write_text(
        json.dumps(lane_marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--adapter-probe", type=Path, required=True)
    parser.add_argument("--render-adapter", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--expected-backend", choices=("metal", "vulkan"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = verify(args)
    except EvidenceError as exc:
        print(f"SUBSTRATIA evidence: FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
