#!/usr/bin/env python3
"""Verify exact-head NEPHELE physical acceptance from raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from _deltae import delta_e_2000, srgb_to_lab  # noqa: E402
from _ssim import ssim  # noqa: E402
if __package__:
    from .nephele_heterogeneous_comparator import SAMPLES as COMPARATOR_SAMPLES, SEED as COMPARATOR_SEED, comparator_cache_key
    from .nephele_majorant_domain_probe import domain_coverage_sha256
    from .nephele_fixture_masks import MASK_FILES, generate_masks
    from .nephele_shader_analyzer import analyze_shaders
    from .record_media_reference_convergence import PREFIX_ARTIFACTS as REFERENCE_PREFIX_ARTIFACTS
else:
    from nephele_heterogeneous_comparator import SAMPLES as COMPARATOR_SAMPLES, SEED as COMPARATOR_SEED, comparator_cache_key
    from nephele_majorant_domain_probe import domain_coverage_sha256
    from nephele_fixture_masks import MASK_FILES, generate_masks
    from nephele_shader_analyzer import analyze_shaders
    from record_media_reference_convergence import PREFIX_ARTIFACTS as REFERENCE_PREFIX_ARTIFACTS


SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DENSITY_SAMPLING = "normalized-clamp-to-edge-linear-texel-center-uN-minus-0.5"
MAJORANT_QUERY = "floor(clamp(unit,0,1)*N)-clamped-to-N-minus-1"
MAJORANT_CONSTRUCTION = (
    "3x3x3-clamped-neighborhood-trilinear-outward-then-f32-extinction-outward"
)
MAJORANT_PROBE_MAPPING = (
    "domain-boundaries-texel-extrema-majorant-boundaries-then-irrational-lattice-v3"
)
SOFTWARE_TOKENS = (
    "basic render driver", "lavapipe", "llvmpipe", "swiftshader", "warp",
    "virtual", "paravirtual", "virtio", "software",
)
REQUIRED_JUNIT_CASES = frozenset(
    ("tests.test_nephele_physical", name)
    for name in (
        "gate1_estimator_majorant_rr",
        "gate2_energy",
        "gate3_realtime_reference",
        "gate4_terrain_coupling",
        "gate5_compute_shadow_ridgeline",
        "gate6_determinism_memory",
    )
)
FIXTURE_FILES = {
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
SCENE_INPUT_ROLES = {
    "camera", "terrain_dem", "terrain", "medium", "sun", "atmosphere", "exposure", "tonemap", "crop", "material"
}
REALTIME_DIAGNOSTIC_KEYS = {
    "majorant_proof", "majorant_valid", "sample_count", "step_count",
    "temporal_history_decision", "temporal_history_reason", "host_visible_bytes",
    "froxel_device_local_bytes", "density_device_local_bytes",
    "majorant_device_local_bytes", "staging_readback_bytes", "adapter", "backend",
    "driver", "source_revision", "executed_multi_scatter",
    "single_scatter_dispatches", "multiple_scatter_dispatches", "terrain_trace_queries",
    "single_scatter_luminance", "multiple_scatter_luminance",
    "energy_accounting_residual", "sun_transmittance_method",
    "sun_transmittance_bias", "sun_transmittance_max_segment_length",
    "sun_transmittance_executed_steps", "sun_transmittance_max_abs_error",
}
RAW_FILES = {
    *FIXTURE_FILES.values(),
    "realtime-rgb.npy",
    "medium-disabled-rgb.npy",
    "terrain-occlusion-disabled-rgb.npy",
    "realtime-termination-slice.npy",
    "froxel-frame-run-a.npy",
    "froxel-frame-run-b.npy",
    "transmittance.npy",
    "in-scatter.npy",
    "cloud-shadow-aov.npy",
    "optical-depth.npy",
    "gate1-statistics.json",
    "homogeneous-ratio-samples.bin",
    "heterogeneous-ratio-samples.bin",
    "heterogeneous-comparator.json",
    "heterogeneous-comparator.samples.bin",
    "majorant-domain-probe.json",
    "majorant-domain-probe.pairs.bin",
    "rr-evidence.json",
    "rr-contributions.bin",
    "gate2-energy.json",
    "gate2-energy-samples.bin",
    "gate5-static.json",
    "memory.json",
    "run-a.json",
    "run-b.json",
    "reference-provenance.json",
    "reference-convergence.json",
    "record_media_reference_convergence.py",
    "homogeneous-slab.json",
}
class EvidenceError(ValueError):
    """A structured, fail-closed evidence validation error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise EvidenceError(code, message)


def _keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            "schema_error",
            f"{label}: keys differ (missing={sorted(expected - actual)}, extra={sorted(actual - expected)})",
        )


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("artifact_error", f"{path.name}: invalid or missing JSON: {exc}")
    if not isinstance(value, dict):
        _fail("schema_error", f"{path.name}: JSON root must be an object")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        _fail("artifact_error", f"{path.name}: cannot hash artifact: {exc}")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        _fail("schema_error", f"{label}: expected a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        _fail("schema_error", f"{label}: expected a finite number ({exc})")
    if not math.isfinite(result):
        _fail("schema_error", f"{label}: expected a finite number")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail("schema_error", f"{label}: expected an integer >= {minimum}")
    return value


def _windows_arch(value: Any) -> str:
    normalized = str(value).strip().lower()
    return "x64" if normalized in {"x64", "amd64", "x86_64"} else normalized


def _identity_hash(value: Any, label: str) -> str:
    if not isinstance(value, dict):
        _fail("schema_error", f"{label}: expected an identity object")
    _keys(value, {"algorithm", "parameters", "sha256"}, label)
    if not isinstance(value["algorithm"], str) or not value["algorithm"].strip() or not isinstance(value["parameters"], dict):
        _fail("schema_error", f"{label}: algorithm or parameters are invalid")
    try:
        encoded = json.dumps(
            {"algorithm": value["algorithm"], "parameters": value["parameters"]},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        _fail("schema_error", f"{label}: identity is not canonical JSON: {exc}")
    derived = hashlib.sha256(encoded).hexdigest()
    if value["sha256"] != derived:
        _fail("schema_error", f"{label}: SHA-256 does not match algorithm and parameters")
    return derived


def _derived_cache_key(comparator: dict[str, Any]) -> str:
    try:
        return comparator_cache_key(comparator)
    except (KeyError, TypeError, ValueError) as exc:
        _fail("schema_error", f"comparator cache-key inputs are not canonical JSON: {exc}")


@dataclass(frozen=True)
class Stats:
    count: int
    mean: float
    variance: float
    standard_error: float


def _accumulator_from_array(values: np.ndarray) -> dict[str, float | int]:
    values = values.astype(np.float64, copy=False)
    return {
        "count": int(values.size),
        "sum": float(values.sum(dtype=np.float64)),
        "sum_squares": float(np.square(values).sum(dtype=np.float64)),
    }


def _stats(
    value: Any,
    label: str,
    *,
    exact_count: int | None = None,
    support: tuple[float, float] | None = None,
) -> Stats:
    if not isinstance(value, dict):
        _fail("schema_error", f"{label}: accumulator must be an object")
    _keys(value, {"count", "sum", "sum_squares"}, label)
    count = _integer(value["count"], f"{label}.count", minimum=2)
    if exact_count is not None and count != exact_count:
        _fail("gate_failure", f"{label}: count must equal {exact_count}")
    total = _finite(value["sum"], f"{label}.sum")
    total_sq = _finite(value["sum_squares"], f"{label}.sum_squares")
    if support is not None:
        low, high = support
        if (
            total < count * low
            or total > count * high
            or total_sq < 0.0
            or total_sq > count * max(low * low, high * high)
            or (low >= 0.0 and total_sq > high * total + 1.0e-12 * max(abs(total), 1.0))
        ):
            _fail("schema_error", f"{label}: accumulator is impossible for support [{low},{high}]")
    mean = total / count
    numerator = total_sq - total * total / count
    scale = max(abs(total_sq), abs(total * total / count), 1.0)
    if numerator < -1.0e-12 * scale:
        _fail("schema_error", f"{label}: impossible negative sample variance")
    variance = max(0.0, numerator) / (count - 1)
    return Stats(count, mean, variance, math.sqrt(variance / count))


def _nonnegative_stats(value: Any, label: str) -> Stats:
    if not isinstance(value, dict):
        _fail("schema_error", f"{label}: accumulator must be an object")
    _keys(value, {"count", "sum", "sum_squares", "minimum", "maximum"}, label)
    result = _stats(
        {key: value[key] for key in ("count", "sum", "sum_squares")}, label
    )
    minimum = _finite(value["minimum"], f"{label}.minimum")
    maximum = _finite(value["maximum"], f"{label}.maximum")
    total = _finite(value["sum"], f"{label}.sum")
    total_sq = _finite(value["sum_squares"], f"{label}.sum_squares")
    if (
        minimum < 0.0
        or maximum < minimum
        or total < result.count * minimum
        or total > result.count * maximum
        or total_sq < result.count * minimum * minimum
        or total_sq > maximum * total + 1.0e-12 * max(abs(total_sq), 1.0)
    ):
        _fail("schema_error", f"{label}: accumulator is impossible for finite nonnegative samples")
    return result


def _array(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        _fail("artifact_error", f"{path.name}: invalid or missing NumPy artifact: {exc}")
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        _fail("schema_error", f"{path.name}: expected a non-object ndarray")
    return value


def _rgb(path: Path) -> np.ndarray:
    value = _array(path)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        _fail("schema_error", f"{path.name}: expected uint8 sRGB shape (H,W,3)")
    return value


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = _array(path)
    if value.dtype != np.bool_ or value.shape != shape or not value.any():
        _fail("schema_error", f"{path.name}: expected a nonempty bool mask with shape {shape}")
    return value


def _roi_crop(mask: np.ndarray, *images: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if not mask[y0:y1, x0:x1].all():
        _fail("schema_error", "godray ROI must be one solid tracked rectangle")
    if y1 - y0 < 11 or x1 - x0 < 11:
        _fail("schema_error", "godray ROI must be at least the SSIM 11x11 window")
    return [image[y0:y1, x0:x1] for image in images]


def _delta_e(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return delta_e_2000(srgb_to_lab(left), srgb_to_lab(right))


def _verify_junit(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        _fail("artifact_error", f"junit.xml: invalid or missing XML: {exc}")
    local = lambda tag: tag.rsplit("}", 1)[-1]
    if local(root.tag) not in {"testsuite", "testsuites"}:
        _fail("junit_error", "junit.xml: root must be testsuite or testsuites")
    suites = [element for element in root.iter() if local(element.tag) == "testsuite"]
    if not suites or (local(root.tag) == "testsuites" and any(local(child.tag) != "testsuite" for child in root)):
        _fail("junit_error", "junit.xml: testsuites must contain validated testsuite elements only")
    parent = {child: element for element in root.iter() for child in element}
    cases = [element for element in root.iter() if local(element.tag) == "testcase"]
    if any(local(parent.get(case, root).tag) != "testsuite" for case in cases):
        _fail("junit_error", "junit.xml: testcase exists outside a validated testsuite")
    actual = [(case.get("classname", ""), case.get("name", "")) for case in cases]
    if len(actual) != 6 or set(actual) != REQUIRED_JUNIT_CASES or len(set(actual)) != len(actual):
        _fail("junit_error", "junit.xml: expected exactly six required physical cases, once each, with no extras")
    failures = errors = skipped = 0
    for case in cases:
        failures += sum(local(child.tag) == "failure" for child in case)
        errors += sum(local(child.tag) == "error" for child in case)
        skipped += sum(local(child.tag) == "skipped" for child in case)
    for suite in suites:
        if any(local(child.tag) not in {"testcase", "properties", "system-out", "system-err"} for child in suite):
            _fail("junit_error", "junit.xml: suite contains unsupported topology")
        descendants = [element for element in suite if local(element.tag) == "testcase"]
        recomputed = {
            "tests": len(descendants),
            "failures": sum(local(child.tag) == "failure" for case in descendants for child in case),
            "errors": sum(local(child.tag) == "error" for case in descendants for child in case),
            "skipped": sum(local(child.tag) == "skipped" for case in descendants for child in case),
        }
        try:
            recorded = {name: int(suite.attrib[name]) for name in recomputed}
        except (KeyError, TypeError, ValueError):
            _fail("junit_error", "junit.xml: every suite aggregate must contain integer tests/failures/errors/skipped")
        if recorded != recomputed:
            _fail("junit_error", "junit.xml: suite aggregate differs from recomputed testcase outcomes")
    if failures or errors or skipped:
        _fail("junit_error", "junit.xml: physical lane requires zero failures, errors, and skips")
    return {"tests": len(cases), "failures": failures, "errors": errors, "skipped": skipped}


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail("identity_error", f"git {' '.join(args)} failed: {exc}")
    return result.stdout.strip()


def _tracked_blob(repo_root: Path, commit: str, path: Path) -> bytes:
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}")
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail("provenance_error", f"{relative} is not a tracked blob at {commit}: {exc}")


def _verify_tool(
    record: Any,
    canonical: str,
    executed: Callable[..., Any],
    repo_root: Path,
    commit: str,
    label: str,
) -> str:
    if not isinstance(record, dict):
        _fail("schema_error", f"{label}: producer_tool must be an object")
    _keys(record, {"path", "sha256"}, f"{label}.producer_tool")
    path = repo_root / canonical
    digest = _sha256(path)
    if (
        record["path"] != canonical
        or record["sha256"] != digest
        or hashlib.sha256(_tracked_blob(repo_root, commit, path)).hexdigest() != digest
        or _sha256(Path(executed.__code__.co_filename)) != digest
    ):
        _fail("provenance_error", f"{label}: tracked producer tool is not the code being executed")
    return digest


def _verify_physical_producer(record: Any, repo_root: Path, commit: str) -> None:
    if not isinstance(record, dict):
        _fail("schema_error", "physical producer identity must be an object")
    _keys(record, {"implementation", "source_revision", "native_source", "producer_tool"}, "physical producer")
    if record["implementation"] != "canonical-ratio-delta-roulette-and-analog-sphere-v1" or record["source_revision"] != commit:
        _fail("provenance_error", "physical producer implementation or source revision differs")
    for key, expected in (("native_source", "src/media_py.rs"), ("producer_tool", "scripts/run_media_physical_capture.py")):
        value = record[key]
        if not isinstance(value, dict):
            _fail("schema_error", f"physical producer {key} must be an object")
        _keys(value, {"path", "sha256"}, f"physical producer {key}")
        path = repo_root / expected
        if value != {"path": expected, "sha256": _sha256(path)} or hashlib.sha256(_tracked_blob(repo_root, commit, path)).hexdigest() != value["sha256"]:
            _fail("provenance_error", f"physical producer {key} is not the exact tracked source")


def _verify_homogeneous_slab(
    record: Any, artifact_dir: Path, repo_root: Path, commit: str
) -> dict[str, Any]:
    if not isinstance(record, dict):
        _fail("schema_error", "homogeneous slab identity must be an object")
    _keys(record, {"path", "sha256"}, "homogeneous slab identity")
    canonical = repo_root / "tests/nephele/homogeneous-slab.json"
    expected = {"path": "tests/nephele/homogeneous-slab.json", "sha256": _sha256(canonical)}
    if (
        record != expected
        or _sha256(artifact_dir / "homogeneous-slab.json") != record["sha256"]
        or hashlib.sha256(_tracked_blob(repo_root, commit, canonical)).hexdigest() != record["sha256"]
    ):
        _fail("provenance_error", "homogeneous slab is not the exact tracked source input")
    slab = _object(artifact_dir / "homogeneous-slab.json")
    _keys(slab, {"schema", "analytic_comparator", "density", "distance", "sigma_a", "sigma_s", "sigma_t_max_channel"}, "homogeneous-slab.json")
    if (
        slab["schema"] != "forge3d.nephele.homogeneous_slab/1"
        or slab["analytic_comparator"] != "exp(-sigma_t_max_channel * density * distance)"
        or not isinstance(slab["sigma_a"], list)
        or not isinstance(slab["sigma_s"], list)
        or len(slab["sigma_a"]) != 3
        or len(slab["sigma_s"]) != 3
    ):
        _fail("schema_error", "homogeneous slab transport definition is incomplete")
    sigma_t = [
        _finite(a, "homogeneous_slab.sigma_a") + _finite(s, "homogeneous_slab.sigma_s")
        for a, s in zip(slab["sigma_a"], slab["sigma_s"])
    ]
    expected_sigma_t = _finite(slab["sigma_t_max_channel"], "homogeneous_slab.sigma_t_max_channel")
    density = _finite(slab["density"], "homogeneous_slab.density")
    distance = _finite(slab["distance"], "homogeneous_slab.distance")
    if expected_sigma_t <= 0.0 or density <= 0.0 or distance <= 0.0 or any(value != expected_sigma_t for value in sigma_t):
        _fail("schema_error", "homogeneous slab extinction or geometry differs from its canonical definition")
    return slab


def _raw_f64(path: Path, record: Any, label: str, count: int) -> np.ndarray:
    if not isinstance(record, dict):
        _fail("schema_error", f"{label}: raw output must be an object")
    expected = {"path", "sha256", "encoding", "samples"}
    if "columns" in record:
        expected.add("columns")
    _keys(record, expected, f"{label}.raw_output")
    if record["path"] != path.name or Path(str(record["path"])).name != record["path"] or record["encoding"] != "little-endian-f64" or record["samples"] != count:
        _fail("provenance_error", f"{label}: raw f64 execution contract differs")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail("artifact_error", f"{label}: raw output is missing: {exc}")
    if hashlib.sha256(payload).hexdigest() != record["sha256"] or len(payload) % 8:
        _fail("provenance_error", f"{label}: raw output bytes are corrupt")
    values = np.frombuffer(payload, dtype="<f8")
    if not np.isfinite(values).all():
        _fail("schema_error", f"{label}: raw output contains non-finite samples")
    return values


def _next_up_f32(value: np.float32) -> np.float32:
    if value == np.float32(0.0) or not np.isfinite(value):
        return value
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return np.asarray(bits + np.uint32(1), dtype=np.uint32).view(np.float32)[()]


def _outward_product_f32(left: np.float32, right: np.float32) -> np.float32:
    exact = float(left) * float(right)
    rounded = np.float32(left * right)
    return _next_up_f32(rounded) if float(rounded) < exact else rounded


def _trilinear_upper_f32(maximum: np.float32) -> np.float32:
    if maximum == np.float32(0.0):
        return maximum
    operations = 13.0
    unit_roundoff = 1.0 / float(1 << 24)
    denominator = 1.0 - operations * unit_roundoff
    half_min_subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0))) * 0.5
    exact = (
        float(maximum) * (1.0 + operations * unit_roundoff / denominator)
        + operations * half_min_subnormal / denominator
    )
    rounded = np.float32(exact)
    return _next_up_f32(rounded) if float(rounded) < exact else rounded


def _verify_transported_medium(medium: Any, label: str = "heterogeneous medium") -> dict[str, Any]:
    if not isinstance(medium, dict):
        _fail("schema_error", f"{label}: expected object")
    _keys(
        medium,
        {
            "schema", "domain", "density_r16", "density_transport", "majorant_cells",
            "majorant_transport", "transport", "sigma_a", "sigma_s", "phase",
            "density_scale",
        },
        label,
    )
    domain = medium["domain"]
    if not isinstance(domain, dict):
        _fail("schema_error", f"{label}.domain: expected object")
    _keys(domain, {"bounds_min", "bounds_max", "grid_shape"}, f"{label}.domain")
    shape = domain["grid_shape"]
    raw = medium["density_r16"]
    if (
        medium["schema"] != "forge3d.nephele.heterogeneous_medium/2"
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(_integer(value, f"{label}.domain.grid_shape", minimum=2) < 2 for value in shape)
        or not isinstance(raw, list)
        or len(raw) != math.prod(shape)
        or any(type(value) is not int or not 0 <= value <= 65535 for value in raw)
    ):
        _fail("schema_error", f"{label}: density grid is invalid")
    represented_f16 = (
        np.asarray(raw, dtype=np.float32) / np.float32(65535.0)
    ).astype(np.float16)
    f16_sha256 = hashlib.sha256(
        represented_f16.astype("<f2", copy=False).tobytes(order="C")
    ).hexdigest()
    expected_density_transport = {
        "schema": "forge3d.nephele.density_transport/1",
        "decode": "unorm16-div-65535-as-f32-then-ieee-f16-rne",
        "storage": "ieee-f16-bits-little-endian",
        "sampling": DENSITY_SAMPLING,
        "f16_sha256": f16_sha256,
    }
    expected_majorant_transport = {
        "schema": "forge3d.nephele.majorant_transport/1",
        "grid_shape": shape,
        "query": MAJORANT_QUERY,
        "construction": MAJORANT_CONSTRUCTION,
    }
    sigma_a, sigma_s = medium["sigma_a"], medium["sigma_s"]
    if any(
        not isinstance(values, list)
        or len(values) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        )
        for values in (sigma_a, sigma_s)
    ):
        _fail("schema_error", f"{label}: coefficient spectra are invalid")
    sigma_t_spectrum = [float(left) + float(right) for left, right in zip(sigma_a, sigma_s)]
    extinction_channel = max(range(3), key=sigma_t_spectrum.__getitem__)
    sigma_t_max = sigma_t_spectrum[extinction_channel]
    expected_transport = {
        "sigma_t_spectrum": sigma_t_spectrum,
        "sigma_t_max_channel": sigma_t_max,
        "extinction_channel": extinction_channel,
        "slab_axis": 2,
    }
    density_scale = _finite(medium["density_scale"], f"{label}.density_scale")
    if density_scale <= 0 or sigma_t_max <= 0:
        _fail("schema_error", f"{label}: density scale and extinction must be positive")
    if (
        medium["density_transport"] != expected_density_transport
        or medium["majorant_transport"] != expected_majorant_transport
        or medium["transport"] != expected_transport
    ):
        _fail("provenance_error", f"{label}: transported representation metadata differs")

    represented = represented_f16.astype(np.float32).reshape((shape[2], shape[1], shape[0]))
    runtime_sigma_t = max(
        float(np.float32(left) + np.float32(right)) for left, right in zip(sigma_a, sigma_s)
    )
    expected_majorants: list[float] = []
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                maximum = np.float32(0.0)
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            maximum = np.maximum(
                                maximum,
                                represented[
                                    min(max(z + dz, 0), shape[2] - 1),
                                    min(max(y + dy, 0), shape[1] - 1),
                                    min(max(x + dx, 0), shape[0] - 1),
                                ],
                            )
                authored = _trilinear_upper_f32(np.float32(maximum))
                physical = _outward_product_f32(authored, np.float32(density_scale))
                expected_majorants.append(
                    float(_outward_product_f32(physical, np.float32(runtime_sigma_t)))
                )
    majorants = medium["majorant_cells"]
    if not isinstance(majorants, list) or majorants != expected_majorants:
        _fail(
            "provenance_error",
            f"{label}: majorants are not the canonical N^3 extinction grid",
        )
    return {
        "grid_shape": shape,
        "density_f16_sha256": f16_sha256,
        "density_scale": density_scale,
        "density_sampling": DENSITY_SAMPLING,
        "majorant_grid_shape": shape,
        "majorant_query": MAJORANT_QUERY,
        "sigma_t_max": sigma_t_max,
    }


def _verify_majorant_domain(
    domain: Any,
    medium_path: Path,
    medium_sha256: str,
) -> None:
    if not isinstance(domain, dict):
        _fail("schema_error", "majorant.domain: expected object")
    expected = {
        "schema", "medium_sha256", "density_f16_sha256", "bounds_min", "bounds_max",
        "grid_shape", "majorant_grid_shape", "exact_domain", "domain_boundary_points",
        "domain_face_interiors",
        "texel_center_extrema", "majorant_cell_centers",
        "texel_centers_are_majorant_cell_centers", "majorant_boundary_sides",
        "majorant_query", "probe_mapping", "sha256",
    }
    _keys(domain, expected, "majorant.domain")
    medium = _object(medium_path)
    transported = _verify_transported_medium(medium)
    bounds_min = domain["bounds_min"]
    bounds_max = domain["bounds_max"]
    grid_shape = domain["grid_shape"]
    if (
        domain["schema"] != "forge3d.nephele.majorant_domain_coverage/3"
        or domain["medium_sha256"] != medium_sha256
        or domain["density_f16_sha256"] != transported["density_f16_sha256"]
        or not isinstance(bounds_min, list) or len(bounds_min) != 3
        or not isinstance(bounds_max, list) or len(bounds_max) != 3
        or not isinstance(grid_shape, list) or len(grid_shape) != 3
        or any(_finite(value, "majorant.domain.bounds_min") >= _finite(bounds_max[index], "majorant.domain.bounds_max") for index, value in enumerate(bounds_min))
        or any(_integer(value, "majorant.domain.grid_shape", minimum=2) < 2 for value in grid_shape)
        or any(medium["domain"].get(key) != value for key,value in (("bounds_min",bounds_min),("bounds_max",bounds_max),("grid_shape",grid_shape)))
        or domain["majorant_grid_shape"] != grid_shape
        or domain["exact_domain"] is not True
        or domain["texel_centers_are_majorant_cell_centers"] is not True
        or domain["majorant_query"] != MAJORANT_QUERY
        or domain["probe_mapping"] != MAJORANT_PROBE_MAPPING
    ):
        _fail("provenance_error", "majorant domain does not exactly cover the committed heterogeneous medium")
    node_count = math.prod(grid_shape)
    boundary_sides = 2 * sum(
        (grid_shape[axis] - 1)
        * math.prod(grid_shape[other] for other in range(3) if other != axis)
        for axis in range(3)
    )
    domain_face_interiors = 2 * sum(
        math.prod(grid_shape[other] for other in range(3) if other != axis)
        for axis in range(3)
    )
    if (
        _integer(domain["domain_boundary_points"], "majorant.domain.domain_boundary_points") != 8
        or _integer(domain["domain_face_interiors"], "majorant.domain.domain_face_interiors") != domain_face_interiors
        or _integer(domain["texel_center_extrema"], "majorant.domain.texel_center_extrema") != node_count
        or _integer(domain["majorant_cell_centers"], "majorant.domain.majorant_cell_centers") != node_count
        or _integer(domain["majorant_boundary_sides"], "majorant.domain.majorant_boundary_sides") != boundary_sides
        or domain["sha256"] != domain_coverage_sha256(domain)
    ):
        _fail("provenance_error", "majorant boundary/interpolation-extrema coverage is incomplete")


def _verify_identity(
    artifact_dir: Path,
    head_sha: str,
    repo_root: Path,
    imported_native_path: Path,
    observed_host: tuple[str, str, str],
) -> dict[str, Any]:
    context = _object(artifact_dir / "run-context.json")
    _keys(
        context,
        {
            "schema",
            "status",
            "head_sha",
            "checked_out_head",
            "tracked_worktree_clean",
            "required_backend",
            "command",
            "fixture_manifest_sha256",
            "gate4_policy_sha256",
            "fixture_commit",
            "runner_os",
            "runner_arch",
            "lane",
        },
        "run-context.json",
    )
    if (
        context["schema"] != "forge3d.nephele.run_context/1"
        or context["status"] != "captured"
        or context["head_sha"] != head_sha
        or context["checked_out_head"] != head_sha
        or context["tracked_worktree_clean"] is not True
        or context["runner_os"] != "Windows"
        or _windows_arch(context["runner_arch"]) != "x64"
        or context["lane"] != "windows-nvidia-vulkan"
        or not SHA_RE.fullmatch(str(context["fixture_commit"]))
        or not isinstance(context["command"], str)
        or not context["command"].strip()
    ):
        _fail("identity_error", "run-context.json: exact clean source identity is not proven")
    observed_os, observed_arch, observed_lane = observed_host
    if (
        observed_os != context["runner_os"]
        or _windows_arch(observed_arch) != _windows_arch(context["runner_arch"])
        or observed_lane != context["lane"]
    ):
        _fail("identity_error", "observed host OS, architecture, or lane differs from run context")
    if _git(repo_root, "rev-parse", "HEAD") != head_sha:
        _fail("identity_error", "repository HEAD does not match selected acceptance SHA")
    _git(repo_root, "cat-file", "-e", f"{head_sha}^{{commit}}")
    if _git(repo_root, "status", "--porcelain"):
        _fail("identity_error", "repository worktree is not actually clean")
    backend = str(context["required_backend"]).lower()
    if backend != "vulkan":
        _fail("identity_error", "run-context.json: windows-nvidia-vulkan lane requires literal Vulkan")

    runtime = _object(artifact_dir / "installed-wheel-runtime.json")
    _keys(
        runtime,
        {
            "schema", "source_revision", "package_version", "wheel_filename", "wheel_sha256",
            "wheel_native_member", "native_sha256", "installed_native_path",
        },
        "installed-wheel-runtime.json",
    )
    wheel_name = runtime["wheel_filename"]
    if not isinstance(wheel_name, str) or Path(wheel_name).name != wheel_name:
        _fail("schema_error", "installed-wheel-runtime.json: wheel_filename must be a basename")
    if (
        runtime["schema"] != "forge3d.nephele.installed_runtime/1"
        or runtime["source_revision"] != head_sha
        or not isinstance(runtime["package_version"], str)
        or not runtime["package_version"]
        or not SHA256_RE.fullmatch(str(runtime["wheel_sha256"]))
        or not SHA256_RE.fullmatch(str(runtime["native_sha256"]))
    ):
        _fail("identity_error", "installed-wheel-runtime.json: wheel/source identity is incomplete")
    if _sha256(artifact_dir / wheel_name) != runtime["wheel_sha256"]:
        _fail("identity_error", "installed wheel bytes do not match recorded SHA-256")
    wheel_path = artifact_dir / wheel_name
    member = runtime["wheel_native_member"]
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            names = [name for name in wheel.namelist() if re.fullmatch(r"forge3d/_forge3d[^/]*\.(?:pyd|so|dylib)", name)]
            if names != [member]:
                _fail("identity_error", "wheel native member is missing, ambiguous, or unrelated")
            member_bytes = wheel.read(member)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        _fail("identity_error", f"wheel is not a valid Forge3D wheel: {exc}")
    native_hash = hashlib.sha256(member_bytes).hexdigest()
    installed_path = Path(str(runtime["installed_native_path"]))
    if (
        not installed_path.is_absolute()
        or installed_path.resolve() != imported_native_path.resolve()
        or native_hash != runtime["native_sha256"]
        or _sha256(artifact_dir / "native-extension.bin") != native_hash
        or _sha256(installed_path) != native_hash
        or (artifact_dir / "native-extension.bin").read_bytes() != member_bytes
        or head_sha.encode("ascii") not in member_bytes
    ):
        _fail("identity_error", "wheel member, retained native bytes, and imported native extension differ")

    adapter = _object(artifact_dir / "adapter-probe.json")
    _keys(adapter, {"schema", "status", "requested_backend", "probe"}, "adapter-probe.json")
    probe = adapter.get("probe")
    if not isinstance(probe, dict):
        _fail("schema_error", "adapter-probe.json: probe must be an object")
    _keys(
        probe,
        {"status", "name", "vendor", "device", "backend", "device_type", "driver", "driver_info", "software_fallback"},
        "adapter-probe.json.probe",
    )
    name = str(probe["name"])
    if (
        adapter["schema"] != "forge3d.nephele.adapter_probe/1"
        or adapter["status"] != "passed"
        or str(adapter["requested_backend"]).lower() != "vulkan"
        or probe["status"] != "ok"
        or str(probe["backend"]).lower() != "vulkan"
        or str(probe["device_type"]).lower() != "discretegpu"
        or probe["software_fallback"] is not False
        or not name or "nvidia" not in name.lower()
        or _integer(probe["vendor"], "adapter.vendor", minimum=1) != 0x10DE
        or _integer(probe["device"], "adapter.device", minimum=1) <= 0
        or not isinstance(probe["driver"], str) or not probe["driver"].strip()
        or not isinstance(probe["driver_info"], str) or not probe["driver_info"].strip()
        or any(token in name.lower() for token in SOFTWARE_TOKENS)
    ):
        _fail("adapter_error", "adapter-probe.json: physical requested-backend execution is not proven")
    return {"context": context, "runtime": runtime, "adapter": probe}


def _verify_canonical_file(
    artifact_path: Path,
    canonical_path: Path,
    recorded_hash: Any,
    label: str,
    *,
    repo_root: Path,
    commit: str,
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(str(recorded_hash)):
        _fail("schema_error", f"run-context.json: {label} hash is invalid")
    artifact_hash = _sha256(artifact_path)
    canonical_hash = _sha256(canonical_path)
    tracked_hash = hashlib.sha256(_tracked_blob(repo_root, commit, canonical_path)).hexdigest()
    if artifact_hash != recorded_hash or canonical_hash != recorded_hash or tracked_hash != recorded_hash:
        _fail("provenance_error", f"{label}: artifact, tracked canonical file, and recorded hash differ")
    return _object(artifact_path)


def _fixture_record(
    record: Any,
    role: str,
    artifact_dir: Path,
    repo_root: Path,
    fixture_commit: str,
) -> None:
    if not isinstance(record, dict):
        _fail("schema_error", f"fixture {role}: expected object")
    _keys(record, {"path", "artifact", "sha256"}, f"fixture {role}")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        _fail("provenance_error", f"fixture {role}: path escapes the repository")
    path = repo_root / relative
    artifact = str(record["artifact"])
    if (
        Path(artifact).name != artifact
        or not SHA256_RE.fullmatch(str(record["sha256"]))
        or _sha256(path) != record["sha256"]
        or _sha256(artifact_dir / artifact) != record["sha256"]
        or hashlib.sha256(_tracked_blob(repo_root, fixture_commit, path)).hexdigest() != record["sha256"]
    ):
        _fail("provenance_error", f"fixture {role}: tracked, working, and artifact bytes differ")


def _verify_fixture_manifest(
    artifact_dir: Path,
    manifest: dict[str, Any],
    repo_root: Path,
    fixture_commit: str,
) -> str:
    if manifest.get("status") != "APPROVED":
        _fail("fixture_unresolved", "tracked NEPHELE fixture manifest is unresolved")
    _keys(
        manifest,
        {
            "schema", "status", "fixture_id", "revision", "source_revision", "color_pipeline",
            "scene_inputs", "files", "mask_generator", "mask_rules",
        },
        "fixture-manifest.json",
    )
    if (
        manifest["schema"] != "forge3d.nephele.fixture_manifest/3"
        or not isinstance(manifest["fixture_id"], str)
        or not manifest["fixture_id"]
        or _integer(manifest["revision"], "fixture.revision", minimum=1) < 1
        or not SHA_RE.fullmatch(str(manifest["source_revision"]))
        or not isinstance(manifest["color_pipeline"], dict)
        or set(manifest["color_pipeline"]) != {
            "linear_input", "tonemap", "output_encoding", "quantization",
            "exposure", "tonemap_contract", "crop",
        }
        or manifest["color_pipeline"].get("linear_input") != "linear-sRGB"
        or manifest["color_pipeline"].get("tonemap") != "aces-fitted"
        or manifest["color_pipeline"].get("output_encoding") != "IEC 61966-2-1 sRGB"
        or manifest["color_pipeline"].get("quantization") != "round-to-nearest uint8"
    ):
        _fail("schema_error", "fixture-manifest.json: fixture identity or color pipeline is incomplete")
    scene_inputs = manifest["scene_inputs"]
    files = manifest["files"]
    if not isinstance(scene_inputs, dict) or set(scene_inputs) != SCENE_INPUT_ROLES:
        _fail("schema_error", "fixture-manifest.json: scene input hashes are incomplete")
    if not isinstance(files, dict) or set(files) != set(FIXTURE_FILES):
        _fail("schema_error", "fixture-manifest.json: reference and mask files are incomplete")
    for role, record in {**scene_inputs, **files}.items():
        _fixture_record(record, role, artifact_dir, repo_root, fixture_commit)
    artifacts = [record["artifact"] for record in [*scene_inputs.values(), *files.values()]]
    if len(artifacts) != len(set(artifacts)):
        _fail("schema_error", "fixture artifact basenames must be unique")
    for role, filename in FIXTURE_FILES.items():
        if files[role]["artifact"] != filename:
            _fail("schema_error", f"fixture {role}: canonical artifact name changed")
    for pipeline_role, scene_role in (
        ("exposure", "exposure"),
        ("tonemap_contract", "tonemap"),
        ("crop", "crop"),
    ):
        if manifest["color_pipeline"][pipeline_role] != scene_inputs[scene_role]["sha256"]:
            _fail(
                "provenance_error",
                f"fixture color pipeline {pipeline_role} is not bound to scene input hash",
            )
    generator = manifest["mask_generator"]
    rules = manifest["mask_rules"]
    for role, record in (("mask_generator", generator), ("mask_rules", rules)):
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            _fail("schema_error", f"fixture {role}: invalid record")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            _fail("provenance_error", f"fixture {role}: path escapes the repository")
        path = repo_root / relative
        if (
            not SHA256_RE.fullmatch(str(record["sha256"]))
            or _sha256(path) != record["sha256"]
            or hashlib.sha256(_tracked_blob(repo_root, fixture_commit, path)).hexdigest() != record["sha256"]
        ):
            _fail("provenance_error", f"fixture {role}: tracked generator provenance mismatch")
    if Path(generator["path"]).as_posix() != "scripts/nephele_fixture_masks.py":
        _fail("provenance_error", "fixture mask generator is not the canonical tracked tool")
    if not Path(rules["path"]).as_posix().startswith("tests/nephele/fixture/"):
        _fail("provenance_error", "fixture mask rules are not under the canonical fixture path")
    if _sha256(Path(generate_masks.__code__.co_filename)) != generator["sha256"]:
        _fail("provenance_error", "fixture mask generator is not the tracked code being executed")
    with tempfile.TemporaryDirectory(prefix="nephele-mask-verify-") as temporary:
        generated = Path(temporary)
        try:
            generate_masks(artifact_dir, repo_root / rules["path"], generated)
        except (OSError, ValueError) as exc:
            _fail("provenance_error", f"fixture mask regeneration failed: {exc}")
        for role, filename in MASK_FILES.items():
            if _sha256(generated / filename) != files[role]["sha256"]:
                _fail("provenance_error", f"fixture {role}: deterministic regeneration differs")
    bundle = {
        "fixture_manifest_sha256": hashlib.sha256(
            (repo_root / "tests/nephele/fixture-manifest.json").read_bytes()
        ).hexdigest(),
        "scene_inputs": {role: scene_inputs[role]["sha256"] for role in sorted(scene_inputs)},
        "files": {role: files[role]["sha256"] for role in sorted(files)},
        "homogeneous_slab_sha256": _sha256(artifact_dir / "homogeneous-slab.json"),
    }
    slab_path = repo_root / "tests/nephele/homogeneous-slab.json"
    if (
        _sha256(slab_path) != bundle["homogeneous_slab_sha256"]
        or hashlib.sha256(_tracked_blob(repo_root, fixture_commit, slab_path)).hexdigest()
        != bundle["homogeneous_slab_sha256"]
    ):
        _fail("provenance_error", "input bundle homogeneous slab is not the tracked fixture-commit source")
    return hashlib.sha256(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bound_reference_file(
    artifact_dir: Path,
    repo_root: Path,
    fixture_commit: str,
    relative_name: Any,
    recorded_sha256: Any,
    label: str,
) -> tuple[Path, str]:
    relative = Path(str(relative_name))
    if relative.is_absolute() or ".." in relative.parts or not SHA256_RE.fullmatch(str(recorded_sha256)):
        _fail("provenance_error", f"{label}: invalid tracked path or SHA-256")
    canonical = repo_root / relative
    artifact_name = relative.name
    if (
        _sha256(canonical) != recorded_sha256
        or _sha256(artifact_dir / artifact_name) != recorded_sha256
        or hashlib.sha256(_tracked_blob(repo_root, fixture_commit, canonical)).hexdigest() != recorded_sha256
    ):
        _fail("provenance_error", f"{label}: tracked, working, artifact, and recorded bytes differ")
    return canonical, artifact_name


def _hybrid_kernel_source(repo_root: Path, source_revision: str) -> bytes:
    components = (
        ("src/shaders/sdf_primitives.wgsl", False),
        ("src/shaders/sdf_operations.wgsl", True),
        ("src/shaders/hybrid_traversal.wgsl", True),
        ("src/shaders/hybrid_terrain_traversal.wgsl", True),
        ("src/shaders/atmosphere/prometheus_spectral_reference.wgsl", False),
        ("src/shaders/hybrid_kernel.wgsl", True),
    )
    assembled: list[str] = []
    for relative, strip in components:
        try:
            source = _tracked_blob(repo_root, source_revision, repo_root / relative).decode("utf-8")
        except UnicodeDecodeError as exc:
            _fail("provenance_error", f"assembled hybrid source is not UTF-8: {exc}")
        if strip:
            source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith("#include")
            )
        assembled.append(source)
    return "\n".join(assembled).encode()


def _reference_module_sha256(repo_root: Path, source_revision: str) -> str:
    query = _tracked_blob(
        repo_root,
        source_revision,
        repo_root / "src/shaders/nephele_terrain_trace_adapter.wgsl",
    )
    return hashlib.sha256(_hybrid_kernel_source(repo_root, source_revision) + b"\n" + query).hexdigest()


def _camera_contract(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("schema_error", f"{label}: camera_contract must be an object")
    vector_keys = {"origin", "look_at", "up", "right", "forward"}
    _keys(value, vector_keys | {"fov_y"}, f"{label}.camera_contract")
    result: dict[str, Any] = {}
    for key in vector_keys:
        raw = value[key]
        if not isinstance(raw, list) or len(raw) != 3:
            _fail("schema_error", f"{label}.camera_contract.{key}: expected three finite f32 values")
        vector = np.asarray([_finite(component, f"{label}.camera_contract.{key}") for component in raw], dtype=np.float32)
        if not np.isfinite(vector).all():
            _fail("schema_error", f"{label}.camera_contract.{key}: expected three finite f32 values")
        result[key] = vector.tolist()
    result["fov_y"] = float(np.float32(_finite(value["fov_y"], f"{label}.camera_contract.fov_y")))
    return result


def _native_runtime(
    value: Any,
    source_revision: str,
    artifact_dir: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict):
        _fail("schema_error", f"{label}: native_runtime must be an object")
    _keys(
        value,
        {"source_revision", "wheel_filename", "wheel_sha256", "wheel_native_member", "native_sha256"},
        f"{label}.native_runtime",
    )
    wheel_filename = value["wheel_filename"]
    member = value["wheel_native_member"]
    if (
        value["source_revision"] != source_revision
        or not isinstance(wheel_filename, str)
        or Path(wheel_filename).name != wheel_filename
        or not wheel_filename.endswith(".whl")
        or not isinstance(member, str)
        or not re.fullmatch(r"forge3d/_forge3d[^/]*\.(?:pyd|so|dylib)", member)
        or not SHA256_RE.fullmatch(str(value["wheel_sha256"]))
        or not SHA256_RE.fullmatch(str(value["native_sha256"]))
    ):
        _fail("provenance_error", f"{label}: reference wheel/native runtime identity is incomplete")
    wheel_path = artifact_dir / wheel_filename
    if _sha256(wheel_path) != value["wheel_sha256"]:
        _fail("provenance_error", f"{label}: retained reference wheel bytes differ from provenance")
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            members = [
                name for name in wheel.namelist()
                if re.fullmatch(r"forge3d/_forge3d[^/]*\.(?:pyd|so|dylib)", name)
            ]
            if members != [member]:
                _fail("provenance_error", f"{label}: retained reference wheel native member is missing or ambiguous")
            native_bytes = wheel.read(member)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        _fail("provenance_error", f"{label}: retained reference wheel is invalid: {exc}")
    if (
        hashlib.sha256(native_bytes).hexdigest() != value["native_sha256"]
        or source_revision.encode("ascii") not in native_bytes
    ):
        _fail("provenance_error", f"{label}: retained reference native bytes differ from provenance or source")
    return {key: str(value[key]) for key in value}


def _spatial_tiles(value: Any, crop: list[int], label: str) -> list[dict[str, int]]:
    if not isinstance(value, list) or not value:
        _fail("schema_error", f"{label}: spatial_tiles must be nonempty")
    x, y, width, height = map(int, crop)
    expected = [
        {"x": x, "y": y + row, "width": width, "height": 1}
        for row in range(height)
    ]
    if value != expected:
        _fail("provenance_error", f"{label}: spatial tiles do not give exact canonical full-domain coverage")
    return expected


def _verify_reference_provenance(
    artifact_dir: Path,
    manifest: dict[str, Any],
    repo_root: Path,
    fixture_commit: str,
) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    convergence_path = repo_root / "tests/nephele/fixture/reference-convergence.json"
    convergence = _verify_canonical_file(
        artifact_dir / convergence_path.name,
        convergence_path,
        _sha256(convergence_path),
        "reference convergence",
        repo_root=repo_root,
        commit=fixture_commit,
    )
    _keys(convergence, {
        "schema", "status", "criterion", "previous", "final", "nested_prefix",
        "spatial_tiles_identity", "mask_identity", "classification_identity",
        "metrics", "next_samples_per_pixel_if_unresolved", "generator", "generator_sha256",
    }, "reference-convergence.json")
    if (
        convergence["schema"] != "forge3d.nephele.reference_convergence/2"
        or convergence["status"] != "CONVERGED"
        or convergence["nested_prefix"] is not True
        or convergence["spatial_tiles_identity"] is not True
        or convergence["next_samples_per_pixel_if_unresolved"] is not None
        or convergence["criterion"] != "actual downstream Gate 3 and approved Gate 4 metrics over exact identical reference-derived masks"
    ):
        _fail("fixture_unresolved", "reference convergence is unresolved or uses an arbitrary criterion")
    generator_path, generator_name = _bound_reference_file(
        artifact_dir, repo_root, fixture_commit, convergence["generator"],
        convergence["generator_sha256"], "reference convergence generator",
    )
    if generator_path.relative_to(repo_root).as_posix() != "scripts/record_media_reference_convergence.py":
        _fail("provenance_error", "reference convergence generator is not canonical")

    final = convergence["final"]
    previous = convergence["previous"]
    if not isinstance(final, dict) or not isinstance(previous, dict):
        _fail("schema_error", "reference convergence endpoints must be objects")
    _keys(final, {"samples_per_pixel", "rgb_path", "rgb_sha256", "provenance_path", "provenance_sha256"}, "reference convergence final")
    _keys(previous, {"samples_per_pixel", "provenance_path", "provenance_sha256", "artifacts"}, "reference convergence previous")
    final_spp = _integer(final["samples_per_pixel"], "reference final samples_per_pixel", minimum=1)
    prior_spp = _integer(previous["samples_per_pixel"], "reference prior samples_per_pixel", minimum=1)
    if prior_spp * 2 != final_spp:
        _fail("provenance_error", "reference convergence is not one exact sample-count doubling")
    final_rgb, final_rgb_name = _bound_reference_file(
        artifact_dir, repo_root, fixture_commit, final["rgb_path"], final["rgb_sha256"], "final reference RGB",
    )
    if final_rgb.relative_to(repo_root).as_posix() != manifest["files"]["reference_rgb"]["path"]:
        _fail("provenance_error", "reference convergence final RGB is not the manifest reference")
    final_provenance_path, final_provenance_name = _bound_reference_file(
        artifact_dir, repo_root, fixture_commit, final["provenance_path"], final["provenance_sha256"], "final reference provenance",
    )
    prior_provenance_path, prior_provenance_name = _bound_reference_file(
        artifact_dir, repo_root, fixture_commit, previous["provenance_path"], previous["provenance_sha256"], "prior reference provenance",
    )

    def provenance(path: Path, expected_spp: int, label: str) -> dict[str, Any]:
        value = _object(path)
        _keys(value, {
            "schema", "algorithm", "samples_per_pixel", "seed", "sample_identity",
            "spatial_tiles", "runtime_partition", "source_revision", "source_inputs", "scene_inputs",
            "assembled_sources", "full_viewport", "crop", "camera_contract", "native_runtime", "diagnostics",
            "acceptance_eligible", "diagnostic_reason",
        }, label)
        if (
            value["schema"] != "forge3d.nephele.reference_provenance/2"
            or value["algorithm"] != "integrated-hybrid-terrain-ratio-delta-tracking-reference"
            or value["samples_per_pixel"] != expected_spp
            or value["seed"] != 0x4E455048
            or not SHA_RE.fullmatch(str(value["source_revision"]))
            or value["source_revision"] != manifest["source_revision"]
            or value["acceptance_eligible"] is not True
            or value["diagnostic_reason"] is not None
        ):
            _fail("provenance_error", f"{label}: algorithm, sample count, seed, revision, or clean-wheel eligibility differs")
        crop = _object(artifact_dir / manifest["scene_inputs"]["crop"]["artifact"])
        expected_crop = [crop.get(key) for key in ("x", "y", "width", "height")]
        if value["full_viewport"] != crop.get("full_viewport") or value["crop"] != expected_crop:
            _fail("provenance_error", f"{label}: full viewport or crop differs from the manifest input")
        sample_identity = value["sample_identity"]
        if not isinstance(sample_identity, dict):
            _fail("schema_error", f"{label}: sample_identity must be an object")
        _keys(sample_identity, {"algorithm", "seed", "range"}, f"{label}.sample_identity")
        if sample_identity != {
            "algorithm": "per-pixel-absolute-sample-index-v1",
            "seed": 0x4E455048,
            "range": [0, expected_spp],
        }:
            _fail("provenance_error", f"{label}: canonical per-pixel sample identity differs")
        _spatial_tiles(value["spatial_tiles"], value["crop"], label)
        runtime_partition = value["runtime_partition"]
        if not isinstance(runtime_partition, dict):
            _fail("schema_error", f"{label}: runtime_partition must be an object")
        _keys(runtime_partition, {"kind", "value"}, f"{label}.runtime_partition")
        if (
            runtime_partition["kind"] != "process_pool_max_workers"
            or _integer(runtime_partition["value"], f"{label}.runtime_partition.value", minimum=1)
            > len(value["spatial_tiles"])
        ):
            _fail("provenance_error", f"{label}: runtime spatial partition diagnostic differs")
        expected_scene_inputs = {
            role: manifest["scene_inputs"][role]["sha256"] for role in sorted(manifest["scene_inputs"])
        }
        if value["scene_inputs"] != expected_scene_inputs:
            _fail("provenance_error", f"{label}: scene-input hashes differ from the fixture manifest")
        value["camera_contract"] = _camera_contract(value["camera_contract"], label)
        value["native_runtime"] = _native_runtime(
            value["native_runtime"], value["source_revision"], artifact_dir, label
        )
        inputs = value["source_inputs"]
        if not isinstance(inputs, dict) or not inputs:
            _fail("schema_error", f"{label}: source_inputs must be nonempty")
        required = {
            "scripts/generate_media_fixture.py", "python/forge3d/media.py", "src/media_py.rs",
            "src/path_tracing/hybrid_compute/media_reference.rs",
            "src/path_tracing/hybrid_compute/terrain_heightfield.rs",
            "src/path_tracing/hybrid_compute/render_terrain.rs",
            "src/terrain/camera.rs", "src/terrain/mod.rs",
            "src/shaders/hybrid_terrain_traversal.wgsl",
            "src/shaders/nephele_terrain_trace_adapter.wgsl", "src/shader_sources.rs",
        }
        try:
            media_sources = set(subprocess.run(
                ["git", "-C", str(repo_root), "ls-tree", "-r", "--name-only", value["source_revision"], "--", "src/media"],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines())
        except subprocess.CalledProcessError as exc:
            _fail("provenance_error", f"{label}: cannot enumerate clean-revision media sources: {exc.stderr.strip()}")
        if set(inputs) != required | media_sources:
            _fail("provenance_error", f"{label}: required source dependencies are incomplete")
        for relative_name, digest in inputs.items():
            relative = Path(str(relative_name))
            if relative.is_absolute() or ".." in relative.parts or not SHA256_RE.fullmatch(str(digest)):
                _fail("provenance_error", f"{label}: invalid source dependency record")
            if hashlib.sha256(_tracked_blob(repo_root, value["source_revision"], repo_root / relative)).hexdigest() != digest:
                _fail("provenance_error", f"{label}: source dependency differs from its clean revision: {relative_name}")
        assembled = value["assembled_sources"]
        if assembled != {"terrain_media_reference_module_sha256": _reference_module_sha256(repo_root, value["source_revision"])}:
            _fail("provenance_error", f"{label}: assembled terrain-media reference module provenance differs")
        diagnostics = value["diagnostics"]
        if not isinstance(diagnostics, dict):
            _fail("schema_error", f"{label}: diagnostics must be an object")
        diagnostic_keys = {
            "majorant_proof", "majorant_valid", "sample_count", "step_count",
            "temporal_history_decision", "temporal_history_reason", "host_visible_bytes",
            "froxel_device_local_bytes", "density_device_local_bytes",
            "majorant_device_local_bytes", "staging_readback_bytes", "adapter", "backend",
            "driver", "source_revision", "executed_multi_scatter",
            "single_scatter_luminance", "multiple_scatter_luminance",
            "energy_accounting_residual",
        }
        _keys(diagnostics, diagnostic_keys, f"{label}.diagnostics")
        backend = str(diagnostics["backend"]).strip().lower()
        driver = diagnostics["driver"]
        expected_sample_count = expected_spp * int(expected_crop[2]) * int(expected_crop[3])
        expected_host_visible_bytes = int(expected_crop[2]) * int(expected_crop[3]) * (5 * 3 * 4 + 4 + 2)
        if (
            diagnostics["source_revision"] != value["source_revision"]
            or not isinstance(diagnostics["adapter"], str)
            or not diagnostics["adapter"].strip()
            or not backend
            or not isinstance(driver, str)
            or (not driver.strip() and backend != "metal")
            or diagnostics["majorant_valid"] is not True
            or diagnostics["majorant_proof"] != "TrilinearConvexHull"
            or _integer(diagnostics["sample_count"], f"{label}.diagnostics.sample_count", minimum=1) != expected_sample_count
            or _integer(diagnostics["step_count"], f"{label}.diagnostics.step_count", minimum=1) <= 0
            or diagnostics["executed_multi_scatter"] is not True
            or diagnostics["temporal_history_decision"] != "not_applicable"
            or diagnostics["temporal_history_reason"] != "independent reference samples do not reuse temporal history"
            or diagnostics["single_scatter_luminance"] is not None
            or diagnostics["multiple_scatter_luminance"] is not None
            or diagnostics["energy_accounting_residual"] is not None
            or _integer(diagnostics["host_visible_bytes"], f"{label}.diagnostics.host_visible_bytes", minimum=1)
            != expected_host_visible_bytes
        ):
            _fail("provenance_error", f"{label}: native reference diagnostics are incomplete")
        for key in (
            "froxel_device_local_bytes", "density_device_local_bytes",
            "majorant_device_local_bytes", "staging_readback_bytes",
        ):
            if _integer(diagnostics[key], f"{label}.diagnostics.{key}") != 0:
                _fail("provenance_error", f"{label}: reference-only device allocation diagnostics must be zero")
        return value

    current_provenance = provenance(final_provenance_path, final_spp, "final reference provenance")
    prior_provenance = provenance(prior_provenance_path, prior_spp, "prior reference provenance")
    if (
        prior_provenance["sample_identity"]["algorithm"] != current_provenance["sample_identity"]["algorithm"]
        or prior_provenance["sample_identity"]["seed"] != current_provenance["sample_identity"]["seed"]
        or prior_provenance["sample_identity"]["range"] != [0, prior_spp]
        or current_provenance["sample_identity"]["range"] != [0, final_spp]
        or prior_provenance["spatial_tiles"] != current_provenance["spatial_tiles"]
        or prior_provenance["full_viewport"] != current_provenance["full_viewport"]
        or prior_provenance["crop"] != current_provenance["crop"]
    ):
        _fail("provenance_error", "reference samples are not an exact canonical per-pixel nested prefix")
    if prior_provenance["camera_contract"] != current_provenance["camera_contract"]:
        _fail("provenance_error", "reference prefixes use different camera contracts")
    if prior_provenance["native_runtime"] != current_provenance["native_runtime"]:
        _fail("provenance_error", "reference prefixes use different native wheel/runtime identities")
    consistent_diagnostics = (
        "adapter", "backend", "driver", "source_revision", "majorant_proof", "majorant_valid",
        "executed_multi_scatter", "host_visible_bytes", "froxel_device_local_bytes",
        "density_device_local_bytes", "majorant_device_local_bytes", "staging_readback_bytes",
        "temporal_history_decision", "temporal_history_reason",
        "single_scatter_luminance", "multiple_scatter_luminance",
        "energy_accounting_residual",
    )
    if any(
        prior_provenance["diagnostics"][key] != current_provenance["diagnostics"][key]
        for key in consistent_diagnostics
    ):
        _fail("provenance_error", "reference prefixes use inconsistent native diagnostics identities")

    artifacts = previous["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(REFERENCE_PREFIX_ARTIFACTS):
        _fail("schema_error", "prior reference prefix artifacts are incomplete")
    prior_paths: dict[str, Path] = {}
    artifact_names = {
        generator_name, final_rgb_name, final_provenance_name, prior_provenance_name,
        convergence_path.name, current_provenance["native_runtime"]["wheel_filename"],
    }
    for canonical_name, record in artifacts.items():
        if not isinstance(record, dict):
            _fail("schema_error", f"prior {canonical_name}: expected record")
        _keys(record, {"path", "sha256"}, f"prior {canonical_name}")
        path, name = _bound_reference_file(
            artifact_dir, repo_root, fixture_commit, record["path"], record["sha256"], f"prior {canonical_name}",
        )
        expected_name = f"{Path(canonical_name).stem}-spp{prior_spp}{Path(canonical_name).suffix}"
        if name != expected_name:
            _fail("provenance_error", f"prior {canonical_name}: canonical prefix name differs")
        prior_paths[canonical_name] = path
        artifact_names.add(name)

    classification_names = (
        "reference-terrain-hit.npy",
        "reference-media-lighting-visibility.npy",
        "reference-terrain-slice.npy",
    )
    classification_identity = {
        name: np.array_equal(_array(prior_paths[name]), _array(artifact_dir / name))
        for name in classification_names
    }
    if convergence["classification_identity"] != classification_identity or not all(classification_identity.values()):
        _fail("gate_failure", "reference prefix does not preserve deterministic terrain classifications")

    with tempfile.TemporaryDirectory(prefix="nephele-reference-prefix-") as temporary:
        temporary_path = Path(temporary)
        old_fixture = temporary_path / "old-fixture"
        old_masks = temporary_path / "old-masks"
        old_fixture.mkdir()
        for name in REFERENCE_PREFIX_ARTIFACTS[1:]:
            (old_fixture / name).write_bytes(prior_paths[name].read_bytes())
        try:
            generate_masks(old_fixture, repo_root / manifest["mask_rules"]["path"], old_masks)
        except (OSError, ValueError) as exc:
            _fail("provenance_error", f"prior reference mask regeneration failed: {exc}")
        regenerated_identity = {
            role: np.array_equal(_array(old_masks / filename), _array(artifact_dir / filename))
            for role, filename in MASK_FILES.items()
        }
    if convergence["mask_identity"] != regenerated_identity or not all(regenerated_identity.values()):
        _fail("gate_failure", "reference prefix does not preserve every downstream mask identity")
    old_rgb = _rgb(prior_paths["reference-rgb.npy"])
    new_rgb = _rgb(final_rgb)
    if old_rgb.shape != new_rgb.shape:
        _fail("schema_error", "reference prefix RGB shapes differ")
    masks = {role: _mask(artifact_dir / filename, new_rgb.shape[:2]) for role, filename in MASK_FILES.items()}
    delta_e = delta_e_2000(srgb_to_lab(old_rgb), srgb_to_lab(new_rgb))
    ys, xs = np.nonzero(masks["godray_roi_mask"])
    old_roi = old_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    new_roi = new_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    shadow_values = delta_e[masks["terrain_mask"] & masks["cloud_shadow_mask"]]
    if not shadow_values.size:
        _fail("schema_error", "reference convergence cloud-shadow terrain population is empty")
    metrics = {
        "gate3_sky_cloud_delta_e_below_2_5_fraction": float(np.mean(delta_e[masks["sky_cloud_mask"]] < 2.5)),
        "gate3_godray_roi_ssim": ssim(old_roi, new_roi, data_range=255.0),
        "gate4_cloud_shadow_terrain_maximum_delta_e": float(shadow_values.max()),
    }
    if not isinstance(convergence["metrics"], dict) or set(convergence["metrics"]) != set(metrics) or any(
        not math.isclose(_finite(convergence["metrics"][name], f"reference convergence metric {name}"), value, rel_tol=1e-12, abs_tol=1e-12)
        for name, value in metrics.items()
    ):
        _fail("provenance_error", "reference convergence downstream metrics differ from recomputation")
    if not (metrics["gate3_sky_cloud_delta_e_below_2_5_fraction"] >= 0.95 and metrics["gate3_godray_roi_ssim"] > 0.95 and metrics["gate4_cloud_shadow_terrain_maximum_delta_e"] < 2.0):
        _fail("fixture_unresolved", "reference convergence does not satisfy downstream Gate 3/4 criteria")
    return (
        {
            "previous_samples_per_pixel": prior_spp,
            "final_samples_per_pixel": final_spp,
            "metrics": metrics,
            "native_runtime": current_provenance["native_runtime"],
        },
        artifact_names,
        current_provenance["camera_contract"],
    )


def _policy_evaluator(policy: dict[str, Any]) -> Callable[[np.ndarray], tuple[float, bool]]:
    if policy.get("schema") != "forge3d.nephele.gate4_policy/1" or policy.get("status") != "APPROVED":
        _fail("policy_unresolved", "Gate 4 aggregation has no owner-approved policy")
    _keys(policy, {"schema", "status", "policy_id", "approved_by", "approved_revision", "aggregation", "definition"}, "gate4-policy.json")
    if not all(isinstance(policy[key], str) and policy[key].strip() for key in ("policy_id", "approved_by", "definition")):
        _fail("schema_error", "gate4-policy.json: approval identity is incomplete")
    if not SHA_RE.fullmatch(str(policy["approved_revision"])):
        _fail("schema_error", "gate4-policy.json: approved_revision must be an exact Git SHA")
    aggregation = policy["aggregation"]
    if aggregation != {"kind": "maximum"}:
        _fail("policy_unresolved", "Gate 4 requires the exact owner-approved maximum aggregation")
    return lambda values: (float(values.max()), bool(values.max() < 2.0))


def _gate1(
    artifact_dir: Path,
    head_sha: str,
    repo_root: Path,
    medium_record: dict[str, Any],
) -> dict[str, Any]:
    raw = _object(artifact_dir / "gate1-statistics.json")
    _keys(raw, {"schema", "producer", "homogeneous_slab", "homogeneous", "heterogeneous", "majorant", "russian_roulette"}, "gate1-statistics.json")
    if raw["schema"] != "forge3d.nephele.gate1_raw/1":
        _fail("schema_error", "gate1-statistics.json: unknown schema")
    _verify_physical_producer(raw["producer"], repo_root, head_sha)
    slab = _verify_homogeneous_slab(raw["homogeneous_slab"], artifact_dir, repo_root, head_sha)
    homogeneous = raw["homogeneous"]
    if not isinstance(homogeneous, dict):
        _fail("schema_error", "gate1-statistics.json.homogeneous: expected object")
    _keys(homogeneous, {"samples", "sigma_t", "density", "distance", "sample_mapping", "raw_output"}, "gate1-statistics.json.homogeneous")
    homogeneous_values = _raw_f64(
        artifact_dir / "homogeneous-ratio-samples.bin",
        homogeneous["raw_output"],
        "homogeneous ratio tracking",
        1_000_000,
    )
    if homogeneous_values.size != 1_000_000 or np.any((homogeneous_values < 0.0) | (homogeneous_values > 1.0)):
        _fail("schema_error", "homogeneous ratio-tracking raw support or count differs")
    recomputed_homogeneous = _accumulator_from_array(homogeneous_values)
    if homogeneous["samples"] != recomputed_homogeneous:
        _fail("schema_error", "homogeneous accumulator differs from implementation raw samples")
    stats = _stats(homogeneous["samples"], "homogeneous.samples", exact_count=1_000_000, support=(0.0, 1.0))
    _identity_hash(homogeneous["sample_mapping"], "homogeneous.sample_mapping")
    expected_homogeneous_mapping = {
        "algorithm": "canonical-sample-identity-v1",
        "parameters": {"frame_u64": 0x4E4550481001, "fields": ["frame", "pixel", "sample", "bounce", "dimension"]},
    }
    expected_homogeneous_mapping["sha256"] = hashlib.sha256(json.dumps(expected_homogeneous_mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if homogeneous["sample_mapping"] != expected_homogeneous_mapping:
        _fail("provenance_error", "homogeneous ratio tracking does not use the canonical sample identity")
    sigma_t = _finite(homogeneous["sigma_t"], "homogeneous.sigma_t")
    density = _finite(homogeneous["density"], "homogeneous.density")
    distance = _finite(homogeneous["distance"], "homogeneous.distance")
    if (sigma_t, density, distance) != (
        float(slab["sigma_t_max_channel"]), float(slab["density"]), float(slab["distance"])
    ):
        _fail("provenance_error", "Gate 1 homogeneous parameters differ from the tracked slab")
    if min(sigma_t, density, distance) < 0.0 or not 0.0 <= stats.mean <= 1.0:
        _fail("schema_error", "Gate 1 homogeneous transmittance inputs are outside the physical domain")
    analytic = math.exp(-sigma_t * density * distance)
    if not (stats.mean - 3 * stats.standard_error <= analytic <= stats.mean + 3 * stats.standard_error):
        _fail("gate_failure", "Gate 1 homogeneous analytic value is outside mean +/- 3 standard errors")

    heterogeneous = raw["heterogeneous"]
    if not isinstance(heterogeneous, dict):
        _fail("schema_error", "gate1-statistics.json.heterogeneous: expected object")
    _keys(heterogeneous, {"samples", "sample_mapping", "transport_representation", "raw_output", "comparator_sha256"}, "gate1-statistics.json.heterogeneous")
    heterogeneous_values = _raw_f64(
        artifact_dir / "heterogeneous-ratio-samples.bin",
        heterogeneous["raw_output"],
        "heterogeneous ratio tracking",
        1_000_000,
    )
    if heterogeneous_values.size != 1_000_000 or np.any((heterogeneous_values < 0.0) | (heterogeneous_values > 1.0)):
        _fail("schema_error", "heterogeneous ratio-tracking raw support or count differs")
    if heterogeneous["samples"] != _accumulator_from_array(heterogeneous_values):
        _fail("schema_error", "heterogeneous accumulator differs from implementation raw samples")
    heterogeneous_stats = _stats(heterogeneous["samples"], "heterogeneous.samples", exact_count=1_000_000, support=(0.0, 1.0))
    _identity_hash(heterogeneous["sample_mapping"], "heterogeneous.sample_mapping")
    expected_heterogeneous_mapping = {
        "algorithm": "canonical-grid3d-z-ray-v1",
        "parameters": {"coordinate_frame_u64": 0x4E4550481101, "tracking_frame_u64": 0x4E4550481102, "fields": ["sample", "dimension"]},
    }
    expected_heterogeneous_mapping["sha256"] = hashlib.sha256(json.dumps(expected_heterogeneous_mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if heterogeneous["sample_mapping"] != expected_heterogeneous_mapping:
        _fail("provenance_error", "heterogeneous ratio tracking does not use the canonical sample identity")
    if not 0.0 <= heterogeneous_stats.mean <= 1.0:
        _fail("schema_error", "Gate 1 heterogeneous estimator mean is outside [0,1]")
    comparator_path = artifact_dir / "heterogeneous-comparator.json"
    if _sha256(comparator_path) != heterogeneous["comparator_sha256"]:
        _fail("provenance_error", "heterogeneous comparator hash mismatch")
    comparator = _object(comparator_path)
    comparator_keys = {"schema", "algorithm", "transport_representation", "extinction_channel", "sigma_t", "seed_identity", "sample_mapping", "source_revision", "command", "generated_at_utc", "producer_tool", "medium", "cache_key", "samples", "statistics", "raw_output", "cache_provenance"}
    _keys(comparator, comparator_keys, "heterogeneous-comparator.json")
    comparator_tool_hash = _verify_tool(
        comparator["producer_tool"],
        "scripts/nephele_heterogeneous_comparator.py",
        comparator_cache_key,
        repo_root,
        head_sha,
        "heterogeneous comparator",
    )
    if not isinstance(comparator["medium"], dict):
        _fail("schema_error", "heterogeneous-comparator.json.medium: expected object")
    _keys(comparator["medium"], {"path", "sha256"}, "heterogeneous-comparator.json.medium")
    if (
        comparator["schema"] != "forge3d.nephele.heterogeneous_comparator/4"
        or comparator["algorithm"] != "independent-f16-texel-center-column-bernoulli-v3"
        or comparator["source_revision"] != head_sha
        or not all(isinstance(comparator[key], str) and comparator[key].strip() for key in ("algorithm", "command", "generated_at_utc"))
        or comparator["medium"] != {"path": medium_record["path"], "sha256": medium_record["sha256"]}
        or comparator["producer_tool"]["sha256"] != comparator_tool_hash
        or _identity_hash(comparator["seed_identity"], "comparator.seed_identity") != comparator["seed_identity"]["sha256"]
        or _identity_hash(comparator["sample_mapping"], "comparator.sample_mapping") != comparator["sample_mapping"]["sha256"]
        or comparator["cache_key"] != _derived_cache_key(comparator)
    ):
        _fail("provenance_error", "heterogeneous-comparator.json: cached comparator provenance is incomplete")
    expected_seed = {"algorithm":"splitmix64","parameters":{"seed_u64":COMPARATOR_SEED}}
    expected_seed["sha256"] = hashlib.sha256(json.dumps(expected_seed,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    expected_mapping = {"algorithm":"indexed-ray-bernoulli-v1","parameters":{"dimensions":["u","v","survival"],"bit_order":"lsb0"}}
    expected_mapping["sha256"] = hashlib.sha256(json.dumps(expected_mapping,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    committed_medium = _object(artifact_dir / medium_record["artifact"])
    transported_medium = _verify_transported_medium(committed_medium)
    if heterogeneous["transport_representation"] != transported_medium:
        _fail(
            "provenance_error",
            "implementation heterogeneous samples are not bound to the exact transported field",
        )
    sigma_a = committed_medium.get("sigma_a")
    sigma_s = committed_medium.get("sigma_s")
    if not isinstance(sigma_a, list) or not isinstance(sigma_s, list) or len(sigma_a) != 3 or len(sigma_s) != 3:
        _fail("schema_error", "heterogeneous medium coefficient spectra are incomplete")
    extinction = [_finite(a, "medium.sigma_a") + _finite(s, "medium.sigma_s") for a, s in zip(sigma_a, sigma_s)]
    expected_channel = max(range(3), key=extinction.__getitem__)
    if comparator["extinction_channel"] != expected_channel or not math.isclose(
        _finite(comparator["sigma_t"], "comparator.sigma_t"), extinction[expected_channel], rel_tol=0.0, abs_tol=0.0
    ):
        _fail("provenance_error", "heterogeneous comparator is not bound to canonical extinction coefficients")
    expected_comparator_representation = {
        "density_f16_sha256": transported_medium["density_f16_sha256"],
        "grid_shape": transported_medium["grid_shape"],
        "density_scale": transported_medium["density_scale"],
        "sampling": DENSITY_SAMPLING,
        "slab_integration": "exact-piecewise-linear-texel-center-column-mean",
    }
    if comparator["transport_representation"] != expected_comparator_representation:
        _fail(
            "provenance_error",
            "heterogeneous comparator is not bound to the exact transported f16 field",
        )
    raw_output, cache = comparator["raw_output"], comparator["cache_provenance"]
    if not isinstance(raw_output,dict) or set(raw_output)!={"path","sha256","encoding","samples"} or raw_output["path"]!="heterogeneous-comparator.samples.bin" or raw_output["encoding"]!="bit-packed-lsb0-bernoulli" or raw_output["samples"]!=COMPARATOR_SAMPLES or cache!={"status":"freshly_executed","cache_key_scope":"inputs-only"} or comparator["seed_identity"]!=expected_seed or comparator["sample_mapping"]!=expected_mapping:
        _fail("provenance_error", "heterogeneous comparator execution contract is not exact")
    try: raw_bits=(artifact_dir/raw_output["path"]).read_bytes()
    except OSError as exc: _fail("artifact_error",f"heterogeneous comparator raw samples are missing: {exc}")
    if len(raw_bits)!=(COMPARATOR_SAMPLES+7)//8 or hashlib.sha256(raw_bits).hexdigest()!=raw_output["sha256"]:
        _fail("provenance_error", "heterogeneous comparator raw sample artifact is missing or corrupt")
    survivors=sum(byte.bit_count() for byte in raw_bits)
    if comparator["samples"]!={"count":COMPARATOR_SAMPLES,"sum":survivors,"sum_squares":survivors}:
        _fail("schema_error", "heterogeneous comparator accumulator differs from raw Bernoulli samples")
    comparator_stats = _stats(comparator["samples"], "heterogeneous_comparator.samples", exact_count=100_000_000, support=(0.0, 1.0))
    reported = comparator["statistics"]
    if not isinstance(reported, dict) or set(reported) != {"mean", "sample_variance", "standard_error"} or any(not math.isclose(_finite(reported[name], f"comparator.statistics.{name}"), getattr(comparator_stats, {"mean":"mean","sample_variance":"variance","standard_error":"standard_error"}[name]), rel_tol=1e-12, abs_tol=1e-15) for name in reported):
        _fail("schema_error", "heterogeneous comparator reported statistics differ from raw samples")
    if not 0.0 <= comparator_stats.mean <= 1.0:
        _fail("schema_error", "Gate 1 heterogeneous comparator mean is outside [0,1]")
    if comparator_stats.mean == 0.0:
        _fail("policy_unresolved", "Gate 1 heterogeneous comparator is zero; owner comparison policy required")
    relative_error = abs(heterogeneous_stats.mean - comparator_stats.mean) / abs(comparator_stats.mean)
    if relative_error >= 0.005:
        _fail("gate_failure", "Gate 1 heterogeneous relative error is not below 0.5 percent")

    majorant = raw["majorant"]
    if not isinstance(majorant, dict):
        _fail("schema_error", "gate1-statistics.json.majorant: expected object")
    _keys(majorant,{"evidence_sha256"},"gate1-statistics.json.majorant")
    probe_path=artifact_dir/"majorant-domain-probe.json"
    if _sha256(probe_path)!=majorant["evidence_sha256"]: _fail("provenance_error","majorant probe artifact hash mismatch")
    probe=_object(probe_path)
    _keys(probe,{"schema","probe_count","violation_count","max_represented_extinction_minus_bound","measured_representation_error","source_revision","producer_tool","medium","transport_representation","sample_mapping","domain","raw_output","summary"},"majorant-domain-probe.json")
    _verify_tool(
        probe["producer_tool"],
        "scripts/nephele_majorant_domain_probe.py",
        domain_coverage_sha256,
        repo_root,
        head_sha,
        "majorant domain probe",
    )
    _verify_majorant_domain(
        probe["domain"], artifact_dir / medium_record["artifact"], medium_record["sha256"]
    )
    expected_probe_representation = {
        "density_f16_sha256": transported_medium["density_f16_sha256"],
        "grid_shape": transported_medium["grid_shape"],
        "majorant_grid_shape": transported_medium["grid_shape"],
        "density_sampling": DENSITY_SAMPLING,
        "majorant_query": MAJORANT_QUERY,
        "sigma_t_max": transported_medium["sigma_t_max"],
    }
    if probe["schema"]!="forge3d.nephele.majorant_probe/3" or probe["source_revision"]!=head_sha or probe["medium"]!={"path":medium_record["path"],"sha256":medium_record["sha256"]} or probe["transport_representation"] != expected_probe_representation or probe["sample_mapping"]!={"algorithm":MAJORANT_PROBE_MAPPING,"count":1_000_000}: _fail("provenance_error","majorant probe identity is incomplete")
    raw_output=probe["raw_output"]
    if not isinstance(raw_output,dict) or set(raw_output)!={"path","sha256","encoding","pairs"} or raw_output["path"]!="majorant-domain-probe.pairs.bin" or raw_output["encoding"]!="little-endian-f32-extinction-bound-pairs" or raw_output["pairs"]!=1_000_000: _fail("schema_error","majorant raw pair contract is invalid")
    try: pair_bytes=(artifact_dir/raw_output["path"]).read_bytes()
    except OSError as exc: _fail("artifact_error",f"majorant raw pairs are missing: {exc}")
    if len(pair_bytes)!=1_000_000*8 or hashlib.sha256(pair_bytes).hexdigest()!=raw_output["sha256"]: _fail("provenance_error","majorant raw pairs are missing or corrupt")
    violations=0; max_excess=-math.inf; es=bs=0.0; emin=bmin=math.inf; emax=bmax=-math.inf
    for extinction,bound in struct.iter_unpack("<ff",pair_bytes):
        if not math.isfinite(extinction) or not math.isfinite(bound) or extinction<0 or bound<0: _fail("schema_error","majorant raw pair is not finite nonnegative represented data")
        violations += extinction>bound; max_excess=max(max_excess,extinction-bound); es+=extinction; bs+=bound; emin=min(emin,extinction); emax=max(emax,extinction); bmin=min(bmin,bound); bmax=max(bmax,bound)
    expected_summary={"extinction":{"minimum":emin,"maximum":emax,"sum":es},"bound":{"minimum":bmin,"maximum":bmax,"sum":bs}}
    if probe["summary"]!=expected_summary or probe["probe_count"]!=1_000_000 or probe["violation_count"]!=violations or not math.isclose(probe["max_represented_extinction_minus_bound"],max_excess,rel_tol=0,abs_tol=0) or violations: _fail("gate_failure","Gate 1 represented extinction exceeds represented majorant or summary differs")
    representation_error=_finite(probe["measured_representation_error"],"majorant.measured_representation_error")

    rr = raw["russian_roulette"]
    if not isinstance(rr, dict):
        _fail("schema_error", "gate1-statistics.json.russian_roulette: expected object")
    _keys(rr, {"evidence_sha256"}, "gate1-statistics.json.russian_roulette")
    rr_path = artifact_dir / "rr-evidence.json"
    if _sha256(rr_path) != rr["evidence_sha256"]:
        _fail("provenance_error", "Russian-roulette evidence hash mismatch")
    rr_evidence = _object(rr_path)
    _keys(rr_evidence, {"schema", "source_revision", "analytic_oracle", "producer_tool", "sample_mapping", "raw_output", "accumulators"}, "rr-evidence.json")
    _verify_tool(
        rr_evidence["producer_tool"],
        "scripts/nephele_heterogeneous_comparator.py",
        comparator_cache_key,
        repo_root,
        head_sha,
        "Russian-roulette contribution producer",
    )
    expected_rr_mapping = {
        "algorithm": "canonical-delta-track-single-scatter-rr-v1",
        "parameters": {"columns": ["rr_on", "rr_off"], "rr_trials_per_collision": 4, "encoding": "little-endian-f64"},
    }
    expected_rr_mapping["sha256"] = hashlib.sha256(json.dumps(expected_rr_mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    raw_output = rr_evidence["raw_output"]
    if (
        rr_evidence["schema"] != "forge3d.nephele.russian_roulette_evidence/3"
        or rr_evidence["source_revision"] != head_sha
        or rr_evidence["sample_mapping"] != expected_rr_mapping
        or not isinstance(raw_output, dict)
        or set(raw_output) != {"path", "sha256", "encoding", "pairs"}
        or raw_output["path"] != "rr-contributions.bin"
        or raw_output["encoding"] != "little-endian-f64-rr-on-off-pairs"
    ):
        _fail("provenance_error", "Russian-roulette evidence execution contract is not exact")
    pair_count = _integer(raw_output["pairs"], "rr-evidence.raw_output.pairs", minimum=2)
    if pair_count != 1_000_000:
        _fail("gate_failure", "Gate 1 RR-on/off requires exactly 1,000,000 paired contributions")
    try:
        rr_bytes = (artifact_dir / raw_output["path"]).read_bytes()
    except OSError as exc:
        _fail("artifact_error", f"Russian-roulette raw contribution artifact is missing: {exc}")
    if len(rr_bytes) != pair_count * 16 or hashlib.sha256(rr_bytes).hexdigest() != raw_output["sha256"]:
        _fail("provenance_error", "Russian-roulette raw contribution artifact is corrupt")
    values = {"on": [], "off": []}
    for on_value, off_value in struct.iter_unpack("<dd", rr_bytes):
        if not math.isfinite(on_value) or not math.isfinite(off_value) or on_value < 0 or off_value < 0:
            _fail("schema_error", "Russian-roulette raw contributions must be finite and nonnegative")
        values["on"].append(on_value); values["off"].append(off_value)
    recomputed = {}
    for name in ("on", "off"):
        samples = values[name]
        recomputed[name] = {
            "count": pair_count,
            "sum": math.fsum(samples),
            "sum_squares": math.fsum(value * value for value in samples),
            "minimum": min(samples),
            "maximum": max(samples),
        }
    if rr_evidence["accumulators"] != recomputed:
        _fail("schema_error", "Russian-roulette accumulators differ from raw contributions")
    rr_on = _nonnegative_stats(recomputed["on"], "russian_roulette.on")
    rr_off = _nonnegative_stats(recomputed["off"], "russian_roulette.off")
    oracle = _finite(rr_evidence["analytic_oracle"], "russian_roulette.analytic_oracle")
    slab_albedo = float(slab["sigma_s"][0]) / float(slab["sigma_t_max_channel"])
    expected_oracle = (
        1.0 - math.exp(-float(slab["sigma_t_max_channel"]) * float(slab["density"]) * float(slab["distance"]))
    ) * slab_albedo
    if not math.isclose(oracle, expected_oracle, rel_tol=0.0, abs_tol=0.0):
        _fail("provenance_error", "Russian-roulette oracle differs from the tracked homogeneous slab")
    if not (0.0 <= rr_on.mean <= 1.0 and 0.0 < rr_off.mean <= 1.0):
        _fail("schema_error", "Gate 1 RR-on/off albedo means must lie in [0,1] and RR-off must be nonzero")
    if not all(value.mean - 3 * value.standard_error <= oracle <= value.mean + 3 * value.standard_error for value in (rr_on, rr_off)):
        _fail("gate_failure", "Gate 1 RR-on/off confidence intervals do not contain the analytic slab oracle")
    rr_delta = abs(rr_on.mean - rr_off.mean) / abs(rr_off.mean)
    if rr_delta >= 0.002:
        _fail("gate_failure", "Gate 1 Russian-roulette relative difference is not below 0.2 percent")
    return {
        "homogeneous": {**stats.__dict__, "analytic": analytic, "three_standard_error_pass": True},
        "heterogeneous": {**heterogeneous_stats.__dict__, "reference": comparator_stats.__dict__, "relative_error": relative_error},
        "majorant": {"probe_count": probe["probe_count"], "violation_count": 0, "max_excess": max_excess, "measured_representation_error": representation_error},
        "russian_roulette": {
            "on": {**rr_on.__dict__, "confidence_95": [rr_on.mean - 1.96 * rr_on.standard_error, rr_on.mean + 1.96 * rr_on.standard_error]},
            "off": {**rr_off.__dict__, "confidence_95": [rr_off.mean - 1.96 * rr_off.standard_error, rr_off.mean + 1.96 * rr_off.standard_error]},
            "relative_difference": rr_delta,
            "analytic_oracle": oracle,
        },
    }


def _gate2(artifact_dir: Path, head_sha: str, repo_root: Path) -> dict[str, Any]:
    raw = _object(artifact_dir / "gate2-energy.json")
    _keys(raw, {"schema", "producer", "homogeneous_slab", "sample_mapping", "normalization", "raw_output", "transmitted", "scattered_out", "absorbed", "incident", "closure_residual"}, "gate2-energy.json")
    if raw["schema"] != "forge3d.nephele.gate2_raw/2" or raw["normalization"] != "three independent analog-transport ensembles share one incident energy unit per sample":
        _fail("schema_error", "gate2-energy.json: schema or normalization missing")
    _verify_physical_producer(raw["producer"], repo_root, head_sha)
    _verify_homogeneous_slab(raw["homogeneous_slab"], artifact_dir, repo_root, head_sha)
    _identity_hash(raw["sample_mapping"], "gate2.sample_mapping")
    expected_mapping = {
        "algorithm": "independent-analog-closed-sphere-v1",
        "parameters": {
            "streams": {"transmitted": 0x4E4550482001, "scattered_out": 0x4E4550482002, "absorbed": 0x4E4550482003},
            "normalization": "one incident energy unit per estimator sample",
        },
    }
    expected_mapping["sha256"] = hashlib.sha256(json.dumps(expected_mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if raw["sample_mapping"] != expected_mapping:
        _fail("provenance_error", "Gate 2 independent analog stream identity differs")
    expected_columns = ["transmitted", "scattered_out", "absorbed", "incident", "closure_residual"]
    if raw["raw_output"].get("columns") != expected_columns:
        _fail("provenance_error", "Gate 2 raw columns or independent-estimator identity differs")
    flat = _raw_f64(
        artifact_dir / "gate2-energy-samples.bin",
        raw["raw_output"],
        "Gate 2 closed-sphere transport",
        1_000_000,
    )
    if flat.size != 5_000_000:
        _fail("schema_error", "Gate 2 raw output must contain five values per sample")
    matrix = flat.reshape(1_000_000, 5)
    if not np.isin(matrix[:, :3], (0.0, 1.0)).all() or not np.all(matrix[:, 3] == 1.0):
        _fail("schema_error", "Gate 2 independent energy samples have invalid support")
    if not np.array_equal(matrix[:, 4], matrix[:, 0] + matrix[:, 1] + matrix[:, 2] - matrix[:, 3]):
        _fail("schema_error", "Gate 2 raw closure residual differs from raw independent terms")
    if not np.any(matrix[:, 4] != 0.0):
        _fail("provenance_error", "Gate 2 closure is tautological rather than independently estimated")
    for index, name in enumerate(expected_columns):
        if raw[name] != _accumulator_from_array(matrix[:, index]):
            _fail("schema_error", f"Gate 2 {name} accumulator differs from raw samples")
    terms = {name: _stats(raw[name], f"gate2.{name}", exact_count=1_000_000) for name in expected_columns}
    counts = {term.count for term in terms.values()}
    if (
        len(counts) != 1
        or terms["incident"].mean <= 0.0
        or any(terms[name].mean < 0.0 for name in ("transmitted", "scattered_out", "absorbed"))
    ):
        _fail("schema_error", "gate2-energy.json: terms require one shared positive incident normalization")
    residual = abs(terms["transmitted"].mean + terms["scattered_out"].mean + terms["absorbed"].mean - terms["incident"].mean) / terms["incident"].mean
    recorded_residual = abs(terms["closure_residual"].mean) / terms["incident"].mean
    if not math.isclose(residual, recorded_residual, rel_tol=1.0e-9, abs_tol=1.0e-12):
        _fail("schema_error", "gate2-energy.json: closure residual accumulator is inconsistent with energy terms")
    if residual > 1.0e-3:
        _fail("gate_failure", "Gate 2 energy residual exceeds 1e-3")
    return {"terms": {name: value.__dict__ for name, value in terms.items()}, "relative_residual": residual}


def _visual_gates(
    artifact_dir: Path,
    evaluator: Callable[[np.ndarray], tuple[float, bool]],
    repo_root: Path,
    head_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference = _rgb(artifact_dir / "reference-rgb.npy")
    realtime = _rgb(artifact_dir / "realtime-rgb.npy")
    medium_disabled = _rgb(artifact_dir / "medium-disabled-rgb.npy")
    terrain_disabled = _rgb(artifact_dir / "terrain-occlusion-disabled-rgb.npy")
    if not (reference.shape == realtime.shape == medium_disabled.shape == terrain_disabled.shape):
        _fail("schema_error", "visual RGB artifacts must have identical shapes")
    shape = reference.shape[:2]
    masks = {name: _mask(artifact_dir / filename, shape) for name, filename in FIXTURE_FILES.items() if name.endswith("mask")}
    roi = _mask(artifact_dir / FIXTURE_FILES["godray_roi_mask"], shape)
    delta_reference = _delta_e(realtime, reference)
    sky_values = delta_reference[masks["sky_cloud_mask"]]
    sky_pass_fraction = float(np.mean(sky_values < 2.5))
    reference_roi, realtime_roi = _roi_crop(roi, reference, realtime)
    shaft_ssim = ssim(reference_roi, realtime_roi, data_range=255.0)
    if sky_pass_fraction < 0.95 or not shaft_ssim > 0.95:
        _fail("gate_failure", "Gate 3 DeltaE pass fraction or godray SSIM failed")

    shadow_values = delta_reference[masks["cloud_shadow_mask"] & masks["terrain_mask"]]
    if not shadow_values.size:
        _fail("schema_error", "cloud-shadow terrain intersection must be nonempty")
    aggregate, aggregate_pass = evaluator(shadow_values)
    medium_delta = _delta_e(medium_disabled, realtime)[masks["terrain_mask"]]
    changed_fraction = float(np.mean(medium_delta > 5.0))
    realtime_roi, terrain_disabled_roi = _roi_crop(roi, realtime, terrain_disabled)
    occlusion_ssim = ssim(realtime_roi, terrain_disabled_roi, data_range=255.0)
    if not aggregate_pass or changed_fraction < 0.10 or not occlusion_ssim < 0.80:
        _fail("gate_failure", "Gate 4 terrain-coupling metric failed")

    static = _object(artifact_dir / "gate5-static.json")
    analyzer_path = repo_root / "scripts/nephele_shader_analyzer.py"
    analyzer_hash = _sha256(analyzer_path)
    if (
        hashlib.sha256(_tracked_blob(repo_root, head_sha, analyzer_path)).hexdigest()
        != analyzer_hash
        or _sha256(Path(analyze_shaders.__code__.co_filename)) != analyzer_hash
    ):
        _fail("provenance_error", "Gate 5 analyzer is not the tracked code being executed")
    try:
        recomputed = analyze_shaders(repo_root)
    except (OSError, UnicodeError, ValueError) as exc:
        _fail("gate_failure", f"Gate 5 exhaustive shader analysis failed: {exc}")
    recomputed["tool_sha256"] = analyzer_hash
    if (
        static != recomputed
        or recomputed["unresolved_source_expressions"]
        or any(
            not any(invocation.startswith(wrapper + "<-") for invocation in recomputed["rust_shader_wrapper_invocations"])
            for wrapper in recomputed["rust_shader_construction_wrappers"]
        )
        or ((repo_root / "Cargo.toml").is_file() and (
            recomputed["naga_validated_assemblies"] != len(recomputed["resolved_source_sha256"])
            or len(recomputed["executed_assembled_naga_contracts"]) != recomputed["naga_validated_assemblies"]
        ))
        or recomputed["compute_entry_texture_sample_compare_calls"] != 0
        or recomputed["stale_disabled_shadow_comments"] != 0
    ):
        _fail("gate_failure", "Gate 5 tracked exhaustive shader analysis differs or fails")
    reference_slice = _array(artifact_dir / "reference-terrain-slice.npy")
    realtime_slice = _array(artifact_dir / "realtime-termination-slice.npy")
    if reference_slice.shape != shape or realtime_slice.shape != shape or not np.issubdtype(reference_slice.dtype, np.number) or not np.issubdtype(realtime_slice.dtype, np.number):
        _fail("schema_error", "Gate 5 depth-slice arrays must be numeric and match the fixture")
    shaft = masks["shaft_mask"] & roi
    if not shaft.any():
        _fail("schema_error", "Gate 5 shaft-mask/ROI intersection must be nonempty")
    agreement = float(np.mean(np.abs(reference_slice[shaft] - realtime_slice[shaft]) <= 1.0))
    if agreement < 0.99:
        _fail("gate_failure", "Gate 5 ridgeline agreement is below 99 percent")
    return (
        {"sky_cloud_delta_e_pass_fraction": sky_pass_fraction, "godray_roi_ssim": shaft_ssim},
        {"shadow_aggregate": aggregate, "medium_ablation_changed_fraction": changed_fraction, "terrain_occlusion_ablation_ssim": occlusion_ssim},
        {"compute_entry_texture_sample_compare_calls": 0, "stale_disabled_shadow_comments": 0, "ridgeline_within_one_slice_fraction": agreement},
    )


def _sun_transmittance_diagnostics(
    diagnostics: dict[str, Any], label: str
) -> dict[str, Any]:
    method = diagnostics["sun_transmittance_method"]
    bias = diagnostics["sun_transmittance_bias"]
    segment_length = diagnostics["sun_transmittance_max_segment_length"]
    executed_steps = diagnostics["sun_transmittance_executed_steps"]
    max_abs_error = diagnostics["sun_transmittance_max_abs_error"]
    if (
        method != "bounded_nested_midpoint"
        or bias != "fine_midpoint_with_coarse_fine_abs_rgb_error"
        or type(segment_length) is not float
        or not math.isfinite(segment_length)
        or segment_length <= 0.0
        or type(executed_steps) is not int
        or executed_steps <= 0
        or type(max_abs_error) is not float
        or not math.isfinite(max_abs_error)
        or not 0.0 <= max_abs_error <= 1.0
    ):
        _fail(
            "schema_error",
            f"{label}: executed sun-transmittance diagnostics are invalid",
        )
    return {
        "method": method,
        "bias": bias,
        "max_segment_length": segment_length,
        "executed_steps": executed_steps,
        "max_abs_error": max_abs_error,
    }


def _validate_realtime_diagnostics(
    diagnostics: Any,
    label: str,
    head_sha: str,
    adapter: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(diagnostics, dict):
        _fail("schema_error", f"{label}: diagnostics must be an object")
    _keys(diagnostics, REALTIME_DIAGNOSTIC_KEYS, label)
    expected_driver = f'{adapter["driver"]} {adapter["driver_info"]}'
    if (
        diagnostics["source_revision"] != head_sha
        or str(diagnostics["backend"]).lower() != "vulkan"
        or diagnostics["adapter"] != adapter["name"]
        or diagnostics["driver"] != expected_driver
        or diagnostics["majorant_valid"] is not True
        or diagnostics["majorant_proof"] != "TrilinearConvexHull"
        or diagnostics["executed_multi_scatter"] is not True
        or _integer(diagnostics["sample_count"], f"{label}.sample_count", minimum=1) < 1
        or _integer(diagnostics["step_count"], f"{label}.step_count", minimum=1) < 1
    ):
        _fail("identity_error", f"{label}: candidate diagnostics identity is invalid")
    for key in (
        "host_visible_bytes", "froxel_device_local_bytes", "density_device_local_bytes",
        "majorant_device_local_bytes", "staging_readback_bytes", "single_scatter_dispatches",
        "multiple_scatter_dispatches", "terrain_trace_queries",
    ):
        _integer(diagnostics[key], f"{label}.{key}", minimum=0)
    for key in (
        "single_scatter_luminance", "multiple_scatter_luminance",
        "energy_accounting_residual",
    ):
        value = diagnostics[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            _fail("schema_error", f"{label}.{key}: expected a finite number")
        if value < 0:
            _fail("schema_error", f"{label}.{key} is negative")
    return _sun_transmittance_diagnostics(diagnostics, label)


def _gate6(
    artifact_dir: Path,
    head_sha: str,
    context: dict[str, Any],
    adapter: dict[str, Any],
    input_bundle_sha256: str,
    reference_camera_contract: dict[str, Any],
) -> dict[str, Any]:
    frame_a = _array(artifact_dir / "froxel-frame-run-a.npy")
    frame_b = _array(artifact_dir / "froxel-frame-run-b.npy")
    hash_a, hash_b = _sha256(artifact_dir / "froxel-frame-run-a.npy"), _sha256(artifact_dir / "froxel-frame-run-b.npy")
    adapter_hash = hashlib.sha256(
        json.dumps(adapter, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    runs = []
    for label, frame_name, frame_hash in (
        ("a", "froxel-frame-run-a.npy", hash_a),
        ("b", "froxel-frame-run-b.npy", hash_b),
    ):
        run = _object(artifact_dir / f"run-{label}.json")
        _keys(
            run,
            {
                "schema", "process_id", "run_nonce", "head_sha", "tracked_worktree_clean",
                "fixture_manifest_sha256", "input_bundle_sha256", "adapter_identity_sha256",
                "backend", "frame_file", "frame_sha256", "camera_contract",
                "presentation_identity", "medium_disabled_rgb_sha256", "diagnostics",
            },
            f"run-{label}.json",
        )
        if (
            run["schema"] != "forge3d.nephele.clean_run/1"
            or _integer(run["process_id"], f"run-{label}.process_id", minimum=1) < 1
            or not SHA256_RE.fullmatch(str(run["run_nonce"]))
            or run["head_sha"] != head_sha
            or run["tracked_worktree_clean"] is not True
            or run["fixture_manifest_sha256"] != context["fixture_manifest_sha256"]
            or run["input_bundle_sha256"] != input_bundle_sha256
            or run["adapter_identity_sha256"] != adapter_hash
            or str(run["backend"]).lower() != "vulkan"
            or run["frame_file"] != frame_name
            or run["frame_sha256"] != frame_hash
        ):
            _fail("identity_error", f"run-{label}.json is not bound to its exact clean render")
        run["camera_contract"] = _camera_contract(run["camera_contract"], f"run-{label}.json")
        if any(
            not np.array_equal(
                np.asarray(run["camera_contract"][key], dtype=np.float32),
                np.asarray(reference_camera_contract[key], dtype=np.float32),
            )
            for key in reference_camera_contract
        ):
            _fail("identity_error", f"run-{label}.json camera differs from the reference bitwise")
        expected_presentation = {
            "capture_api": "TerrainRenderer._capture_nephele_acceptance",
            "color_pipeline": ["white_balance", "exposure", "aces", "lut", "iec_61966_2_1_srgb8"],
            "no_medium_same_call": label == "a",
        }
        expected_no_medium_hash = _sha256(artifact_dir / "medium-disabled-rgb.npy") if label == "a" else None
        if (
            run["presentation_identity"] != expected_presentation
            or run["medium_disabled_rgb_sha256"] != expected_no_medium_hash
        ):
            _fail("identity_error", f"run-{label}.json does not bind the shared acceptance presentation path")
        diagnostics = run["diagnostics"]
        _validate_realtime_diagnostics(
            diagnostics,
            f"run-{label}.json.diagnostics",
            head_sha,
            adapter,
        )
        runs.append(run)
    if runs[0]["process_id"] == runs[1]["process_id"] or runs[0]["run_nonce"] == runs[1]["run_nonce"]:
        _fail("identity_error", "Gate 6 requires two distinct process identities and run nonces")
    if runs[0]["diagnostics"] != runs[1]["diagnostics"]:
        _fail(
            "gate_failure",
            "Gate 6 candidate diagnostics, including actual executed step_count, are not exact across clean runs",
        )
    if (
        frame_a.dtype != frame_b.dtype
        or frame_a.shape != frame_b.shape
        or not np.issubdtype(frame_a.dtype, np.number)
        or not np.isfinite(frame_a).all()
        or not np.isfinite(frame_b).all()
        or hash_a != hash_b
        or not np.array_equal(frame_a, frame_b)
    ):
        _fail("gate_failure", "Gate 6 clean-run output frames are not bitwise identical")
    memory = _object(artifact_dir / "memory.json")
    _keys(memory, {"schema", "output_dimensions", "internal_dimensions", "include_no_medium", "froxel_depth", "termination_integration_steps", "peak_host_visible_bytes", "froxel_device_local_bytes", "density_majorant_device_local_bytes", "staging_bytes", "readback_bytes"}, "memory.json")
    if memory["schema"] != "forge3d.nephele.memory/2":
        _fail("schema_error", "memory.json: unknown schema")
    output_dimensions = memory["output_dimensions"]
    internal_dimensions = memory["internal_dimensions"]
    if (
        not isinstance(output_dimensions, list)
        or len(output_dimensions) != 2
        or not isinstance(internal_dimensions, list)
        or len(internal_dimensions) != 2
        or any(_integer(value, "memory dimensions", minimum=1) < 1 for value in output_dimensions + internal_dimensions)
        or output_dimensions != [frame_a.shape[1], frame_a.shape[0]]
        or internal_dimensions != output_dimensions
        or type(memory["include_no_medium"]) is not bool
        or memory["include_no_medium"]
        != runs[0]["presentation_identity"]["no_medium_same_call"]
    ):
        _fail("schema_error", "memory.json dimensions or conditional no-medium capture differ")
    values = {
        key: _integer(memory[key], f"memory.{key}")
        for key in {
            "peak_host_visible_bytes", "froxel_device_local_bytes",
            "density_majorant_device_local_bytes", "staging_bytes", "readback_bytes",
        }
    }
    output_pixels = math.prod(output_dimensions)
    internal_pixels = math.prod(internal_dimensions)
    expected_readback_bytes = (
        (36 + 4 * int(memory["include_no_medium"])) * output_pixels
        + 24 * internal_pixels
    )
    if values["readback_bytes"] != expected_readback_bytes:
        _fail(
            "gate_failure",
            "Gate 6 readback bytes differ from exact output/internal dimensions and no-medium capture",
        )
    froxel_depth = _integer(memory["froxel_depth"], "memory.froxel_depth", minimum=1)
    termination = _array(artifact_dir / "realtime-termination-slice.npy")
    if (
        termination.shape != (internal_dimensions[1], internal_dimensions[0])
        or not np.issubdtype(termination.dtype, np.number)
        or not np.isfinite(termination).all()
        or np.any(termination < 0)
        or np.any(termination > froxel_depth - 1)
        or np.any(termination != np.floor(termination))
    ):
        _fail("schema_error", "Gate 6 termination slices do not define exact integration work")
    expected_integration_steps = int(
        np.asarray(termination, dtype=np.uint64).sum(dtype=np.uint64) + termination.size
    )
    if (
        _integer(memory["termination_integration_steps"], "memory.termination_integration_steps", minimum=1)
        != expected_integration_steps
        or runs[0]["diagnostics"]["step_count"] < expected_integration_steps
    ):
        _fail(
            "gate_failure",
            "Gate 6 actual step_count does not include the exact termination integration work",
        )
    if values["peak_host_visible_bytes"] >= 512 * 1024**2:
        _fail("gate_failure", "Gate 6 peak host-visible bytes are not below 512 MiB")
    aov_files = {
            "transmittance": "transmittance.npy",
            "in_scatter": "in-scatter.npy",
            "cloud_shadow": "cloud-shadow-aov.npy",
            "optical_depth": "optical-depth.npy",
    }
    for filename in aov_files.values():
        aov = _array(artifact_dir / filename)
        if not np.issubdtype(aov.dtype, np.number) or not np.isfinite(aov).all():
            _fail("schema_error", f"{filename}: AOV must contain finite numeric values")
    aov_hashes = {name: _sha256(artifact_dir / filename) for name, filename in aov_files.items()}
    return {
        "frame_hashes": [hash_a, hash_b],
        "run_manifests": runs,
        "sun_transmittance": _sun_transmittance_diagnostics(
            runs[0]["diagnostics"], "run-a.json.diagnostics"
        ),
        "memory": values,
        "aov_hashes_non_gating": aov_hashes,
    }


def build_report(
    artifact_dir: Path,
    *,
    head_sha: str,
    repo_root: Path = ROOT,
    imported_native_path: Path | None = None,
    observed_host: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    if not SHA_RE.fullmatch(head_sha):
        _fail("identity_error", "head_sha must be one full lowercase Git SHA")
    repo_root = repo_root.resolve()
    if imported_native_path is None:
        try:
            native = importlib.import_module("forge3d._forge3d")
            imported_native_path = Path(native.__file__).resolve()
        except (ImportError, TypeError) as exc:
            _fail("identity_error", f"installed Forge3D native extension is not importable: {exc}")
    if observed_host is None:
        observed_host = (
            platform.system(),
            platform.machine(),
            os.environ.get("FORGE3D_NEPHELE_LANE", ""),
        )
    identity = _verify_identity(
        artifact_dir, head_sha, repo_root, imported_native_path, observed_host
    )
    context = identity["context"]
    fixture_manifest_path = repo_root / "tests/nephele/fixture-manifest.json"
    gate4_policy_path = repo_root / "tests/nephele/gate4-policy.json"
    manifest = _verify_canonical_file(
        artifact_dir / "fixture-manifest.json",
        fixture_manifest_path,
        context["fixture_manifest_sha256"],
        "fixture manifest",
        repo_root=repo_root,
        commit=context["fixture_commit"],
    )
    input_bundle_sha256 = _verify_fixture_manifest(
        artifact_dir, manifest, repo_root, context["fixture_commit"]
    )
    reference_convergence, reference_artifacts, reference_camera_contract = _verify_reference_provenance(
        artifact_dir, manifest, repo_root, context["fixture_commit"]
    )
    policy = _verify_canonical_file(
        artifact_dir / "gate4-policy.json",
        gate4_policy_path,
        context["gate4_policy_sha256"],
        "Gate 4 policy",
        repo_root=repo_root,
        commit=head_sha,
    )
    evaluator = _policy_evaluator(policy)
    junit = _verify_junit(artifact_dir / "junit.xml")
    gate1 = _gate1(
        artifact_dir,
        head_sha,
        repo_root,
        manifest["scene_inputs"]["medium"],
    )
    gate2 = _gate2(artifact_dir, head_sha, repo_root)
    gate3, gate4, gate5 = _visual_gates(
        artifact_dir, evaluator, repo_root, head_sha
    )
    gate6 = _gate6(
        artifact_dir, head_sha, context, identity["adapter"], input_bundle_sha256,
        reference_camera_contract,
    )
    input_names = sorted(
        RAW_FILES
        | reference_artifacts
        | {record["artifact"] for record in manifest["scene_inputs"].values()}
        | {
            "run-context.json",
            "adapter-probe.json",
            "installed-wheel-runtime.json",
            identity["runtime"]["wheel_filename"],
            "native-extension.bin",
            "fixture-manifest.json",
            "gate4-policy.json",
            "junit.xml",
        }
    )
    return {
        "schema": "forge3d.nephele.verification/1",
        "status": "PASS",
        "head_sha": head_sha,
        "identity": {"adapter": identity["adapter"], "runtime": identity["runtime"], "junit": junit},
        "fixture": {
            "id": manifest["fixture_id"],
            "revision": manifest["revision"],
            "commit": context["fixture_commit"],
            "manifest_sha256": context["fixture_manifest_sha256"],
            "input_bundle_sha256": input_bundle_sha256,
            "homogeneous_slab_sha256": _sha256(artifact_dir / "homogeneous-slab.json"),
            "reference_convergence": reference_convergence,
        },
        "gate4_policy": policy,
        "gates": {"gate1": gate1, "gate2": gate2, "gate3": gate3, "gate4": gate4, "gate5": gate5, "gate6": gate6},
        "raw_artifact_sha256": {name: _sha256(artifact_dir / name) for name in input_names},
    }


def _error_record(exc: EvidenceError) -> dict[str, str]:
    return {"schema": "forge3d.nephele.verification_error/1", "status": "FAIL", "code": exc.code, "message": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.artifact_dir, head_sha=args.head_sha)
    except EvidenceError as exc:
        error = _error_record(exc)
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        (args.artifact_dir / "verification-error.json").write_text(json.dumps(error, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2
    output = args.artifact_dir / "verification-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.artifact_dir / "lane-ran.json").write_text(json.dumps({"schema": "forge3d.nephele.lane/1", "status": "RAN", "head_sha": args.head_sha}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
