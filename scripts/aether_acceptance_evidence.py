#!/usr/bin/env python3
"""Build fail-closed, exact-head evidence for the AETHER Metal closure lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

if __package__:
    from .assert_junit_zero_skips import verify_junit
else:
    from assert_junit_zero_skips import verify_junit


SHA_RE = re.compile(r"[0-9a-f]{40}")
PHYSICAL_METAL_TYPES = {"discretegpu", "integratedgpu"}
SOFTWARE_TOKENS = ("basic render driver", "lavapipe", "llvmpipe", "swiftshader", "warp")
SUN_ELEVATION_LABELS = ("-5", "0", "5", "10", "30", "60", "89")
SKY_CASE_LABELS = tuple(
    f"az{azimuth}_x{x}_y{y}"
    for azimuth in (20, 90, 160)
    for y in (8, 20, 28)
    for x in (8, 32, 56)
)
EXPECTED_DELTA_E_KEYS = frozenset(
    f"{elevation}:{case}"
    for elevation in SUN_ELEVATION_LABELS
    for case in SKY_CASE_LABELS
)
REQUIRED_JUNIT_CASES = frozenset(
    {
        (
            "tests.test_atmosphere_reference",
            "test_sky_delta_e2000_under_two_for_full_sun_elevation_sweep",
        ),
        (
            "tests.test_atmosphere_reference",
            "test_terrain_saturation_falloff_matches_scattering_law_within_ten_percent",
        ),
    }
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: JSON root must be an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_physical_metal(record: dict[str, Any]) -> dict[str, Any]:
    probe = record.get("probe")
    if not isinstance(probe, dict):
        raise ValueError("adapter-probe.json: probe must be an object")
    name = str(probe.get("name", ""))
    if record.get("status") != "passed" or record.get("mode") != "aether-metal":
        raise ValueError("adapter-probe.json: AETHER probe did not pass")
    if str(record.get("requested_backend", "")).lower() != "metal":
        raise ValueError("adapter-probe.json: requested backend must be Metal")
    if probe.get("status") != "ok" or str(probe.get("backend", "")).lower() != "metal":
        raise ValueError("adapter-probe.json: active adapter is not a successful Metal device")
    if str(probe.get("device_type", "")).lower() not in PHYSICAL_METAL_TYPES:
        raise ValueError("adapter-probe.json: device is not a physical Metal GPU")
    if probe.get("software_fallback") is not False:
        raise ValueError("adapter-probe.json: software fallback cannot prove AETHER")
    if not name or any(token in name.lower() for token in SOFTWARE_TOKENS):
        raise ValueError("adapter-probe.json: adapter name identifies software or is empty")
    return probe


def _require_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    if metrics.get("schema_version") != 1:
        raise ValueError("metrics.json: schema_version must be 1")
    sweep = metrics.get("delta_e_sweep")
    saturation = metrics.get("saturation_falloff")
    if not isinstance(sweep, dict):
        raise ValueError("metrics.json: full DeltaE sweep is missing")
    actual_keys = frozenset(str(key) for key in sweep)
    if actual_keys != EXPECTED_DELTA_E_KEYS:
        missing = sorted(EXPECTED_DELTA_E_KEYS - actual_keys)
        extra = sorted(actual_keys - EXPECTED_DELTA_E_KEYS)
        raise ValueError(
            "metrics.json: DeltaE sweep topology mismatch "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )
    try:
        scores = [float(value) for value in sweep.values()]
    except (TypeError, ValueError) as exc:
        raise ValueError("metrics.json: DeltaE sweep values must be numeric") from exc
    if not scores or any(not (0.0 <= value < 2.0) for value in scores):
        raise ValueError("metrics.json: every DeltaE score must be finite and below 2")
    if not isinstance(saturation, dict):
        raise ValueError("metrics.json: saturation falloff evidence is missing")
    try:
        distances = [float(value) for value in saturation["distances_m"]]
        measured = [float(value) for value in saturation["measured_saturation"]]
        predicted = [float(value) for value in saturation["predicted_saturation"]]
        recorded_measured_ratio = float(saturation["measured_far_near_ratio"])
        recorded_predicted_ratio = float(saturation["predicted_far_near_ratio"])
        recorded_relative_error = float(saturation["relative_error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metrics.json: saturation raw evidence is invalid") from exc
    raw_values = [
        *distances,
        *measured,
        *predicted,
        recorded_measured_ratio,
        recorded_predicted_ratio,
        recorded_relative_error,
    ]
    if (
        len(distances) != 2
        or len(measured) != 2
        or len(predicted) != 2
        or any(not math.isfinite(value) for value in raw_values)
        or not (distances[0] > 0.0 and distances[1] > distances[0] * 1.10)
        or not (
            0.0 <= measured[1] < measured[0] <= 1.0
            and measured[0] > 0.05
            and measured[0] - measured[1] > 0.005
        )
        or not (
            0.0 <= predicted[1] < predicted[0] <= 1.0
            and predicted[0] > 0.05
            and predicted[0] - predicted[1] > 0.005
        )
    ):
        raise ValueError("metrics.json: saturation raw evidence is outside the gate domain")
    measured_ratio = measured[1] / measured[0]
    predicted_ratio = predicted[1] / predicted[0]
    relative_error = abs(measured_ratio - predicted_ratio) / max(
        abs(predicted_ratio), 1.0e-8
    )
    for recorded, computed, label in (
        (recorded_measured_ratio, measured_ratio, "measured ratio"),
        (recorded_predicted_ratio, predicted_ratio, "predicted ratio"),
        (recorded_relative_error, relative_error, "relative error"),
    ):
        if not math.isclose(recorded, computed, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError(f"metrics.json: saturation {label} is inconsistent with raw values")
    if not 0.0 <= relative_error <= 0.10:
        raise ValueError("metrics.json: saturation falloff error exceeds 10 percent")
    return {"max_delta_e_2000": max(scores), "saturation_relative_error": relative_error}


def _require_junit_cases(path: Path) -> None:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"junit.xml: cannot inspect required AETHER cases: {exc}") from exc
    actual = frozenset(
        (testcase.attrib.get("classname", ""), testcase.attrib.get("name", ""))
        for testcase in root.iter("testcase")
    )
    missing = sorted(REQUIRED_JUNIT_CASES - actual)
    if missing:
        raise ValueError(f"junit.xml: required AETHER gate cases are missing: {missing}")


def build_summary(
    artifact_dir: Path,
    *,
    head_sha: str,
    repository: str,
    run_id: str,
    run_attempt: str,
    job: str,
) -> dict[str, Any]:
    if not SHA_RE.fullmatch(head_sha):
        raise ValueError("head_sha must be one full lowercase Git SHA")
    checked_out = (artifact_dir / "checked-out-head.txt").read_text(encoding="utf-8").strip()
    if checked_out != head_sha:
        raise ValueError("checked-out head does not match the selected acceptance SHA")

    adapter_path = artifact_dir / "adapter-probe.json"
    junit_path = artifact_dir / "junit.xml"
    metrics_path = artifact_dir / "metrics.json"
    adapter = _require_physical_metal(_read_object(adapter_path))
    counts = verify_junit(junit_path)
    _require_junit_cases(junit_path)
    thresholds = _require_metrics(_read_object(metrics_path))

    return {
        "schema_version": 1,
        "status": "passed",
        "repository": repository,
        "head_sha": head_sha,
        "checked_out_head": checked_out,
        "run_id": str(run_id),
        "run_attempt": str(run_attempt),
        "job": job,
        "adapter": adapter,
        "junit": counts.as_dict(),
        "thresholds": {
            "delta_e_2000_strict_upper_bound": 2.0,
            "saturation_relative_error_upper_bound": 0.10,
            **thresholds,
        },
        "sha256": {
            "adapter_probe": _sha256(adapter_path),
            "junit": _sha256(junit_path),
            "metrics": _sha256(metrics_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    summary = build_summary(
        args.artifact_dir,
        head_sha=args.head_sha,
        repository=args.repository,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        job=args.job,
    )
    output = args.artifact_dir / "acceptance-summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
