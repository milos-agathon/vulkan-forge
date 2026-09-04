"""Assemble fail-closed CARTOGRAPHER-PRIME runner evidence.

The gated test writes solver measurements. This post-test step validates the
JUnit report (including zero skips), resolves runtime versions and the actual
checked-out Git SHA, then creates the single artifact consumed by the
cross-runner verifier.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from scripts.assert_junit_zero_skips import verify_junit


EVIDENCE_SCHEMA = "forge3d.cartographer-prime.runner-evidence.v1"
MEASUREMENTS_SCHEMA = "forge3d.cartographer-prime.measurements.v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_BUILD_PROFILE = "release-lto"


def _command(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_measurements(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MEASUREMENTS_SCHEMA:
        raise ValueError("measurement schema mismatch")
    if not HASH_RE.fullmatch(str(payload.get("plan_hash", ""))):
        raise ValueError("measurement plan_hash must be lowercase sha256")
    gap = payload.get("optimality_gap")
    if not isinstance(gap, (int, float)) or isinstance(gap, bool):
        raise ValueError("measurement optimality_gap must be numeric")
    occluded = payload.get("occluded_placement_count")
    if not isinstance(occluded, int) or isinstance(occluded, bool):
        raise ValueError("measurement occluded_placement_count must be an integer")
    count = payload.get("tested_instance_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("measurement tested_instance_count must be positive")
    return payload


def build_evidence(
    *,
    measurements_path: Path,
    junit_path: Path,
    expected_sha: str,
    runner_config: str,
    runner_os: str,
    runner_arch: str,
    build_profile: str,
    repository: Path,
) -> dict:
    measurements = _load_measurements(measurements_path)
    counts = verify_junit(junit_path)
    actual_sha = _command("git", "-C", str(repository), "rev-parse", "HEAD")
    if not SHA_RE.fullmatch(expected_sha):
        raise ValueError("expected SHA must be a 40-character lowercase Git SHA")
    if actual_sha != expected_sha:
        raise ValueError(
            f"checked-out SHA mismatch: expected {expected_sha}, got {actual_sha}"
        )
    rustc = _command("rustc", "--version", "--verbose")
    if not all((runner_config, runner_os, runner_arch, build_profile, rustc)):
        raise ValueError("runner and runtime provenance fields must be non-empty")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (
        3,
        11,
    ):
        raise ValueError("acceptance evidence requires CPython 3.11")
    if build_profile != EXPECTED_BUILD_PROFILE:
        raise ValueError(
            f"acceptance evidence requires build profile {EXPECTED_BUILD_PROFILE}"
        )

    return {
        "schema": EVIDENCE_SCHEMA,
        "git_sha": actual_sha,
        "runner": {
            "config": runner_config,
            "os": runner_os,
            "arch": runner_arch,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "rustc": rustc,
            "build_profile": build_profile,
        },
        "metrics": {
            "plan_hash": measurements["plan_hash"],
            "optimality_gap": measurements["optimality_gap"],
            "occluded_placement_count": measurements["occluded_placement_count"],
            "tested_instance_count": measurements["tested_instance_count"],
        },
        "tests": counts.as_dict(),
        "evidence_scope": "hosted-runner-config",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--runner-config", required=True)
    parser.add_argument("--runner-os", required=True)
    parser.add_argument("--runner-arch", required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        evidence = build_evidence(
            measurements_path=args.measurements,
            junit_path=args.junit,
            expected_sha=args.expected_sha,
            runner_config=args.runner_config,
            runner_os=args.runner_os,
            runner_arch=args.runner_arch,
            build_profile=args.build_profile,
            repository=args.repository,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"CARTOGRAPHER-PRIME evidence creation failed: {exc}"
        ) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
