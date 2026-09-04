"""Verify CARTOGRAPHER-PRIME evidence from all required hosted runners."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

EVIDENCE_SCHEMA = "forge3d.cartographer-prime.runner-evidence.v1"
GOLDEN_SCHEMA = "forge3d.cartographer-prime.golden.v1"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PYTHON_VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)$"
)
LLVM_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)$"
)
RUSTC_BANNER_RE = re.compile(
    r"^rustc (?P<release>[0-9]+\.[0-9]+\.[0-9]+) "
    r"\((?P<short_hash>[0-9a-f]{7,40}) "
    r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\)$"
)
EXPECTED_BUILD_PROFILE = "release-lto"
MAX_GAP = 0.02  # Definition-of-done threshold in 07-cartographer-prime.md.


class EvidenceError(ValueError):
    """Evidence is absent, inconsistent, or fails the acceptance contract."""


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return value


def _parse_rustc(value: str, config: str) -> dict[str, str]:
    lines = value.splitlines()
    if not lines:
        raise EvidenceError(f"invalid rustc provenance for {config}")
    banner = RUSTC_BANNER_RE.fullmatch(lines[0])
    fields = {}
    for line in lines[1:]:
        key, separator, field_value = line.partition(": ")
        if not separator or not key or not field_value or key in fields:
            raise EvidenceError(f"invalid rustc provenance for {config}")
        fields[key] = field_value
    required = {"binary", "commit-hash", "commit-date", "host", "release", "LLVM version"}
    if not banner or not required <= set(fields):
        raise EvidenceError(f"invalid rustc provenance for {config}")
    if fields["binary"] != "rustc":
        raise EvidenceError(f"invalid rustc binary for {config}")
    commit_hash = fields["commit-hash"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit_hash):
        raise EvidenceError(f"invalid rustc commit hash for {config}")
    if not commit_hash.startswith(banner.group("short_hash")):
        raise EvidenceError(f"inconsistent rustc commit hash for {config}")
    if fields["commit-date"] != banner.group("date"):
        raise EvidenceError(f"inconsistent rustc commit date for {config}")
    if fields["release"] != banner.group("release"):
        raise EvidenceError(f"inconsistent rustc release for {config}")
    llvm_version = fields["LLVM version"]
    if not LLVM_RE.fullmatch(llvm_version):
        raise EvidenceError(f"invalid rustc LLVM version for {config}")
    return {
        "release": fields["release"],
        "commit_hash": commit_hash,
        "commit_date": fields["commit-date"],
        "host": fields["host"],
        "llvm_version": llvm_version,
    }


def _parse_python_version(value: str, config: str) -> tuple[int, int, int]:
    match = PYTHON_VERSION_RE.fullmatch(value)
    if not match:
        raise EvidenceError(f"wrong Python runtime for {config}")
    version = tuple(int(match.group(field)) for field in ("major", "minor", "patch"))
    if version[:2] != (3, 11):
        raise EvidenceError(f"wrong Python runtime for {config}")
    return version


def verify_evidence(artifacts: Path, golden_path: Path, expected_sha: str) -> dict:
    if not SHA_RE.fullmatch(expected_sha):
        raise EvidenceError("expected SHA must be a 40-character lowercase Git SHA")
    golden = _load_json(golden_path)
    if golden.get("schema") != GOLDEN_SCHEMA:
        raise EvidenceError("golden schema mismatch")
    golden_hash = golden.get("plan_hash")
    if not isinstance(golden_hash, str) or not HASH_RE.fullmatch(golden_hash):
        raise EvidenceError("golden plan_hash must be lowercase sha256")
    expected_test_count = golden.get("expected_test_count")
    if (
        not isinstance(expected_test_count, int)
        or isinstance(expected_test_count, bool)
        or expected_test_count <= 0
    ):
        raise EvidenceError("golden expected_test_count must be a positive integer")
    expected_instance_count = golden.get("expected_tested_instance_count")
    if (
        not isinstance(expected_instance_count, int)
        or isinstance(expected_instance_count, bool)
        or expected_instance_count <= 0
    ):
        raise EvidenceError(
            "golden expected_tested_instance_count must be a positive integer"
        )
    required_rows = golden.get("required_runner_configs")
    if not isinstance(required_rows, list) or not required_rows:
        raise EvidenceError("golden required_runner_configs must be non-empty")
    required = {}
    for row in required_rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "runner_os",
            "runner_arch",
            "rust_host",
        }:
            raise EvidenceError("invalid required runner config entry")
        config_id = row["id"]
        if not all(
            isinstance(row[key], str) and row[key]
            for key in ("id", "runner_os", "runner_arch", "rust_host")
        ):
            raise EvidenceError(
                "required runner config fields must be non-empty strings"
            )
        if config_id in required:
            raise EvidenceError(f"duplicate required runner config: {config_id}")
        required[config_id] = row

    paths = sorted(artifacts.rglob("cartographer-prime-evidence.json"))
    if not paths:
        raise EvidenceError("no CARTOGRAPHER-PRIME runner evidence found")
    found = {}
    runtime_identities = {}
    for path in paths:
        evidence = _load_json(path)
        if evidence.get("schema") != EVIDENCE_SCHEMA:
            raise EvidenceError(f"evidence schema mismatch: {path}")
        runner = evidence.get("runner")
        if not isinstance(runner, dict):
            raise EvidenceError(f"missing runner provenance: {path}")
        config = runner.get("config")
        if not isinstance(config, str) or not config:
            raise EvidenceError(f"invalid runner config: {path}")
        if config in found:
            raise EvidenceError(f"duplicate runner config: {config}")
        if config not in required:
            raise EvidenceError(f"unexpected runner config: {config}")
        found[config] = evidence

        if evidence.get("git_sha") != expected_sha:
            raise EvidenceError(f"wrong SHA for {config}")
        expected_runner = required[config]
        if runner.get("os") != expected_runner["runner_os"]:
            raise EvidenceError(f"wrong runner OS for {config}")
        if runner.get("arch") != expected_runner["runner_arch"]:
            raise EvidenceError(f"wrong runner arch for {config}")
        runtime = evidence.get("runtime")
        if not isinstance(runtime, dict) or not all(
            isinstance(runtime.get(key), str) and runtime[key]
            for key in ("python", "python_implementation", "rustc", "build_profile")
        ):
            raise EvidenceError(f"missing runtime provenance for {config}")
        python_version = runtime["python"]
        if runtime["python_implementation"] != "CPython":
            raise EvidenceError(f"wrong Python runtime for {config}")
        python_semver = _parse_python_version(python_version, config)
        if runtime["build_profile"] != EXPECTED_BUILD_PROFILE:
            raise EvidenceError(f"wrong build profile for {config}")
        rustc = _parse_rustc(runtime["rustc"], config)
        if rustc["host"] != expected_runner["rust_host"]:
            raise EvidenceError(f"wrong rustc host for {config}")
        runtime_identities[config] = {
            "python": python_version,
            "python_implementation": runtime["python_implementation"],
            "python_major_minor": python_semver[:2],
            "rust_release": rustc["release"],
            "rust_commit_hash": rustc["commit_hash"],
            "rust_commit_date": rustc["commit_date"],
            "rust_llvm_version": rustc["llvm_version"],
            "build_profile": runtime["build_profile"],
        }
        if evidence.get("evidence_scope") != "hosted-runner-config":
            raise EvidenceError(f"wrong evidence scope for {config}")
        tests = evidence.get("tests")
        if not isinstance(tests, dict) or not all(
            isinstance(tests.get(field), int) and not isinstance(tests[field], bool)
            for field in ("tests", "failures", "errors", "skipped")
        ):
            raise EvidenceError(f"invalid test counts for {config}")
        if tests["tests"] <= 0:
            raise EvidenceError(f"no tests recorded for {config}")
        if tests["tests"] != expected_test_count:
            raise EvidenceError(
                f"wrong test count for {config}: expected {expected_test_count}, "
                f"got {tests['tests']}"
            )
        if any(tests.get(field) != 0 for field in ("failures", "errors", "skipped")):
            raise EvidenceError(f"non-clean or skipped tests for {config}")

        metrics = evidence.get("metrics")
        if not isinstance(metrics, dict):
            raise EvidenceError(f"missing metrics for {config}")
        plan_hash = metrics.get("plan_hash")
        if not isinstance(plan_hash, str) or not HASH_RE.fullmatch(plan_hash):
            raise EvidenceError(f"invalid plan hash for {config}")
        gap = metrics.get("optimality_gap")
        if (
            not isinstance(gap, (int, float))
            or isinstance(gap, bool)
            or not math.isfinite(gap)
            or gap < 0.0
            or gap > MAX_GAP
        ):
            raise EvidenceError(f"optimality gap failure for {config}: {gap!r}")
        occluded = metrics.get("occluded_placement_count")
        if (
            not isinstance(occluded, int)
            or isinstance(occluded, bool)
            or occluded != 0
        ):
            raise EvidenceError(f"occluded placement failure for {config}")
        tested = metrics.get("tested_instance_count")
        if not isinstance(tested, int) or isinstance(tested, bool) or tested <= 0:
            raise EvidenceError(f"invalid tested instance count for {config}")
        if tested != expected_instance_count:
            raise EvidenceError(
                f"wrong tested instance count for {config}: "
                f"expected {expected_instance_count}, got {tested}"
            )

    missing = sorted(set(required) - set(found))
    if missing:
        raise EvidenceError(f"missing runner configs: {', '.join(missing)}")
    test_counts = {value["tests"]["tests"] for value in found.values()}
    if len(test_counts) != 1:
        raise EvidenceError("cross-runner test counts disagree")
    instance_counts = {
        value["metrics"]["tested_instance_count"] for value in found.values()
    }
    if len(instance_counts) != 1:
        raise EvidenceError("cross-runner tested instance counts disagree")
    for field in (
        "python_implementation",
        "python_major_minor",
        "rust_release",
        "rust_commit_hash",
        "rust_commit_date",
        "rust_llvm_version",
        "build_profile",
    ):
        values = {identity[field] for identity in runtime_identities.values()}
        if len(values) != 1:
            raise EvidenceError(f"cross-runner {field} provenance disagrees")
    hashes = {value["metrics"]["plan_hash"] for value in found.values()}
    if len(hashes) != 1:
        raise EvidenceError("runner plan hashes disagree")
    measured_hash = next(iter(hashes))
    if measured_hash != golden_hash:
        raise EvidenceError("runner plan hash does not match committed golden")

    return {
        "schema": "forge3d.cartographer-prime.verification.v1",
        "git_sha": expected_sha,
        "runner_configs": sorted(found),
        "python_versions": {
            config: runtime_identities[config]["python"] for config in sorted(found)
        },
        "plan_hash": measured_hash,
        "maximum_optimality_gap": max(
            value["metrics"]["optimality_gap"] for value in found.values()
        ),
        "occluded_placement_count": sum(
            value["metrics"]["occluded_placement_count"] for value in found.values()
        ),
        "evidence_scope": "three-hosted-runner-configs",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_evidence(args.artifacts, args.golden, args.expected_sha)
    except EvidenceError as exc:
        raise SystemExit(f"CARTOGRAPHER-PRIME verification failed: {exc}") from exc
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
