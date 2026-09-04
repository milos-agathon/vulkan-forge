#!/usr/bin/env python
# scripts/ci_pytest_lane.py
# CENSOR validation profiles. The fast profile protects architectural truth on
# routine changes. The full profile retains main's split non-slow/slow lanes for
# explicit acceptance and release validation.
# RELEVANT FILES: tests/UNRUN.toml, tests/_toml_compat.py, .github/workflows/ci.yml
"""Run a focused or full CENSOR pytest profile, forwarding pytest arguments."""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import deque
from os import environ
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
UNRUN_TOML = TESTS / "UNRUN.toml"
SLOW_LANE_SELECTOR = "--slow-lane"

# tests/_toml_compat.py is the shared loader (stdlib tomllib on >=3.11, tiny
# hand-rolled fallback on 3.10 where CI still runs).
sys.path.insert(0, str(TESTS))
from _toml_compat import load_toml  # noqa: E402


FAST_LANE_FILES = [
    "tests/test_aether_acceptance_evidence.py",
    "tests/test_install_smoke.py",
    "tests/test_license.py",
    "tests/test_api_contracts.py",
    "tests/test_solar_spa.py",
    "tests/test_capability_negotiation.py",
    "tests/test_budget_enforce.py",
    "tests/test_memory_budget_policy.py",
    "tests/test_device_init_failure.py",
    "tests/test_allocation_gate.py",
    "tests/test_dead_render_structure_gate.py",
    "tests/test_pipeline_validation_gate.py",
    "tests/test_degradation_behavior.py",
    "tests/test_certificate_verifier.py",
    "tests/test_render_certificate.py",
    "tests/test_render_certificate_contract.py",
    "tests/test_astro_ephemeris.py",
    "tests/test_determinism_matrix.py",
    "tests/test_no_silent_degradation.py",
    "tests/test_substratia_evidence_report.py",
]


def unrun_files() -> list[str]:
    """Return repo-relative files quarantined from the full profile."""
    if not UNRUN_TOML.exists():
        return []
    data = load_toml(UNRUN_TOML)
    return [str(entry["file"]) for entry in data.get("entries", [])]


def _all_test_files() -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for p in TESTS.glob("test_*.py"))


def _tracked_test_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "tests/test_*.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def full_lane_files() -> list[str]:
    """Every acceptance test file except the honest UNRUN quarantine.

    Reject working-only or missing files so a direct local invocation selects
    the same tracked suite as a clean CI checkout.
    """
    working = set(_all_test_files())
    tracked = set(_tracked_test_files())
    if working != tracked:
        raise RuntimeError(
            "tracked test inventory differs from the working tree: "
            f"untracked={sorted(working - tracked)}, missing={sorted(tracked - working)}"
        )
    unrun = set(unrun_files())
    return sorted(tracked - unrun)


def fast_lane_files() -> list[str]:
    """Focused routine checks for permanent contracts and CPU acceptance gates."""
    missing = [path for path in FAST_LANE_FILES if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"fast CENSOR lane names missing tests: {missing}")
    return list(FAST_LANE_FILES)


def profile_files(profile: str) -> list[str]:
    if profile == "fast":
        return fast_lane_files()
    if profile == "full":
        return full_lane_files()
    raise ValueError(f"unknown CENSOR validation profile: {profile}")


def build_pytest_args(
    profile: str, passthrough: list[str], *, slow: bool = False
) -> list[str]:
    """Compose pytest argv from a profile, marker selection, and passthrough.

    We pass the file list explicitly rather than `tests/ --ignore=<file>`
    to make the lane's accounting directly inspectable and to prevent UNRUN
    files that fail at collection time from ever being imported.

    ``--slow-lane`` is private to this wrapper and valid only for the exhaustive
    full profile. It is removed before pytest sees argv.
    """
    forwarded = list(passthrough)
    if SLOW_LANE_SELECTOR in forwarded:
        slow = True
        forwarded.remove(SLOW_LANE_SELECTOR)
    if slow and profile != "full":
        raise ValueError("--slow-lane is valid only with --profile full")
    marker = (
        "slow and not interactive_viewer"
        if slow
        else "not slow and not interactive_viewer"
    )
    return [*profile_files(profile), "-m", marker, *forwarded]


def _github_escape(message: str) -> str:
    """Escape a string for GitHub workflow command annotations."""
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _parse_args(argv: list[str]) -> tuple[str, bool, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run an explicit CENSOR validation profile before pytest options."
    )
    parser.add_argument("--profile", choices=("fast", "full"), required=True)
    parser.add_argument(SLOW_LANE_SELECTOR, action="store_true")
    known, passthrough = parser.parse_known_args(argv)
    return known.profile, known.slow_lane, passthrough


def main(argv: list[str]) -> int:
    profile, slow, passthrough = _parse_args(argv)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *build_pytest_args(profile, passthrough, slow=slow),
    ]
    tail: deque[str] = deque(maxlen=180)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
    code = proc.wait()
    if code and environ.get("GITHUB_ACTIONS") == "true":
        message = "\n".join(tail)
        if len(message) > 3500:
            message = message[-3500:]
        print(
            f"::error title={profile.title()} Python lane failed::{_github_escape(message)}",
            flush=True,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
