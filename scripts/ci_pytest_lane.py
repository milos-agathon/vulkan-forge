#!/usr/bin/env python
# scripts/ci_pytest_lane.py
# CENSOR validation profiles. The fast profile protects architectural truth on
# routine changes. The full profile retains main's split non-slow/slow lanes for
# explicit acceptance and release validation.
# RELEVANT FILES: tests/UNRUN.toml, tests/_toml_compat.py, .github/workflows/ci.yml
"""Run a focused or full CENSOR pytest profile, forwarding pytest arguments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from os import environ
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
UNRUN_TOML = TESTS / "UNRUN.toml"
SLOW_LANE_SELECTOR = "--slow-lane"
SELECTION_LEDGER_OPTION = "--selection-ledger"
ZERO_SKIP_ENV = "FORGE3D_GENERIC_FULL_ZERO_SKIP"
LEDGER_ENV = "FORGE3D_PYTEST_SELECTION_LEDGER"
DEDICATED_LANE_MARKERS = (
    "recipe_golden",
    "anamnesis_physical",
    "sidera_vulkan",
    "cross_backend",
    "nvidia_vulkan",
    "limes_physical",
    "helios_physical",
    "gpu_lane",
    "f3dz_physical",
    "apple_metal_physical",
)

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
    marker += "".join(f" and not {name}" for name in DEDICATED_LANE_MARKERS)
    return [*profile_files(profile), "-m", marker, *forwarded]


def _github_escape(message: str) -> str:
    """Escape a string for GitHub workflow command annotations."""
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _parse_args(argv: list[str]) -> tuple[str, bool, Path | None, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run an explicit CENSOR validation profile before pytest options."
    )
    parser.add_argument("--profile", choices=("fast", "full"), required=True)
    parser.add_argument(SLOW_LANE_SELECTOR, action="store_true")
    parser.add_argument(SELECTION_LEDGER_OPTION, type=Path)
    known, passthrough = parser.parse_known_args(argv)
    return known.profile, known.slow_lane, known.selection_ledger, passthrough


_COLLECTION_SKIPS: list[str] = []
_DESELECTED: list[dict[str, object]] = []


def _zero_skip_enabled() -> bool:
    return environ.get(ZERO_SKIP_ENV) == "1"


def pytest_collectreport(report) -> None:
    if _zero_skip_enabled() and report.skipped:
        _COLLECTION_SKIPS.append(f"{report.nodeid}: {report.longrepr}")


def _ledger_record(item) -> dict[str, object]:
    return {
        "nodeid": item.nodeid,
        "markers": sorted({marker.name for marker in item.iter_markers()}),
    }


def pytest_deselected(items) -> None:
    if _zero_skip_enabled():
        _DESELECTED.extend(_ledger_record(item) for item in items)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items) -> None:
    if not _zero_skip_enabled():
        return
    forbidden = []
    ledger = []
    for item in items:
        ledger.append(_ledger_record(item))
        if item.get_closest_marker("skip") is not None:
            forbidden.append(f"{item.nodeid}: skip")
        if item.get_closest_marker("xfail") is not None:
            forbidden.append(f"{item.nodeid}: xfail")
        for marker in item.iter_markers("skipif"):
            if marker.args and bool(marker.args[0]):
                forbidden.append(f"{item.nodeid}: active skipif")
    ledger_path = environ.get(LEDGER_ENV)
    if ledger_path:
        output = Path(ledger_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema": "forge3d.pytest-selection.v1",
                    "selected": ledger,
                    "deselected": _DESELECTED,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if forbidden:
        raise pytest.UsageError(
            "generic full Python lane selected skip/xfail tests:\n"
            + "\n".join(forbidden)
        )


def pytest_collection_finish(session) -> None:
    del session
    if _zero_skip_enabled() and _COLLECTION_SKIPS:
        raise pytest.UsageError(
            "generic full Python lane skipped test modules during collection:\n"
            + "\n".join(_COLLECTION_SKIPS)
        )


def main(argv: list[str]) -> int:
    profile, slow, selection_ledger, passthrough = _parse_args(argv)
    child_env = environ.copy()
    if profile == "full":
        child_env[ZERO_SKIP_ENV] = "1"
        if selection_ledger is not None:
            child_env[LEDGER_ENV] = str(selection_ledger)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "scripts.ci_pytest_lane",
        *build_pytest_args(profile, passthrough, slow=slow),
    ]
    tail: deque[str] = deque(maxlen=180)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        env=child_env,
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
