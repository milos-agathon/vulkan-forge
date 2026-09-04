"""Fail-closed contracts for CARTOGRAPHER-PRIME hosted-runner evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "labels" / "optimal_plan_hash.json"
WORKFLOW = ROOT / ".github" / "workflows" / "cartographer-prime.yml"
EVIDENCE_PATH = ROOT / ".github" / "scripts" / "cartographer_prime_evidence.py"
VERIFIER_PATH = (
    ROOT / ".github" / "scripts" / "verify_cartographer_prime_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_cartographer_prime_evidence", VERIFIER_PATH
)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
EvidenceError = VERIFIER.EvidenceError
verify_evidence = VERIFIER.verify_evidence
EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "cartographer_prime_evidence", EVIDENCE_PATH
)
assert EVIDENCE_SPEC and EVIDENCE_SPEC.loader
EVIDENCE = importlib.util.module_from_spec(EVIDENCE_SPEC)
EVIDENCE_SPEC.loader.exec_module(EVIDENCE)
SHA = "a" * 40
PLAN_HASH = "8ad1f1330b00f9752ac6cffeec7de36ab2017ccbf0de907832096f42862bacaf"
RUST_RELEASE = "1.90.0"
RUST_COMMIT = "1" * 40
RUST_DATE = "2025-09-14"
LLVM_VERSION = "20.1.8"
RUNNERS = {
    "ubuntu-x64": ("Linux", "X64"),
    "macos-arm64": ("macOS", "ARM64"),
    "windows-x64": ("Windows", "X64"),
}
RUST_HOSTS = {
    "ubuntu-x64": "x86_64-unknown-linux-gnu",
    "macos-arm64": "aarch64-apple-darwin",
    "windows-x64": "x86_64-pc-windows-msvc",
}


def _rustc(
    config: str,
    *,
    release: str = RUST_RELEASE,
    commit: str = RUST_COMMIT,
    date: str = RUST_DATE,
    host: str | None = None,
    llvm: str = LLVM_VERSION,
) -> str:
    return "\n".join(
        (
            f"rustc {release} ({commit[:9]} {date})",
            "binary: rustc",
            f"commit-hash: {commit}",
            f"commit-date: {date}",
            f"host: {host or RUST_HOSTS[config]}",
            f"release: {release}",
            f"LLVM version: {llvm}",
        )
    )


def _evidence(config: str) -> dict:
    runner_os, runner_arch = RUNNERS[config]
    return {
        "schema": "forge3d.cartographer-prime.runner-evidence.v1",
        "git_sha": SHA,
        "runner": {"config": config, "os": runner_os, "arch": runner_arch},
        "runtime": {
            "python": "3.11.9",
            "python_implementation": "CPython",
            "rustc": _rustc(config),
            "build_profile": "release-lto",
        },
        "metrics": {
            "plan_hash": PLAN_HASH,
            "optimality_gap": 0.0,
            "occluded_placement_count": 0,
            "tested_instance_count": 5,
        },
        "tests": {"tests": 16, "failures": 0, "errors": 0, "skipped": 0},
        "evidence_scope": "hosted-runner-config",
    }


def _write_artifacts(root: Path, payloads: dict[str, dict] | None = None) -> None:
    source = payloads or {key: _evidence(key) for key in RUNNERS}
    for config, payload in source.items():
        folder = root / config
        folder.mkdir(parents=True)
        (folder / "cartographer-prime-evidence.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_evidence_assembler_binds_clean_measurements_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measurements = tmp_path / "measurements.json"
    measurements.write_text(
        json.dumps(
            {
                "schema": "forge3d.cartographer-prime.measurements.v1",
                "plan_hash": PLAN_HASH,
                "optimality_gap": 0.0,
                "occluded_placement_count": 0,
                "tested_instance_count": 5,
            }
        ),
        encoding="utf-8",
    )

    class CleanCounts:
        @staticmethod
        def as_dict() -> dict[str, int]:
            return {"tests": 16, "failures": 0, "errors": 0, "skipped": 0}

    def command(*args: str) -> str:
        if args[:3] == ("git", "-C", str(ROOT)):
            return SHA
        if args == ("rustc", "--version", "--verbose"):
            return _rustc("macos-arm64")
        raise AssertionError(args)

    monkeypatch.setattr(EVIDENCE, "_command", command)
    monkeypatch.setattr(EVIDENCE, "verify_junit", lambda _path: CleanCounts())
    monkeypatch.setattr(EVIDENCE.platform, "python_version", lambda: "3.11.9")
    monkeypatch.setattr(
        EVIDENCE.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(EVIDENCE.sys, "version_info", (3, 11, 9))
    payload = EVIDENCE.build_evidence(
        measurements_path=measurements,
        junit_path=tmp_path / "junit.xml",
        expected_sha=SHA,
        runner_config="macos-arm64",
        runner_os="macOS",
        runner_arch="ARM64",
        build_profile="release-lto",
        repository=ROOT,
    )
    assert payload["git_sha"] == SHA
    assert payload["metrics"]["plan_hash"] == PLAN_HASH
    assert payload["runtime"] == {
        "python": "3.11.9",
        "python_implementation": "CPython",
        "rustc": _rustc("macos-arm64"),
        "build_profile": "release-lto",
    }
    assert payload["tests"]["skipped"] == 0


def test_verifier_accepts_exact_three_runner_evidence(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    result = verify_evidence(tmp_path, GOLDEN, SHA)
    assert result == {
        "schema": "forge3d.cartographer-prime.verification.v1",
        "git_sha": SHA,
        "runner_configs": ["macos-arm64", "ubuntu-x64", "windows-x64"],
        "python_versions": {
            "macos-arm64": "3.11.9",
            "ubuntu-x64": "3.11.9",
            "windows-x64": "3.11.9",
        },
        "plan_hash": PLAN_HASH,
        "maximum_optimality_gap": 0.0,
        "occluded_placement_count": 0,
        "evidence_scope": "three-hosted-runner-configs",
    }


def test_verifier_rejects_missing_runner(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS if key != "macos-arm64"}
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match="missing runner configs: macos-arm64"):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_wrong_sha(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["windows-x64"]["git_sha"] = "b" * 40
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match="wrong SHA for windows-x64"):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_schema_tamper(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["ubuntu-x64"]["schema"] = "forge3d.cartographer-prime.runner-evidence.v0"
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match="evidence schema mismatch"):
        verify_evidence(tmp_path, GOLDEN, SHA)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("python", "3.12.1", "wrong Python runtime"),
        ("python", "3.11", "wrong Python runtime"),
        ("python", "3.11.latest", "wrong Python runtime"),
        ("python", "03.11.9", "wrong Python runtime"),
        ("python", "3.11.1١", "wrong Python runtime"),
        ("python_implementation", "PyPy", "wrong Python runtime"),
        ("build_profile", "dev", "wrong build profile"),
        ("rustc", "rustc 1.90.0", "invalid rustc provenance"),
    ],
)
def test_verifier_rejects_wrong_runtime_provenance(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["ubuntu-x64"]["runtime"][field] = value
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match=message):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_rustc_host_mismatch(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["ubuntu-x64"]["runtime"]["rustc"] = _rustc(
        "ubuntu-x64", host="x86_64-unknown-linux-musl"
    )
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match="wrong rustc host for ubuntu-x64"):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_forged_llvm_version(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["ubuntu-x64"]["runtime"]["rustc"] = _rustc(
        "ubuntu-x64", llvm="forged"
    )
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(
        EvidenceError, match="invalid rustc LLVM version for ubuntu-x64"
    ):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_cross_runner_llvm_drift(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["macos-arm64"]["runtime"]["rustc"] = _rustc(
        "macos-arm64", llvm="99.0.0"
    )
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(
        EvidenceError, match="cross-runner rust_llvm_version provenance disagrees"
    ):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_cross_runner_rustc_drift(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["macos-arm64"]["runtime"]["rustc"] = _rustc(
        "macos-arm64",
        release="1.91.0",
        commit="2" * 40,
        date="2025-10-30",
    )
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(
        EvidenceError, match="cross-runner rust_release provenance disagrees"
    ):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_accepts_cross_runner_python_patch_drift(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["ubuntu-x64"]["runtime"]["python"] = "3.11.15"
    _write_artifacts(tmp_path, payloads)
    result = verify_evidence(tmp_path, GOLDEN, SHA)
    assert result["python_versions"] == {
        "macos-arm64": "3.11.9",
        "ubuntu-x64": "3.11.15",
        "windows-x64": "3.11.9",
    }


def test_verifier_rejects_wrong_exact_test_count(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["windows-x64"]["tests"]["tests"] = 1
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match="wrong test count for windows-x64"):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_wrong_exact_tested_instance_count(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["macos-arm64"]["metrics"]["tested_instance_count"] = 1
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(
        EvidenceError, match="wrong tested instance count for macos-arm64"
    ):
        verify_evidence(tmp_path, GOLDEN, SHA)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimality_gap", 0.021, "optimality gap failure"),
        ("optimality_gap", float("nan"), "optimality gap failure"),
        ("occluded_placement_count", 1, "occluded placement failure"),
        ("occluded_placement_count", False, "occluded placement failure"),
    ],
)
def test_verifier_rejects_wrong_metric(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["ubuntu-x64"]["metrics"][field] = value
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match=message):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_tampered_hash_disagreement(tmp_path: Path) -> None:
    payloads = {key: _evidence(key) for key in RUNNERS}
    payloads["macos-arm64"]["metrics"]["plan_hash"] = "b" * 64
    _write_artifacts(tmp_path, payloads)
    with pytest.raises(EvidenceError, match="runner plan hashes disagree"):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_verifier_rejects_duplicate_config(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    duplicate_payload = copy.deepcopy(_evidence("ubuntu-x64"))
    (duplicate / "cartographer-prime-evidence.json").write_text(
        json.dumps(duplicate_payload), encoding="utf-8"
    )
    with pytest.raises(EvidenceError, match="duplicate runner config: ubuntu-x64"):
        verify_evidence(tmp_path, GOLDEN, SHA)


def test_workflow_builds_native_gate_and_verifies_exact_head() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    jobs = parsed["jobs"]
    runner = jobs["runner-evidence"]
    assert runner["strategy"]["matrix"]["include"] == [
        {"config": "ubuntu-x64", "runner": "ubuntu-latest"},
        {"config": "macos-arm64", "runner": "macos-15"},
        {"config": "windows-x64", "runner": "windows-latest"},
    ]
    runner_text = workflow.split("  runner-evidence:", 1)[1].split(
        "\n  verify-three-runners:", 1
    )[0]
    expected_ref = "${{ github.event.pull_request.head.sha || github.sha }}"
    assert f"ref: {expected_ref}" in runner_text
    assert 'test "$(git rev-parse HEAD)" = "$CARTOGRAPHER_EXPECTED_SHA"' in runner_text
    assert "python -m maturin build --profile release-lto --out wheelhouse" in runner_text
    assert runner["env"]["FORGE3D_NO_BOOTSTRAP"] == "1"
    assert runner["env"]["FORGE3D_TEST_INSTALLED_WHEEL"] == "1"
    assert "tests/test_label_optimal_solver.py" in runner_text
    assert "scripts/assert_junit_zero_skips.py" in runner_text
    assert ".github/scripts/cartographer_prime_evidence.py" in runner_text
    evidence_step = next(
        step
        for step in runner["steps"]
        if step.get("name") == "Bind measurements to runner and exact head"
    )
    assert evidence_step["shell"] == "bash"
    assert "--build-profile release-lto" in evidence_step["run"]
    verifier = jobs["verify-three-runners"]
    assert verifier["needs"] == "runner-evidence"
    assert verifier["if"] == "always()"
    checkout = verifier["steps"][0]
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["if"] == "always()"
    download = next(
        step
        for step in verifier["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download["if"] == "always()"
    verification_step = next(
        step
        for step in verifier["steps"]
        if step.get("name") == "Reject incomplete or inconsistent evidence"
    )
    assert verification_step["if"] == "always()"
    result_guard = verifier["steps"][-1]
    assert result_guard["name"] == "Require every runner evidence leg to succeed"
    assert result_guard["if"] == "always()"
    assert result_guard == next(
        step
        for step in verifier["steps"]
        if step.get("name") == "Require every runner evidence leg to succeed"
    )
    assert result_guard["shell"] == "bash"
    assert "needs.runner-evidence.result" in result_guard["run"]
    assert "= success" in result_guard["run"]
    assert ".github/scripts/verify_cartographer_prime_evidence.py" in workflow
    assert "retention-days:" not in workflow
    assert "physical" not in workflow.lower()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert golden["expected_test_count"] == 16
    assert golden["expected_tested_instance_count"] == 5
    assert {
        row["id"]: row["rust_host"] for row in golden["required_runner_configs"]
    } == RUST_HOSTS
