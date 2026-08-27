from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    "ci.yml",
    "determinism-matrix.yml",
    "build-wheel.yml",
    "test-python-wheel.yml",
)
BINDER = "./.github/actions/bind-artifact-head"
EXPECTED_SHA = {
    "ci.yml": "${{ github.event.pull_request.head.sha || github.sha }}",
    "determinism-matrix.yml": "${{ inputs.ref }}",
    "build-wheel.yml": "${{ inputs.ref }}",
    "test-python-wheel.yml": "${{ inputs.ref }}",
}


def _yaml(path: Path) -> dict:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(data, dict)
    return data


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _normalize_condition(value: str) -> str:
    value = "".join(value.split())
    if value.startswith("${{") and value.endswith("}}"):
        value = value[3:-2]
    return value


def _marker_is_selected(marker: str, upload_paths: list[str]) -> bool:
    if marker in upload_paths:
        return True
    return any(
        (path.endswith("/") and marker.startswith(path))
        or fnmatch.fnmatchcase(marker, path)
        for path in upload_paths
    )


def test_every_reachable_upload_is_bound_to_the_exact_checked_out_head() -> None:
    violations = []
    for workflow_name in WORKFLOWS:
        jobs = _yaml(ROOT / ".github" / "workflows" / workflow_name)["jobs"]
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            steps = job.get("steps") or []
            for index, upload in enumerate(steps):
                if upload.get("uses") != "actions/upload-artifact@v4":
                    continue
                artifact = (upload.get("with") or {}).get("name", "<unnamed>")
                label = f"{workflow_name}:{job_name}:{artifact}"
                if index == 0 or steps[index - 1].get("uses") != BINDER:
                    violations.append(f"{label}: missing immediately preceding binder")
                    continue

                binder = steps[index - 1]
                binder_id = binder.get("id")
                if not binder_id:
                    violations.append(f"{label}: binder has no id")
                    continue
                binder_with = binder.get("with") or {}
                upload_with = upload.get("with") or {}
                if binder_with.get("expected-sha") != EXPECTED_SHA[workflow_name]:
                    violations.append(f"{label}: wrong expected SHA expression")

                original_paths = _lines(binder_with.get("artifact-paths", ""))
                upload_paths = _lines(upload_with.get("path", ""))
                marker = binder_with.get("output-path", "")
                if not original_paths or any(path not in upload_paths for path in original_paths):
                    violations.append(f"{label}: upload dropped an original evidence path")
                if not marker or not _marker_is_selected(marker, upload_paths):
                    violations.append(f"{label}: upload does not select the SHA marker")

                upload_policy = upload_with.get("if-no-files-found", "warn")
                if binder_with.get("if-no-files-found", "warn") != upload_policy:
                    violations.append(f"{label}: no-files policy drift")

                bound_clause = f"steps.{binder_id}.outputs.bound=='true'"
                upload_condition = _normalize_condition(upload.get("if", ""))
                binder_condition = _normalize_condition(binder.get("if", ""))
                expected_conditions = (
                    {
                        f"({binder_condition})&&{bound_clause}",
                        f"{binder_condition}&&{bound_clause}",
                    }
                    if binder_condition
                    else {bound_clause}
                )
                if upload_condition not in expected_conditions:
                    violations.append(
                        f"{label}: upload condition must preserve the binder condition "
                        "and require its bound output"
                    )

    assert not violations, "\n".join(violations)


def _action_script() -> str:
    action = _yaml(ROOT / ".github" / "actions" / "bind-artifact-head" / "action.yml")
    assert action["runs"]["using"] == "composite"
    steps = action["runs"]["steps"]
    assert len(steps) == 1
    return steps[0]["run"]


def _git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Artifact Test",
            "-c",
            "user.email=artifact@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo, sha


def _run_action(
    repo: Path,
    sha: str,
    *,
    paths: str,
    output: str,
    policy: str,
    expected: str | None = None,
) -> subprocess.CompletedProcess[str]:
    github_output = repo / "github-output.txt"
    env = os.environ.copy()
    env.update(
        EXPECTED_SHA=expected or sha,
        ARTIFACT_PATHS=paths,
        OUTPUT_PATH=output,
        IF_NO_FILES_FOUND=policy,
        GITHUB_OUTPUT=str(github_output),
    )
    return subprocess.run(
        ["bash", "-c", _action_script()],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )


def test_binder_writes_full_sha_only_for_existing_content(tmp_path: Path) -> None:
    repo, sha = _git_repo(tmp_path)
    evidence = repo / "evidence"
    evidence.mkdir()
    (evidence / "result.ABSENT").write_text("no adapter\n", encoding="utf-8")

    result = _run_action(
        repo,
        sha,
        paths="evidence/",
        output="evidence/checked-out-head.txt",
        policy="error",
    )

    assert result.returncode == 0, result.stderr
    assert (evidence / "checked-out-head.txt").read_text(encoding="utf-8") == f"{sha}\n"
    assert (repo / "github-output.txt").read_text(encoding="utf-8") == "bound=true\n"


@pytest.mark.parametrize("policy, returncode", [("error", 1), ("warn", 0), ("ignore", 0)])
def test_binder_preserves_no_content_policy_without_manufacturing_an_artifact(
    tmp_path: Path, policy: str, returncode: int
) -> None:
    repo, sha = _git_repo(tmp_path)

    result = _run_action(
        repo,
        sha,
        paths="missing/",
        output="missing/checked-out-head.txt",
        policy=policy,
    )

    assert result.returncode == returncode
    assert not (repo / "missing").exists()
    assert (repo / "github-output.txt").read_text(encoding="utf-8") == "bound=false\n"


@pytest.mark.parametrize("expected", ["a" * 12, "A" * 40, "f" * 40])
def test_binder_rejects_truncated_nonlowercase_and_wrong_heads(
    tmp_path: Path, expected: str
) -> None:
    repo, sha = _git_repo(tmp_path)
    (repo / "evidence").mkdir()
    (repo / "evidence" / "result.json").write_text("{}\n", encoding="utf-8")

    result = _run_action(
        repo,
        sha,
        paths="evidence/",
        output="evidence/checked-out-head.txt",
        policy="error",
        expected=expected,
    )

    assert result.returncode != 0
    assert not (repo / "evidence" / "checked-out-head.txt").exists()


def test_binder_rejects_an_existing_mismatched_marker(tmp_path: Path) -> None:
    repo, sha = _git_repo(tmp_path)
    evidence = repo / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text("{}\n", encoding="utf-8")
    marker = evidence / "checked-out-head.txt"
    marker.write_text("0" * 40 + "\n", encoding="utf-8")

    result = _run_action(
        repo,
        sha,
        paths="evidence/",
        output="evidence/checked-out-head.txt",
        policy="error",
    )

    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "0" * 40 + "\n"


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_binder_accepts_one_exact_marker_line(
    tmp_path: Path, ending: str
) -> None:
    repo, sha = _git_repo(tmp_path)
    evidence = repo / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text("{}\n", encoding="utf-8")
    marker = evidence / "checked-out-head.txt"
    marker_bytes = f"{sha}{ending}".encode()
    marker.write_bytes(marker_bytes)

    result = _run_action(
        repo,
        sha,
        paths="evidence/",
        output="evidence/checked-out-head.txt",
        policy="error",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_bytes() == marker_bytes
    assert (repo / "github-output.txt").read_text(encoding="utf-8") == "bound=true\n"


@pytest.mark.parametrize("invalid_ending", ["\n\n", "\r"])
def test_binder_rejects_invalid_marker_terminators(
    tmp_path: Path, invalid_ending: str
) -> None:
    repo, sha = _git_repo(tmp_path)
    evidence = repo / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text("{}\n", encoding="utf-8")
    marker = evidence / "checked-out-head.txt"
    marker_bytes = f"{sha}{invalid_ending}".encode()
    marker.write_bytes(marker_bytes)

    result = _run_action(
        repo,
        sha,
        paths="evidence/",
        output="evidence/checked-out-head.txt",
        policy="error",
    )

    assert result.returncode != 0
    assert "exactly one full SHA line" in result.stdout
    assert marker.read_bytes() == marker_bytes


def test_binder_rejects_a_sha_split_across_marker_lines(tmp_path: Path) -> None:
    repo, sha = _git_repo(tmp_path)
    evidence = repo / "evidence"
    evidence.mkdir()
    (evidence / "result.json").write_text("{}\n", encoding="utf-8")
    marker = evidence / "checked-out-head.txt"
    split_marker = f"{sha[:20]}\n{sha[20:]}\n"
    marker.write_text(split_marker, encoding="utf-8")

    result = _run_action(
        repo,
        sha,
        paths="evidence/",
        output="evidence/checked-out-head.txt",
        policy="error",
    )

    assert result.returncode != 0
    assert "exactly one full SHA line" in result.stdout
    assert marker.read_text(encoding="utf-8") == split_marker


def test_binder_does_not_treat_an_existing_marker_as_payload(tmp_path: Path) -> None:
    repo, sha = _git_repo(tmp_path)
    evidence = repo / "evidence"
    evidence.mkdir()
    marker = evidence / "checked-out-head.txt"
    marker.write_text(f"{sha}\n", encoding="utf-8")

    result = _run_action(
        repo,
        sha,
        paths="evidence/",
        output="evidence/checked-out-head.txt",
        policy="error",
    )

    assert result.returncode == 1
    assert marker.read_text(encoding="utf-8") == f"{sha}\n"
    assert (repo / "github-output.txt").read_text(encoding="utf-8") == "bound=false\n"
