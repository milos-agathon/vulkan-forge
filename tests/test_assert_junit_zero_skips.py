"""Adversarial fixtures for the required-lane JUnit verifier."""

import json
import os
from pathlib import Path
import sys

import pytest
import yaml

from scripts.assert_junit_zero_skips import JUnitValidationError, verify_junit
from scripts.summarize_m06_evidence import (
    build_summary,
    github_notice,
    main as summarize_m06_main,
    markdown_summary,
    write_summary,
)


def _contract_root() -> Path:
    return Path(
        os.environ.get(
            "FORGE3D_CI_CONTRACT_ROOT", Path(__file__).resolve().parents[1]
        )
    )


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "junit.xml"
    path.write_text(body, encoding="utf-8")
    return path


def test_zero_tests_rejected(tmp_path):
    path = _write(
        tmp_path, '<testsuite tests="0" failures="0" errors="0" skipped="0"/>'
    )
    with pytest.raises(JUnitValidationError, match="no tests"):
        verify_junit(path)


def test_one_clean_test_accepted(tmp_path):
    path = _write(
        tmp_path,
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="clean"/></testsuite>',
    )
    assert verify_junit(path).as_dict() == {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


@pytest.mark.parametrize(
    ("outcome", "counter"),
    [("failure", "failures"), ("error", "errors"), ("skipped", "skipped")],
)
def test_nonclean_outcome_rejected(tmp_path, outcome, counter):
    path = _write(
        tmp_path,
        f'<testsuite tests="1" failures="{int(counter == "failures")}" '
        f'errors="{int(counter == "errors")}" skipped="{int(counter == "skipped")}">'
        f'<testcase name="bad"><{outcome}/></testcase></testsuite>',
    )
    with pytest.raises(JUnitValidationError, match="not clean"):
        verify_junit(path)


def test_xfail_encoded_as_skip_rejected(tmp_path):
    path = _write(
        tmp_path,
        '<testsuite tests="1" failures="0" errors="0" skipped="1">'
        '<testcase name="xfail"><skipped type="pytest.xfail">expected</skipped>'
        "</testcase></testsuite>",
    )
    with pytest.raises(JUnitValidationError, match="zero-skip"):
        verify_junit(path)


@pytest.mark.parametrize("outcome", ["failure", "skipped"])
def test_clean_parent_cannot_hide_nested_nonclean_child(tmp_path, outcome):
    child_counter = "failures" if outcome == "failure" else "skipped"
    path = _write(
        tmp_path,
        '<testsuites tests="1" failures="0" errors="0" skipped="0">'
        '<testsuite name="parent" tests="1" failures="0" errors="0" skipped="0">'
        f'<testsuite name="child" tests="1" failures="{int(child_counter == "failures")}" '
        f'errors="0" skipped="{int(child_counter == "skipped")}">'
        f'<testcase name="nested"><{outcome}/></testcase>'
        "</testsuite></testsuite></testsuites>",
    )
    with pytest.raises(JUnitValidationError, match="contradictory"):
        verify_junit(path)


def test_nested_aggregate_totals_are_not_double_counted(tmp_path):
    path = _write(
        tmp_path,
        '<testsuites tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="parent" tests="2" failures="0" errors="0" skipped="0">'
        '<testsuite name="a" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="a"/></testsuite>'
        '<testsuite name="b" tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="b"/></testsuite>'
        "</testsuite></testsuites>",
    )
    assert verify_junit(path).tests == 2


def test_missing_file_rejected(tmp_path):
    with pytest.raises(JUnitValidationError, match="does not exist"):
        verify_junit(tmp_path / "missing.xml")


def test_malformed_xml_rejected(tmp_path):
    path = _write(tmp_path, '<testsuite tests="1"><testcase>')
    with pytest.raises(JUnitValidationError, match="malformed"):
        verify_junit(path)


@pytest.mark.parametrize("value", ["one", "-1"])
def test_invalid_counters_rejected(tmp_path, value):
    path = _write(
        tmp_path,
        f'<testsuite tests="{value}" failures="0" errors="0" skipped="0">'
        '<testcase name="x"/></testsuite>',
    )
    with pytest.raises(JUnitValidationError, match="counter"):
        verify_junit(path)


def test_declared_and_actual_totals_must_agree(tmp_path):
    path = _write(
        tmp_path,
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase name="only"/></testsuite>',
    )
    with pytest.raises(JUnitValidationError, match="contradictory tests"):
        verify_junit(path)


def test_m06_evidence_summary_extracts_adapter_and_junit_counts(tmp_path):
    (tmp_path / "run-context.json").write_text(
        json.dumps(
            {
                "repository": "milos-agathon/forge3d",
                "head_sha": "abc123",
                "run_id": "42",
                "run_attempt": "1",
                "runner_name": "forge3d-rtx3070",
                "runner_os": "Windows",
                "runner_arch": "X64",
                "required_labels": [
                    "self-hosted",
                    "Windows",
                    "forge3d-gpu",
                    "gpu-nvidia",
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "checked-out-head.txt").write_text("abc123\n", encoding="utf-8")
    (tmp_path / "adapter-probe.json").write_text(
        json.dumps(
            {
                "requested_backend": "vulkan",
                "probe": {
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": 0x10DE,
                    "device": 1234,
                    "backend": "Vulkan",
                    "device_type": "DiscreteGpu",
                    "driver": "test-driver",
                    "driver_info": "test-info",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "viewer-adapter.json").write_text(
        json.dumps(
            {
                "validated_identity": {
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": 0x10DE,
                    "device": 1234,
                    "backend": "vulkan",
                    "device_type": "discretegpu",
                    "driver": "test-driver",
                    "driver_info": "test-info",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "junit.xml").write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="a" name="one"/>'
        '<testcase classname="a" name="two"/>'
        "</testsuite>",
        encoding="utf-8",
    )

    summary = write_summary(tmp_path)

    assert summary["status"] == "pass"
    assert summary["exact_head"] is True
    assert summary["adapter"]["vendor_hex"] == "0x10de"
    assert summary["adapter"]["backend"] == "vulkan"
    assert summary["junit"] == {
        "exists": True,
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "zero_skip_clean": True,
    }
    assert (tmp_path / "m06-public-evidence-summary.json").is_file()
    rendered = markdown_summary(build_summary(tmp_path))
    assert "tests=2 failures=0 errors=0 skipped=0" in rendered
    assert "vendor=0x10de backend=vulkan" in rendered
    annotation = github_notice(summary)
    assert annotation.startswith("::notice title=M-06 exact-head evidence::")
    assert "head_sha=abc123" in annotation
    assert "checked_out_head=abc123 exact_head=true" in annotation
    assert "adapter=NVIDIA GeForce RTX 3070" in annotation
    assert "tests=2 failures=0 errors=0 skipped=0" in annotation


def test_m06_evidence_summary_fails_closed_on_synthetic_merge_checkout(
    tmp_path, monkeypatch
):
    (tmp_path / "run-context.json").write_text(
        json.dumps({"head_sha": "pr-head"}), encoding="utf-8"
    )
    (tmp_path / "checked-out-head.txt").write_text(
        "synthetic-merge\n", encoding="utf-8"
    )
    (tmp_path / "adapter-probe.json").write_text(
        json.dumps(
            {
                "probe": {
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": 0x10DE,
                    "backend": "vulkan",
                    "device_type": "discretegpu",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "junit.xml").write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase classname="m06" name="acceptance"/>'
        "</testsuite>",
        encoding="utf-8",
    )

    summary = build_summary(tmp_path)

    assert summary["exact_head"] is False
    assert summary["status"] == "incomplete"
    assert "exact_head=false" in github_notice(summary)
    monkeypatch.setattr(sys, "argv", ["summarize_m06_evidence.py", str(tmp_path)])
    assert summarize_m06_main() == 1


def _checkout_refs(workflow_text: str) -> list[str | None]:
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    refs: list[str | None] = []
    for job in workflow.get("jobs", {}).values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []):
            if step.get("uses") == "actions/checkout@v4":
                refs.append(step.get("with", {}).get("ref"))
    return refs


def test_ci_checkout_steps_pin_pull_requests_to_the_exact_head():
    root = _contract_root()
    pr_head_ref = "${{ github.event.pull_request.head.sha || github.sha }}"
    reusable_ref = "${{ inputs.ref }}"
    # Semantic discovery avoids the old brittle checkout-count assertion: adding
    # a properly pinned job must not break every Python lane, while an unpinned
    # checkout in any PR-reachable reusable workflow must still fail preflight.
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ci_jobs = yaml.load(ci, Loader=yaml.BaseLoader)["jobs"]
    preflight_refs = _checkout_refs(
        yaml.dump({"jobs": {"preflight": ci_jobs["preflight"]}})
    )
    live_base_ref = (
        "${{ github.event_name == 'pull_request' && "
        "format('refs/heads/{0}', github.base_ref) || github.sha }}"
    )
    trusted_base_ref = "${{ steps.policy-base.outputs.sha }}"
    assert preflight_refs == [live_base_ref, trusted_base_ref]
    for job_name, job in ci_jobs.items():
        if job_name == "preflight" or not isinstance(job, dict):
            continue
        for index, checkout_ref in enumerate(
            _checkout_refs(yaml.dump({"jobs": {job_name: job}})), start=1
        ):
            assert checkout_ref == pr_head_ref, (
                f"ci.yml:{job_name} checkout step {index} is not exact-head pinned"
            )

    for name in ("build-wheel.yml", "test-python-wheel.yml", "determinism-matrix.yml"):
        workflow = (root / ".github" / "workflows" / name).read_text(encoding="utf-8")
        checkout_refs = _checkout_refs(workflow)
        assert checkout_refs, f"{name} has no checkout provenance to verify"
        for index, checkout_ref in enumerate(checkout_refs, start=1):
            assert checkout_ref == reusable_ref, (
                f"{name} checkout step {index} is not exact-head pinned"
            )

    for reusable in ("build-wheel.yml", "test-python-wheel.yml", "determinism-matrix.yml"):
        callers = [
            job
            for job in ci_jobs.values()
            if isinstance(job, dict)
            and job.get("uses") == f"./.github/workflows/{reusable}"
        ]
        assert callers, f"CI does not call {reusable}"
        for caller in callers:
            assert caller.get("with", {}).get("ref") == pr_head_ref

    certificate = (
        root / ".github" / "workflows" / "certificate-refresh.yml"
    ).read_text(encoding="utf-8")
    assert _checkout_refs(certificate) == ["${{ github.sha }}"]


def test_preflight_uses_evaluated_state_without_weakening_source_provenance():
    root = _contract_root()
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    jobs = yaml.load(workflow, Loader=yaml.BaseLoader)["jobs"]
    preflight = jobs["preflight"]
    checkout_refs = _checkout_refs(yaml.dump({"jobs": {"preflight": preflight}}))
    live_base_ref = (
        "${{ github.event_name == 'pull_request' && "
        "format('refs/heads/{0}', github.base_ref) || github.sha }}"
    )
    trusted_base_ref = "${{ steps.policy-base.outputs.sha }}"
    assert checkout_refs == [live_base_ref, trusted_base_ref]
    if checkout_refs == ["${{ github.sha }}"]:
        run_blocks = "\n".join(
            step.get("run", "")
            for step in preflight["steps"]
            if isinstance(step, dict)
        )
        assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in run_blocks
    else:
        assert checkout_refs == [live_base_ref, trusted_base_ref]
        steps = preflight["steps"]
        base_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Checkout live policy base"
        )
        manual_base_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Resolve protected policy base for manual acceptance"
        )
        snapshot_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Snapshot live policy base"
        )
        trusted_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Checkout trusted CI contracts"
        )
        merge_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Materialize current-base candidate tree"
        )
        manual_merge_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Materialize manual-dispatch candidate tree"
        )
        setup_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses") == "actions/setup-python@v5"
        )
        validate_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Validate workflow and cost-control contracts"
        )
        assert (
            base_index
            < manual_base_index
            < snapshot_index
            < trusted_index
            < merge_index
            < manual_merge_index
            < setup_index
            < validate_index
        )
        base_checkout = steps[base_index]
        assert base_checkout["with"] == {
            "ref": live_base_ref,
            "fetch-depth": "0",
        }
        manual_base_step = steps[manual_base_index]
        assert manual_base_step["if"] == "github.event_name == 'workflow_dispatch'"
        assert manual_base_step["env"] == {
            "POLICY_BRANCH": "${{ github.event.repository.default_branch }}"
        }
        assert 'refs/remotes/origin/${POLICY_BRANCH}' in manual_base_step["run"]
        snapshot_step = steps[snapshot_index]
        assert snapshot_step["id"] == "policy-base"
        assert 'sha=$(git rev-parse HEAD)' in snapshot_step["run"]
        trusted_checkout = steps[trusted_index]
        assert trusted_checkout["with"] == {
            "ref": trusted_base_ref,
            "path": ".ci-contracts",
        }
        merge_step = steps[merge_index]
        assert merge_step["if"] == "github.event_name == 'pull_request'"
        assert merge_step["env"] == {
            "POLICY_BASE_SHA": trusted_base_ref,
            "PR_NUMBER": "${{ github.event.pull_request.number }}",
            "PR_HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
        }
        merge_run = merge_step["run"]
        assert 'test "$(git rev-parse HEAD)" = "$POLICY_BASE_SHA"' in merge_run
        assert (
            "+refs/pull/${PR_NUMBER}/head:refs/remotes/pull/${PR_NUMBER}/head"
            in merge_run
        )
        assert (
            'test "$(git rev-parse refs/remotes/pull/${PR_NUMBER}/head)" = '
            '"$PR_HEAD_SHA"' in merge_run
        )
        assert 'git config --local user.name "github-actions[bot]"' in merge_run
        assert (
            'git config --local user.email '
            '"41898282+github-actions[bot]@users.noreply.github.com"' in merge_run
        )
        assert 'git merge --no-commit --no-ff "$PR_HEAD_SHA"' in merge_run
        assert 'test "$(git rev-parse HEAD)" = "$POLICY_BASE_SHA"' in merge_run
        assert 'test "$(git rev-parse MERGE_HEAD)" = "$PR_HEAD_SHA"' in merge_run
        manual_merge_step = steps[manual_merge_index]
        assert manual_merge_step["if"] == "github.event_name == 'workflow_dispatch'"
        assert manual_merge_step["env"] == {
            "POLICY_BASE_SHA": trusted_base_ref,
            "CANDIDATE_HEAD_SHA": "${{ github.sha }}",
        }
        manual_merge_run = manual_merge_step["run"]
        assert 'git fetch --no-tags origin "$CANDIDATE_HEAD_SHA"' in manual_merge_run
        assert 'test "$(git rev-parse FETCH_HEAD)" = "$CANDIDATE_HEAD_SHA"' in manual_merge_run
        assert 'git merge --no-commit --no-ff "$CANDIDATE_HEAD_SHA"' in manual_merge_run
        assert 'test "$(git rev-parse MERGE_HEAD)" = "$CANDIDATE_HEAD_SHA"' in manual_merge_run
        validate_step = steps[validate_index]
        assert validate_step["working-directory"] == ".ci-contracts"
        assert validate_step["env"] == {
            "FORGE3D_NO_BOOTSTRAP": "1",
            "FORGE3D_CI_CONTRACT_ROOT": "${{ github.workspace }}",
            "PYTHONPATH": "${{ github.workspace }}/.ci-contracts",
        }

    pr_head_ref = "${{ github.event.pull_request.head.sha || github.sha }}"
    for job_name, job in jobs.items():
        if job_name == "preflight" or not isinstance(job, dict):
            continue
        for checkout_ref in _checkout_refs(yaml.dump({"jobs": {job_name: job}})):
            assert checkout_ref == pr_head_ref
        if isinstance(job.get("uses"), str) and job["uses"].startswith(
            "./.github/workflows/"
        ):
            assert job.get("with", {}).get("ref") == pr_head_ref


def test_checkout_contract_cannot_be_satisfied_by_a_comment_or_sibling_key():
    deceptive = """
name: deceptive
on:
  pull_request:
jobs:
  bad:
    runs-on: ubuntu-latest
    steps:
      # ref: ${{ github.event.pull_request.head.sha || github.sha }}
      - uses: actions/checkout@v4
      - name: "ref: ${{ github.event.pull_request.head.sha || github.sha }}"
        run: echo not-a-checkout-ref
"""
    assert _checkout_refs(deceptive) == [None]
