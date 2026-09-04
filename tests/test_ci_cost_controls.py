from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

import yaml


ROOT = Path(
    os.environ.get("FORGE3D_CI_CONTRACT_ROOT", Path(__file__).resolve().parents[1])
)
UPLOAD_STEP = re.compile(
    r"(?ms)^      - (?:name|uses):.*?"
    r"(?=^      - (?:name|uses):|^  [A-Za-z0-9_-]+:|\Z)"
)


class _UniqueKeyLoader(yaml.BaseLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key ({key})",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _workflow_data(name: str) -> dict:
    data = yaml.load(_workflow(name), Loader=yaml.BaseLoader)
    assert isinstance(data, dict), f"{name} is not a workflow mapping"
    assert isinstance(data.get("jobs"), dict), f"{name} has no jobs mapping"
    return data


def test_paths_filter_literals_are_valid_yaml_without_duplicate_keys() -> None:
    workflow = _workflow("ci.yml")
    steps = re.findall(
        r"(?ms)^      - .*?uses: dorny/paths-filter@v3.*?(?=^      - |^  \S|\Z)",
        workflow,
    )
    assert steps, "ci.yml has no dorny/paths-filter steps"
    for step in steps:
        match = re.search(
            r"(?ms)^          filters: \|\n(?P<body>(?:^ {12}.*(?:\n|\Z))*)",
            step,
        )
        assert match, "dorny/paths-filter step has no literal filters mapping"
        filters = yaml.load(
            textwrap.dedent(match.group("body")), Loader=_UniqueKeyLoader
        )
        assert isinstance(filters, dict) and filters


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:.*?(?=^  [A-Za-z0-9_-]+:|\Z)",
        workflow,
    )
    assert match, f"missing workflow job: {name}"
    return match.group()


def _artifact_step(workflow: str, artifact: str) -> str:
    for step in _upload_steps(workflow):
        if re.search(rf"(?m)^\s+name: {re.escape(artifact)}$", step):
            return step
    raise AssertionError(f"missing artifact upload: {artifact}")


def _upload_steps(workflow: str) -> list[str]:
    return [
        step
        for step in UPLOAD_STEP.findall(workflow)
        if "uses: actions/upload-artifact@v4" in step
    ]


def test_ci_cost_controls_are_scoped_and_retained() -> None:
    workflow = _workflow("ci.yml")
    jobs = _workflow_data("ci.yml")["jobs"]
    tessella_enabled = "test-tessella-gpu" in jobs

    concurrency = re.search(r"(?ms)^concurrency:\n.*?(?=^env:)", workflow)
    assert concurrency
    body = concurrency.group()
    assert "format('pr-{0}', github.event.pull_request.number)" in body
    assert "format('manual-{0}', github.ref_name)" in body
    assert "format('run-{0}', github.run_id)" in body
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request'" in body

    trigger = workflow.split("\npermissions:", 1)[0]
    assert "types: [opened, synchronize, reopened, ready_for_review]" in trigger
    assert "labeled" not in trigger
    assert "unlabeled" not in trigger
    scopes = ["core", "full", "determinism", "m06", "f3dz", "anamnesis", "helios"]
    if tessella_enabled:
        scopes.append("tessella")
    for scope in scopes:
        assert f"          - {scope}" in trigger

    preflight = _job(workflow, "preflight")
    assert "name: Checkout trusted CI contracts" in preflight
    if "name: Checkout trusted CI contracts" in preflight:
        live_base_ref = (
            "ref: ${{ github.event_name == 'pull_request' && "
            "format('refs/heads/{0}', github.base_ref) || github.sha }}"
        )
        assert preflight.count(live_base_ref) == 1
        assert "name: Snapshot live policy base" in preflight
        assert "id: policy-base" in preflight
        assert 'echo "sha=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"' in preflight
        assert "name: Resolve protected policy base for manual acceptance" in preflight
        assert "POLICY_BRANCH: ${{ github.event.repository.default_branch }}" in preflight
        assert 'refs/remotes/origin/${POLICY_BRANCH}' in preflight
        assert "ref: ${{ steps.policy-base.outputs.sha }}" in preflight
        assert "path: .ci-contracts" in preflight
        assert "POLICY_BASE_SHA: ${{ steps.policy-base.outputs.sha }}" in preflight
        assert 'test "$(git rev-parse HEAD)" = "$POLICY_BASE_SHA"' in preflight
        assert 'git config --local user.name "github-actions[bot]"' in preflight
        assert (
            "git config --local user.email "
            '"41898282+github-actions[bot]@users.noreply.github.com"' in preflight
        )
        assert 'git merge --no-commit --no-ff "$PR_HEAD_SHA"' in preflight
        assert 'test "$(git rev-parse MERGE_HEAD)" = "$PR_HEAD_SHA"' in preflight
        assert "name: Materialize manual-dispatch candidate tree" in preflight
        assert "CANDIDATE_HEAD_SHA: ${{ github.sha }}" in preflight
        assert 'git merge --no-commit --no-ff "$CANDIDATE_HEAD_SHA"' in preflight
        assert 'test "$(git rev-parse MERGE_HEAD)" = "$CANDIDATE_HEAD_SHA"' in preflight
        assert "working-directory: .ci-contracts" in preflight
        assert "FORGE3D_CI_CONTRACT_ROOT: ${{ github.workspace }}" in preflight
        assert "PYTHONPATH: ${{ github.workspace }}/.ci-contracts" in preflight
    else:
        assert "name: Checkout evaluated workflow state" in preflight
        assert "ref: ${{ github.sha }}" in preflight
        assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in preflight
        assert "base-added policy tests are available" in preflight
    assert "tests/test_ci_cost_controls.py" in preflight
    assert "tests/test_ci_lfs_fanout.py" in preflight
    assert "tests/test_determinism_matrix.py" in preflight
    for job_name in (
        "prepare-lfs-fixtures",
        "terrain-golden-paths",
        "test-rust",
        "build-wheel-windows",
        "build-wheel-linux",
        "build-wheel-macos",
        "build-wheel-linux-arm",
        "build-docs",
    ):
        assert "needs: preflight" in _job(workflow, job_name)

    for platform in ("windows", "linux", "macos", "linux-arm"):
        job = _job(workflow, f"build-wheel-{platform}")
        assert "uses: ./.github/workflows/build-wheel.yml" in job
        assert f"artifact: wheels-{platform}" in job

    core = _job(workflow, "test-python-core")
    assert "runner: ubuntu-latest" in core
    assert "python_versions: '[\"3.11\"]'" in core
    assert "test_mode: full" in core
    for job_name in (
        "test-python-compat-linux",
        "test-python-compat-windows",
        "test-python-compat-macos",
    ):
        assert "test_mode: compatibility" in _job(workflow, job_name)
    assert 'python_versions: \'["3.10", "3.13"]\'' in _job(
        workflow, "test-python-compat-linux"
    )
    for job_name in (
        "test-python-full-linux",
        "test-python-full-windows",
        "test-python-full-macos",
    ):
        job = _job(workflow, job_name)
        assert "github.event_name == 'schedule'" in job
        assert "inputs.scope == 'full'" in job
        assert 'python_versions: \'["3.10", "3.11", "3.12", "3.13"]\'' in job

    assert "retention-days: 1" in _artifact_step(
        _job(workflow, "prepare-lfs-fixtures"), "lfs-fixture-bundles"
    )
    wheel_workflow = _workflow("build-wheel.yml")
    assert "retention-days: 1" in _artifact_step(
        _job(wheel_workflow, "build"), "${{ inputs.artifact }}"
    )
    for job, artifact in (
        ("test-terminus-fuzz", "terminus-fuzzer-evidence"),
        ("test-golden-images", "golden-lane-marker"),
        ("test-golden-images", "visual-golden-diffs"),
    ):
        assert "retention-days: 7" in _artifact_step(_job(workflow, job), artifact)

    physical_artifacts = [
        ("test-substratia-gpu", "substratia-metal-diagnostic-evidence"),
        ("test-golden-images-nvidia", "visual-gpu-evidence"),
        ("test-substratia-gpu-nvidia", "substratia-nvidia-physical-gpu-evidence"),
        ("test-m06-full-geospatial-viewer", "m06-full-geospatial-viewer-evidence"),
        ("test-f3dz-gpu", "f3dz-physical-gpu-evidence"),
        ("test-anamnesis-portability-seed", "anamnesis-physical-portable-store"),
        ("test-anamnesis-portability", "anamnesis-portability-evidence"),
        ("test-anamnesis-production", "anamnesis-production-evidence"),
    ]
    if tessella_enabled:
        physical_artifacts.append(
            ("test-tessella-gpu", "tessella-physical-gpu-evidence")
        )
    if "test-helios-gpu" in jobs:
        physical_artifacts.extend(
            [
                (
                    "test-helios-rust-conservative",
                    "helios-rust-conservative-${{ github.run_id }}-${{ github.run_attempt }}",
                ),
                ("test-helios-gpu", "helios-physical-gpu-evidence"),
            ]
        )
    for job, artifact in physical_artifacts:
        assert "retention-days: 90" in _artifact_step(_job(workflow, job), artifact)

    substratia = _job(workflow, "test-substratia-gpu")
    assert "runs-on: macos-14" in substratia
    assert "inputs.scope == 'full'" in substratia
    assert "FORGE3D_ALLOW_SOFTWARE_GOLDENS" not in substratia
    assert "substratia_evidence_report.py" in substratia

    visual_nvidia = _job(workflow, "test-golden-images-nvidia")
    substratia_nvidia = _job(workflow, "test-substratia-gpu-nvidia")
    for job in (visual_nvidia, substratia_nvidia):
        assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in job
        assert "inputs.scope == 'full'" in job
        assert "WGPU_BACKEND: vulkan" in job
        assert "--require-nvidia-vulkan" in job
        assert "FORGE3D_ALLOW_SOFTWARE_GOLDENS" not in job
    assert "name: wheels-windows" in visual_nvidia
    assert "name: wheels-windows" in substratia_nvidia
    assert "substratia_evidence_report.py" in substratia_nvidia

    assert "sphinx" not in workflow.lower()
    assert "name: documentation" not in workflow
    assert "  build-docs:" in workflow
    assert "cargo doc --workspace" in _job(workflow, "build-docs")

    assert "refresh-recipe-certificates" not in workflow
    certificate = _workflow("certificate-refresh.yml")
    certificate_trigger = certificate.split("\npermissions:", 1)[0]
    assert "  workflow_dispatch:" in certificate_trigger
    assert "  push:" not in certificate_trigger
    assert "  pull_request:" not in certificate_trigger
    assert certificate.count("uses: ./.github/workflows/build-wheel.yml") == 1
    assert "artifact: certificate-refresh-wheel-windows" in certificate
    assert "--require-nvidia-vulkan" in certificate
    assert "ref: ${{ github.sha }}" in certificate
    assert "github.ref == 'refs/heads/main'" in certificate
    assert "github.ref_protected" in certificate
    assert "environment: production-signing" in certificate
    assert "group: certificate-refresh-production" in certificate
    assert "retention-days: 90" in _artifact_step(
        _job(certificate, "refresh"), "refreshed-recipe-certificates"
    )


def test_only_the_small_core_jobs_can_run_on_pr_or_push_events() -> None:
    """Any new CI job is acceptance-only until this allowlist is reviewed."""
    jobs = _workflow_data("ci.yml")["jobs"]
    routine = {"preflight", "test-fast-contract", "pr-core-success"}
    assert routine <= set(jobs)

    assert jobs["preflight"].get("if") is None
    assert jobs["test-fast-contract"].get("if") is None
    assert jobs["pr-core-success"].get("if") == "always()"

    for name, job in jobs.items():
        if name in routine:
            continue
        condition = str(job.get("if", ""))
        assert condition, f"{name} lacks an explicit acceptance-only condition"
        assert "github.event_name == 'pull_request'" not in condition, name
        assert "github.event_name == 'push'" not in condition, name
        assert "run-physical" not in condition, name
        assert (
            "github.event_name == 'schedule'" in condition
            or "github.event_name == 'workflow_dispatch'" in condition
        ), f"{name} is not routed through schedule or explicit dispatch"


def test_aether_closure_lane_is_explicit_and_optional_on_metal() -> None:
    workflow = _workflow("ci.yml")
    golden = _job(workflow, "test-golden-images")
    probe = (ROOT / "scripts" / "terrain_ci_probe.py").read_text(encoding="utf-8")

    # VirtualGpu and software adapters may be useful development paths, but they
    # are not physical-Metal closure evidence.
    assert 'PHYSICAL_DEVICE_TYPES = {"discretegpu", "integratedgpu"}' in probe
    assert 'args.mode in {"sidera-metal", "aether-metal"}' in probe
    aether_branch = probe.split('if args.mode == "aether-metal":', 1)[1].split(
        "# Exit-code contract", 1
    )[0]
    assert "_adapter_is_physical_metal(probe)" in aether_branch
    assert "_smoke_render(args.mode)" in aether_branch
    assert 'write_evidence("failed")' in aether_branch
    assert "return 3" in aether_branch

    assert "python scripts/terrain_ci_probe.py --mode aether-metal" in golden
    assert "tests/test_atmosphere_spectral.py" in golden
    assert "tests/test_atmosphere_reference.py" in golden
    assert "tests/test_atmosphere_pt_reference.py" in golden
    assert "tests/test_atmosphere_golden.py" in golden
    assert 'python scripts/assert_junit_zero_skips.py "$junit"' in golden
    assert "FORGE3D_AETHER_PHYSICAL_METAL: '1'" in golden
    assert "FORGE3D_TEST_INSTALLED_WHEEL: '1'" in golden
    assert "aether_lane=absent" in golden
    assert "software execution does not prove physical Metal" in golden
    # Candidate acceptance must never receive production certificate material.
    assert "FORGE3D_CERT_SIGNING_KEY" not in golden
    assert "FORGE3D_REQUIRE_PRODUCTION_SIGNING" not in golden

    summary = _job(workflow, "full-acceptance-summary")
    # The protected NVIDIA lane is the required full-acceptance proof. Metal is
    # an explicitly opt-in diagnostic and therefore must not gate the summary.
    assert "needs.test-golden-images.outputs.aether_lane" not in summary
    assert "test-golden-images," not in summary.split("\n    runs-on:", 1)[0]


def test_aether_offline_bake_feature_is_explicit_in_acceptance_and_wheel() -> None:
    workflow = _workflow("ci.yml")
    rust_job = _job(workflow, "test-rust")
    golden_job = _job(workflow, "test-golden-images")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    maturin = pyproject.split("[tool.maturin]", 1)[1].split("\n[", 1)[0]

    assert "--features atmosphere-bake core::atmosphere" in rust_job
    assert "atmosphere-bake" in maturin
    assert "atmosphere_bake_luts(" in golden_job
    assert "assert h.precomputed is False" in golden_job
    assert "tests/test_atmosphere_lut_handoff.py" in golden_job


def test_publish_cost_controls_are_tag_only_and_retention_aware() -> None:
    workflow = _workflow("publish.yml")
    trigger = workflow.split("\npermissions:", 1)[0]
    assert "  pull_request:" not in trigger
    assert re.search(r"(?m)^  push:\n    tags:\n      - 'v\*'$", trigger)
    assert "  workflow_dispatch:" in trigger

    concurrency = re.search(r"(?ms)^concurrency:\n.*?(?=^jobs:)", workflow)
    assert concurrency
    assert "group: ${{ github.workflow }}-${{ github.ref }}" in concurrency.group()
    assert "cancel-in-progress: false" in concurrency.group()
    assert "queue: max" in concurrency.group()

    publish = _job(workflow, "publish")
    condition = re.search(r"(?m)^    if: (.+)$", publish)
    assert condition
    assert "startsWith(github.ref, 'refs/tags/v')" in condition.group(1)
    assert "github.event_name == 'push'" in condition.group(1)
    assert "github.event_name == 'workflow_dispatch'" in condition.group(1)
    assert "github.event.inputs.dry_run == 'false'" in condition.group(1)

    expected = "retention-days: ${{ startsWith(github.ref, 'refs/tags/v') && 90 || 1 }}"
    uploads = _upload_steps(workflow)
    assert uploads
    assert all(expected in upload for upload in uploads)


def test_determinism_cost_controls_keep_pr_cancellation_and_evidence() -> None:
    workflow = _workflow("determinism-matrix.yml")
    trigger = workflow.split("\npermissions:", 1)[0]
    assert "  workflow_call:" in trigger
    assert "  workflow_dispatch:" not in trigger
    assert "  push:" not in trigger
    assert "  pull_request:" not in trigger
    assert "maturin develop" not in workflow
    assert "PyO3/maturin-action" not in workflow
    assert "actions/download-artifact@v4" in workflow
    for family in ("run_render", "run_f3dz", "run_anamnesis"):
        assert family in trigger

    uploads = _upload_steps(workflow)
    assert uploads
    assert all("retention-days: 7" in upload for upload in uploads)
    for artifact in (
        "f3dz-determinism-${{ matrix.os }}",
        "determinism-hash-${{ matrix.leg }}",
        "determinism-hash-wasm",
        "anamnesis-store-linux",
        "anamnesis-hosted-consumer-evidence",
    ):
        assert "retention-days: 7" in _artifact_step(workflow, artifact)


def test_all_workflows_parse_and_local_reusable_calls_are_well_formed() -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    parsed = {
        path.name: _workflow_data(path.name) for path in workflow_dir.glob("*.yml")
    }
    assert parsed

    allowed_call_keys = {
        "name",
        "uses",
        "with",
        "secrets",
        "strategy",
        "needs",
        "if",
        "permissions",
        "concurrency",
    }
    for caller_name, workflow in parsed.items():
        for job_name, job in workflow["jobs"].items():
            if not isinstance(job, dict):
                continue
            uses = job.get("uses", "")
            if not uses.startswith("./.github/workflows/"):
                continue
            assert set(job) <= allowed_call_keys, (
                f"{caller_name}:{job_name} uses unsupported keys for a reusable call: "
                f"{sorted(set(job) - allowed_call_keys)}"
            )
            target_name = Path(uses).name
            assert (
                target_name in parsed
            ), f"{caller_name}:{job_name} calls missing {uses}"
            target_on = parsed[target_name].get("on", {})
            assert (
                isinstance(target_on, dict) and "workflow_call" in target_on
            ), f"{caller_name}:{job_name} target {target_name} is not reusable"
            call_contract = target_on["workflow_call"] or {}
            declared_inputs = call_contract.get("inputs", {})
            provided_inputs = job.get("with", {}) or {}
            assert set(provided_inputs) <= set(declared_inputs), (
                f"{caller_name}:{job_name} passes unknown inputs to {target_name}: "
                f"{sorted(set(provided_inputs) - set(declared_inputs))}"
            )
            required_inputs = {
                name
                for name, contract in declared_inputs.items()
                if str(contract.get("required", "false")).casefold() == "true"
            }
            assert required_inputs <= set(provided_inputs), (
                f"{caller_name}:{job_name} omits required {target_name} inputs: "
                f"{sorted(required_inputs - set(provided_inputs))}"
            )
            for input_name, value in provided_inputs.items():
                input_type = declared_inputs[input_name].get("type")
                if input_type == "boolean" and not str(value).startswith("${{"):
                    assert str(value).casefold() in {
                        "true",
                        "false",
                    }, f"{caller_name}:{job_name} passes non-boolean {input_name}={value}"


def test_tessella_acceptance_is_absent_or_exactly_scoped() -> None:
    workflow = _workflow("ci.yml")
    paths = _job(workflow, "terrain-golden-paths")
    assert "tessella: ${{ steps.filter.outputs.tessella }}" in paths
    assert "            tessella:" in paths

    baseline_paths = (
        "src/terrain/renderer/virtual_texture.rs",
        "src/core/feedback_buffer.rs",
        "src/core/screen_space_effects/hzb.rs",
        "src/shaders/hzb_build.wgsl",
        "tests/test_terrain_vt_pbr_families.py",
        "tests/test_tv20_virtual_texturing.py",
    )
    for path in baseline_paths:
        assert f"              - '{path}'" in paths

    # Main may reserve the classifier without running a TESSELLA lane. If a
    # candidate introduces one, admit only the exact fail-closed physical lane
    # and aggregate contract reviewed here.
    jobs = _workflow_data("ci.yml")["jobs"]
    tessella_jobs = []
    tessella_markers = baseline_paths + (
        "src/terrain/culling/two_phase.rs",
        "src/shaders/hzb_cull.wgsl",
        "scripts/tessella_evidence_contract.py",
        "scripts/tessella_evidence_provenance.py",
        "scripts/tessella_evidence_report.py",
        "scripts/tessella_evidence_thresholds.py",
        "tests/test_vt_out_of_core.py",
        "tests/test_hzb_culling.py",
        "tests/test_visibility_buffer.py",
        "tests/test_bc_encoders.py",
        "tests/test_flythrough_popping.py",
        "tests/test_vt_request_retention.py",
        "tests/test_tessella_certificate_evidence.py",
        "tests/test_tessella_evidence_report.py",
    )
    for name, body in jobs.items():
        if name == "terrain-golden-paths":
            continue
        serialized = yaml.safe_dump(body).casefold()
        physical_domain_job = "self-hosted" in serialized and any(
            marker in serialized for marker in tessella_markers
        )
        if (
            "tessella" in name.casefold()
            or "tessella" in serialized
            or physical_domain_job
        ):
            tessella_jobs.append(name)
    if "test-tessella-gpu" not in jobs:
        assert (
            tessella_jobs == []
        ), f"TESSELLA jobs require an explicit scoped contract: {tessella_jobs}"
        return

    assert tessella_jobs == ["test-tessella-gpu", "full-acceptance-summary"]

    for path in tessella_markers[len(baseline_paths) :]:
        assert f"              - '{path}'" in paths

    lane = jobs["test-tessella-gpu"]
    expected_condition = " ".join(
        """
        github.event_name == 'schedule' ||
        (github.event_name == 'workflow_dispatch' &&
         (inputs.scope == 'full' || inputs.scope == 'tessella'))
        """.split()
    )
    assert " ".join(lane["if"].split()) == expected_condition
    assert lane["needs"] == ["build-wheel-windows", "terrain-golden-paths"]
    assert lane["runs-on"] == [
        "self-hosted",
        "Windows",
        "X64",
        "forge3d-gpu",
        "gpu-nvidia",
    ]
    assert lane["timeout-minutes"] == "90"
    assert lane["env"]["WGPU_BACKEND"] == "vulkan"
    assert lane["env"]["FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN"] == "1"
    assert lane["env"]["FORGE3D_NO_BOOTSTRAP"] == "1"
    assert lane["env"]["FORGE3D_TEST_INSTALLED_WHEEL"] == "1"
    assert lane["env"]["FORGE3D_TESSELLA_REQUIRED_GPU"] == "1"

    lane_text = _job(workflow, "test-tessella-gpu")
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in lane_text
    assert "--require-nvidia-vulkan" in lane_text
    assert "rustup default stable" in lane_text
    assert "rust-toolchain.txt" in lane_text
    assert lane_text.index("rustup default stable") < lane_text.index(
        "cargo test --release --lib"
    )
    assert (
        "terrain::clipmap::gpu_lod::tests::"
        "gpu_and_cpu_select_identical_tile_sets_for_1000_cameras" in lane_text
    )
    assert (
        "terrain::culling::two_phase::tests::"
        "hzb_cull_shader_matches_the_cpu_occlusion_predicate" in lane_text
    )
    assert lane_text.count("--ignored --exact --nocapture") == 2
    acceptance_files = (
        "test_vt_out_of_core.py",
        "test_hzb_culling.py",
        "test_visibility_buffer.py",
        "test_bc_encoders.py",
        "test_flythrough_popping.py",
        "test_vt_request_retention.py",
        "test_tessella_certificate_evidence.py",
    )
    for test_file in acceptance_files:
        assert test_file in lane_text
    assert "scripts/tessella_evidence_report.py" in lane_text

    steps = lane["steps"]
    exact_head_steps = [
        step for step in steps if "TESSELLA checkout mismatch" in step.get("run", "")
    ]
    assert len(exact_head_steps) == 1
    exact_head_run = exact_head_steps[0]["run"]
    assert "git rev-parse HEAD" in exact_head_run
    assert "checked-out-head.txt" in exact_head_run
    assert "run-context.json" in exact_head_run

    acceptance_steps = [
        step for step in steps if "tests/test_vt_out_of_core.py" in step.get("run", "")
    ]
    assert len(acceptance_steps) == 1
    acceptance_run = acceptance_steps[0]["run"]
    for test_file in acceptance_files:
        assert test_file in acceptance_run
    assert '--junitxml="$junit"' in acceptance_run
    assert 'python scripts/assert_junit_zero_skips.py "$junit"' in acceptance_run
    assert "if ($pytestCode -ne 0) { exit $pytestCode }" in acceptance_run
    assert "exit $verifyCode" in acceptance_run

    evidence = _artifact_step(lane_text, "tessella-physical-gpu-evidence")
    assert "if: always()" in evidence
    assert "if-no-files-found: error" in evidence
    assert "retention-days: 90" in evidence

    summary = jobs["full-acceptance-summary"]
    assert "test-tessella-gpu" in summary["needs"]
    assert "always()" in summary["if"]
    assert "github.event_name == 'schedule'" in summary["if"]
    assert "github.event_name == 'workflow_dispatch'" in summary["if"]
    assert "github.event_name == 'pull_request'" not in summary["if"]
    assert "github.event_name == 'push'" not in summary["if"]
    assert "run-physical" not in summary["if"]

    enforcement_steps = [
        step for step in summary["steps"] if "check_selected()" in step.get("run", "")
    ]
    assert len(enforcement_steps) == 1
    enforcement_run = enforcement_steps[0]["run"]
    assert (
        "tessella_selected=\"${{ github.event_name == 'schedule' || "
        "(github.event_name == 'workflow_dispatch' && (inputs.scope == 'full' || "
        "inputs.scope == 'tessella')) }}\"" in enforcement_run
    )
    assert (
        'check_selected "$tessella_selected" '
        "'${{ needs.test-tessella-gpu.result }}' tessella-physical" in enforcement_run
    )


def test_physical_selection_truth_table_and_job_conditions() -> None:
    jobs = _workflow_data("ci.yml")["jobs"]

    def normalize(value: str) -> str:
        return " ".join(value.split())

    def expected_condition(scope: str) -> str:
        return normalize(
            f"""
            github.event_name == 'schedule' ||
            (github.event_name == 'workflow_dispatch' &&
             (inputs.scope == 'full' || inputs.scope == '{scope}'))
            """
        )

    expected = {
        "test-m06-full-geospatial-viewer": expected_condition("m06"),
        "test-f3dz-gpu": expected_condition("f3dz"),
        "test-anamnesis-portability-seed": expected_condition("anamnesis"),
        "test-anamnesis-portability": expected_condition("anamnesis"),
        "test-anamnesis-production": expected_condition("anamnesis"),
    }
    if "test-tessella-gpu" in jobs:
        expected["test-tessella-gpu"] = expected_condition("tessella")
    for job_name, condition in expected.items():
        assert normalize(jobs[job_name]["if"]) == condition

    def selected(
        event: str,
        *,
        scope: str = "core",
        internal: bool = False,
        labeled: bool = False,
        path_selected: bool = False,
        family: str = "m06",
    ) -> bool:
        return (
            event == "schedule"
            or (event == "workflow_dispatch" and scope in {"full", family})
        )

    cases = (
        ({"event": "push", "path_selected": True}, False),
        ({"event": "pull_request", "path_selected": True}, False),
        (
            {
                "event": "pull_request",
                "internal": True,
                "path_selected": True,
            },
            False,
        ),
        (
            {
                "event": "pull_request",
                "internal": True,
                "labeled": True,
                "path_selected": False,
            },
            False,
        ),
        (
            {
                "event": "pull_request",
                "internal": True,
                "labeled": True,
                "path_selected": True,
            },
            False,
        ),
        ({"event": "schedule"}, True),
        ({"event": "workflow_dispatch", "scope": "core"}, False),
        ({"event": "workflow_dispatch", "scope": "full"}, True),
        ({"event": "workflow_dispatch", "scope": "m06"}, True),
        ({"event": "workflow_dispatch", "scope": "f3dz"}, False),
    )
    for arguments, wanted in cases:
        assert selected(**arguments) is wanted
    if "test-tessella-gpu" in jobs:
        assert (
            selected("workflow_dispatch", scope="tessella", family="tessella") is True
        )
        assert selected("workflow_dispatch", scope="m06", family="tessella") is False
