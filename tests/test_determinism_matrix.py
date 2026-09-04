import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_determinism_hashes.py"
DUPLA_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dupla_proof.py"
NVIDIA_RUNNER = (
    Path(__file__).parents[1] / "scripts" / "run_nvidia_determinism_acceptance.py"
)
CONTRACT_ROOT = Path(
    os.environ.get("FORGE3D_CI_CONTRACT_ROOT", Path(__file__).parents[1])
)
WORKFLOW = CONTRACT_ROOT / ".github" / "workflows" / "determinism-matrix.yml"
SCENE = "terra_determinata_v1"
SHA = "d" * 64


def _artifact(root, leg, *, sha=None, adapter=True, marker=None):
    path = root / f"determinism-hash-{leg}"
    path.mkdir(parents=True)
    if sha:
        (path / f"{SCENE}.sha256").write_text(sha + "\n")
        if adapter:
            (path / f"{SCENE}.json").write_text(
                json.dumps(
                    {
                        "adapter": {
                            "name": "NVIDIA GeForce RTX 3070",
                            "backend": "Vulkan",
                            "device_type": "DiscreteGpu",
                            "vendor": 0x10DE,
                            "device": 0x2484,
                            "software_fallback": False,
                        }
                    }
                )
            )
    if marker:
        (path / f"{SCENE}.{marker}").write_text(f"{leg} {marker.lower()}\n")
    return path


def _run(tmp_path, golden=SHA):
    golden_file = tmp_path / "golden.sha256"
    if golden is not None:
        golden_file.write_text(golden + "\n")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hashes",
            str(tmp_path / "hashes"),
            "--golden",
            str(golden_file),
            "--scene",
            SCENE,
        ],
        capture_output=True,
        text=True,
    )


def _dupla_proof() -> dict:
    operation = {
        "generated_count": 100_000_000,
        "adversarial_count": 1_000_000,
        "mismatch_count": 0,
        "max_err_u2": 1.0,
        "cited_bound_u2": 3.0,
    }
    return {
        "schema": "forge3d.dupla-proof.v1",
        "backend": "vulkan",
        "adapter": "hardware",
        "selftest": {"passed": True, "mismatch_count": 0},
        "harness": {name: dict(operation) for name in ("add", "mul", "div", "sqrt")},
        "jitter": {
            "dd_max_error_px": 0.001,
            "raw_over_one_px": 100,
            "dd_hash_a": SHA,
            "dd_hash_b": SHA,
        },
    }


def _validate_dupla(tmp_path, *legs):
    return subprocess.run(
        [
            sys.executable,
            str(DUPLA_SCRIPT),
            "--validate-artifacts",
            str(tmp_path),
            "--expected-legs",
            *legs,
        ],
        capture_output=True,
        text=True,
    )


def test_matrix_rejects_zero_hardware_hashes(tmp_path):
    (tmp_path / "hashes").mkdir()
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "no hardware-backed leg produced a hash" in result.stderr


def test_render_acceptance_has_no_required_metal_leg():
    workflow = WORKFLOW.read_text()
    hosted = workflow.split("  render:\n", 1)[1].split("\n  render-nvidia:\n", 1)[0]
    nvidia = workflow.split("  render-nvidia:\n", 1)[1].split(
        "\n  # Optional Apple/Metal", 1
    )[0]
    diagnostic = workflow.split("  metal-diagnostic:\n", 1)[1].split(
        "\n  wasm-policy:\n", 1
    )[0]
    required = hosted + nvidia
    assert "leg: apple" not in required
    assert "backend: metal" not in required
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in nvidia
    assert "name: wheels-windows" in nvidia
    assert "shell: bash" not in nvidia
    assert "shell: pwsh" in nvidia
    assert "terrain_ci_probe.py" in nvidia
    assert "--require-nvidia-vulkan" in nvidia
    assert "run_nvidia_determinism_acceptance.py" in nvidia
    runner = NVIDIA_RUNNER.read_text()
    assert '_render_once(args, artifact_dir, "first")' in runner
    assert '_render_once(args, artifact_dir, "repeat")' in runner
    assert "render/probe" in runner
    assert "repeat hash differs" in runner
    assert "FORGE3D_RUN_METAL_DIAGNOSTIC" in diagnostic
    assert "determinism-metal-diagnostic-apple" in diagnostic
    assert "continue-on-error: true" in diagnostic
    assert "--expected-legs intel amd nvidia" in workflow


def _nvidia_adapter(*, device=0x2484):
    return {
        "name": "NVIDIA GeForce RTX 3070",
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "vendor": 0x10DE,
        "device": device,
        "software_fallback": False,
    }


def _nvidia_probe(tmp_path):
    path = tmp_path / "nvidia-adapter-probe.json"
    path.write_text(
        json.dumps({"requested_backend": "vulkan", "probe": _nvidia_adapter()})
    )
    return path


def _run_nvidia_acceptance(monkeypatch, tmp_path, render_payloads):
    from scripts import run_nvidia_determinism_acceptance as runner

    payloads = iter(render_payloads)

    def fake_run(command, **_kwargs):
        pixels, adapter = next(payloads)
        png_path = Path(command[command.index("--out-png") + 1])
        png_path.write_bytes(pixels)
        record = {
            "scene": SCENE,
            "sha256": hashlib.sha256(pixels).hexdigest(),
            "adapter": adapter,
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(record) + "\n", stderr=""
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("FORGE3D_DETERMINISTIC", "1")
    monkeypatch.setenv("WGPU_BACKENDS", "vulkan")
    artifact_dir = tmp_path / "hash-out"
    result = runner.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--adapter-probe",
            str(_nvidia_probe(tmp_path)),
            "--scene",
            SCENE,
            "--width",
            "512",
            "--height",
            "512",
        ]
    )
    return result, artifact_dir


def test_nvidia_acceptance_binds_two_matching_frames_to_probe(monkeypatch, tmp_path):
    pixels = b"physical-nvidia-vulkan-frame"
    result, artifact_dir = _run_nvidia_acceptance(
        monkeypatch,
        tmp_path,
        [(pixels, _nvidia_adapter()), (pixels, _nvidia_adapter())],
    )
    assert result == 0
    assert (artifact_dir / f"{SCENE}.sha256").is_file()
    assert (artifact_dir / f"{SCENE}.repeat.sha256").is_file()
    assert (artifact_dir / f"{SCENE}.json").is_file()
    assert (artifact_dir / f"{SCENE}.repeat.json").is_file()
    assert not (artifact_dir / f"{SCENE}.FAILED").exists()


def test_nvidia_acceptance_rejects_probe_render_identity_drift(
    monkeypatch, tmp_path
):
    pixels = b"physical-nvidia-vulkan-frame"
    result, artifact_dir = _run_nvidia_acceptance(
        monkeypatch,
        tmp_path,
        [(pixels, _nvidia_adapter(device=0x2684)), (pixels, _nvidia_adapter())],
    )
    assert result == 1
    failure = (artifact_dir / f"{SCENE}.FAILED").read_text()
    assert "first render/probe adapter differs at device" in failure


def test_nvidia_acceptance_rejects_repeat_hash_drift(monkeypatch, tmp_path):
    result, artifact_dir = _run_nvidia_acceptance(
        monkeypatch,
        tmp_path,
        [
            (b"physical-nvidia-vulkan-frame-a", _nvidia_adapter()),
            (b"physical-nvidia-vulkan-frame-b", _nvidia_adapter()),
        ],
    )
    assert result == 1
    failure = (artifact_dir / f"{SCENE}.FAILED").read_text()
    assert "repeat hash differs" in failure


def test_matrix_rejects_gated_failure_without_required_nvidia_hash(tmp_path):
    _artifact(tmp_path / "hashes", "intel", marker="FAILED")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "required NVIDIA/Vulkan leg produced no hash" in result.stderr


def test_matrix_rejects_unattributed_hash(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA, adapter=False)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "invalid attributable adapter metadata" in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "Dx12"),
        ("device_type", "IntegratedGpu"),
        ("vendor", 0x1002),
        ("name", "AMD Radeon RX 7900 XT"),
        ("device", "not-a-device"),
        ("device", -1),
        ("software_fallback", True),
        ("software_fallback", None),
    ],
)
def test_nvidia_hash_requires_strict_physical_vulkan_identity(tmp_path, field, value):
    artifact = _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    meta_path = artifact / f"{SCENE}.json"
    meta = json.loads(meta_path.read_text())
    if value is None:
        meta["adapter"].pop(field)
    else:
        meta["adapter"][field] = value
    meta_path.write_text(json.dumps(meta))
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "invalid attributable adapter metadata" in result.stderr


def test_non_nvidia_hardware_hash_cannot_replace_required_anchor(tmp_path):
    _artifact(tmp_path / "hashes", "amd", sha=SHA)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "required NVIDIA/Vulkan leg produced no hash" in result.stderr


def test_determinism_record_persists_strict_adapter_identity(
    monkeypatch, tmp_path, capsys
):
    import forge3d as f3d
    from forge3d import determinism

    monkeypatch.setenv("WGPU_BACKENDS", "vulkan")
    monkeypatch.setattr(
        determinism,
        "_render_reference_inprocess",
        lambda *_args, **_kwargs: SHA,
    )
    monkeypatch.setattr(
        f3d,
        "device_probe",
        lambda _backend=None: {
            "status": "ok",
            "name": "NVIDIA GeForce RTX 3070",
            "backend": "Vulkan",
            "device_type": "DiscreteGpu",
            "vendor": 0x10DE,
            "device": 0x2484,
            "software_fallback": False,
        },
    )

    assert determinism._main(["--out-png", str(tmp_path / "unused.png")]) == 0
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["adapter"] == {
        "name": "NVIDIA GeForce RTX 3070",
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "vendor": 0x10DE,
        "device": 0x2484,
        "software_fallback": False,
    }


def test_matrix_accepts_matching_hardware_hash_with_documented_gated_failure(tmp_path):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "intel", marker="FAILED")
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "GATED-FAILURE" in result.stdout


@pytest.mark.parametrize("actual", ["e" * 64, "f" * 64])
def test_matrix_rejects_golden_or_pairwise_mismatch(tmp_path, actual):
    _artifact(tmp_path / "hashes", "nvidia", sha=SHA)
    _artifact(tmp_path / "hashes", "amd", sha=actual)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "mismatch" in result.stderr


def test_f3dz_stream_hashes_run_on_two_hosted_platforms():
    workflow = WORKFLOW.read_text()
    assert "f3dz-stream:" in workflow
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert "tools/f3dz_determinism_report.py" in workflow
    assert "test_error_bound_stored_page_error_nan_and_determinism" in workflow
    assert "test_cross_platform_determinism_hashes" in workflow
    assert "f3dz-determinism-${{ matrix.os }}" in workflow


def test_shadow_shader_classifier_is_retained_without_pr_path_gating():
    ci = (WORKFLOW.parent / "ci.yml").read_text()
    classifier = ci.split("            determinism_render:\n", 1)[1].split(
        "\n            determinism_f3dz:\n", 1
    )[0]
    assert "'src/shaders/shadows.wgsl'" in classifier
    assert "'src/shaders/includes/shadow_moments.wgsl'" in classifier
    assert "'src/shaders/csm.wgsl'" not in classifier

    caller = ci.split("  determinism-render:", 1)[1].split(
        "\n  determinism-f3dz:", 1
    )[0]
    assert "needs.terrain-golden-paths.outputs.determinism_render" not in caller
    assert "github.event_name == 'schedule'" in caller
    assert "inputs.scope == 'full'" in caller
    assert "inputs.scope == 'determinism'" in caller
    assert "github.event_name == 'pull_request'" not in caller
    assert "uses: ./.github/workflows/determinism-matrix.yml" in caller
    assert "run_render: true" in caller


def test_matrix_reuses_caller_wheels_instead_of_rebuilding_extensions():
    workflow = WORKFLOW.read_text()
    assert "  workflow_call:" in workflow
    assert "maturin develop" not in workflow
    assert "PyO3/maturin-action" not in workflow
    for artifact in ("wheels-linux", "wheels-windows"):
        assert artifact in workflow
    acceptance = workflow.split("  render:\n", 1)[1].split("\n  metal-diagnostic:\n", 1)[0]
    diagnostic = workflow.split("  metal-diagnostic:\n", 1)[1].split(
        "\n  wasm-policy:\n", 1
    )[0]
    assert "wheels-macos" not in acceptance
    assert "wheels-macos" in diagnostic
    assert "ref: ${{ inputs.ref }}" in workflow
    assert "FORGE3D_NO_BOOTSTRAP: '1'" in workflow
    anamnesis = workflow.split("  anamnesis-seed:", 1)[1].split(
        "\n  anamnesis-portability:", 1
    )[0]
    assert anamnesis.index("Classify the hosted Vulkan adapter") < anamnesis.index(
        "Gate ANAMNESIS incrementality"
    )
    summary = workflow.split("  diff:", 1)[1]
    for family in (
        "f3dz-stream",
        "render",
        "render-nvidia",
        "wasm-policy",
        "anamnesis-seed",
        "anamnesis-portability",
    ):
        assert f"needs.{family}.result" in summary


def test_dupla_aggregation_accepts_verified_and_explicit_absence(tmp_path):
    (tmp_path / "dupla-proof-nvidia.json").write_text(json.dumps(_dupla_proof()))
    (tmp_path / "dupla-proof-intel.ABSENT").write_text("no physical adapter\n")
    result = _validate_dupla(tmp_path, "nvidia", "intel")
    assert result.returncode == 0, result.stderr
    assert "VERIFIED" in result.stdout and "ABSENT" in result.stdout


def test_dupla_aggregation_rejects_failed_or_missing_proof(tmp_path):
    (tmp_path / "dupla-proof-amd.FAILED").write_text("bound exceeded\n")
    failed = _validate_dupla(tmp_path, "amd")
    assert failed.returncode == 1
    assert "DUPLA proof failed" in failed.stderr
    missing = _validate_dupla(tmp_path, "nvidia")
    assert missing.returncode == 1
    assert "expected exactly one DUPLA result" in missing.stderr


def test_dupla_aggregation_rejects_invalid_evidence(tmp_path):
    proof = _dupla_proof()
    proof["harness"]["mul"]["max_err_u2"] = 8.0
    proof["harness"]["mul"]["cited_bound_u2"] = 7.0
    (tmp_path / "dupla-proof-nvidia.json").write_text(json.dumps(proof))
    result = _validate_dupla(tmp_path, "nvidia")
    assert result.returncode == 1
    assert "cited bound exceeded" in result.stderr
