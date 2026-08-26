# tests/test_certificate_verifier.py
# CENSOR Task 10: offline verifier tests for forge3d.certificate. These run
# WITHOUT the native _forge3d module — certificate.py is pure Python — so a
# third party can re-verify a signed RenderCertificate with a stock CPython.
# RELEVANT FILES: python/forge3d/certificate.py, python/forge3d/_ed25519.py

import copy
import hashlib
import json
import os
import shutil
import sys
import subprocess
import textwrap
from pathlib import Path

import pytest

from forge3d import certificate
from scripts.run_apple_metal_acceptance import expected_recipe_ids

FIXTURE = {
    "schema": "forge3d.render_certificate/1",
    "engine": {"version": "1.30.1", "git_sha": "deadbeef", "wgsl_module_hashes": {"terrain.main": "ab" * 32}},
    "adapter": {"vendor": "v", "device": "d", "backend": "dx12", "driver_info": "i"},
    "capabilities": {"requested": ["timestamp_query"], "granted": [], "limits": {"max_bind_groups": 4}},
    "passes": [{"label": "terrain.main", "gpu_ms": 1.25, "draw_calls": 7}],
    "allocations": {"peak_host_visible_bytes": 1, "peak_device_local_bytes": 2, "by_label": {"x": 1}},
    "degradations": [],
}

SIGNED_FIXTURE = {
    **copy.deepcopy(FIXTURE),
    "signature": {
        "alg": "ed25519",
        "pubkey": "92f33f5957bd8532b7e878f34b95aabe077f8ab962cea2c4b06acf2c91491917",
        "sig": "74e25234b4c64840fe5fe94468fa2ce5cfedb69284bdb18442f9cba14743e8d8c2ac2950e377a162a8b1ef07a18c096334a25d3d6a1d39313a9306fed7545607",
        "signed_fields": [
            "adapter",
            "allocations",
            "capabilities",
            "degradations",
            "engine",
            "passes (gpu_ms excluded)",
            "schema",
        ],
    },
}


def test_sign_then_verify_roundtrip(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    p = tmp_path / "cert.json"
    certificate.write_certificate(cert, p)
    assert certificate.verify(p, cert["signature"]["pubkey"]) is True


def test_verify_accepts_in_memory_certificate_mapping():
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    assert certificate.verify(cert, cert["signature"]["pubkey"]) is True


def test_committed_certificates_use_pinned_production_key_not_dev_key():
    cert_dir = Path(__file__).parent / "golden" / "certificates"
    pinned = (cert_dir / "signing.pub").read_text(encoding="utf-8").strip()
    dev_pub = certificate.sign_certificate(
        copy.deepcopy(FIXTURE), seed=certificate.DEV_SIGNING_SEED
    )["signature"]["pubkey"]
    assert pinned != dev_pub
    for path in cert_dir.glob("*.json"):
        committed = json.loads(path.read_text(encoding="utf-8"))
        assert committed["signature"]["pubkey"] == pinned
        assert certificate.verify(path, pinned) is True


def test_public_acceptance_covers_catalog_and_rejects_committed_tamper():
    cert_dir = Path(__file__).parent / "golden" / "certificates"
    pinned = (cert_dir / "signing.pub").read_text(encoding="utf-8").strip()
    paths = sorted(cert_dir.glob("*.json"))
    assert {path.stem for path in paths} == set(expected_recipe_ids())
    for path in paths:
        committed = json.loads(path.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(committed)
        tampered["_acceptance_tamper_probe"] = True
        assert certificate.verify(committed, pinned) is True
        assert certificate.verify(tampered, pinned) is False


def _workflow_step_script(name: str) -> str:
    workflow = (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    step = workflow.split(f"- name: {name}", 1)[1].split("\n      - name:", 1)[0]
    run = step.split("run:", 1)[1].lstrip()
    if not run.startswith("|"):
        return run.splitlines()[0]
    return textwrap.dedent(run[1:])


def _public_verifier_script(*, through_pytest_install: bool = False) -> str:
    script = _workflow_step_script(
        "Verify exact-head committed recipe certificates"
    )
    lines = script.splitlines()
    install_index = next(
        index
        for index, line in enumerate(lines)
        if "python -m pip install pytest" in line
    )
    end = install_index + int(through_pytest_install)
    if through_pytest_install and end < len(lines) and lines[end].strip() == ")":
        end += 1
    script = "\n".join(lines[:end])
    return "\n".join(
        line for line in script.splitlines() if "exec > >(tee " not in line
    )


def _prepare_public_verifier_candidate(tmp_path: Path) -> Path:
    root = Path(__file__).parents[1]
    candidate = tmp_path / "candidate"
    trusted = candidate / ".ci-contracts"
    shutil.copytree(
        root / "tests/golden/certificates", candidate / "tests/golden/certificates"
    )
    shutil.copytree(
        root / "tests/golden/certificates", trusted / "tests/golden/certificates"
    )
    package = trusted / "python/forge3d"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("certificate.py", "_canonical_json.py", "_ed25519.py"):
        shutil.copy2(root / "python/forge3d" / name, package / name)
    (candidate / ".gitignore").write_text(
        ".ci-contracts/\nevidence/\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", candidate], check=True)
    subprocess.run(
        ["git", "-C", candidate, "config", "user.name", "certificate-test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", candidate, "config", "user.email", "test@example.invalid"],
        check=True,
    )
    return candidate


def _run_public_verifier(
    candidate: Path, *, through_pytest_install: bool = False
) -> subprocess.CompletedProcess[str]:
    subprocess.run(["git", "-C", candidate, "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            candidate,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "candidate",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", candidate, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    trusted_bin = candidate.parent / "trusted-bin"
    trusted_bin.mkdir()
    (trusted_bin / "python").symlink_to(sys.executable)
    env = os.environ.copy()
    env.update(
        {
            "CERTIFICATE_ACCEPTANCE_DIR": str(candidate / "evidence"),
            "EXPECTED_HEAD": head,
            "GITHUB_WORKSPACE": str(candidate),
            "PYTHONNOUSERSITE": "1",
            "PATH": f"{trusted_bin}{os.pathsep}{env['PATH']}",
        }
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            _public_verifier_script(through_pytest_install=through_pytest_install),
        ],
        cwd=candidate,
        env=env,
        capture_output=True,
        text=True,
    )


def _resign_candidate_catalog(candidate: Path, seed: bytes) -> None:
    public_key = certificate._ed25519.public_key_from_private(seed).hex()
    cert_dir = candidate / "tests/golden/certificates"
    (cert_dir / "signing.pub").write_text(public_key + "\n", encoding="utf-8")
    for path in cert_dir.glob("*.json"):
        signed = json.loads(path.read_text(encoding="utf-8"))
        message = certificate.SIGN_CONTEXT + hashlib.sha256(
            certificate.canonical_payload_bytes(signed)
        ).digest()
        signed["signature"]["pubkey"] = public_key
        signed["signature"]["sig"] = certificate._ed25519.sign(seed, message).hex()
        certificate.write_certificate(signed, path)


def _write_acceptance_shadow(package: Path, marker: Path) -> None:
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "certificate.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )


def test_public_verifier_accepts_authentic_catalog_without_signing_secret(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    result = _run_public_verifier(candidate)
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_verifier_rejects_candidate_key_replacement_and_resigning(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    _resign_candidate_catalog(candidate, b"candidate-controlled-key-seed!!!")
    result = _run_public_verifier(candidate)
    assert result.returncode != 0, result.stdout + result.stderr


def test_public_verifier_ignores_candidate_root_forge3d_shadow(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    marker = candidate / "candidate-forge3d-executed"
    _write_acceptance_shadow(candidate / "forge3d", marker)
    result = _run_public_verifier(candidate)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_public_verifier_ignores_candidate_sitecustomize_shadow(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    marker = candidate / "candidate-sitecustomize-executed"
    (candidate / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "root = Path(__file__).parent / 'forge3d'\n"
        "root.mkdir(exist_ok=True)\n"
        "(root / '__init__.py').write_text('')\n",
        encoding="utf-8",
    )
    result = _run_public_verifier(candidate)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_public_verifier_ignores_candidate_pip_shadow(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    marker = candidate / "candidate-pip-executed"
    package = candidate / "pip"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    result = _run_public_verifier(candidate, through_pytest_install=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_zero_skip_verifier_ignores_candidate_pythonpath_shadow(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    checker = candidate / ".ci-contracts/scripts/assert_junit_zero_skips.py"
    checker.parent.mkdir(parents=True)
    shutil.copy2(Path(__file__).parents[1] / "scripts" / checker.name, checker)
    evidence = candidate / "evidence"
    evidence.mkdir()
    (evidence / "junit.xml").write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="clean"/></testsuite>',
        encoding="utf-8",
    )
    marker = candidate / "candidate-sitecustomize-executed"
    (candidate / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    trusted_bin = candidate.parent / "trusted-zero-skip-bin"
    trusted_bin.mkdir()
    (trusted_bin / "python").symlink_to(sys.executable)
    env = os.environ.copy()
    env.update(
        {
            "CERTIFICATE_ACCEPTANCE_DIR": str(evidence),
            "GITHUB_WORKSPACE": str(candidate),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(candidate),
            "PATH": f"{trusted_bin}{os.pathsep}{env['PATH']}",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script("Require clean zero-skip certificate acceptance"),
        ],
        cwd=candidate,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_public_verifier_rejects_same_certificate_replay(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    certs = sorted((candidate / "tests/golden/certificates").glob("*.json"))
    replay = certs[0].read_bytes()
    for path in certs[1:]:
        path.write_bytes(replay)
    result = _run_public_verifier(candidate)
    assert result.returncode != 0, result.stdout + result.stderr


def test_public_verifier_rejects_two_filename_swap(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    first, second = sorted(
        (candidate / "tests/golden/certificates").glob("*.json")
    )[:2]
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    first.write_bytes(second_bytes)
    second.write_bytes(first_bytes)
    result = _run_public_verifier(candidate)
    assert result.returncode != 0, result.stdout + result.stderr


def test_public_verifier_ignores_candidate_verifier_substitution(tmp_path):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    marker = candidate / "candidate-verifier-executed"
    _write_acceptance_shadow(candidate / "python/forge3d", marker)
    result = _run_public_verifier(candidate)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("attack", ("payload", "missing", "extra"))
def test_public_verifier_rejects_remaining_candidate_attacks(tmp_path, attack):
    candidate = _prepare_public_verifier_candidate(tmp_path)
    cert_dir = candidate / "tests/golden/certificates"
    paths = sorted(cert_dir.glob("*.json"))
    if attack == "payload":
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
        payload["_candidate_tamper"] = True
        paths[0].write_text(json.dumps(payload), encoding="utf-8")
    elif attack == "missing":
        paths[0].unlink()
    elif attack == "extra":
        shutil.copy2(paths[0], cert_dir / "extra.json")
    result = _run_public_verifier(candidate)
    assert result.returncode != 0, result.stdout + result.stderr


def test_gpu_ms_is_not_signed(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    cert["passes"][0]["gpu_ms"] = 99.0
    p = tmp_path / "cert.json"; certificate.write_certificate(cert, p)
    assert certificate.verify(p, cert["signature"]["pubkey"]) is True


def test_any_signed_byte_tamper_fails(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    cert["allocations"]["peak_host_visible_bytes"] += 1
    p = tmp_path / "cert.json"; certificate.write_certificate(cert, p)
    assert certificate.verify(p, cert["signature"]["pubkey"]) is False


def test_model_assumption_tamper_fails_verification():
    fixture = copy.deepcopy(FIXTURE)
    fixture["models"] = {
        "astro.twilight": "civil-to-astronomical smoothstep over solar altitude"
    }
    cert = certificate.sign_certificate(fixture)
    assert "models" in cert["signature"]["signed_fields"]
    cert["models"]["astro.twilight"] = "tampered visibility model"
    assert certificate.verify(cert, cert["signature"]["pubkey"]) is False


def test_payload_bytes_are_deterministic():
    a = certificate.canonical_payload_bytes(copy.deepcopy(FIXTURE))
    b = certificate.canonical_payload_bytes(json.loads(json.dumps(FIXTURE)))
    assert a == b


def test_payload_normalizes_negative_zero():
    fixture = copy.deepcopy(FIXTURE)
    fixture["value"] = -0.0
    assert b'"value":0.0' in certificate.canonical_payload_bytes(fixture)


def test_payload_rejects_non_finite_float():
    fixture = copy.deepcopy(FIXTURE)
    fixture["value"] = float("nan")
    with pytest.raises(ValueError):
        certificate.canonical_payload_bytes(fixture)


def test_signing_is_deterministic():
    # Ed25519 signing is deterministic (RFC 8032): the same certificate and
    # seed must yield identical payload digests AND identical signature bytes.
    first = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    second = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    assert certificate.payload_sha256(first) == certificate.payload_sha256(second)
    assert first["signature"]["sig"] == second["signature"]["sig"]


def test_signing_uses_native_ed25519_dalek(monkeypatch, tmp_path):
    def reject_python_signing(*_args, **_kwargs):
        raise AssertionError("certificate signing must use native ed25519-dalek")

    monkeypatch.setattr(certificate._ed25519, "sign", reject_python_signing)
    monkeypatch.setattr(
        certificate._ed25519, "public_key_from_private", reject_python_signing
    )

    signed = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    path = tmp_path / "native-signed.json"
    certificate.write_certificate(signed, path)
    assert certificate.verify(path, signed["signature"]["pubkey"])


def test_cli_verify(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    p = tmp_path / "cert.json"; certificate.write_certificate(cert, p)
    k = tmp_path / "k.pub"; k.write_text(cert["signature"]["pubkey"])
    # Resolve forge3d for the subprocess from the SAME location this test
    # imported it, so the CLI is exercised against the code under test rather
    # than whatever ambient install `sys.path` happens to expose (the dev venv
    # may point its `.pth` at a sibling worktree).
    pkg_parent = str(Path(certificate.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    r = subprocess.run([sys.executable, "-m", "forge3d.certificate", "verify", str(p), "--pubkey", str(k)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "VALID" in r.stdout, r.stderr


def test_cli_verify_without_native_module(tmp_path):
    package = tmp_path / "forge3d"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source_dir = Path(certificate.__file__).resolve().parent
    for name in ("certificate.py", "_canonical_json.py", "_ed25519.py"):
        shutil.copy2(source_dir / name, package / name)

    cert_path = tmp_path / "cert.json"
    cert_path.write_text(json.dumps(SIGNED_FIXTURE), encoding="utf-8")
    key_path = tmp_path / "key.pub"
    key_path.write_text(SIGNED_FIXTURE["signature"]["pubkey"], encoding="ascii")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "forge3d.certificate",
            "verify",
            str(cert_path),
            "--pubkey",
            str(key_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert not (package / "_forge3d.pyd").exists()
    assert result.returncode == 0 and "VALID" in result.stdout, result.stderr
