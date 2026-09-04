# tests/test_certificate_verifier.py
# CENSOR Task 10: offline verifier tests for forge3d.certificate. These run
# WITHOUT the native _forge3d module — certificate.py is pure Python — so a
# third party can re-verify a signed RenderCertificate with a stock CPython.
# RELEVANT FILES: python/forge3d/certificate.py, python/forge3d/_ed25519.py

import copy
import json
import os
import shutil
import sys
import subprocess
from pathlib import Path

import pytest

from forge3d import certificate

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
