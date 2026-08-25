# tests/test_provenance_offline_verify.py
# VERITAS: proves the standalone verifier re-verifies a natively-sealed
# provenance triple WITHOUT the compiled _forge3d extension. Runs the CLI in
# a subprocess with an import hook that masks forge3d._forge3d, forcing the
# pure-Python _ed25519 fallback, against the committed fixture triple.
# RELEVANT FILES: tools/verify_provenance.py, python/forge3d/provenance.py,
# python/forge3d/_ed25519.py, tests/test_provenance_veritas.py

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "tools" / "verify_provenance.py"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "provenance"
SOURCE_MAP = FIXTURE_DIR / "source_map.npy"
MANIFEST = FIXTURE_DIR / "provenance.json"

# Bootstrap that masks the compiled extension before anything imports it,
# then runs the standalone verifier exactly as a third party would.
_MASKED_BOOTSTRAP = textwrap.dedent(
    """
    import importlib.abc
    import runpy
    import sys

    class _MaskNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "forge3d._forge3d":
                raise ImportError("forge3d._forge3d masked for offline verification test")
            return None

    sys.meta_path.insert(0, _MaskNative())
    tool, *argv = sys.argv[1:]
    sys.argv = [tool] + argv
    runpy.run_path(tool, run_name="__main__")
    """
)


def _run_verifier_without_native(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _MASKED_BOOTSTRAP, str(VERIFIER), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


@pytest.fixture()
def provenance_fixture(tmp_path):
    from PIL import Image

    assert SOURCE_MAP.is_file(), "tracked provenance source map is required"
    assert MANIFEST.is_file(), "tracked provenance manifest is required"
    source_map = np.asarray(np.load(SOURCE_MAP), dtype=np.uint32)
    rgba = np.zeros(source_map.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = np.where(source_map == 1, 220, 30)
    rgba[..., 1] = np.where(source_map == 2, 220, 30)
    rgba[..., 2] = 30
    rgba[..., 3] = 255
    image = tmp_path / "image.png"
    Image.fromarray(rgba, mode="RGBA").save(image)
    return image, SOURCE_MAP, MANIFEST


def test_offline_verifier_verifies_native_manifest_without_extension(
    provenance_fixture,
) -> None:
    image, source_map, manifest = provenance_fixture
    result = _run_verifier_without_native(image, source_map, manifest)
    assert result.returncode == 0, f"verifier failed:\n{result.stdout}\n{result.stderr}"
    assert "merkle_root_match: True" in result.stdout
    assert "signature_valid: True" in result.stdout
    assert "image_dims_match: True" in result.stdout
    assert "tamper_probe_single_texel_detected: True" in result.stdout
    assert "verified: True" in result.stdout
    # Per-source pixel coverage is reported for at least two real sources.
    coverage_lines = [
        line for line in result.stdout.splitlines() if line.startswith("coverage source_id=")
    ]
    assert len(coverage_lines) >= 2, result.stdout


def test_offline_verifier_detects_tampered_source_map(
    tmp_path, provenance_fixture
) -> None:
    image, source_map_path, manifest = provenance_fixture
    source_map = np.load(source_map_path)
    source_map = np.asarray(source_map, dtype=np.uint32).copy()
    source_map[0, 0] ^= 1
    tampered_path = tmp_path / "source_map_tampered.npy"
    np.save(tampered_path, source_map)

    result = _run_verifier_without_native(image, tampered_path, manifest)
    assert result.returncode != 0
    assert "merkle_root_match: False" in result.stdout
    assert "verified: False" in result.stdout


def test_pure_python_report_on_fixture(provenance_fixture) -> None:
    """In-process check of the pure-Python path (no native calls involved)."""
    from forge3d import provenance as prov

    _image, source_map_path, manifest_path = provenance_fixture
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == prov.SCHEMA_VERSION
    source_map = np.asarray(np.load(source_map_path), dtype=np.uint32)

    report = prov.verify_provenance_offline(source_map, manifest)
    assert report["ok"] is True
    assert report["root_match"] is True
    assert report["signature_valid"] is True
    assert len(report["coverage"]) >= 2
    assert prov.SOURCE_ID_NONE not in report["coverage"]
    assert not report["unknown_source_ids"]

    # The signature must be pinned to the committed public key + root.
    from forge3d import _ed25519

    assert _ed25519.verify(
        bytes.fromhex(manifest["public_key"]),
        prov.SIGN_CONTEXT + bytes.fromhex(manifest["merkle_root"]),
        bytes.fromhex(manifest["signature"]),
    )
