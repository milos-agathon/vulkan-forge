# tests/test_trust_boundary_diagnostics.py
# Trust-boundary diagnostics: native import root-cause capture, informative
# missing-symbol errors, the device_probe status taxonomy, and the guarantee
# that GPU acquisition failures raise catchable RuntimeError, never
# pyo3_runtime.PanicException.
# RELEVANT FILES:python/forge3d/_native.py,python/forge3d/_gpu.py,python/forge3d/__init__.py,src/core/gpu.rs

from __future__ import annotations

import glob
import os
import subprocess
import sys

import pytest

import forge3d as f3d
from forge3d import _gpu as gpu_mod
from forge3d import _native as native_mod

NATIVE_AVAILABLE = native_mod.NATIVE_AVAILABLE


# ---------------------------------------------------------------------------
# native_import_error()
# ---------------------------------------------------------------------------


def test_native_import_error_none_when_native_loaded():
    if not NATIVE_AVAILABLE:
        pytest.skip("native extension not available in this environment")
    assert f3d.native_import_error() is None


def test_native_import_error_returns_captured_exception(monkeypatch):
    boom = ImportError("DLL load failed while importing _forge3d: boom")
    monkeypatch.setattr(native_mod, "NATIVE_IMPORT_ERROR", boom)
    assert f3d.native_import_error() is boom


def test_load_native_records_root_cause(monkeypatch):
    import importlib

    boom = ImportError("simulated ABI mismatch")

    def _raise(name):
        raise boom

    monkeypatch.setattr(importlib, "import_module", _raise)
    assert native_mod._load_native() is None
    try:
        assert native_mod.NATIVE_IMPORT_ERROR is boom
    finally:
        # Restore truthful state for later tests in this process.
        monkeypatch.undo()
        native_mod.refresh_native_module()


# ---------------------------------------------------------------------------
# forge3d.__getattr__ for native-only symbols
# ---------------------------------------------------------------------------


def test_missing_native_symbol_message_includes_root_cause(monkeypatch):
    boom = ImportError("DLL load failed: The specified module could not be found")
    monkeypatch.setattr(f3d, "_NATIVE_MODULE", None)
    monkeypatch.setattr(native_mod, "NATIVE_IMPORT_ERROR", boom)

    with pytest.raises(AttributeError) as excinfo:
        f3d.__getattr__("Scene")

    message = str(excinfo.value)
    assert "forge3d.Scene requires the native extension" in message
    assert repr(boom) in message
    assert "maturin develop --release" in message


def test_missing_native_symbol_without_cause_mentions_not_built(monkeypatch):
    monkeypatch.setattr(f3d, "_NATIVE_MODULE", None)
    monkeypatch.setattr(native_mod, "NATIVE_IMPORT_ERROR", None)

    with pytest.raises(AttributeError, match="is not built"):
        f3d.__getattr__("TerrainRenderer")


def test_feature_gated_symbol_message_mentions_cargo_feature(monkeypatch):
    class _FakeNative:
        pass  # deliberately lacks every symbol

    monkeypatch.setattr(f3d, "_NATIVE_MODULE", _FakeNative())

    with pytest.raises(AttributeError) as excinfo:
        f3d.__getattr__("copc_read_node_points")

    message = str(excinfo.value)
    assert "not provided by this build" in message
    assert "Cargo feature" in message


def test_unknown_attribute_still_plain_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        f3d.__getattr__("definitely_not_a_forge3d_symbol")


def test_hasattr_probe_stays_false_not_raising(monkeypatch):
    """hasattr() probes across the repo must keep returning False, not raise."""
    monkeypatch.setattr(f3d, "_NATIVE_MODULE", None)
    monkeypatch.setattr(native_mod, "NATIVE_IMPORT_ERROR", None)
    monkeypatch.delitem(f3d.__dict__, "Scene", raising=False)
    assert hasattr(f3d, "Scene") is False


# ---------------------------------------------------------------------------
# device_probe() status taxonomy
# ---------------------------------------------------------------------------


def _assert_diagnostic_fields(probe: dict, expected_status: str) -> None:
    assert probe["status"] == expected_status
    assert isinstance(probe.get("reason"), str) and probe["reason"]
    assert isinstance(probe.get("remediation"), str) and probe["remediation"]


def test_device_probe_native_missing(monkeypatch):
    monkeypatch.setattr(gpu_mod, "get_native_module", lambda: None)
    probe = gpu_mod.device_probe()
    _assert_diagnostic_fields(probe, "native_missing")
    assert "maturin develop" in probe["remediation"]


def test_device_probe_native_missing_preserves_import_error(monkeypatch):
    boom = ImportError("simulated missing DLL")
    monkeypatch.setattr(gpu_mod, "get_native_module", lambda: None)
    monkeypatch.setattr(native_mod, "NATIVE_IMPORT_ERROR", boom)
    probe = gpu_mod.device_probe()
    _assert_diagnostic_fields(probe, "native_missing")
    assert repr(boom) in probe["reason"]


def test_device_probe_probe_error(monkeypatch):
    class _BrokenNative:
        @staticmethod
        def device_probe(*args):
            raise RuntimeError("simulated driver crash")

        enumerate_adapters = None
        get_device = None

    monkeypatch.setattr(gpu_mod, "get_native_module", lambda: _BrokenNative())
    probe = gpu_mod.device_probe()
    _assert_diagnostic_fields(probe, "probe_error")
    assert "simulated driver crash" in probe["reason"]


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension not available")
def test_device_probe_no_adapter_for_webgpu_backend():
    """BROWSER_WEBGPU never exposes a native desktop adapter, so this
    deterministically exercises the no_adapter branch on any host."""
    probe = f3d.device_probe("webgpu")
    _assert_diagnostic_fields(probe, "no_adapter")
    assert "WGPU_BACKENDS" in probe["remediation"]


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension not available")
@pytest.mark.apple_metal_physical
def test_device_probe_ok_carries_adapter_fields():
    probe = f3d.device_probe()
    if probe.get("status") != "ok":
        pytest.skip(f"no adapter on this host: {probe.get('reason', 'unknown')}")
    assert probe["name"]
    assert probe["backend"]
    assert isinstance(probe["software_fallback"], bool)


# ---------------------------------------------------------------------------
# GPU acquisition failures raise RuntimeError, never PanicException
# ---------------------------------------------------------------------------


def _run_engine_info_subprocess(extra_env: dict) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("WGPU_BACKENDS", None)
    env.pop("WGPU_BACKEND", None)
    env.pop("FORGE3D_DETERMINISTIC", None)
    # Prepend the source tree ONLY when it actually contains the built native
    # module (local `maturin develop` layout). On a clean CI checkout the
    # source tree has no _forge3d.* — prepending it would shadow the installed
    # wheel and turn every child into an ImportError instead of exercising the
    # real adapter-acquisition path (found by the first exhaustive-lane CI run).
    repo_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
    if glob.glob(os.path.join(repo_python, "forge3d", "_forge3d*")):
        env["PYTHONPATH"] = os.pathsep.join(
            [repo_python, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [repo_python]
        )
    env.update(extra_env)
    code = (
        "from forge3d import _forge3d\n"
        "_forge3d.engine_info()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension not available")
def test_no_adapter_raises_runtime_error_not_panic():
    """Pin the backend to BROWSER_WEBGPU (never available on native desktop):
    even the software fallback cannot resolve, so acquisition must fail with a
    catchable RuntimeError carrying remediation text — not a PanicException."""
    result = _run_engine_info_subprocess({"WGPU_BACKENDS": "webgpu"})
    assert result.returncode != 0
    assert "PanicException" not in result.stderr
    assert "RuntimeError" in result.stderr
    assert "No suitable GPU adapter" in result.stderr
    assert "WGPU_BACKENDS" in result.stderr


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension not available")
def test_deterministic_without_backend_pin_raises_runtime_error():
    result = _run_engine_info_subprocess({"FORGE3D_DETERMINISTIC": "1"})
    assert result.returncode != 0
    assert "PanicException" not in result.stderr
    assert "RuntimeError" in result.stderr
    assert "FORGE3D_DETERMINISTIC" in result.stderr


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native extension not available")
@pytest.mark.apple_metal_physical
def test_engine_info_reports_fallback_honesty():
    """On hosts with any adapter (hardware or WARP/lavapipe), engine_info must
    succeed and disclose whether a software fallback adapter is in use."""
    result = _run_engine_info_subprocess({})
    if result.returncode != 0:
        pytest.skip(f"no adapter at all on this host: {result.stderr.strip()[:200]}")
    info = native_mod.get_native_module().engine_info()
    assert "software_fallback" in info
    assert isinstance(info["software_fallback"], bool)
    assert "device_type" in info
