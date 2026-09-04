from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import _terrain_runtime as terrain_runtime
import pytest
from forge3d import map_scene


def test_running_on_unsupported_hosted_macos_ci_detects_github_actions(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(terrain_runtime.platform, "system", lambda: "Darwin")

    assert terrain_runtime._running_on_unsupported_hosted_macos_ci() is True


def test_running_on_unsupported_hosted_macos_ci_ignores_local_macos(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(terrain_runtime.platform, "system", lambda: "Darwin")

    assert terrain_runtime._running_on_unsupported_hosted_macos_ci() is False


def test_hosted_windows_gpu_override_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(terrain_runtime.platform, "system", lambda: "Windows")

    assert terrain_runtime._running_on_unsupported_hosted_windows_ci() is True
    monkeypatch.setenv("FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN", "1")
    assert terrain_runtime._running_on_unsupported_hosted_windows_ci() is False


def test_mapscene_allows_the_explicit_hosted_windows_gpu_override(monkeypatch) -> None:
    map_scene._terrain_renderer_runtime_available.cache_clear()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN", "1")
    monkeypatch.setattr(map_scene.platform, "system", lambda: "Windows")
    fake_forge3d = types.SimpleNamespace(
        Session=lambda **_: object(),
        TerrainRenderer=lambda *_: object(),
        MaterialSet=object,
        IBL=object,
        TerrainRenderParams=object,
        has_gpu=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "forge3d", fake_forge3d)
    monkeypatch.setattr(
        map_scene.subprocess,
        "run",
        lambda *_, **__: types.SimpleNamespace(returncode=0),
    )
    try:
        assert map_scene._terrain_renderer_runtime_available() is True
    finally:
        map_scene._terrain_renderer_runtime_available.cache_clear()


def test_terrain_rendering_available_short_circuits_on_hosted_macos_ci(monkeypatch) -> None:
    terrain_runtime._terrain_rendering_unavailable_reason.cache_clear()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(terrain_runtime.platform, "system", lambda: "Darwin")

    def fail_if_called():
        raise AssertionError("terrain GPU probe should not run on hosted macOS CI")

    monkeypatch.setattr(terrain_runtime.f3d, "has_gpu", fail_if_called)

    try:
        assert terrain_runtime.terrain_rendering_available() is False
    finally:
        terrain_runtime._terrain_rendering_unavailable_reason.cache_clear()


def test_terrain_rendering_available_uses_child_probe(monkeypatch) -> None:
    terrain_runtime._terrain_rendering_unavailable_reason.cache_clear()
    # This unit test exercises the child-probe plumbing itself; the hosted-CI
    # blanket guards would short-circuit before subprocess.run on GitHub
    # runners, so disable them explicitly for the mock scenario.
    monkeypatch.setattr(
        terrain_runtime, "_running_on_unsupported_hosted_macos_ci", lambda: False
    )
    monkeypatch.setattr(
        terrain_runtime, "_running_on_unsupported_hosted_windows_ci", lambda: False
    )
    monkeypatch.setattr(terrain_runtime.f3d, "has_gpu", lambda: True)
    monkeypatch.setattr(terrain_runtime.f3d, "device_probe", lambda _: {"status": "ok"})
    monkeypatch.setattr(terrain_runtime, "_adapter_is_terrain_safe", lambda _: True)
    monkeypatch.setattr(terrain_runtime, "REQUIRED_SYMBOLS", ())

    class Result:
        returncode = 1

    child_call = {}

    def run_child(*_, **kwargs):
        child_call.update(kwargs)
        return Result()

    monkeypatch.setattr(terrain_runtime.subprocess, "run", run_child)

    try:
        assert terrain_runtime.terrain_rendering_available() is False
        child_pythonpath = child_call["env"]["PYTHONPATH"].split(terrain_runtime.os.pathsep)
        repo = terrain_runtime.Path(terrain_runtime.__file__).resolve().parents[1]
        assert child_pythonpath[0] == str(repo / "tests")
        assert str(repo / "python") not in child_pythonpath
    finally:
        terrain_runtime._terrain_rendering_unavailable_reason.cache_clear()


def test_nvidia_vulkan_terrain_constructor_child_smoke() -> None:
    """The public terrain constructor must return on a qualifying Vulkan GPU.

    Keep this in a fresh child process: the native GPU context is global and a
    prior test must not select another backend before this constructor seam is
    exercised.  The children run the DEFAULT instance-flag configuration, so
    ambient ``WGPU_DEBUG``/``WGPU_VALIDATION`` overrides and Vulkan loader
    layer filters are stripped from their environment — the historic
    regression (validation layer crashing on debug-info-laden terrain SPIR-V)
    only reproduces under specific flag combinations and must not be masked
    or spuriously reintroduced by the invoking shell.  A non-qualifying host
    is an honest absence for this physical adapter regression gate, not a
    synthetic pass; ``device_probe`` never raises, so a probe child that
    exits nonzero, times out, or prints no JSON is an operational failure,
    not adapter absence.
    """

    env = dict(os.environ)
    for override in (
        "WGPU_DEBUG",
        "WGPU_VALIDATION",
        "WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER",
        "VK_INSTANCE_LAYERS",
        "VK_LOADER_LAYERS_ENABLE",
        "VK_LOADER_LAYERS_DISABLE",
    ):
        env.pop(override, None)
    env.update(WGPU_BACKENDS="vulkan", WGPU_BACKEND="vulkan", PYTHONUNBUFFERED="1")
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                "import json, forge3d as f3d; print(json.dumps(f3d.device_probe('vulkan')))",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "NVIDIA Vulkan adapter probe child timed out after 120s; "
            "device_probe reports absence as a status dict, so a hang is an "
            f"operational failure: stdout={exc.stdout!r} stderr={exc.stderr!r}",
            pytrace=False,
        )
    if probe.returncode != 0:
        exit_hex = f"0x{probe.returncode & 0xFFFFFFFF:08X}"
        pytest.fail(
            "NVIDIA Vulkan adapter probe child failed; device_probe never "
            f"raises, so a nonzero exit is an operational failure: "
            f"exit={probe.returncode} ({exit_hex}) stdout={probe.stdout!r} "
            f"stderr={probe.stderr!r}",
            pytrace=False,
        )
    try:
        adapter = json.loads(probe.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        pytest.fail(
            "NVIDIA Vulkan adapter probe child exited 0 without a JSON "
            f"probe dict: stdout={probe.stdout!r} stderr={probe.stderr!r}",
            pytrace=False,
        )
    if not isinstance(adapter, dict) or adapter.get("status") not in ("ok", "no_adapter"):
        pytest.fail(
            "NVIDIA Vulkan adapter probe returned no structurally valid "
            "verdict (expected a dict with status 'ok' or 'no_adapter'): "
            f"{adapter!r} stderr={probe.stderr!r}",
            pytrace=False,
        )
    vendor = adapter.get("vendor")
    if adapter.get("status") == "ok" and type(vendor) is not int:
        pytest.fail(
            "NVIDIA Vulkan adapter probe returned status 'ok' without an "
            f"integer vendor id: {adapter!r}",
            pytrace=False,
        )
    if not (
        adapter.get("status") == "ok"
        and str(adapter.get("backend", "")).lower() == "vulkan"
        and str(adapter.get("device_type", "")).lower() == "discretegpu"
        and vendor == 0x10DE
        and not bool(adapter.get("software_fallback", True))
    ):
        pytest.skip(f"qualifying NVIDIA Vulkan adapter absent: {adapter!r}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                "import forge3d as f3d; "
                "session = f3d.Session(window=False); "
                "f3d.TerrainRenderer(session)",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "NVIDIA Vulkan TerrainRenderer constructor child timed out after 120s: "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}",
            pytrace=False,
        )
    exit_hex = f"0x{result.returncode & 0xFFFFFFFF:08X}"
    assert result.returncode == 0, (
        "NVIDIA Vulkan TerrainRenderer constructor child failed: "
        f"exit={result.returncode} ({exit_hex}) stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
