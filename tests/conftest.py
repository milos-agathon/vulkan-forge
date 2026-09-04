# Ensure `import forge3d` works from a fresh clone:
# 1) Put repo/python on sys.path so the package is importable without prior install.
# 2) If the native extension is missing, auto-build once.
#    - In a virtualenv we use `maturin develop --release`.
#    - Outside a virtualenv we build a wheel and extract `_forge3d` into
#      `python/forge3d/` so the repo-local package can import it.
# Set FORGE3D_NO_BOOTSTRAP=1 to disable autobuild (e.g., in CI with preinstalled wheel).
import importlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_installed_wheel_paths(package_path: Path, native_path: Path) -> None:
    """Fail closed when an installed-wheel lane resolves repo-local Python code."""

    source_package = _repo_root() / "python" / "forge3d"
    if _is_within(package_path, source_package):
        raise RuntimeError(
            "FORGE3D_TEST_INSTALLED_WHEEL resolved the repo-local Python package: "
            f"{package_path}"
        )
    if _is_within(native_path, source_package):
        raise RuntimeError(
            "FORGE3D_TEST_INSTALLED_WHEEL resolved a repo-local native extension: "
            f"{native_path}"
        )
    if not package_path.is_file() or not native_path.is_file():
        raise RuntimeError(
            "FORGE3D_TEST_INSTALLED_WHEEL requires importable installed package and native files: "
            f"package={package_path}, native={native_path}"
        )


def _require_installed_wheel_runtime() -> None:
    if os.environ.get("FORGE3D_NO_BOOTSTRAP") != "1":
        raise RuntimeError(
            "FORGE3D_TEST_INSTALLED_WHEEL requires FORGE3D_NO_BOOTSTRAP=1"
        )
    package = importlib.import_module("forge3d")
    native = importlib.import_module("forge3d._forge3d")
    _validate_installed_wheel_paths(Path(package.__file__), Path(native.__file__))


def _ensure_python_path():
    repo = _repo_root()
    pkg_dir = repo / "python"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    # Also make test helpers (e.g. _license_test_keys) importable.
    tests_dir = repo / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))


def _install_maturin():
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "maturin"], check=True)


def _in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _maturin_develop():
    repo = _repo_root()
    env = os.environ.copy()
    subprocess.run(
        ["maturin", "develop", "--release"],
        cwd=str(repo),
        env=env,
        check=True,
    )


def _maturin_build_wheel() -> Path:
    repo = _repo_root()
    out_dir = repo / "dist-test"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    subprocess.run(
        ["maturin", "build", "--profile", "release-lto", "-o", str(out_dir)],
        cwd=str(repo),
        env=env,
        check=True,
    )
    wheels = sorted(out_dir.glob("forge3d-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError("maturin build produced no forge3d wheel")
    return wheels[-1]


def _extract_native_from_wheel(wheel_path: Path) -> Path:
    pkg_dir = _repo_root() / "python" / "forge3d"
    native_members: list[str] = []
    with zipfile.ZipFile(wheel_path) as wheel:
        for name in wheel.namelist():
            if name.startswith("forge3d/_forge3d") and Path(name).suffix.lower() in {".pyd", ".so", ".dylib"}:
                native_members.append(name)
        if not native_members:
            raise RuntimeError(f"No native forge3d extension found in {wheel_path.name}")
        native_member = sorted(native_members)[0]
        destination = pkg_dir / Path(native_member).name
        with wheel.open(native_member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return destination


def _clear_forge3d_modules() -> None:
    for name in list(sys.modules):
        if name == "forge3d" or name.startswith("forge3d."):
            sys.modules.pop(name, None)


def _native_runtime_ready(module) -> bool:
    native = importlib.import_module("forge3d._native")
    return bool(getattr(native, "NATIVE_AVAILABLE", False)) and all(
        hasattr(module, name) for name in ("Session", "TerrainRenderer", "MaterialSet", "IBL")
    )


def _bootstrap_native_extension() -> str:
    _install_maturin()
    if _in_virtualenv():
        _maturin_develop()
        return "maturin develop"

    wheel_path = _maturin_build_wheel()
    native_path = _extract_native_from_wheel(wheel_path)
    return f"maturin build ({wheel_path.name} -> {native_path.name})"


def _needs_build_from_exc(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        isinstance(exc, ModuleNotFoundError) and "forge3d" in msg
    ) or (
        isinstance(exc, ImportError) and ("_forge3d" in msg or "forge3d._forge3d" in msg)
    )


def _license_api():
    module = importlib.import_module("forge3d._license")
    return module._reset_license_state, module.set_license_key


@pytest.fixture
def pro_license():
    """Enable a deterministic test license for Pro-gated APIs."""
    from _license_test_keys import sign_test_key

    _reset_license_state, set_license_key = _license_api()
    _reset_license_state()
    set_license_key(sign_test_key("PRO", "20991231"))
    try:
        yield
    finally:
        set_license_key("")


def pytest_configure(config):
    """Register custom markers for Workstream I tasks and P5.7."""

    config.addinivalue_line(
        "markers",
        "gpu_lane: TESSELLA physical-GPU acceptance gate",
    )
    config.addinivalue_line(
        "markers", "viewer: tests for interactive viewer functionality (Workstream I1)"
    )
    config.addinivalue_line(
        "markers", "offscreen: tests for offscreen rendering and Jupyter integration (Workstream I2)"
    )
    config.addinivalue_line("markers", "opbr: tests for PBR rendering")
    config.addinivalue_line("markers", "olighting: tests for lighting")
    config.addinivalue_line(
        "markers", "interactive_viewer: tests requiring interactive viewer"
    )
    config.addinivalue_line("markers", "pro: tests requiring a Pro license")
    config.addinivalue_line("markers", "slow: slow tests")


def pytest_collection_modifyitems(config, items):
    """Auto-tag tests that require the Pro license fixture."""

    del config

    for item in items:
        if "pro_license" in item.fixturenames:
            item.add_marker(pytest.mark.pro)


# Track P5.7 test results for artifact generation
_p57_results = {}


def pytest_runtest_logreport(report):
    """Track P5.7 test results."""

    if report.when == "call":
        if (
            "test_p5_ssao" in report.nodeid
            or "test_p5_ssgi" in report.nodeid
            or "test_p5_ssr" in report.nodeid
        ):
            _p57_results[report.nodeid] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    """Write p5_PASS.txt if all P5.7 tests passed."""

    import hashlib
    import json
    from datetime import datetime

    del session, exitstatus

    if not _p57_results:
        return

    passed = [k for k, v in _p57_results.items() if v == "passed"]
    failed = [k for k, v in _p57_results.items() if v == "failed"]
    skipped = [k for k, v in _p57_results.items() if v == "skipped"]

    repo_root = _repo_root()
    report_dir = repo_root / "reports" / "p5"
    report_dir.mkdir(parents=True, exist_ok=True)

    if failed:
        fail_file = report_dir / "p5_FAIL.txt"
        with open(fail_file, "w", encoding="utf-8") as f:
            f.write("P5.7 Acceptance Tests FAILED\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Passed: {len(passed)}\n")
            f.write(f"Failed: {len(failed)}\n")
            f.write(f"Skipped: {len(skipped)}\n")
            f.write("\nFailed tests:\n")
            for test_name in failed:
                f.write(f"  - {test_name}\n")
    elif passed:
        metrics = {
            "passed_count": len(passed),
            "skipped_count": len(skipped),
            "timestamp": datetime.now().isoformat(),
            "tests": passed,
        }
        metrics_hash = hashlib.sha256(
            json.dumps(metrics, sort_keys=True).encode()
        ).hexdigest()

        pass_file = report_dir / "p5_PASS.txt"
        with open(pass_file, "w", encoding="utf-8") as f:
            f.write("P5.7 Acceptance Tests PASSED\n")
            f.write(f"Timestamp: {metrics['timestamp']}\n")
            f.write(f"Passed: {len(passed)}\n")
            f.write(f"Skipped: {len(skipped)}\n")
            f.write(f"Metrics hash: {metrics_hash}\n")
            f.write("\nPassed tests:\n")
            for test_name in passed:
                f.write(f"  - {test_name}\n")


def pytest_sessionstart(session):
    del session

    # Always make test helpers importable, even in no-bootstrap mode.
    tests_dir = _repo_root() / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    if os.environ.get("FORGE3D_TEST_INSTALLED_WHEEL") == "1":
        _require_installed_wheel_runtime()
        return

    if os.environ.get("FORGE3D_NO_BOOTSTRAP") == "1":
        return

    _ensure_python_path()
    needs_bootstrap = False
    try:
        import forge3d

        if _native_runtime_ready(forge3d):
            return
        needs_bootstrap = True
    except Exception as exc:
        if not _needs_build_from_exc(exc):
            raise
        needs_bootstrap = True

    if not needs_bootstrap:
        return

    _clear_forge3d_modules()
    _ensure_python_path()

    try:
        mode = _bootstrap_native_extension()
        importlib.invalidate_caches()
        _clear_forge3d_modules()
        _ensure_python_path()
        import forge3d

        if not _native_runtime_ready(forge3d):
            raise RuntimeError("forge3d bootstrap built the extension but native runtime is still unavailable")
        print(f"forge3d bootstrap: built via {mode}", flush=True)
    except Exception as build_exc:
        raise RuntimeError(
            "forge3d bootstrap failed. Ensure Rust toolchain and (on Windows) "
            f"MSVC Build Tools are installed. Original error: {build_exc}"
        ) from build_exc
