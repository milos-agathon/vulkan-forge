"""Smoke tests for package installation metadata and public API."""

from importlib.metadata import distribution, metadata
from pathlib import Path
import re
import sys

import pytest

from tests.conftest import _validate_installed_wheel_paths

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None


def _load_project_urls(pyproject: Path) -> dict[str, str]:
    """Return the ``[project.urls]`` table without requiring Python 3.11+."""
    if tomllib is not None:
        with pyproject.open("rb") as fh:
            return tomllib.load(fh)["project"]["urls"]

    urls: dict[str, str] = {}
    in_urls = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[project.urls]":
            in_urls = True
            continue
        if in_urls and line.startswith("["):
            break
        if not in_urls:
            continue

        match = re.match(r'^"?(.+?)"?\s*=\s*"(.+)"$', line)
        if match:
            urls[match.group(1)] = match.group(2)

    if not urls:
        raise AssertionError("No [project.urls] table found in pyproject.toml")
    return urls


def test_load_project_urls_falls_back_without_tomllib(monkeypatch, tmp_path):
    """Python 3.10 still loads project URLs without stdlib tomllib."""

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "forge3d"

[project.urls]
Homepage = "https://example.com"
"Bug Tracker" = "https://example.com/issues"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys.modules[__name__], "tomllib", None)

    assert _load_project_urls(pyproject) == {
        "Homepage": "https://example.com",
        "Bug Tracker": "https://example.com/issues",
    }


def test_python_version_floor():
    """forge3d requires Python 3.10+."""
    assert sys.version_info >= (3, 10), "forge3d requires Python 3.10+"


def test_import_forge3d():
    """Package imports without error and exposes a version."""
    import forge3d

    assert forge3d.__version__


def test_public_api_surface():
    """Key public symbols are accessible from the package root."""
    import forge3d

    required = [
        "open_viewer",
        "open_viewer_async",
        "Renderer",
        "RendererConfig",
        "MapPlate",
        "Legend",
        "ScaleBar",
        "has_gpu",
        "enumerate_adapters",
        "fetch_dataset",
        "set_license_key",
        "LicenseError",
        "__version__",
    ]
    for name in required:
        assert hasattr(forge3d, name), f"Missing public symbol: {name}"
    assert not hasattr(forge3d, "RenderView"), "RenderView should not be exported from package root"


def test_fetch_dataset_alias_matches_datasets_module():
    """The package root keeps the documented fetch_dataset alias."""
    import forge3d

    assert callable(forge3d.fetch_dataset)
    assert forge3d.fetch_dataset is forge3d.datasets.fetch
    assert not hasattr(forge3d, "fetch"), "Root package should expose fetch_dataset, not fetch"


def test_version_consistency():
    """Package version stays in sync with pyproject.toml."""
    import forge3d

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not available in this environment")

    match = re.search(
        r'^version\s*=\s*"(.+?)"',
        pyproject.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "No version entry found in pyproject.toml"
    assert forge3d.__version__ == match.group(1)


def test_enumerate_adapters_smoke():
    """Adapter enumeration should not crash, even on GPU-less CI."""
    import forge3d

    adapters = forge3d.enumerate_adapters()
    assert isinstance(adapters, list)


def test_legacy_render_api_removed():
    """The legacy render_raster/render_polygons/render_raytrace_mesh API is gone."""
    import forge3d

    for name in ("render_raster", "render_polygons", "render_raytrace_mesh"):
        assert not hasattr(forge3d, name), f"Legacy API should be removed: {name}"


def test_installed_project_urls_match_public_metadata():
    """Installed metadata should point at the live repository and docs."""

    meta = metadata("forge3d")
    project_urls = meta.get_all("Project-URL") or meta.get_all("Project-Url") or []
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not available in this environment")

    expected_urls = _load_project_urls(pyproject)

    for label, url in expected_urls.items():
        assert f"{label}, {url}" in project_urls
    assert all("github.com/forge3d/forge3d" not in value for value in project_urls)


def test_installs_interactive_viewer_console_script():
    """The wheel exposes the interactive viewer command via console_scripts."""
    entry_points = distribution("forge3d").entry_points
    assert any(
        ep.group == "console_scripts"
        and ep.name == "interactive_viewer"
        and ep.value == "forge3d._viewer_entry:main"
        for ep in entry_points
    )


def test_sidera_third_party_notice_is_shipped_with_embedded_coefficients():
    """The MIT notice travels with wheels containing ``moon_terms.bin``."""
    import forge3d

    root = Path(__file__).resolve().parent.parent
    source_package = root / "python" / "forge3d"
    package_path = Path(forge3d.__file__).resolve()
    if package_path.is_relative_to(source_package.resolve()):
        pyproject = root / "pyproject.toml"
        if tomllib is not None:
            with pyproject.open("rb") as fh:
                license_files = tomllib.load(fh)["project"]["license-files"]
            assert "assets/astro/THIRD_PARTY_NOTICES.md" in license_files
        else:
            project = pyproject.read_text(encoding="utf-8").split("[project]", 1)[1]
            project = project.split("\n[", 1)[0]
            assert '"assets/astro/THIRD_PARTY_NOTICES.md"' in project
        return

    installed = distribution("forge3d")
    members = [
        member
        for member in installed.files or ()
        if member.name == "THIRD_PARTY_NOTICES.md"
    ]
    assert len(members) == 1, (
        "installed forge3d wheel omits assets/astro/THIRD_PARTY_NOTICES.md"
    )
    notice = installed.locate_file(members[0]).read_text(encoding="utf-8")
    assert "Copyright (c) 2016 Commenthol" in notice
    assert "Permission is hereby granted" in notice


def test_installed_wheel_path_gate_rejects_repo_local_package(tmp_path):
    repo_package = Path(__file__).resolve().parent.parent / "python" / "forge3d"
    native = tmp_path / "site-packages" / "forge3d" / "_forge3d.abi3.so"
    native.parent.mkdir(parents=True)
    native.touch()

    with pytest.raises(RuntimeError, match="repo-local Python package"):
        _validate_installed_wheel_paths(repo_package / "__init__.py", native)


def test_installed_wheel_path_gate_rejects_repo_local_native(tmp_path):
    package = tmp_path / "site-packages" / "forge3d" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.touch()
    repo_native = (
        Path(__file__).resolve().parent.parent
        / "python"
        / "forge3d"
        / "_forge3d.abi3.so"
    )

    with pytest.raises(RuntimeError, match="repo-local native extension"):
        _validate_installed_wheel_paths(package, repo_native)


def test_installed_wheel_path_gate_accepts_external_package_and_native(tmp_path):
    package = tmp_path / "site-packages" / "forge3d" / "__init__.py"
    native = tmp_path / "site-packages" / "forge3d" / "_forge3d.abi3.so"
    package.parent.mkdir(parents=True)
    package.touch()
    native.touch()

    _validate_installed_wheel_paths(package, native)
