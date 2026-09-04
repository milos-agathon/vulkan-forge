# tests/test_no_silent_degradation.py
# CENSOR Task 13: the CI honesty gate. One test function per lettered gate:
#   (a) committed RenderCertificates carry no un-allowlisted degradation
#   (b) zero raw wgpu allocation sites bypass the tracked ledger
#   (c) every Cargo feature is referenced, and the CI --features list is curated
#   (d) the wheel ships the features its public APIs need; the built-in CRS
#       engine is authoritative, while optional PROJ and GEOS remain honest
#   (e) the full profile accounts for every tracked test, while the routine fast
#       profile retains every mandatory CENSOR truth contract
# RELEVANT FILES: scripts/ci_pytest_lane.py, tests/UNRUN.toml,
#   tests/degradation_allowlist.toml, tests/allocation_allowlist.toml,
#   tests/_toml_compat.py, Cargo.toml, pyproject.toml, .github/workflows/ci.yml
"""Static + behavioural honesty gates for CENSOR."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from _toml_compat import load_toml
from tests._golden_variants import (
    assert_nvidia_vulkan_golden_adapter,
    selected_golden_path,
    selected_golden_variant,
)

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
CERT_DIR = TESTS / "golden" / "certificates"


def _cargo_alias_features(alias: str | list[str]) -> set[str]:
    tokens = shlex.split(alias) if isinstance(alias, str) else alias
    index = tokens.index("--features")
    return set(tokens[index + 1].split(","))

# Make sibling helpers importable regardless of pytest rootdir insertion order.
for _p in (str(TESTS), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_allocation_gate import _raw_sites  # noqa: E402  (reuse the source gate)
import ci_pytest_lane  # noqa: E402  (validation profiles are the source of truth)


# ---------------------------------------------------------------------------
# (a) certificate degradations
# ---------------------------------------------------------------------------
def test_a_committed_certificates_have_no_unallowlisted_degradations():
    certs = sorted(CERT_DIR.glob("*.json"))
    assert certs, "no committed certificates found -- expected tests/golden/certificates/*.json"

    allow = load_toml(TESTS / "degradation_allowlist.toml").get("entries", [])
    allowed = {}
    for entry in allow:
        assert date.fromisoformat(entry["expires"]) >= date.today(), f"expired degradation allowlist entry: {entry}"
        allowed[(entry["kind"], entry["name"])] = entry

    offenders = []
    for cert in certs:
        data = json.loads(cert.read_text(encoding="utf-8"))
        for deg in data.get("degradations", []) or []:
            key = (deg.get("kind"), deg.get("name"))
            if key not in allowed:
                offenders.append(f"{cert.name}: {key} -> {deg.get('consequence')}")

    assert offenders == [], "certificates carry un-allowlisted degradations:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# (b) source allocation gate (reused)
# ---------------------------------------------------------------------------
def test_b_zero_raw_allocation_sites():
    allow = load_toml(TESTS / "allocation_allowlist.toml")["entries"]
    allowed = {e["site"].rsplit(":", 1)[0] for e in allow}
    stray = [s for s in _raw_sites() if s.rsplit(":", 1)[0] not in allowed]
    assert stray == [], f"raw wgpu allocation sites bypass the tracked ledger: {stray}"


# ---------------------------------------------------------------------------
# (c) feature gate
# ---------------------------------------------------------------------------
# The single source of truth for what CI's `cargo check`/`cargo test`/`cargo doc`
# compile on every Rust CI platform. Platform-bound and wheel-only features are
# exercised by separate commands/jobs and verified below.
PORTABLE_CI_CARGO_FEATURES = {
    "default",  # baseline: images + enable-gpu-instancing + enable-staging-rings
    "async_readback",
    "copc_laz",
    "cog_streaming",
    "gis-remote",
    "geos-topology",
    "weighted-oit",
    "wsI_bigbuf",
    "wsI_double_buf",
    "enable-pbr",
    "enable-tbn",
    "enable-normal-mapping",
    "enable-hdr-offscreen",
    "enable-renderer-config",
    "enable-staging-rings",
    "shader-contract-asserts",
}
DEDICATED_SYSTEM_FEATURES = {"proj"}
DEDICATED_ACCEPTANCE_FEATURES = {"atmosphere-bake"}


def _cargo_features() -> set[str]:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    section = re.search(r"\[features\](.*?)(?:\n\[)", text, re.DOTALL)
    assert section, "could not locate [features] in Cargo.toml"
    names = set()
    for line in section.group(1).splitlines():
        stripped = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_\-]+)\s*=", stripped)
        if m:
            names.add(m.group(1))
    return names


def _cargo_feature_table() -> dict[str, list[str]]:
    """Parse Cargo.toml's [features] table with a regex.

    Deliberately NOT load_toml: on Python 3.10 the tiny _toml_compat fallback
    parser only understands the UNRUN/allowlist schema and returns a dict with
    no "features" key, which made this gate error (not fail honestly) on every
    3.10 CI leg — unseen until the exhaustive lane first ran there.
    """
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    section = re.search(r"\[features\](.*?)(?:\n\[)", text, re.DOTALL)
    assert section, "could not locate [features] in Cargo.toml"
    table: dict[str, list[str]] = {}
    for m in re.finditer(
        r"^([A-Za-z0-9_\-]+)\s*=\s*\[([^\]]*)\]", section.group(1), re.MULTILINE
    ):
        table[m.group(1)] = re.findall(r'"([^"]+)"', m.group(2))
    return table


def _feature_closure(features: set[str]) -> set[str]:
    table = _cargo_feature_table()
    closure = set(features)
    pending = list(features)
    while pending:
        feature = pending.pop()
        for dependency in table.get(feature, []):
            if dependency in table and dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return closure


def _feature_referenced(feat: str) -> bool:
    needle = f'feature = "{feat}"'
    for base in ("src", "tests", "benches"):
        d = ROOT / base
        if not d.exists():
            continue
        for path in d.rglob("*.rs"):
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                return True
    build_rs = ROOT / "build.rs"
    if build_rs.exists() and needle in build_rs.read_text(encoding="utf-8", errors="ignore"):
        return True
    return False


def test_c_every_feature_referenced_and_ci_list_curated():
    declared = _cargo_features()

    # Every non-`default` feature must be referenced somewhere in Rust source.
    unreferenced = sorted(f for f in declared if f != "default" and not _feature_referenced(f))
    assert unreferenced == [], f"declared Cargo features with no `feature = \"..\"` reference (dead advertising): {unreferenced}"

    assert PORTABLE_CI_CARGO_FEATURES <= declared, (
        f"CI feature set names undeclared features: {PORTABLE_CI_CARGO_FEATURES - declared}"
    )
    assert DEDICATED_SYSTEM_FEATURES <= declared

    # Portable check/test/doc commands must agree exactly, while PROJ has a
    # dedicated Ubuntu check with its system dependencies installed.
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lists = re.findall(r"--features\s+([A-Za-z0-9_,\-]+)", ci_yml)
    assert lists, "no cargo --features lists found in ci.yml"
    portable_lists = [raw for raw in lists if "default" in raw.split(",") and len(raw.split(",")) > 1]
    assert len(portable_lists) >= 3, "expected portable cargo check/test/doc feature lists"
    for raw in portable_lists:
        got = set(raw.split(","))
        assert got == PORTABLE_CI_CARGO_FEATURES, (
            f"ci.yml --features {sorted(got)} != portable set {sorted(PORTABLE_CI_CARGO_FEATURES)}"
        )
        assert got <= declared, f"ci.yml advertises undeclared features: {got - declared}"
    assert any(set(raw.split(",")) == DEDICATED_SYSTEM_FEATURES for raw in lists), (
        "ci.yml lacks a dedicated native-PROJ compile check"
    )
    for package in ("libproj-dev", "libsqlite3-dev", "sqlite3", "pkg-config"):
        assert package in ci_yml, f"PROJ CI check does not install {package}"

    # The wheel's maturin list is the extension-module compile lane. Together,
    # portable/default closure + system lane + wheel lane must cover everything.
    maturin = _maturin_features()
    wheel_yml = (ROOT / ".github" / "workflows" / "build-wheel.yml").read_text(
        encoding="utf-8"
    )
    assert "uses: ./.github/workflows/build-wheel.yml" in ci_yml
    assert "PyO3/maturin-action" in wheel_yml, (
        "reusable CI wheel builder does not exercise maturin features"
    )
    assert any(
        set(raw.split(",")) == DEDICATED_ACCEPTANCE_FEATURES for raw in lists
    ), "ci.yml lacks a dedicated atmosphere-bake acceptance check"
    covered = (
        _feature_closure(PORTABLE_CI_CARGO_FEATURES)
        | DEDICATED_SYSTEM_FEATURES
        | DEDICATED_ACCEPTANCE_FEATURES
        | maturin
    )
    assert covered == declared, f"declared features not compiled by any CI lane: {sorted(declared - covered)}"

    # Routine linting stays deliberately small. Explicit acceptance linting
    # covers the portable surface plus extension-module without system PROJ.
    aliases = load_toml(ROOT / ".cargo" / "config.toml")["alias"]
    routine = _cargo_alias_features(aliases["forge3d-clippy"])
    acceptance = _cargo_alias_features(aliases["forge3d-clippy-acceptance"])
    assert routine == {
        "default",
        "extension-module",
        "cog_streaming",
        "shader-contract-asserts",
    }, (
        f"routine clippy expanded beyond the small contract: {sorted(routine)}"
    )
    assert acceptance == (
        PORTABLE_CI_CARGO_FEATURES
        | DEDICATED_ACCEPTANCE_FEATURES
        | {"extension-module"}
    ), (
        f"acceptance clippy feature drift: {sorted(acceptance)}"
    )


def test_clippy_alias_feature_parser_accepts_string_and_array_forms():
    expected = {"extension-module", "default", "enable-pbr"}
    for alias in [
        "clippy --workspace --features extension-module,default,enable-pbr -- -D warnings",
        [
            "clippy",
            "--workspace",
            "--features",
            "extension-module,default,enable-pbr",
            "--",
            "-D",
            "warnings",
        ],
    ]:
        assert _cargo_alias_features(alias) == expected


# ---------------------------------------------------------------------------
# (d) wheel gate
# ---------------------------------------------------------------------------
# Features the shipped wheel MUST compile in because documented public APIs
# depend on them at runtime.
WHEEL_REQUIRED_FEATURES = {
    "extension-module",
    "enable-tbn",
    "weighted-oit",
    "enable-gpu-instancing",
    "enable-staging-rings",
    "copc_laz",
    "cog_streaming",
    "gis-remote",
    # MENSURA ships real topology ops (pure-Rust `geo` crate) as a wheel
    # feature; the public forge3d.gis topology surface requires it.
    "geos-topology",
    # AETHER exposes an explicit offline bake API in the shipped wheel while
    # normal rendering still consumes its shipped LUT bank.
    "atmosphere-bake",
}


def test_d_aether_bake_api_is_feature_gated_and_runtime_surface_is_shipped():
    maturin = _maturin_features()
    assert "atmosphere-bake" in maturin
    init_source = (ROOT / "python" / "forge3d" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert '"atmosphere_bake_luts"' in init_source
    boundary = (ROOT / "src" / "py_functions" / "atmosphere.rs").read_text(
        encoding="utf-8"
    )
    assert "Custom AETHER LUT inputs require" in boundary
    assert "no nearest shipped table or legacy" in boundary


def _maturin_features() -> set[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"\[tool\.maturin\](.*?)(?:\n\[)", text, re.DOTALL)
    assert section, "could not locate [tool.maturin] in pyproject.toml"
    m = re.search(r"features\s*=\s*\[([^\]]*)\]", section.group(1))
    assert m, "could not locate maturin `features` list in pyproject.toml"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_d_wheel_features_and_native_gis_backends_are_honest():
    maturin = _maturin_features()
    missing = WHEEL_REQUIRED_FEATURES - maturin
    assert not missing, f"wheel omits features required by public APIs: {sorted(missing)}"

    # PROJ is deliberately NOT shipped (it links a C library). MENSURA's
    # built-in pure-Rust dispatcher is the authoritative runtime transform
    # engine; optional PROJ is a differential-test oracle only. A wheel must
    # therefore transform a supported pair without pyproj or a degradation.
    # geos-topology IS shipped (pure-Rust `geo` crate) and is asserted present
    # via WHEEL_REQUIRED_FEATURES above.
    assert "proj" not in maturin, (
        "proj is expected to be compiled OUT of the wheel"
    )

    import forge3d.crs as crs
    assert crs.proj_available() is True
    projected = crs.transform_coords([[1.0, 1.0]], "EPSG:4326", "EPSG:3857")
    assert projected.shape == (1, 2)
    assert abs(float(projected[0, 0])) > 100_000.0
    crs_source = (ROOT / "python" / "forge3d" / "crs.py").read_text(encoding="utf-8")
    assert "_native.CrsTransform.from_crs" in crs_source
    assert "never as a transform backend" in crs_source
    transform_source = crs_source.split("def transform_coords(", 1)[1].split(
        "\ndef reproject_geom", 1
    )[0]
    assert "transformer = _crs_transform" in transform_source
    assert "pyproj.Transformer" not in transform_source
    assert "pyproj.transform(" not in transform_source

    # geos-topology: even though the wheel now ships it, the Rust boundary keeps
    # an explicit require_topology_backend / BackendUnavailable gate so a minimal
    # build (feature absent) returns an honest error instead of a silent wrong
    # result. Assert that honest wiring still exists in the source.
    topo = (ROOT / "src" / "gis" / "geometry" / "topology.rs").read_text(encoding="utf-8")
    assert "require_topology_backend" in topo and "BackendUnavailable" in topo, (
        "geos-topology fallback is not visibly diagnostic-bearing in src/gis/geometry/topology.rs"
    )


# ---------------------------------------------------------------------------
# (e) UNRUN accounting
# ---------------------------------------------------------------------------
def _tracked_test_files() -> set[str]:
    working = {path.relative_to(ROOT).as_posix() for path in TESTS.glob("test_*.py")}
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "tests/test_*.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    tracked = {line.strip() for line in out.splitlines() if line.strip()}
    untracked = sorted(working - tracked)
    assert untracked == [], f"test files exist locally but would disappear from CI: {untracked}"
    return working


def _explicit_lane_files() -> set[str]:
    """Files a non-default CI lane runs explicitly (golden lane) or by marker (viewer lane)."""
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # Golden lane: every `tests/<file>.py` token that appears in a pytest command.
    golden = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", ci_yml))
    # Interactive viewer lane runs `pytest tests/ -m interactive_viewer`; the
    # owning files are those carrying the marker.
    viewer = set()
    if "-m interactive_viewer" in ci_yml:
        for path in _tracked_test_files():
            fp = ROOT / path
            if fp.exists() and "interactive_viewer" in fp.read_text(encoding="utf-8", errors="ignore"):
                viewer.add(path)
    return golden | viewer


def _workflow_job(workflow: str, name: str) -> str:
    match = re.search(
        rf"^  {re.escape(name)}:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"workflow job not found: {name}"
    return match.group(0)


def test_e_validation_profiles_are_exhaustive_and_honest():
    universe = _tracked_test_files()
    unrun = set(ci_pytest_lane.unrun_files())
    explicit = _explicit_lane_files()

    # No UNRUN entry may name a nonexistent / untracked file.
    missing = sorted(f for f in unrun if f not in universe)
    assert missing == [], f"UNRUN names files absent from the tracked suite: {missing}"

    # Quarantine entries are unique, owner-attributed, and non-expired.
    data = load_toml(TESTS / "UNRUN.toml")
    entries = data.get("entries", [])
    files = [entry.get("file") for entry in entries]
    assert len(files) == len(set(files)), f"duplicate UNRUN entries: {files}"
    for entry in entries:
        assert "reason" in entry and entry["reason"], f"UNRUN entry lacks a reason: {entry}"
        assert "owner" in entry and entry["owner"], f"UNRUN entry lacks an owner: {entry}"
        assert date.fromisoformat(entry["expires"]) >= date.today(), f"expired UNRUN entry: {entry}"

    # A file may not be BOTH quarantined and claimed by an explicit lane.
    both = sorted(unrun & explicit)
    assert both == [], f"files are both UNRUN and run by an explicit lane: {both}"

    # The full profile collects everything not UNRUN; the accounting remains
    # total even though routine pull requests use the focused profile.
    script_lane = set(ci_pytest_lane.full_lane_files())
    full_lane = universe - unrun
    unrun_names = {Path(f).name for f in unrun}
    assert {Path(f).name for f in script_lane} & unrun_names == set(), (
        "full profile selects quarantined files"
    )
    assert script_lane == full_lane, (
        "lane script and tracked full-profile accounting differ: "
        f"missing={sorted(full_lane - script_lane)}, extra={sorted(script_lane - full_lane)}"
    )

    fast_lane = set(ci_pytest_lane.fast_lane_files())
    expected_fast = {
        "tests/test_aether_acceptance_evidence.py",
        "tests/test_install_smoke.py",
        "tests/test_license.py",
        "tests/test_api_contracts.py",
        "tests/test_solar_spa.py",
        "tests/test_capability_negotiation.py",
        "tests/test_budget_enforce.py",
        "tests/test_memory_budget_policy.py",
        "tests/test_device_init_failure.py",
        "tests/test_allocation_gate.py",
        "tests/test_dead_render_structure_gate.py",
        "tests/test_pipeline_validation_gate.py",
        "tests/test_degradation_behavior.py",
        "tests/test_certificate_verifier.py",
        "tests/test_render_certificate.py",
        "tests/test_render_certificate_contract.py",
        "tests/test_astro_ephemeris.py",
        "tests/test_determinism_matrix.py",
        "tests/test_no_silent_degradation.py",
        "tests/test_substratia_evidence_report.py",
    }
    assert fast_lane == expected_fast, (
        "fast profile changed without updating the architectural-contract lock: "
        f"missing={sorted(expected_fast - fast_lane)}, extra={sorted(fast_lane - expected_fast)}"
    )
    assert fast_lane <= full_lane, (
        f"fast profile selects quarantined or unknown tests: {sorted(fast_lane - full_lane)}"
    )
    conftest = (ROOT / "conftest.py").read_text(encoding="utf-8")
    assert "pytest_ignore_collect" not in conftest, "root conftest silently bypasses test collection"
    assert explicit <= universe


def test_e_slow_lane_is_marker_selected_and_accounted():
    default_args = ci_pytest_lane.build_pytest_args("full", [])
    slow_args = ci_pytest_lane.build_pytest_args(
        "full", [ci_pytest_lane.SLOW_LANE_SELECTOR]
    )
    assert default_args[default_args.index("-m") + 1] == (
        "not slow and not interactive_viewer"
    )
    assert slow_args[slow_args.index("-m") + 1] == (
        "slow and not interactive_viewer"
    )
    assert ci_pytest_lane.SLOW_LANE_SELECTOR not in slow_args

    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    slow_job = ci_yml.split("  test-python-slow:", 1)[1].split(
        "\n  # ============================================================================\n  # TERMINUS", 1
    )[0]
    assert "python scripts/ci_pytest_lane.py --profile full --slow-lane" in slow_job
    pr_core = _workflow_job(ci_yml, "pr-core-success")
    acceptance = _workflow_job(ci_yml, "full-acceptance-summary")
    assert "test-python-slow" not in pr_core.split("\n    runs-on:", 1)[0]
    assert "test-python-slow" in acceptance.split("\n    runs-on:", 1)[0]


def test_e_anamnesis_physical_jobs_are_acceptance_scoped_honestly():
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    paths_job = ci_yml.split("  terrain-golden-paths:", 1)[1].split(
        "\n  # ============================================================================\n  # Rust Tests", 1
    )[0]
    assert "anamnesis: ${{ steps.filter.outputs.anamnesis }}" in paths_job
    anamnesis_paths = paths_job.split("            anamnesis:\n", 1)[1]
    for broad_path in ("'src/**'", "'python/**'"):
        assert broad_path not in anamnesis_paths
    for path in (
        "'src/core/anamnesis/**'",
        "'src/core/framegraph_impl/**'",
        "'src/core/ibl.rs'",
        "'src/core/ibl/**'",
        "'src/core/session.rs'",
        "'src/core/shader_registry.rs'",
        "'src/core/hdr.rs'",
        "'src/core/tonemap.rs'",
        "'src/core/resource_tracker.rs'",
        "'src/core/material.rs'",
        "'src/core/hdr_readback.rs'",
        "'src/core/provenance.rs'",
        "'src/formats/hdr.rs'",
        "'src/lighting/types.rs'",
        "'src/lighting/light_buffer/**'",
        "'src/offscreen/**'",
        "'src/path_tracing/**'",
        "'src/shader_sources.rs'",
        "'src/shadows/**'",
        "'src/py_functions/adjudication.rs'",
        "'src/py_functions/mod.rs'",
        "'src/render/material_set.rs'",
        "'src/render/material_set/**'",
        "'src/terrain/renderer/**'",
        "'src/terrain/render_params/**'",
        "'src/py_module/classes.rs'",
        "'src/py_module/functions/rendering.rs'",
        "'src/py_types/frame.rs'",
        "'src/lib.rs'",
        "'src/util/memory_budget.rs'",
        "'src/py_module/functions/anamnesis.rs'",
        "'python/forge3d/anamnesis.py'",
        "'python/forge3d/determinism.py'",
        "'python/forge3d/_native.py'",
        "'python/forge3d/_gpu.py'",
        "'src/shaders/adjudication_raster.wgsl'",
        "'src/shaders/ao_from_aovs.wgsl'",
        "'src/shaders/pt_*.wgsl'",
        "'src/shaders/terrain_*.wgsl'",
        "'src/shaders/heightfield_*.wgsl'",
        "'src/shaders/brdf/**'",
        "'src/shaders/includes/determinism.wgsl'",
        "'src/shaders/shadow_blur.wgsl'",
        "'scripts/check_anamnesis_portability.py'",
        "'scripts/terrain_ci_probe.py'",
        "'scripts/assert_junit_zero_skips.py'",
        "'tests/anamnesis_gpu_acceptance.py'",
        "'tests/goldens/determinism/**'",
        "'.github/workflows/ci.yml'",
        "'.github/workflows/build-wheel.yml'",
    ):
        assert path in anamnesis_paths

    required = (
        "github.event_name == 'schedule'",
        "inputs.scope == 'full'",
        "inputs.scope == 'anamnesis'",
    )
    for job_name in (
        "test-anamnesis-portability-seed",
        "test-anamnesis-portability",
        "test-anamnesis-production",
    ):
        job = ci_yml.split(f"  {job_name}:", 1)[1].split(
            "\n    runs-on:", 1
        )[0]
        for fragment in required:
            assert fragment in job
        for forbidden in (
            "github.event_name == 'push'",
            "github.event_name == 'pull_request'",
            "run-physical",
            "needs.terrain-golden-paths.outputs.anamnesis",
        ):
            assert forbidden not in job
    production = ci_yml.split("  test-anamnesis-production:", 1)[1].split(
        "\n  # ============================================================================\n  # Hosted determinism families", 1
    )[0]
    assert "test_real_gpu_600_frame_acceptance" in production
    aggregate = ci_yml.split("  full-acceptance-summary:", 1)[1]
    assert "anamnesis_physical_selected=" in aggregate
    for job_name in (
        "test-anamnesis-portability-seed",
        "test-anamnesis-portability",
        "test-anamnesis-production",
    ):
        assert (
            f"check_selected \"$anamnesis_physical_selected\" "
            f"'${{{{ needs.{job_name}.result }}}}'"
        ) in aggregate


# ---------------------------------------------------------------------------
# (f) visual-golden lane honesty
# ---------------------------------------------------------------------------
def test_f_backend_golden_variants_are_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "FORGE3D_TERRAIN_GOLDEN_VARIANT"
    monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("WGPU_BACKEND", raising=False)
    assert selected_golden_variant(env_name, implicit_metal=True) is None

    monkeypatch.setenv("WGPU_BACKEND", "metal")
    assert selected_golden_variant(env_name, implicit_metal=True) == "metal"
    assert selected_golden_path(
        Path("goldens"), "scene", env_name, implicit_metal=True
    ) == Path("goldens/scene.metal.png")

    monkeypatch.setenv("WGPU_BACKEND", "vulkan")
    assert selected_golden_variant(env_name, implicit_metal=True) is None
    assert selected_golden_path(
        Path("goldens"), "scene", env_name, implicit_metal=True
    ) == Path("goldens/scene.png")
    monkeypatch.setenv(env_name, "nvidia-vulkan")
    assert selected_golden_variant(env_name, implicit_metal=True) == "nvidia-vulkan"
    assert selected_golden_path(
        Path("goldens"), "scene", env_name, implicit_metal=True
    ) == Path("goldens/scene.nvidia-vulkan.png")
    assert_nvidia_vulkan_golden_adapter(
        env_name,
        {
            "status": "ok",
            "backend": "Vulkan",
            "device_type": "DiscreteGpu",
            "vendor": 0x10DE,
            "name": "NVIDIA test adapter",
            "software_fallback": False,
        },
    )
    with pytest.raises(AssertionError):
        assert_nvidia_vulkan_golden_adapter(
            env_name,
            {
                "status": "ok",
                "backend": "Vulkan",
                "device_type": "DiscreteGpu",
                "vendor": 0x1002,
                "name": "wrong adapter",
                "software_fallback": False,
            },
        )

    monkeypatch.setenv("WGPU_BACKEND", "metal")
    with pytest.raises(ValueError, match="requires WGPU_BACKEND"):
        selected_golden_variant(env_name, implicit_metal=True)

    monkeypatch.setenv("WGPU_BACKEND", "vulkan")
    monkeypatch.setenv(env_name, "generic-vulkan")
    with pytest.raises(ValueError, match="Unknown golden variant"):
        selected_golden_variant(env_name, implicit_metal=True)


def test_f_nvidia_visual_acceptance_is_physical_and_fail_closed():
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    fast_job = _workflow_job(ci_yml, "test-fast-contract")
    golden_job = _workflow_job(ci_yml, "test-golden-images-nvidia")
    metal_diagnostic = _workflow_job(ci_yml, "test-golden-images")
    pytest_step = golden_job.split("- name: Run visual golden tests", 1)[1].split("\n      - name:", 1)[0]
    sidera_step = golden_job.split("- name: Run SIDERA NVIDIA Vulkan night golden", 1)[1].split(
        "\n      - name:", 1
    )[0]
    probe_step = golden_job.split("- name: Require physical NVIDIA Vulkan terrain adapter", 1)[
        1
    ].split("\n      - name:", 1)[0]
    aggregate = _workflow_job(ci_yml, "full-acceptance-summary")

    assert "github.event_name == 'pull_request'" not in golden_job
    assert "inputs.scope == 'full'" in golden_job
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in golden_job
    assert "WGPU_BACKEND: vulkan" in golden_job
    assert "name: wheels-windows" in golden_job
    assert "name: wheels-macos" not in golden_job
    assert "--require-nvidia-vulkan" in probe_step
    assert "continue-on-error" not in probe_step
    assert "FORGE3D_ALLOW_SOFTWARE_GOLDENS" not in golden_job
    assert "FORGE3D_UPDATE_TERRAIN_GOLDENS" not in golden_job
    assert "FORGE3D_UPDATE_RECIPE_GOLDENS" not in golden_job
    assert "FORGE3D_UPDATE_TERRAIN_GOLDENS" not in pytest_step
    assert "FORGE3D_UPDATE_RECIPE_GOLDENS" not in pytest_step
    assert "FORGE3D_TERRAIN_GOLDEN_VARIANT: nvidia-vulkan" in golden_job
    assert "FORGE3D_RECIPE_GOLDEN_VARIANT: nvidia-vulkan" in golden_job
    assert "FORGE3D_SUBSTRATIA_GOLDEN_VARIANT: nvidia-vulkan" in golden_job
    assert "continue-on-error" not in pytest_step, "golden pytest mismatch is incorrectly non-fatal"
    assert "run_nvidia_visual_acceptance.py --suite visual" in pytest_step
    visual_runner = (ROOT / "scripts/run_nvidia_visual_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert (
        "test_recipe_goldens_render_and_match[mapscene_terrain_raster]"
        not in visual_runner
    )
    assert "test_nvidia_vulkan_recipe_pixel_golden_render_and_match" in visual_runner
    recipe_source = (ROOT / "tests/test_recipe_goldens.py").read_text(encoding="utf-8")
    certificate_test = recipe_source.split(
        "def test_recipe_goldens_render_and_match", 1
    )[1].split("def test_nvidia_vulkan_recipe_pixel_golden_render_and_match", 1)[0]
    nvidia_pixel_test = recipe_source.split(
        "def test_nvidia_vulkan_recipe_pixel_golden_render_and_match", 1
    )[1]
    assert "_render_recipe_golden_pixels" in certificate_test
    assert "_emit_or_verify_certificate(spec)" in certificate_test
    assert "_render_recipe_golden_pixels" in nvidia_pixel_test
    assert "_emit_or_verify_certificate" not in nvidia_pixel_test
    assert "FORGE3D_CERT_SIGNING_KEY" not in golden_job
    assert "FORGE3D_REQUIRE_PRODUCTION_SIGNING" not in golden_job
    assert "assert_junit_zero_skips.py" in pytest_step
    assert "tests/test_astro_night_golden.py" in sidera_step
    assert "assert_junit_zero_skips.py" in sidera_step
    assert "continue-on-error" not in sidera_step
    assert "sidera_lane:" in golden_job
    assert "FORGE3D_EXPECTED_ADAPTER_PROBE" in golden_job
    for path, evidence_name in (
        ("test_terrain_visual_goldens.py", "terrain-render-adapter.json"),
        ("test_terrain_tv10_goldens.py", "tv10-render-adapter.json"),
        ("test_recipe_goldens.py", "recipe-render-adapter.json"),
    ):
        source = (TESTS / path).read_text(encoding="utf-8")
        assert "assert_nvidia_vulkan_golden_adapter" in source
        assert evidence_name in source
        if path != "test_recipe_goldens.py":
            assert "selected_golden_path(" in source
    assert "visual-gpu-evidence" in golden_job and "retention-days: 90" in golden_job
    assert "Require production certificate signing key" not in golden_job
    certificate_refresh = (
        ROOT / ".github/workflows/certificate-refresh.yml"
    ).read_text(encoding="utf-8")
    assert "FORGE3D_CERT_SIGNING_KEY" in certificate_refresh
    assert "FORGE3D_REQUIRE_PRODUCTION_SIGNING" in certificate_refresh
    assert "github.ref == 'refs/heads/main'" in certificate_refresh
    assert "github.ref_protected" in certificate_refresh
    assert "FORGE3D_CERT_SIGNING_KEY" not in fast_job
    assert "FORGE3D_REQUIRE_PRODUCTION_SIGNING" not in fast_job
    assert "FORGE3D_RUN_TERRAIN_GOLDENS" not in fast_job
    assert "test_recipe_goldens.py" not in fast_job
    assert (
        "check_selected \"$full_selected\" "
        "'${{ needs.test-golden-images-nvidia.result }}' visual-goldens-nvidia"
        in aggregate
    )
    assert "needs.test-golden-images-nvidia.outputs.lane" in aggregate
    assert "needs.test-golden-images-nvidia.outputs.sidera_lane" in aggregate
    assert 'if [ "$sidera_lane" != "ran" ]' in aggregate
    assert "SIDERA physical NVIDIA Vulkan lane was selected" in aggregate
    assert "FORGE3D_RUN_METAL_DIAGNOSTIC" in metal_diagnostic
    assert "continue-on-error: true" in metal_diagnostic
    assert "test-golden-images," not in aggregate.split("\n    runs-on:", 1)[0]


def test_f_pr_core_is_lightweight_and_full_profiles_are_acceptance_only():
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    fast_job = _workflow_job(ci_yml, "test-fast-contract")
    pr_core = _workflow_job(ci_yml, "pr-core-success")
    acceptance = _workflow_job(ci_yml, "full-acceptance-summary")

    assert "--profile fast" in fast_job
    pr_needs = pr_core.split("\n    runs-on:", 1)[0]
    assert "test-fast-contract" in pr_needs
    for heavy in (
        "test-golden-images",
        "test-m06-full-geospatial-viewer",
        "test-anamnesis-production",
        "test-python-full-linux",
        "test-python-full-windows",
        "test-python-full-macos",
    ):
        assert heavy not in pr_needs, f"PR Core Success depends on heavyweight lane {heavy}"
    assert "FORGE3D_CERT_SIGNING_KEY" not in pr_core
    assert "full acceptance" in acceptance.lower()

    for name in (
        "test-python-full-linux",
        "test-python-full-windows",
        "test-python-full-macos",
    ):
        job = _workflow_job(ci_yml, name)
        assert "test_mode: full" in job, f"{name} does not select the exhaustive profile"
        assert "github.event_name == 'pull_request'" not in job
    reusable_python = (
        ROOT / ".github" / "workflows" / "test-python-wheel.yml"
    ).read_text(encoding="utf-8")
    assert "python scripts/ci_pytest_lane.py --profile full" in reusable_python
    slow_job = _workflow_job(ci_yml, "test-python-slow")
    assert "python scripts/ci_pytest_lane.py --profile full --slow-lane" in slow_job
    assert "github.event_name == 'pull_request'" not in slow_job


def test_substratia_physical_evidence_is_exact_head_and_cannot_be_bypassed():
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = _workflow_job(ci_yml, "test-substratia-gpu-nvidia")
    metal_diagnostic = _workflow_job(ci_yml, "test-substratia-gpu")
    core = _workflow_job(ci_yml, "pr-core-success")
    acceptance = _workflow_job(ci_yml, "full-acceptance-summary")
    probe = (ROOT / "scripts" / "terrain_ci_probe.py").read_text(encoding="utf-8")
    lane = (ROOT / "scripts" / "ci_pytest_lane.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / "run_nvidia_visual_acceptance.py").read_text(
        encoding="utf-8"
    )

    # Physical acceptance remains outside the stable hosted PR context.
    assert "test-substratia-gpu-nvidia" not in core.split("\n    runs-on:", 1)[0]
    assert "github.event_name == 'pull_request'" not in job
    assert "inputs.scope == 'full'" in job
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in job
    assert "WGPU_BACKEND: vulkan" in job
    assert "--require-nvidia-vulkan" in job
    assert "FORGE3D_ALLOW_SOFTWARE_GOLDENS" not in job
    assert "continue-on-error" not in job

    # The lane is bound to an explicit clean candidate and produces exact test,
    # image, adapter, and verifier evidence rather than trusting artifact presence.
    assert "FORGE3D_SUBSTRATIA_CANDIDATE_SHA" in job
    assert 'git status --porcelain --untracked-files=no' in job
    assert "run_nvidia_visual_acceptance.py --suite substratia" in job
    assert "assert_junit_zero_skips.py" in job
    for test_name in (
        "test_normal_family_changes_lighting_ssim",
        "test_all_families_page_within_budget",
        "test_missing_family_is_fatal",
        "test_partial_normal_residency_degrades_gracefully",
    ):
        assert test_name in launcher
    assert "scripts/substratia_evidence_report.py" in job
    assert '--candidate-sha "$env:FORGE3D_SUBSTRATIA_CANDIDATE_SHA"' in job
    assert "--render-adapter" in job
    assert "adapter-probe.json" in job
    assert "lane-ran.json" in job and "verification.json" in job
    assert "if-no-files-found: error" in job
    assert "retention-days: 90" in job

    # Full Acceptance consumes explicit RAN and PASS outputs; a successful upload
    # alone is not sufficient.
    assert "test-substratia-gpu-nvidia" in acceptance.split("\n    runs-on:", 1)[0]
    assert "needs.test-substratia-gpu-nvidia.outputs.lane" in acceptance
    assert "needs.test-substratia-gpu-nvidia.outputs.verifier" in acceptance
    assert "SUBSTRATIA physical lane did not record RAN" in acceptance
    assert "SUBSTRATIA evidence verifier did not record PASS" in acceptance

    # The SUBSTRATIA physical proof excludes virtual devices even though the
    # general CI-safe probe retains main's virtual-GPU support.
    for token in ("software", "virtual", "paravirtual", "virtio", "llvmpipe"):
        assert f'"{token}"' in probe
    assert "return 2" in probe and "return 3" in probe
    assert "tests/test_substratia_evidence_report.py" in lane
    assert "FORGE3D_RUN_METAL_DIAGNOSTIC" in metal_diagnostic
    assert "continue-on-error: true" in metal_diagnostic
    assert "test-substratia-gpu," not in acceptance.split("\n    runs-on:", 1)[0]
