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

import ast
import json
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pytest
import yaml

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
    "shader-contract-asserts",
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


def _workflow_step(job: str, name: str) -> str:
    return job.split(f"- name: {name}", 1)[1].split("\n      - name:", 1)[0]


def _assert_pwsh_pytest_and_verifier_are_both_authoritative(
    step: str, pytest_command: str
) -> None:
    lines = [line.strip() for line in step.splitlines()]
    pytest_index = lines.index(pytest_command)
    pytest_status_index = lines.index("$pytestCode = $LASTEXITCODE")
    verifier_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("python scripts/assert_junit_zero_skips.py")
    )
    verifier_status_index = lines.index("$verifyCode = $LASTEXITCODE")
    pytest_exit_index = lines.index("if ($pytestCode -ne 0) { exit $pytestCode }")
    verifier_exit_index = lines.index("exit $verifyCode")

    assert (
        pytest_index
        < pytest_status_index
        < verifier_index
        < verifier_status_index
        < pytest_exit_index
        < verifier_exit_index
    )


PHYSICAL_FAMILY_NODES = {
    "anamnesis_physical": {
        "tests/test_anamnesis_portability.py::test_portable_store_hits_and_capability_mismatch_misses",
        "tests/test_anamnesis_incremental.py::test_real_gpu_600_frame_acceptance",
        "tests/test_anamnesis_inertness.py::test_native_terrain_cache_restores_all_graph_passes",
        "tests/test_anamnesis_inertness.py::test_native_terrain_cache_rejects_moment_shadow_techniques",
        "tests/test_anamnesis_p1.py::test_public_gpu_graph_cache_restores_intermediate_texture",
    },
    "sidera_vulkan": {
        "tests/test_astro_night_golden.py::test_night_golden_matches_committed_vulkan_bytes",
        "tests/test_astro_night_golden.py::test_golden_refresh_does_not_rewrite_the_committed_file_when_disabled",
        "tests/test_determinism_hash.py::test_sidera_night_vulkan_reference_replays",
    },
    "cross_backend": {
        "tests/test_determinism_hash.py::test_device_probe_reports_initialized_render_adapter",
        "tests/test_shadow_tip.py::test_shadow_mask_is_identical_on_dx12_and_vulkan",
    },
    "f3dz_physical": {
        "tests/test_f3dz_codec.py::test_gpu_matches_cpu_for_every_corpus_page",
    },
    "nvidia_vulkan": {
        "tests/test_recipe_goldens.py::test_nvidia_vulkan_recipe_pixel_golden_render_and_match",
        "tests/test_terrain_runtime.py::test_nvidia_vulkan_terrain_constructor_child_smoke",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_normal_family_changes_lighting_ssim",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_all_families_page_within_budget",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_missing_family_is_fatal",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_missing_family_offline_preflight_leaves_no_active_session",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_partial_normal_residency_degrades_gracefully",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_unusable_family_source_is_fatal",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_partial_mask_residency_uses_neutral_fallback",
        "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_gpu_shader_feedback_preserves_family_coordinates",
    },
    "limes_physical": {
        "tests/test_vector_coverage.py::test_torture_plus_100k_road_segments_frame_time_within_budget",
    },
    "helios_physical": {
        "tests/test_shadow_tip.py::test_shadow_mask_golden",
    },
}

PHYSICAL_FAMILY_MODULES = {
    "tests/test_preset_visual_parity.py": "nvidia_vulkan",
    "tests/test_terrain_tv10_goldens.py": "nvidia_vulkan",
    "tests/test_terrain_visual_goldens.py": "nvidia_vulkan",
}


def _function_markers(nodeid: str) -> set[str]:
    path_text, *qualname = nodeid.split("::")
    function_name = qualname[-1]
    tree = ast.parse((ROOT / path_text).read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    assert len(functions) == 1, nodeid
    markers = set()
    for decorator in functions[0].decorator_list:
        call = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(call, ast.Attribute)
            and isinstance(call.value, ast.Attribute)
            and isinstance(call.value.value, ast.Name)
            and call.value.value.id == "pytest"
            and call.value.attr == "mark"
        ):
            markers.add(call.attr)
    return markers


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
    dedicated = "".join(
        f" and not {marker}" for marker in ci_pytest_lane.DEDICATED_LANE_MARKERS
    )
    assert default_args[default_args.index("-m") + 1] == (
        "not slow and not interactive_viewer" + dedicated
    )
    assert slow_args[slow_args.index("-m") + 1] == (
        "slow and not interactive_viewer" + dedicated
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


def test_e_recipe_goldens_are_routed_only_by_explicit_backend_lane(monkeypatch):
    monkeypatch.delenv("FORGE3D_RECIPE_GOLDEN_VARIANT", raising=False)
    generic_args = ci_pytest_lane.build_pytest_args("full", [])
    generic_marker = generic_args[generic_args.index("-m") + 1]
    assert "not recipe_golden" in generic_marker

    monkeypatch.setenv("WGPU_BACKEND", "metal")
    monkeypatch.setenv("FORGE3D_RECIPE_GOLDEN_VARIANT", "metal")
    physical_args = ci_pytest_lane.build_pytest_args("full", [])
    physical_marker = physical_args[physical_args.index("-m") + 1]
    assert "not recipe_golden" in physical_marker


def test_e_helios_pixel_golden_is_owned_by_the_existing_dedicated_lane():
    source = (ROOT / "tests" / "test_shadow_tip.py").read_text(encoding="utf-8")
    routed_nodes = (
        "test_shadow_mask_golden",
        "test_shadow_mask_is_identical_on_dx12_and_vulkan",
    )
    for name in routed_nodes:
        decorators = source.split(f"def {name}()", 1)[0].rsplit("\n\n", 1)[-1]
        assert "@pytest.mark.recipe_golden" in decorators
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    lane = workflow.split("test-helios-gpu:", 1)[1]
    assert "FORGE3D_RUN_TERRAIN_GOLDENS: '1'" in lane
    assert lane.count("tests/test_shadow_tip.py") == 1
    assert (
        "tests/test_determinism_hash.py::test_device_probe_reports_initialized_render_adapter"
        in lane
    )
    assert "assert_junit_zero_skips.py" in lane

    reusable = (
        ROOT / ".github" / "workflows" / "test-python-wheel.yml"
    ).read_text(encoding="utf-8")
    assert "FORGE3D_RECIPE_GOLDEN_VARIANT" not in reusable
    assert "WGPU_BACKEND" not in reusable

    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    nvidia_job = _workflow_job(ci_yml, "test-golden-images-nvidia")
    assert "WGPU_BACKEND: vulkan" in nvidia_job
    assert "FORGE3D_RECIPE_GOLDEN_VARIANT: nvidia-vulkan" in nvidia_job
    assert "python -m pip install -r tests/requirements.txt" in nvidia_job

    certificate_refresh = (
        ROOT / ".github" / "workflows" / "certificate-refresh.yml"
    ).read_text(encoding="utf-8")
    assert "WGPU_BACKEND: vulkan" in certificate_refresh
    assert "FORGE3D_RECIPE_GOLDEN_VARIANT: nvidia-vulkan" in certificate_refresh
    assert "python -m pip install -r tests/requirements.txt" in certificate_refresh

    recipe_source = (ROOT / "tests" / "test_recipe_goldens.py").read_text(
        encoding="utf-8"
    )
    assert "@pytest.mark.recipe_golden" in recipe_source
    render_helper = recipe_source.split("def _render_recipe_golden_pixels", 1)[1]
    assert render_helper.index("report = scene.render()") < render_helper.index(
        "active_adapter = _active_render_adapter()"
    )
    assert render_helper.index("active_adapter = _active_render_adapter()") < render_helper.index(
        "_assert_active_recipe_golden_adapter(active_adapter)"
    )
    nvidia_decorators = recipe_source.split(
        "def test_nvidia_vulkan_recipe_pixel_golden_render_and_match", 1
    )[0].rsplit("\n\n", 1)[-1]
    assert (
        '@pytest.mark.parametrize("spec", NVIDIA_VULKAN_RECIPE_GOLDENS'
        in nvidia_decorators
    )
    metal_decorators = recipe_source.split(
        "def test_metal_recipe_pixel_golden_render_and_match", 1
    )[0].rsplit("\n\n", 1)[-1]
    assert '@pytest.mark.parametrize("spec", RECIPE_GOLDENS' in metal_decorators


def test_e_tv6_example_is_owned_by_the_existing_m06_viewer_lane():
    source = (
        ROOT / "tests" / "test_terrain_tv6_heterogeneous_volumetrics.py"
    ).read_text(encoding="utf-8")
    target = source.split(
        "def test_tv6_example_renders_real_dem_and_reports_budget", 1
    )[0]
    assert target.rstrip().endswith("@pytest.mark.interactive_viewer")

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    lane = _workflow_job(workflow, "test-m06-full-geospatial-viewer")
    assert "RUN_M06_VIEWER_CI: '1'" in lane
    assert "FORGE3D_VIEWER_BINARY" in lane
    assert (
        "tests/test_terrain_tv6_heterogeneous_volumetrics.py::"
        "test_tv6_example_renders_real_dem_and_reports_budget"
    ) in lane
    assert "assert_junit_zero_skips.py" in lane


def test_e_local_asset_closure_examples_are_shipped_and_skip_gates_do_not_return():
    examples = (
        "_terrain_feature_demo.py",
        "terrain_tv4_material_variation_demo.py",
        "terrain_tv6_heterogeneous_volumetrics_demo.py",
        "terrain_tv10_subsurface_demo.py",
        "terrain_tv21_blending_demo.py",
        "terrain_tv24_reflection_probe_demo.py",
    )
    assert all((ROOT / "examples" / name).is_file() for name in examples)

    former_gates = {
        "test_california_cigar_smoke_hybrid.py": "California cache fixtures unavailable",
        "test_lighting_alignment.py": "sample_dem.tif and studio_small_08_4k.hdr required",
        "test_perspective_projection.py": "studio_small_08_4k.hdr not found",
        "test_provenance_offline_verify.py": "Provenance fixture missing",
        "test_shadow_techniques.py": "studio_small_08_4k.hdr not found",
        "test_terrain_render_color_space.py": "studio_small_08_4k.hdr missing",
        "test_terrain_tv6_heterogeneous_volumetrics.py": "interactive_viewer binary not found",
    }
    for filename, text in former_gates.items():
        source = (TESTS / filename).read_text(encoding="utf-8")
        assert text not in source, f"skip gate returned in {filename}: {text}"


def test_e_full_python_profiles_install_one_manifest_and_reject_skips():
    ci = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    reusable = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "test-python-wheel.yml").read_text(
            encoding="utf-8"
        )
    )
    profiles = (
        (
            reusable["jobs"]["test"]["steps"],
            "Install wheel and full-suite dependencies",
            "Run full default Python lane",
            "Verify full Python lane has zero skips",
            "Upload full Python lane JUnit",
        ),
        (
            ci["jobs"]["test-python-slow"]["steps"],
            "Install wheel and slow-lane dependencies",
            "Run accounted slow Python lane",
            "Verify slow Python lane has zero skips",
            "Upload slow Python lane JUnit",
        ),
    )
    full_steps = reusable["jobs"]["test"]["steps"]
    slow_steps = ci["jobs"]["test-python-slow"]["steps"]
    full_pip_installs = [
        line.strip()
        for step in full_steps
        if "compatibility" not in str(step.get("if", ""))
        for line in str(step.get("run", "")).splitlines()
        if "pip install" in line
    ]
    slow_pip_installs = [
        line.strip()
        for step in slow_steps
        for line in str(step.get("run", "")).splitlines()
        if "pip install" in line
    ]
    manifest_install = ["python -m pip install -r tests/requirements.txt"]
    assert full_pip_installs == manifest_install
    assert slow_pip_installs == manifest_install

    artifact_names = []
    for steps, install_name, lane_name, verifier_name, upload_name in profiles:
        indexes = {step.get("name"): index for index, step in enumerate(steps)}
        install = steps[indexes[install_name]]
        lane = steps[indexes[lane_name]]
        verifier = steps[indexes[verifier_name]]
        upload = steps[indexes[upload_name]]

        pip_installs = [
            line.strip()
            for line in install["run"].splitlines()
            if "pip install" in line
        ]
        assert pip_installs == ["python -m pip install -r tests/requirements.txt"]
        assert indexes[lane_name] < indexes[verifier_name] < indexes[upload_name]
        assert lane["run"].count("python scripts/ci_pytest_lane.py") == 1
        assert "python -m pytest" not in lane["run"]
        assert "always()" in verifier["if"] and "always()" in upload["if"]
        assert upload["uses"] == "actions/upload-artifact@v4"
        assert upload["with"]["if-no-files-found"] == "error"

        junit = re.search(r"--junitxml=([^\s]+)", lane["run"])
        assert junit is not None
        junit_path = junit.group(1)
        assert verifier["run"].split()[-1] == junit_path
        artifact_prefix = junit_path.removesuffix(".xml")
        assert upload["with"]["path"] == artifact_prefix + "*"
        assert f"--selection-ledger={artifact_prefix}-selection.json" in lane["run"]
        artifact_names.append(upload["with"]["name"])

    assert len(artifact_names) == len(set(artifact_names))
    assert "${{ runner.os }}" in artifact_names[0]
    assert "${{ matrix.python-version }}" in artifact_names[0]

    requirements = {
        re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower()
        for line in (TESTS / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required = {
        "geopandas",
        "ipywidgets",
        "maturin",
        "mypy",
        "numpy",
        "packaging",
        "pillow",
        "pyproj",
        "pytest",
        "pyyaml",
        "rasterio",
        "requests",
        "scipy",
        "shapely",
        "xarray",
    }
    assert required <= requirements, (
        f"full-profile dependencies missing: {sorted(required - requirements)}"
    )
    assert {"pytest-asyncio", "wgpu"}.isdisjoint(requirements)


def test_e_full_acceptance_requires_authoritative_apple_metal_lane():
    ci = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    job = ci["jobs"]["test-apple-metal-acceptance"]
    runner_label = "forge3d-pr170-apple-metal"
    assert job["runs-on"] == [runner_label]
    assert job["needs"] == ["build-wheel-macos", "prepare-lfs-fixtures"]
    assert "continue-on-error" not in job
    assert "vars." not in str(job["if"])
    assert "github.event_name == 'schedule'" not in job["if"]
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert "inputs.scope == 'full'" in job["if"]
    assert (
        "github.ref == 'refs/heads/codex/refactor-forge3d-20260812'" in job["if"]
    )
    label_owners = []
    workflows = ROOT / ".github" / "workflows"
    workflow_paths = sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml")))
    for workflow_path in workflow_paths:
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job_name, candidate in workflow.get("jobs", {}).items():
            if runner_label in yaml.safe_dump(candidate):
                label_owners.append((workflow_path.name, job_name))
    assert label_owners == [("ci.yml", "test-apple-metal-acceptance")]

    env = job["env"]
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["FORGE3D_NO_BOOTSTRAP"] == "1"
    assert env["FORGE3D_TEST_INSTALLED_WHEEL"] == "1"
    assert env["FORGE3D_APPLE_METAL_ACCEPTANCE"] == "1"
    assert env["WGPU_BACKEND"] == "metal"
    assert env["WGPU_BACKENDS"] == "metal"
    assert env["FORGE3D_RECIPE_GOLDEN_VARIANT"] == "metal"
    assert "FORGE3D_TESSELLA_REQUIRED_GPU" not in env
    assert "FORGE3D_TESSELLA_TIMING_REQUIRED" not in env
    assert env["FORGE3D_RUN_LIVE_TEXT_GPU"] == "1"

    steps = {step["name"]: step for step in job["steps"] if "name" in step}
    host = steps["Prove dedicated physical Apple Silicon runner"]
    assert host["shell"] == "bash"
    for contract in (
        "set -euo pipefail",
        'test "$RUNNER_NAME" = "forge3d-pr170-apple-metal"',
        'test "$RUNNER_OS" = "macOS"',
        'test "$RUNNER_ARCH" = "ARM64"',
        'test "$(uname -m)" = "arm64"',
        'test "$(sysctl -n hw.optional.arm64)" = "1"',
        'test "$(sysctl -n kern.hv_vmm_present)" = "0"',
        "machdep.cpu.brand_string",
        "Apple\\ *) ;;",
        '*) echo "::error::runner is not physical Apple Silicon"; exit 1 ;;',
        "host-identity.txt",
    ):
        assert contract in host["run"]
    assert job["steps"][0]["name"] == "Prove dedicated physical Apple Silicon runner"
    download = steps["Download shared LFS fixture artifact"]
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"] == {
        "name": "lfs-fixture-bundles",
        "path": "lfs-fixture-bundles",
    }
    restore = steps["Restore and verify Apple Metal TIFF fixtures"]["run"]
    assert restore.count("python-tiffs.zip") == 1
    assert "assets/tif/dem_rainier.tif" in restore
    assert "was not restored" in restore
    assert "is still an LFS pointer" in restore
    step_names = [step.get("name") for step in job["steps"]]
    assert step_names.index("Download shared LFS fixture artifact") < step_names.index(
        "Restore and verify Apple Metal TIFF fixtures"
    ) < step_names.index("Run authoritative Apple Metal acceptance matrix")
    tv21_source = (ROOT / "tests" / "test_terrain_tv21_demo.py").read_text(
        encoding="utf-8"
    )
    assert "@pytest.mark.apple_metal_physical" in tv21_source
    assert 'REPO_ROOT / "assets" / "tif" / "dem_rainier.tif"' in tv21_source

    install = steps["Install exact macOS wheel and acceptance dependencies"]["run"]
    assert "python scripts/install_compatible_wheel.py dist" in install
    assert "python -m pip install -r tests/requirements.txt" in install
    assert "maturin" not in install

    run = steps["Run authoritative Apple Metal acceptance matrix"]["run"]
    assert run.count("python scripts/run_apple_metal_acceptance.py") == 1
    assert "--junit" in run and "--evidence-dir" in run
    assert "!scripts/run_apple_metal_acceptance.py" in (ROOT / ".gitignore").read_text(
        encoding="utf-8"
    )
    verifier = steps["Verify Apple Metal acceptance has zero skips"]
    assert "always()" in verifier["if"]
    assert "scripts/assert_junit_zero_skips.py" in verifier["run"]
    upload = steps["Upload Apple Metal acceptance evidence"]
    assert "always()" in upload["if"]
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["if-no-files-found"] == "error"

    summary = ci["jobs"]["full-acceptance-summary"]
    assert "test-apple-metal-acceptance" in summary["needs"]
    summary_run = summary["steps"][0]["run"]
    assert (
        "apple_metal_selected=\"${{ github.event_name == 'workflow_dispatch' && "
        "inputs.scope == 'full' && github.ref == "
        "'refs/heads/codex/refactor-forge3d-20260812' }}\""
    ) in summary_run
    assert (
        "check_selected \"$apple_metal_selected\" "
        "'${{ needs.test-apple-metal-acceptance.result }}' apple-metal"
    ) in summary_run
    diagnostic = (
        ROOT / ".github" / "workflows" / "determinism-matrix.yml"
    ).read_text(encoding="utf-8").split("  metal-diagnostic:", 1)[1].split(
        "\n  wasm-policy:", 1
    )[0]
    assert "runs-on: ${{ matrix.os }}" in diagnostic
    assert "os: macos-14" in diagnostic
    assert "continue-on-error: true" in diagnostic
    assert "FORGE3D_RUN_METAL_DIAGNOSTIC" in diagnostic
    policy = (ROOT / ".claude" / "rules" / "build-and-ci.md").read_text(
        encoding="utf-8"
    )
    assert "test-apple-metal-acceptance" in policy
    assert "required only by manual `scope=full`" in policy
    assert "on `codex/refactor-forge3d-20260812`" in policy
    assert "Scheduled full acceptance must leave that Apple job skipped" in policy
    assert "prerequisite for scheduled" not in policy


def test_e_apple_metal_selection_is_one_checked_fail_closed_manifest():
    from scripts import run_apple_metal_acceptance as metal

    phases = metal.load_manifest()
    assert [phase.name for phase in phases] == [
        "physical-family",
        "tv20-normal-fresh-1",
        "tv20-normal-fresh-2",
        "contract-matrix",
    ]
    normal_node = (
        "tests/test_tv20_virtual_texturing.py::TestTerrainMaterialVirtualTexturing::"
        "test_vt_normal_family_changes_normal_aov_and_reports_dual_residency"
    )
    physical = phases[0]
    assert physical.nodes == ("tests",)
    assert physical.marker is not None
    assert "apple_metal_physical" in physical.marker
    assert "not apple_metal_contract" in physical.marker
    assert phases[1].nodes == (normal_node,)
    assert phases[2].nodes == (normal_node,)

    matrix = phases[3].nodes
    required = {
        "tests/test_tv20_virtual_texturing.py::TestTerrainMaterialVirtualTexturing::test_vt_enabled_changes_albedo_and_reports_residency",
        "tests/test_aov.py::TestAovRendering::test_aov_numpy_outputs_are_real_and_normalized",
        "tests/test_aov.py::TestAovRendering::test_aov_outputs_match_beauty_size_after_scaling_and_msaa",
        "tests/test_flythrough_popping.py::test_depth_aov_matches_known_flat_plane_distance",
        "tests/test_msdf_fidelity.py::test_live_gpu_shader_readback_matches_independent_quad_oracle",
        "tests/test_msdf_fidelity.py::test_live_gpu_native_text_is_exact_across_two_scenes",
        "tests/test_astro_night_golden.py::test_night_golden_is_cross_process_repeatable_on_pinned_backend",
        "tests/test_cam_phi_wiring.py::test_cam_phi_changes_output",
        "tests/test_cam_phi_wiring.py::test_cam_phi_four_quadrants",
        "tests/test_determinism_hash.py::test_intra_backend_bit_identity",
        "tests/test_determinism_hash.py::test_matches_committed_golden",
        "tests/test_visibility_buffer.py::test_visibility_resolve_pays_once_and_picking_is_stable_for_10000_pixels",
        "tests/test_flythrough_popping.py::test_visibility_shading_is_identical_and_hole_free_at_flythrough_settings",
        "tests/test_recipe_goldens.py::test_metal_recipe_pixel_golden_render_and_match",
    }
    assert set(matrix) == required
    ordered = (
        "tests/test_cam_phi_wiring.py::test_cam_phi_changes_output",
        "tests/test_cam_phi_wiring.py::test_cam_phi_four_quadrants",
        "tests/test_determinism_hash.py::test_intra_backend_bit_identity",
    )
    start = matrix.index(ordered[0])
    assert matrix[start : start + len(ordered)] == ordered
    recipe_ids = metal.expected_recipe_ids()
    assert len(recipe_ids) == 22
    assert len(set(recipe_ids)) == len(recipe_ids)

    valid_adapter = {
        "backend": "Metal",
        "device_type": "IntegratedGpu",
        "software_fallback": False,
        "name": "Apple M4",
        "vendor": 0x106B,
        "device": 0x1234,
        "raw_vendor": 0,
        "raw_device": 0,
    }
    metal._require_physical_apple_metal(valid_adapter, "test")
    for invalid in (
        {**valid_adapter, "backend": "Vulkan"},
        {**valid_adapter, "device_type": "VirtualGpu"},
        {**valid_adapter, "software_fallback": True},
        {**valid_adapter, "name": "Apple Paravirtual GPU"},
        {**valid_adapter, "name": "NVIDIA RTX"},
        {key: value for key, value in valid_adapter.items() if key != "vendor"},
        {key: value for key, value in valid_adapter.items() if key != "device"},
        {**valid_adapter, "vendor": 0},
        {**valid_adapter, "vendor": 0x10DE},
        {**valid_adapter, "device": 0},
        {key: value for key, value in valid_adapter.items() if key != "raw_vendor"},
        {key: value for key, value in valid_adapter.items() if key != "raw_device"},
    ):
        with pytest.raises(RuntimeError):
            metal._require_physical_apple_metal(invalid, "test")

    manifest = metal._read_manifest()
    assert manifest["recipe_cases"] == 22
    assert manifest["expected_tests"] == 266
    runner_source = (ROOT / "scripts" / "run_apple_metal_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ["FORGE3D_APPLE_METAL_ACCEPTANCE"] = "1"' in runner_source
    assert "--runxfail" not in runner_source
    assert "--maxfail" not in runner_source
    assert "retry" not in runner_source.casefold()
    assert "_validate_phase_adapter_records(before, phases, evidence_dir)" in runner_source
    assert "_phase_result_junits(phase, phase_junit, code, evidence_dir)" in runner_source
    for filename in ("test_msdf_fidelity.py", "test_astro_night_golden.py"):
        required_source = (TESTS / filename).read_text(encoding="utf-8")
        assert "FORGE3D_APPLE_METAL_ACCEPTANCE" in required_source
        assert "raise RuntimeError" in required_source


def test_e_apple_metal_skip_audit_runs_after_marker_deselection():
    from scripts import run_apple_metal_acceptance as metal

    hook = metal.pytest_collection_modifyitems.pytest_impl
    assert hook["trylast"] is True


def _import_sidera_night_for_collection(monkeypatch, probe, *, apple=False):
    import runpy
    import types

    session_calls = []

    def session(**kwargs):
        session_calls.append(kwargs)
        raise AssertionError("Session called during test-module import")

    forge3d = types.ModuleType("forge3d")
    forge3d.Session = session
    forge3d.device_probe = probe
    native = types.ModuleType("forge3d._forge3d")
    native.engine_info = lambda: {}
    diagnostics = types.ModuleType("forge3d.diagnostics")
    diagnostics.render_certificate = lambda **kwargs: {}
    forge3d._forge3d = native
    monkeypatch.setitem(sys.modules, "forge3d", forge3d)
    monkeypatch.setitem(sys.modules, "forge3d._forge3d", native)
    monkeypatch.setitem(sys.modules, "forge3d.diagnostics", diagnostics)
    monkeypatch.setenv("FORGE3D_DETERMINISM_TEST_BACKEND", "vulkan")
    monkeypatch.delenv("FORGE3D_EXPECTED_ADAPTER_PROBE", raising=False)
    if apple:
        monkeypatch.setenv("FORGE3D_APPLE_METAL_ACCEPTANCE", "1")
    else:
        monkeypatch.delenv("FORGE3D_APPLE_METAL_ACCEPTANCE", raising=False)
    return runpy.run_path(TESTS / "test_astro_night_golden.py"), session_calls


def test_e_sidera_generic_no_adapter_imports_without_session(monkeypatch):
    probe_calls = []

    def no_adapter(backend):
        probe_calls.append(backend)
        return {"status": "no_adapter"}

    module, session_calls = _import_sidera_night_for_collection(monkeypatch, no_adapter)

    assert session_calls == []
    assert probe_calls == ["vulkan", "vulkan"]
    tests = {
        name: value
        for name, value in module.items()
        if name.startswith("test_") and callable(value)
    }
    assert len(tests) == 5
    for name, test in tests.items():
        assert any(
            marker.name == "apple_metal_physical"
            for marker in getattr(test, "pytestmark", ())
        ), name


def test_e_sidera_strict_apple_no_adapter_fails_closed(monkeypatch):
    with pytest.raises(RuntimeError, match="required Apple Metal SIDERA adapter"):
        _import_sidera_night_for_collection(
            monkeypatch, lambda backend: {"status": "no_adapter"}, apple=True
        )


def test_e_sidera_unexpected_probe_error_propagates(monkeypatch):
    class ProbeError(Exception):
        pass

    def broken_probe(backend):
        raise ProbeError(backend)

    with pytest.raises(ProbeError, match="vulkan"):
        _import_sidera_night_for_collection(monkeypatch, broken_probe)


def test_e_apple_metal_skip_audit_rejects_selected_decorators(monkeypatch):
    from scripts import run_apple_metal_acceptance as metal

    class Item:
        def __init__(self, nodeid, markers):
            self.nodeid = nodeid
            self._markers = markers

        def iter_markers(self, name=None):
            return (
                marker
                for marker in self._markers
                if name is None or marker.name == name
            )

        def get_closest_marker(self, name):
            return next(self.iter_markers(name), None)

    items = [
        Item("tests/test_selected.py::test_skip", (pytest.mark.skip.mark,)),
        Item(
            "tests/test_selected.py::test_skipif",
            (pytest.mark.skipif(True, reason="selected").mark,),
        ),
        Item("tests/test_selected.py::test_xfail", (pytest.mark.xfail.mark,)),
    ]
    monkeypatch.setenv("FORGE3D_APPLE_METAL_ACCEPTANCE", "1")

    with pytest.raises(pytest.UsageError) as error:
        metal.pytest_collection_modifyitems(items)

    message = str(error.value)
    assert "test_skip: skip" in message
    assert "test_skipif: active skipif" in message
    assert "test_xfail: xfail" in message


def test_e_apple_metal_collection_audits_only_selected_items(tmp_path):
    selection = tmp_path / "test_selection.py"
    selection.write_text(
        """\
import pytest

@pytest.mark.apple_metal_physical
def test_selected_clean():
    pass

@pytest.mark.nvidia_vulkan
@pytest.mark.skipif(True, reason="off lane")
def test_deselected_skipif():
    pass

@pytest.mark.apple_metal_contract
@pytest.mark.skip
def test_selected_skip():
    pass

@pytest.mark.apple_metal_contract
@pytest.mark.skipif(True, reason="selected")
def test_selected_skipif():
    pass

@pytest.mark.apple_metal_contract
@pytest.mark.xfail
def test_selected_xfail():
    pass
""",
        encoding="utf-8",
    )
    child_env = {**os.environ, "FORGE3D_APPLE_METAL_ACCEPTANCE": "1"}

    def collect(marker):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "scripts.run_apple_metal_acceptance",
                str(selection),
                "-m",
                marker,
                "--collect-only",
                "-q",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=child_env,
        )
        return result, result.stdout + result.stderr

    collected, output = collect("apple_metal_physical")
    assert collected.returncode == 0, output
    assert "test_selected_clean" in collected.stdout
    assert "test_deselected_skipif" not in collected.stdout

    rejected, output = collect("apple_metal_contract")
    assert rejected.returncode == int(pytest.ExitCode.USAGE_ERROR), output
    assert "test_selected_skip: skip" in output
    assert "test_selected_skipif: active skipif" in output
    assert "test_selected_xfail: xfail" in output


def test_e_apple_metal_merged_junit_preserves_failures(tmp_path):
    from scripts import run_apple_metal_acceptance as metal
    from scripts.assert_junit_zero_skips import JUnitValidationError, verify_junit

    clean = tmp_path / "clean.xml"
    clean.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0">'
        '<testcase name="clean"/></testsuite>',
        encoding="utf-8",
    )
    failed = tmp_path / "failed.xml"
    metal._failure_junit(failed, "required adapter missing")
    merged = tmp_path / "merged.xml"
    metal._merge_junit([clean, failed], merged)

    with pytest.raises(JUnitValidationError, match="zero-skip"):
        verify_junit(merged)
    assert metal._junit_test_count([merged]) == 2


def test_e_apple_metal_phase_process_records_rendered_adapter(
    monkeypatch, tmp_path
):
    from scripts import run_apple_metal_acceptance as metal

    phase = metal.Phase("rendered-phase", ("tests/test_pixels.py::test_render",))
    adapter = {
        "backend": "metal",
        "adapter_name": "Apple M4",
        "device_name": "Apple M4",
        "device_type": "integratedgpu",
        "software_fallback": False,
        "vendor": 0x106B,
        "device": 0x1234,
        "raw_vendor": 0,
        "raw_device": 0,
    }
    events = []

    def fake_pytest_main(args):
        events.append(("render", tuple(args)))
        return 0

    def fake_engine_info():
        events.append(("engine_info", None))
        return adapter

    monkeypatch.setattr(metal, "_pytest_main", fake_pytest_main)
    monkeypatch.setattr(metal, "_initialized_engine_info", fake_engine_info)
    record = tmp_path / "phase-adapter.json"

    assert metal._run_phase_in_process(phase, tmp_path / "phase.xml", record) == 0
    assert [event[0] for event in events] == ["render", "engine_info"]
    assert json.loads(record.read_text(encoding="utf-8")) == {
        "phase": phase.name,
        "active_adapter": adapter,
    }


def test_e_apple_metal_preflight_probes_the_initialized_context(
    monkeypatch, tmp_path
):
    import forge3d as f3d
    from scripts import run_apple_metal_acceptance as metal

    active = {
        "backend": "metal",
        "adapter_name": "Apple M4",
        "device_name": "Apple M4",
        "device_type": "integratedgpu",
        "software_fallback": False,
        "vendor": 0x106B,
        "device": 0x1234,
        "raw_vendor": 0,
        "raw_device": 0,
    }
    probe = {
        **active,
        "name": active["adapter_name"],
        "backend": "Metal",
        "device_type": "IntegratedGpu",
    }
    events = []

    def engine_info():
        events.append("engine_info")
        return active

    def device_probe(backend):
        events.append(f"device_probe:{backend}")
        return probe

    monkeypatch.setattr(metal, "_initialized_engine_info", engine_info)
    monkeypatch.setattr(f3d, "device_probe", device_probe)
    record = tmp_path / "adapter-before.json"

    assert metal._active_adapter_record(record) == {
        "requested_backend": "metal",
        "probe": probe,
        "active_adapter": active,
    }
    assert events == ["engine_info", "device_probe:metal"]
    assert json.loads(record.read_text(encoding="utf-8"))["probe"]["device"] == active[
        "device"
    ]


def test_e_apple_metal_requires_every_render_phase_adapter_and_exact_identity(
    tmp_path,
):
    from scripts import run_apple_metal_acceptance as metal

    probe = {
        "backend": "Metal",
        "name": "Apple M4",
        "device_type": "IntegratedGpu",
        "software_fallback": False,
        "vendor": 0x106B,
        "device": 0x1234,
        "raw_vendor": 0,
        "raw_device": 0,
    }
    active = {
        "backend": "metal",
        "adapter_name": "Apple M4",
        "device_name": "Apple M4",
        "device_type": "integratedgpu",
        "software_fallback": False,
        "vendor": 0x106B,
        "device": 0x1234,
        "raw_vendor": 0,
        "raw_device": 0,
    }
    before = {"requested_backend": "metal", "probe": probe, "active_adapter": active}
    phases = (
        metal.Phase("first", ("tests/test_pixels.py::test_first",)),
        metal.Phase("second", ("tests/test_pixels.py::test_second",)),
        metal.Phase("third", ("tests/test_pixels.py::test_third",)),
    )
    for phase in phases:
        (tmp_path / f"{phase.name}-adapter.json").write_text(
            json.dumps({"phase": phase.name, "active_adapter": active}),
            encoding="utf-8",
        )

    metal._validate_phase_adapter_records(before, phases, tmp_path)
    mismatched = {**active, "adapter_name": "Apple M3", "device_name": "Apple M3"}
    (tmp_path / "second-adapter.json").write_text(
        json.dumps({"phase": "second", "active_adapter": mismatched}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="second.*identity"):
        metal._validate_phase_adapter_records(before, phases, tmp_path)
    mismatched = {**active, "device": 0x5678}
    (tmp_path / "second-adapter.json").write_text(
        json.dumps({"phase": "second", "active_adapter": mismatched}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="second.*identity"):
        metal._validate_phase_adapter_records(before, phases, tmp_path)
    mismatched = {**active, "raw_device": 1}
    (tmp_path / "second-adapter.json").write_text(
        json.dumps({"phase": "second", "active_adapter": mismatched}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="second.*identity"):
        metal._validate_phase_adapter_records(before, phases, tmp_path)
    (tmp_path / "second-adapter.json").unlink()
    with pytest.raises(RuntimeError, match="second.*missing"):
        metal._validate_phase_adapter_records(before, phases, tmp_path)


def test_e_apple_metal_nonzero_phase_exit_adds_authoritative_junit_error(tmp_path):
    from scripts import run_apple_metal_acceptance as metal
    from scripts.assert_junit_zero_skips import JUnitValidationError, verify_junit

    phase = metal.Phase("rendered-phase", ("tests/test_pixels.py::test_render",))
    clean = tmp_path / "clean.xml"
    clean.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0" time="2.5">'
        '<testcase name="clean" time="2.5"/></testsuite>',
        encoding="utf-8",
    )
    other = tmp_path / "other.xml"
    other.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0" time="1.25">'
        '<testcase name="other" time="1.25"/></testsuite>',
        encoding="utf-8",
    )
    inputs = [other, *metal._phase_result_junits(phase, clean, 1, tmp_path)]
    merged = tmp_path / "merged.xml"
    metal._merge_junit(inputs, merged)
    root = ET.parse(merged).getroot()

    assert root.attrib == {
        "name": "Apple Metal acceptance",
        "tests": "3",
        "failures": "0",
        "errors": "1",
        "skipped": "0",
        "time": "3.75",
    }
    assert "rendered-phase exited 1" in ET.tostring(root, encoding="unicode")
    with pytest.raises(JUnitValidationError, match="zero-skip"):
        verify_junit(merged)


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
    assert "-m anamnesis_physical" in production
    for path in (
        "tests/test_anamnesis_incremental.py",
        "tests/test_anamnesis_inertness.py",
        "tests/test_anamnesis_p1.py",
        "tests/test_anamnesis_portability.py",
    ):
        assert production.count(path) == 1
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
    )[1].split("def test_metal_recipe_pixel_golden_render_and_match", 1)[0]
    metal_pixel_test = recipe_source.split(
        "def test_metal_recipe_pixel_golden_render_and_match", 1
    )[1].split("def test_nvidia_vulkan_recipe_pixel_golden_render_and_match", 1)[0]
    nvidia_pixel_test = recipe_source.split(
        "def test_nvidia_vulkan_recipe_pixel_golden_render_and_match", 1
    )[1]
    assert "_render_recipe_golden_pixels" in certificate_test
    assert "_emit_or_verify_certificate(spec)" in certificate_test
    assert "_render_recipe_golden_pixels" in metal_pixel_test
    assert "_emit_or_verify_certificate" not in metal_pixel_test
    assert "_render_recipe_golden_pixels" in nvidia_pixel_test
    assert "_emit_or_verify_certificate" not in nvidia_pixel_test
    assert "FORGE3D_CERT_SIGNING_KEY" not in golden_job
    assert "FORGE3D_REQUIRE_PRODUCTION_SIGNING" not in golden_job
    assert "assert_junit_zero_skips.py" in pytest_step
    assert "run_nvidia_visual_acceptance.py --suite sidera" in sidera_step
    for nodeid in PHYSICAL_FAMILY_NODES["sidera_vulkan"]:
        assert visual_runner.count(f'"{nodeid}"') == 1
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
        assert evidence_name in source
        if path == "test_recipe_goldens.py":
            assert "_active_render_adapter" in source
            assert "engine_info" in source
        else:
            assert "assert_nvidia_vulkan_golden_adapter" in source
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


def test_physical_records_have_semantic_family_markers():
    for marker, nodes in PHYSICAL_FAMILY_NODES.items():
        for nodeid in nodes:
            assert marker in _function_markers(nodeid), nodeid

    for path, marker in PHYSICAL_FAMILY_MODULES.items():
        source = (ROOT / path).read_text(encoding="utf-8")
        assert f"pytestmark = pytest.mark.{marker}" in source
        assert "allow_module_level=True" not in source


def test_generic_profiles_semantically_deselect_physical_families():
    registered = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    excluded = {*PHYSICAL_FAMILY_NODES, *PHYSICAL_FAMILY_MODULES.values()}
    for marker in excluded:
        assert f"    {marker}:" in registered

    for slow in (False, True):
        args = ci_pytest_lane.build_pytest_args("full", [], slow=slow)
        expression = args[args.index("-m") + 1]
        for marker in excluded:
            assert f"not {marker}" in expression


def test_generic_full_selection_ledger_records_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Item:
        def __init__(self, nodeid: str, markers: tuple[str, ...]):
            self.nodeid = nodeid
            self._markers = tuple(pytest.mark.__getattr__(name).mark for name in markers)

        def iter_markers(self, name=None):
            return (
                marker
                for marker in self._markers
                if name is None or marker.name == name
            )

        def get_closest_marker(self, name):
            return next(self.iter_markers(name), None)

    ledger = tmp_path / "selection.json"
    monkeypatch.setenv(ci_pytest_lane.ZERO_SKIP_ENV, "1")
    monkeypatch.setenv(ci_pytest_lane.LEDGER_ENV, str(ledger))
    ci_pytest_lane._DESELECTED.clear()
    try:
        ci_pytest_lane.pytest_deselected(
            [Item("tests/test_physical.py::test_gpu", ("gpu_lane",))]
        )
        ci_pytest_lane.pytest_collection_modifyitems(
            [Item("tests/test_portable.py::test_cpu", ())]
        )
        payload = json.loads(ledger.read_text(encoding="utf-8"))
    finally:
        ci_pytest_lane._DESELECTED.clear()

    assert payload == {
        "schema": "forge3d.pytest-selection.v1",
        "selected": [
            {"nodeid": "tests/test_portable.py::test_cpu", "markers": []}
        ],
        "deselected": [
            {
                "nodeid": "tests/test_physical.py::test_gpu",
                "markers": ["gpu_lane"],
            }
        ],
    }


def test_nvidia_visual_and_sidera_families_have_complete_zero_skip_evidence():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_nvidia_visual_acceptance.py").read_text(
        encoding="utf-8"
    )
    job = _workflow_job(workflow, "test-golden-images-nvidia")
    summary = _workflow_job(workflow, "full-acceptance-summary")
    visual_step = _workflow_step(job, "Run visual golden tests")
    sidera_step = _workflow_step(job, "Run SIDERA NVIDIA Vulkan night golden")

    for path in PHYSICAL_FAMILY_MODULES:
        assert runner.count(f'"{path}"') == 1
    for nodeid in PHYSICAL_FAMILY_NODES["nvidia_vulkan"]:
        assert runner.count(f'"{nodeid}"') == 1
    for nodeid in PHYSICAL_FAMILY_NODES["sidera_vulkan"]:
        assert runner.count(f'"{nodeid}"') == 1
    assert "--suite visual" in job
    assert "--suite sidera" in job
    _assert_pwsh_pytest_and_verifier_are_both_authoritative(
        visual_step,
        'python scripts/run_nvidia_visual_acceptance.py --suite visual --junit "$junit"',
    )
    _assert_pwsh_pytest_and_verifier_are_both_authoritative(
        sidera_step,
        'python scripts/run_nvidia_visual_acceptance.py --suite sidera --junit "$junit"',
    )
    assert "if: always()" in job and "uses: actions/upload-artifact@v4" in job
    assert "test-golden-images-nvidia" in summary


def test_f3dz_physical_family_has_one_complete_zero_skip_junit():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = _workflow_job(workflow, "test-f3dz-gpu")
    summary = _workflow_job(workflow, "full-acceptance-summary")
    step = _workflow_step(job, "Run all F3DZ corpus and CPU-GPU identity gates")

    assert job.count("python -m pytest tests/test_f3dz_codec.py") == 1
    _assert_pwsh_pytest_and_verifier_are_both_authoritative(
        step,
        'python -m pytest tests/test_f3dz_codec.py -v --tb=short --junitxml="$env:FORGE3D_F3DZ_ARTIFACT_DIR/junit.xml" *>&1 | Tee-Object "$env:FORGE3D_F3DZ_ARTIFACT_DIR/pytest.log"',
    )
    assert "if: always()" in job and "uses: actions/upload-artifact@v4" in job
    assert "test-f3dz-gpu" in summary


def test_anamnesis_family_has_one_complete_zero_skip_junit():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = _workflow_job(workflow, "test-anamnesis-production")
    summary = _workflow_job(workflow, "full-acceptance-summary")
    hosted_workflow = (
        ROOT / ".github" / "workflows" / "determinism-matrix.yml"
    ).read_text(encoding="utf-8")
    hosted = _workflow_job(hosted_workflow, "anamnesis-seed")
    physical_step = _workflow_step(
        job, "Run the complete ANAMNESIS physical family with zero skips"
    )

    for nodeid in PHYSICAL_FAMILY_NODES["anamnesis_physical"]:
        path = nodeid.split("::", 1)[0]
        assert job.count(path) == 1
    assert "-m anamnesis_physical" in job
    assert "FORGE3D_DETERMINISTIC: '1'" in job
    assert "FORGE3D_RUN_GPU_ANAMNESIS: '1'" in job
    assert job.count("--junitxml=") == 1
    assert job.count("assert_junit_zero_skips.py") == 1
    _assert_pwsh_pytest_and_verifier_are_both_authoritative(
        physical_step,
        "python -m pytest tests/test_anamnesis_incremental.py "
        "tests/test_anamnesis_inertness.py tests/test_anamnesis_p1.py "
        "tests/test_anamnesis_portability.py -m anamnesis_physical -v "
        '--tb=short --basetemp="$env:FORGE3D_ANAMNESIS_BASE_TEMP" '
        '--junitxml="$junit"',
    )
    assert "if: always()" in job and "uses: actions/upload-artifact@v4" in job
    assert "test-anamnesis-production" in summary
    assert '-m "not anamnesis_physical"' in hosted
    assert (
        "tests/test_anamnesis_incremental.py::test_real_gpu_600_frame_acceptance"
        not in hosted
    )


def test_cross_backend_family_is_selected_once_by_required_helios():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = _workflow_job(workflow, "test-helios-gpu")
    summary = _workflow_job(workflow, "full-acceptance-summary")
    device_probe = (
        "tests/test_determinism_hash.py::"
        "test_device_probe_reports_initialized_render_adapter"
    )

    assert job.count(device_probe) == 1
    assert job.count("tests/test_shadow_tip.py") == 1
    assert job.count("--junitxml=") == 1
    assert job.count("assert_junit_zero_skips.py") == 1
    assert "if: always()" in job and "uses: actions/upload-artifact@v4" in job
    assert "test-helios-gpu" in summary


def test_limes_and_approved_tv6_share_required_m06_evidence_once():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    job = _workflow_job(workflow, "test-m06-full-geospatial-viewer")
    summary = _workflow_job(workflow, "full-acceptance-summary")

    assert job.count("'tests/test_vector_coverage.py'") == 1
    assert job.count(
        "'tests/test_terrain_tv6_heterogeneous_volumetrics.py::"
        "test_tv6_example_renders_real_dem_and_reports_budget'"
    ) == 1
    assert job.count("--junitxml=") == 1
    assert job.count("assert_junit_zero_skips.py") == 1
    assert "if: always()" in job and "uses: actions/upload-artifact@v4" in job
    assert "test-m06-full-geospatial-viewer" in summary


def test_all_interactive_viewer_records_are_owned_once_by_required_m06(
    tmp_path: Path,
):
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    job = _workflow_job(workflow, "test-m06-full-geospatial-viewer")
    step = _workflow_step(job, "Run M-06 source and live acceptance")
    summary = _workflow_job(workflow, "full-acceptance-summary")
    informational = _workflow_job(workflow, "test-interactive-viewer-macos")

    tests_block = step.split("$tests = @(", 1)[1].split("\n          )", 1)[0]
    m06_nodes = re.findall(r"'([^']+)'", tests_block)
    assert len(m06_nodes) == len(set(m06_nodes))

    ledger = tmp_path / "generic-full-selection.json"
    collected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ci_pytest_lane.py"),
            "--profile",
            "full",
            f"--selection-ledger={ledger}",
            "--collect-only",
            "-qq",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    interactive = [
        record["nodeid"]
        for record in payload["deselected"]
        if "interactive_viewer" in record["markers"]
    ]
    assert len(interactive) == 48
    assert len(interactive) == len(set(interactive))

    def ownership_count(nodeid: str) -> int:
        return sum(
            nodeid == target
            or nodeid.startswith(target + "[")
            or ("::" not in target and nodeid.startswith(target + "::"))
            for target in m06_nodes
        )

    assert {nodeid: ownership_count(nodeid) for nodeid in interactive} == {
        nodeid: 1 for nodeid in interactive
    }
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in job
    assert "WGPU_BACKEND: vulkan" in job
    assert "RUN_INTERACTIVE_VIEWER_CI: '1'" in job
    assert job.count("--junitxml=") == 1
    assert job.count("assert_junit_zero_skips.py") == 1
    assert "if: always()" in job and "uses: actions/upload-artifact@v4" in job
    assert "test-m06-full-geospatial-viewer" in summary
    assert "continue-on-error: true" in informational
    assert "test-interactive-viewer-macos" not in summary
