from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from scripts.ci_pytest_lane import full_lane_files


ROOT = Path(
    os.environ.get(
        "FORGE3D_CI_CONTRACT_ROOT", Path(__file__).resolve().parents[1]
    )
)
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _continued_command_arguments(block: str, command: str) -> tuple[str, ...]:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != f"{command} \\":
            continue
        arguments = []
        for argument_line in lines[index + 1 :]:
            argument = argument_line.strip()
            continued = argument.endswith("\\")
            arguments.append(argument.removesuffix("\\").strip())
            if not continued:
                return tuple(arguments)
    raise AssertionError(f"continued command not found: {command}")


def test_ci_downloads_verified_lfs_media_once_and_shares_one_artifact() -> None:
    workflow = _workflow()
    python_workflow = (
        ROOT / ".github" / "workflows" / "test-python-wheel.yml"
    ).read_text(encoding="utf-8")

    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\bgit\s+lfs\s+(?:pull|fetch|checkout)\b", text), (
            f"{path.name} reintroduced per-job LFS transfer"
        )
        data = yaml.load(text, Loader=yaml.BaseLoader)
        for job_name, job in data.get("jobs", {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps", []):
                if step.get("uses") != "actions/checkout@v4":
                    continue
                lfs = step.get("with", {}).get("lfs", "false").casefold()
                assert lfs not in {"true", "1", "yes", "on"}, (
                    f"{path.name}:{job_name} checkout reintroduced LFS fanout"
                )
    assert (
        "https://media.githubusercontent.com/media/milos-agathon/forge3d/"
        "92a86baa3c8f6ba3c3a7368e4f80d4004905a433"
    ) in workflow
    assert (
        "https://media.githubusercontent.com/media/milos-agathon/forge3d/"
        "19028116dfbe9b954eac63e0a211be0d544226aa"
    ) in workflow
    assert workflow.count("sha256sum --check -") == 1
    assert "cff39b4e02d7ba13c48f3d8b1a4080d40ada753ade62fa951459fe4e01e98b48" in workflow
    assert "875b243474b151175f76037acd60c2149ac2e46fba9ba2bbce0c9a6998015dd3" in workflow
    assert "d09d229fa265749720a6b4bd40c440799f43286bf2d401d732ea77f89d0bd478" in workflow
    assert "6b254585be4982ed9e8da63b8536ecc2f5fa4c64c6545db06c73eb1fe39a8f7f" in workflow
    assert "575e4b57d69d5db1bc16d5cd7edc8c83c570e280717e9e7ca225c060fd0aeb3d" in workflow
    assert "is still an LFS pointer" in workflow
    prepare = workflow.split("  prepare-lfs-fixtures:", 1)[1].split(
        "  terrain-golden-paths:", 1
    )[0]
    assert prepare.count("name: lfs-fixture-bundles") == 1
    assert python_workflow.count("name: lfs-fixture-bundles") == 1
    assert "if: inputs.restore_lfs" in python_workflow
    assert workflow.count("uses: actions/upload-artifact@v4") >= 1
    assert "retention-days: 1" in workflow


def test_ci_lfs_manifest_contains_only_lane_fixtures() -> None:
    workflow = _workflow()
    prepare = workflow.split("  prepare-lfs-fixtures:", 1)[1].split(
        "  terrain-golden-paths:", 1
    )[0]

    assert "assets/tif/Mount_Fuji_30m.tif" in prepare
    assert "assets/tif/dem_rainier.tif" in prepare
    assert "assets/tif/switzerland_dem.tif" in prepare
    assert prepare.count("assets/tif/switzerland_land_cover.tif") == 3
    assert "assets/tif/moon_south_pole_lola.tif" in prepare
    assert "python/forge3d/forge3d.pdb" not in prepare
    assert "assets/highres.png" not in prepare
    assert "assets/swiss-legend.png" not in prepare
    assert "assets/tif/Bryce_Canyon.tif" not in prepare
    assert "assets/tif/luxembourg_dem.tif" not in prepare

    python_bundle = _continued_command_arguments(
        prepare, "zip -q lfs-fixture-bundles/python-tiffs.zip"
    )
    assert len(python_bundle) == 5
    assert set(python_bundle) == {
        "assets/tif/Mount_Fuji_30m.tif",
        "assets/tif/dem_rainier.tif",
        "assets/tif/switzerland_dem.tif",
        "assets/tif/switzerland_land_cover.tif",
        "assets/tif/moon_south_pole_lola.tif",
    }

    m06_bundle = _continued_command_arguments(
        prepare, "zip -q lfs-fixture-bundles/m06-dem.zip"
    )
    assert m06_bundle == ("assets/tif/switzerland_dem.tif",)


def test_python_golden_and_m06_restore_only_their_fixture_bundles() -> None:
    workflow = _workflow()
    python_workflow = (
        ROOT / ".github" / "workflows" / "test-python-wheel.yml"
    ).read_text(encoding="utf-8")
    slow_job = workflow.split("  test-python-slow:", 1)[1].split(
        "\n  # ============================================================================\n  # TERMINUS", 1
    )[0]
    golden_job = workflow.split("  test-golden-images:", 1)[1].split(
        "  test-m06-full-geospatial-viewer:", 1
    )[0]
    m06_job = workflow.split("  test-m06-full-geospatial-viewer:", 1)[1].split(
        "\n  # ============================================================================\n  # COMPENDIUM F3DZ", 1
    )[0]

    assert python_workflow.count("python-tiffs.zip") == 1
    assert "m06-dem.zip" not in python_workflow
    assert "forge3d.pdb" not in python_workflow
    assert "needs: [build-wheel-linux, prepare-lfs-fixtures]" in slow_job
    assert slow_job.count(".zip") == 1
    assert "python-tiffs.zip" in slow_job
    assert "m06-dem.zip" not in slow_job
    for full_job in (
        "test-python-core",
        "test-python-full-linux",
        "test-python-full-windows",
        "test-python-full-macos",
    ):
        assert "restore_lfs: true" in workflow.split(f"  {full_job}:", 1)[1].split(
            "\n\n", 1
        )[0]
    for smoke_job in (
        "test-python-compat-linux",
        "test-python-compat-windows",
        "test-python-compat-macos",
    ):
        assert "restore_lfs:" not in workflow.split(f"  {smoke_job}:", 1)[1].split(
            "\n\n", 1
        )[0]
    python_restore = (
        "run: python -m zipfile -e lfs-fixture-bundles/python-tiffs.zip ."
    )
    assert (
        "needs: [build-wheel-windows, prepare-lfs-fixtures, terrain-golden-paths]"
        in golden_job
    )
    assert golden_job.count("name: lfs-fixture-bundles") == 1
    assert golden_job.count(python_restore) == 1
    assert "m06-dem.zip" not in golden_job
    assert "python -m pip install pytest numpy pyproj pillow rasterio" in golden_job
    assert "name: Run strict OBLIQUA NVIDIA Vulkan gates" in golden_job
    assert "FORGE3D_RUN_OBLIQUA_GPU: '1'" in golden_job
    assert "name: Upload OBLIQUA artifacts\n        if: always()" in golden_job

    assert "needs: [build-wheel-windows, prepare-lfs-fixtures, terrain-golden-paths]" in m06_job
    assert "m06-dem.zip" in m06_job
    assert "python-tiffs.zip" not in m06_job
    assert "Get-PSDrive -PSProvider FileSystem" in m06_job
    assert "Sort-Object Free -Descending" in m06_job
    assert (
        '$scratchScope = "$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT-$env:GITHUB_JOB"'
        in m06_job
    )
    assert "FORGE3D_M06_SCRATCH_DIR" in m06_job
    assert '"CARGO_TARGET_DIR=$targetDir"' in m06_job
    assert '"FORGE3D_VIEWER_BINARY=$viewerBinary"' in m06_job
    assert "name: Clean M-06 build scratch" in m06_job
    assert "(Split-Path -Leaf $parentDir) -ne 'forge3d-ci-scratch'" in m06_job
    assert m06_job.index("name: Upload M-06 evidence") < m06_job.index(
        "name: Clean M-06 build scratch"
    )


def test_m06_path_filter_and_aggregator_contract() -> None:
    workflow = _workflow()
    paths_job = workflow.split("  terrain-golden-paths:", 1)[1].split(
        "\n  # ============================================================================\n  # Rust Tests", 1
    )[0]
    assert "m06: ${{ steps.filter.outputs.m06 }}" in paths_job
    m06_paths = paths_job.split("            m06:\n", 1)[1].split(
        "\n            f3dz:\n", 1
    )[0]
    for pattern in (
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        ".cargo/config.toml",
        "pyproject.toml",
        "pytest.ini",
        "conftest.py",
        "src/**",
        "python/forge3d/**",
        "scripts/install_compatible_wheel.py",
        "scripts/terrain_ci_probe.py",
        "scripts/assert_junit_zero_skips.py",
        "scripts/summarize_m06_evidence.py",
        "scripts/ci_pytest_lane.py",
        "tests/*.py",
        "tests/*.toml",
        "tests/golden/certificates/**",
        "tests/golden/recipes/mapscene_terrain_raster.png",
        "tests/data/vector_torture/cases.json",
        "assets/lidar/MtStHelens.laz",
        "assets/tif/switzerland_dem.tif",
        "assets/fonts/**",
        "assets/geoid/egm96_n120.bin",
    ):
        assert f"              - '{pattern}'" in m06_paths
    for broad_pattern in (
        "docs/**",
        "examples/**",
        "python/**",
        "tests/**",
        "assets/**",
    ):
        assert f"              - '{broad_pattern}'" not in m06_paths
    for workflow_path in (
        ".github/workflows/ci.yml",
        ".github/workflows/build-wheel.yml",
    ):
        assert f"              - '{workflow_path}'" in m06_paths

    m06_job = workflow.split("  test-m06-full-geospatial-viewer:", 1)[1].split(
        "\n  # ============================================================================\n  # COMPENDIUM F3DZ", 1
    )[0]
    assert "needs: [build-wheel-windows, prepare-lfs-fixtures, terrain-golden-paths]" in m06_job
    assert "if: >-" in m06_job
    for clause in (
        "github.event_name == 'schedule'",
        "inputs.scope == 'full'",
        "inputs.scope == 'm06'",
    ):
        assert clause in m06_job
    for forbidden in (
        "github.event_name == 'pull_request'",
        "github.event_name == 'push'",
        "run-physical",
        "needs.terrain-golden-paths.outputs.m06",
    ):
        assert forbidden not in m06_job

    aggregate = workflow.split("  full-acceptance-summary:", 1)[1]
    assert (
        "m06_selected=\"${{ github.event_name == 'schedule' || "
        "(github.event_name == 'workflow_dispatch' && (inputs.scope == 'full' || "
        "inputs.scope == 'm06')) }}\"" in aggregate
    )
    assert (
        "check_selected \"$m06_selected\" '${{ needs.test-m06-full-geospatial-viewer.result }}' m06-physical"
        in aggregate
    )


def test_m06_acceptance_keeps_only_unique_physical_coverage() -> None:
    workflow = _workflow()
    m06_job = workflow.split("  test-m06-full-geospatial-viewer:", 1)[1].split(
        "\n  # ============================================================================\n  # COMPENDIUM F3DZ", 1
    )[0]
    acceptance = m06_job.split(
        "      - name: Run M-06 source and live acceptance", 1
    )[1].split("      - name: Summarize public M-06 evidence", 1)[0]

    assert "      - name: Run M-06 Rust ABI and adapter gates" not in m06_job
    canonical_lane = set(full_lane_files())
    for redundant in (
        "tests/test_m06_anchoring_boundary.py",
        "tests/test_world_coord_f32_gate.py",
        "tests/test_m06_viewer_matrix_contract.py",
        "tests/test_m06_single_rebase_contract.py",
        "tests/test_m06_temporal_resource_contract.py",
        "tests/test_m06_scene_review_transaction.py",
        "tests/test_m06_command_transaction.py",
        "tests/test_m06_python_viewer_contracts.py",
        "tests/test_gis_raster.py",
        "tests/test_gis_crs_affine.py",
        "tests/test_3dtiles_parse.py",
        "tests/test_buildings_cityjson.py",
        "tests/test_viewer_ipc.py",
        "tests/test_api_contracts.py",
        "tests/test_install_smoke.py",
        "tests/test_allocation_gate.py",
        "tests/test_no_silent_degradation.py",
        "tests/test_certificate_verifier.py",
        "tests/test_render_certificate_contract.py",
        "tests/test_recipe_goldens.py::test_certificate_refresh_rejects_capability_degradation",
        "tests/test_recipe_goldens.py::test_recipe_golden_gate_rejects_pixel_regression",
    ):
        assert redundant.split("::", 1)[0] in canonical_lane
        assert f"'{redundant}'" not in acceptance

    for physical in (
        "tests/test_shadow_techniques.py::TestEvsmExposureParity::test_evsm_is_not_black",
        "tests/test_shadow_techniques.py::TestEvsmExposureParity::test_evsm_banding_is_bounded_in_raw_visibility",
        "tests/test_picking_ipc.py",
        "tests/test_m06_full_geospatial_viewer.py",
        "tests/test_terrain_viewer_pbr.py",
        "tests/test_vector_overlay_rendering.py",
        "tests/test_vector_coverage.py",
    ):
        assert f"'{physical}'" in acceptance

    picking = (ROOT / "tests/test_picking_ipc.py").read_text(encoding="utf-8")
    shadow = (ROOT / "tests/test_shadow_techniques.py").read_text(encoding="utf-8")
    assert "pytest.mark.interactive_viewer" in picking
    assert 'os.environ.get("RUN_M06_VIEWER_CI") == "1"' in picking
    assert "required M-06 NVIDIA/Vulkan viewer lane only" in picking
    assert "pytest.mark.interactive_viewer" in shadow
    assert 'os.environ.get("RUN_M06_VIEWER_CI") != "1"' in shadow
