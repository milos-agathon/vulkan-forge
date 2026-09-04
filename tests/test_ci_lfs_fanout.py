from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def test_ci_downloads_lfs_once_and_shares_one_artifact() -> None:
    workflow = _workflow()
    prepare = workflow.split("  prepare-lfs-fixtures:", 1)[1].split(
        "  terrain-golden-paths:", 1
    )[0]

    assert workflow.count("git lfs pull") == 1
    assert "lfs: true" not in workflow
    assert workflow.count("name: lfs-fixture-bundles") == 4
    assert prepare.count("uses: actions/upload-artifact@v4") == 1
    assert "retention-days: 1" in prepare


def test_ci_lfs_manifest_contains_only_lane_fixtures() -> None:
    workflow = _workflow()
    prepare = workflow.split("  prepare-lfs-fixtures:", 1)[1].split(
        "  terrain-golden-paths:", 1
    )[0]

    assert "assets/tif/Mount_Fuji_30m.tif" in prepare
    assert "assets/tif/dem_rainier.tif" in prepare
    assert "assets/tif/switzerland_dem.tif" in prepare
    assert prepare.count("assets/tif/switzerland_land_cover.tif") == 3
    assert "python/forge3d/forge3d.pdb" not in prepare
    assert "assets/highres.png" not in prepare
    assert "assets/swiss-legend.png" not in prepare
    assert "assets/tif/Bryce_Canyon.tif" not in prepare
    assert "assets/tif/luxembourg_dem.tif" not in prepare

    python_bundle = _continued_command_arguments(
        prepare, "zip -q lfs-fixture-bundles/python-tiffs.zip"
    )
    assert len(python_bundle) == 4
    assert set(python_bundle) == {
        "assets/tif/Mount_Fuji_30m.tif",
        "assets/tif/dem_rainier.tif",
        "assets/tif/switzerland_dem.tif",
        "assets/tif/switzerland_land_cover.tif",
    }

    m06_bundle = _continued_command_arguments(
        prepare, "zip -q lfs-fixture-bundles/m06-dem.zip"
    )
    assert m06_bundle == ("assets/tif/switzerland_dem.tif",)


def test_python_golden_and_m06_restore_only_their_fixture_bundles() -> None:
    workflow = _workflow()
    python_job = workflow.split("  test-python:", 1)[1].split(
        "  test-terminus-fuzz:", 1
    )[0]
    golden_job = workflow.split("  test-golden-images:", 1)[1].split(
        "  refresh-recipe-certificates:", 1
    )[0]
    m06_job = workflow.split("  test-m06-full-geospatial-viewer:", 1)[1].split(
        "  build-docs:", 1
    )[0]

    python_restore = (
        "run: python -m zipfile -e lfs-fixture-bundles/python-tiffs.zip ."
    )
    m06_restore = (
        "run: Expand-Archive lfs-fixture-bundles/m06-dem.zip "
        "-DestinationPath . -Force"
    )

    assert "needs: [build-wheels, prepare-lfs-fixtures]" in python_job
    assert python_job.count(python_restore) == 1
    assert "m06-dem.zip" not in python_job
    assert "forge3d.pdb" not in python_job

    assert (
        "needs: [build-wheels, terrain-golden-paths, prepare-lfs-fixtures]"
        in golden_job
    )
    assert "name: lfs-fixture-bundles" in golden_job
    assert golden_job.count(python_restore) == 1
    assert "m06-dem.zip" not in golden_job

    assert "needs: [build-wheels, prepare-lfs-fixtures]" in m06_job
    assert m06_job.count(m06_restore) == 1
    assert "python-tiffs.zip" not in m06_job
