from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

import forge3d as f3d


def _load_module_by_path(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "terrain_tv6_heterogeneous_volumetrics_demo", str(path)
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _rasterio_available() -> bool:
    try:
        import rasterio

        return not getattr(rasterio, "__forge3d_stub__", False)
    except Exception:
        return False


@pytest.mark.interactive_viewer
def test_tv6_example_renders_real_dem_and_reports_budget(tmp_path: Path) -> None:
    assert _rasterio_available(), "TV6 dedicated viewer lane requires rasterio"
    repo = Path(__file__).resolve().parents[1]
    example_path = repo / "examples" / "terrain_tv6_heterogeneous_volumetrics_demo.py"
    assert example_path.exists(), "TV6 example script is required"
    mod = _load_module_by_path(example_path)

    result = mod.render_demo(
        dem_path=mod.DEFAULT_DEM,
        output_dir=tmp_path / "tv6-demo",
        width=960,
        height=600,
        max_dem_size=768,
        timeout=90.0,
    )

    baseline_path = Path(result["baseline_path"])
    contact_sheet_path = Path(result["contact_sheet_path"])
    manifest_path = Path(result["manifest_path"])

    assert baseline_path.exists()
    assert contact_sheet_path.exists()
    assert manifest_path.exists()
    assert baseline_path.stat().st_size > 0
    assert contact_sheet_path.stat().st_size > 0
    assert manifest_path.stat().st_size > 0

    baseline = f3d.png_to_numpy(baseline_path)
    contact_sheet = f3d.png_to_numpy(contact_sheet_path)
    assert baseline.shape == (600, 960, 4)
    assert contact_sheet.shape == (600, 3870, 4)

    baseline_report = result["baseline_report"]
    assert baseline_report["active_volume_count"] == 0
    assert baseline_report["texture_bytes"] == 0

    scenes = result["scenes"]
    assert set(scenes.keys()) == {"valley_fog", "plume", "localized_haze"}

    for name, scene in scenes.items():
        path = Path(scene["path"])
        report = scene["report"]

        assert path.exists(), f"{name} snapshot missing"
        assert path.stat().st_size > 0, f"{name} snapshot empty"
        assert report["active_volume_count"] == 1
        assert report["texture_bytes"] > 0
        assert report["texture_bytes"] <= report["memory_budget_bytes"]
        assert report["atlas_dimensions"][0] > 0
        assert report["atlas_dimensions"][1] > 0
        assert report["atlas_dimensions"][2] > 0
        assert report["raymarch_steps"] >= 40
        assert scene["mean_abs_diff"] > 1.0, f"{name} image delta too small"
        assert scene["changed_pixels"] > 20_000, f"{name} changed too few pixels"
        assert scene["render_seconds"] < 45.0, f"{name} render time unexpectedly high"

    assert result["baseline_render_seconds"] < 45.0
    assert result["total_render_seconds"] < 120.0
