# tests/test_terrain_tv4_demo.py
# Executes the real-DEM TV4 example and validates that the example writes
# non-trivial image output from repo assets.

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available


pytestmark = pytest.mark.apple_metal_physical


def _load_module_by_path(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("terrain_tv4_material_variation_demo", str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_tv4_example_renders_real_dem(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    example_path = repo / "examples" / "terrain_tv4_material_variation_demo.py"
    assert example_path.exists(), "tracked TV4 example script is missing"
    mod = _load_module_by_path(example_path)

    result = mod.render_demo(
        dem_path=mod.DEFAULT_DEM,
        output_dir=tmp_path / "tv4-demo",
        width=960,
        height=600,
        max_dem_size=768,
    )

    baseline_path = Path(result["baseline_path"])
    varied_path = Path(result["varied_path"])
    comparison_path = Path(result["comparison_path"])

    assert baseline_path.exists()
    assert varied_path.exists()
    assert comparison_path.exists()
    assert baseline_path.stat().st_size > 0
    assert varied_path.stat().st_size > 0
    assert comparison_path.stat().st_size > 0

    baseline = f3d.png_to_numpy(baseline_path)
    varied = f3d.png_to_numpy(varied_path)
    comparison = f3d.png_to_numpy(comparison_path)

    assert baseline.shape == (600, 960, 4)
    assert varied.shape == (600, 960, 4)
    assert comparison.shape == (600, 1932, 4)

    mean_abs = float(np.mean(np.abs(baseline[..., :3].astype(np.float32) - varied[..., :3].astype(np.float32))))
    assert mean_abs > 0.10
    assert float(result["mean_abs_diff"]) >= mean_abs - 1e-3
