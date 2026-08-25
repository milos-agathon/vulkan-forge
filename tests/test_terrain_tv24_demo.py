# Executes the real-DEM TV24 example and validates that the example writes
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
    spec = importlib.util.spec_from_file_location("terrain_tv24_reflection_probe_demo", str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_tv24_example_renders_real_dem(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    example_path = repo / "examples" / "terrain_tv24_reflection_probe_demo.py"
    assert example_path.exists(), "tracked TV24 example script is missing"
    mod = _load_module_by_path(example_path)

    result = mod.render_demo(
        dem_path=mod.DEFAULT_DEM,
        output_dir=tmp_path / "tv24-demo",
        width=960,
        height=600,
        max_dem_size=768,
    )

    diffuse_path = Path(result["diffuse_path"])
    reflection_path = Path(result["reflection_path"])
    reflection_debug_path = Path(result["reflection_debug_path"])
    reflection_weight_path = Path(result["reflection_weight_path"])
    comparison_path = Path(result["comparison_path"])

    for path in [
        diffuse_path,
        reflection_path,
        reflection_debug_path,
        reflection_weight_path,
        comparison_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0

    diffuse = f3d.png_to_numpy(diffuse_path)
    reflection = f3d.png_to_numpy(reflection_path)
    reflection_debug = f3d.png_to_numpy(reflection_debug_path)
    reflection_weight = f3d.png_to_numpy(reflection_weight_path)
    comparison = f3d.png_to_numpy(comparison_path)

    assert diffuse.shape == (600, 960, 4)
    assert reflection.shape == (600, 960, 4)
    assert reflection_debug.shape == (600, 960, 4)
    assert reflection_weight.shape == (600, 960, 4)
    assert comparison.shape == (600, 1932, 4)

    mean_abs = float(np.mean(np.abs(diffuse[..., :3].astype(np.float32) - reflection[..., :3].astype(np.float32))))
    water_mean_abs = float(result["water_mean_abs_diff"])
    assert water_mean_abs > 0.10
    assert float(result["mean_abs_diff"]) >= mean_abs - 1e-3
    assert float(np.mean(reflection_debug[..., :3])) > 1.0
    assert float(np.mean(reflection_weight[..., 0])) > 1.0
    assert int(result["water_pixels"]) > 0
    assert int(result["rendered_water_pixels"]) > 0
    assert int(result["reflection_probe_memory"]["probe_count"]) > 0
