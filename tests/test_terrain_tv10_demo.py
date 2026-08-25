# tests/test_terrain_tv10_demo.py
# Executes the real-DEM TV10 example and validates that it writes non-trivial
# image output for two repo DEM scenes.

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
    spec = importlib.util.spec_from_file_location("terrain_tv10_subsurface_demo", str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_tv10_example_renders_real_dems(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    example_path = repo / "examples" / "terrain_tv10_subsurface_demo.py"
    assert example_path.exists(), "tracked TV10 example script is missing"
    mod = _load_module_by_path(example_path)

    result = mod.render_demo(
        output_dir=tmp_path / "tv10-demo",
        width=960,
        height=600,
        max_dem_size=768,
    )

    summary_path = Path(result["summary_path"])
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0

    summary = f3d.png_to_numpy(summary_path)
    assert summary.shape[1] == 1932

    scenes = result["scenes"]
    assert len(scenes) == 2
    for scene in scenes:
        baseline_path = Path(scene["baseline_path"])
        subsurface_path = Path(scene["subsurface_path"])
        comparison_path = Path(scene["comparison_path"])
        assert baseline_path.exists()
        assert subsurface_path.exists()
        assert comparison_path.exists()

        baseline = f3d.png_to_numpy(baseline_path)
        subsurface = f3d.png_to_numpy(subsurface_path)
        comparison = f3d.png_to_numpy(comparison_path)

        assert baseline.shape == (600, 960, 4)
        assert subsurface.shape == (600, 960, 4)
        assert comparison.shape == (600, 1932, 4)

        diff = np.abs(baseline[..., :3].astype(np.float32) - subsurface[..., :3].astype(np.float32))
        mean_abs = float(np.mean(diff))
        peak_p99 = float(np.percentile(diff, 99.0))
        assert mean_abs > 0.08
        assert peak_p99 > 3.0
        assert float(scene["mean_abs_diff"]) >= mean_abs - 1e-3
        assert float(scene["peak_p99_diff"]) >= peak_p99 - 1e-3
