from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import terrain_rendering_available


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "terrain_tv21_blending_demo.py"
DEFAULT_DEM = REPO_ROOT / "assets" / "tif" / "dem_rainier.tif"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("terrain_tv21_blending_demo", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.apple_metal_physical
def test_tv21_demo_renders_real_dem_outputs() -> None:
    if not terrain_rendering_available():
        pytest.skip("TV21 example requires GPU-backed terrain runtime")
    assert EXAMPLE_PATH.exists(), "tracked TV21 example script is missing"

    module = _load_example_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = module.render_tv21_demo(
            dem_path=DEFAULT_DEM,
            output_dir=Path(tmpdir),
            width=320,
            height=220,
            max_dem_size=768,
            crop_size=160,
        )

        cases = list(summary["cases"])
        assert len(cases) == 3

        contact_sheet = f3d.png_to_numpy(summary["contact_sheet_path"])
        assert contact_sheet.ndim == 3
        assert contact_sheet.shape[2] == 4
        assert int(np.count_nonzero(contact_sheet[..., :3])) > 1000

        for case in cases:
            assert case["changed_pixels"] >= 120
            assert case["mean_delta"] > 0.05
            for key in ("baseline_path", "tv21_path", "diff_path"):
                path = Path(case[key])
                assert path.exists()
                image = f3d.png_to_numpy(str(path))
                assert image.shape == (220, 320, 4)
