#!/usr/bin/env python3
"""Run the fixed NVIDIA/Vulkan visual acceptance selections."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


SUBSTRATIA_TESTS = (
    "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_normal_family_changes_lighting_ssim",
    "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_all_families_page_within_budget",
    "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_missing_family_is_fatal",
    "tests/test_terrain_vt_pbr_families.py::TestTerrainVTPbrFamilies::test_partial_normal_residency_degrades_gracefully",
)
VISUAL_TESTS = (
    "tests/test_terrain_visual_goldens.py",
    "tests/test_terrain_tv10_goldens.py",
    "tests/test_terrain_vt_pbr_families.py",
    "tests/test_recipe_goldens.py::test_recipe_golden_gate_rejects_pixel_regression",
    "tests/test_recipe_goldens.py::test_nvidia_vulkan_recipe_pixel_golden_render_and_match",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("visual", "substratia"), required=True)
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args()
    selected = VISUAL_TESTS if args.suite == "visual" else SUBSTRATIA_TESTS
    return int(pytest.main([*selected, f"--junitxml={args.junit}", "-v", "--tb=short"]))


if __name__ == "__main__":
    raise SystemExit(main())
