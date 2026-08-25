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
    "tests/test_preset_visual_parity.py",
    "tests/test_terrain_visual_goldens.py",
    "tests/test_terrain_tv10_goldens.py",
    "tests/test_terrain_vt_pbr_families.py",
    "tests/test_recipe_goldens.py::test_recipe_golden_gate_rejects_pixel_regression",
    "tests/test_recipe_goldens.py::test_nvidia_vulkan_recipe_pixel_golden_render_and_match",
    "tests/test_terrain_runtime.py::test_nvidia_vulkan_terrain_constructor_child_smoke",
)
SIDERA_TESTS = (
    "tests/test_astro_night_golden.py::test_night_golden_matches_committed_vulkan_bytes",
    "tests/test_astro_night_golden.py::test_golden_refresh_does_not_rewrite_the_committed_file_when_disabled",
    "tests/test_determinism_hash.py::test_sidera_night_vulkan_reference_replays",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("visual", "substratia", "sidera"), required=True
    )
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    selected = {
        "visual": VISUAL_TESTS,
        "substratia": SUBSTRATIA_TESTS,
        "sidera": SIDERA_TESTS,
    }[args.suite]
    pytest_args = [*selected, f"--junitxml={args.junit}", "-v", "--tb=short"]
    if args.collect_only:
        pytest_args.extend(("--collect-only", "-q"))
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
