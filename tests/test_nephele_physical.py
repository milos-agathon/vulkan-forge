"""Exact six-case, zero-skip NEPHELE physical acceptance suite.

Collect with ``-o python_functions=gate*`` so JUnit names are the six binding
contract names without a synthetic ``test_`` prefix.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.nephele_evidence_report import (
    _gate1,
    _gate2,
    _gate6,
    _object,
    _policy_evaluator,
    _verify_fixture_manifest,
    _verify_identity,
    _verify_reference_provenance,
    _visual_gates,
)


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def physical() -> dict[str, Any]:
    raw = os.environ.get("FORGE3D_NEPHELE_ARTIFACT_DIR")
    if not raw:
        raise RuntimeError("FORGE3D_NEPHELE_ARTIFACT_DIR is required; physical acceptance never skips")
    artifact = Path(raw).resolve()
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    runtime = _object(artifact / "installed-wheel-runtime.json")
    identity = _verify_identity(
        artifact,
        head,
        ROOT,
        Path(runtime["installed_native_path"]),
        ("Windows", os.environ.get("RUNNER_ARCH", ""), "windows-nvidia-vulkan"),
    )
    context = identity["context"]
    manifest = _object(artifact / "fixture-manifest.json")
    bundle = _verify_fixture_manifest(artifact, manifest, ROOT, context["fixture_commit"])
    _, _, reference_camera_contract = _verify_reference_provenance(
        artifact, manifest, ROOT, context["fixture_commit"]
    )
    evaluator = _policy_evaluator(_object(artifact / "gate4-policy.json"))
    visual = _visual_gates(artifact, evaluator, ROOT, head)
    return {
        "artifact": artifact,
        "head": head,
        "identity": identity,
        "manifest": manifest,
        "bundle": bundle,
        "visual": visual,
        "reference_camera_contract": reference_camera_contract,
    }


def gate1_estimator_majorant_rr(physical: dict[str, Any]) -> None:
    gate = _gate1(
        physical["artifact"],
        physical["head"],
        ROOT,
        physical["manifest"]["scene_inputs"]["medium"],
    )
    assert gate["homogeneous"]["three_standard_error_pass"] is True
    assert gate["heterogeneous"]["relative_error"] < 0.005
    assert gate["majorant"]["probe_count"] == 1_000_000
    assert gate["majorant"]["violation_count"] == 0
    assert gate["russian_roulette"]["relative_difference"] < 0.002


def gate2_energy(physical: dict[str, Any]) -> None:
    assert _gate2(physical["artifact"], physical["head"], ROOT)["relative_residual"] <= 1.0e-3


def gate3_realtime_reference(physical: dict[str, Any]) -> None:
    gate = physical["visual"][0]
    assert gate["sky_cloud_delta_e_pass_fraction"] >= 0.95
    assert gate["godray_roi_ssim"] > 0.95


def gate4_terrain_coupling(physical: dict[str, Any]) -> None:
    gate = physical["visual"][1]
    assert gate["shadow_aggregate"] < 2.0
    assert gate["medium_ablation_changed_fraction"] >= 0.10
    assert gate["terrain_occlusion_ablation_ssim"] < 0.80


def gate5_compute_shadow_ridgeline(physical: dict[str, Any]) -> None:
    gate = physical["visual"][2]
    assert gate["compute_entry_texture_sample_compare_calls"] == 0
    assert gate["stale_disabled_shadow_comments"] == 0
    assert gate["ridgeline_within_one_slice_fraction"] >= 0.99


def gate6_determinism_memory(physical: dict[str, Any]) -> None:
    gate = _gate6(
        physical["artifact"],
        physical["head"],
        physical["identity"]["context"],
        physical["identity"]["adapter"],
        physical["bundle"],
        physical["reference_camera_contract"],
    )
    assert gate["frame_hashes"][0] == gate["frame_hashes"][1]
    assert gate["memory"]["peak_host_visible_bytes"] < 512 * 1024**2
    assert gate["sun_transmittance"]["method"] == "bounded_nested_midpoint"
    assert (
        gate["sun_transmittance"]["bias"]
        == "fine_midpoint_with_coarse_fine_abs_rgb_error"
    )
    assert gate["sun_transmittance"]["max_segment_length"] > 0.0
    assert gate["sun_transmittance"]["executed_steps"] > 0
    assert 0.0 <= gate["sun_transmittance"]["max_abs_error"] <= 1.0
