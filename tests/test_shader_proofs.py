from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
from _toml_compat import load_toml

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE

if not NATIVE_AVAILABLE:
    pytest.skip("shader proof tests require the compiled native extension", allow_module_level=True)

from forge3d import verify


ROOT = pathlib.Path(__file__).resolve().parent.parent


def _stable(report: dict) -> dict:
    report = copy.deepcopy(report)
    for verdict in report["verdicts"]:
        verdict["timing_ms"] = 0.0
    report["ablations"]["height_range_div"]["timing_ms"] = 0.0
    report["ablations"]["pt_shade_delete_guard"]["timing_ms"] = 0.0
    report["unsafe_fixture"]["timing_ms"] = 0.0
    return report


def test_proven_list_clean_and_large_enough():
    report = verify.shader_report()
    assert report["status"] == "ok"
    assert report["proven_count"] >= 10
    assert report["module_count"] >= 8
    assert report["suppressions"] == []
    assert all(v["proof_status"] == "proven" for v in report["verdicts"])
    assert sum(v["timing_ms"] for v in report["verdicts"]) < 300_000


def test_unsafe_fixture_rejected_and_accumulation_proved():
    report = verify.shader_report()
    unsafe = report["unsafe_fixture"]
    assert unsafe["proof_status"] == "unproven"
    assert any(a["kind"] == "possible_nan_or_inf" for a in unsafe["alarms"])

    hybrid = next(v for v in report["verdicts"] if v["module"] == "hybrid_terrain_traversal")
    assert hybrid["proof_status"] == "proven"
    assert hybrid["alarms"] == []
    assert "declared_output_ranges" in hybrid["claims"]


def test_seeded_ablations_are_caught():
    report = verify.shader_report()
    div = report["ablations"]["height_range_div"]
    assert div["baseline_proof_status"] == "proven"
    assert div["proof_status"] == "unproven"
    assert div["alarms"]

    guard = report["ablations"]["pt_shade_delete_guard"]
    assert guard["baseline_proof_status"] == "proven"
    assert guard["proof_status"] == "unproven"
    assert guard["alarms"]


def test_runtime_contract_assert_mode_fails_closed_without_observations():
    runtime_assert = verify.shader_report()["runtime_assert"]
    if runtime_assert["status"] != "not_run":
        pytest.skip("a prior test in this process already recorded runtime shader-contract observations")
    assert runtime_assert["status"] == "not_run"
    assert runtime_assert["checked_scenes"] == 0
    assert runtime_assert["observed_inputs"] is False

@pytest.mark.apple_metal_physical
def test_runtime_contract_asserts_observed_gpu_inputs():
    if not f3d.has_gpu():
        pytest.skip("runtime shader-contract assertions require a GPU render")

    f3d.render_brdf_tile("lambert", 0.4, 32, 32, certificate=True)
    runtime_assert = verify.shader_report()["runtime_assert"]
    if not runtime_assert["feature_enabled"]:
        pytest.skip("native extension was not built with shader-contract-asserts")

    assert runtime_assert["status"] == "passed"
    assert runtime_assert["checked_scenes"] >= 1
    assert runtime_assert["observed_inputs"] is True
    brdf = next(
        entry for entry in runtime_assert["checked_entries"] if entry["module"] == "brdf_tile"
    )
    assert brdf["entry_point"] == "fs_main"
    assert brdf["pipeline"] == "brdf_tile.pipeline"
    names = {binding["name"] for binding in brdf["checked_bindings"]}
    assert {
        "params.roughness",
        "shading.roughness",
        "debug_buffer",
        "render_target.samples",
    } <= names
    assert all(binding["status"] == "passed" for binding in brdf["checked_bindings"])

    from forge3d.path_tracing import hybrid_render_terrain_reference

    hybrid_render_terrain_reference(
        np.zeros((4, 4), dtype=np.float32),
        8,
        8,
        {
            "origin": (0.0, 3.0, 8.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0.0, 1.0, 0.0),
            "fov_y": 45.0,
            "exposure": 1.0,
        },
        sun_intensity=2.5,
        max_frames=4,
        min_frames=2,
        variance_threshold=1e30,
        certificate=True,
    )
    hybrid_report = verify.shader_report()["runtime_assert"]
    assert hybrid_report["status"] == "passed"
    hybrid = next(
        entry
        for entry in hybrid_report["checked_entries"]
        if entry["module"] == "hybrid_terrain_traversal"
    )
    assert hybrid["scene"].endswith("-4f")
    checks = {binding["name"]: binding for binding in hybrid["checked_bindings"]}
    assert checks["uniforms.frame_index"]["observed_max"] == 3.0
    assert checks["terrain_reservoirs_prev.m"]["observed_max"] > 0.0
    assert checks["out_tex.samples"]["status"] == "passed"


def test_verdicts_are_deterministic_and_containment_checked():
    first = _stable(verify.shader_report())
    second = _stable(verify.shader_report())
    assert first == second
    assert first["containment"]["status"] == "passed"
    assert first["containment"]["samples_per_op"] == 1_000_000


def test_ledger_is_exhaustive_and_unexpired():
    report = verify.shader_report()
    assert report["ledger"]["missing"] == []
    assert report["ledger"]["expired"] == []

    ledger = load_toml(ROOT / "tests/shader_proofs_ledger.toml")
    for row in ledger["unproven"]:
        assert row["path"].startswith("src/")
        assert row["reason"]
        assert row["owner"]
        assert row["expiry"] >= "2026-07-17"
