"""Focused fail-closed tests for the nine-result TESSELLA evidence report."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tessella_evidence_report.py"
HEAD_SHA = "344ce414737bd3e6c76571e6c16d87487b856dba"
LOD_TEST_NAME = (
    "terrain::clipmap::gpu_lod::tests::"
    "gpu_and_cpu_select_identical_tile_sets_for_1000_cameras"
)
HZB_TEST_NAME = (
    "terrain::culling::two_phase::tests::"
    "hzb_cull_shader_matches_the_cpu_occlusion_predicate"
)
SUPPLEMENTAL_GATES = (
    "bc5_fidelity_flat_baseline",
    "bc7_fidelity_smooth_baseline",
    "bindless_atlas",
    "flythrough_crack_detector_control",
    "flythrough_pop_gate_control",
    "flythrough_visibility_coverage",
)


def _valid_results() -> dict[str, dict]:
    return {
        "vt_out_of_core": {
            "gate": "vt_out_of_core",
            "width": 3840,
            "height": 2160,
            "logical_texel_bytes": 256 * 1024**3,
            "settling_frames": 8,
            "fallback_texels": 0,
            "peak_host_visible_bytes": 512 * 1024**2 - 1,
            "atlas_device_local_bytes": 300,
            "atlas_uncompressed_equivalent_bytes": 1_000,
            "atlas_compression_ratio": 10 / 3,
            "atlas_device_local_bytes_albedo": 100,
            "atlas_device_local_bytes_normal": 100,
            "atlas_device_local_bytes_mask": 100,
            "atlas_uncompressed_equivalent_bytes_albedo": 400,
            "atlas_uncompressed_equivalent_bytes_normal": 200,
            "atlas_uncompressed_equivalent_bytes_mask": 400,
            "atlas_compression_ratio_albedo": 4.0,
            "atlas_compression_ratio_normal": 2.0,
            "atlas_compression_ratio_mask": 4.0,
        },
        "hzb_occlusion": {
            "gate": "hzb_occlusion",
            "cull_percent": 79.0,
            "frustum_passing": 100,
            "phase1_drawn": 20,
            "phase1_rejected": 80,
            "phase2_recovered": 1,
            "final_drawn": 21,
            "baseline_gpu_ms": 18.0,
            "culled_gpu_ms": 10.0,
            "speedup": 1.8,
            "speedup_target": 1.8,
            "speedup_gate": 1.7,
            "timestamp_query": True,
            "bitwise_identical": True,
        },
        "hzb_history_recovery": {
            "gate": "hzb_history_recovery",
            "phase1_rejected": 12,
            "phase2_recovered": 3,
            "bitwise_identical": True,
        },
        "visibility_buffer": {
            "gate": "visibility_buffer",
            "visible_pixels": 1_000,
            "background_pixels": 0,
            "visibility_feedback_records": 1_000,
            "forward_feedback_records": 1_200,
            "material_invocations": 1_000,
            "forward_material_invocations": 1_200,
            "measured_overdraw_factor": 1.2,
            "fallback_texels": 0,
            "picking_samples": 10_000,
            "picking_hits": 5_000,
            "gpu_picking_repeat_matches": 10_000,
            "gpu_cpu_picking_compared": 4_000,
            "gpu_cpu_picking_excluded": 6_000,
            "gpu_cpu_picking_matches": 4_000,
            "bitwise_identical_to_forward": True,
        },
        "bc7_fidelity": {
            "gate": "bc7_fidelity",
            "texture_family": "albedo",
            "fixture": "hard_albedo_256",
            "source_bytes": 262_144,
            "encoded_bytes": 65_536,
            "compression_ratio": 4.0,
            "ssim": 0.99,
            "mean_delta_e_2000": 1.0,
            "ssim_gate": 0.98,
            "mean_delta_e_gate": 1.5,
        },
        "bc5_fidelity": {
            "gate": "bc5_fidelity",
            "texture_family": "normal",
            "fixture": "steep_normals_256",
            "source_bytes": 131_072,
            "encoded_bytes": 65_536,
            "compression_ratio": 2.0,
            "mean_angular_error_degrees": 0.5,
            "max_angular_error_degrees": 3.0,
            "mean_angle_gate_degrees": 1.0,
            "max_angle_gate_degrees": 4.0,
        },
        "flythrough_popping": {
            "gate": "flythrough_popping",
            "frames": 600,
            "rendered_frames_total": 600,
            "width": 1280,
            "height": 720,
            "worst_frame_crack_count": 0,
            "crack_count": 0,
            "depth_sample_count": 20,
            "frames_crack_checked": 600,
            "max_delta_e_2000": 0.9,
            "camera_step_px": 0.01,
            "camera_path_distance_m": 165.0,
            "distinct_camera_positions": 600,
            "clipmap_center_step_m": 1.0,
            "clipmap_center_path_m": 599.0,
            "actual_clipmap_center_transitions": 569,
            "distinct_clipmap_centers": 481,
            "regions_on_screen": 3,
        },
        "vt_request_retention": {
            "gate": "vt_request_retention",
            "feedback_not_ready_frames": 30,
            "convergence_budget_frames": 8,
            "convergence_frames": 8,
            "retained_set_size": 12,
            "retained_set_identical_every_not_ready_frame": True,
            "retained_requests_after_convergence": 0,
            "tiles_streamed": 12,
        },
        "capability_degradations": {
            "gate": "capability_degradations",
            "adapter": "NVIDIA GeForce RTX 3070",
            "backend": "Vulkan",
            "degradations": [],
            "degradation_count": 0,
            "tessella_capabilities_degraded": [],
        },
    }


def _write_results(artifact_dir: Path, results: dict[str, dict] | None = None) -> None:
    for gate, payload in (results or _valid_results()).items():
        (artifact_dir / f"{gate}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _write_provenance(artifact_dir)


def _write_provenance(artifact_dir: Path) -> None:
    (artifact_dir / "checked-out-head.txt").write_text(
        f"{HEAD_SHA}\n", encoding="utf-8"
    )
    (artifact_dir / "run-context.json").write_text(
        json.dumps(
            {
                "repository": "milos-agathon/forge3d",
                "head_sha": HEAD_SHA,
                "checked_out_head": HEAD_SHA,
                "runner_name": "forge3d-rtx3070",
                "runner_os": "Windows",
                "runner_arch": "X64",
                "required_backend": "vulkan",
                "required_labels": [
                    "self-hosted",
                    "Windows",
                    "X64",
                    "forge3d-gpu",
                    "gpu-nvidia",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "adapter-probe.json").write_text(
        json.dumps(
            {
                "requested_backend": "vulkan",
                "probe": {
                    "status": "ok",
                    "name": "NVIDIA GeForce RTX 3070",
                    "vendor": 0x10DE,
                    "device": 0x2484,
                    "backend": "Vulkan",
                    "device_type": "DiscreteGpu",
                    "software_fallback": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "gpu-cpu-lod-differential.log").write_text(
        "running 1 test\n"
        f"test {LOD_TEST_NAME} ... ok\n\n"
        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
        "693 filtered out; finished in 0.10s\n",
        encoding="utf-8",
    )
    (artifact_dir / "hzb-conservativeness-differential.log").write_text(
        "running 1 test\n"
        f"test {HZB_TEST_NAME} ... ok\n\n"
        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; "
        "693 filtered out; finished in 0.10s\n",
        encoding="utf-8",
    )
    (artifact_dir / "junit.xml").write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="tessella" name="gate_one"/>'
        '<testcase classname="tessella" name="gate_two"/>'
        "</testsuite>",
        encoding="utf-8",
    )


def _run(artifact_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(artifact_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_report_accepts_all_nine_core_results_and_is_deterministic(
    tmp_path: Path,
) -> None:
    _write_results(tmp_path)
    for gate in SUPPLEMENTAL_GATES:
        (tmp_path / f"{gate}.json").write_text(
            json.dumps({"gate": gate}), encoding="utf-8"
        )

    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr
    report_path = tmp_path / "verification-report.json"
    first_bytes = report_path.read_bytes()
    report = json.loads(first_bytes)

    assert report["schema"] == "forge3d.tessella_verification/1"
    assert report["status"] == "pass"
    assert report["core_gate_count"] == 9
    assert [item["gate"] for item in report["results"]] == list(_valid_results())
    assert all(item["status"] == "pass" for item in report["results"])
    hzb = next(item for item in report["results"] if item["gate"] == "hzb_occlusion")
    assert hzb["thresholds"]["speedup_min"] == 1.8
    assert report["supplemental_json_files"] == [
        f"{gate}.json" for gate in SUPPLEMENTAL_GATES
    ]
    assert report["provenance"]["exact_head"] is True
    assert report["provenance"]["checked_out_head"] == HEAD_SHA
    assert report["provenance"]["junit"] == {
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert report["provenance"]["gpu_cpu_lod_differential"] == {
        "test": LOD_TEST_NAME,
        "camera_count": 1_000,
        "status": "pass",
    }
    assert report["provenance"]["hzb_conservativeness_differential"] == {
        "test": HZB_TEST_NAME,
        "status": "pass",
    }
    assert set(report["provenance"]["input_sha256"]) == {
        *(f"{gate}.json" for gate in _valid_results()),
        *(f"{gate}.json" for gate in SUPPLEMENTAL_GATES),
        "adapter-probe.json",
        "checked-out-head.txt",
        "gpu-cpu-lod-differential.log",
        "hzb-conservativeness-differential.log",
        "junit.xml",
        "run-context.json",
    }
    assert "9/9 core gates passed; status=PASS" in first.stdout

    second = _run(tmp_path)
    assert second.returncode == 0, second.stderr
    assert report_path.read_bytes() == first_bytes


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing", "missing core evidence file: hzb_occlusion.json"),
        ("invalid_json", "hzb_occlusion.json: invalid JSON"),
        ("non_object", "hzb_occlusion.json: JSON root must be an object"),
        ("wrong_gate", "gate identity mismatch"),
        ("missing_field", "missing required field 'max_angular_error_degrees'"),
        ("boolean_numeric", "boolean is not numeric evidence"),
        ("numeric_string", "expected numeric value"),
        ("non_finite", "non-finite numeric value at $.ssim"),
        ("positive_infinity", "non-finite numeric value at $.ssim"),
        ("empty_evidence", "missing required field 'cull_percent'"),
        ("weak_hzb", "clear the recorded regression floor"),
        ("inconsistent_hzb", "speedup is inconsistent"),
        ("inconsistent_cull", "cull_percent is inconsistent"),
        ("inconsistent_bc", "compression_ratio is inconsistent"),
        ("inconsistent_degradations", "degradation_count must equal"),
        ("wrong_resolution", "render size must equal 3840x2160"),
        ("peak_boundary", "peak_host_visible_bytes must be > 0 and < 512 MiB"),
        ("zero_peak", "peak_host_visible_bytes must be > 0 and < 512 MiB"),
        ("wrong_family_ratio", "normal atlas compression ratio must equal 2"),
        ("family_sum", "per-family device-local bytes must sum"),
        ("stationary_camera", "camera_step_px must be > 0"),
        (
            "missing_fly_runtime_field",
            "missing required field 'rendered_frames_total'",
        ),
        (
            "fly_rendered_frames_boundary",
            "frames, rendered_frames_total, and frames_crack_checked must equal 600",
        ),
        (
            "fly_center_transitions_boundary",
            "actual_clipmap_center_transitions must be between 540 and 599",
        ),
        ("fly_center_path_boundary", "clipmap_center_path_m must be > 0"),
        (
            "fly_center_step_inconsistent",
            "clipmap_center_step_m must equal actual path divided by 599 transitions",
        ),
        (
            "fly_distinct_centers_boundary",
            "distinct_clipmap_centers must be between 480 and 600",
        ),
        ("fly_regions_boundary", "regions_on_screen must be >= 3"),
        ("bc7_delta_boundary", "mean_delta_e_2000 must be < 1.5"),
        ("bc5_mean_boundary", "mean_angular_error_degrees must be < 1.0"),
        ("bc5_max_boundary", "max_angular_error_degrees must be < 4.0"),
        ("fly_delta_boundary", "max_delta_e_2000 must be < 1.0"),
        ("altered_bc7_gate", "ssim_gate must retain the literal 0.98 threshold"),
        ("missing_provenance", "missing provenance file: run-context.json"),
        ("head_mismatch", "exact-head mismatch"),
        ("software_adapter", "software_fallback must be false"),
        ("lod_failure", "exact 1,000-camera differential must report PASS"),
        ("hzb_failure", "exact real-shader HZB differential must report PASS"),
        ("junit_skip", "required lane was not clean and zero-skip"),
    ],
)
def test_report_fails_closed_on_invalid_core_evidence(
    tmp_path: Path, case: str, error: str
) -> None:
    results = deepcopy(_valid_results())
    _write_results(tmp_path, results)

    if case == "missing":
        (tmp_path / "hzb_occlusion.json").unlink()
        # A supplemental file with the missing gate identity cannot substitute
        # for the required identity-named core file.
        (tmp_path / "replacement.json").write_text(
            json.dumps(results["hzb_occlusion"]), encoding="utf-8"
        )
    elif case == "invalid_json":
        (tmp_path / "hzb_occlusion.json").write_text("{", encoding="utf-8")
    elif case == "non_object":
        (tmp_path / "hzb_occlusion.json").write_text("[]", encoding="utf-8")
    elif case == "wrong_gate":
        results["hzb_occlusion"]["gate"] = "visibility_buffer"
        _write_results(tmp_path, results)
    elif case == "missing_field":
        del results["bc5_fidelity"]["max_angular_error_degrees"]
        _write_results(tmp_path, results)
    elif case == "boolean_numeric":
        results["vt_out_of_core"]["settling_frames"] = True
        _write_results(tmp_path, results)
    elif case == "numeric_string":
        results["vt_out_of_core"]["logical_texel_bytes"] = str(256 * 1024**3)
        _write_results(tmp_path, results)
    elif case == "non_finite":
        results["bc7_fidelity"]["ssim"] = float("nan")
        _write_results(tmp_path, results)
    elif case == "positive_infinity":
        results["bc7_fidelity"]["ssim"] = float("inf")
        _write_results(tmp_path, results)
    elif case == "empty_evidence":
        (tmp_path / "hzb_occlusion.json").write_text(
            json.dumps({"gate": "hzb_occlusion"}), encoding="utf-8"
        )
    elif case == "weak_hzb":
        results["hzb_occlusion"]["baseline_gpu_ms"] = 16.9
        results["hzb_occlusion"]["speedup"] = 1.69
        _write_results(tmp_path, results)
    elif case == "inconsistent_hzb":
        results["hzb_occlusion"]["speedup"] = 2.0
        _write_results(tmp_path, results)
    elif case == "inconsistent_cull":
        results["hzb_occlusion"]["cull_percent"] = 80.0
        _write_results(tmp_path, results)
    elif case == "inconsistent_bc":
        results["bc7_fidelity"]["compression_ratio"] = 4.1
        _write_results(tmp_path, results)
    elif case == "inconsistent_degradations":
        results["capability_degradations"]["degradations"] = ["terrain_hzb_two_phase"]
        _write_results(tmp_path, results)
    elif case == "wrong_resolution":
        results["vt_out_of_core"]["width"] = 1920
        _write_results(tmp_path, results)
    elif case == "peak_boundary":
        results["vt_out_of_core"]["peak_host_visible_bytes"] = 512 * 1024**2
        _write_results(tmp_path, results)
    elif case == "zero_peak":
        results["vt_out_of_core"]["peak_host_visible_bytes"] = 0
        _write_results(tmp_path, results)
    elif case == "wrong_family_ratio":
        results["vt_out_of_core"]["atlas_compression_ratio_normal"] = 4.0
        _write_results(tmp_path, results)
    elif case == "family_sum":
        results["vt_out_of_core"]["atlas_device_local_bytes_mask"] = 99
        _write_results(tmp_path, results)
    elif case == "stationary_camera":
        results["flythrough_popping"]["camera_step_px"] = 0.0
        _write_results(tmp_path, results)
    elif case == "missing_fly_runtime_field":
        del results["flythrough_popping"]["rendered_frames_total"]
        _write_results(tmp_path, results)
    elif case == "fly_rendered_frames_boundary":
        results["flythrough_popping"]["rendered_frames_total"] = 599
        _write_results(tmp_path, results)
    elif case == "fly_center_transitions_boundary":
        results["flythrough_popping"]["actual_clipmap_center_transitions"] = 539
        _write_results(tmp_path, results)
    elif case == "fly_center_path_boundary":
        results["flythrough_popping"]["clipmap_center_path_m"] = 0.0
        _write_results(tmp_path, results)
    elif case == "fly_center_step_inconsistent":
        results["flythrough_popping"]["clipmap_center_step_m"] = 2.0
        _write_results(tmp_path, results)
    elif case == "fly_distinct_centers_boundary":
        results["flythrough_popping"]["distinct_clipmap_centers"] = 479
        _write_results(tmp_path, results)
    elif case == "fly_regions_boundary":
        results["flythrough_popping"]["regions_on_screen"] = 2
        _write_results(tmp_path, results)
    elif case == "bc7_delta_boundary":
        results["bc7_fidelity"]["mean_delta_e_2000"] = 1.5
        _write_results(tmp_path, results)
    elif case == "bc5_mean_boundary":
        results["bc5_fidelity"]["mean_angular_error_degrees"] = 1.0
        _write_results(tmp_path, results)
    elif case == "bc5_max_boundary":
        results["bc5_fidelity"]["max_angular_error_degrees"] = 4.0
        _write_results(tmp_path, results)
    elif case == "fly_delta_boundary":
        results["flythrough_popping"]["max_delta_e_2000"] = 1.0
        _write_results(tmp_path, results)
    elif case == "altered_bc7_gate":
        results["bc7_fidelity"]["ssim_gate"] = 0.97
        _write_results(tmp_path, results)
    elif case == "missing_provenance":
        (tmp_path / "run-context.json").unlink()
    elif case == "head_mismatch":
        (tmp_path / "checked-out-head.txt").write_text(
            "b" * 40 + "\n", encoding="utf-8"
        )
    elif case == "software_adapter":
        adapter_path = tmp_path / "adapter-probe.json"
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter["probe"]["software_fallback"] = True
        adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    elif case == "lod_failure":
        (tmp_path / "gpu-cpu-lod-differential.log").write_text(
            f"test {LOD_TEST_NAME} ... FAILED\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; "
            "693 filtered out; finished in 0.10s\n",
            encoding="utf-8",
        )
    elif case == "hzb_failure":
        (tmp_path / "hzb-conservativeness-differential.log").write_text(
            f"test {HZB_TEST_NAME} ... FAILED\n"
            "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; "
            "693 filtered out; finished in 0.10s\n",
            encoding="utf-8",
        )
    elif case == "junit_skip":
        (tmp_path / "junit.xml").write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="1">'
            '<testcase classname="tessella" name="skipped"><skipped/></testcase>'
            "</testsuite>",
            encoding="utf-8",
        )

    completed = _run(tmp_path)
    assert completed.returncode == 1
    assert error in completed.stderr
    report = json.loads(
        (tmp_path / "verification-report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "fail"
    assert any(error in item for item in report["validation_errors"])


def test_report_accepts_all_inclusive_threshold_boundaries(tmp_path: Path) -> None:
    results = _valid_results()
    results["hzb_occlusion"]["cull_percent"] = 60.0
    results["hzb_occlusion"]["phase2_recovered"] = 20
    results["hzb_occlusion"]["final_drawn"] = 40
    results["bc7_fidelity"]["ssim"] = 0.98
    _write_results(tmp_path, results)

    completed = _run(tmp_path)

    assert completed.returncode == 0, completed.stderr
