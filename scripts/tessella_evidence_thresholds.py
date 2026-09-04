"""Literal quantitative thresholds and cross-invariants for TESSELLA evidence."""

from __future__ import annotations

import math
from typing import Any


THRESHOLDS = {
    "vt_out_of_core": {
        "width": 3840,
        "height": 2160,
        "logical_texel_bytes_min": 256 * 1024**3,
        "settling_frames_max": 8,
        "fallback_texels_max": 0,
        "peak_host_visible_bytes_exclusive_max": 512 * 1024**2,
    },
    "hzb_occlusion": {
        "cull_percent_min": 60.0,
        "speedup_min": 1.8,
        "bitwise_identical_required": True,
    },
    "visibility_buffer": {
        "feedback_records_equal_visible_pixels": True,
        "material_invocations_equal_visible_pixels": True,
        "picking_samples": 10_000,
        "gpu_cpu_picking_match_percent": 100.0,
        "fallback_texels_max": 0,
    },
    "bc7_fidelity": {"ssim_min": 0.98, "mean_delta_e_2000_exclusive_max": 1.5},
    "bc5_fidelity": {
        "mean_angular_error_degrees_exclusive_max": 1.0,
        "max_angular_error_degrees_exclusive_max": 4.0,
    },
    "flythrough_popping": {
        "frames": 600,
        "rendered_frames_total": 600,
        "max_delta_e_2000_exclusive_max": 1.0,
        "crack_count_max": 0,
        "actual_clipmap_center_transitions_min": 540,
        "distinct_clipmap_centers_min": 480,
        "regions_on_screen_min": 3,
    },
    "vt_request_retention": {
        "feedback_not_ready_frames": 30,
        "convergence_frames_max": 8,
        "retained_requests_after_convergence": 0,
    },
}


def _close(actual: float, expected: float) -> bool:
    # Runtime VT statistics cross the PyO3 boundary as f32 values, so exact
    # recomputation legitimately carries single-precision rounding.
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12)


def threshold_errors(gate: str, d: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def need(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{gate}: {message}")

    if gate == "vt_out_of_core":
        need(
            d["width"] == 3840 and d["height"] == 2160,
            "render size must equal 3840x2160",
        )
        need(
            d["logical_texel_bytes"] >= 256 * 1024**3,
            "logical_texel_bytes must be >= 256 GiB",
        )
        need(0 <= d["settling_frames"] <= 8, "settling_frames must be between 0 and 8")
        need(d["fallback_texels"] == 0, "fallback_texels must equal 0")
        need(
            0 < d["peak_host_visible_bytes"] < 512 * 1024**2,
            "peak_host_visible_bytes must be > 0 and < 512 MiB",
        )
        need(d["atlas_device_local_bytes"] > 0, "atlas_device_local_bytes must be > 0")
        need(
            d["atlas_uncompressed_equivalent_bytes"] >= d["atlas_device_local_bytes"],
            "atlas uncompressed bytes must be >= device-local bytes",
        )
        if d["atlas_device_local_bytes"] > 0:
            expected = (
                d["atlas_uncompressed_equivalent_bytes"] / d["atlas_device_local_bytes"]
            )
            need(
                _close(d["atlas_compression_ratio"], expected),
                "atlas_compression_ratio is inconsistent with atlas footprints",
            )
        family_ratios = {"albedo": 4.0, "normal": 2.0, "mask": 4.0}
        device_local_total = 0
        uncompressed_total = 0
        for family, required_ratio in family_ratios.items():
            device_local = d[f"atlas_device_local_bytes_{family}"]
            uncompressed = d[f"atlas_uncompressed_equivalent_bytes_{family}"]
            ratio = d[f"atlas_compression_ratio_{family}"]
            need(device_local > 0, f"{family} atlas device-local bytes must be > 0")
            need(
                uncompressed > device_local,
                f"{family} atlas uncompressed bytes must exceed device-local bytes",
            )
            if device_local > 0:
                need(
                    _close(ratio, uncompressed / device_local),
                    f"{family} atlas compression ratio is inconsistent",
                )
            need(
                _close(ratio, required_ratio),
                f"{family} atlas compression ratio must equal {required_ratio:g}",
            )
            device_local_total += device_local
            uncompressed_total += uncompressed
        need(
            device_local_total == d["atlas_device_local_bytes"],
            "per-family device-local bytes must sum to aggregate atlas bytes",
        )
        need(
            uncompressed_total == d["atlas_uncompressed_equivalent_bytes"],
            "per-family uncompressed bytes must sum to aggregate atlas bytes",
        )
    elif gate == "hzb_occlusion":
        need(d["cull_percent"] >= 60.0, "cull_percent must be >= 60.0")
        need(d["bitwise_identical"] is True, "bitwise_identical must be true")
        need(d["timestamp_query"] is True, "timestamp_query must be true")
        need(d["speedup_target"] == 1.8, "speedup_target must retain the literal 1.8")
        need(d["speedup_gate"] == 1.7, "speedup_gate must retain the 1.7 floor")
        need(
            d["baseline_gpu_ms"] > 0 and d["culled_gpu_ms"] > 0,
            "GPU timings must be > 0",
        )
        if d["baseline_gpu_ms"] > 0 and d["culled_gpu_ms"] > 0:
            measured = d["baseline_gpu_ms"] / d["culled_gpu_ms"]
            need(
                _close(d["speedup"], measured),
                "speedup is inconsistent with baseline_gpu_ms / culled_gpu_ms",
            )
            need(
                measured >= d["speedup_gate"],
                "recomputed speedup must clear the recorded regression floor",
            )
        need(
            d["phase1_drawn"] + d["phase2_recovered"] == d["final_drawn"],
            "phase1_drawn + phase2_recovered must equal final_drawn",
        )
        need(d["frustum_passing"] > 0, "frustum_passing must be > 0")
        need(
            d["phase1_drawn"] + d["phase1_rejected"] == d["frustum_passing"],
            "phase1_drawn + phase1_rejected must equal frustum_passing",
        )
        need(
            d["phase2_recovered"] <= d["phase1_rejected"],
            "phase2_recovered must be <= phase1_rejected",
        )
        if d["frustum_passing"] > 0:
            measured_cull_percent = (
                100.0 * (d["frustum_passing"] - d["final_drawn"]) / d["frustum_passing"]
            )
            need(
                _close(d["cull_percent"], measured_cull_percent),
                "cull_percent is inconsistent with indirect draw counts",
            )
    elif gate == "hzb_history_recovery":
        need(d["phase1_rejected"] > 0, "phase1_rejected must be > 0")
        need(
            0 < d["phase2_recovered"] <= d["phase1_rejected"],
            "phase2_recovered must be > 0 and <= phase1_rejected",
        )
        need(d["bitwise_identical"] is True, "bitwise_identical must be true")
    elif gate == "visibility_buffer":
        need(d["visible_pixels"] > 0, "visible_pixels must be > 0")
        need(
            d["visibility_feedback_records"] == d["visible_pixels"],
            "visibility_feedback_records must equal visible_pixels",
        )
        need(
            d["material_invocations"] == d["visible_pixels"],
            "material_invocations must equal visible_pixels",
        )
        need(
            d["forward_feedback_records"] == d["forward_material_invocations"],
            "forward feedback and material invocations must match",
        )
        need(
            d["forward_material_invocations"] >= d["visible_pixels"],
            "forward_material_invocations must be >= visible_pixels",
        )
        need(d["fallback_texels"] == 0, "fallback_texels must equal 0")
        need(d["picking_samples"] == 10_000, "picking_samples must equal 10000")
        need(
            d["gpu_picking_repeat_matches"] == d["picking_samples"],
            "repeated GPU picking must match every sample",
        )
        need(d["gpu_cpu_picking_compared"] > 0, "GPU/CPU compared set must be nonempty")
        need(
            d["gpu_cpu_picking_compared"] + d["gpu_cpu_picking_excluded"]
            == d["picking_samples"],
            "GPU/CPU compared and excluded counts must cover every sample",
        )
        need(
            d["gpu_cpu_picking_matches"] == d["gpu_cpu_picking_compared"],
            "GPU/CPU picking must match every unambiguous visible sample",
        )
        need(
            d["bitwise_identical_to_forward"] is True,
            "bitwise_identical_to_forward must be true",
        )
    elif gate in {"bc7_fidelity", "bc5_fidelity"}:
        family = "albedo" if gate == "bc7_fidelity" else "normal"
        need(d["texture_family"] == family, f"texture_family must equal '{family}'")
        need(
            d["source_bytes"] > 0 and d["encoded_bytes"] > 0,
            "source_bytes and encoded_bytes must be > 0",
        )
        if d["source_bytes"] > 0 and d["encoded_bytes"] > 0:
            need(
                _close(d["compression_ratio"], d["source_bytes"] / d["encoded_bytes"]),
                "compression_ratio is inconsistent with source_bytes / encoded_bytes",
            )
        if gate == "bc7_fidelity":
            need(
                d["ssim_gate"] == 0.98,
                "ssim_gate must retain the literal 0.98 threshold",
            )
            need(
                d["mean_delta_e_gate"] == 1.5,
                "mean_delta_e_gate must retain the literal 1.5 threshold",
            )
            need(d["ssim"] >= 0.98, "ssim must be >= 0.98")
            need(d["mean_delta_e_2000"] < 1.5, "mean_delta_e_2000 must be < 1.5")
        else:
            need(
                d["mean_angle_gate_degrees"] == 1.0,
                "mean_angle_gate_degrees must retain the literal 1.0 threshold",
            )
            need(
                d["max_angle_gate_degrees"] == 4.0,
                "max_angle_gate_degrees must retain the literal 4.0 threshold",
            )
            need(
                d["mean_angular_error_degrees"] < 1.0,
                "mean_angular_error_degrees must be < 1.0",
            )
            need(
                d["max_angular_error_degrees"] < 4.0,
                "max_angular_error_degrees must be < 4.0",
            )
    elif gate == "flythrough_popping":
        need(
            d["frames"] == 600
            and d["rendered_frames_total"] == 600
            and d["frames_crack_checked"] == 600,
            "frames, rendered_frames_total, and frames_crack_checked must equal 600",
        )
        need(d["width"] > 0 and d["height"] > 0, "render dimensions must be > 0")
        need(d["depth_sample_count"] > 0, "depth_sample_count must be > 0")
        need(
            d["worst_frame_crack_count"] == 0 and d["crack_count"] == 0,
            "all crack counts must equal 0",
        )
        need(d["max_delta_e_2000"] < 1.0, "max_delta_e_2000 must be < 1.0")
        need(d["camera_step_px"] > 0, "camera_step_px must be > 0")
        need(
            d["camera_path_distance_m"] > 0,
            "camera_path_distance_m must be > 0",
        )
        need(
            d["distinct_camera_positions"] == 600,
            "distinct_camera_positions must equal 600",
        )
        need(
            540 <= d["actual_clipmap_center_transitions"] < d["frames"],
            "actual_clipmap_center_transitions must be between 540 and 599",
        )
        need(
            d["clipmap_center_path_m"] > 0,
            "clipmap_center_path_m must be > 0",
        )
        need(
            _close(
                d["clipmap_center_step_m"],
                d["clipmap_center_path_m"] / (d["frames"] - 1),
            ),
            "clipmap_center_step_m must equal actual path divided by 599 transitions",
        )
        need(
            480 <= d["distinct_clipmap_centers"] <= d["frames"],
            "distinct_clipmap_centers must be between 480 and 600",
        )
        need(d["regions_on_screen"] >= 3, "regions_on_screen must be >= 3")
    elif gate == "vt_request_retention":
        need(
            d["feedback_not_ready_frames"] == 30,
            "feedback_not_ready_frames must equal 30",
        )
        need(
            d["convergence_budget_frames"] == 8,
            "convergence_budget_frames must equal 8",
        )
        need(
            0 <= d["convergence_frames"] <= 8,
            "convergence_frames must be between 0 and 8",
        )
        need(d["retained_set_size"] > 0, "retained_set_size must be > 0")
        need(
            d["retained_set_identical_every_not_ready_frame"] is True,
            "retained set must be identical for every not-ready frame",
        )
        need(
            d["retained_requests_after_convergence"] == 0,
            "retained_requests_after_convergence must equal 0",
        )
    elif gate == "capability_degradations":
        need(
            d["degradation_count"] == len(d["degradations"]),
            "degradation_count must equal len(degradations)",
        )
        need(
            set(d["tessella_capabilities_degraded"]).issubset(d["degradations"]),
            "TESSELLA degradation list must be a subset of degradations",
        )
    return errors
