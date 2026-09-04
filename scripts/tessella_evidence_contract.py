"""Canonical nine-file TESSELLA evidence schema and hard thresholds."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .tessella_evidence_thresholds import THRESHOLDS, threshold_errors
else:
    from tessella_evidence_thresholds import THRESHOLDS, threshold_errors


CORE_GATES = (
    "vt_out_of_core",
    "hzb_occlusion",
    "hzb_history_recovery",
    "visibility_buffer",
    "bc7_fidelity",
    "bc5_fidelity",
    "flythrough_popping",
    "vt_request_retention",
    "capability_degradations",
)


@dataclass(frozen=True)
class GateSpec:
    numeric: tuple[str, ...] = ()
    booleans: tuple[str, ...] = ()
    strings: tuple[str, ...] = ()
    string_lists: tuple[str, ...] = ()


SPECS = {
    "vt_out_of_core": GateSpec(
        numeric=(
            "width",
            "height",
            "logical_texel_bytes",
            "settling_frames",
            "fallback_texels",
            "peak_host_visible_bytes",
            "atlas_device_local_bytes",
            "atlas_uncompressed_equivalent_bytes",
            "atlas_compression_ratio",
            "atlas_device_local_bytes_albedo",
            "atlas_device_local_bytes_normal",
            "atlas_device_local_bytes_mask",
            "atlas_uncompressed_equivalent_bytes_albedo",
            "atlas_uncompressed_equivalent_bytes_normal",
            "atlas_uncompressed_equivalent_bytes_mask",
            "atlas_compression_ratio_albedo",
            "atlas_compression_ratio_normal",
            "atlas_compression_ratio_mask",
        )
    ),
    "hzb_occlusion": GateSpec(
        numeric=(
            "cull_percent",
            "frustum_passing",
            "phase1_drawn",
            "phase1_rejected",
            "phase2_recovered",
            "final_drawn",
            "baseline_gpu_ms",
            "culled_gpu_ms",
            "speedup",
            "speedup_target",
            "speedup_gate",
        ),
        booleans=("timestamp_query", "bitwise_identical"),
    ),
    "hzb_history_recovery": GateSpec(
        numeric=("phase1_rejected", "phase2_recovered"),
        booleans=("bitwise_identical",),
    ),
    "visibility_buffer": GateSpec(
        numeric=(
            "visible_pixels",
            "background_pixels",
            "visibility_feedback_records",
            "forward_feedback_records",
            "material_invocations",
            "forward_material_invocations",
            "measured_overdraw_factor",
            "fallback_texels",
            "picking_samples",
            "picking_hits",
            "gpu_picking_repeat_matches",
            "gpu_cpu_picking_compared",
            "gpu_cpu_picking_excluded",
            "gpu_cpu_picking_matches",
        ),
        booleans=("bitwise_identical_to_forward",),
    ),
    "bc7_fidelity": GateSpec(
        numeric=(
            "source_bytes",
            "encoded_bytes",
            "compression_ratio",
            "ssim",
            "mean_delta_e_2000",
            "ssim_gate",
            "mean_delta_e_gate",
        ),
        strings=("texture_family", "fixture"),
    ),
    "bc5_fidelity": GateSpec(
        numeric=(
            "source_bytes",
            "encoded_bytes",
            "compression_ratio",
            "mean_angular_error_degrees",
            "max_angular_error_degrees",
            "mean_angle_gate_degrees",
            "max_angle_gate_degrees",
        ),
        strings=("texture_family", "fixture"),
    ),
    "flythrough_popping": GateSpec(
        numeric=(
            "frames",
            "rendered_frames_total",
            "width",
            "height",
            "worst_frame_crack_count",
            "crack_count",
            "depth_sample_count",
            "frames_crack_checked",
            "max_delta_e_2000",
            "camera_step_px",
            "camera_path_distance_m",
            "distinct_camera_positions",
            "clipmap_center_step_m",
            "clipmap_center_path_m",
            "actual_clipmap_center_transitions",
            "distinct_clipmap_centers",
            "regions_on_screen",
        )
    ),
    "vt_request_retention": GateSpec(
        numeric=(
            "feedback_not_ready_frames",
            "convergence_budget_frames",
            "convergence_frames",
            "retained_set_size",
            "retained_requests_after_convergence",
            "tiles_streamed",
        ),
        booleans=("retained_set_identical_every_not_ready_frame",),
    ),
    "capability_degradations": GateSpec(
        numeric=("degradation_count",),
        strings=("adapter", "backend"),
        string_lists=("degradations", "tessella_capabilities_degraded"),
    ),
}


def non_finite_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        return [
            nested
            for key in sorted(value, key=str)
            for nested in non_finite_paths(value[key], f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            nested
            for index, item in enumerate(value)
            for nested in non_finite_paths(item, f"{path}[{index}]")
        ]
    return []


def _field_errors(gate: str, data: dict[str, Any]) -> list[str]:
    spec = SPECS[gate]
    errors: list[str] = []
    all_fields = (*spec.numeric, *spec.booleans, *spec.strings, *spec.string_lists)
    for field in all_fields:
        if field not in data:
            errors.append(f"{gate}: missing required field '{field}'")
    for field in spec.numeric:
        if field in data and (
            isinstance(data[field], bool) or not isinstance(data[field], (int, float))
        ):
            errors.append(
                f"{gate}.{field}: expected numeric value (boolean is not numeric evidence)"
            )
    for field in spec.booleans:
        if field in data and not isinstance(data[field], bool):
            errors.append(f"{gate}.{field}: expected boolean value")
    for field in spec.strings:
        if field in data and (not isinstance(data[field], str) or not data[field]):
            errors.append(f"{gate}.{field}: expected non-empty string")
    for field in spec.string_lists:
        if field in data and (
            not isinstance(data[field], list)
            or any(not isinstance(item, str) or not item for item in data[field])
        ):
            errors.append(f"{gate}.{field}: expected list of non-empty strings")
    errors.extend(
        f"{gate}: non-finite numeric value at {path}" for path in non_finite_paths(data)
    )
    return errors


def load_gate(path: Path, gate: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, [f"missing core evidence file: {path.name} (gate '{gate}')"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [f"{path.name}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{path.name}: JSON root must be an object"]
    if data.get("gate") != gate:
        return data, [
            f"{path.name}: gate identity mismatch: expected {gate!r}, got {data.get('gate')!r}"
        ]
    errors = _field_errors(gate, data)
    if not errors:
        errors.extend(threshold_errors(gate, data))
    return data, errors
