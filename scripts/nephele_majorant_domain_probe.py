#!/usr/bin/env python3
"""Execute the fixed million-point exact-domain NEPHELE majorant probe."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROBES = 1_000_000
DENSITY_SAMPLING = "normalized-clamp-to-edge-linear-texel-center-uN-minus-0.5"
MAJORANT_QUERY = "floor(clamp(unit,0,1)*N)-clamped-to-N-minus-1"
MAJORANT_CONSTRUCTION = (
    "3x3x3-clamped-neighborhood-trilinear-outward-then-f32-extinction-outward"
)
PROBE_MAPPING = "domain-boundaries-texel-extrema-majorant-boundaries-then-irrational-lattice-v3"


def domain_coverage_sha256(record: dict[str, Any]) -> str:
    encoded = json.dumps(
        {key: value for key, value in record.items() if key != "sha256"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def represented_density(
    raw: list[int], shape: tuple[int, int, int]
) -> tuple[np.ndarray, str]:
    if len(raw) != math.prod(shape):
        raise ValueError("density payload does not match grid shape")
    decoded = (np.asarray(raw, dtype=np.float32) / np.float32(65535.0)).astype(np.float16)
    bits = decoded.astype("<f2", copy=False).tobytes(order="C")
    return decoded.astype(np.float32).reshape((shape[2], shape[1], shape[0])), hashlib.sha256(
        bits
    ).hexdigest()


def _next_up(value: np.float32) -> np.float32:
    if value == np.float32(0.0) or not np.isfinite(value):
        return value
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return np.asarray(bits + np.uint32(1), dtype=np.uint32).view(np.float32)[()]


def _outward_mul(a: np.float32, b: np.float32) -> np.float32:
    exact = float(a) * float(b)
    rounded = np.float32(a * b)
    if float(rounded) < exact:
        rounded = _next_up(rounded)
    return rounded


def _trilinear_upper(maximum: np.float32) -> np.float32:
    if maximum == np.float32(0.0):
        return maximum
    operations = 13.0
    unit_roundoff = 1.0 / float(1 << 24)
    half_min_subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0))) * 0.5
    denominator = 1.0 - operations * unit_roundoff
    gamma = operations * unit_roundoff / denominator
    exact = float(maximum) * (1.0 + gamma) + operations * half_min_subnormal / denominator
    rounded = np.float32(exact)
    if float(rounded) < exact:
        rounded = _next_up(rounded)
    return rounded


def canonical_majorant_cells(
    represented: np.ndarray, density_scale: float, sigma_t_max: float
) -> list[float]:
    """Independently reproduce the canonical Rust N^3 extinction-majorant grid."""
    nz, ny, nx = represented.shape
    scale_f32 = np.float32(density_scale)
    sigma_f32 = np.float32(sigma_t_max)
    result: list[float] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                maximum = np.float32(0.0)
                for dz, dy, dx in itertools.product(range(-1, 2), repeat=3):
                    node = represented[
                        min(max(z + dz, 0), nz - 1),
                        min(max(y + dy, 0), ny - 1),
                        min(max(x + dx, 0), nx - 1),
                    ]
                    maximum = np.maximum(maximum, np.float32(node))
                physical = _outward_mul(_trilinear_upper(maximum), scale_f32)
                result.append(float(_outward_mul(physical, sigma_f32)))
    return result


def _axis_indices(unit: float, size: int) -> tuple[int, int, np.float32]:
    unit_f32 = np.float32(min(max(np.float32(unit), np.float32(0.0)), np.float32(1.0)))
    coordinate = np.float32(unit_f32 * np.float32(size) - np.float32(0.5))
    floor = math.floor(float(coordinate))
    lower = min(max(floor, 0), size - 1)
    upper = min(max(floor + 1, 0), size - 1)
    fraction = np.float32(coordinate - np.float32(math.floor(float(coordinate))))
    return lower, upper, fraction


def _sample(
    grid: np.ndarray, shape: tuple[int, int, int], point: tuple[float, float, float]
) -> np.float32:
    axes = [_axis_indices(coordinate, size) for coordinate, size in zip(point, shape)]
    value = np.float32(0.0)
    one = np.float32(1.0)
    for z in range(2):
        for y in range(2):
            for x in range(2):
                ix = axes[0][x]
                iy = axes[1][y]
                iz = axes[2][z]
                wx = np.float32(one - axes[0][2]) if x == 0 else axes[0][2]
                wy = np.float32(one - axes[1][2]) if y == 0 else axes[1][2]
                wz = np.float32(one - axes[2][2]) if z == 0 else axes[2][2]
                weight = np.float32(np.float32(wx * wy) * wz)
                value = np.float32(value + np.float32(np.float32(grid[iz, iy, ix]) * weight))
    return value


def _majorant_cell(shape: tuple[int, int, int], point: tuple[float, float, float]) -> int:
    cell = []
    for coordinate, size in zip(point, shape):
        unit = np.float32(min(max(np.float32(coordinate), np.float32(0.0)), np.float32(1.0)))
        cell.append(min(math.floor(float(np.float32(unit * np.float32(size)))), size - 1))
    return (cell[2] * shape[1] + cell[1]) * shape[0] + cell[0]


def _coverage_prefix(shape: tuple[int, int, int]) -> tuple[list[tuple[float, float, float]], dict[str, int | bool]]:
    domain_corners = list(itertools.product((0.0, 1.0), repeat=3))
    centers = [
        ((x + 0.5) / shape[0], (y + 0.5) / shape[1], (z + 0.5) / shape[2])
        for z in range(shape[2])
        for y in range(shape[1])
        for x in range(shape[0])
    ]
    boundary_sides: list[tuple[float, float, float]] = []
    domain_faces: list[tuple[float, float, float]] = []
    for axis in range(3):
        transverse = [dimension for dimension in range(3) if dimension != axis]
        for first in range(shape[transverse[0]]):
            for second in range(shape[transverse[1]]):
                for side in (0.0, 1.0):
                    point = [0.0, 0.0, 0.0]
                    point[axis] = side
                    point[transverse[0]] = (first + 0.5) / shape[transverse[0]]
                    point[transverse[1]] = (second + 0.5) / shape[transverse[1]]
                    domain_faces.append(tuple(point))
        for boundary_index in range(1, shape[axis]):
            boundary = np.float32(boundary_index / shape[axis])
            sides = (
                float(np.nextafter(boundary, np.float32(0.0))),
                float(np.nextafter(boundary, np.float32(1.0))),
            )
            for first in range(shape[transverse[0]]):
                for second in range(shape[transverse[1]]):
                    base = [0.0, 0.0, 0.0]
                    base[transverse[0]] = (first + 0.5) / shape[transverse[0]]
                    base[transverse[1]] = (second + 0.5) / shape[transverse[1]]
                    for side in sides:
                        point = base.copy()
                        point[axis] = side
                        boundary_sides.append(tuple(point))
    counts: dict[str, int | bool] = {
        "domain_boundary_points": len(domain_corners),
        "domain_face_interiors": len(domain_faces),
        "texel_center_extrema": len(centers),
        "majorant_cell_centers": len(centers),
        "texel_centers_are_majorant_cell_centers": True,
        "majorant_boundary_sides": len(boundary_sides),
    }
    return domain_corners + domain_faces + centers + boundary_sides, counts


def produce(
    medium_path: Path, source_revision: str, raw_path: Path, *, probes: int = PROBES
) -> dict[str, Any]:
    medium = json.loads(medium_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema",
        "domain",
        "density_r16",
        "density_transport",
        "majorant_cells",
        "majorant_transport",
        "density_scale",
        "sigma_a",
        "sigma_s",
        "phase",
        "transport",
    }
    domain = medium.get("domain", {})
    shape = tuple(domain.get("grid_shape", ()))
    bounds_min, bounds_max = domain.get("bounds_min"), domain.get("bounds_max")
    raw = medium.get("density_r16")
    sigma_a, sigma_s = medium.get("sigma_a"), medium.get("sigma_s")
    density_scale = medium.get("density_scale")
    valid_coefficients = all(
        isinstance(values, list)
        and len(values) == 3
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
            for value in values
        )
        for values in (sigma_a, sigma_s)
    )
    if (
        set(medium) != expected_keys
        or medium.get("schema") != "forge3d.nephele.heterogeneous_medium/2"
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
        or len(shape) != 3
        or any(type(size) is not int or size < 2 for size in shape)
        or not isinstance(raw, list)
        or len(raw) != math.prod(shape)
        or any(type(value) is not int or not 0 <= value <= 65535 for value in raw)
        or not isinstance(bounds_min, list)
        or not isinstance(bounds_max, list)
        or len(bounds_min) != 3
        or len(bounds_max) != 3
        or any(
            not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
            for lower, upper in zip(bounds_min, bounds_max)
        )
        or not valid_coefficients
        or isinstance(density_scale, bool)
        or not isinstance(density_scale, (int, float))
        or not math.isfinite(density_scale)
        or density_scale <= 0
        or probes != PROBES
    ):
        raise ValueError("invalid exact transport domain")

    represented, f16_sha256 = represented_density(raw, shape)
    density_transport = medium.get("density_transport")
    expected_density_transport = {
        "schema": "forge3d.nephele.density_transport/1",
        "decode": "unorm16-div-65535-as-f32-then-ieee-f16-rne",
        "storage": "ieee-f16-bits-little-endian",
        "sampling": DENSITY_SAMPLING,
        "f16_sha256": f16_sha256,
    }
    expected_majorant_transport = {
        "schema": "forge3d.nephele.majorant_transport/1",
        "grid_shape": list(shape),
        "query": MAJORANT_QUERY,
        "construction": MAJORANT_CONSTRUCTION,
    }
    sigma_t_spectrum = [float(a) + float(s) for a, s in zip(sigma_a, sigma_s)]
    extinction_channel = max(range(3), key=sigma_t_spectrum.__getitem__)
    sigma_t_max = sigma_t_spectrum[extinction_channel]
    expected_transport = {
        "sigma_t_spectrum": sigma_t_spectrum,
        "sigma_t_max_channel": sigma_t_max,
        "extinction_channel": extinction_channel,
        "slab_axis": 2,
    }
    runtime_sigma_t_max = max(
        float(np.float32(a) + np.float32(s)) for a, s in zip(sigma_a, sigma_s)
    )
    expected_majorants = canonical_majorant_cells(
        represented, density_scale, runtime_sigma_t_max
    )
    if (
        density_transport != expected_density_transport
        or medium.get("majorant_transport") != expected_majorant_transport
        or medium.get("transport") != expected_transport
        or medium.get("majorant_cells") != expected_majorants
        or sigma_t_max <= 0
    ):
        raise ValueError("transport representation or canonical majorant metadata mismatch")

    prefix, coverage_counts = _coverage_prefix(shape)
    if len(prefix) > probes:
        raise ValueError("domain boundary/extrema prefix exceeds fixed probe count")
    sigma_f32 = np.float32(runtime_sigma_t_max)
    scale_f32 = np.float32(density_scale)
    majorants = [np.float32(value) for value in expected_majorants]
    violations = 0
    maximum = -math.inf
    measured = 0.0
    extinction_sum = bound_sum = 0.0
    extinction_min = bound_minimum = math.inf
    extinction_max = bound_maximum = -math.inf
    with raw_path.open("wb") as raw_stream:
        for index in range(probes):
            point = (
                prefix[index]
                if index < len(prefix)
                else (
                    (index * 0.7548776662466927) % 1.0,
                    (index * 0.5698402909980532) % 1.0,
                    (index * 0.4385790210124317) % 1.0,
                )
            )
            density = _sample(represented, shape, point)
            physical = np.float32(density * scale_f32)
            extinction = np.float32(physical * sigma_f32)
            bound = majorants[_majorant_cell(shape, point)]
            measured = max(measured, abs(float(extinction) - float(np.float32(extinction))))
            excess = float(extinction) - float(bound)
            maximum = max(maximum, excess)
            violations += extinction > bound
            raw_stream.write(struct.pack("<ff", float(extinction), float(bound)))
            extinction_sum += float(extinction)
            bound_sum += float(bound)
            extinction_min = min(extinction_min, float(extinction))
            extinction_max = max(extinction_max, float(extinction))
            bound_minimum = min(bound_minimum, float(bound))
            bound_maximum = max(bound_maximum, float(bound))

    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    medium_hash = hashlib.sha256(medium_path.read_bytes()).hexdigest()
    coverage: dict[str, Any] = {
        "schema": "forge3d.nephele.majorant_domain_coverage/3",
        "medium_sha256": medium_hash,
        "density_f16_sha256": f16_sha256,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "grid_shape": list(shape),
        "majorant_grid_shape": list(shape),
        "exact_domain": True,
        **coverage_counts,
        "majorant_query": MAJORANT_QUERY,
        "probe_mapping": PROBE_MAPPING,
        "sha256": "",
    }
    coverage["sha256"] = domain_coverage_sha256(coverage)
    tool = Path(__file__).resolve()
    return {
        "schema": "forge3d.nephele.majorant_probe/3",
        "probe_count": probes,
        "violation_count": violations,
        "max_represented_extinction_minus_bound": maximum,
        "measured_representation_error": measured,
        "source_revision": source_revision,
        "producer_tool": {
            "path": "scripts/nephele_majorant_domain_probe.py",
            "sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        },
        "medium": {"path": medium_path.as_posix(), "sha256": medium_hash},
        "transport_representation": {
            "density_f16_sha256": f16_sha256,
            "grid_shape": list(shape),
            "majorant_grid_shape": list(shape),
            "density_sampling": DENSITY_SAMPLING,
            "majorant_query": MAJORANT_QUERY,
            "sigma_t_max": sigma_t_max,
        },
        "sample_mapping": {"algorithm": PROBE_MAPPING, "count": probes},
        "domain": coverage,
        "raw_output": {
            "path": raw_path.name,
            "sha256": raw_hash,
            "encoding": "little-endian-f32-extinction-bound-pairs",
            "pairs": probes,
        },
        "summary": {
            "extinction": {
                "minimum": extinction_min,
                "maximum": extinction_max,
                "sum": extinction_sum,
            },
            "bound": {
                "minimum": bound_minimum,
                "maximum": bound_maximum,
                "sum": bound_sum,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("medium", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args(argv)
    try:
        record = produce(
            args.medium, args.source_revision, args.output.with_suffix(".pairs.bin")
        )
        if record["probe_count"] != PROBES or record["violation_count"]:
            raise ValueError("majorant probe failed")
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"NEPHELE majorant probe failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
