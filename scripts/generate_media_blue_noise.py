#!/usr/bin/env python3
"""Generate and verify NEPHELE's deterministic void-and-cluster rank tile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TILE = ROOT / "assets/media/nephele_blue_noise_8x8.txt"
PROVENANCE = ROOT / "assets/media/nephele_blue_noise_provenance.json"
SIZE = 8
SIGMA = 1.5
SEED = 0x4E455048454C45


def _tie_break(point: tuple[int, int]) -> int:
    value = (SEED + point[0] + SIZE * point[1]) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _distance_squared(a: tuple[int, int], b: tuple[int, int]) -> int:
    dx = min(abs(a[0] - b[0]), SIZE - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), SIZE - abs(a[1] - b[1]))
    return dx * dx + dy * dy


def _energy(point: tuple[int, int], occupied: set[tuple[int, int]]) -> float:
    return sum(
        math.exp(-_distance_squared(point, other) / (2.0 * SIGMA * SIGMA))
        for other in occupied
        if other != point
    )


def generate() -> list[int]:
    """Return a toroidal void-and-cluster permutation of ``range(64)``."""
    points = [(x, y) for y in range(SIZE) for x in range(SIZE)]
    occupied: set[tuple[int, int]] = {(0, 0)}
    while len(occupied) < len(points) // 2:
        candidates = [point for point in points if point not in occupied]
        occupied.add(min(candidates, key=lambda point: (_energy(point, occupied), _tie_break(point))))

    ranks: dict[tuple[int, int], int] = {}
    lower = set(occupied)
    for rank in range(len(points) // 2 - 1, -1, -1):
        cluster = max(lower, key=lambda point: (_energy(point, lower), _tie_break(point)))
        ranks[cluster] = rank
        lower.remove(cluster)

    upper = set(occupied)
    for rank in range(len(points) // 2, len(points)):
        candidates = [point for point in points if point not in upper]
        void = min(candidates, key=lambda point: (_energy(point, upper), _tie_break(point)))
        ranks[void] = rank
        upper.add(void)
    return [ranks[point] for point in points]


def tile_bytes(ranks: list[int]) -> bytes:
    rows = [ranks[index : index + SIZE] for index in range(0, len(ranks), SIZE)]
    body = "\n".join(" ".join(str(value) for value in row) for row in rows)
    return (
        "# NEPHELE deterministic 8x8 void-and-cluster blue-noise rank tile.\n"
        "# Row-major permutation 0..63; generator and provenance are tracked.\n"
        f"{body}\n"
    ).encode("ascii")


def provenance_bytes(data: bytes) -> bytes:
    record = {
        "schema": "forge3d.nephele.blue_noise_provenance/1",
        "algorithm": "deterministic-toroidal-void-and-cluster-rank-v1",
        "dimensions": [SIZE, SIZE],
        "generator": "scripts/generate_media_blue_noise.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "license": "MIT",
        "license_file": "LICENSE",
        "parameters": {
            "initial_pattern": "deterministic-lowest-energy-best-candidate-half-fill",
            "periodic_boundary": True,
            "ranking": "cluster-removal-then-void-insertion",
            "sigma": SIGMA,
            "tie_break": "splitmix64-point-hash",
            "tie_break_seed_u64": SEED,
        },
        "source": "repository-generated; no third-party tile bytes",
        "tile": TILE.relative_to(ROOT).as_posix(),
        "tile_sha256": hashlib.sha256(data).hexdigest(),
    }
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = tile_bytes(generate())
    provenance = provenance_bytes(data)
    if args.write:
        TILE.write_bytes(data)
        PROVENANCE.write_bytes(provenance)
        return 0
    if TILE.read_bytes() != data or PROVENANCE.read_bytes() != provenance:
        raise SystemExit("tracked blue-noise tile or provenance differs from the generator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
