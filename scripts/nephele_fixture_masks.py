#!/usr/bin/env python3
"""Regenerate NEPHELE masks from deterministic reference classifications."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MASK_FILES = {
    "sky_cloud_mask": "sky-cloud-mask.npy",
    "godray_roi_mask": "godray-roi-mask.npy",
    "terrain_mask": "terrain-mask.npy",
    "cloud_shadow_mask": "cloud-shadow-mask.npy",
    "shaft_mask": "shaft-mask.npy",
}
EXPECTED_RULES = {
    "terrain_mask": "terrain_hit",
    "sky_cloud_mask": "not terrain_hit and min(reference transmittance) < 1",
    "cloud_shadow_mask": "terrain_hit and min(reference cloud shadow) < 1",
    "shaft_mask": "terrain_hit and reference midpoint sun visibility is terrain_blocked",
    "godray_roi_mask": "bounding rectangle of reference midpoint terrain_blocked lighting",
}
TECHNICAL_CONTRACTS = {
    "godray_roi_minimum_shape": {
        "source": "tests/_ssim.py",
        "parameter": "ssim.win_size",
        "value": 11,
    },
}


def _load(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.hasobject or not np.isfinite(value).all():
        raise ValueError(f"{path.name}: expected a finite non-object array")
    if shape is not None and value.shape[:2] != shape:
        raise ValueError(f"{path.name}: reference shapes differ")
    return value


def _save_mask(path: Path, mask: np.ndarray, rule: str) -> None:
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise ValueError(f"{path.name}: {rule} generated an empty or invalid semantic mask")
    if mask.all():
        raise ValueError(f"{path.name}: {rule} generated a full semantic mask")
    with path.open("wb") as stream:
        np.save(stream, mask, allow_pickle=False)


def generate_masks(fixture_dir: Path, rules_path: Path, output_dir: Path) -> None:
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    if rules != {
        "schema": "forge3d.nephele.mask_rules/3",
        "rules": EXPECTED_RULES,
        "technical_contracts": TECHNICAL_CONTRACTS,
    }:
        raise ValueError("mask rules must state the reviewed reference-only equations exactly")

    terrain_hit = _load(fixture_dir / "reference-terrain-hit.npy")
    if terrain_hit.ndim != 2 or terrain_hit.dtype != np.bool_:
        raise ValueError("reference-terrain-hit.npy must be an exact boolean plane")
    shape = terrain_hit.shape
    transmittance = _load(fixture_dir / "reference-transmittance.npy", shape)
    cloud_shadow = _load(fixture_dir / "reference-cloud-shadow-aov.npy", shape)
    lighting = _load(fixture_dir / "reference-media-lighting-visibility.npy", shape)
    terrain_slice = _load(fixture_dir / "reference-terrain-slice.npy", shape)
    if transmittance.shape != (*shape, 3) or cloud_shadow.shape != (*shape, 3):
        raise ValueError("reference transmittance and cloud shadow must be RGB AOVs")
    if lighting.shape != shape or lighting.dtype != np.uint8 or not np.isin(lighting, [0, 1, 2]).all():
        raise ValueError("reference media-lighting classification codes must be uint8 0, 1, or 2")

    terrain = terrain_hit
    sky_cloud = ~terrain & (np.min(transmittance, axis=-1) < 1.0)
    cloud_shadow_mask = terrain & (np.min(cloud_shadow, axis=-1) < 1.0)
    blocked = lighting == 1
    shaft = terrain & blocked
    if not shaft.any():
        raise ValueError("reference classification contains no terrain-hit shaft pixels")
    if not np.isfinite(terrain_slice[shaft]).all() or not (terrain_slice[shaft] < 64.0).all():
        raise ValueError("every shaft pixel must carry a finite exact terrain-hit depth slice")
    ys, xs = np.nonzero(blocked)
    godray_roi = np.zeros(shape, dtype=np.bool_)
    godray_roi[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    window = TECHNICAL_CONTRACTS["godray_roi_minimum_shape"]["value"]
    if godray_roi.any(axis=1).sum() < window or godray_roi.any(axis=0).sum() < window:
        raise ValueError("reference-classified godray ROI is smaller than tests/_ssim.py win_size=11")

    output_dir.mkdir(parents=True, exist_ok=True)
    for role, mask in {
        "sky_cloud_mask": sky_cloud,
        "godray_roi_mask": godray_roi,
        "terrain_mask": terrain,
        "cloud_shadow_mask": cloud_shadow_mask,
        "shaft_mask": shaft,
    }.items():
        _save_mask(output_dir / MASK_FILES[role], mask, EXPECTED_RULES[role])
