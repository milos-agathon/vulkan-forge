#!/usr/bin/env python3
"""Evaluate one exact nested-prefix reference doubling against Gate 3/4 metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nephele_fixture_masks import MASK_FILES, generate_masks
from tests._deltae import delta_e_2000, srgb_to_lab
from tests._ssim import ssim


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/nephele/fixture"
PREFIX_ARTIFACTS = (
    "reference-rgb.npy",
    "reference-transmittance.npy",
    "reference-in-scatter.npy",
    "reference-cloud-shadow-aov.npy",
    "reference-optical-depth.npy",
    "reference-terrain-hit.npy",
    "reference-media-lighting-visibility.npy",
    "reference-terrain-slice.npy",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _tracked_name(name: str, samples_per_pixel: int) -> str:
    path = Path(name)
    return f"{path.stem}-spp{samples_per_pixel}{path.suffix}"


def _roi_crop(mask: np.ndarray, *images: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.nonzero(mask)
    return [image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] for image in images]


def _validate_spatial_tiles(provenance: dict) -> None:
    crop_x, crop_y, width, height = provenance["crop"]
    coverage = np.zeros((height, width), dtype=np.uint8)
    for tile in provenance["spatial_tiles"]:
        x, y, tile_width, tile_height = (
            tile["x"], tile["y"], tile["width"], tile["height"]
        )
        if (
            tile_width <= 0 or tile_height <= 0
            or x < crop_x or y < crop_y
            or x + tile_width > crop_x + width
            or y + tile_height > crop_y + height
        ):
            raise ValueError("reference spatial tile lies outside the tracked crop")
        coverage[
            y - crop_y : y - crop_y + tile_height,
            x - crop_x : x - crop_x + tile_width,
        ] += 1
    if not np.array_equal(coverage, np.ones((height, width), dtype=np.uint8)):
        raise ValueError("reference spatial tiles must cover every crop pixel exactly once")


def downstream_metrics(old: np.ndarray, new: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    delta_e = delta_e_2000(srgb_to_lab(old), srgb_to_lab(new))
    shadow_values = delta_e[masks["terrain_mask"] & masks["cloud_shadow_mask"]]
    if not shadow_values.size:
        raise ValueError("reference convergence cloud-shadow terrain population is empty")
    old_roi, new_roi = _roi_crop(masks["godray_roi_mask"], old, new)
    return {
        "gate3_sky_cloud_delta_e_below_2_5_fraction": float(
            np.mean(delta_e[masks["sky_cloud_mask"]] < 2.5)
        ),
        "gate3_godray_roi_ssim": ssim(old_roi, new_roi, data_range=255.0),
        "gate4_cloud_shadow_terrain_maximum_delta_e": float(shadow_values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", type=Path, required=True)
    args = parser.parse_args()
    prior = _object(args.previous_dir / "reference-provenance.json")
    final = _object(FIXTURE / "reference-provenance.json")
    if prior.get("acceptance_eligible") is not True or final.get("acceptance_eligible") is not True:
        raise ValueError("diagnostic references cannot seed acceptance convergence")
    _validate_spatial_tiles(prior)
    _validate_spatial_tiles(final)
    if prior["samples_per_pixel"] * 2 != final["samples_per_pixel"]:
        raise ValueError("reference convergence requires one exact sample-count doubling")
    for key in (
        "algorithm",
        "seed",
        "source_revision",
        "source_inputs",
        "scene_inputs",
        "assembled_sources",
        "native_runtime",
        "camera_contract",
        "full_viewport",
        "crop",
        "spatial_tiles",
    ):
        if prior.get(key) != final.get(key):
            raise ValueError(f"reference convergence identity changed: {key}")
    prior_identity = prior.get("sample_identity")
    final_identity = final.get("sample_identity")
    if (
        prior_identity != {
            "algorithm": "per-pixel-absolute-sample-index-v1",
            "seed": prior["seed"],
            "range": [0, prior["samples_per_pixel"]],
        }
        or final_identity != {
            "algorithm": "per-pixel-absolute-sample-index-v1",
            "seed": final["seed"],
            "range": [0, final["samples_per_pixel"]],
        }
    ):
        raise ValueError("reference convergence requires exact per-pixel [0,N) sample prefixes")

    previous_records: dict[str, dict[str, object]] = {}
    for name in PREFIX_ARTIFACTS:
        source = args.previous_dir / name
        target = FIXTURE / _tracked_name(name, prior["samples_per_pixel"])
        shutil.copyfile(source, target)
        previous_records[name] = {
            "path": target.relative_to(ROOT).as_posix(),
            "sha256": _sha256(target),
        }
    tracked_provenance = FIXTURE / _tracked_name(
        "reference-provenance.json", prior["samples_per_pixel"]
    )
    shutil.copyfile(args.previous_dir / "reference-provenance.json", tracked_provenance)

    with tempfile.TemporaryDirectory(prefix="nephele-prefix-masks-") as temporary:
        temporary = Path(temporary)
        old_fixture = temporary / "old-fixture"
        old_masks = temporary / "old-masks"
        new_masks = temporary / "new-masks"
        old_fixture.mkdir()
        for name in PREFIX_ARTIFACTS[1:]:
            shutil.copyfile(args.previous_dir / name, old_fixture / name)
        rules = FIXTURE / "mask-rules.json"
        generate_masks(old_fixture, rules, old_masks)
        generate_masks(FIXTURE, rules, new_masks)
        mask_identity = {
            role: np.array_equal(
                np.load(old_masks / filename, allow_pickle=False),
                np.load(new_masks / filename, allow_pickle=False),
            )
            for role, filename in MASK_FILES.items()
        }
        masks = {
            role: np.load(new_masks / filename, allow_pickle=False)
            for role, filename in MASK_FILES.items()
        }

    classification_identity = {
        name: np.array_equal(
            np.load(args.previous_dir / name, allow_pickle=False),
            np.load(FIXTURE / name, allow_pickle=False),
        )
        for name in (
            "reference-terrain-hit.npy",
            "reference-media-lighting-visibility.npy",
            "reference-terrain-slice.npy",
        )
    }

    old = np.load(args.previous_dir / "reference-rgb.npy", allow_pickle=False)
    new = np.load(FIXTURE / "reference-rgb.npy", allow_pickle=False)
    if old.dtype != np.uint8 or new.dtype != np.uint8 or old.shape != new.shape:
        raise ValueError("reference convergence requires matching uint8 sRGB arrays")
    metrics = downstream_metrics(old, new, masks)
    spatial_tiles_identity = prior["spatial_tiles"] == final["spatial_tiles"]
    converged = (
        spatial_tiles_identity
        and
        all(mask_identity.values())
        and all(classification_identity.values())
        and metrics["gate3_sky_cloud_delta_e_below_2_5_fraction"] >= 0.95
        and metrics["gate3_godray_roi_ssim"] > 0.95
        and metrics["gate4_cloud_shadow_terrain_maximum_delta_e"] < 2.0
    )
    generator = Path(__file__).resolve()
    record = {
        "schema": "forge3d.nephele.reference_convergence/2",
        "status": "CONVERGED" if converged else "UNRESOLVED",
        "criterion": "actual downstream Gate 3 and approved Gate 4 metrics over exact identical reference-derived masks",
        "previous": {
            "samples_per_pixel": prior["samples_per_pixel"],
            "provenance_path": tracked_provenance.relative_to(ROOT).as_posix(),
            "provenance_sha256": _sha256(tracked_provenance),
            "artifacts": previous_records,
        },
        "final": {
            "samples_per_pixel": final["samples_per_pixel"],
            "rgb_path": "tests/nephele/fixture/reference-rgb.npy",
            "rgb_sha256": _sha256(FIXTURE / "reference-rgb.npy"),
            "provenance_path": "tests/nephele/fixture/reference-provenance.json",
            "provenance_sha256": _sha256(FIXTURE / "reference-provenance.json"),
        },
        "nested_prefix": True,
        "spatial_tiles_identity": spatial_tiles_identity,
        "mask_identity": mask_identity,
        "classification_identity": classification_identity,
        "metrics": metrics,
        "next_samples_per_pixel_if_unresolved": None if converged else final["samples_per_pixel"] * 2,
        "generator": generator.relative_to(ROOT).as_posix(),
        "generator_sha256": _sha256(generator),
    }
    (FIXTURE / "reference-convergence.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not converged:
        raise SystemExit(
            "reference remains UNRESOLVED; render the recorded next exact nested-prefix doubling"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
