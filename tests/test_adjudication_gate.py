# tests/test_adjudication_gate.py
# AEQUITAS perceptual adjudication gate: PT-vs-raster ground-truth parity.
#
# Section 1 (no GPU): unit tests for tests/_deltae.py against the Sharma et al.
# (2005) CIEDE2000 reference vectors and mask/band sanity checks.
# Section 2 (GPU): renders the committed reference scene both ways via
# forge3d.render_adjudication_pair and asserts the measurable win:
#   dE2000 < 2.0 on >= 95% of lit pixels  AND  SSIM > 0.96 on the shadow band.
# Follows the recipe-golden skip convention (skip on unsupported hosted GPUs,
# hard-fail on regression) and persists goldens under tests/golden/adjudication/.

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _deltae import (
    band_bbox,
    delta_e_2000,
    lit_mask,
    shadow_boundary_band,
    srgb_to_lab,
)
from _ssim import ssim
from _terrain_runtime import terrain_rendering_available

import forge3d as f3d

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "adjudication"
UPDATE_GOLDENS = os.environ.get("FORGE3D_UPDATE_ADJUDICATION_GOLDENS") == "1"
ARTIFACT_DIR = os.environ.get("FORGE3D_ADJUDICATION_ARTIFACT_DIR")

# Gate parameters (locked; changing them invalidates the committed goldens).
GATE_WIDTH = 512
GATE_HEIGHT = 512
GATE_SPP = 4096
DELTA_E_MAX = 2.0
LIT_PASS_FRACTION_MIN = 0.95
BAND_SSIM_MIN = 0.96
# Drift thresholds vs committed goldens (terrain-golden convention).
DRIFT_SSIM_MIN = 0.995
DRIFT_MEAN_ABS_MAX = 2.0


# ---------------------------------------------------------------------------
# Section 1: metric primitives (always run; no GPU required)
# ---------------------------------------------------------------------------

# Reference pairs from Sharma, Wu & Dalal (2005), Table 1 (kL=kC=kH=1).
SHARMA_VECTORS = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA_VECTORS)
def test_deltae2000_sharma_vectors(lab1, lab2, expected):
    got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
    assert abs(got - expected) < 1e-3, f"dE2000({lab1},{lab2}) = {got:.4f}, want {expected:.4f}"
    # Symmetry (kL=kC=kH=1 makes CIEDE2000 symmetric).
    rev = float(delta_e_2000(np.array(lab2), np.array(lab1)))
    assert abs(rev - expected) < 1e-3


def test_srgb_to_lab_white_and_black():
    lab = srgb_to_lab(np.array([[[255, 255, 255]]], dtype=np.uint8))
    assert np.allclose(lab[0, 0], [100.0, 0.0, 0.0], atol=1e-2)
    lab0 = srgb_to_lab(np.array([[[0, 0, 0]]], dtype=np.uint8))
    assert np.allclose(lab0[0, 0], [0.0, 0.0, 0.0], atol=1e-6)


def test_lit_mask_and_boundary_band_shapes():
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[:, 24:, :] = 200  # right half lit
    lit = lit_mask(img)
    assert lit.shape == (32, 48) and lit.dtype == bool
    assert not lit[:, :24].any() and lit[:, 24:].all()

    band = shadow_boundary_band(img, band_px=3)
    assert band.shape == (32, 48) and band.dtype == bool
    # Band straddles the lit/shadow edge at x=24 and is thin.
    assert band[:, 24].all()
    assert not band[:, :19].any() and not band[:, 40:].any()
    ys, xs = band_bbox(band)
    assert xs.start >= 19 and xs.stop <= 40


def test_deltae_identical_images_is_zero():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    lab = srgb_to_lab(img)
    de = delta_e_2000(lab, lab)
    assert de.shape == (16, 16)
    assert float(np.abs(de).max()) < 1e-9


# ---------------------------------------------------------------------------
# Section 2: the adjudication gate (GPU; recipe-golden skip convention)
# ---------------------------------------------------------------------------


def _save_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f3d.numpy_to_png(str(path), rgba)


def _write_failure_artifacts(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not ARTIFACT_DIR:
        return
    out = Path(ARTIFACT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    _save_png(out / f"{name}_actual.png", actual)
    _save_png(out / f"{name}_expected.png", expected)
    diff = np.abs(
        actual[..., :3].astype(np.int16) - expected[..., :3].astype(np.int16)
    ).astype(np.uint8)
    _save_png(out / f"{name}_diff.png", diff)


def _assert_matches_golden(name: str, actual: np.ndarray) -> None:
    golden_path = GOLDEN_DIR / f"{name}.png"
    if UPDATE_GOLDENS:
        _save_png(golden_path, actual)
        return
    assert golden_path.exists(), (
        f"Missing adjudication golden {golden_path}. "
        "Regenerate with FORGE3D_UPDATE_ADJUDICATION_GOLDENS=1."
    )
    expected = f3d.png_to_numpy(str(golden_path))
    assert actual.shape == expected.shape
    mean_abs = float(
        np.mean(np.abs(actual[..., :3].astype(np.float32) - expected[..., :3].astype(np.float32)))
    )
    score = ssim(actual[..., :3], expected[..., :3], data_range=255.0)
    if score < DRIFT_SSIM_MIN or mean_abs > DRIFT_MEAN_ABS_MAX:
        _write_failure_artifacts(name, actual, expected)
    assert score >= DRIFT_SSIM_MIN, f"{name} drift: SSIM too low vs golden: {score:.6f}"
    assert mean_abs <= DRIFT_MEAN_ABS_MAX, f"{name} drift: mean abs diff too high: {mean_abs:.4f}"


@pytest.mark.apple_metal_physical
def test_adjudication_gate():
    if not terrain_rendering_available():
        pytest.skip(
            "Adjudication gate requires a terrain-capable hardware-backed forge3d runtime"
        )

    pt_rgba, raster_rgba, meta = f3d.render_adjudication_pair(
        GATE_WIDTH, GATE_HEIGHT, GATE_SPP
    )
    assert pt_rgba.shape == (GATE_HEIGHT, GATE_WIDTH, 4) and pt_rgba.dtype == np.uint8
    assert raster_rgba.shape == (GATE_HEIGHT, GATE_WIDTH, 4) and raster_rgba.dtype == np.uint8

    # Both renders must come from the single ReferenceSceneDesc: identical
    # camera/light metadata for both paths.
    assert meta["pt"] == meta["raster"], (
        f"camera/light metadata mismatch between PT and raster paths: {meta}"
    )

    # Constant sky/ambient contract: both paths must report the literal
    # constants from the single ReferenceSceneDesc (no gradient fields).
    for key in ("ambient_r", "ambient_g", "ambient_b", "sky_r", "sky_g", "sky_b"):
        assert key in meta["pt"], f"missing constant ambient/sky metadata key {key}"
        assert meta["pt"][key] == meta["raster"][key]

    # --- Metric 1: dE2000 over lit pixels of the PT reference ---
    lit = lit_mask(pt_rgba)
    assert lit.any(), "lit-pixel mask is empty; scene/exposure regression"
    de = delta_e_2000(srgb_to_lab(pt_rgba), srgb_to_lab(raster_rgba))
    lit_pass_fraction = float((de[lit] < DELTA_E_MAX).mean())

    # --- Metric 2: SSIM on the shadow-boundary band of the PT reference ---
    band = shadow_boundary_band(pt_rgba)
    ys, xs = band_bbox(band)
    band_ssim = ssim(
        pt_rgba[ys, xs, :3], raster_rgba[ys, xs, :3], data_range=255.0
    )

    print(
        f"\nADJUDICATION: dE2000<{DELTA_E_MAX} on {lit_pass_fraction * 100.0:.4f}% "
        f"of lit pixels (need >= {LIT_PASS_FRACTION_MIN * 100.0:.1f}%); "
        f"shadow-boundary SSIM = {band_ssim:.6f} (need > {BAND_SSIM_MIN})"
    )

    if UPDATE_GOLDENS:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        (GOLDEN_DIR / "scores.json").write_text(
            json.dumps(
                {
                    "width": GATE_WIDTH,
                    "height": GATE_HEIGHT,
                    "spp": GATE_SPP,
                    "lit_pass_fraction": lit_pass_fraction,
                    "shadow_band_ssim": band_ssim,
                },
                indent=2,
            )
            + "\n"
        )

    # The measurable win (hard gate; not xfail, not skip-on-error).
    assert lit_pass_fraction >= LIT_PASS_FRACTION_MIN, (
        f"dE2000 gate failed: only {lit_pass_fraction * 100.0:.4f}% of lit pixels "
        f"under dE {DELTA_E_MAX}"
    )
    assert band_ssim > BAND_SSIM_MIN, (
        f"shadow-boundary SSIM gate failed: {band_ssim:.6f} <= {BAND_SSIM_MIN}"
    )

    # Drift detection against the committed reference renders.
    _assert_matches_golden("pt_reference", pt_rgba)
    _assert_matches_golden("raster_reference", raster_rgba)
