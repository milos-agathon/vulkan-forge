from __future__ import annotations

import numpy as np

import forge3d as f3d
from _bc_fixtures import (
    flat_normals,
    hard_albedo,
    normal_tilt_degrees,
    smooth_albedo,
    steep_normals,
)
from _deltae import delta_e_2000, srgb_to_lab
from _ssim import ssim
from _tessella_evidence import record_tessella_result

# TESSELLA win 4 thresholds. These are the spec's numbers and are never
# relaxed: if a fixture fails, the encoder is fixed, not the gate.
BC7_SSIM_GATE = 0.98
BC7_DELTA_E_GATE = 1.5
BC5_MEAN_ANGLE_GATE = 1.0
BC5_MAX_ANGLE_GATE = 4.0

# The fixtures are hash-derived, so their bytes are part of the gate.
HARD_ALBEDO_SHA256 = "58c09a0bd9e16469f4a285515aa7f67a8a0f2693f8a01e2038839651ff541925"
STEEP_NORMALS_RG_SHA256 = (
    "3b3ee1b94f97d540d9a330be32d1b862ac5857cc9a12b25bc647c8affcf014db"
)


def _bc7_roundtrip(source: np.ndarray):
    height, width = source.shape[:2]
    encoded = f3d.encode_bc7_rgba8(source.tobytes(), width, height)
    decoded = np.frombuffer(
        f3d.decode_bc7_rgba8(encoded, width, height), dtype=np.uint8
    ).reshape(source.shape)
    measured_ssim = float(ssim(source[..., :3], decoded[..., :3]))
    delta = delta_e_2000(srgb_to_lab(source[..., :3]), srgb_to_lab(decoded[..., :3]))
    return encoded, measured_ssim, delta


def _bc5_roundtrip(normals: np.ndarray):
    side = normals.shape[0]
    rg = np.round((normals[..., :2] * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    encoded = f3d.encode_bc5_rg8(rg.tobytes(), side, side)
    decoded = np.frombuffer(
        f3d.decode_bc5_rg8(encoded, side, side), dtype=np.uint8
    ).reshape(side, side, 2)
    decoded_xy = decoded.astype(np.float64) / 127.5 - 1.0
    decoded_z = np.sqrt(np.maximum(1.0 - np.sum(decoded_xy * decoded_xy, axis=-1), 0.0))
    decoded_normals = np.dstack([decoded_xy, decoded_z])
    dots = np.clip(np.sum(normals * decoded_normals, axis=-1), -1.0, 1.0)
    return rg, encoded, np.degrees(np.arccos(dots))


def test_bc7_mode6_clears_the_fidelity_gate_on_a_demanding_albedo():
    """The gate case: multi-octave detail, saturated hard edges, checkerboards
    and thin high-contrast lines -- content a single-subset BC7 mode has to
    work for, not a gradient it cannot fail."""
    source = hard_albedo()
    encoded, measured_ssim, delta = _bc7_roundtrip(source)

    assert len(encoded) * 4 == source.nbytes
    assert measured_ssim >= BC7_SSIM_GATE, measured_ssim
    assert float(delta.mean()) < BC7_DELTA_E_GATE, float(delta.mean())
    record_tessella_result(
        "bc7_fidelity",
        {
            "texture_family": "albedo",
            "fixture": "hard_albedo_256",
            "source_bytes": source.nbytes,
            "encoded_bytes": len(encoded),
            "compression_ratio": source.nbytes / len(encoded),
            "ssim": measured_ssim,
            "mean_delta_e_2000": float(delta.mean()),
            "p99_delta_e_2000": float(np.percentile(delta, 99)),
            "max_delta_e_2000": float(delta.max()),
            "ssim_gate": BC7_SSIM_GATE,
            "mean_delta_e_gate": BC7_DELTA_E_GATE,
        },
    )


def test_bc7_smooth_baseline_is_easier_than_the_gate_fixture():
    """Locks the claim that the gate fixture is the harder of the two, so a
    future edit cannot quietly swap the gate back to best-case content."""
    smooth_encoded, smooth_ssim, smooth_delta = _bc7_roundtrip(smooth_albedo())
    _, hard_ssim, hard_delta = _bc7_roundtrip(hard_albedo())

    assert smooth_ssim > hard_ssim
    assert float(smooth_delta.mean()) < float(hard_delta.mean())
    assert smooth_ssim >= BC7_SSIM_GATE
    record_tessella_result(
        "bc7_fidelity_smooth_baseline",
        {
            "fixture": "smooth_albedo_64",
            "encoded_bytes": len(smooth_encoded),
            "ssim": smooth_ssim,
            "mean_delta_e_2000": float(smooth_delta.mean()),
        },
    )


def test_bc5_normal_angular_error_clears_gate_on_steep_normals():
    """The gate case: a 5-octave surface whose normals tilt far past the
    near-flat dome the earlier fixture used."""
    normals = steep_normals()
    rg, encoded, angular = _bc5_roundtrip(normals)
    tilt = normal_tilt_degrees(normals)

    assert len(encoded) * 2 == rg.nbytes
    assert float(angular.mean()) < BC5_MEAN_ANGLE_GATE, float(angular.mean())
    assert float(angular.max()) < BC5_MAX_ANGLE_GATE, float(angular.max())
    record_tessella_result(
        "bc5_fidelity",
        {
            "texture_family": "normal",
            "fixture": "steep_normals_256",
            "source_bytes": rg.nbytes,
            "encoded_bytes": len(encoded),
            "compression_ratio": rg.nbytes / len(encoded),
            "mean_angular_error_degrees": float(angular.mean()),
            "max_angular_error_degrees": float(angular.max()),
            "source_tilt_mean_degrees": float(tilt.mean()),
            "source_tilt_max_degrees": float(tilt.max()),
            "mean_angle_gate_degrees": BC5_MEAN_ANGLE_GATE,
            "max_angle_gate_degrees": BC5_MAX_ANGLE_GATE,
        },
    )


def test_bc5_flat_baseline_is_easier_than_the_gate_fixture():
    _, _, steep_angular = _bc5_roundtrip(steep_normals())
    _, flat_encoded, flat_angular = _bc5_roundtrip(flat_normals())

    assert float(flat_angular.mean()) < float(steep_angular.mean())
    assert (
        normal_tilt_degrees(flat_normals()).max()
        < normal_tilt_degrees(steep_normals()).max()
    )
    record_tessella_result(
        "bc5_fidelity_flat_baseline",
        {
            "fixture": "flat_normals_64",
            "encoded_bytes": len(flat_encoded),
            "mean_angular_error_degrees": float(flat_angular.mean()),
            "max_angular_error_degrees": float(flat_angular.max()),
        },
    )


def test_bc_fixtures_are_bit_stable():
    """Win 4 is a numeric gate, so its inputs must be reproducible bytes, not
    a NumPy RNG stream that may change between releases."""
    import hashlib

    first = hard_albedo()
    assert first.dtype == np.uint8
    assert np.array_equal(first, hard_albedo())
    assert hashlib.sha256(first.tobytes()).hexdigest() == HARD_ALBEDO_SHA256

    rg, _, _ = _bc5_roundtrip(steep_normals())
    assert rg.dtype == np.uint8
    assert hashlib.sha256(rg.tobytes()).hexdigest() == STEEP_NORMALS_RG_SHA256
