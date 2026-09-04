"""Deterministic analytic ridge fixture for OBLIQUA camera proofs.

Two north/south bars have vertical walls at known world X coordinates.  A camera
35 degrees from nadir views those walls along their full 700-unit length.
``ORTHOGRAPHIC_HALF_HEIGHT`` is derived from the projected half-length plus the
projected ridge height, with a 20 percent framing margin.  Consequently the
orthographic walls are parallel by construction, while the 8 degree pinhole
camera has analytically measurable perspective convergence.
"""
from __future__ import annotations

import math

import numpy as np

FIXTURE_SIZE = 1024
RIDGE_BAR_WIDTH = 48.0
RIDGE_BAR_SPACING = 384.0
RIDGE_BAR_LENGTH = 700.0
RIDGE_HEIGHT = 72.0
OBLIQUE_TILT_DEG = 35.0
PINHOLE_FOV_Y = 8.0
_FRAME_MARGIN = 1.20
_TILT = math.radians(OBLIQUE_TILT_DEG)
ORTHOGRAPHIC_HALF_HEIGHT = _FRAME_MARGIN * (
    0.5 * RIDGE_BAR_LENGTH * math.cos(_TILT) + RIDGE_HEIGHT * math.sin(_TILT)
)
CAMERA_TARGET = (0.0, 0.5 * RIDGE_HEIGHT, 0.0)
CAMERA_DISTANCE = ORTHOGRAPHIC_HALF_HEIGHT / math.tan(math.radians(PINHOLE_FOV_Y / 2.0))
CAMERA_UP = (0.0, math.sin(_TILT), -math.cos(_TILT))
EDGE_WORLD_X0 = -0.5 * RIDGE_BAR_SPACING + 0.5 * RIDGE_BAR_WIDTH
EDGE_WORLD_X1 = 0.5 * RIDGE_BAR_SPACING - 0.5 * RIDGE_BAR_WIDTH
EXPECTED_EDGE_X0 = (0.5 + EDGE_WORLD_X0 / (2.0 * ORTHOGRAPHIC_HALF_HEIGHT)) * FIXTURE_SIZE - 0.5
EXPECTED_EDGE_X1 = (0.5 + EDGE_WORLD_X1 / (2.0 * ORTHOGRAPHIC_HALF_HEIGHT)) * FIXTURE_SIZE - 0.5
EDGE_ROI_HALF_WIDTH = 14
RIDGE_ENDPOINTS_WORLD = (
    ((EDGE_WORLD_X0, RIDGE_HEIGHT, -0.5 * RIDGE_BAR_LENGTH), (EDGE_WORLD_X0, RIDGE_HEIGHT, 0.5 * RIDGE_BAR_LENGTH)),
    ((EDGE_WORLD_X1, RIDGE_HEIGHT, -0.5 * RIDGE_BAR_LENGTH), (EDGE_WORLD_X1, RIDGE_HEIGHT, 0.5 * RIDGE_BAR_LENGTH)),
)


def make_ridge_dem(size: int = FIXTURE_SIZE) -> np.ndarray:
    """Return a float32 DEM with two rectangular, parallel raised bars."""
    if size < 32:
        raise ValueError("ridge fixture size must be at least 32")
    scale = (FIXTURE_SIZE - 1) / (size - 1)
    x = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * scale
    z = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * scale
    xx, zz = np.meshgrid(x, z)
    bars = (
        (np.abs(xx + RIDGE_BAR_SPACING / 2.0) <= RIDGE_BAR_WIDTH / 2.0)
        | (np.abs(xx - RIDGE_BAR_SPACING / 2.0) <= RIDGE_BAR_WIDTH / 2.0)
    ) & (np.abs(zz) <= RIDGE_BAR_LENGTH / 2.0)
    return np.where(bars, RIDGE_HEIGHT, 0.0).astype(np.float32)


def make_camera(model: str = "orthographic", oblique: bool = True) -> dict:
    """Return a camera dictionary using the fixture's derived framing."""
    if model not in {"orthographic", "pinhole"}:
        raise ValueError("model must be 'orthographic' or 'pinhole'")
    tilt = _TILT if oblique else 0.0
    distance = CAMERA_DISTANCE
    target = np.asarray(CAMERA_TARGET, dtype=np.float64)
    origin = target + np.array((0.0, distance * math.cos(tilt), distance * math.sin(tilt)))
    camera = {
        "model": model,
        "origin": tuple(origin),
        "look_at": CAMERA_TARGET,
        "up": (0.0, math.sin(tilt), -math.cos(tilt)),
    }
    if model == "orthographic":
        camera["half_height"] = ORTHOGRAPHIC_HALF_HEIGHT
        camera["fov_y"] = None
    else:
        camera["fov_y"] = PINHOLE_FOV_Y
    return camera


def _tls_angle(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if direction[1] < 0.0:
        direction = -direction
    return math.degrees(math.atan2(float(direction[0]), float(direction[1])))


def fit_ridge_edge_angles(
    depth: np.ndarray,
    expected_x0: float,
    expected_x1: float,
    roi_half_width: int,
) -> tuple[float, float]:
    """Fit the two depth-discontinuity loci with total least squares."""
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("depth must be a 2D array")
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    discontinuity = np.abs(np.diff(filled, axis=1))
    discontinuity[~(finite[:, :-1] | finite[:, 1:])] = 0.0
    angles = []
    rows = np.arange(values.shape[0])
    # Exclude the projected bar end caps.  Their horizontal discontinuities
    # share these X ROIs but are not samples of the two longitudinal walls.
    longitudinal = (rows >= 0.20 * values.shape[0]) & (rows < 0.80 * values.shape[0])
    for expected in (expected_x0, expected_x1):
        lo = max(0, int(math.floor(expected - roi_half_width)))
        hi = min(discontinuity.shape[1], int(math.ceil(expected + roi_half_width + 1)))
        roi = discontinuity[:, lo:hi]
        strongest = np.argmax(roi, axis=1)
        strength = roi[np.arange(roi.shape[0]), strongest]
        positive = strength[strength > 0.0]
        # Perspective depth changes smoothly across every ground pixel.  A
        # fixed >0 test therefore fits the ground gradient instead of the wall.
        # The true wall jump is >0.05 world units for this locked fixture.
        threshold = max(0.05, float(np.quantile(positive, 0.05))) if positive.size else math.inf
        keep = longitudinal & (strength > threshold)
        if np.count_nonzero(keep) < 32:
            raise AssertionError("fewer than 32 depth-discontinuity samples in ridge ROI")

        # Locate the discontinuity at subpixel precision by taking the centroid
        # of the depth derivative around its peak.  Four neighbours on either
        # side cover the one-cell bilinear height transition and remove the
        # integer-staircase angle bias without fitting an analytic line.
        subpixel_x = np.empty(values.shape[0], dtype=np.float64)
        for y, peak in enumerate(strongest):
            start = max(0, int(peak) - 4)
            stop = min(roi.shape[1], int(peak) + 5)
            weights = roi[y, start:stop]
            centers = lo + np.arange(start, stop, dtype=np.float64) + 0.5
            subpixel_x[y] = (
                float(np.dot(centers, weights) / weights.sum())
                if weights.sum() > 0.0
                else lo + float(peak) + 0.5
            )
        points = np.column_stack((subpixel_x[keep], rows[keep] + 0.5))
        angles.append(_tls_angle(points))
    return (angles[0], angles[1])


def _project(camera: dict, point: tuple[float, float, float], width: int, height: int) -> np.ndarray:
    origin = np.asarray(camera["origin"], dtype=np.float64)
    target = np.asarray(camera["look_at"], dtype=np.float64)
    forward = target - origin
    forward /= np.linalg.norm(forward)
    supplied_up = np.asarray(camera["up"], dtype=np.float64)
    right = np.cross(forward, supplied_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(point, dtype=np.float64) - origin
    z = float(np.dot(relative, forward))
    if z <= 0.0:
        raise ValueError("ridge endpoint is behind the camera")
    half_y = math.tan(math.radians(float(camera["fov_y"])) / 2.0)
    ndc_x = float(np.dot(relative, right)) / (z * half_y * (width / height))
    ndc_y = float(np.dot(relative, up)) / (z * half_y)
    return np.array(((ndc_x + 1.0) * width / 2.0 - 0.5, (1.0 - ndc_y) * height / 2.0 - 0.5))


def expected_pinhole_edge_angle_spread(
    camera: dict,
    ridge_endpoints_world,
    image_width: int,
    image_height: int,
) -> float:
    """Project 3D endpoints analytically and return edge-angle spread in degrees."""
    angles = []
    for endpoints in ridge_endpoints_world:
        p0 = _project(camera, endpoints[0], image_width, image_height)
        p1 = _project(camera, endpoints[1], image_width, image_height)
        delta = p1 - p0
        if delta[1] < 0.0:
            delta = -delta
        angles.append(math.degrees(math.atan2(float(delta[0]), float(delta[1]))))
    return abs(angles[1] - angles[0])
