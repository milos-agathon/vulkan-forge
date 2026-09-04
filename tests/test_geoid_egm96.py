# tests/test_geoid_egm96.py
# MENSURA win 3 (CI-gated): EGM96 geoid.
# N(lat, lon) at degree/order 120 matches the committed NGA-published test
# values to < 0.5 m, and a DEM tagged orthometric converts to ellipsoidal
# heights differing from the raw values by exactly N (per-pixel, 1e-6 m).
# RELEVANT FILES: src/geo/geoid.rs, assets/geoid/egm96_n120.bin,
#                 tests/data/egm96_test_values.txt

import hashlib
import math
import platform
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

import forge3d


_BYTE_LOCK_POINTS = [
    (-89.5, 0.5),
    (-75.25, 42.75),
    (-60.0, -120.0),
    (-45.5, 179.5),
    (-30.25, -179.75),
    (-15.0, 90.0),
    (0.0, 0.0),
    (0.5, 179.5),
    (12.345, 67.89),
    (23.5, -45.5),
    (35.0, 120.0),
    (46.87, 102.45),
    (51.5074, -0.1278),
    (60.0, 10.0),
    (70.25, -135.0),
    (80.0, 179.0),
    (89.5, 359.5),
    (-33.8688, 151.2093),
    (27.9881, 86.925),
    (-22.9068, -43.1729),
]

# Rust's f64 trigonometric operations use the target platform's libm, so an
# optimized payload is byte-stable per target rather than portable across
# targets. These release-build hashes lock the exact pre-SELENE EGM96 payload
# on the three targets that run this gate. The macOS arm64 value was reproduced
# directly from pre-refactor commit 7fa1b984 with the same release toolchain;
# the Linux x86_64 and Windows AMD64 values are stable hosted-wheel payloads.
_BYTE_LOCK_SHA256 = {
    ("darwin", "arm64"): "86291cb905156dabb987bf57c53b42f124e2cf1047dc5b9145e4e46c0a856a17",
    ("linux", "x86_64"): "5deafb3b0a40962cd947c714bead4c5e86a038793723d9fd252dfb90e751ee61",
    ("win32", "amd64"): "ab9469d5e078dbfaa5df9b02219733c482ee21ed1f872fa251ed7056da27a639",
}


def _reference_points():
    path = Path(__file__).parent / "data" / "egm96_test_values.txt"
    points = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lat, lon, n_ref, source = line.split()
        points.append((float(lat), float(lon), float(n_ref), source))
    return points


def test_egm96_degree_120_matches_nga_published_values():
    points = _reference_points()
    assert len(points) == 20, "expected the 20 committed NGA reference points"
    worst = 0.0
    worst_at = None
    for lat, lon, n_ref, source in points:
        n = forge3d.geoid_undulation(lat, lon)
        err = abs(n - n_ref)
        if err > worst:
            worst, worst_at = err, (lat, lon, source)
        assert err < 0.5, (
            f"EGM96 residual {err:.3f} m at ({lat}, {lon}) [{source}]: "
            f"got {n:.3f}, want {n_ref:.3f}"
        )
    print(
        "EGM96 degree-120 worst residual vs published degree-360 values: "
        f"{worst:.4f} m at {worst_at}"
    )


def test_egm96_refactor_is_byte_identical():
    payload = b"".join(
        struct.pack("<d", forge3d.geoid_undulation(lat, lon))
        for lat, lon in _BYTE_LOCK_POINTS
    )
    target = (sys.platform, platform.machine().lower())
    actual = hashlib.sha256(payload).hexdigest()
    expected = _BYTE_LOCK_SHA256.get(target)
    assert expected is not None, (
        f"no reviewed EGM96 release baseline for {target}; "
        f"measured release payload sha256={actual}"
    )
    assert actual == expected


def test_known_undulation_signs_and_magnitudes():
    # Sanity anchors: strongly negative over the Indian Ocean low, strongly
    # positive over the North Atlantic/Iceland high.
    assert forge3d.geoid_undulation(5.0, 78.0) < -80.0
    assert forge3d.geoid_undulation(64.0, -22.0) > 50.0


def test_dem_orthometric_to_ellipsoidal_differs_by_exactly_n():
    rng = np.random.default_rng(7)
    rows, cols = 12, 16
    dem = rng.uniform(-100.0, 3000.0, (rows, cols))
    bounds = (13.0, 52.0, 13.4, 52.3)  # (left, bottom, right, top), EPSG:4326
    out = forge3d.dem_orthometric_to_ellipsoidal(dem, bounds)
    assert out.shape == (rows, cols)
    assert out.dtype == np.float64

    left, bottom, right, top = bounds
    worst = 0.0
    for r in range(rows):
        lat = top - (r + 0.5) * (top - bottom) / rows
        for c in range(cols):
            lon = left + (c + 0.5) * (right - left) / cols
            n = forge3d.geoid_undulation(lat, lon)
            expected = dem[r, c] + n
            worst = max(worst, abs(out[r, c] - expected))
    assert worst < 1e-6, f"per-pixel residual {worst} m exceeds 1e-6"


def test_scalar_height_conversions_are_exact_inverses():
    lat, lon, h = 46.8743190, 102.4487290, 812.5
    n = forge3d.geoid_undulation(lat, lon)
    ell = forge3d.orthometric_to_ellipsoidal(h, lat, lon)
    assert abs(ell - (h + n)) < 1e-12
    back = forge3d.ellipsoidal_to_orthometric(ell, lat, lon)
    assert abs(back - h) < 1e-12


# --- MENSURA M-03: EGM96 boundary cases -------------------------------------


def test_egm96_poles_and_equator_are_finite():
    # The poles and the equator evaluate to finite undulations (no NaN/Inf).
    for lat in (90.0, -90.0, 0.0):
        assert math.isfinite(forge3d.geoid_undulation(lat, 0.0))


def test_egm96_pole_undulation_is_longitude_independent():
    # At a geographic pole the point is a single location, so the synthesized
    # undulation must not depend on the (degenerate) longitude.
    north = forge3d.geoid_undulation(90.0, 0.0)
    south = forge3d.geoid_undulation(-90.0, 0.0)
    for lon in (0.0, 42.0, 137.5, -168.0, 360.0):
        assert forge3d.geoid_undulation(90.0, lon) == pytest.approx(north, abs=1e-9)
        assert forge3d.geoid_undulation(-90.0, lon) == pytest.approx(south, abs=1e-9)


def test_egm96_longitude_wrap_equivalence():
    # 0 and 360 (and -179 and 181) name the same meridian; the geoid is periodic
    # in longitude to floating-point precision.
    for lat in (0.0, 37.5, -48.25, 64.0):
        assert forge3d.geoid_undulation(lat, 0.0) == pytest.approx(
            forge3d.geoid_undulation(lat, 360.0), abs=1e-9
        )
        assert forge3d.geoid_undulation(lat, -179.0) == pytest.approx(
            forge3d.geoid_undulation(lat, 181.0), abs=1e-9
        )


@pytest.mark.parametrize(
    "lat,lon",
    [(90.0001, 0.0), (-90.5, 0.0), (120.0, 0.0), (-181.0, 0.0)],
)
def test_egm96_latitude_out_of_range_raises(lat, lon):
    # Latitudes outside [-90, 90] are rejected at the Python boundary rather than
    # silently synthesizing a nonsense value.
    with pytest.raises(ValueError):
        forge3d.geoid_undulation(lat, lon)


@pytest.mark.parametrize(
    "lat,lon",
    [
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, float("inf")),
        (0.0, float("-inf")),
    ],
)
def test_egm96_non_finite_inputs_raise(lat, lon):
    with pytest.raises(ValueError):
        forge3d.geoid_undulation(lat, lon)
