"""SIDERA ephemeris gates against the committed JPL Horizons oracle.

Every threshold here is a Definition-of-Done gate from
``docs/prompts/fable5-moonshots/28-sidera.md``. None of them compares SIDERA
against a stored copy of its own output: the position gates use JPL Horizons
vectors, the ΔT gate uses Horizons' own time-scale columns, and the sidereal
and refraction gates cross-check two independently published formulae.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import pytest

import forge3d as f3d
from forge3d import astro, sky
from forge3d._native import get_native_module

# stdlib `tomllib` is 3.11+; the package supports 3.10, so use the repo shim.
from _toml_compat import load_toml

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data" / "horizons_vectors.dat"
THRESHOLDS = {
    "sun": 10.0,
    "moon": 30.0,
    "mercury": 60.0,
    "venus": 60.0,
    "mars": 60.0,
    "jupiter": 60.0,
    "saturn": 60.0,
}


def test_hashed_text_oracles_are_checked_out_with_canonical_lf_bytes():
    attributes = {
        line.strip()
        for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    }
    assert "assets/astro/*.dat text eol=lf" in attributes
    assert "tests/data/horizons_vectors.dat text eol=lf" in attributes


def _separation_arcsec(a, b):
    azimuth_delta = math.radians(a[0] - b[0])
    altitude_a = math.radians(a[1])
    altitude_b = math.radians(b[1])
    cosine = (
        math.sin(altitude_a) * math.sin(altitude_b)
        + math.cos(altitude_a) * math.cos(altitude_b) * math.cos(azimuth_delta)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3_600.0


def _utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _oracle_rows():
    """Yield the parsed body rows of the committed Horizons vector file."""
    for line in DATA.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0].startswith("@"):
            continue
        yield fields


def _tagged_rows(tag):
    for line in DATA.read_text(encoding="ascii").splitlines():
        if line.startswith(tag + " "):
            yield line.split()


def test_public_ephemeris_meets_committed_horizons_gates(capsys):
    """DoD 1–3: Sun ≤ 10″, Moon ≤ 30″ + phase/semidiameter, planets ≤ 60″.

    The NOAA Solar Calculator baseline that predates SIDERA
    (``src/lighting/ephemeris.rs``, exported as ``forge3d.sun_position_utc``)
    is measured on the very same vectors and reported in the same table, as
    DoD 1 requires.
    """
    maxima = defaultdict(float)
    worst = {}
    noaa_maximum = 0.0
    noaa_worst = None
    phase_max = semidiameter_max = 0.0
    combinations = set()
    rows = 0
    phase_rows = 0

    for line in DATA.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "@moon_phase":
            phase_rows += 1
            phase = astro.moon_phase(_utc(fields[1]))
            phase_max = max(phase_max, abs(phase[0] - float(fields[2]) / 100.0))
            semidiameter_max = max(semidiameter_max, abs(phase[2] - float(fields[3]) * 0.5))
            continue
        if fields[0].startswith("@"):
            continue
        rows += 1
        site, iso, body = fields[0], fields[1], fields[2]
        combinations.add((site, iso))
        latitude, longitude, height = float(fields[3]), float(fields[4]), float(fields[5])
        reference = (float(fields[6]), float(fields[7]))
        deviation = _separation_arcsec(
            astro.body_position(
                body, _utc(iso), latitude, longitude, height_m=height
            ),
            reference,
        )
        if deviation > maxima[body]:
            maxima[body] = deviation
            worst[body] = f"{site} {iso}"
        if body == "sun":
            when = _utc(iso)
            baseline = f3d.sun_position_utc(
                latitude,
                longitude,
                when.year,
                when.month,
                when.day,
                when.hour,
                when.minute,
                int(when.second),
            )
            noaa_deviation = _separation_arcsec(
                (baseline.azimuth, baseline.elevation), reference
            )
            if noaa_deviation > noaa_maximum:
                noaa_maximum = noaa_deviation
                noaa_worst = f"{site} {iso}"

    # DoD 5 of the oracle spec: at least 40 epoch/site combinations.
    assert len(combinations) >= 40, f"only {len(combinations)} epoch/site combinations"
    assert rows == len(combinations) * len(THRESHOLDS), rows
    assert phase_rows >= len(combinations) // len({c[0] for c in combinations})
    assert set(maxima) == set(THRESHOLDS)

    lunar_window_rows = 0
    regression_present = False
    for fields in _tagged_rows("@moon_window"):
        lunar_window_rows += 1
        _, site, iso, latitude, longitude, height, azimuth, altitude = fields
        regression_present |= site == "tromso" and iso == "2008-11-14T00:00:00Z"
        deviation = _separation_arcsec(
            astro.body_position(
                "moon",
                _utc(iso),
                float(latitude),
                float(longitude),
                height_m=float(height),
            ),
            (float(azimuth), float(altitude)),
        )
        if deviation > maxima["moon"]:
            maxima["moon"] = deviation
            worst["moon"] = f"{site} {iso} [30-day window sweep]"
    assert lunar_window_rows == 621
    assert regression_present, "the independently observed Tromso regression is missing"

    with capsys.disabled():
        print(
            f"\nSIDERA vs JPL Horizons - {rows} vectors over "
            f"{len(combinations)} epoch/site combinations (2000-2050)"
        )
        print(f"{'body':10s}{'max dev (arcsec)':>18s}{'gate':>8s}{'margin':>10s}  worst vector")
        for body in THRESHOLDS:
            print(
                f"{body:10s}{maxima[body]:18.3f}{THRESHOLDS[body]:8.1f}"
                f"{THRESHOLDS[body] - maxima[body]:10.3f}  {worst[body]}"
            )
        print(
            f"{'sun (NOAA)':10s}{noaa_maximum:18.3f}{'-':>8s}{'-':>10s}  {noaa_worst}"
            "   [pre-SIDERA baseline, src/lighting/ephemeris.rs]"
        )
        print(
            f"moon illuminated fraction max |err| {phase_max:.6f} (gate 0.005); "
            f"semidiameter max |err| {semidiameter_max:.4f}\" (gate 1.0\")"
        )

    for body, threshold in THRESHOLDS.items():
        assert maxima[body] <= threshold, (
            f"{body} missed its {threshold}\" gate at {maxima[body]:.3f}\" "
            f"on vector {worst[body]}"
        )
    assert phase_max <= 0.005
    assert semidiameter_max <= 1.0
    # The whole point of the new core: it must beat the NOAA day-calculator on
    # the Sun by a wide margin, not merely match it.
    assert maxima["sun"] * 4.0 < noaa_maximum, (
        f"SIDERA sun {maxima['sun']:.3f}\" is not decisively better than the "
        f"NOAA baseline {noaa_maximum:.3f}\""
    )


def test_delta_t_matches_the_horizons_time_scale_columns(capsys):
    """The committed ΔT fit's stated residual, gated against the oracle.

    Horizons reports ``TDB−UT`` (col. 30, with UT = UTC) and ``UT1−UTC``
    (col. 49) per epoch. TT−UT1 = (TDB−UTC) − (UT1−UTC) up to the TDB−TT
    periodic term, which stays below 2 ms. SIDERA's piecewise-linear fit must
    reproduce that to well under a second.
    """
    native = get_native_module()
    worst = 0.0
    worst_epoch = None
    manifest = load_toml(ROOT / "assets" / "astro" / "MANIFEST.toml")
    delta_asset = next(entry for entry in manifest["asset"] if entry["path"] == "delta_t_fit.dat")
    declared_residual = float(delta_asset["max_validated_midmonth_residual_seconds"])
    oracle_manifest = load_toml(DATA.with_name("horizons_vectors.MANIFEST.toml"))
    epochs = 0
    for fields in _tagged_rows("@delta_t_midmonth"):
        epochs += 1
        when = _utc(fields[1])
        expected = float(fields[2])
        actual = native.astro_delta_t_seconds(
            when.year, when.month, when.day, when.hour, when.minute, float(when.second)
        )
        if abs(actual - expected) > worst:
            worst = abs(actual - expected)
            worst_epoch = fields[1]
    assert epochs == oracle_manifest["delta_t_midmonth_vectors"] == 612
    with capsys.disabled():
        print(
            f"\ndelta-T (TT-UT1) vs Horizons over {epochs} epochs: "
            f"max |residual| {worst:.6f} s (declared gate {declared_residual} s) "
            f"at {worst_epoch}"
        )
    assert worst <= declared_residual, f"ΔT residual {worst} s at {worst_epoch}"
    assert declared_residual < 1.0


def test_observation_sets_every_rendered_body_and_star_catalog():
    summary = sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    assert summary["star_count"] == 9_096
    assert set(THRESHOLDS) <= summary.keys()
    assert len(summary["moon_phase"]) == 3


def test_observation_routes_to_active_subprocess_viewer():
    class FakeViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    fake = FakeViewer()
    sky._set_active_viewer(fake)
    try:
        sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        assert fake.command["cmd"] == "set_observation"
    finally:
        sky._set_active_viewer(None)


def test_observation_replays_when_viewer_opens_later():
    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )

    class FakeViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    fake = FakeViewer()
    try:
        sky._set_active_viewer(fake)
        assert fake.command["cmd"] == "set_observation"
    finally:
        sky._set_active_viewer(None)


def test_observation_replay_survives_viewer_restart():
    class FakeViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    first = FakeViewer()
    second = FakeViewer()
    sky._set_active_viewer(first)
    try:
        sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        sky._remove_active_viewer(first)
        first.is_running = False
        sky._set_active_viewer(second)
        assert second.command["cmd"] == "set_observation"
    finally:
        sky._set_active_viewer(None)


def test_temporary_viewer_restores_previous_observation_target(monkeypatch):
    class FakeViewer:
        is_running = True

        def __init__(self):
            self.commands = []

        def send_ipc(self, command):
            self.commands.append(dict(command))

    monkeypatch.setattr(sky, "get_native_module", lambda: None)
    sky._set_active_viewer(None)
    sky._clear_observation_replay()
    previous = FakeViewer()
    temporary = FakeViewer()
    sky._set_active_viewer(previous)
    try:
        sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        sky._set_active_viewer(temporary)
        sky.set_observation(
            datetime(2027, 7, 26, 22, tzinfo=timezone.utc), 19.8207, -155.4681
        )
        sky._remove_active_viewer(temporary)
        assert sky._get_active_viewer() is previous
        assert [command["year"] for command in previous.commands] == [2026, 2027]
        assert [command["year"] for command in temporary.commands] == [2026, 2027]
    finally:
        sky._set_active_viewer(None)
        sky._clear_observation_replay()


def test_promoted_viewer_replay_failure_is_cleanup_safe_and_retryable(monkeypatch):
    class FakeViewer:
        is_running = True

        def __init__(self):
            self.commands = []
            self.fail = False

        def send_ipc(self, command):
            self.commands.append(dict(command))
            if self.fail:
                raise RuntimeError("viewer IPC disappeared")

    monkeypatch.setattr(sky, "get_native_module", lambda: None)
    sky._set_active_viewer(None)
    sky._clear_observation_replay()
    previous = FakeViewer()
    temporary = FakeViewer()
    sky._set_active_viewer(previous)
    try:
        sky._set_active_viewer(temporary)
        sky.set_observation(
            datetime(2027, 7, 26, 22, tzinfo=timezone.utc), 19.8207, -155.4681
        )
        previous.fail = True
        sky._remove_active_viewer(temporary)
        assert sky._get_active_viewer() is previous
        assert previous.commands[-1]["year"] == 2027

        previous.fail = False
        sky._set_active_viewer(previous)
        assert previous.commands[-1]["year"] == 2027
    finally:
        sky._set_active_viewer(None)
        sky._clear_observation_replay()


def test_viewer_close_keeps_replay_cleanup_best_effort(monkeypatch):
    from forge3d import viewer as viewer_module

    handle = object.__new__(viewer_module.ViewerHandle)
    handle._socket = None
    handle._process = None
    handle._cleanup_paths = []
    monkeypatch.setattr(
        sky,
        "_remove_active_viewer",
        lambda _viewer: (_ for _ in ()).throw(RuntimeError("broken replay")),
    )
    handle.close()


def test_blocking_viewer_keeps_direct_launch_without_observation(monkeypatch):
    from forge3d import viewer as viewer_module

    class Process:
        def wait(self):
            return 7

    launched = []
    sky._clear_observation_replay()
    monkeypatch.setattr(viewer_module, "_find_viewer_binary", lambda: "viewer")
    monkeypatch.setattr(
        viewer_module.subprocess,
        "Popen",
        lambda command: launched.append(command) or Process(),
    )
    assert viewer_module.open_viewer(width=320, height=200) == 7
    assert launched == [["viewer", "--size", "320x200", "--fov", "60.0"]]


def test_observation_aware_blocking_snapshot_closes_after_capture(monkeypatch):
    from forge3d import viewer as viewer_module

    class Process:
        def wait(self):
            raise AssertionError("snapshot flow must not wait for an interactive close")

    class Viewer:
        _process = Process()

        def __init__(self):
            self.snapshot_path = None
            self.closed = False

        def snapshot(self, path, *, width, height):
            self.snapshot_path = (path, width, height)

        def close(self):
            self.closed = True

    handle = Viewer()
    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    monkeypatch.setattr(viewer_module, "open_viewer_async", lambda **_kwargs: handle)
    assert viewer_module.open_viewer(snapshot_path="night.png") == 0
    assert handle.snapshot_path == ("night.png", 1280, 720)
    assert handle.closed
    sky._clear_observation_replay()


@pytest.mark.parametrize(
    "command",
    [
        {"cmd": "lit_sun", "azimuth_deg": 180.0, "elevation_deg": 30.0},
        {
            "cmd": "set_terrain_sun",
            "azimuth_deg": 180.0,
            "elevation_deg": 30.0,
            "intensity": 1.0,
        },
        {"cmd": "set_terrain", "sun_azimuth": 180.0},
        {"cmd": "set_terrain", "sun_elevation": 30.0},
        {"cmd": "set_terrain", "sun_intensity": 1.0},
    ],
)
def test_manual_sun_success_clears_stale_observation_replay(command):
    from forge3d import viewer as viewer_module

    class Socket:
        def sendall(self, _request):
            pass

        def recv(self, _size):
            return b'{"ok": true}\n'

    class FakeViewer:
        is_running = True
        command = None

        def send_ipc(self, command):
            self.command = command

    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    handle = object.__new__(viewer_module.ViewerHandle)
    handle._socket = Socket()
    handle._send_command(command)
    viewer = FakeViewer()
    try:
        sky._set_active_viewer(viewer)
        assert viewer.command is None
    finally:
        sky._set_active_viewer(None)


def test_non_sun_terrain_success_preserves_observation_replay():
    from forge3d import viewer as viewer_module

    class Socket:
        def sendall(self, _request):
            pass

        def recv(self, _size):
            return b'{"ok": true}\n'

    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    handle = object.__new__(viewer_module.ViewerHandle)
    handle._socket = Socket()
    try:
        handle._send_command(
            {"cmd": "set_terrain", "zscale": 2.0, "sun_azimuth": None}
        )
        assert sky._has_observation_replay()
    finally:
        sky._clear_observation_replay()


def test_inactive_viewer_manual_sun_preserves_active_viewer_replay():
    from forge3d import viewer as viewer_module

    class Socket:
        def sendall(self, _request):
            pass

        def recv(self, _size):
            return b'{"ok": true}\n'

    class ActiveViewer:
        is_running = True

        def __init__(self):
            self.command = None

        def send_ipc(self, command):
            self.command = command

    inactive = object.__new__(viewer_module.ViewerHandle)
    inactive._socket = Socket()
    active = ActiveViewer()
    sky._set_active_viewer(active)
    try:
        sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        assert active.command["cmd"] == "set_observation"
        inactive._send_command(
            {"cmd": "lit_sun", "azimuth_deg": 180.0, "elevation_deg": 30.0}
        )
        assert sky._get_active_viewer() is active
        assert sky._has_observation_replay()
    finally:
        sky._set_active_viewer(None)
        sky._clear_observation_replay()


def test_rejected_manual_terrain_sun_preserves_observation_replay():
    from forge3d import viewer as viewer_module

    class Socket:
        def sendall(self, _request):
            pass

        def recv(self, _size):
            return b'{"ok": false, "error": "terrain rejected"}\n'

    sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    handle = object.__new__(viewer_module.ViewerHandle)
    handle._socket = Socket()
    try:
        with pytest.raises(viewer_module.ViewerError, match="terrain rejected"):
            handle._send_command(
                {
                    "cmd": "set_terrain_sun",
                    "azimuth_deg": 180.0,
                    "elevation_deg": 30.0,
                    "intensity": 1.0,
                }
            )
        assert sky._has_observation_replay()
    finally:
        sky._clear_observation_replay()


def test_observation_can_preseed_viewer_without_native(monkeypatch):
    monkeypatch.setattr(sky, "get_native_module", lambda: None)
    summary = sky.set_observation(
        datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
    )
    assert summary["status"] == "pending_viewer"
    with pytest.raises(ValueError):
        sky.set_observation(
            datetime(2051, 1, 1, tzinfo=timezone.utc), 0.0, 0.0
        )

    class ClosedViewer:
        is_running = False

    closed = ClosedViewer()
    sky._set_active_viewer(closed)
    try:
        summary = sky.set_observation(
            datetime(2026, 7, 26, 22, tzinfo=timezone.utc), 52.3676, 4.9041
        )
        assert summary["status"] == "pending_viewer"
    finally:
        sky._set_active_viewer(None)


def test_invalid_or_out_of_window_inputs_raise():
    with pytest.raises(ValueError):
        astro.body_position("sun", datetime(2026, 1, 1), 0.0, 0.0)
    with pytest.raises(ValueError):
        astro.body_position(
            "sun", datetime(2051, 1, 1, tzinfo=timezone.utc), 0.0, 0.0
        )


def _gmst_iau1982_seconds(jd_ut1: float) -> float:
    """Independent USNO/IAU-1982 (Aoki et al.) GMST, in seconds of time.

    USNO Circular 179 eq. 2.24 / Meeus eq. 12.4. Written here, in Python, so
    the reference is visibly independent of the Rust ERA-based IAU 2006 kernel
    it is used to check.
    """
    t = (jd_ut1 - 2_451_545.0) / 36_525.0
    seconds = (
        67_310.548_41
        + 3_164_400_184.812_866 * t
        + 0.093_104 * t * t
        - 6.2e-6 * t * t * t
    )
    return seconds % 86_400.0


def _bennett_refraction_arcminutes(apparent_altitude_deg: float) -> float:
    """Independent Bennett (1982) refraction, Meeus eq. 16.3, in arcminutes."""
    return 1.0 / math.tan(
        math.radians(apparent_altitude_deg + 7.31 / (apparent_altitude_deg + 4.4))
    )


def test_sidereal_time_matches_the_independent_usno_formula(capsys):
    """DoD 4a: GMST within 0.1 s of the cited USNO/IAU formula across 2000–2050.

    Two independence guarantees: the native metric sweeps the window against an
    IAU-1982 polynomial coded separately in Rust, and this test re-derives the
    same reference in Python and compares it to the *public* sidereal-time API.
    """
    metrics = get_native_module().astro_validation_metrics()
    assert metrics["gmst_samples"] > 1_000, metrics["gmst_samples"]
    assert metrics["gmst_max_seconds"] <= 0.1, metrics

    worst = 0.0
    worst_epoch = None
    checked = 0
    for year in range(2000, 2051, 2):
        for month, day, hour in ((1, 1, 0), (4, 15, 7), (7, 26, 22), (10, 2, 13)):
            when = datetime(year, month, day, hour, tzinfo=timezone.utc)
            gmst_deg, gast_deg = astro.sidereal_time(when)
            # JD(UT1) = JD(UTC) + (UT1 − UTC), and
            # UT1 − UTC = (TAI − UTC) + 32.184 s − ΔT.
            jd_utc = _julian_day_utc(when)
            tt_minus_utc = _tai_minus_utc_seconds(when) + 32.184
            jd_ut1 = jd_utc + (tt_minus_utc - astro.delta_t_seconds(when)) / 86_400.0
            reference = _gmst_iau1982_seconds(jd_ut1)
            actual = gmst_deg * 240.0  # degrees -> seconds of time
            delta = actual - reference
            delta -= round(delta / 86_400.0) * 86_400.0
            checked += 1
            if abs(delta) > worst:
                worst = abs(delta)
                worst_epoch = when.isoformat()
            # The equation of the equinoxes is at most ~1.1 s.
            equation_of_equinoxes = (gast_deg - gmst_deg) * 240.0
            equation_of_equinoxes -= round(equation_of_equinoxes / 86_400.0) * 86_400.0
            assert abs(equation_of_equinoxes) < 1.2, (when, equation_of_equinoxes)

    with capsys.disabled():
        print(
            f"\nGMST: native sweep {metrics['gmst_max_seconds']:.6f} s over "
            f"{metrics['gmst_samples']} samples; public-API cross-check "
            f"{worst:.6f} s over {checked} epochs (gate 0.1 s, worst {worst_epoch})"
        )
    assert checked >= 100
    assert worst <= 0.1, f"GMST off by {worst} s at {worst_epoch}"


#: IERS Bulletin C leap seconds. Written out here so the sidereal-time
#: cross-check does not lean on the same committed table the Rust side reads.
_LEAP_SECONDS = (
    (datetime(2000, 1, 1, tzinfo=timezone.utc), 32.0),
    (datetime(2006, 1, 1, tzinfo=timezone.utc), 33.0),
    (datetime(2009, 1, 1, tzinfo=timezone.utc), 34.0),
    (datetime(2012, 7, 1, tzinfo=timezone.utc), 35.0),
    (datetime(2015, 7, 1, tzinfo=timezone.utc), 36.0),
    (datetime(2017, 1, 1, tzinfo=timezone.utc), 37.0),
)


def _tai_minus_utc_seconds(when: datetime) -> float:
    offset = None
    for effective, value in _LEAP_SECONDS:
        if when >= effective:
            offset = value
    assert offset is not None, when
    return offset


def _julian_day_utc(when: datetime) -> float:
    year, month = when.year, when.month
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    day_fraction = (when.hour + (when.minute + when.second / 60.0) / 60.0) / 24.0
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + when.day
        + b
        - 1524.5
        + day_fraction
    )


def test_refraction_matches_bennett_at_five_degrees_apparent(capsys):
    """DoD 4b: refraction at 5° apparent altitude within 0.2′ of Bennett."""
    metrics = get_native_module().astro_validation_metrics()
    sidera = metrics["refraction_sidera_arcminutes"]
    bennett = metrics["refraction_bennett_arcminutes"]
    # Guard against the gate degenerating into a tautology: the two published
    # fits are different functions and must not be bit-identical.
    assert sidera != bennett
    assert abs(bennett - _bennett_refraction_arcminutes(5.0)) < 1e-9, bennett
    with capsys.disabled():
        print(
            f"\nrefraction at 5 deg apparent: SIDERA (Saemundsson, inverted) "
            f"{sidera:.4f}' vs Bennett {bennett:.4f}' -> "
            f"|delta| {abs(sidera - bennett):.4f}' (gate 0.2')"
        )
    assert metrics["refraction_error_arcminutes"] <= 0.2, metrics
    assert abs(sidera - bennett) <= 0.2

    # The public wrapper must agree with the metric, and refraction must fall
    # monotonically with altitude and vanish outside the fit's declared domain.
    assert astro.refraction_arcminutes(5.0) > astro.refraction_arcminutes(45.0) > 0.0
    # Sæmundsson at *true* altitude 0 is ~28.98′ (the familiar ~34′ figure is
    # Bennett at *apparent* altitude 0, a different argument).
    assert 28.0 < astro.refraction_arcminutes(0.0) < 30.0
    assert astro.refraction_arcminutes(89.5) == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_refraction_rejects_non_finite_altitude(value):
    with pytest.raises(ValueError, match="true_altitude_deg must be finite"):
        astro.refraction_arcminutes(value)


def test_reduction_ablation_gates(capsys):
    """DoD 5: each reduction is load-bearing, measured by removing exactly one.

    The precession ablation is measured on the *rendered catalog* — all 9,096
    stars through the real ``star_instances`` chain — not on a synthetic axis
    vector, because DoD 5 is stated about star positions. Precession is a
    rotation about the ecliptic pole, so a star near that pole barely moves;
    the min/median are reported alongside the max so the claim is not oversold.
    """
    metrics = get_native_module().astro_validation_metrics()
    assert metrics["precession_star_count"] == 9_096, metrics
    with capsys.disabled():
        print(
            f"\nablation, precession removed from the 2026-epoch star pipeline "
            f"({metrics['precession_star_count']} stars): "
            f"max {metrics['precession_arcminutes']:.3f}', "
            f"median {metrics['precession_median_arcminutes']:.3f}', "
            f"min {metrics['precession_min_arcminutes']:.3f}' (gate > 20')"
        )
        print(
            f"ablation, lunar parallax removed: "
            f"{metrics['lunar_parallax_arcminutes']:.3f}' (gate > 30')"
        )
    assert metrics["precession_arcminutes"] > 20.0, metrics
    # Not just one lucky star: half the rendered sky must move appreciably.
    assert metrics["precession_median_arcminutes"] > 15.0, metrics
    assert metrics["lunar_parallax_arcminutes"] > 30.0, metrics


def test_set_sun_handler_is_live_not_print_only():
    """DoD 6: the print-only `SetSunDirection` stub is fixed, not left lying.

    This must read the stub's own file. An earlier revision asserted only on
    the *downstream* files, so reverting `ipc_command.rs` to its pre-SIDERA
    body left every assertion satisfied while `set_sun` was a `println!` again.
    """
    ipc = (
        Path(__file__).parents[1] / "src" / "viewer" / "cmd" / "ipc_command.rs"
    ).read_text(encoding="utf-8")
    arm = ipc.split("ViewerCmd::SetSunDirection", 1)[1].split("ViewerCmd::SetObservation", 1)[0]
    assert "viewer.apply_manual_sun(" in arm
    # The exact shape of the old stub must be gone from this arm. (Other arms
    # in this file legitimately print, so the check is arm-scoped.)
    assert "let _dir" not in arm
    assert "println!" not in arm
    assert "Sun direction: azimuth" not in ipc

    frame_setup = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "render"
        / "main_loop"
        / "frame_setup.rs"
    ).read_text(encoding="utf-8")
    # The sky pass must read the observation-driven sun direction, and the sun
    # disc/scattering must fade on the declared ramp rather than stepping to
    # zero at altitude 0 (which pops a frame before the sky starts darkening).
    assert "self.sky_sun_direction_ws" in frame_setup
    assert "SUN_FADE_START_DEG" in frame_setup and "SUN_FADE_END_DEG" in frame_setup
    assert "smoothstep(SUN_FADE_START_DEG, SUN_FADE_END_DEG, solar_altitude_deg)" in frame_setup
    helpers = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "state"
        / "viewer_helpers"
        / "core.rs"
    ).read_text(encoding="utf-8")
    assert "lit_sun_direction_ws[1] > 0.0" in helpers
    assert "crate::astro::night::horizontal_direction" in helpers
    assert "self.lit_sun_intensity = 1.0" in helpers
    scene_command = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "cmd"
        / "scene_command.rs"
    ).read_text(encoding="utf-8")
    lit_sun = scene_command.split("ViewerCmd::SetLitSun", 1)[1].split(
        "ViewerCmd::SetLitIbl", 1
    )[0]
    assert "viewer.sync_terrain_sun_to_lit()" in lit_sun
    terrain_command = (
        Path(__file__).parents[1]
        / "src"
        / "viewer"
        / "cmd"
        / "terrain_command.rs"
    ).read_text(encoding="utf-8")
    load_handler = terrain_command.split("ViewerCmd::LoadTerrain", 1)[1].split(
        "ViewerCmd::SetTerrainCamera", 1
    )[0]
    assert "viewer.sync_terrain_sun_to_lit()" in load_handler

    viewer = (Path(__file__).parents[1] / "python" / "forge3d" / "viewer.py").read_text(
        encoding="utf-8"
    )
    replay = viewer.split("sky._set_active_viewer(handle)", 1)
    assert "handle.close()" in replay[1]
    blocking = viewer.split("def open_viewer(", 1)[1]
    assert "handle = open_viewer_async(" in blocking


def test_sidera_reference_probe_requires_physical_nvidia_vulkan():
    from scripts import terrain_ci_probe

    physical_nvidia = {
        "status": "ok",
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "name": "NVIDIA GeForce RTX 3070",
        "vendor": 0x10DE,
        "software_fallback": False,
    }
    assert terrain_ci_probe._adapter_is_ci_safe(
        physical_nvidia, required_backend="vulkan", require_nvidia=True
    )
    assert not terrain_ci_probe._adapter_is_ci_safe(
        {**physical_nvidia, "name": "NVIDIA Paravirtual device"},
        required_backend="vulkan",
        require_nvidia=True,
    )
    assert not terrain_ci_probe._adapter_is_ci_safe(
        {**physical_nvidia, "device_type": "IntegratedGpu"},
        required_backend="vulkan",
        require_nvidia=True,
    )
    assert not terrain_ci_probe._adapter_is_ci_safe(
        {**physical_nvidia, "backend": "Metal"},
        required_backend="vulkan",
        require_nvidia=True,
    )
    for unsafe in (
        {**physical_nvidia, "software_fallback": True},
        {
            key: value
            for key, value in physical_nvidia.items()
            if key != "software_fallback"
        },
        {**physical_nvidia, "vendor": 0x1002},
        {**physical_nvidia, "name": "AMD Radeon RX 7900 XT"},
    ):
        assert not terrain_ci_probe._adapter_is_ci_safe(
            unsafe, required_backend="vulkan", require_nvidia=True
        )


def test_snapshot_sky_uses_dedicated_camera_and_motion_coverage_union():
    snapshot = (
        ROOT / "src" / "viewer" / "render" / "main_loop" / "snapshot_sky.rs"
    ).read_text(encoding="utf-8")
    allocator = (
        ROOT
        / "src"
        / "viewer"
        / "state"
        / "viewer_helpers"
        / "snapshot_sky.rs"
    ).read_text(encoding="utf-8")
    secondary = (
        ROOT / "src" / "viewer" / "render" / "main_loop" / "secondary.rs"
    ).read_text(encoding="utf-8")
    motion = (
        ROOT / "src" / "viewer" / "terrain" / "render" / "motion_blur.rs"
    ).read_text(encoding="utf-8")
    union = (
        ROOT / "src" / "viewer" / "terrain" / "motion_blur_depth.rs"
    ).read_text(encoding="utf-8")
    assert "frame.projection(width, height)" in snapshot
    assert "viewer.sky.snapshot.camera" in allocator
    assert "cache.camera" in snapshot and "self.sky_camera" not in snapshot
    assert "present_snapshot_sky" in secondary
    assert "self.snapshot_depth_texture = Some(coverage_depth)" in motion
    assert "CompareFunction::Less" in union and "texture_depth_2d" in union


def test_sidera_assets_stay_under_two_mebibytes_and_match_their_manifest():
    assets = ROOT / "assets" / "astro"
    total = sum(path.stat().st_size for path in assets.iterdir() if path.is_file())
    total += (ROOT / "tests" / "data" / "horizons_vectors.dat").stat().st_size
    assert total <= 2 * 1024**2, total
    manifest = load_toml(assets / "MANIFEST.toml")
    declared = {entry["path"]: entry for entry in manifest["asset"]}
    # Every declared asset must exist and hash to its manifest entry, and every
    # committed asset must be declared — provenance with no unlisted payload.
    on_disk = {path.name for path in assets.iterdir() if path.is_file()} - {
        "MANIFEST.toml",
        manifest["third_party_notices"],
    }
    assert on_disk == set(declared), (on_disk, set(declared))
    notices = (assets / manifest["third_party_notices"]).read_text(encoding="utf-8")
    # moon_terms.bin is MIT and ships inside the wheel, so its permission
    # notice must travel with it.
    assert "MIT License" in notices and "Permission is hereby granted" in notices
    for name, entry in declared.items():
        actual = hashlib.sha256((assets / name).read_bytes()).hexdigest()
        assert actual == entry["sha256"], (
            f"{name}: manifest says {entry['sha256']}, bytes hash to {actual}"
        )
        for field in ("source", "source_url", "license"):
            assert entry.get(field), f"{name} has no declared {field}"
    vsop = declared["vsop87d.bin"]
    retained = sum(
        sum(counts)
        for body in vsop["term_counts"].values()
        for counts in body.values()
    )
    assert retained == 14_793 < 25_659
    assert vsop["max_omitted_longitude_radians_per_body"] <= 5.0e-8
    assert vsop["max_omitted_latitude_radians_per_body"] <= 5.0e-8
    assert vsop["max_omitted_radius_au_per_body"] <= 2.0e-8


def test_horizons_oracle_manifest_locks_payload_and_generation_settings():
    manifest = load_toml(DATA.with_name("horizons_vectors.MANIFEST.toml"))
    payload = DATA.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]
    assert manifest["ephemeris"] == "DE441"
    assert manifest["eop_files"].startswith("eop.")
    for setting in ("GEODETIC", "ICRF", "AIRLESS", "DE441", "4/10/13/20/24/30/49"):
        assert setting in manifest["settings"]

    lines = DATA.read_text(encoding="ascii").splitlines()
    ordinary = [line for line in lines if line and not line.startswith(("#", "@"))]
    assert len(ordinary) == manifest["vectors"] == 280
    assert sum(line.startswith("@moon_phase ") for line in lines) == manifest["moon_phase_vectors"]
    assert sum(line.startswith("@moon_window ") for line in lines) == manifest["moon_window_vectors"]
    assert (
        sum(line.startswith("@delta_t_midmonth ") for line in lines)
        == manifest["delta_t_midmonth_vectors"]
    )
    header = "\n".join(lines[:12])
    assert f"Ephemeris: {manifest['ephemeris']}" in header
    assert f"EOP: {manifest['eop_files']}" in header


def test_manifest_truncation_budgets_are_gates_not_prose(capsys):
    """The declared per-asset error budgets must bound the measured deviations.

    ``MANIFEST.toml`` publishes a maximum Horizons deviation per theory asset.
    Without this test those numbers are decoration: a thinner term table could
    drift right up to the looser DoD threshold while the manifest kept
    advertising the old, tighter figure — an undeclared truncation, which the
    task's operating rules forbid.
    """
    from datetime import datetime

    manifest = load_toml(ROOT / "assets" / "astro" / "MANIFEST.toml")
    declared = {entry["path"]: entry for entry in manifest["asset"]}
    budgets = {
        "vsop87d.bin": (
            "max_horizons_position_error_arcsec",
            ["sun", "mercury", "venus", "mars", "jupiter", "saturn"],
        ),
        "moon_terms.bin": ("max_horizons_topocentric_position_error_arcsec", ["moon"]),
    }
    data = (ROOT / "tests" / "data" / "horizons_vectors.dat").read_text(encoding="ascii")
    measured = {}
    for line in data.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] in {"@moon_phase", "@delta_t_midmonth"}:
            continue
        if fields[0] == "@moon_window":
            _, _, iso, latitude, longitude, height, azimuth, altitude = fields
            body = "moon"
        else:
            _, iso, body, latitude, longitude, height, azimuth, altitude, *_ = fields
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        actual = astro.body_position(
            body, when, float(latitude), float(longitude), height_m=float(height)
        )
        deviation = _separation_arcsec(actual, (float(azimuth), float(altitude)))
        measured[body] = max(measured.get(body, 0.0), deviation)

    with capsys.disabled():
        print("\nmanifest declared budgets vs measured maxima:")
    for asset, (field, bodies) in budgets.items():
        budget = float(declared[asset][field])
        worst = max(measured[body] for body in bodies)
        with capsys.disabled():
            print(f"  {asset:16s} {field} = {budget:6.2f}\"  measured {worst:6.3f}\"")
        assert worst <= budget, (
            f"{asset} declares {field} = {budget}\" but the committed oracle "
            f"measures {worst:.3f}\" — update the declaration or the theory"
        )
