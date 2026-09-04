"""Regenerate SIDERA's compact public-domain astronomy assets."""

from __future__ import annotations

import argparse
import ast
import calendar
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import struct
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASTRO = ROOT / "assets" / "astro"
VSOP_URL = "https://ftp.imcce.fr/pub/ephem/planets/vsop87/"
VSOP_FILES = ("mer", "ven", "ear", "mar", "jup", "sat")
VSOP_HEADER = re.compile(r"VARIABLE\s+(\d).*T\*\*(\d)\s+(\d+)\s+TERMS")
# A conservative contribution budget over |t| <= 0.051 VSOP millennia
# (2000-01-01 through 2050-12-31).  Terms are removed smallest-bound first,
# across each body's complete coordinate series.  The sum of every omitted
# |A*t^power| is therefore bounded by these values before the Horizons gate is
# applied.  L/B are radians; R is AU.
VSOP_MAX_ABS_T = 0.051
VSOP_OMITTED_CONTRIBUTION_BUDGET = (5.0e-8, 5.0e-8, 2.0e-8)
MOON_URL = "https://raw.githubusercontent.com/commenthol/astronomia/master/src/moonposition.js"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
BSC_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
    "-source=V%2F50%2Fcatalog&-out=HR,RAJ2000,DEJ2000,Vmag,B-V&-out.max=unlimited"
)
MOON_TEXTURE_URL = (
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_2k.jpg"
)
SITES = (
    ("amsterdam", 52.3676, 4.9041, 0.0),
    ("mauna_kea", 19.8207, -155.4681, 4.205),
    ("sydney", -33.8688, 151.2093, 0.058),
    ("quito", -0.1807, -78.4678, 2.85),
    ("longyearbyen", 78.2232, 15.6469, 0.028),
)
EPOCHS = (
    "2000-01-01 00:00",
    "2005-06-21 12:00",
    "2012-03-20 06:00",
    "2020-12-21 18:00",
    "2026-07-26 22:00",
    "2035-09-23 03:00",
    "2044-02-29 12:00",
    "2050-12-31 23:00",
)
LUNAR_SWEEP_SITE = ("tromso", 69.6492, 18.9553, 0.010)
LUNAR_SWEEP_START = "2000-01-01"
LUNAR_SWEEP_STOP = "2050-12-31"
LUNAR_SWEEP_STEP = "30 d"
BODIES = (
    ("sun", "10"),
    ("moon", "301"),
    ("mercury", "199"),
    ("venus", "299"),
    ("mars", "499"),
    ("jupiter", "599"),
    ("saturn", "699"),
)


def generate_catalog() -> None:
    expected_count = 9_096
    source = urllib.request.urlopen(BSC_URL, timeout=60).read().decode("utf-8")
    stars = []
    for line in source.splitlines():
        fields = line.split("\t")
        if len(fields) != 5 or not fields[0].strip().isdigit():
            continue
        ra, dec, magnitude, bv = (field.strip() for field in fields[1:])
        if not ra or not dec or not magnitude:
            continue
        hours, minutes, seconds = map(float, ra.split())
        sign = -1.0 if dec.startswith("-") else 1.0
        degrees, arcminutes, arcseconds = map(float, dec[1:].split())
        stars.append(
            (
                15.0 * (hours + minutes / 60.0 + seconds / 3_600.0),
                sign * (degrees + arcminutes / 60.0 + arcseconds / 3_600.0),
                float(magnitude),
                float(bv) if bv else float("nan"),
            )
        )
    if len(stars) != expected_count:
        raise RuntimeError(f"expected {expected_count} usable BSC rows, received {len(stars)}")

    output = ASTRO / "bright_stars.bin"
    with output.open("wb") as stream:
        stream.write(b"F3DBSC01")
        stream.write(struct.pack("<I", len(stars)))
        for star in stars:
            stream.write(struct.pack("<ffff", *star))
    print(f"{output}: {len(stars)} stars, {output.stat().st_size} bytes")


def generate_moon_texture() -> None:
    from PIL import Image

    source = urllib.request.urlopen(MOON_TEXTURE_URL, timeout=60).read()
    image = Image.open(io.BytesIO(source)).convert("L").resize((128, 64), Image.Resampling.LANCZOS)
    output = ASTRO / "moon_albedo.bin"
    output.write_bytes(b"F3DMAP01" + struct.pack("<II", *image.size) + image.tobytes())
    print(f"{output}: {image.width}x{image.height}, {output.stat().st_size} bytes")


def generate_vsop() -> None:
    output = ASTRO / "vsop87d.bin"
    sections: list[tuple[int, int, list[tuple[float, float, float]]]] = []
    bodies: list[tuple[str, list[int]]] = []
    for body in VSOP_FILES:
        text = urllib.request.urlopen(f"{VSOP_URL}VSOP87D.{body}", timeout=60).read().decode("ascii")
        body_sections: list[int] = []
        lines = iter(text.splitlines())
        for line in lines:
            match = VSOP_HEADER.search(line)
            if not match:
                continue
            variable, power, count = map(int, match.groups())
            terms = []
            for _ in range(count):
                values = next(lines).split()
                terms.append(tuple(map(float, values[-3:])))
            body_sections.append(len(sections))
            sections.append((variable - 1, power, terms))
        bodies.append((body, body_sections))

    kept_indices: dict[int, set[int]] = {
        index: set(range(len(terms)))
        for index, (_, _, terms) in enumerate(sections)
    }
    omitted_bounds: dict[str, list[float]] = {}
    for body, body_sections in bodies:
        body_bounds = []
        for coordinate, budget in enumerate(VSOP_OMITTED_CONTRIBUTION_BUDGET):
            candidates = []
            for section_index in body_sections:
                variable, power, terms = sections[section_index]
                if variable != coordinate:
                    continue
                for term_index, (amplitude, _, _) in enumerate(terms):
                    contribution = abs(amplitude) * VSOP_MAX_ABS_T**power
                    candidates.append((contribution, section_index, term_index))
            omitted = 0.0
            for contribution, section_index, term_index in sorted(candidates):
                # Every published power remains represented, even when all of
                # its terms fall below the aggregate contribution budget.
                if len(kept_indices[section_index]) <= 1 or omitted + contribution > budget:
                    continue
                kept_indices[section_index].remove(term_index)
                omitted += contribution
            body_bounds.append(omitted)
        omitted_bounds[body] = body_bounds

    sections = [
        (variable, power, [term for i, term in enumerate(terms) if i in kept_indices[index]])
        for index, (variable, power, terms) in enumerate(sections)
    ]

    with output.open("wb") as stream:
        stream.write(b"F3DVSOP1")
        stream.write(struct.pack("<II", len(bodies), len(sections)))
        for name, indices in bodies:
            stream.write(name.encode("ascii"))
            stream.write(struct.pack("<B", len(indices)))
            stream.write(struct.pack(f"<{len(indices)}I", *indices))
        for variable, power, terms in sections:
            stream.write(struct.pack("<BBHI", variable, power, 0, len(terms)))
            for term in terms:
                stream.write(struct.pack("<ddd", *term))
    total_terms = sum(len(terms) for _, _, terms in sections)
    print(f"{output}: {output.stat().st_size} bytes, {total_terms} retained terms")
    for body, body_sections in bodies:
        counts = {coordinate: [] for coordinate in range(3)}
        for section_index in body_sections:
            variable, power, terms = sections[section_index]
            counts[variable].append((power, len(terms)))
        print(f"  {body}: L/B/R={counts}; omitted bounds={omitted_bounds[body]}")


def generate_moon() -> None:
    source = urllib.request.urlopen(MOON_URL, timeout=60).read().decode("utf-8")
    longitude_radius = ast.literal_eval(
        "[" + re.search(r"const ta = \[(.*?)\n  \]", source, re.DOTALL).group(1) + "]"
    )
    latitude = ast.literal_eval(
        "[" + re.search(r"const tb = \[(.*?)\n  \]", source, re.DOTALL).group(1) + "]"
    )
    output = ASTRO / "moon_terms.bin"
    with output.open("wb") as stream:
        stream.write(b"F3DMOON1")
        stream.write(struct.pack("<II", len(longitude_radius), len(latitude)))
        for d, m, mp, f, longitude, radius in longitude_radius:
            stream.write(struct.pack("<bbbbii", d, m, mp, f, int(longitude), int(radius)))
        for d, m, mp, f, coefficient in latitude:
            stream.write(struct.pack("<bbbbi", d, m, mp, f, int(coefficient)))
    print(f"{output}: {output.stat().st_size} bytes")


def _horizons_result(parameters: dict[str, str]) -> tuple[str, str, str]:
    url = HORIZONS_URL + "?" + urllib.parse.urlencode(parameters)
    result = json.load(urllib.request.urlopen(url, timeout=60))["result"]
    preamble, remainder = result.split("$$SOE", 1)
    block = remainder.split("$$EOE", 1)[0]
    ephemerides = set(re.findall(r"source:\s*(DE\d+)", preamble))
    if ephemerides != {"DE441"}:
        raise RuntimeError(f"expected Horizons DE441, received {sorted(ephemerides)}")
    eop_match = re.search(r"^EOP file\s*:\s*(\S+)", preamble, re.MULTILINE)
    return block, "DE441", eop_match.group(1) if eop_match else "unreported"


def _atomic_write_oracle(data: str, manifest: str) -> None:
    directory = ROOT / "tests" / "data"
    data_path = directory / "horizons_vectors.dat"
    manifest_path = directory / "horizons_vectors.MANIFEST.toml"
    temporary_paths = []
    try:
        for target, content in ((data_path, data), (manifest_path, manifest)):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="ascii", newline="\n", dir=directory, delete=False
            ) as stream:
                stream.write(content)
                temporary_paths.append((Path(stream.name), target))
        for temporary, target in temporary_paths:
            os.replace(temporary, target)
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)


def _parse_horizons_calendar(value: str) -> dt.datetime:
    value = value.strip()
    for format_string in ("%Y-%b-%d %H:%M:%S.%f", "%Y-%b-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(value, format_string)
        except ValueError:
            pass
    raise ValueError(f"unexpected Horizons calendar value: {value!r}")


def generate_horizons() -> None:
    rows = []
    served_ephemerides: set[str] = set()
    served_eop_files: set[str] = set()
    for site, latitude, longitude, height_km in SITES:
        for body, command in BODIES:
            parameters = {
                "format": "json",
                "COMMAND": f"'{command}'",
                "OBJ_DATA": "NO",
                "MAKE_EPHEM": "YES",
                "EPHEM_TYPE": "OBSERVER",
                "CENTER": "'coord@399'",
                "COORD_TYPE": "GEODETIC",
                "SITE_COORD": f"'{longitude},{latitude},{height_km}'",
                "TLIST": " ".join(f"'{epoch}'" for epoch in EPOCHS),
                "TLIST_TYPE": "CAL",
                "QUANTITIES": "'4,10,13,20,24,30,49'",
                "ANG_FORMAT": "DEG",
                "APPARENT": "AIRLESS",
                "CSV_FORMAT": "YES",
                "REF_SYSTEM": "ICRF",
                "CAL_FORMAT": "CAL",
                "TIME_DIGITS": "SECONDS",
            }
            block, ephemeris, eop_file = _horizons_result(parameters)
            served_ephemerides.add(ephemeris)
            served_eop_files.add(eop_file)
            body_rows = list(csv.reader(io.StringIO(block.strip())))
            if len(body_rows) != len(EPOCHS):
                raise RuntimeError(f"Horizons returned {len(body_rows)} rows for {site}/{body}")
            for epoch, values in zip(EPOCHS, body_rows):
                numeric = [value.strip() for value in values[3:]]
                rows.append(
                    (
                        site,
                        epoch.replace(" ", "T") + ":00Z",
                        body,
                        latitude,
                        longitude,
                        height_km * 1_000.0,
                        *numeric,
                    )
                )

    phase_parameters = {
        "format": "json",
        "COMMAND": "'301'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "TLIST": " ".join(f"'{epoch}'" for epoch in EPOCHS),
        "TLIST_TYPE": "CAL",
        "QUANTITIES": "'10,13,20,24'",
        "CSV_FORMAT": "YES",
        "CAL_FORMAT": "CAL",
        "TIME_DIGITS": "SECONDS",
    }
    phase_block, ephemeris, eop_file = _horizons_result(phase_parameters)
    served_ephemerides.add(ephemeris)
    served_eop_files.add(eop_file)
    phase_rows = list(csv.reader(io.StringIO(phase_block.strip())))

    sweep_site, sweep_latitude, sweep_longitude, sweep_height_km = LUNAR_SWEEP_SITE
    sweep_parameters = {
        "format": "json",
        "COMMAND": "'301'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'coord@399'",
        "COORD_TYPE": "GEODETIC",
        "SITE_COORD": f"'{sweep_longitude},{sweep_latitude},{sweep_height_km}'",
        "START_TIME": f"'{LUNAR_SWEEP_START}'",
        "STOP_TIME": f"'{LUNAR_SWEEP_STOP}'",
        "STEP_SIZE": f"'{LUNAR_SWEEP_STEP}'",
        "QUANTITIES": "'4'",
        "ANG_FORMAT": "DEG",
        "APPARENT": "AIRLESS",
        "CSV_FORMAT": "YES",
        "REF_SYSTEM": "ICRF",
        "CAL_FORMAT": "CAL",
        "TIME_DIGITS": "SECONDS",
    }
    sweep_block, ephemeris, eop_file = _horizons_result(sweep_parameters)
    served_ephemerides.add(ephemeris)
    served_eop_files.add(eop_file)
    sweep_rows = list(csv.reader(io.StringIO(sweep_block.strip())))

    midmonth_parameters = {
        "format": "json",
        "COMMAND": "'10'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "START_TIME": "'2000-01-15'",
        "STOP_TIME": "'2050-12-15'",
        "STEP_SIZE": "'1 month'",
        "QUANTITIES": "'30,49'",
        "CSV_FORMAT": "YES",
        "CAL_FORMAT": "CAL",
        "TIME_DIGITS": "SECONDS",
    }
    midmonth_block, ephemeris, eop_file = _horizons_result(midmonth_parameters)
    served_ephemerides.add(ephemeris)
    served_eop_files.add(eop_file)
    midmonth_rows = list(csv.reader(io.StringIO(midmonth_block.strip())))

    # Every quantitative claim in the header is derived from what was actually
    # requested and actually served, so re-running with a different ephemeris,
    # site list or epoch list cannot leave a stale provenance record behind.
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    if served_ephemerides != {"DE441"}:
        raise RuntimeError(f"mixed Horizons ephemerides: {sorted(served_ephemerides)}")
    ephemeris = next(iter(served_ephemerides))
    eop_files = ",".join(sorted(served_eop_files))
    header = (
        f"# JPL Horizons topocentric observer vectors, generated {generated}.\n"
        f"# Ephemeris: {ephemeris}; EOP: {eop_files}; center: coord@399; frame: ICRF; apparent: AIRLESS.\n"
        "# Quantities: 4,10,13,20,24,30,49; angular format: decimal degrees; CSV.\n"
        "# Generator: tools/generate_sidera_assets.py --horizons\n"
        f"# {len(SITES)} WGS84 sites x {len(EPOCHS)} UTC epochs = "
        f"{len(SITES) * len(EPOCHS)} epoch/site combinations x {len(BODIES)} bodies "
        f"= {len(rows)} vectors.\n"
        "# columns: site utc body lat_deg lon_deg height_m az_deg alt_deg illum_percent\n"
        "#          angular_diameter_arcsec distance_au range_rate_km_s phase_angle_deg\n"
        "#          tdb_minus_ut_seconds ut1_minus_utc_seconds\n"
        "# @moon_phase rows are geocentric: utc illum_percent diameter_arcsec distance_au phase_deg.\n"
        "# @moon_window rows are a 30-day Tromso sweep: utc lat lon height_m az_deg alt_deg.\n"
        "# @delta_t_midmonth rows are independent monthly: utc TT_minus_UT1_seconds.\n"
    )
    stream = io.StringIO()
    stream.write(header)
    for row in rows:
        stream.write(" ".join(map(str, row)).rstrip() + "\n")
    for epoch, values in zip(EPOCHS, phase_rows):
        numeric = [value.strip() for value in values[3:]]
        stream.write(
            "@moon_phase "
            + epoch.replace(" ", "T")
            + ":00Z "
            + " ".join(numeric[0:1] + numeric[1:2] + numeric[2:3] + numeric[4:5])
            + "\n"
        )
    for values in sweep_rows:
        epoch = _parse_horizons_calendar(values[0])
        stream.write(
            f"@moon_window {sweep_site} {epoch:%Y-%m-%dT%H:%M:%SZ} "
            f"{sweep_latitude} {sweep_longitude} {sweep_height_km * 1_000.0} "
            f"{values[3].strip()} {values[4].strip()}\n"
        )
    for values in midmonth_rows:
        epoch = _parse_horizons_calendar(values[0])
        delta_t = float(values[3]) - float(values[4])
        stream.write(f"@delta_t_midmonth {epoch:%Y-%m-%dT%H:%M:%SZ} {delta_t:.6f}\n")
    data = stream.getvalue()
    digest = hashlib.sha256(data.encode("ascii")).hexdigest()
    manifest = (
        "format_version = 2\n"
        "path = \"horizons_vectors.dat\"\n"
        "source = \"NASA/JPL Horizons API\"\n"
        "source_url = \"https://ssd-api.jpl.nasa.gov/doc/horizons.html\"\n"
        "license = \"NASA/JPL public ephemeris data\"\n"
        "generation_command = \"python tools/generate_sidera_assets.py --horizons\"\n"
        "settings = \"coord@399, GEODETIC, ICRF, AIRLESS, DE441, quantities 4/10/13/20/24/30/49\"\n"
        f"generated_utc = \"{generated}\"\n"
        f"ephemeris = \"{ephemeris}\"\n"
        f"eop_files = \"{eop_files}\"\n"
        f"combinations = {len(SITES) * len(EPOCHS)}\n"
        f"unique_epochs = {len(EPOCHS)}\n"
        f"vectors = {len(rows)}\n"
        f"moon_phase_vectors = {len(phase_rows)}\n"
        f"moon_window_vectors = {len(sweep_rows)}\n"
        f"delta_t_midmonth_vectors = {len(midmonth_rows)}\n"
        f"sha256 = \"{digest}\"\n"
    )
    _atomic_write_oracle(data, manifest)
    output = ROOT / "tests" / "data" / "horizons_vectors.dat"
    print(
        f"{output}: {len(rows)} body, {len(sweep_rows)} lunar-window, "
        f"{len(midmonth_rows)} delta-T vectors, {output.stat().st_size} bytes"
    )


def generate_delta_t() -> None:
    parameters = {
        "format": "json",
        "COMMAND": "'10'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "START_TIME": "'2000-01-01'",
        "STOP_TIME": "'2051-01-01'",
        "STEP_SIZE": "'1 month'",
        "QUANTITIES": "'30,49'",
        "CSV_FORMAT": "YES",
        "CAL_FORMAT": "JD",
    }
    block, ephemeris, eop_file = _horizons_result(parameters)
    dut1 = [float(row[4]) for row in csv.reader(io.StringIO(block.strip()))]
    dates = []
    year, month = 2000, 1
    while (year, month) <= (2051, 1):
        dates.append(dt.date(year, month, 1))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    if len(dates) != len(dut1):
        raise RuntimeError(f"Horizons returned {len(dut1)} ΔT rows for {len(dates)} dates")

    def decimal_year(date: dt.date) -> float:
        days = 366 if calendar.isleap(date.year) else 365
        return date.year + (date.timetuple().tm_yday - 1) / days

    def tai_minus_utc(date: dt.date) -> int:
        offset = 32
        for effective, value in (
            (dt.date(2006, 1, 1), 33),
            (dt.date(2009, 1, 1), 34),
            (dt.date(2012, 7, 1), 35),
            (dt.date(2015, 7, 1), 36),
            (dt.date(2017, 1, 1), 37),
        ):
            if date >= effective:
                offset = value
        return offset

    values = [tai_minus_utc(date) + 32.184 - delta for date, delta in zip(dates, dut1)]
    output = ASTRO / "delta_t_fit.dat"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="ascii", newline="\n", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write("# Piecewise-linear TT-UT1 fit to monthly JPL Horizons/IERS EOP nodes.\n")
        stream.write(
            f"# Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d}; "
            f"ephemeris {ephemeris}; EOP {eop_file}.\n"
        )
        stream.write("# columns: start_year end_year origin_year c0_seconds c1_seconds_per_year\n")
        for date_a, date_b, value_a, value_b in zip(dates, dates[1:], values, values[1:]):
            start, end = decimal_year(date_a), decimal_year(date_b)
            slope = (value_b - value_a) / (end - start)
            stream.write(f"{start:.12f} {end:.12f} {start:.12f} {value_a:.9f} {slope:.9f}\n")
    os.replace(temporary, output)
    print(f"{output}: {len(values) - 1} monthly segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vsop", action="store_true")
    parser.add_argument("--moon", action="store_true")
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--moon-texture", action="store_true")
    parser.add_argument("--horizons", action="store_true")
    parser.add_argument("--delta-t", action="store_true")
    options = parser.parse_args()
    if options.vsop:
        generate_vsop()
    if options.moon:
        generate_moon()
    if options.catalog:
        generate_catalog()
    if options.moon_texture:
        generate_moon_texture()
    if options.horizons:
        generate_horizons()
    if options.delta_t:
        generate_delta_t()
