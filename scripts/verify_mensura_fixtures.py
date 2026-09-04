"""Verify the committed, network-independent MENSURA fixture artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    "assets/geoid/egm96_n120.bin": (
        236_168,
        "b640e9dcefd1040f0b184a101e1eab2740486a85680a560080ec091eab796fe4",
    ),
    "assets/geoid/mars_areoid_n179.bin": (
        260_664,
        "a36346d659c4533ab8a3280069b9011e3e4ce650a23fa7070a3d6cadcda510b9",
    ),
    "assets/geoid/mars_areoid_n179.manifest.json": (
        1_461,
        "40a1f3ce1daace6bfddff75bdc6db3e1f4451c6241f7d00387644165855b215b",
    ),
    "tests/data/geodtest_subset.dat": (
        9_854,
        "8dbea72eabe48c0da21497e1f140090a234382ad9ba298ed111b9566feebcdf3",
    ),
    "tests/data/egm96_test_values.txt": (
        1_980,
        "b7a6f64b2f6ea17918b82efdc19bb2b8dd3941e6c58d28fb134b158b14e4958b",
    ),
    "tests/data/mars_areoid_reference.txt": (
        1_839,
        "8e321576ed85712a31f4ed22012f2cad14133af31ad927b98dcc9fcd969bd3d7",
    ),
}

GEODTEST_SHORT_SHA256 = "31d376e8158f7af26277d887d7a1e7726e14d172db5a848a1730dd4886ab6121"
GEODTEST_ROW_INDICES = (
    0, 121, 242, 363, 484, 605, 726, 847, 968, 1089, 1210, 1331, 1452,
    1573, 1694, 1815, 1936, 2841, 3110, 3231, 3352, 3473, 3594, 3715,
    3836, 3957, 4078, 4199, 4320, 4441, 4562, 4683, 4804, 4925, 6046,
    6167, 6288, 6409, 6531, 6652, 6773, 6895, 7016, 7138, 7260, 7381,
    7504, 7625, 7746, 7867,
)
EGM96_SPHERICAL_SHA256 = "1f21ab8151c1b9fe25f483a4f6b78acdbf5306daf923725017b83d87a5f33472"
EGM96_MEMBER_SHA256 = "1e5e6c30343989b8e2eda0bb96bde06ef05981eec69934d6e44aace4f0d6a9d5"
CORRCOEF_MEMBER_SHA256 = "72a3dbcf1c5cd60602770e38b5145d09b5ba5c47e72625b2159bdf48139d8b2d"


def verify_geodtest_source(path: Path) -> None:
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == GEODTEST_SHORT_SHA256
    with gzip.open(path, "rt", encoding="ascii") as source:
        rows = [line for line in source.read().splitlines() if line and not line.startswith("#")]
    selected = [rows[index] for index in GEODTEST_ROW_INDICES]
    assert all(float(row.split()[7]) <= 179.0 for row in selected)
    fixture = [
        line
        for line in (ROOT / "tests/data/geodtest_subset.dat").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert selected == fixture
    print(f"verified {path}: upstream SHA-256 and deterministic 50-row extraction")


def coefficient_pairs(data: bytes, minimum_degree: int) -> list[tuple[float, float]]:
    pairs = []
    for line in data.decode("ascii").splitlines():
        fields = line.replace("D", "E").split()
        degree, _order = map(int, fields[:2])
        if minimum_degree <= degree <= 120:
            pairs.append((float(fields[2]), float(fields[3])))
    return pairs


def verify_egm96_source(path: Path) -> None:
    archive_data = path.read_bytes()
    assert hashlib.sha256(archive_data).hexdigest() == EGM96_SPHERICAL_SHA256
    with zipfile.ZipFile(path) as archive:
        egm96 = archive.read("EGM96")
        corrcoef = archive.read("CORRCOEF")
    assert hashlib.sha256(egm96).hexdigest() == EGM96_MEMBER_SHA256
    assert hashlib.sha256(corrcoef).hexdigest() == CORRCOEF_MEMBER_SHA256
    expected = coefficient_pairs(egm96, 2) + coefficient_pairs(corrcoef, 0)
    binary = (ROOT / "assets/geoid/egm96_n120.bin").read_bytes()
    header = struct.unpack_from("<8sIIII", binary)
    assert header == (b"F3DEGM96", 1, 120, 7378, 7381)
    assert list(struct.iter_unpack("<dd", binary[24:])) == expected
    print(f"verified {path}: upstream SHA-256 and deterministic n=120 coefficient extraction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--egm96-spherical", type=Path)
    parser.add_argument("--geodtest-short", type=Path)
    args = parser.parse_args()
    for relative, (expected_size, expected_sha256) in FIXTURES.items():
        path = ROOT / relative
        data = path.read_bytes()
        canonical = data if path.suffix == ".bin" else data.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(canonical).hexdigest()
        assert len(canonical) == expected_size, (relative, len(canonical), expected_size)
        assert digest == expected_sha256, (relative, digest, expected_sha256)
        print(f"verified {relative}: {len(canonical)} bytes sha256={digest}")
    if args.geodtest_short:
        verify_geodtest_source(args.geodtest_short)
    if args.egm96_spherical:
        verify_egm96_source(args.egm96_spherical)


if __name__ == "__main__":
    main()
