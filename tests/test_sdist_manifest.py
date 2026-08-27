from __future__ import annotations

import shlex
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.in"


def test_sdist_manifest_inputs_and_archive_members(tmp_path: Path) -> None:
    literal_includes = []
    for raw_line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = shlex.split(raw_line, comments=True)
        if fields[:1] == ["include"]:
            literal_includes.extend(
                field
                for field in fields[1:]
                if not any(char in field for char in "*?[]")
            )

    assert literal_includes
    assert [path for path in literal_includes if not (ROOT / path).is_file()] == []

    result = subprocess.run(
        [sys.executable, "-m", "maturin", "sdist", "--out", str(tmp_path)],
        cwd=ROOT,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "rust-toolchain.toml" not in result.stdout

    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        members = {member.name.partition("/")[2] for member in archive.getmembers()}

    expected = set(literal_includes) | {
        "src/lib.rs",
        "python/forge3d/__init__.py",
        "python/forge3d/__init__.pyi",
        "tests/test_api_contracts.py",
        "examples/terrain_demo.py",
        "python/forge3d/data/mini_dem.npy",
        "assets/fonts/PROVENANCE.md",
        "shaders/contracts/overlays.toml",
    }
    assert expected <= members
    assert "rust-toolchain.toml" not in members
