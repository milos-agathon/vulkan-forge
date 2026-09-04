import re
from datetime import date
from pathlib import Path

from _toml_compat import load_toml, parse_toml_fallback

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "src" / "core" / "resource_tracker.rs"

# Adjustment (binding, see task-7-brief.md step 1 note): the brief's original
# pattern `create_buffer_init\(` substring-matches the tracked wrapper call
# `tracked_create_buffer_init(`. The negative lookbehind excludes any
# `create_buffer_init(` that is immediately preceded by `tracked_`, so raw
# `device.create_buffer_init(` still trips the gate while
# `tracked_create_buffer_init(` does not.
# Hardened (CENSOR audit F-09): the gate also catches UFCS calls
# (`Device::create_buffer(...)`), `create_texture_with_data` (wgpu DeviceExt —
# previously deliberately uncovered because unused in-tree), and calls split
# across lines (`device.\n    create_buffer(`) via the whitespace-collapsed
# sweep below. `tracked_*` wrapper calls stay excluded by the lookbehinds.
RAW = re.compile(
    r"\.create_buffer\(|\.create_texture\(|(?<!tracked_)create_buffer_init\("
    # UFCS evasion must name the wgpu type (`wgpu::Device::create_buffer(&dev,`).
    # Plain `Self::create_texture(`/`Foo::create_texture(` helpers that route
    # through the tracked wrappers are legitimate and stay unmatched.
    r"|Device::create_buffer\(|Device::create_texture\("
    r"|(?<!tracked_)create_texture_with_data\("
)


def _strip_line_comments(text: str) -> str:
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def _raw_sites():
    sites = []
    for path in (ROOT / "src").rglob("*.rs"):
        if path == TRACKER:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if RAW.search(line) and not line.lstrip().startswith("//"):
                sites.append(f"{path.relative_to(ROOT).as_posix()}:{i}")
    return sites


def _line_split_evasions():
    """Catch allocation calls that a line-based regex misses because the
    receiver and the method name are separated by a newline."""
    hits = []
    for path in (ROOT / "src").rglob("*.rs"):
        if path == TRACKER:
            continue
        flat = re.sub(r"\s+", "", _strip_line_comments(path.read_text(encoding="utf-8")))
        if RAW.search(flat):
            hits.append(path.relative_to(ROOT).as_posix())
    return hits


def test_zero_raw_allocation_sites_outside_tracker():
    allow = load_toml(ROOT / "tests" / "allocation_allowlist.toml")["entries"]
    allowed = {e["site"].rsplit(":", 1)[0] for e in allow}
    stray = [s for s in _raw_sites() if s.rsplit(":", 1)[0] not in allowed]
    assert stray == [], f"raw wgpu allocation sites bypass the ledger: {stray}"


def test_zero_line_split_allocation_evasions():
    allow = load_toml(ROOT / "tests" / "allocation_allowlist.toml")["entries"]
    allowed = {e["site"].rsplit(":", 1)[0] for e in allow}
    stray = [f for f in _line_split_evasions() if f not in allowed]
    assert stray == [], (
        f"allocation calls split across lines evade the per-line gate: {stray}"
    )


def test_allowlist_entries_not_expired_and_not_stale():
    allow = load_toml(ROOT / "tests" / "allocation_allowlist.toml")["entries"]
    live_files = {s.rsplit(":", 1)[0] for s in _raw_sites()}
    for e in allow:
        assert date.fromisoformat(e["expires"]) >= date.today(), f"expired: {e}"
        assert e["site"].rsplit(":", 1)[0] in live_files, f"stale allowlist entry: {e}"


def test_gate_regex_excludes_tracked_wrapper():
    """Regression test for the substring-match bug in the brief's original regex.

    `create_buffer_init(` must still be caught when called raw (e.g. on
    `device`), but must NOT be caught when called through the
    `tracked_create_buffer_init(` wrapper.
    """
    assert RAW.search("let b = device.create_buffer_init(&desc);")
    assert not RAW.search("let b = tracked_create_buffer_init(&device, &tracker, &desc);")
    # Sanity: the other two raw-allocation patterns are unaffected.
    assert RAW.search("let b = device.create_buffer(&desc);")
    assert RAW.search("let t = device.create_texture(&desc);")
    # Hardening (F-09): UFCS, DeviceExt::create_texture_with_data, and
    # line-split calls are all caught; tracked wrappers still are not.
    assert RAW.search("let b = wgpu::Device::create_buffer(&device, &desc);")
    assert RAW.search("let t = wgpu::Device::create_texture(&device, &desc);")
    assert RAW.search("let t = queue_ext.create_texture_with_data(&queue, &desc, data);")
    assert not RAW.search("let t = tracked_create_texture_with_data(&device, &queue, &desc, data);")
    assert not RAW.search("let t = tracked_create_texture(&device, &desc);")
    split = re.sub(r"\s+", "", "device.\n    create_buffer(&desc)")
    assert RAW.search(split)


def test_toml_fallback_parser_handles_empty_entries():
    """Exercise the no-tomllib fallback path directly (see _toml_compat.py).

    The venv here is Python 3.13 (tomllib available), so `load_toml` would
    otherwise never touch the fallback branch. Parse the allowlist's exact
    schema directly through `parse_toml_fallback` so the fallback logic is
    covered locally even though CI on py3.10 is the only place it's load-bearing.
    """
    sample = (
        "# comment header\n"
        "entries = []\n"
    )
    assert parse_toml_fallback(sample) == {"entries": []}


def test_toml_fallback_parser_handles_array_of_tables():
    sample = (
        "entries = []\n"
        "\n"
        "[[entries]]\n"
        'site = "src/foo.rs:12"\n'
        'reason = "temporary"\n'
        'owner = "someone"\n'
        'expires = "2099-01-01"\n'
    )
    parsed = parse_toml_fallback(sample)
    assert parsed["entries"] == [
        {
            "site": "src/foo.rs:12",
            "reason": "temporary",
            "owner": "someone",
            "expires": "2099-01-01",
        }
    ]


def test_toml_fallback_parser_handles_named_tables_and_scalar_arrays():
    sample = (
        'format = "forge3d-f3dz-corpus/1"\n'
        "grid_shape = [256, 256]\n"
        "\n"
        "[alpine]\n"
        'eps_0_1 = "635bdde3"\n'
    )
    assert parse_toml_fallback(sample) == {
        "format": "forge3d-f3dz-corpus/1",
        "grid_shape": [256, 256],
        "alpine": {"eps_0_1": "635bdde3"},
    }


def test_toml_fallback_parser_handles_nested_aot_tables_and_multiline_arrays():
    sample = (
        "[[asset]]\n"
        'path = "vsop87d.bin"\n'
        "\n"
        "[asset.term_counts.mercury]\n"
        "longitude = [986, 174, 10]\n"
        "\n"
        "[alias]\n"
        "forge3d-clippy = [\n"
        '    "clippy",\n'
        '    "--workspace",\n'
        "]\n"
    )
    assert parse_toml_fallback(sample) == {
        "asset": [
            {
                "path": "vsop87d.bin",
                "term_counts": {"mercury": {"longitude": [986, 174, 10]}},
            }
        ],
        "alias": {"forge3d-clippy": ["clippy", "--workspace"]},
    }


def test_toml_fallback_parser_preserves_hashes_inside_strings():
    sample = (
        '[meta] # an actual comment\n'
        'policy = "excludes #[cfg(test)] modules" # another comment\n'
    )
    assert parse_toml_fallback(sample) == {
        "meta": {"policy": "excludes #[cfg(test)] modules"}
    }
