"""TERMINUS panic/unwrap ratchet gate."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

from _toml_compat import load_toml

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("gis", "vector", "labels", "py_functions", "terrain")
TOKEN_PATTERNS = {
    "panic": re.compile(r"panic!\("),
    "unwrap": re.compile(r"\.unwrap\(\)"),
    "expect": re.compile(r"\.expect\("),
}


def _production_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    skip = False
    depth = 0
    pending_cfg_test = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#[cfg(test)]"):
            pending_cfg_test = True
            continue
        if pending_cfg_test and re.match(r"(pub\s+)?mod\s+tests\b", stripped):
            skip = True
            pending_cfg_test = False
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                depth = 1
            continue
        if pending_cfg_test and stripped and not stripped.startswith("#"):
            pending_cfg_test = False
        if skip:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                skip = False
                depth = 0
            continue
        out.append(line)
    return "\n".join(out)


def _is_test_fixture_path(path: Path) -> bool:
    text = path.as_posix()
    return (
        "/tests/" in text
        or text.endswith("/tests.rs")
        or text.endswith("_tests.rs")
        or "/gsub_tests/" in text
        or "/gpos_tests" in text
        or "/bidi_conformance_tests" in text
        or "/linebreak_conformance_tests" in text
    )


def _iter_production_sources(
    root: Path = ROOT,
    source_overrides: Mapping[str, str] | None = None,
):
    overrides = source_overrides or {}
    for module in MODULES:
        for path in (root / "src" / module).rglob("*.rs"):
            if _is_test_fixture_path(path):
                continue
            rel = path.relative_to(root).as_posix()
            raw = overrides.get(rel)
            text = _production_text(path) if raw is None else _production_text_value(raw)
            yield module, rel, text


def _production_text_value(raw: str) -> str:
    """Apply the production scanner to source text without touching the tree."""

    lines = raw.splitlines()
    out: list[str] = []
    skip = False
    depth = 0
    pending_cfg_test = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#[cfg(test)]"):
            pending_cfg_test = True
            continue
        if pending_cfg_test and re.match(r"(pub\s+)?mod\s+tests\b", stripped):
            skip = True
            pending_cfg_test = False
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                depth = 1
            continue
        if pending_cfg_test and stripped and not stripped.startswith("#"):
            pending_cfg_test = False
        if skip:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                skip = False
                depth = 0
            continue
        out.append(line)
    return "\n".join(out)


def _counts(
    root: Path = ROOT,
    source_overrides: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    result = {module: {"panic": 0, "unwrap": 0, "expect": 0} for module in MODULES}
    for module, _rel, text in _iter_production_sources(root, source_overrides):
        for token, pattern in TOKEN_PATTERNS.items():
            result[module][token] += len(pattern.findall(text))
    return result


def _source_counts(
    root: Path = ROOT,
    source_overrides: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for _module, rel, text in _iter_production_sources(root, source_overrides):
        counts = {token: len(pattern.findall(text)) for token, pattern in TOKEN_PATTERNS.items()}
        if any(counts.values()):
            result[rel] = counts
    return result


def _module_offenders(current, ratchet) -> list[str]:
    entries = {entry["module"]: entry for entry in ratchet["modules"]}
    offenders = []
    for module, counts in current.items():
        expected = entries[module]
        for token, actual in counts.items():
            limit = int(expected[token])
            if actual > limit:
                offenders.append(f"{module}.{token}: {actual} > ratchet {limit}")
    return offenders


def _allowlist_offenders(current, ratchet) -> list[str]:
    entries = {entry["path"]: entry for entry in ratchet["source_allowlist"]}
    offenders = []
    missing = sorted(set(current) - set(entries))
    stale = sorted(set(entries) - set(current))
    if missing or stale:
        offenders.append(f"source allowlist paths differ: missing={missing}, stale={stale}")
    for path, counts in current.items():
        if path not in entries:
            continue
        entry = entries[path]
        if not str(entry.get("reason", "")).strip():
            offenders.append(f"{path}: missing allowlist reason")
        for token, actual in counts.items():
            limit = int(entry[token])
            if actual > limit:
                offenders.append(f"{path}.{token}: {actual} > allowlist {limit}")
    return offenders


def test_robustness_ratchet_counts_do_not_increase():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    current = _counts()
    offenders = _module_offenders(current, ratchet)
    assert offenders == [], "TERMINUS robustness ratchet increased:\n" + "\n".join(offenders)


def test_robustness_ratchet_records_required_burndown():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    meta = ratchet.get("meta", ratchet)
    before = int(meta["step0_reachable_panic_unwrap"])
    after = sum(int(entry["panic"]) + int(entry["unwrap"]) for entry in ratchet["modules"])
    required_pct = float(meta["required_panic_unwrap_reduction_pct"])
    actual_pct = 100.0 * (before - after) / before
    assert actual_pct >= required_pct


def test_robustness_ratchet_records_exact_step0_baseline():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    meta = ratchet.get("meta", ratchet)
    baseline = {
        entry["module"]: {
            token: int(entry[f"baseline_{token}"])
            for token in TOKEN_PATTERNS
        }
        for entry in ratchet["modules"]
    }
    assert set(baseline) == set(MODULES)
    assert sum(counts["panic"] + counts["unwrap"] for counts in baseline.values()) == int(
        meta["step0_reachable_panic_unwrap"]
    )
    assert sum(counts["panic"] for counts in baseline.values()) == 1
    assert sum(counts["unwrap"] for counts in baseline.values()) == 93
    assert sum(counts["expect"] for counts in baseline.values()) == 27


def test_remaining_sources_match_reasoned_allowlist():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    current = _source_counts()
    offenders = _allowlist_offenders(current, ratchet)
    assert offenders == [], "TERMINUS source allowlist increased:\n" + "\n".join(offenders)


def test_cog_unwrap_ablation_fails_module_and_source_allowlist_without_rewriting_tree():
    """Red proof for the exact COG failure mode found during the audit."""

    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    rel = "src/terrain/cog/cog_reader.rs"
    original = (ROOT / rel).read_text(encoding="utf-8")
    injection = "\nfn terminus_reachable_unwrap_probe() { let _ = Some(1_u8).unwrap(); }\n"
    overrides = {rel: original + injection}

    assert _module_offenders(_counts(), ratchet) == []
    assert _allowlist_offenders(_source_counts(), ratchet) == []

    module_offenders = _module_offenders(_counts(source_overrides=overrides), ratchet)
    source_offenders = _allowlist_offenders(
        _source_counts(source_overrides=overrides), ratchet
    )
    assert "terrain.unwrap: 32 > ratchet 31" in module_offenders
    assert any(
        rel in offender and ("missing=" in offender or ".unwrap: 1 > allowlist 0" in offender)
        for offender in source_offenders
    )
    assert (ROOT / rel).read_text(encoding="utf-8") == original
