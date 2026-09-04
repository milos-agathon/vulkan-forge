from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _doc(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_vt_support_matrix_states_exact_family_runtime_status():
    text = _doc("docs/guides/virtual_texturing_support_matrix.md")

    assert "| Albedo terrain VT family | `supported` | Runtime pages BC7 albedo" in text
    assert "| Normal terrain VT family | `supported` | Runtime pages BC5 normal" in text
    assert "| Mask terrain VT family | `supported` | Runtime pages BC7 mask" in text
    assert "exactly `albedo`, `normal`, and `mask`" in text
    assert "Height streaming is a separate" in text


def test_vt_docs_disallow_silent_skip_or_support_overclaim():
    text = _doc("docs/guides/virtual_texturing_support_matrix.md").lower()

    assert "must not silently skip" in text
    assert "normal terrain vt family | `unsupported`" not in text
    assert "mask terrain vt family | `unsupported`" not in text
    assert "native runtime pages only `albedo`" not in text


def test_vt_docs_report_per_family_physical_footprint_semantics():
    text = _doc("docs/guides/virtual_texturing_support_matrix.md")
    squashed = " ".join(text.split())

    assert "albedo and mask are 4:1" in squashed
    assert "normal is 2:1" in squashed
    assert "10:3 (about 3.33:1)" in squashed
    assert "atlas_device_local_bytes_{albedo,normal,mask}" in text
    assert "aggregate physical/uncompressed ratio is exactly 1:1" in squashed
    assert "per-family footprint fields are `0`" in squashed
