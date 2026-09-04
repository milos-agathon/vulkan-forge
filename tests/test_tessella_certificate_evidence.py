"""Spec verification item (9): the certificate's degradations on the CI adapter.

The other gate tests report their own numbers. This one exists so the
*degradation list itself* is captured as evidence rather than argued about,
and so a lane that silently loses a capability leaves a record behind.
"""

from __future__ import annotations

import pytest

import forge3d as f3d
from _tessella_evidence import record_tessella_result
from _terrain_runtime import terrain_rendering_available

# TESSELLA's five negotiated capabilities, by their stable capability names in
# src/core/capabilities.rs WANTED.
TESSELLA_CAPABILITIES = (
    "indirect_first_instance",
    "multi_draw_indirect",
    "multi_draw_indirect_count",
    "texture_binding_array",
    "sampled_texture_and_storage_buffer_array_non_uniform_indexing",
    "texture_compression_bc",
    "timestamp_query",
)


@pytest.mark.gpu_lane
@pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)
def test_certificate_degradations_are_recorded_as_evidence():
    import tempfile
    from pathlib import Path

    from forge3d.diagnostics import render_certificate
    from _terrain_runtime import _write_test_hdr
    from test_terrain_clipmap_streaming import _make_params, _render_rgba, _steep_dem

    # The certificate drains a render-local degradation capture, so it only
    # exists after a real render. Drive one through the TESSELLA path.
    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        params = _make_params(size_px=(64, 64))
        _render_rgba(renderer, params, _steep_dem(64), ibl)

    probe = f3d.device_probe()
    certificate = render_certificate(sign=False)
    degradations = certificate.get("degradations", [])
    names = sorted({str(entry["name"]) for entry in degradations})

    # Not an assertion that the list is empty: on a partially capable adapter
    # it must NOT be. The assertion is that every degraded capability is
    # *named*, which is what the spec's operating rule requires.
    for entry in degradations:
        assert entry.get("kind"), entry
        assert entry.get("name"), entry
        assert entry.get("consequence"), entry

    record_tessella_result(
        "capability_degradations",
        {
            "adapter": str(probe.get("adapter_name", probe.get("name", "unknown"))),
            "backend": str(probe.get("backend", "unknown")),
            "degradations": names,
            "degradation_count": len(degradations),
            "tessella_capabilities_degraded": sorted(
                name for name in names if name in TESSELLA_CAPABILITIES
            ),
        },
    )
