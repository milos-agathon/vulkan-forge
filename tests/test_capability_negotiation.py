import pytest
import forge3d as f3d
from forge3d.diagnostics import capabilities
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_capabilities_reports_requested_and_granted():
    if not f3d.has_gpu():
        pytest.skip("no GPU adapter")
    caps = capabilities()
    assert set(caps) >= {"requested", "granted", "limits"}
    assert "timestamp_query" in caps["requested"]
    assert set(caps["granted"]) <= set(caps["requested"])


def test_absent_capability_is_recorded_not_fatal():
    if not f3d.has_gpu():
        pytest.skip("no GPU adapter")
    caps = capabilities()
    from forge3d._forge3d import native_degradations

    degs = [d for d in native_degradations() if d["kind"] == "capability_absent"]
    absent = set(caps["requested"]) - set(caps["granted"])
    assert absent == {d["name"] for d in degs}


def test_production_device_paths_do_not_request_empty_features():
    for relative in (
        "src/viewer/init/device_init.rs",
        "src/terrain/spike/constructor.rs",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "required_features: wgpu::Features::empty()" not in source, relative


def test_no_default_device_descriptor_requests_in_production_code():
    """`request_device(&wgpu::DeviceDescriptor::default(), ...)` is
    `Features::empty()` in disguise: a device created that way bypasses
    capability negotiation entirely (audit finding F-08, extrude_polygon_gpu_py).
    Test modules are exempt (heuristic: a hit after the file's first
    `#[cfg(test)]` marker, or a `tests.rs` file, is test code)."""
    hits = []
    for path in (ROOT / "src").rglob("*.rs"):
        if path.name == "tests.rs" or path.parent.name == "tests":
            continue
        text = path.read_text(encoding="utf-8")
        cfg_test = text.find("#[cfg(test)]")
        for i, line in enumerate(text.splitlines(), 1):
            if "request_device(&wgpu::DeviceDescriptor::default()" in line:
                offset = sum(
                    len(l) + 1 for l in text.splitlines()[: i - 1]
                )
                if cfg_test != -1 and offset > cfg_test:
                    continue  # inside the trailing test module
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{i}")
    assert hits == [], (
        f"un-negotiated DeviceDescriptor::default() device requests: {hits}"
    )


CAPABILITY_DECISION_OWNER = "src/core/capabilities.rs"

# TESSELLA: every terrain indirect-draw capability decision must go through
# `IndirectDrawMode` in src/core/capabilities.rs, which is the only place that
# also records the `rendering_fallback` the decision implies. An inline
# `device.features().contains(...)` anywhere else is exactly the regression that
# made the CPU draw-loop fallback invisible to the render certificate.
GUARDED_FEATURES = (
    "Features::MULTI_DRAW_INDIRECT",
    "Features::MULTI_DRAW_INDIRECT_COUNT",
    "Features::INDIRECT_FIRST_INSTANCE",
)


def test_indirect_draw_capability_checks_live_only_in_the_recorder():
    hits = []
    for path in (ROOT / "src").rglob("*.rs"):
        rel = path.relative_to(ROOT).as_posix()
        if rel == CAPABILITY_DECISION_OWNER:
            continue
        text = path.read_text(encoding="utf-8")
        cfg_test = text.find("#[cfg(test)]")
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            if not any(feature in line for feature in GUARDED_FEATURES):
                continue
            offset = sum(len(l) + 1 for l in lines[: i - 1])
            if cfg_test != -1 and offset > cfg_test:
                continue  # inside the trailing test module
            hits.append(f"{rel}:{i}")
    assert hits == [], (
        "indirect-draw capability checks bypass IndirectDrawMode::negotiate "
        f"(and therefore record no degradation): {hits}"
    )


def test_terrain_indirect_fallback_is_recorded_at_the_taken_branch():
    """The certificate drains a render-LOCAL degradation capture, so the
    negotiation-time `capability_absent` rows never reach it. The fallback has to
    be recorded inside the render, and only on a frame that actually issues
    indirect draws (`culling="none"` and Grid geometry both draw directly)."""
    owner = (ROOT / CAPABILITY_DECISION_OWNER).read_text(encoding="utf-8")
    assert "pub fn record_fallbacks(&self, granted: Features)" in owner
    assert '"terrain_multi_draw_indirect"' in owner
    assert '"terrain_indirect_first_instance"' in owner

    execute = (ROOT / "src/terrain/renderer/draw/execute.rs").read_text(encoding="utf-8")
    assert "IndirectDrawMode::negotiate(granted)" in execute
    assert "if indirect.is_some() {" in execute
    assert "draw_mode.record_fallbacks(granted)" in execute


def test_indirect_draw_fallback_matches_granted_capabilities():
    """Biconditional, so it is non-tautological on any adapter: a fully capable
    device fails if the record fires spuriously, and a device missing any of the
    three bits fails if the record is absent. The second render is a settled
    frame, which is what proves the record is emitted per frame rather than once
    per renderer construction."""
    import tempfile

    from _terrain_runtime import _write_test_hdr, terrain_rendering_available

    if not terrain_rendering_available():
        pytest.skip("terrain rendering runtime unavailable on this adapter")

    from forge3d.diagnostics import render_certificate
    from test_terrain_clipmap_streaming import _make_params, _render_rgba, _steep_dem

    granted = set(capabilities()["granted"])
    gpu_multi_draw = {
        "indirect_first_instance",
        "multi_draw_indirect",
        "multi_draw_indirect_count",
    } <= granted

    with tempfile.TemporaryDirectory() as td:
        hdr_path = Path(td) / "probe.hdr"
        _write_test_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        dem = _steep_dem(64)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        params = _make_params(culling="frustum")
        for frame in range(2):
            _render_rgba(renderer, params, dem, ibl)
            certificate = render_certificate(sign=False)
            assert "clipmap_lod_select" in certificate["engine"]["wgsl_module_hashes"], (
                "the invariant below only applies when the clipmap indirect path ran"
            )
            degradations = {
                (entry["kind"], entry["name"]) for entry in certificate["degradations"]
            }
            recorded = ("rendering_fallback", "terrain_multi_draw_indirect") in degradations
            assert recorded == (not gpu_multi_draw), (
                f"frame {frame}: granted={sorted(granted)} implies "
                f"gpu_multi_draw={gpu_multi_draw} but degradations={sorted(degradations)}"
            )


def test_viewer_gpu_timing_readback_is_non_blocking():
    source = (
        ROOT / "src/viewer/state/viewer_helpers/gi/reexecute.rs"
    ).read_text(encoding="utf-8")
    assert "Maintain::Wait" not in source
    assert "pollster::block_on(timer.get_results())" not in source
    assert "timer.try_get_results()" in source
