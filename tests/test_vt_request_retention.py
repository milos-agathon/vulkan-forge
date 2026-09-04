from __future__ import annotations

import tempfile
from pathlib import Path

import forge3d as f3d
import numpy as np
import pytest

from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from forge3d.terrain_params import TerrainVTSettings, VTLayerFamily
from test_terrain_clipmap_streaming import _make_params, _render_rgba, _steep_dem


@pytest.mark.gpu_lane
@pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)
def test_thirty_not_ready_frames_preserve_requests_and_converge():
    side = 512
    vt = TerrainVTSettings(
        enabled=True,
        layers=[
            VTLayerFamily(
                family=family,
                virtual_size_px=(side, side),
                tile_size=120,
                tile_border=4,
            )
            for family in ("albedo", "normal", "mask")
        ],
        atlas_size=1024,
        residency_budget_mb=32.0,
        max_mip_levels=4,
        use_feedback=True,
    )
    sources = {
        "albedo": np.full((side, side, 4), [112, 132, 96, 255], dtype=np.uint8),
        "normal": np.full((side, side, 4), [128, 128, 255, 255], dtype=np.uint8),
        "mask": np.full((side, side, 4), [255, 128, 255, 255], dtype=np.uint8),
    }
    with tempfile.TemporaryDirectory() as td:
        hdr = Path(td) / "probe.hdr"
        _write_test_hdr(hdr)
        ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        for material_index in range(4):
            for family, source in sources.items():
                renderer.register_material_vt_source(
                    material_index,
                    family,
                    source,
                    (side, side),
                    [0.5, 0.5, 1.0, 1.0],
                )
        params = _make_params(size_px=(64, 64), vt=vt)
        dem = _steep_dem(64)
        _render_rgba(renderer, params, dem, ibl)
        renderer.force_vt_feedback_not_ready_for_test(30)

        # SET IDENTITY, not cardinality. `retained_requests` alone cannot tell
        # a preserved request set from one that was silently dropped and
        # refilled with different keys at the same count, which is exactly the
        # failure the `None` feedback branch used to have.
        baseline = {tuple(key) for key in renderer.read_retained_vt_requests()}
        # One distinct key per registered source (4 materials x 3 families), so
        # the comparison below is a real set comparison and not a restatement
        # of the count.
        assert len(baseline) == 12, sorted(baseline)
        assert len({key[3:] for key in baseline}) == 12, sorted(baseline)
        assert len(baseline) == int(
            renderer.get_material_vt_stats()["retained_requests"]
        )
        for frame in range(30):
            _render_rgba(renderer, params, dem, ibl)
            retained = {tuple(key) for key in renderer.read_retained_vt_requests()}
            assert retained == baseline, {
                "not_ready_frame": frame,
                "dropped": sorted(baseline - retained),
                "invented": sorted(retained - baseline),
            }
            assert len(retained) == int(
                renderer.get_material_vt_stats()["retained_requests"]
            )
        convergence_frame = 0
        for convergence_frame in range(1, 9):
            _render_rgba(renderer, params, dem, ibl)
            if renderer.get_material_vt_stats()["retained_requests"] == 0:
                break
        final_stats = renderer.get_material_vt_stats()
        assert final_stats["retained_requests"] == 0
        assert renderer.read_retained_vt_requests() == []
        record_tessella_result(
            "vt_request_retention",
            {
                "feedback_not_ready_frames": 30,
                "convergence_budget_frames": 8,
                "convergence_frames": convergence_frame,
                "retained_set_size": len(baseline),
                "retained_set_identical_every_not_ready_frame": True,
                "retained_requests_after_convergence": int(
                    final_stats["retained_requests"]
                ),
                "tiles_streamed": int(final_stats["tiles_streamed"]),
            },
        )


def test_every_vt_source_reaches_the_atlas_through_the_store_trait():
    """TESSELLA spec item 4, source-level gate (cf. tests/test_allocation_gate.py).

    `register_source(..., data: Vec<u8>, ...)` is replaced by a store handle:
    the ingest boundary builds a `MemoryPageStore` and the renderer holds no
    raw image and no `MipImage`. There must be exactly ONE way a VT tile's
    bytes are obtained anywhere in the renderer -- `store.page(...)` inside
    `build_tile_data` -- so a future "fast path" that slices an image inline
    cannot reappear without failing here.
    """

    root = Path(__file__).resolve().parents[1] / "src"
    runtime = (root / "terrain/renderer/virtual_texture.rs").read_text(encoding="utf-8")
    store = (root / "terrain/vt/store.rs").read_text(encoding="utf-8")

    # The in-RAM special case is gone, name and all.
    forbidden = ("MipImage", "PreparedVTSourcePayload", "VTSourcePayload")
    offenders = [
        f"{path.relative_to(root)}:{number}:{line.strip()}"
        for path in root.rglob("*.rs")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for symbol in forbidden
        if symbol in line
    ]
    assert offenders == []

    # Both source kinds are the same type at rest and after preparation.
    assert "store: Arc<dyn crate::terrain::vt::VirtualTextureStore>," in runtime
    assert "MemoryPageStore::new(" in runtime
    assert "MmapPageStore::open(" in runtime
    assert "rebind_tile_geometry(" in runtime

    # One page-acquisition call site in the whole renderer, and it is the one
    # inside build_tile_data.
    page_calls = [
        number
        for number, line in enumerate(runtime.splitlines(), 1)
        if ".store.page(" in line or "store.page(page_key)" in line
    ]
    assert len(page_calls) == 1, page_calls
    build_tile_data = runtime.split("fn build_tile_data(", 1)[1].split(
        "\n    fn ", 1
    )[0]
    assert "source.store.page(page_key)?" in build_tile_data

    # MemoryPageStore is a real implementation of the trait, not a wrapper the
    # renderer can bypass.
    assert "impl VirtualTextureStore for MemoryPageStore" in store
    assert "pub struct MemoryPageStore" in store


def test_no_shipped_synthetic_reader_callers():
    root = Path(__file__).resolve().parents[1] / "src"
    callers = []
    for path in root.rglob("*.rs"):
        if path.name == "readers.rs":
            continue
        text = path.read_text(encoding="utf-8")
        for symbol in ("SyntheticHeightReader", "SyntheticOverlayReader"):
            for line_number, line in enumerate(text.splitlines(), 1):
                if symbol in line and "pub use" not in line:
                    callers.append(f"{path.relative_to(root)}:{line_number}:{line.strip()}")
    assert callers == []


def test_retention_fault_injection_delays_the_real_feedback_map():
    root = Path(__file__).resolve().parents[1] / "src"
    feedback = (root / "core/feedback_buffer.rs").read_text(encoding="utf-8")
    runtime = (root / "terrain/renderer/virtual_texture.rs").read_text(
        encoding="utf-8"
    )
    assert "forced_not_ready_polls" in feedback
    assert "try_read_feedback_entries" in feedback
    assert "force_not_ready_polls_for_test" in runtime
    assert "feedback_not_ready_frames" not in runtime


def test_gpu_lod_and_visibility_shaders_have_live_callsites():
    root = Path(__file__).resolve().parents[1]
    geometry = (root / "src/terrain/renderer/geometry.rs").read_text()
    execute = (root / "src/terrain/renderer/draw/execute.rs").read_text()
    assert "GpuLodSelector" in geometry
    assert "encode_indirect" in geometry
    assert "terrain_visbuffer_write.shader" in execute
    assert "stage_visibility_stats" in execute
    assert "encode_visibility_resolve_pass" in execute
    assert "pass.draw(0..3, 0..1)" in execute
    pipeline = (root / "src/terrain/renderer/pipeline_cache.rs").read_text()
    resolve_pipeline = pipeline.split(
        "create_clipmap_visibility_resolve_pipeline(", 1
    )[1]
    assert 'entry_point: "vs_clipmap_main"' in resolve_pipeline
    assert 'entry_point: "fs_visibility_geometry"' in resolve_pipeline
    shader = (
        root / "src/shaders/terrain_visibility_fullscreen.wgsl"
    ).read_text()
    assert "visibility_barycentrics" in shader
    assert "terrain_visibility_indices" in shader


def test_height_is_the_fourth_feedback_driven_vt_family():
    root = Path(__file__).resolve().parents[1]
    residency = (root / "src/terrain/vt_family_residency.rs").read_text()
    streaming = (root / "src/terrain/renderer/streaming.rs").read_text()
    py_api = (root / "src/terrain/renderer/py_api.rs").read_text()
    assert "VT_FAMILY_COUNT: usize = 4" in residency
    assert "HeightVtFamilyRuntime" in streaming
    assert "RetainedRequestSet" in streaming
    assert "FamilyResidencyTracker" in streaming
    assert "ReaderPageStore" in streaming
    assert "VirtualTextureStore for ReaderPageStore" in streaming
    assert "latest_feedback_uvs" in py_api


def test_public_vt_stats_diagnostics_surface_is_registered():
    from forge3d import diagnostics

    assert callable(diagnostics.vt_stats)


def test_retained_request_set_accessor_is_registered():
    assert hasattr(f3d.TerrainRenderer, "read_retained_vt_requests")
    stub = (
        Path(__file__).resolve().parents[1] / "python/forge3d/__init__.pyi"
    ).read_text(encoding="utf-8")
    assert "def read_retained_vt_requests(" in stub


def test_captured_feedback_provenance_accessor_is_registered():
    assert hasattr(
        f3d.TerrainRenderer, "resolve_captured_vt_feedback_provenance"
    )
    stub = (
        Path(__file__).resolve().parents[1] / "python/forge3d/__init__.pyi"
    ).read_text(encoding="utf-8")
    assert "def resolve_captured_vt_feedback_provenance(" in stub


def test_selected_tessella_acceptance_is_a_required_zero_skip_hardware_lane():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    tessella_paths = workflow.split("\n            tessella:\n", 1)[1].split(
        "\n\n  # ============================================================================",
        1,
    )[0]
    for source_owner in (
        "src/terrain/clipmap/**",
        "src/terrain/renderer/virtual_texture.rs",
        "src/terrain/renderer/visibility_buffer.rs",
        "src/terrain/renderer/py_api.rs",
        "python/forge3d/__init__.pyi",
    ):
        assert f"- '{source_owner}'" in tessella_paths
    job = workflow.split("\n  test-tessella-gpu:", 1)[1].split(
        "\n  # ============================================================================",
        1,
    )[0]
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in job
    assert "terrain_ci_probe.py --mode terrain --require-nvidia-vulkan" in job
    assert "scripts/assert_junit_zero_skips.py" in job
    for test_file in (
        "test_vt_out_of_core.py",
        "test_hzb_culling.py",
        "test_visibility_buffer.py",
        "test_bc_encoders.py",
        "test_flythrough_popping.py",
        "test_vt_request_retention.py",
    ):
        assert test_file in job
    aggregator = workflow.split("\n  full-acceptance-summary:", 1)[1]
    assert "test-tessella-gpu" in aggregator
    assert "tessella_selected=" in aggregator
