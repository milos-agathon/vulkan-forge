from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _tessella_evidence import record_tessella_result
from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from forge3d.diagnostics import render_certificate, visibility_stats, vt_stats
from forge3d.mem import get_budget_policy, memory_metrics
from forge3d.terrain_params import (
    PomSettings,
    TerrainVTSettings,
    VTLayerFamily,
    make_terrain_params_config,
)
from test_terrain_clipmap_streaming import _render_rgba


VIRTUAL_SIDE = 1 << 18
LOGICAL_MIN_BYTES = 256 * 1024**3
LOGICAL_BYTES = VIRTUAL_SIDE * VIRTUAL_SIDE * 3 * 4

# The store's recorded materialization plan, pinned by the CLI flags below.
COARSE_MIN_MIP = 6
DETAIL_MAX_MIP = 5
# The CLI default window is 8 pages. The committed gate packs 2 so that the
# camera's whole coarse working set (3840 pages) plus every page the shader's
# fine-mip feedback can reach (3 families x 6 mips x 2x2 = 72) stays inside the
# 8192/128 -> 64x64 = 4096 atlas slots. A wider window would make eviction
# unavoidable and the no-thrash assertion untestable rather than false.
DETAIL_WINDOW_PAGES = 2

# 2^18 / 128 = 2048 pages on each axis at mip 0, so the pyramid has 12 levels.
PAGES_AT_MIP = {mip: max(1, -(-2048 // (1 << mip))) for mip in range(12)}
EXPECTED_PER_BAND = {
    mip: (
        PAGES_AT_MIP[mip] ** 2
        if mip >= COARSE_MIN_MIP
        else min(DETAIL_WINDOW_PAGES, PAGES_AT_MIP[mip]) ** 2
    )
    for mip in range(12)
}
EXPECTED_PAGE_COUNT = 3 * sum(EXPECTED_PER_BAND.values())  # 4167
# Header (96 B) + one 64-byte directory entry and one 128x128 BC page
# ((128/4)^2 * 16 = 16384 B) per page.
EXPECTED_STORE_BYTES = 96 + EXPECTED_PAGE_COUNT * (64 + 16384)

# The camera below sits at desired mip 6 over the full [0,1] UV rect, so the
# request set is the complete mip-6 and mip-7 grids for all three families:
# 3 * (32*32 + 16*16) = 3840 pages.
COARSE_WORKING_SET_PAGES = 3 * (
    PAGES_AT_MIP[COARSE_MIN_MIP] ** 2 + PAGES_AT_MIP[COARSE_MIN_MIP + 1] ** 2
)

def _build_sparse_store(tmp_path: Path) -> tuple[f3d.VTStore, dict]:
    store_path = tmp_path / "switzerland-procedural.f3dvt"
    manifest_path = tmp_path / "switzerland-procedural.manifest.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--bin",
            "forge3d-vtpack",
            "--",
            "--procedural",
            "--output",
            str(store_path),
            "--manifest",
            str(manifest_path),
            "--virtual-width",
            str(VIRTUAL_SIDE),
            "--virtual-height",
            str(VIRTUAL_SIDE),
            "--tile-size",
            "128",
            "--tile-border",
            "0",
            "--seed",
            "19",
            "--coarse-min-mip",
            str(COARSE_MIN_MIP),
            "--detail-max-mip",
            str(DETAIL_MAX_MIP),
            "--detail-window-pages",
            str(DETAIL_WINDOW_PAGES),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    # The manifest carries one digest per page (hundreds of KiB), so it is read
    # from disk rather than from the packer's stdout summary.
    manifest = json.loads(manifest_path.read_text())
    return f3d.open_vt_store(store_path), manifest


def _manifest_digests(manifest: dict) -> dict[tuple[int, int, int, int], str]:
    return {
        (
            page["key"]["family"],
            page["key"]["mip"],
            page["key"]["x"],
            page["key"]["y"],
        ): page["sha256"]
        for page in manifest["pages"]
    }


def test_sparse_store_declares_at_least_256_gib_without_allocating_it(tmp_path):
    store, manifest = _build_sparse_store(tmp_path)
    assert int(manifest["logical_texel_bytes"]) == LOGICAL_BYTES
    assert int(manifest["logical_texel_bytes"]) >= LOGICAL_MIN_BYTES
    assert manifest["procedural"] is True
    assert manifest["page_order"] == "family,mip,morton2"
    assert Path(store.path).read_bytes()[:8] == b"F3DVT1\0\0"
    assert all(len(page["sha256"]) == 64 for page in manifest["pages"])

    # The store materializes an explicit, recorded plan -- not three aliased
    # pages standing in for the whole address space.
    assert manifest["min_materialized_mip"] == COARSE_MIN_MIP
    assert manifest["materialization_plan"] == {
        "coarse_min_mip": COARSE_MIN_MIP,
        "detail_max_mip": DETAIL_MAX_MIP,
        "detail_window_pages": DETAIL_WINDOW_PAGES,
    }
    assert manifest["page_count"] == EXPECTED_PAGE_COUNT
    assert len(manifest["pages"]) == EXPECTED_PAGE_COUNT

    # Anti-degeneracy: no two physical pages may share a payload.
    assert manifest["distinct_page_digests"] == EXPECTED_PAGE_COUNT
    assert len({page["sha256"] for page in manifest["pages"]}) == EXPECTED_PAGE_COUNT

    per_band = Counter(
        (page["key"]["family"], page["key"]["mip"]) for page in manifest["pages"]
    )
    for family in range(3):
        for mip, expected in EXPECTED_PER_BAND.items():
            assert per_band[(family, mip)] == expected, (family, mip)

    size = Path(store.path).stat().st_size
    assert size == EXPECTED_STORE_BYTES
    assert int(manifest["logical_texel_bytes"]) // size > 5_000


def test_page_payloads_are_keyed_by_page_identity(tmp_path):
    """A wrong-tile bug is only detectable if content depends on the full key."""

    _store, manifest = _build_sparse_store(tmp_path)
    digests = _manifest_digests(manifest)
    assert len(set(digests.values())) == len(digests)

    # Vary exactly one key component at a time and require the payload to move.
    coarse_grid = PAGES_AT_MIP[COARSE_MIN_MIP]
    sampled = 0
    for y in range(coarse_grid):
        for x in range(0, coarse_grid - 1, 5):
            left = digests[(0, COARSE_MIN_MIP, x, y)]
            right = digests[(0, COARSE_MIN_MIP, x + 1, y)]
            assert left != right, ("x", x, y)
            sampled += 1
    assert sampled >= 200

    for x in range(coarse_grid):
        assert digests[(0, COARSE_MIN_MIP, x, 0)] != digests[(0, COARSE_MIN_MIP, x, 1)]
        for family in (1, 2):
            assert (
                digests[(family, COARSE_MIN_MIP, x, 0)]
                != digests[(0, COARSE_MIN_MIP, x, 0)]
            )
    for x in range(PAGES_AT_MIP[COARSE_MIN_MIP + 1]):
        assert (
            digests[(0, COARSE_MIN_MIP + 1, x, 0)] != digests[(0, COARSE_MIN_MIP, x, 0)]
        )


@pytest.mark.gpu_lane
@pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="requires the TESSELLA physical-GPU lane",
)
def test_256_gib_store_settles_within_eight_frames_under_host_budget(tmp_path):
    store, manifest = _build_sparse_store(tmp_path)
    hdr = tmp_path / "probe.hdr"
    _write_test_hdr(hdr)
    ibl = f3d.IBL.from_hdr(str(hdr), intensity=1.0)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    layers = [
        VTLayerFamily(
            family=family,
            virtual_size_px=(VIRTUAL_SIDE, VIRTUAL_SIDE),
            tile_size=128,
            tile_border=0,
        )
        for family in ("albedo", "normal", "mask")
    ]
    config = make_terrain_params_config(
        size_px=(3840, 2160),
        render_scale=1.0,
        terrain_span=50.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        camera_mode="clipmap:4:32:32:10:0.3",
        culling="frustum",
        shading="visibility",
        vt_store=store,
        vt_upload_budget_bytes=64 * 1024 * 1024,
        prefetch_horizon_ms=100.0,
        vt=TerrainVTSettings(
            enabled=True,
            layers=layers,
            # 8192/128 -> 64x64 = 4096 atlas slots. The 256 MiB logical budget
            # represents 4096 128x128 RGBA pages (4095 after equal family
            # splitting), enough for the 3840 coarse pages plus 72 fine pages.
            # A 192 MiB budget holds only 3072 logical pages and only looked
            # sufficient while compressed physical bytes stood in for logical
            # residency cost.
            atlas_size=8192,
            residency_budget_mb=256.0,
            max_mip_levels=8,
            use_feedback=True,
        ),
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
    )
    params = f3d.TerrainRenderParams(config)
    # Frame one is a shader-demand probe. A one-byte budget is smaller than a
    # single BC page, so predictive visible-rect requests cannot prefill any
    # family before the shader runs. The following seven frames restore the
    # production 64 MiB upload budget and must still settle within the literal
    # eight-frame gate.
    feedback_probe_params = f3d.TerrainRenderParams(
        replace(config, vt_upload_budget_bytes=1)
    )
    dem = np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96)

    stats = {}
    contributing_by_key = {}
    shader_feedback_keys = set()
    feedback_probe_keys = set()
    for settling_frame in range(1, 9):
        frame_params = feedback_probe_params if settling_frame == 1 else params
        frame = _render_rgba(renderer, frame_params, dem, ibl)
        # Capture provenance while shader demand is populated. Once every
        # desired page is resident the shader intentionally emits no paging
        # request, so inspecting only the final settled frame is vacuous.
        for tile in renderer.read_contributing_tiles():
            key = (
                tile["family_slot"],
                tile["mip_level"],
                tile["tile_x"],
                tile["tile_y"],
            )
            contributing_by_key[key] = tile
        latest_feedback = {
            tuple(int(value) for value in entry)
            for entry in renderer.read_latest_vt_shader_feedback()
        }
        shader_feedback_keys.update(latest_feedback)
        stats = renderer.get_material_vt_stats()
        if settling_frame == 1:
            feedback_probe_keys = latest_feedback
            assert stats["tiles_streamed"] == 0.0, stats
            assert stats["uploaded_bytes"] == 0.0, stats
            assert stats["resident_pages"] == 0.0, stats
        if stats["retained_requests"] == 0 and stats["miss_rate"] == 0:
            break

    assert frame.shape == (2160, 3840, 4)
    assert settling_frame <= 8
    assert stats["retained_requests"] == 0
    assert stats["evictions"] == 0
    assert stats["cache_misses"] == 0
    assert stats["cache_budget_mb"] == 256.0
    assert stats["resident_megabytes"] <= 256.0
    assert stats["resident_bytes_total"] <= 256 * 1024**2
    public_stats = vt_stats()
    assert all(public_stats[key] == value for key, value in stats.items())
    assert stats["atlas_device_local_bytes"] > 0
    assert (
        stats["atlas_uncompressed_equivalent_bytes"]
        >= stats["atlas_device_local_bytes"]
    )
    assert stats["atlas_compression_ratio"] >= 1.0
    assert stats["bindless_bc"] == 1.0
    family_footprints = {
        family: {
            "uncompressed": int(stats[f"atlas_uncompressed_equivalent_bytes_{family}"]),
            "device_local": int(stats[f"atlas_device_local_bytes_{family}"]),
            "ratio": float(stats[f"atlas_compression_ratio_{family}"]),
        }
        for family in ("albedo", "normal", "mask")
    }
    assert {
        family: values["ratio"] for family, values in family_footprints.items()
    } == {
        "albedo": 4.0,
        "normal": 2.0,
        "mask": 4.0,
    }
    assert sum(item["uncompressed"] for item in family_footprints.values()) == int(
        stats["atlas_uncompressed_equivalent_bytes"]
    )
    assert sum(item["device_local"] for item in family_footprints.values()) == int(
        stats["atlas_device_local_bytes"]
    )
    assert visibility_stats()["fallback_texels"] == 0

    # A store miss is explicit and counted, never a silent substitution: the
    # committed camera's working set is inside the materialization plan, so it
    # must be exactly zero.
    assert stats["store_page_misses"] == 0
    assert stats["store_min_materialized_mip"] == COARSE_MIN_MIP
    # Distinct physical pages actually read. An aliasing store would report a
    # handful no matter how large the camera's working set is.
    assert stats["store_pages_fetched_distinct"] >= COARSE_WORKING_SET_PAGES
    # The whole coarse working set is resident at once -- the real no-thrash
    # claim, and impossible with fewer atlas slots than requested pages.
    assert stats["resident_pages"] >= COARSE_WORKING_SET_PAGES

    # The upload-blocked first frame proves all families produced raw shader
    # demand in this exact 256-GiB fixture. Subsequent production-budget frames
    # may legitimately omit already-resident keys, so preserve the union while
    # validating the first snapshot independently.
    assert feedback_probe_keys
    assert {entry[0] for entry in feedback_probe_keys} == {0, 1, 2}
    assert shader_feedback_keys
    assert {entry[0] for entry in shader_feedback_keys} == {0, 1, 2}
    for family_slot in range(3):
        family_feedback = {
            entry for entry in shader_feedback_keys if entry[0] == family_slot
        }
        assert len(family_feedback) >= 2, family_feedback
        assert any(entry[3] != 0 or entry[4] != 0 for entry in family_feedback)
    digests = _manifest_digests(manifest)
    # The first upload-blocked frame is historical demand: once albedo becomes
    # resident it no longer appears in *latest unresolved* feedback. Resolve
    # that retained snapshot through the current CPU page-table mirror so every
    # raw family demand is proved against the exact resident page bytes.
    captured_tiles = renderer.resolve_captured_vt_feedback_provenance(
        sorted(feedback_probe_keys)
    )
    for tile in captured_tiles:
        key = (
            tile["family_slot"],
            tile["mip_level"],
            tile["tile_x"],
            tile["tile_y"],
        )
        contributing_by_key[key] = tile
    tiles = list(contributing_by_key.values())
    assert tiles
    assert {tile["family_slot"] for tile in tiles} == {0, 1, 2}
    for family_slot in range(3):
        family_tiles = [tile for tile in tiles if tile["family_slot"] == family_slot]
        assert len(family_tiles) >= 2, family_tiles
        assert any(tile["tile_x"] != 0 or tile["tile_y"] != 0 for tile in family_tiles)
    for tile in tiles:
        key = (tile["family_slot"], tile["mip_level"], tile["tile_x"], tile["tile_y"])
        assert key in digests, key
        assert tile["content_hash"] == digests[key], (key, tile["content_hash"])
    assert len({tile["content_hash"] for tile in tiles}) == len(tiles)

    certificate = render_certificate(sign=False)
    degradations = certificate["degradations"]
    assert not {
        "terrain_vt_bc_atlas",
        "terrain_vt_bindless_atlas",
    }.intersection(entry["name"] for entry in degradations), degradations
    metrics = memory_metrics()
    assert get_budget_policy() == "enforce", metrics
    assert metrics.get("budget_policy") == "enforce", metrics
    peak_host = int(certificate["allocations"]["peak_host_visible_bytes"])
    assert peak_host > 0, certificate["allocations"]
    assert peak_host < 512 * 1024**2, {
        "peak_host_visible_bytes": peak_host,
        "page_count": manifest["page_count"],
        "vt_stats": stats,
    }
    record_tessella_result(
        "vt_out_of_core",
        {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "logical_texel_bytes": int(manifest["logical_texel_bytes"]),
            "sparse_store_bytes": Path(store.path).stat().st_size,
            "page_count": int(manifest["page_count"]),
            "distinct_page_digests": int(manifest["distinct_page_digests"]),
            "min_materialized_mip": int(manifest["min_materialized_mip"]),
            "settling_frames": settling_frame,
            "retained_requests": int(stats["retained_requests"]),
            "miss_rate": float(stats["miss_rate"]),
            "evictions": int(stats["evictions"]),
            "resident_pages": int(stats["resident_pages"]),
            "store_page_misses": int(stats["store_page_misses"]),
            "store_pages_fetched_distinct": int(stats["store_pages_fetched_distinct"]),
            "contributing_tiles_verified": len(tiles),
            "fallback_texels": int(visibility_stats()["fallback_texels"]),
            # Device-local footprint is declared separately from the
            # host-visible budget: textures never enter host_visible_bytes.
            "atlas_device_local_bytes": int(stats["atlas_device_local_bytes"]),
            # One-level page-table atlas: base 2048 rows plus the packed
            # half-height mip tail, three family layers, RGBA32F entries.
            "device_local_page_table_bytes": 2048 * 3072 * 3 * 16,
            "atlas_uncompressed_equivalent_bytes": int(
                stats["atlas_uncompressed_equivalent_bytes"]
            ),
            "atlas_compression_ratio": float(stats["atlas_compression_ratio"]),
            **{
                f"atlas_device_local_bytes_{family}": values["device_local"]
                for family, values in family_footprints.items()
            },
            **{
                f"atlas_uncompressed_equivalent_bytes_{family}": values["uncompressed"]
                for family, values in family_footprints.items()
            },
            **{
                f"atlas_compression_ratio_{family}": values["ratio"]
                for family, values in family_footprints.items()
            },
            "peak_host_visible_bytes": int(peak_host),
        },
    )
