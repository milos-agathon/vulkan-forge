# tests/test_provenance_veritas.py
# VERITAS DoD gate: per-pixel source attribution through the streaming VT
# path, sealed with a SHA256 Merkle root + Ed25519 signature. CPU sections
# pin the canonical encoding/Merkle rules (cross-checked against the Rust
# implementation); the GPU class renders >=2 overlapping albedo VT sources
# and asserts 100% attribution, root match, signature validity, and
# single-texel tamper detection.
# RELEVANT FILES: src/core/provenance.rs, python/forge3d/provenance.py,
# tools/verify_provenance.py, src/py_functions/provenance.rs

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _terrain_runtime import _build_heightmap, terrain_rendering_available
from forge3d import provenance as prov
from forge3d.terrain_params import (
    AovSettings,
    PomSettings,
    TerrainVTSettings,
    VTLayerFamily,
    make_terrain_params_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "tools" / "verify_provenance.py"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "provenance"
GPU_AVAILABLE = terrain_rendering_available()

# Fixed, committed test keypair seed (fixtures only — not a production key).
TEST_PRIVATE_KEY = hashlib.sha256(b"forge3d-veritas-test-key").digest()

# Authored albedo source colors (sRGB bytes), one per material index.
SOURCE_COLORS = (
    (220, 30, 30),
    (30, 220, 30),
    (30, 30, 220),
    (220, 220, 30),
)
VIRTUAL_SIZE = 512


# ---------------------------------------------------------------------------
# CPU contracts (no GPU required)
# ---------------------------------------------------------------------------

def _tile(family_slot=0, source_id=1, x=0, y=0, mip=0, payload=b"payload"):
    return prov.encode_tile_leaf(
        family_slot, source_id, x, y, mip, hashlib.sha256(payload).digest()
    )


def test_tile_leaf_encoding_round_trips() -> None:
    content_hash = hashlib.sha256(b"content").digest()
    leaf = prov.encode_tile_leaf(0, 3, 7, 11, 2, content_hash)
    assert len(leaf) == 56
    assert leaf[:4] == b"VTLF"
    decoded = prov.decode_tile_leaf(leaf)
    assert decoded == {
        "family_slot": 0,
        "source_id": 3,
        "tile_x": 7,
        "tile_y": 11,
        "mip_level": 2,
        "content_hash": content_hash,
    }


def test_source_map_leaf_encoding() -> None:
    digest = hashlib.sha256(b"map").digest()
    leaf = prov.encode_source_map_leaf(640, 480, digest)
    assert len(leaf) == 44
    assert leaf[:4] == b"VTSM"
    assert leaf[12:] == digest


def test_merkle_root_is_order_independent() -> None:
    leaves = [_tile(x=i, y=i * 2, source_id=(i % 4) + 1) for i in range(7)]
    shuffled = list(reversed(leaves))
    shuffled[0], shuffled[3] = shuffled[3], shuffled[0]
    assert prov.build_merkle_root(leaves) == prov.build_merkle_root(shuffled)


def test_merkle_odd_node_promotion_rule() -> None:
    leaves = sorted(_tile(source_id=i + 1, x=i) for i in range(3))
    h = [hashlib.sha256(l).digest() for l in leaves]
    left = hashlib.sha256(h[0] + h[1]).digest()
    expected = hashlib.sha256(left + h[2]).digest()  # lone h[2] promoted
    assert prov.build_merkle_root(leaves) == expected


def test_merkle_empty_sentinel() -> None:
    assert (
        prov.build_merkle_root([])
        == hashlib.sha256(b"forge3d.provenance.v1.empty").digest()
    )


def test_known_answer_root_matches_rust_vector() -> None:
    """Cross-language lock: the same vector is asserted in
    src/core/provenance.rs::known_answer_root_vector."""
    leaves = [
        prov.encode_tile_leaf(0, 1, 0, 0, 0, hashlib.sha256(b"source-a").digest()),
        prov.encode_tile_leaf(0, 2, 1, 0, 1, hashlib.sha256(b"source-b").digest()),
        prov.encode_source_map_leaf(4, 2, hashlib.sha256(b"source-map").digest()),
    ]
    assert (
        prov.build_merkle_root(leaves).hex()
        == "67e632b879b8d0f52360148abad03584b213f9065714e2b722db766e03e980c4"
    )


def _sample_tiles():
    return [
        {
            "family": "albedo",
            "family_slot": 0,
            "source_id": 1,
            "tile_x": 0,
            "tile_y": 0,
            "mip_level": 0,
            "content_hash": hashlib.sha256(b"src-a").hexdigest(),
        },
        {
            "family": "albedo",
            "family_slot": 0,
            "source_id": 2,
            "tile_x": 1,
            "tile_y": 0,
            "mip_level": 0,
            "content_hash": hashlib.sha256(b"src-b").hexdigest(),
        },
    ]


def test_offline_seal_verify_round_trip_and_tamper() -> None:
    source_map = np.zeros((16, 24), dtype=np.uint32)
    source_map[:8, :] = 1
    source_map[8:, :12] = 2  # remaining quadrant stays SOURCE_ID_NONE

    manifest = prov.seal_provenance_offline(source_map, _sample_tiles(), TEST_PRIVATE_KEY)
    report = prov.verify_provenance_offline(source_map, manifest)
    assert report["ok"] is True
    assert report["root_match"] is True
    assert report["signature_valid"] is True

    # SOURCE_ID_NONE is never counted as a real attribution.
    assert prov.SOURCE_ID_NONE not in report["coverage"]
    assert report["coverage"] == {1: 8 * 24, 2: 8 * 12}
    assert report["unattributed_pixels"] == 8 * 12

    # Single-texel tamper: the recomputed root must break.
    tampered = source_map.copy()
    tampered[0, 0] ^= 1
    tampered_report = prov.verify_provenance_offline(tampered, manifest)
    assert tampered_report["root_match"] is False
    assert tampered_report["ok"] is False


def test_offline_seal_is_deterministic() -> None:
    source_map = np.arange(48, dtype=np.uint32).reshape(6, 8) % 3
    a = prov.seal_provenance_offline(source_map, _sample_tiles(), TEST_PRIVATE_KEY)
    b = prov.seal_provenance_offline(source_map, list(reversed(_sample_tiles())), TEST_PRIVATE_KEY)
    assert a == b, "seal must be independent of contributing-tile order"


@pytest.mark.skipif(
    not hasattr(f3d, "seal_provenance"), reason="native forge3d extension not available"
)
def test_native_and_offline_seals_agree() -> None:
    source_map = np.zeros((10, 10), dtype=np.uint32)
    source_map[2:7, 3:9] = 1
    tiles = _sample_tiles()
    native = f3d.seal_provenance(source_map, tiles, TEST_PRIVATE_KEY)
    offline = prov.seal_provenance_offline(source_map, tiles, TEST_PRIVATE_KEY)
    native_manifest = json.loads(bytes(native))
    offline_manifest = json.loads(offline)
    assert native_manifest["merkle_root"] == offline_manifest["merkle_root"]
    assert native_manifest["signature"] == offline_manifest["signature"]
    assert native_manifest["public_key"] == offline_manifest["public_key"]
    # Cross-verification in both directions.
    assert f3d.verify_provenance(source_map, offline) is True
    assert prov.verify_provenance_offline(source_map, bytes(native))["ok"] is True


def test_mapscene_render_validates_provenance_kwargs(tmp_path) -> None:
    """MapScene emission flag is off by default and validates its key without
    touching the GPU (validation happens before any render work)."""
    from forge3d.map_scene import (
        LightingPreset,
        MapScene,
        OrbitCamera,
        OutputSpec,
        SceneRecipe,
        TerrainSource,
    )

    heightmap = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    recipe = SceneRecipe(
        terrain=TerrainSource(data=heightmap, metadata={"asset_status": "fixture"}),
        camera=OrbitCamera(),
        lighting=LightingPreset(),
        output=OutputSpec(path=str(tmp_path / "out.png"), width=32, height=32),
    )
    scene = MapScene(recipe)
    with pytest.raises(ValueError, match="32-byte Ed25519 seed"):
        scene.render(emit_provenance=True)
    with pytest.raises(ValueError, match="32-byte Ed25519 seed"):
        scene.render(emit_provenance=True, provenance_signing_key=b"short")
    with pytest.raises(ValueError, match="requires emit_provenance"):
        scene.render(provenance_signing_key=TEST_PRIVATE_KEY)


# ---------------------------------------------------------------------------
# GPU DoD gate
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _flat_source(color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((VIRTUAL_SIZE, VIRTUAL_SIZE, 4), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = color
    img[..., 3] = 255
    return np.ascontiguousarray(img)


def _register_sources(renderer) -> None:
    renderer.clear_material_vt_sources()
    for material_index, color in enumerate(SOURCE_COLORS):
        renderer.register_material_vt_source(
            material_index,
            "albedo",
            _flat_source(color),
            (VIRTUAL_SIZE, VIRTUAL_SIZE),
            # Neutral gray fallback, deliberately unlike every source color,
            # so a fallback-shaded pixel can never masquerade as a source.
            [0.5, 0.5, 0.5, 1.0],
        )


def _build_params(*, source_id: bool = True) -> "f3d.TerrainRenderParams":
    config = make_terrain_params_config(
        size_px=(256, 192),
        render_scale=1.0,
        terrain_span=8.0,
        msaa_samples=1,
        z_scale=1.6,
        exposure=1.0,
        domain=(0.0, 1.0),
        albedo_mode="material",
        colormap_strength=0.0,
        ibl_enabled=True,
        ibl_intensity=1.8,
        light_azimuth_deg=136.0,
        light_elevation_deg=24.0,
        sun_intensity=2.2,
        cam_radius=4.0,
        cam_phi_deg=142.0,
        cam_theta_deg=58.0,
        fov_y_deg=50.0,
        camera_mode="mesh",
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        aov=AovSettings(
            enabled=True, albedo=True, normal=False, depth=False, source_id=source_id
        ),
    )
    config.cam_target = [0.0, 0.0, 0.0]
    config.vt = TerrainVTSettings(
        enabled=True,
        atlas_size=4096,
        residency_budget_mb=192.0,
        max_mip_levels=6,
        layers=[VTLayerFamily(family="albedo", virtual_size_px=(VIRTUAL_SIZE, VIRTUAL_SIZE))],
    )
    return f3d.TerrainRenderParams(config)


def _write_test_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 164, 128]))


def _build_test_ibl():
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_test_hdr(hdr_path)
        return f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)


def _expected_ids_from_albedo(albedo: np.ndarray) -> np.ndarray:
    """Ground-truth source layout recovered from the pre-lighting albedo AOV.

    The terrain splat emits ``albedo = sum_i w_i * c_i`` with known,
    linearly independent flat source colors ``c_i`` and ``sum(w) == 1``, so
    the per-pixel layer weights are recoverable exactly by solving the 4x4
    system ``[c_i^T; 1] w = [albedo; 1]``. The shader attributes each pixel
    to its dominant layer (argmax w) — the same rule applied here.

    The sampled set keeps only pixels where the dominance margin
    (w_max - w_second) exceeds 0.30: the shader's unconditional
    slope/elevation hue variation (strength 0.08 in terrain_pbr_pom.wgsl)
    perturbs recovered weights by at most ~0.21 in this scene, so a 0.30
    margin cannot be flipped by it. Ambiguous pixels (blend zones, fallback
    gray, background) return 0 and are excluded from the fixed sampled set.
    """
    colors = np.stack([_srgb_to_linear(np.array(c)) for c in SOURCE_COLORS])  # (4, 3)
    system = np.vstack([colors.T, np.ones(4)])  # (4, 4)
    system_inv = np.linalg.inv(system)

    height, width, _ = albedo.shape
    flat = albedo.reshape(-1, 3).astype(np.float64)
    rhs = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    weights = (rhs @ system_inv.T).reshape(height, width, 4)

    reconstructed = (weights.reshape(-1, 4) @ colors).reshape(height, width, 3)
    residual = np.linalg.norm(reconstructed - albedo, axis=-1)
    ordered = np.sort(weights, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2]
    best = np.argmax(weights, axis=-1)
    totals = albedo.sum(axis=-1)

    unambiguous = (margin > 0.30) & (residual < 0.02) & (totals > 0.05)
    return np.where(unambiguous, best + 1, 0).astype(np.uint32)


@pytest.fixture()
def provenance_render_env():
    if not GPU_AVAILABLE:
        pytest.skip("VERITAS DoD test requires a terrain-capable GPU runtime")
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    heightmap = _build_heightmap(160)
    ibl = _build_test_ibl()
    renderer.clear_material_vt_sources()
    try:
        yield renderer, material_set, ibl, heightmap
    finally:
        try:
            renderer.clear_material_vt_sources()
        except RuntimeError:
            pass


@pytest.mark.apple_metal_physical
@pytest.mark.skipif(not GPU_AVAILABLE, reason="VERITAS DoD test requires GPU-backed forge3d")
class TestVeritasProvenanceDoD:
    def _render_triple(self, env):
        renderer, material_set, ibl, heightmap = env
        _register_sources(renderer)
        params = _build_params()
        frame = aov_frame = None
        # Warm the streaming path: feedback-driven residency settles within a
        # few frames for a fully-resident VT of this size.
        for _ in range(3):
            frame, aov_frame = renderer.render_with_aov(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
            )
        assert aov_frame.has_source_id
        rgba = np.asarray(frame.to_numpy())
        source_map = np.asarray(aov_frame.source_id(), dtype=np.uint32)
        albedo = np.asarray(aov_frame.albedo(), dtype=np.float32)
        tiles = renderer.read_contributing_tiles()
        return rgba, albedo, source_map, tiles

    def test_measurable_win(self, provenance_render_env, tmp_path):
        rgba, albedo, source_map, tiles = self._render_triple(provenance_render_env)
        assert source_map.shape == albedo.shape[:2] == rgba.shape[:2]
        assert tiles, "streaming VT feedback produced no contributing tiles"
        albedo_tiles = [t for t in tiles if t["family"] == "albedo"]
        assert albedo_tiles, "no albedo-family contributing tiles"

        # --- (1) 100% correct attribution over the fixed sampled set -------
        expected = _expected_ids_from_albedo(albedo)
        sampled = expected != prov.SOURCE_ID_NONE
        sampled_count = int(sampled.sum())
        assert sampled_count >= 1024, f"only {sampled_count} unambiguous sampled pixels"
        mismatches = int((source_map[sampled] != expected[sampled]).sum())
        assert mismatches == 0, (
            f"{mismatches}/{sampled_count} sampled pixels attributed to the wrong source"
        )
        distinct = set(np.unique(source_map[sampled]).tolist())
        assert len(distinct) >= 2, f"expected >=2 overlapping sources, saw {distinct}"

        # Every attributed pixel id must correspond to a contributing source.
        tile_ids = {t["source_id"] for t in albedo_tiles}
        assert distinct <= tile_ids

        # --- (2)+(3) seal: Merkle root match + signature True ---------------
        manifest_bytes = f3d.seal_provenance(source_map, tiles, TEST_PRIVATE_KEY)
        assert f3d.verify_provenance(source_map, manifest_bytes) is True
        offline_report = prov.verify_provenance_offline(source_map, bytes(manifest_bytes))
        assert offline_report["root_match"] is True
        assert offline_report["signature_valid"] is True
        assert offline_report["ok"] is True
        assert not offline_report["unknown_source_ids"]

        # --- (4) single-texel tamper detection ------------------------------
        tampered = source_map.copy()
        tampered[0, 0] ^= 1
        assert f3d.verify_provenance(tampered, manifest_bytes) is False
        assert prov.verify_provenance_offline(tampered, bytes(manifest_bytes))["ok"] is False

        # --- standalone verifier CLI over the emitted triple ----------------
        image_path = tmp_path / "image.png"
        smap_path = tmp_path / "source_map.npy"
        manifest_path = tmp_path / "provenance.json"
        f3d.numpy_to_png(str(image_path), rgba)
        np.save(smap_path, source_map)
        manifest_path.write_bytes(bytes(manifest_bytes))

        result = subprocess.run(
            [sys.executable, str(VERIFIER), str(image_path), str(smap_path), str(manifest_path)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"verifier failed:\n{result.stdout}\n{result.stderr}"
        assert "merkle_root_match: True" in result.stdout
        assert "signature_valid: True" in result.stdout
        assert "tamper_probe_single_texel_detected: True" in result.stdout
        assert "verified: True" in result.stdout

        # --- refresh the committed offline-verification fixture on demand ---
        if os.environ.get("FORGE3D_UPDATE_PROVENANCE_FIXTURE") == "1":
            FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, FIXTURE_DIR / "image.png")
            shutil.copy2(smap_path, FIXTURE_DIR / "source_map.npy")
            shutil.copy2(manifest_path, FIXTURE_DIR / "provenance.json")

    def test_seal_is_deterministic_across_renders(self, provenance_render_env):
        _, _, source_map_a, tiles_a = self._render_triple(provenance_render_env)
        manifest_a = json.loads(bytes(f3d.seal_provenance(source_map_a, tiles_a, TEST_PRIVATE_KEY)))
        _, _, source_map_b, tiles_b = self._render_triple(provenance_render_env)
        manifest_b = json.loads(bytes(f3d.seal_provenance(source_map_b, tiles_b, TEST_PRIVATE_KEY)))
        assert manifest_a["merkle_root"] == manifest_b["merkle_root"]
        assert manifest_a["signature"] == manifest_b["signature"]

    def test_source_id_rejects_msaa(self, provenance_render_env):
        renderer, material_set, ibl, heightmap = provenance_render_env
        _register_sources(renderer)
        config = make_terrain_params_config(
            size_px=(128, 96),
            render_scale=1.0,
            terrain_span=8.0,
            msaa_samples=4,
            z_scale=1.6,
            exposure=1.0,
            domain=(0.0, 1.0),
            albedo_mode="material",
            colormap_strength=0.0,
            camera_mode="mesh",
            cam_radius=4.0,
            pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
            aov=AovSettings(enabled=True, albedo=False, normal=False, depth=False, source_id=True),
        )
        config.cam_target = [0.0, 0.0, 0.0]
        config.vt = TerrainVTSettings(
            enabled=True,
            atlas_size=4096,
            residency_budget_mb=192.0,
            max_mip_levels=6,
            layers=[VTLayerFamily(family="albedo", virtual_size_px=(VIRTUAL_SIZE, VIRTUAL_SIZE))],
        )
        params = f3d.TerrainRenderParams(config)
        with pytest.raises(RuntimeError, match="msaa_samples=1"):
            renderer.render_with_aov(
                material_set=material_set,
                env_maps=ibl,
                params=params,
                heightmap=heightmap,
            )
