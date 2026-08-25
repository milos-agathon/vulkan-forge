# tests/test_render_certificate.py
# CENSOR Task 11: end-to-end RenderCertificate coverage over a real native
# render. Exercises the live-pass ledger, the merged degradation list, the
# Ed25519 seal, and the determinism guarantee of the signed payload.
# RELEVANT FILES: python/forge3d/diagnostics.py, python/forge3d/certificate.py,
# src/core/certificate.rs, tests/test_recipe_goldens.py

from __future__ import annotations

import pytest

import forge3d as f3d
from forge3d.diagnostics import render_certificate

from _terrain_runtime import terrain_rendering_available
from test_recipe_goldens import RECIPE_GOLDENS


def _skip_without_terrain() -> None:
    if not terrain_rendering_available():
        pytest.skip("RenderCertificate tests require a terrain-capable forge3d runtime")


def _clear_sinks() -> None:
    from forge3d._forge3d import clear_native_degradations
    from forge3d import _degradation

    clear_native_degradations()
    _degradation.clear()


def _build_and_render(tmp_path):
    """Build RECIPE_GOLDENS[0] and drive one native render, returning the scene."""
    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)
    scene.render()
    assert scene.last_render_backend == "gpu_terrain"
    return scene


@pytest.mark.apple_metal_physical
def test_certificate_has_live_passes_and_empty_degradations(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    _build_and_render(tmp_path)

    cert = render_certificate()
    assert cert["schema"] == "forge3d.render_certificate/1"
    labels = [entry["label"] for entry in cert["passes"]]
    assert labels[-1] == "mapscene.finalize"
    assert any(label.startswith("terrain.") for label in labels), labels
    assert cert["allocations"]["peak_device_local_bytes"] > 0
    assert cert["allocations"]["by_label"], cert["allocations"]

    granted = set(cert["capabilities"]["granted"])
    timed = [p for p in cert["passes"] if p["gpu_ms"] > 0]
    if "timestamp_query" in granted:
        assert len(timed) >= 5, cert["passes"]
    else:
        assert any(d["name"] == "timestamp_query" for d in cert["degradations"])

    # A healthy render records no degradations except the (expected) absence of
    # a negotiated capability such as timestamp_query.
    assert cert["degradations"] == [] or "timestamp_query" not in granted

    assert cert["signature"]["alg"] == "ed25519"


@pytest.mark.apple_metal_physical
def test_pre_render_python_degradation_does_not_leak(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    from forge3d import _degradation

    _degradation.record("pre_render_test", "must_not_leak", "recorded before render")
    _build_and_render(tmp_path)

    assert not any(
        entry["name"] == "must_not_leak"
        for entry in render_certificate(sign=False)["degradations"]
    )


@pytest.mark.apple_metal_physical
def test_signed_payload_deterministic_across_two_renders(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    from forge3d.certificate import payload_sha256

    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)

    hashes = []
    for _ in range(2):
        scene.render()
        hashes.append(payload_sha256(render_certificate()))

    assert hashes[0] == hashes[1], (
        "signed certificate payload must be byte-identical across two renders "
        f"of the same scene on the same adapter: {hashes}"
    )


@pytest.mark.apple_metal_physical
def test_certificate_excludes_shaders_owned_by_another_renderer(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    # Compile Scene-owned pipelines first. A terrain certificate must not copy
    # these labels merely because they exist in the same process-wide cache.
    f3d.Scene(2, 2)
    _build_and_render(tmp_path)

    hashes = render_certificate(sign=False)["engine"]["wgsl_module_hashes"]
    assert hashes, "terrain render must report its renderer-owned WGSL modules"
    unrelated = {
        "mesh_basic_shader",
        "overlays_shader",
        "ssao-compute",
        "text_overlay_shader",
    }
    assert unrelated.isdisjoint(hashes), hashes


@pytest.mark.apple_metal_physical
def test_scene_certificate_reports_only_scene_owned_shaders(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    _build_and_render(tmp_path)
    scene = f3d.Scene(2, 2)
    scene.render_rgba()

    hashes = render_certificate(sign=False)["engine"]["wgsl_module_hashes"]
    assert hashes, "Scene render must report its renderer-owned WGSL modules"
    assert "terrain_pbr_pom.shader" not in hashes, hashes


@pytest.mark.apple_metal_physical
def test_scene_allocations_ignore_unrelated_live_scene():
    _skip_without_terrain()
    _clear_sinks()

    scene = f3d.Scene(8, 8)
    scene.render_rgba()
    baseline = render_certificate(sign=False)["allocations"]

    unrelated = f3d.Scene(8, 8)
    scene.render_rgba()
    with_unrelated = render_certificate(sign=False)["allocations"]

    assert with_unrelated == baseline
    assert unrelated is not None


@pytest.mark.apple_metal_physical
def test_scene_allocations_include_lazily_enabled_feature():
    _skip_without_terrain()
    _clear_sinks()

    scene = f3d.Scene(32, 32)
    scene.enable_clouds("low")
    scene.render_rgba()

    labels = render_certificate(sign=False)["allocations"]["by_label"]
    assert labels.get("cloud_uniform_buffer", 0) > 0, labels


@pytest.mark.apple_metal_physical
def test_scene_shader_hashes_follow_lazy_feature_use():
    _skip_without_terrain()
    _clear_sinks()

    scene = f3d.Scene(32, 32)
    scene.enable_clouds("low")
    scene.render_rgba()
    enabled = render_certificate(sign=False)["engine"]["wgsl_module_hashes"]
    assert "cloud_shader" in enabled, enabled

    scene.disable_clouds()
    scene.render_rgba()
    disabled = render_certificate(sign=False)["engine"]["wgsl_module_hashes"]
    assert "cloud_shader" not in disabled, disabled


@pytest.mark.apple_metal_physical
def test_python_synthetic_render_replaces_stale_gpu_certificate():
    _skip_without_terrain()
    _clear_sinks()

    f3d.Scene(16, 16).render_rgba()

    from forge3d.path_tracing import render_aovs

    render_aovs(
        8,
        8,
        scene=None,
        synthetic_ok=True,
        certificate=True,
    )
    cert = render_certificate(sign=False)

    assert cert["engine"]["wgsl_module_hashes"] == {}
    assert cert["adapter"]["backend"] == "cpu"
    assert [entry["label"] for entry in cert["passes"]] == [
        "python.path_tracing.render_aovs"
    ]
    assert any(
        entry["kind"] == "synthetic_output"
        and entry["name"] == "path_tracing.render_aovs"
        for entry in cert["degradations"]
    )


def test_progressive_python_render_accepts_certificate_contract():
    _clear_sinks()

    from forge3d.path_tracing import PathTracer

    image = PathTracer(8, 8).render_progressive(
        synthetic_ok=True,
        certificate=True,
    )
    assert image.shape == (8, 8, 4)
    cert = render_certificate(sign=False)
    assert cert["passes"][0]["label"] == "python.path_tracing.render_progressive"


@pytest.mark.apple_metal_physical
def test_certificate_kwarg_writes_signed_file(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    from forge3d import certificate as _certificate

    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)
    cert_path = tmp_path / "render_certificate.json"
    scene.render(certificate=cert_path)

    assert cert_path.exists(), "certificate= path must write the signed certificate JSON"

    # The stashed payload sha matches the written certificate, and the written
    # certificate verifies against its embedded dev public key.
    metadata = scene.last_render_metadata or {}
    assert "certificate_payload_sha256" in metadata

    import json

    written = json.loads(cert_path.read_text(encoding="utf-8"))
    assert _certificate.payload_sha256(written) == metadata["certificate_payload_sha256"]

    pubkey = written["signature"]["pubkey"]
    assert _certificate.verify(cert_path, pubkey) is True


@pytest.mark.apple_metal_physical
def test_certificate_kwarg_false_leaves_metadata_clean(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)
    scene.render()  # certificate defaults to False

    metadata = scene.last_render_metadata or {}
    assert "certificate_payload_sha256" not in metadata
