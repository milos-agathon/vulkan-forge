from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.anamnesis import render_sequence
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)
from forge3d.helpers.offscreen import render_offscreen_rgba
from forge3d.terrain_params import ShadowSettings


def test_native_recompute_control_is_required_on_physical_gpu_ci():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    production_job = workflow.split("  test-anamnesis-production:", 1)[1].split(
        "\n  # ============================================================================\n  # Hosted determinism families", 1
    )[0]
    required_fragments = (
        "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]",
        "WGPU_BACKEND: vulkan",
        "FORGE3D_RUN_GPU_ANAMNESIS: '1'",
        "test_native_terrain_cache_restores_all_graph_passes",
        "--junitxml",
        "scripts/assert_junit_zero_skips.py",
        "anamnesis-p0-adapter.json",
        '$actual | Set-Content "$env:RUNNER_TEMP/anamnesis-checked-out-head.txt"',
        "Get-PSDrive -PSProvider FileSystem",
        "Sort-Object Free -Descending",
        '$scratchScope = "$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT-$env:GITHUB_JOB"',
        "FORGE3D_ANAMNESIS_SCRATCH_DIR",
        "FORGE3D_ANAMNESIS_BASE_TEMP",
        '--basetemp="$env:FORGE3D_ANAMNESIS_BASE_TEMP"',
        "name: Clean ANAMNESIS scratch",
        "(Split-Path -Leaf $parentDir) -ne 'forge3d-ci-scratch'",
    )
    for fragment in required_fragments:
        assert fragment in production_job
    assert production_job.count("--basetemp=") == 1
    assert production_job.index(
        "name: Upload ANAMNESIS production evidence"
    ) < production_job.index("name: Clean ANAMNESIS scratch")


def test_cache_none_is_byte_identical_to_enabled_cold_render(tmp_path):
    recipe = {
        "terrain": {"dem_sha256": "ab" * 32},
        "camera": {"path": "inertness"},
        "layers": [{"kind": "label", "text": "A", "visible_frames": [1]}],
        "output": {"width": 32, "height": 32, "samples": 2},
    }
    uncached = render_sequence(recipe, frames=range(4), cache=None)
    cached = render_sequence(recipe, frames=range(4), cache=tmp_path)
    assert uncached.frame_blobs == cached.frame_blobs
    assert uncached.frame_hashes == cached.frame_hashes
    assert uncached.cache_report.hits == []


def test_callback_requires_explicit_renderer_fingerprint(tmp_path):
    recipe = {"terrain": {}, "output": {}}
    try:
        render_sequence(recipe, frames=[0], cache=tmp_path, render_frame=lambda _r, _f: b"x")
    except ValueError as error:
        assert "render_frame_fingerprint" in str(error)
    else:
        raise AssertionError("hidden callback code identity must not be cacheable")
    with pytest.raises(ValueError, match="render_frame_context"):
        render_sequence(
            recipe,
            frames=[0],
            cache=tmp_path,
            render_frame=lambda _r, _f: b"x",
            render_frame_fingerprint=b"renderer",
        )


def test_opaque_renderer_recipe_change_alone_cannot_serve_stale_hit(tmp_path):
    calls: list[str] = []

    def renderer(recipe, _frame):
        identity = str(recipe["scene"]["identity"])
        calls.append(identity)
        return identity.encode("ascii")

    recipe_a = {"scene": {"identity": "A"}, "output": {"format": "png"}}
    recipe_b = {"scene": {"identity": "B"}, "output": {"format": "png"}}
    first = render_sequence(
        recipe_a,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-external-context",
    )
    second = render_sequence(
        recipe_b,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-external-context",
    )

    assert first.frame_blobs == [b"A"]
    assert second.frame_blobs == [b"B"]
    assert calls == ["A", "B"]
    assert (0, "frame.output") in second.observed_recompute


def test_opaque_renderer_fingerprint_change_alone_cannot_serve_stale_hit(tmp_path):
    calls: list[str] = []

    def renderer_a(_recipe, _frame):
        calls.append("a")
        return b"A"

    def renderer_b(_recipe, _frame):
        calls.append("b")
        return b"B"

    recipe = {"scene": {"identity": "same"}, "output": {"format": "png"}}
    first = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_a,
        render_frame_fingerprint=b"renderer-A",
        render_frame_context=b"same-external-context",
    )
    second = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_b,
        render_frame_fingerprint=b"renderer-B",
        render_frame_context=b"same-external-context",
    )

    assert first.frame_blobs == [b"A"]
    assert second.frame_blobs == [b"B"]
    assert calls == ["a", "b"]
    assert (0, "frame.output") in second.observed_recompute


def test_opaque_renderer_context_change_alone_cannot_serve_stale_hit(tmp_path):
    calls: list[str] = []

    def renderer_a(_recipe, _frame):
        calls.append("a")
        return b"A"

    def renderer_b(_recipe, _frame):
        calls.append("b")
        return b"B"

    recipe = {"scene": {"identity": "same"}, "output": {"format": "png"}}
    first = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_a,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"captured-input-A",
    )
    second = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer_b,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"captured-input-B",
    )

    assert first.frame_blobs == [b"A"]
    assert second.frame_blobs == [b"B"]
    assert calls == ["a", "b"]
    assert (0, "frame.output") in second.observed_recompute


def test_reference_work_factor_change_cannot_serve_stale_hit(tmp_path):
    recipe = {
        "terrain": {"dem": [0, 1]},
        "camera": {"eye": [1, 2, 3]},
        "output": {"format": "png"},
    }
    first = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        reference_work_factor=0,
    )
    second = render_sequence(
        recipe,
        frames=[0],
        cache=tmp_path,
        reference_work_factor=1,
    )
    uncached_second = render_sequence(
        recipe,
        frames=[0],
        cache=None,
        reference_work_factor=1,
    )

    assert first.frame_blobs != second.frame_blobs
    assert second.frame_blobs == uncached_second.frame_blobs
    assert (0, "frame.output") in second.observed_recompute


def test_output_destination_is_proven_irrelevant_to_pixel_keys(tmp_path):
    calls: list[str] = []

    def renderer(_recipe, _frame):
        calls.append("render")
        return b"same pixels"

    first = {
        "terrain": {"dem": [0, 1]},
        "output": {"format": "png", "path": "first/result.png"},
    }
    second = {
        "terrain": {"dem": [0, 1]},
        "output": {"format": "png", "path": "second/result.png"},
    }
    render_sequence(
        first,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-captured-inputs",
    )
    rerender = render_sequence(
        second,
        frames=[0],
        cache=tmp_path,
        render_frame=renderer,
        render_frame_fingerprint=b"same-renderer",
        render_frame_context=b"same-captured-inputs",
    )
    assert calls == ["render"]
    assert rerender.observed_recompute == []
    assert rerender.predicted_recompute == []


def test_offscreen_helper_cache_is_inert_and_serves_identical_bytes(tmp_path):
    uncached = render_offscreen_rgba(8, 6, seed=9, frames=1, cache=None)
    first = render_offscreen_rgba(8, 6, seed=9, frames=1, cache=str(tmp_path))
    second = render_offscreen_rgba(8, 6, seed=9, frames=1, cache=str(tmp_path))
    assert uncached.tobytes() == first.tobytes() == second.tobytes()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_native_terrain_cache_restores_all_graph_passes(tmp_path):
    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    params = f3d.TerrainRenderParams(_canonical_params_config()(64, 64))
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)

    uncached = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=None
    ).to_numpy()
    uncached_report = dict(renderer.last_anamnesis_cache_report)
    first = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native"
    ).to_numpy()
    first_report = dict(renderer.last_anamnesis_cache_report)
    second = renderer.render_terrain_pbr_pom(
        material_set, env_maps, params, heightmap, cache=tmp_path / "native"
    ).to_numpy()
    second_report = dict(renderer.last_anamnesis_cache_report)

    camera_config = _canonical_params_config()(64, 64)
    camera_config.cam_phi_deg = 55.0
    changed_camera_params = f3d.TerrainRenderParams(camera_config)
    camera_changed = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        changed_camera_params,
        heightmap,
        cache=tmp_path / "native",
    ).to_numpy()
    camera_changed_report = dict(renderer.last_anamnesis_cache_report)
    camera_restored = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        changed_camera_params,
        heightmap,
        cache=tmp_path / "native",
    ).to_numpy()
    camera_restored_report = dict(renderer.last_anamnesis_cache_report)

    changed_heightmap = heightmap.copy()
    changed_heightmap[0, 0] += np.float32(0.25)
    changed = renderer.render_terrain_pbr_pom(
        material_set,
        env_maps,
        params,
        changed_heightmap,
        cache=tmp_path / "native",
    ).to_numpy()
    changed_report = dict(renderer.last_anamnesis_cache_report)

    assert uncached.tobytes() == first.tobytes() == second.tobytes()
    assert uncached_report["hits"] == []
    assert uncached_report["misses"] == []
    assert uncached_report["bytes_read"] == 0
    assert uncached_report["bytes_written"] == 0
    assert uncached_report["graph_command_submissions"] == 3
    pass_labels = [
        "terrain.prepare",
        "terrain.shadow",
        "terrain.forward",
        "terrain.resolve",
    ]
    assert first_report["hits"] == []
    assert first_report["misses"] == pass_labels
    assert first_report["bytes_written"] > 0
    assert first_report["hit_rate"] == 0.0
    assert first_report["graph_command_submissions"] == 6
    assert second_report["hits"] == pass_labels
    assert second_report["misses"] == []
    assert second_report["bytes_read"] > 0
    assert second_report["bytes_written"] == 0
    assert second_report["hit_rate"] == 1.0
    assert second_report["graph_command_submissions"] == 0
    assert camera_changed_report["hits"] == []
    assert camera_changed_report["misses"] == [
        "terrain.prepare",
        "terrain.shadow",
        "terrain.forward",
        "terrain.resolve",
    ]
    assert camera_changed_report["bytes_read"] == 0
    assert camera_changed_report["bytes_written"] > 0
    assert camera_changed_report["graph_command_submissions"] == 6
    assert camera_changed.tobytes() != second.tobytes()
    assert camera_restored_report["hits"] == pass_labels
    assert camera_restored_report["misses"] == []
    assert camera_restored_report["graph_command_submissions"] == 0
    assert camera_restored.tobytes() == camera_changed.tobytes()
    assert changed_report["hits"] == []
    assert changed_report["misses"] == pass_labels
    assert changed_report["bytes_written"] > 0
    assert changed_report["hit_rate"] == 0.0
    assert changed_report["graph_command_submissions"] == 6
    assert changed.tobytes() != second.tobytes()


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_native_terrain_cache_rejects_moment_shadow_techniques(tmp_path):
    hdr_path = tmp_path / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    config = _canonical_params_config()(64, 64)
    config.shadows = ShadowSettings(
        True, "VSM", 512, 2, 250.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 9.0, 0.9
    )
    params = f3d.TerrainRenderParams(config)
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)

    with pytest.raises(RuntimeError, match="rejects moment-shadow techniques"):
        renderer.render_terrain_pbr_pom(
            material_set,
            env_maps,
            params,
            heightmap,
            cache=tmp_path / "native-vsm",
        )
