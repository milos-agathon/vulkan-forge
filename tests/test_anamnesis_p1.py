from __future__ import annotations

import os
from pathlib import Path

import forge3d as f3d
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _job(workflow: str, name: str, next_name: str) -> str:
    return workflow.split(f"  {name}:", 1)[1].split(f"\n  {next_name}:", 1)[0]


def test_production_renderers_consume_real_framegraph_barriers():
    framegraph = (ROOT / "src/core/framegraph_impl/mod.rs").read_text(encoding="utf-8")
    forward = (ROOT / "src/offscreen/forward.rs").read_text(encoding="utf-8")
    terrain = (ROOT / "src/terrain/renderer/draw/mod.rs").read_text(encoding="utf-8")
    terrain_aov = (ROOT / "src/terrain/renderer/aov.rs").read_text(encoding="utf-8")
    diagnostics = (ROOT / "src/py_functions/diagnostics.rs").read_text(encoding="utf-8")

    assert "pub struct RendererGraphExecution" in framegraph
    assert "pub enum RendererGraphResource" in framegraph
    assert "pub fn begin_execution" in framegraph
    assert "pub fn run_pass" in framegraph
    assert "pub fn encoder(&mut self)" in framegraph
    assert "queue.submit" in framegraph
    assert 'execute_with_barriers("offscreen.forward"' in forward
    assert 'execute_with_barriers("offscreen.readback"' in forward
    assert "offscreen readback lost its compiled color transition" in forward
    assert ".begin_execution(self.device.clone(), self.queue.clone())" in terrain
    assert 'execution.run_pass("terrain.forward"' in terrain
    assert 'execution.run_pass("terrain.resolve"' in terrain
    assert "execution.bind_texture(handles.shadow" in terrain
    assert "execution.bind_buffer(handles.prepared" in terrain
    assert "prepare_declaration.extend_from_slice(&prepared_bytes)" in terrain
    assert "execute_with_barriers" not in terrain
    assert "let _ = cache" not in (ROOT / "src/terrain/renderer/py_api.rs").read_text(
        encoding="utf-8"
    )
    assert "does not yet serialize the multi-attachment AOV graph" in (
        ROOT / "src/terrain/renderer/py_api.rs"
    ).read_text(encoding="utf-8")
    assert '"FORGE3D_TERRAIN_SHADOW_DEBUG"' in (
        ROOT / "src/terrain/renderer/py_api.rs"
    ).read_text(encoding="utf-8")
    assert "lost its compiled shadow transition" in terrain
    assert "lost its compiled beauty transition" in terrain
    assert 'execute_with_barriers("terrain.forward_aov"' in terrain_aov
    assert 'execute_with_barriers("terrain.resolve_aov"' in terrain_aov
    assert "last_renderer_graph_report" in diagnostics
    assert "RendererGraphBuilder" not in diagnostics
    assert "compile_renderer_graph" not in diagnostics

    direct_constructors = []
    for source in (ROOT / "src").rglob("*.rs"):
        if source == ROOT / "src/core/framegraph_impl/mod.rs":
            continue
        if "FrameGraph::new()" in source.read_text(encoding="utf-8"):
            direct_constructors.append(source.relative_to(ROOT).as_posix())
    assert direct_constructors == []


def test_p1_rejecting_controls_cover_recompute_and_complete_disk_bound():
    inertness = (ROOT / "tests/test_anamnesis_inertness.py").read_text(encoding="utf-8")
    incremental = (ROOT / "tests/test_anamnesis_incremental.py").read_text(
        encoding="utf-8"
    )
    adversarial = (ROOT / "tests/test_anamnesis_adversarial_keys.py").read_text(
        encoding="utf-8"
    )
    native_acceptance = (ROOT / "tests/anamnesis_gpu_acceptance.py").read_text(
        encoding="utf-8"
    )

    assert "test_opaque_renderer_recipe_change_alone_cannot_serve_stale_hit" in inertness
    assert "test_output_destination_is_proven_irrelevant_to_pixel_keys" in inertness
    assert "set(incremental.predicted_recompute)" in incremental
    assert "set(incremental.observed_recompute)" in incremental
    assert "test_complete_store_footprint_is_hard_bounded" in adversarial
    assert "test_tiny_budget_rejects_unrepresentable_self_describing_entry" in adversarial
    assert "cache=None" not in native_acceptance
    assert '"incremental_native_invocations": 600' in native_acceptance
    assert '"graph_command_submissions"' in native_acceptance
    assert "expected_seed_misses = list(_NATIVE_PASSES) * 600" in native_acceptance
    assert 'seed_report["misses"] != expected_seed_misses' in native_acceptance
    assert 'or seed_report["hits"]' in native_acceptance
    assert 'seed_report["graph_command_submissions"] != 3600' in native_acceptance
    assert 'result["matching_frames"] != 600' in native_acceptance
    assert 'result["speedup"] < 20.0' in native_acceptance
    assert "1801" not in native_acceptance
    assert "2402" not in native_acceptance
    assert "changed_report[\"misses\"] != list(_NATIVE_PASSES)" in native_acceptance


def test_p1_portability_and_production_lanes_fail_closed():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    matrix = (ROOT / ".github/workflows/determinism-matrix.yml").read_text(
        encoding="utf-8"
    )

    hosted_seed = _job(matrix, "anamnesis-seed", "anamnesis-portability")
    hosted_consumer = _job(matrix, "anamnesis-portability", "diff")
    for job in (hosted_seed, hosted_consumer):
        assert "scripts/terrain_ci_probe.py" in job
        assert "probe_status" in job
        assert 'exit "$probe_status"' in job
        assert "grep -Eq" not in job
    assert hosted_seed.index("Classify the hosted Vulkan adapter") < hosted_seed.index(
        "Gate ANAMNESIS incrementality"
    )

    physical_seed = _job(
        ci, "test-anamnesis-portability-seed", "test-anamnesis-portability"
    )
    physical_consumer = _job(
        ci, "test-anamnesis-portability", "test-anamnesis-production"
    )
    production = _job(ci, "test-anamnesis-production", "determinism-render")
    physical_runner = (
        "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]"
    )
    assert physical_runner in physical_seed
    assert physical_runner in physical_consumer
    assert "anamnesis-producer" not in physical_seed
    assert "anamnesis-consumer" not in physical_consumer
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in production
    for job in (physical_seed, physical_consumer, production):
        assert "Prove exact-head checkout" in job
        assert "ANAMNESIS.ABSENT" not in job
    assert "--require-nvidia-vulkan" in physical_seed
    assert '--json "anamnesis-portability-producer-adapter.json"' in physical_seed
    assert "RUNNER_TEMP/anamnesis-portability-producer-adapter.json" not in physical_seed
    assert "WGPU_BACKEND: dx12" in physical_consumer
    assert '--json "anamnesis-portability-consumer-adapter.json"' in physical_consumer
    assert "RUNNER_TEMP/anamnesis-portability-consumer-adapter.json" not in physical_consumer
    assert "anamnesis-portability-result.json" in physical_consumer
    assert "Win32_ComputerSystemProduct" in physical_seed
    assert "Win32_ComputerSystemProduct" in physical_consumer
    assert "--machine-id-file" in physical_seed
    assert "--machine-id-file" in physical_consumer
    assert "--runner-name '${{ runner.name }}'" in physical_seed
    assert "--runner-name '${{ runner.name }}'" in physical_consumer
    assert "test_real_gpu_600_frame_acceptance" in production
    assert "test_public_gpu_graph_cache_restores_intermediate_texture" in production
    assert "scripts/assert_junit_zero_skips.py" in production


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
    reason="set FORGE3D_RUN_GPU_ANAMNESIS=1 on a hardware-backed runner",
)
def test_public_gpu_graph_cache_restores_intermediate_texture(tmp_path):
    _, cold_raster, cold_meta = f3d.render_adjudication_pair(
        16, 12, 1, cache=tmp_path
    )
    _, warm_raster, warm_meta = f3d.render_adjudication_pair(
        16, 12, 1, cache=tmp_path
    )

    cold_cache = dict(cold_meta["cache"])
    warm_cache = dict(warm_meta["cache"])
    assert cold_cache["hits"] == []
    assert cold_cache["misses"] == ["offscreen.forward", "offscreen.readback"]
    assert warm_cache["hits"] == ["offscreen.forward"]
    assert warm_cache["misses"] == ["offscreen.readback"]
    assert warm_cache["bytes_read"] > 0
    assert warm_cache["wall_ms_saved"] > 0.0
    np.testing.assert_array_equal(
        np.asarray(warm_raster),
        np.asarray(cold_raster),
        err_msg="rehydrated GPU texture differs from the cold rendered resource",
    )
