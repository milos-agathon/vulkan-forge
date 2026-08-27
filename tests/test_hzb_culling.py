from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from _tessella_evidence import record_tessella_result
from forge3d.diagnostics import culling_stats, render_certificate
from forge3d.terrain_params import make_terrain_params_config

from _terrain_runtime import _write_test_hdr, terrain_rendering_available
from test_terrain_clipmap_streaming import _make_params, _render_rgba


requires_terrain = pytest.mark.skipif(
    not terrain_rendering_available(),
    reason="Terrain rendering runtime unavailable on this adapter",
)

WIN2_SIZE = (3840, 2160)
HZB_SPEEDUP_TARGET = 1.8
HZB_SPEEDUP_GATE = 1.7
# Exercise the production-default clipmap density. The shared terrain test
# helper deliberately uses a 32x32 fast-test mesh; at that density the fixed
# HZB dispatch overhead dominates after the canyon culls ~95% of the tiles and
# the historical fixture measured only 1.70-1.83x. Keep 1.8x as the reported
# target and 1.7x as the variance-tolerant regression floor. Production uses
# 64x64 for both the center and rings, while the same conservative HZB
# partition and bitwise baseline comparison below remain authoritative.
WIN2_CAMERA_MODE = "clipmap:4:64:64:10:0.3"


def _canyon_dem(size: int = 96) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, _ = np.meshgrid(x, x)
    return np.clip(np.abs(xx) * 3.0, 0.0, 1.0).astype(np.float32)


def _win2_params(culling: str):
    return _make_params(
        camera_mode=WIN2_CAMERA_MODE,
        culling=culling,
        terrain_span=50.0,
        cam_radius=7.0,
        z_scale=50.0,
        theta_deg=88.0,
        phi_deg=0.0,
        size_px=WIN2_SIZE,
    )


def _terrain_main_gpu_ms() -> float:
    passes = render_certificate(sign=False)["passes"]
    return float(
        next(item["gpu_ms"] for item in passes if item["label"] == "terrain.main")
    )


def _tessella_timing_required() -> bool:
    return os.environ.get("FORGE3D_TESSELLA_TIMING_REQUIRED") == "1"


def _function_ast(path: Path, name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_tessella_acceptance_uses_cold_single_samples_without_residency_warmup():
    root = Path(__file__).resolve().parent
    hzb_gate = _function_ast(
        root / "test_hzb_culling.py",
        "test_two_phase_hzb_is_bitwise_identical_to_unculled_render",
    )
    hzb_render_calls = [
        node
        for node in ast.walk(hzb_gate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_render_rgba"
    ]
    assert len(hzb_render_calls) == 2
    assert not any(
        isinstance(node, (ast.For, ast.While, ast.comprehension))
        for node in ast.walk(hzb_gate)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "median"
        for node in ast.walk(hzb_gate)
    )

    flythrough_gate = _function_ast(
        root / "test_flythrough_popping.py",
        "test_600_frame_streaming_flythrough_has_no_pop_or_crack",
    )
    calls = [node for node in ast.walk(flythrough_gate) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Name)
        and call.func.id == "_warm_streaming_to_full_residency"
        for call in calls
    )
    stream_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "stream_height_tiles"
    ]
    assert len(stream_calls) == 1
    frame_loop = next(
        node
        for node in ast.walk(flythrough_gate)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "index"
    )
    assert stream_calls[0] in ast.walk(frame_loop)
    warmup_values = [
        value
        for node in ast.walk(flythrough_gate)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "streaming_warmup_steps"
    ]
    assert len(warmup_values) == 1
    assert isinstance(warmup_values[0], ast.Constant)
    assert warmup_values[0].value == 0


def test_culling_parameter_contract():
    required = {
        "size_px": (64, 64),
        "render_scale": 1.0,
        "terrain_span": 10.0,
        "msaa_samples": 1,
        "z_scale": 1.0,
        "exposure": 1.0,
        "domain": (0.0, 1.0),
    }
    params = f3d.TerrainRenderParams(
        make_terrain_params_config(**required, culling="hzb_two_phase")
    )
    assert params.culling == "hzb_two_phase"
    with pytest.raises(ValueError, match="culling must be one of"):
        make_terrain_params_config(**required, culling="not-a-culling-mode")


def test_tessella_hardware_requirement_does_not_select_timing_gate(monkeypatch):
    monkeypatch.delenv("FORGE3D_TESSELLA_TIMING_REQUIRED", raising=False)
    monkeypatch.setenv("FORGE3D_TESSELLA_REQUIRED_GPU", "1")
    assert not _tessella_timing_required()

    monkeypatch.setenv("FORGE3D_TESSELLA_TIMING_REQUIRED", "1")
    assert _tessella_timing_required()


def test_hzb_reuses_prometheus_terrain_minmax_pyramid():
    root = Path(__file__).resolve().parents[1]
    geometry = (root / "src/terrain/renderer/geometry.rs").read_text(encoding="utf-8")
    assert "TerrainMinMaxPyramid::from_heightfield" in geometry
    assert "build_minmax_mips(" not in geometry


def test_two_phase_conservativeness_is_backed_by_the_real_shader():
    """The CPU conservativeness model in `src/terrain/culling/two_phase.rs` was
    a hand-written re-implementation of `hzb_cull.wgsl` and would have survived
    any change to the shader. Lock in both halves of the fix:

    1. `cpu_predicate_tracks_the_compiled_wgsl_source` parses the background
       cutoff, the depth slack and the MAX reduce out of the same `include_str!`
       the pipeline compiles, so it fails on shader drift with no GPU at all.
    2. `hzb_cull_shader_matches_the_cpu_occlusion_predicate` dispatches the real
       kernel (through the production layout/pipeline constructors) over a
       synthetic occluder set and compares the accept/reject partition. It is
       `#[ignore]`d like the 1,000-camera LOD differential and runs in the
       TESSELLA hardware lane.
    """
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/terrain/culling/two_phase.rs").read_text(encoding="utf-8")
    assert 'include_str!("../../shaders/hzb_cull.wgsl")' in source
    assert "const HZB_CULL_SOURCE: &str" in source
    assert "fn cpu_predicate_tracks_the_compiled_wgsl_source()" in source
    assert "fn hzb_cull_shader_matches_the_cpu_occlusion_predicate()" in source
    # The differential must build its pipeline from the production constructors,
    # not a bespoke copy of the bind-group layout.
    assert source.count("create_cull_layout(device)") >= 2
    assert source.count("create_cull_pipeline(device, &layout)") >= 2


@pytest.mark.gpu_lane
@requires_terrain
def test_two_phase_hzb_is_bitwise_identical_to_unculled_render():
    require_performance = _tessella_timing_required()

    with tempfile.TemporaryDirectory() as td:
        hdr_path = Path(td) / "probe.hdr"
        _write_test_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        dem = _canyon_dem()

        baseline_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        baseline = _render_rgba(baseline_renderer, _win2_params("none"), dem, ibl)
        if require_performance:
            baseline_gpu_ms = _terrain_main_gpu_ms()

        culled_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        culled = _render_rgba(
            culled_renderer, _win2_params("hzb_two_phase"), dem, ibl
        )
        if require_performance:
            culled_gpu_ms = _terrain_main_gpu_ms()

    np.testing.assert_array_equal(culled, baseline)
    stats = culling_stats()
    assert stats["cull_percent"] >= 60.0, stats
    assert stats["phase1_drawn"] + stats["phase2_recovered"] == stats["final_drawn"]
    certificate = render_certificate(sign=False)
    assert "hzb_cull" in certificate["engine"]["wgsl_module_hashes"]
    assert "p5.hzb.build.shader" in certificate["engine"]["wgsl_module_hashes"]

    if not require_performance:
        return

    assert "timestamp_query" in certificate["capabilities"]["granted"], certificate[
        "capabilities"
    ]
    assert baseline_gpu_ms > 0.0 and culled_gpu_ms > 0.0
    assert baseline_gpu_ms / culled_gpu_ms >= HZB_SPEEDUP_GATE, {
        "baseline_gpu_ms": baseline_gpu_ms,
        "culled_gpu_ms": culled_gpu_ms,
        "gate": HZB_SPEEDUP_GATE,
    }
    record_tessella_result(
        "hzb_occlusion",
        {
            "cull_percent": float(stats["cull_percent"]),
            "frustum_passing": int(stats["frustum_passing"]),
            "phase1_drawn": int(stats["phase1_drawn"]),
            "phase1_rejected": int(stats["phase1_rejected"]),
            "phase2_recovered": int(stats["phase2_recovered"]),
            "final_drawn": int(stats["final_drawn"]),
            "baseline_gpu_ms": baseline_gpu_ms,
            "culled_gpu_ms": culled_gpu_ms,
            "speedup": baseline_gpu_ms / culled_gpu_ms,
            "speedup_target": HZB_SPEEDUP_TARGET,
            "speedup_gate": HZB_SPEEDUP_GATE,
            "timestamp_query": True,
            "bitwise_identical": True,
        },
    )


@pytest.mark.gpu_lane
@requires_terrain
def test_fresh_hzb_recovers_tiles_rejected_by_camera_history():
    with tempfile.TemporaryDirectory() as td:
        hdr_path = Path(td) / "probe.hdr"
        _write_test_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        dem = _canyon_dem()
        size = (640, 360)

        culled_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        warm = _make_params(
            culling="hzb_two_phase",
            terrain_span=50.0,
            cam_radius=7.0,
            z_scale=50.0,
            theta_deg=88.0,
            phi_deg=0.0,
            size_px=size,
        )
        current = _make_params(
            culling="hzb_two_phase",
            terrain_span=50.0,
            cam_radius=7.0,
            z_scale=50.0,
            theta_deg=88.0,
            phi_deg=20.0,
            size_px=size,
        )
        _render_rgba(culled_renderer, warm, dem, ibl)
        culled = _render_rgba(culled_renderer, current, dem, ibl)
        stats = culling_stats()

        baseline_renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        baseline = _render_rgba(
            baseline_renderer,
            _make_params(
                culling="none",
                terrain_span=50.0,
                cam_radius=7.0,
                z_scale=50.0,
                theta_deg=88.0,
                phi_deg=20.0,
                size_px=size,
            ),
            dem,
            ibl,
        )

    np.testing.assert_array_equal(culled, baseline)
    assert stats["phase1_rejected"] > 0, stats
    assert stats["phase2_recovered"] > 0, stats
    record_tessella_result(
        "hzb_history_recovery",
        {
            "phase1_rejected": int(stats["phase1_rejected"]),
            "phase2_recovered": int(stats["phase2_recovered"]),
            "bitwise_identical": True,
        },
    )
