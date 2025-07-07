import pytest

import vulkan_forge as vf
from vulkan_forge.perf_helpers import create_performance_mesh


@pytest.fixture(autouse=True)
def _patch_performance_helpers(monkeypatch):
    import tests.test_performance as tp

    monkeypatch.setattr(
        tp.TestMeshPerformance,
        "create_performance_mesh",
        staticmethod(create_performance_mesh),
        raising=False,
    )
    monkeypatch.setattr(
        tp.TestRenderingPerformance,
        "create_performance_mesh",
        staticmethod(create_performance_mesh),
        raising=False,
    )
    monkeypatch.setattr(
        vf, "create_performance_mesh", create_performance_mesh, raising=False
    )

    def _estimate_gpu_performance(mesh: vf.Mesh, target_fps: int = 1000) -> dict:
        vertex_count = mesh.data.vertex_count
        triangle_count = mesh.data.triangle_count
        vertex_size = mesh.data.vertex_size_bytes
        index_size = mesh.data.index_size_bytes

        vertex_processing_time = vertex_count / 100_000_000
        triangle_processing_time = triangle_count / 100_000_000
        memory_transfer_time = (vertex_size + index_size) / (500 * 1024**3)

        estimated_frame_time = max(
            vertex_processing_time, triangle_processing_time, memory_transfer_time
        )
        estimated_fps = (
            1.0 / estimated_frame_time if estimated_frame_time > 0 else float("inf")
        )

        return {
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
            "vertex_size_mb": vertex_size / (1024**2),
            "estimated_frame_time_ms": estimated_frame_time * 1000,
            "estimated_fps": estimated_fps,
            "meets_target": estimated_fps >= target_fps,
            "performance_ratio": estimated_fps / target_fps,
            "bottleneck": (
                "vertices"
                if vertex_processing_time == estimated_frame_time
                else (
                    "triangles"
                    if triangle_processing_time == estimated_frame_time
                    else "memory"
                )
            ),
        }

    monkeypatch.setattr(
        tp.TestRenderingPerformance,
        "estimate_gpu_performance",
        staticmethod(_estimate_gpu_performance),
        raising=False,
    )
    yield
