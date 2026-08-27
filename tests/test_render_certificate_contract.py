from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import forge3d as f3d
from forge3d import determinism, geometry, path_tracing, terrain_demo
from forge3d import media
from forge3d.helpers.offscreen import render_offscreen_rgba
from forge3d import sdf
from forge3d.legend import Legend
from forge3d.lighting import RestirDI
from forge3d.map_scene import MapScene
from forge3d.map_plate import MapPlate
from forge3d.north_arrow import NorthArrow
from forge3d.offline import render_offline
from forge3d.scale_bar import ScaleBar
from forge3d.sdf import HybridRenderer, SdfPrimitive, SdfScene
import forge3d.smoke as smoke_module
from forge3d.smoke import SmokeDomain
from forge3d.vector import VectorScene
import forge3d._forge3d as _native
from forge3d.denoise import atrous_denoise
from forge3d.denoise_oidn import oidn_denoise
from forge3d.bench import run_benchmark
from forge3d.export import export_pdf, export_svg
from forge3d.viewer import ViewerHandle
from forge3d.widgets import ViewerWidget


# Public callables matching `render_*` (or documented pixel producers) that are
# deliberately OUTSIDE CENSOR's render definition ("every render: offscreen,
# golden, MapScene, PT reference" — 14-censor.md). Each entry carries the reason
# it does not take certificate=. Additions require the same justification.
DOCUMENTED_EXCLUSIONS = {
    # Pure I/O of an existing array — no rendering executes.
    "numpy_to_png": "array-to-PNG conversion, no render",
    "png_to_numpy": "PNG-to-array conversion, no render",
    # Image-space filters over an already-rendered image.
    "oidn_denoise": "post-hoc denoise filter over an existing image",
    "atrous_denoise": "post-hoc denoise filter over an existing image",
    # Composition of images that were themselves rendered under certificates.
    "MapPlate.compose": "composites pre-rendered certified images",
    "MapPlate.export_png": "writes a composed image, no render",
    "MapPlate.export_jpeg": "writes a composed image, no render",
    # Interactive viewer subprocess: CENSOR certifies offscreen renders.
    "ViewerHandle.snapshot": "interactive viewer subprocess, outside offscreen scope",
    "ViewerHandle.render_animation": "interactive viewer subprocess, outside offscreen scope",
    "ViewerWidget.snapshot": "interactive viewer/widget snapshot, outside offscreen scope",
    # Vector (non-pixel) exports.
    "export_svg": "vector output, not a pixel render",
    "export_pdf": "vector output, not a pixel render",
    # Benchmark harness (never a product image).
    "run_benchmark": "benchmark timing harness",
    # Certificate-report getter: returns the last render's execution report,
    # renders nothing itself.
    "render_execution_report": "execution-report getter, produces no pixels",
    # CPU validation fixture used as an input to the separately certified PT
    # closure harness; it is not a product render and submits no GPU work.
    "atmosphere_generate_environment": "CPU validation fixture, no GPU render",
}


RENDER_ENTRYPOINTS = {
    "Scene.render_rgba": f3d.Scene.render_rgba,
    "Scene.render_png": f3d.Scene.render_png,
    "render_debug_pattern_frame": _native.render_debug_pattern_frame,
    "render_brdf_tile": _native.render_brdf_tile,
    "render_brdf_tile_overrides": _native.render_brdf_tile_overrides,
    "TerrainRenderer.render_terrain_pbr_pom": f3d.TerrainRenderer.render_terrain_pbr_pom,
    "TerrainRenderer.render_with_aov": f3d.TerrainRenderer.render_with_aov,
    "render_adjudication_pair": f3d.render_adjudication_pair,
    "hybrid_render_terrain_reference": f3d.hybrid_render_terrain_reference,
    "hybrid_render_aether_spectral_reference": f3d.hybrid_render_aether_spectral_reference,
    "native.hybrid_render_aether_spectral_reference": _native.hybrid_render_aether_spectral_reference,
    "render_offscreen_rgba": render_offscreen_rgba,
    "Renderer.render_triangle_rgba": f3d.Renderer.render_triangle_rgba,
    "Renderer.render_triangle_png": f3d.Renderer.render_triangle_png,
    "MapScene.render": MapScene.render,
    "render_offline": render_offline,
    "determinism.render_reference": determinism.render_reference,
    "PathTracer.render_rgba": path_tracing.PathTracer.render_rgba,
    "PathTracer.render_progressive": path_tracing.PathTracer.render_progressive,
    "path_tracing.render_aovs": path_tracing.render_aovs,
    "path_tracing.render_rgba": path_tracing.render_rgba,
    "path_tracing.hybrid_render_terrain_reference": path_tracing.hybrid_render_terrain_reference,
    "media.render_volumetric_reference": media.render_volumetric_reference,
    "HybridRenderer.render_sdf_scene": HybridRenderer.render_sdf_scene,
    "sdf.render_simple_scene": sdf.render_simple_scene,
    "Legend.render": Legend.render,
    "NorthArrow.render": NorthArrow.render,
    "ScaleBar.render": ScaleBar.render,
    "RestirDI.render_frame": RestirDI.render_frame,
    "geometry.instance_mesh_gpu_render": geometry.instance_mesh_gpu_render,
    "SmokeDomain.render_rgba": SmokeDomain.render_rgba,
    "SmokeDomain.render_projection_rgba": SmokeDomain.render_projection_rgba,
    "VectorScene.render_oit": VectorScene.render_oit,
    "VectorScene.render_pick_map": VectorScene.render_pick_map,
    "VectorScene.render_oit_and_pick": VectorScene.render_oit_and_pick,
    "vector_render_oit_py": f3d.vector_render_oit_py,
    "vector_render_oit_edl_py": f3d.vector_render_oit_edl_py,
    "vector_render_pick_map_py": f3d.vector_render_pick_map_py,
    "vector_render_oit_and_pick_py": f3d.vector_render_oit_and_pick_py,
    "vector_render_polygons_fill_py": f3d.vector_render_polygons_fill_py,
    "vector_render_analytic_py": f3d.vector_render_analytic_py,
    "vector_oit_and_pick_demo": f3d.vector_oit_and_pick_demo,
}


def test_brdf_certificate_contract_is_in_public_stub() -> None:
    stub = (Path(f3d.__file__).with_name("__init__.pyi")).read_text(encoding="utf-8")
    for name in ("render_brdf_tile", "render_brdf_tile_overrides"):
        declaration = stub.split(f"def {name}(", 1)
        assert len(declaration) == 2, f"{name} missing from __init__.pyi"
        assert "certificate:" in declaration[1].split(") ->", 1)[0]


def test_public_render_entrypoints_expose_certificate_keyword() -> None:
    missing = [
        name
        for name, entrypoint in RENDER_ENTRYPOINTS.items()
        if "certificate" not in inspect.signature(entrypoint).parameters
    ]
    assert not missing, f"render entrypoints missing certificate= contract: {missing}"


# Render entrypoints that are certified but NOT backed by the ANAMNESIS graph
# cache. Each entry carries the reason, on the same terms as
# DOCUMENTED_EXCLUSIONS above.
CACHE_EXCLUSIONS = {
    # LIMES owns a bounded internal compiled-scene cache rather than a
    # caller-supplied ANAMNESIS render graph, so `vector_render_analytic_py`
    # takes certificate= but not cache=.
    "vector_render_analytic_py": "internal compiled-scene cache, no ANAMNESIS graph",
}


def test_public_render_entrypoints_expose_anamnesis_cache_keyword() -> None:
    missing = [
        name
        for name, entrypoint in RENDER_ENTRYPOINTS.items()
        if name not in CACHE_EXCLUSIONS
        and "cache" not in inspect.signature(entrypoint).parameters
    ]
    assert not missing, f"render entrypoints missing cache= contract: {missing}"


def test_cache_exclusions_are_still_certificate_bearing() -> None:
    """An entrypoint may opt out of the cache, never out of the certificate."""

    for name in CACHE_EXCLUSIONS:
        entrypoint = RENDER_ENTRYPOINTS[name]
        assert "certificate" in inspect.signature(entrypoint).parameters, name


def test_brdf_tile_emits_a_live_certified_pass() -> None:
    import pytest

    if not f3d.has_gpu():
        pytest.skip("BRDF tile render requires a GPU adapter")

    from forge3d.diagnostics import render_certificate

    rgba = f3d.render_brdf_tile("lambert", 0.4, 32, 32, certificate=True)
    certificate = render_certificate(sign=False)

    assert rgba.shape == (32, 32, 4)
    assert [entry["label"] for entry in certificate["passes"]] == ["brdf.tile"]
    if "timestamp_query" not in certificate["capabilities"]["granted"]:
        assert certificate["passes"][0]["gpu_ms"] == 0.0
    elif certificate["passes"][0]["gpu_ms"] == 0.0:
        assert {
            "kind": "timing_unavailable",
            "name": "brdf.tile",
            "consequence": "timestamp query resolved invalid; certificate passes[].gpu_ms reported as 0",
        } in certificate["degradations"]
    else:
        assert certificate["passes"][0]["gpu_ms"] > 0.0
    assert set(certificate["engine"]["wgsl_module_hashes"]) == {"brdf_tile.shader"}


def test_render_surface_sweep_has_no_uncertified_entrypoints() -> None:
    """Auto-discovery guard (CENSOR audit F-05): the hardcoded enumeration above
    cannot notice a NEWLY ADDED public render entrypoint. Sweep every public
    callable named `render_*` on `forge3d` and `forge3d._forge3d`: each must
    either accept `certificate=` or carry a documented exclusion."""
    candidates: dict[str, object] = {}
    for module in (f3d, _native, media):
        for name in dir(module):
            if name.startswith("render_"):
                obj = getattr(module, name)
                if callable(obj):
                    candidates.setdefault(name, obj)

    offenders = []
    for name, obj in sorted(candidates.items()):
        if name in DOCUMENTED_EXCLUSIONS:
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            offenders.append(f"{name} (signature not introspectable)")
            continue
        if "certificate" not in params:
            offenders.append(name)
    assert offenders == [], (
        "public render_* callables lack both a certificate= contract and a "
        f"documented exclusion: {offenders}"
    )


def _discover_public_render_surface() -> dict[str, object]:
    """Discover module functions and class methods instead of trusting a list."""
    modules = (f3d, _native, determinism, geometry, path_tracing, terrain_demo, sdf, media)
    candidates: dict[str, object] = {}
    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            if name.startswith("render_") and callable(obj):
                candidates.setdefault(name, obj)
    classes = {
        cls.__name__: cls
        for cls in (
            f3d.Scene,
            f3d.Renderer,
            f3d.TerrainRenderer,
            MapScene,
            MapPlate,
            Legend,
            NorthArrow,
            ScaleBar,
            path_tracing.PathTracer,
            HybridRenderer,
            RestirDI,
            SmokeDomain,
            VectorScene,
            ViewerHandle,
            ViewerWidget,
        )
    }
    for class_name, cls in classes.items():
        for method_name in dir(cls):
            if not method_name.startswith("render"):
                continue
            obj = getattr(cls, method_name)
            if callable(obj):
                candidates[f"{class_name}.{method_name}"] = obj
    return candidates


def test_render_surface_sweep_has_no_uncached_entrypoints() -> None:
    """ANAMNESIS API gate: newly added render surfaces cannot evade cache=."""
    offenders = []
    for name, obj in sorted(_discover_public_render_surface().items()):
        if name in DOCUMENTED_EXCLUSIONS or name.rsplit(".", 1)[-1] in DOCUMENTED_EXCLUSIONS:
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            offenders.append(f"{name} (signature not introspectable)")
            continue
        if "cache" not in params:
            offenders.append(name)
    assert offenders == [], (
        "public render callables lack both a cache= contract and a documented "
        f"non-render exclusion: {offenders}"
    )


def test_documented_exclusions_explain_their_certificate_scope() -> None:
    exclusions = {
        "MapPlate.compose": MapPlate.compose,
        "MapPlate.export_png": MapPlate.export_png,
        "MapPlate.export_jpeg": MapPlate.export_jpeg,
        "oidn_denoise": oidn_denoise,
        "atrous_denoise": atrous_denoise,
        "numpy_to_png": f3d.numpy_to_png,
        "png_to_numpy": f3d.png_to_numpy,
        "ViewerHandle.snapshot": ViewerHandle.snapshot,
        "ViewerWidget.snapshot": ViewerWidget.snapshot,
        "export_svg": export_svg,
        "export_pdf": export_pdf,
        "run_benchmark": run_benchmark,
        "atmosphere_generate_environment": f3d.atmosphere_generate_environment,
    }
    for name, entrypoint in exclusions.items():
        assert name in DOCUMENTED_EXCLUSIONS
        assert "Outside CENSOR's render-certificate scope" in (entrypoint.__doc__ or ""), name

    native_diagnostics = (
        Path(__file__).resolve().parents[1] / "src" / "py_functions" / "diagnostics.rs"
    ).read_text(encoding="utf-8")
    assert "Outside CENSOR's render-certificate scope" in native_diagnostics


def test_vector_render_certificate_is_fresh_and_exact():
    if not f3d.has_gpu() or not f3d.is_weighted_oit_available():
        import pytest

        pytest.skip("weighted OIT GPU path unavailable")

    from forge3d.diagnostics import render_certificate

    scene = VectorScene()
    scene.add_point(0.0, 0.0)
    rgba = scene.render_oit(16, 16, certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (16, 16, 4)
    assert [entry["label"] for entry in cert["passes"]] == [
        "vector.oit",
        "vector.oit.compose",
    ]
    assert set(cert["engine"]["wgsl_module_hashes"]) == {
        "point_instanced.wgsl",
        "vf.Vector.OIT.Compose",
    }


def test_cloud_render_keeps_following_vector_allocation_usable():
    if not f3d.has_gpu() or not f3d.is_weighted_oit_available():
        import pytest

        pytest.skip("cloud/vector GPU path unavailable")

    baseline = f3d.Scene(32, 32).render_rgba()
    scene = f3d.Scene(32, 32)
    scene.enable_clouds("low")
    scene.set_realtime_cloud_density(1.0)
    scene.set_realtime_cloud_coverage(1.0)
    scene.set_cloud_render_mode("volumetric")
    cloud_rgba = scene.render_rgba()

    changed = np.any(cloud_rgba[:, :, :3] != baseline[:, :, :3], axis=2)
    assert np.count_nonzero(changed) >= 100

    vector = VectorScene()
    vector.add_point(0.0, 0.0)
    rgba = vector.render_oit(16, 16)

    assert rgba.shape == (16, 16, 4)


def test_native_sdf_render_uses_the_supplied_scene_and_emits_certificate(monkeypatch):
    if not sdf.NATIVE_AVAILABLE or not hasattr(sdf.forge3d_native, "hybrid_render"):
        import pytest

        pytest.skip("native SDF renderer unavailable")

    from forge3d.diagnostics import render_certificate

    monkeypatch.setattr(sdf, "USE_NATIVE_SDF", True)
    scene = SdfScene()
    scene.add_primitive(SdfPrimitive.sphere((0.0, 0.0, 0.0), 1.0, 1))

    rgba = HybridRenderer(17, 17).render_sdf_scene(scene, certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (17, 17, 4)
    assert rgba[8, 8].tolist() == [204, 51, 51, 255]
    assert [entry["label"] for entry in cert["passes"]] == ["sdf.native_cpu"]
    assert cert["adapter"]["backend"] == "cpu"
    assert cert["engine"]["wgsl_module_hashes"] == {}
    assert cert["degradations"] == []


def test_polygon_fill_certificate_names_only_the_polygon_shader():
    if not f3d.has_gpu():
        import pytest

        pytest.skip("GPU path unavailable")

    from forge3d.diagnostics import render_certificate

    exterior = np.asarray(
        [[-0.8, -0.8], [0.8, -0.8], [0.0, 0.8]], dtype=np.float64
    )
    rgba = f3d.vector_render_polygons_fill_py(
        16,
        16,
        [exterior],
        coordinates_are_ndc=True,
        certificate=True,
    )
    cert = render_certificate(sign=False)

    assert rgba.shape == (16, 16, 4)
    assert [entry["label"] for entry in cert["passes"]] == ["vector.polygon_fill"]
    assert set(cert["engine"]["wgsl_module_hashes"]) == {"polygon_fill.wgsl"}


def test_native_smoke_certificate_uses_cpu_identity():
    from forge3d.diagnostics import render_certificate

    density = np.zeros((3, 3, 3), dtype=np.float32)
    density[1, 1, 1] = 1.0
    domain = SmokeDomain.from_density(density)
    rgba = domain.render_projection_rgba(8, 8, certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (8, 8, 4)
    assert [entry["label"] for entry in cert["passes"]] == ["smoke.cpu_projection"]
    assert cert["adapter"]["backend"] == "cpu"
    assert cert["engine"]["wgsl_module_hashes"] == {}
    if smoke_module._HAS_NATIVE_SMOKE:
        assert cert["degradations"] == []
    else:
        assert {(item["kind"], item["name"]) for item in cert["degradations"]} == {
            ("cpu_fallback", "smoke.render")
        }


def test_fallback_renderer_discloses_cpu_degradation():
    from forge3d.diagnostics import render_certificate

    rgba = f3d.Renderer(8, 8).render_triangle_rgba(certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (8, 8, 4)
    assert [entry["label"] for entry in cert["passes"]] == ["renderer.cpu_triangle"]
    assert cert["adapter"]["backend"] == "cpu"
    assert {(entry["kind"], entry["name"]) for entry in cert["degradations"]} == {
        ("cpu_fallback", "renderer.triangle")
    }


def test_instanced_mesh_certificate_names_only_the_instancing_shader():
    if not f3d.has_gpu() or not geometry.gpu_instancing_available():
        import pytest

        pytest.skip("GPU instancing path unavailable")

    from forge3d.diagnostics import render_certificate

    mesh = geometry.MeshBuffers(
        positions=np.asarray(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float32,
        ),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        indices=np.asarray([[0, 1, 2]], dtype=np.uint32),
    )
    transforms = np.eye(4, dtype=np.float32).reshape(1, 16)

    rgba = geometry.instance_mesh_gpu_render(
        mesh, transforms, 16, 16, certificate=True
    )
    cert = render_certificate(sign=False)

    assert rgba.shape == (16, 16, 4)
    assert [entry["label"] for entry in cert["passes"]] == [
        "geometry.instanced_mesh"
    ]
    assert set(cert["engine"]["wgsl_module_hashes"]) == {"mesh_instanced_shader"}


def test_terrain_sequence_never_substitutes_triangle_placeholder(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(terrain_demo.f3d, "has_gpu", lambda: False)
    with pytest.raises(RuntimeError, match="fallback triangle placeholder has been removed"):
        terrain_demo.render_sunrise_to_noon_sequence(
            dem_path=tmp_path / "missing-dem.tif",
            hdr_path=tmp_path / "missing-env.hdr",
            output_dir=tmp_path / "frames",
        )
