"""P0.1 API Contract Tests -- Baseline snapshot of the public native API surface.

These tests lock the **current** (pre-consolidation) contract so that
refactoring in P0.2-P0.5 cannot silently remove or rename symbols that
downstream code depends on.

Tested contracts:
  - _forge3d native module loads and exposes expected classes/functions
  - Scene class has render_rgba and set_msaa_samples instance methods
  - Scene class exposes key feature-enable/disable methods
  - Key native classes are registered and accessible
  - TBN mesh functions are exported and return correct dict structure (P0.4)
  - Previously-orphaned pyclass types (Frame, SdfPrimitive, etc.) are now registered (P0.3)

Each test is minimal and non-trivial: it asserts something specific about
the current API surface, not just that imports succeed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE, get_native_module


# ---------------------------------------------------------------------------
# Skip entire module when native extension is absent (e.g., pure-Python CI)
# ---------------------------------------------------------------------------
if not NATIVE_AVAILABLE:
    pytest.skip(
        "Contract tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

_native = get_native_module()


def _try_create_terrain_spike():
    try:
        return _native.TerrainSpike(64, 64)
    except Exception:
        return None


_TERRAIN_SPIKE_AVAILABLE = (
    f3d.has_gpu()
    and hasattr(_native, "TerrainSpike")
    and _try_create_terrain_spike() is not None
)


# ===========================================================================
# Section 1: Core native module symbols
# ===========================================================================
class TestNativeModuleSymbols:
    """Verify that the native module exports the expected top-level symbols."""

    # ---- Registered classes (m.add_class in lib.rs) ----

    EXPECTED_CLASSES = [
        "Scene",
        "Session",
        "Colormap1D",
        "MaterialSet",
        "IBL",
        "OverlayLayer",
        "TerrainRenderParams",
        "TerrainRenderer",
        "AovFrame",
        "HdrFrame",
        "OfflineBatchResult",
        "OfflineMetrics",
        "CameraKeyframe",
        "CameraAnimation",
        "CameraState",
        "ClipmapConfig",
        "ClipmapMesh",
        "CogDataset",
        "SunPosition",
        "AtmosphereLutHandle",
        # P0.3: Previously-orphaned classes now registered
        "Frame",
        "SdfPrimitive",
        "SdfScene",
        "SdfSceneBuilder",
        "SmokeDomain",
        "SmokeEmitter",
        "SmokeStepSettings",
        "SmokeRenderSettings",
        "RasterInfo",
        "VectorInfo",
        "AffineTransform",
        "CrsTransform",
        # CARTOGRAPHER-PRIME: grounded optimal-declutter rationale
        "LabelRationale",
        # CENSOR: typed GPU-error exceptions
        "MemoryBudgetExceeded",
        "DegradedCapability",
        # MENSURA M-05: structured raster-reprojection failure exception
        "TransformFailed",
        # LITTERA: immutable native shaping result
        "ShapedText",
    ]

    @pytest.mark.parametrize("cls_name", EXPECTED_CLASSES)
    def test_registered_class_exists(self, cls_name: str):
        """Each registered pyclass must be accessible on the native module."""
        assert hasattr(_native, cls_name), (
            f"_forge3d.{cls_name} not found -- "
            f"was it removed from m.add_class in lib.rs?"
        )
        obj = getattr(_native, cls_name)
        assert isinstance(obj, type), (
            f"_forge3d.{cls_name} should be a class, got {type(obj)}"
        )

    # ---- Registered free functions (wrap_pyfunction in lib.rs) ----

    EXPECTED_FUNCTIONS = [
        "open_viewer",
        "open_terrain_viewer",
        "enumerate_adapters",
        "device_probe",
        "sun_position",
        "sun_position_utc",
        "solar_position",
        "terrain_viewshed",
        "terrain_shadow_mask",
        "terrain_shadow_tip",
        "astro_body_position",
        "astro_moon_phase",
        "astro_delta_t_seconds",
        "astro_sidereal_time",
        "astro_refraction_arcminutes",
        "sky_set_observation",
        "astro_validation_metrics",
        "clipmap_generate_py",
        "engine_info",
        "hybrid_render",
        "configure_csm",
        "global_memory_metrics",
        "set_memory_budget_policy",
        "get_memory_budget_policy",
        "io_import_gltf_with_materials_py",
        "project_3d_to_2d",
        "project_2d_to_screen",
        "decode_b3dm_py",
        "tiles3d_traverse_py",
        "decode_pnts_py",
        "vector_render_oit_edl_py",
        "vector_render_analytic_py",
        "vector_coverage_primitives_py",
        "read_laz_point_attributes",
        "copc_read_node_points",
        # DUPLA: verified GPU double-float proof surface
        "dd_selftest",
        "dd_harness",
        "dd_jitter_demo",
        # AEQUITAS: PT-vs-raster adjudication pair
        "render_adjudication_pair",
        # PROMETHEUS: GPU terrain path-traced reference
        "hybrid_render_terrain_reference",
        "hybrid_render_aether_spectral_reference",
        # AETHER: spectral atmosphere public surface
        "atmosphere_bake_luts",
        "atmosphere_spectral_to_linear_rgb",
        "atmosphere_generate_environment",
        "atmosphere_reference_aerial",
        # VERITAS: per-pixel cryptographic provenance
        "seal_provenance",
        "verify_provenance",
        "verify_license_signature",
        "license_public_key_hex",
        # P0.4: TBN mesh generation
        "mesh_generate_cube_tbn",
        "mesh_generate_plane_tbn",
        # CARTOGRAPHER-PRIME: bounded-optimal label declutter
        "_reserve_label_depth_host_allocation",
        "declutter",
        "declutter_optimal",
        "anamnesis_leaf_key",
        "anamnesis_pass_key",
        "anamnesis_engine_fingerprint",
        "anamnesis_store_verify",
        "anamnesis_store_gc",
        "anamnesis_store_put_leaf",
        "anamnesis_store_get",
        "anamnesis_restore_rgba8",
        # G-002a1: Rust-backed GIS raster metadata/write foundation
        "read_raster_info",
        "read_raster",
        "write_raster",
        "read_vector",
        "reproject_vector",
        "geometry_type",
        "vector_schema",
        "feature_count",
        "vector_crs",
        "vector_bounds",
        "validate_geometry",
        "is_valid",
        "repair_geometry",
        "geometry_measure",
        "measure_geometries",
        # MENSURA geodesy surface (src/py_functions/geodesy.rs)
        "body_info",
        "areoid_undulation",
        "geoid_undulation",
        "orthometric_to_ellipsoidal",
        "ellipsoidal_to_orthometric",
        "geodesic_inverse",
        "geodesic_direct",
        "wgs84_to_ecef",
        "ecef_to_wgs84",
        "dem_orthometric_to_ellipsoidal",
        "geometry_centroid",
        "representative_point",
        "interpolate_line",
        "union_geometries",
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        "dissolve_vector",
        "buffer_geometry",
        "clip_vector",
        "intersect_vectors",
        "simplify_geometry",
        "load_boundary",
        "rasterize_vectors",
        "geometry_mask",
        "mask_raster",
        "normalize_raster",
        "classify_raster",
        # G-002b: Rust-backed GIS CRS/affine/warp operations
        "parse_crs",
        "inspect_crs",
        "raster_crs",
        "assign_crs",
        "create_crs_transformer",
        "transform_bounds",
        "web_mercator_bounds",
        "raster_transform",
        "transform_from_origin",
        "transform_from_bounds",
        "array_bounds",
        "raster_bounds",
        "raster_resolution",
        "validate_transform",
        "pixel_convention",
        "rowcol",
        "xy",
        "index",
        "apply_nodata",
        "read_raster_mask",
        "resample_raster",
        "assert_grid_compatible",
        "align_raster_grid",
        "align_raster_to",
        "reproject_raster",
        "calculate_default_transform",
        "warped_vrt_info",
        "window_from_bounds",
        "read_raster_window",
        "window_transform",
        "bounds",
        # G-002 Later: domain and remote helpers
        "fetch_remote_geodata",
        "cache_geodata",
        "fetch_vector",
        "read_cog",
        "slippy_tile_index",
        "query_osm_features",
        "parse_osm_features",
        "load_context_vectors",
        "prepare_osm_scene",
        "prepare_dem",
        "prepare_terrain_derivatives",
        "read_gridded_dataset",
        "subset_grid",
        "decode_terrarium_dem",
        "build_terrarium_dem",
        "prepare_landcover_raster",
        "prepare_population_raster",
        "load_building_footprints",
        "extract_building_heights",
        "estimate_local_utm",
        # LITTERA: native text pipeline registration seam
        "text_shape",
        "rasterize_shaped_run",
        "bake_msdf_atlas",
        "bake_msdf_atlas_shaped",
        # CENSOR: global degradation sink
        "native_degradations",
        "clear_native_degradations",
        "terrain_culling_stats",
        "terrain_visibility_stats",
        "terrain_vt_stats",
        "terrain_seam_stats",
        "encode_bc7_rgba8",
        "decode_bc7_rgba8",
        "encode_bc5_rg8",
        "decode_bc5_rg8",
        # CENSOR: negotiated GPU capability report
        "capabilities",
        # CENSOR: last-render execution certificate JSON
        "render_execution_report",
        "begin_render_execution_capture",
        "finish_render_execution_capture",
        "abort_render_execution_capture",
        # CENSOR: Ed25519 certificate signing through the Rust implementation
        "sign_render_certificate_digest",
        # CENSOR: budget-enforce test helper
        "request_host_visible_allocation_for_test",
        # CENSOR: certified BRDF pixel renders
        "render_brdf_tile",
        "render_brdf_tile_overrides",
        # PROBATUM: WGSL proof report
        "shader_report",
        # COMPENDIUM: certified DEM codec
        "compress_dem",
        "decompress_dem",
        "verify_dem",
    ]

    @pytest.mark.parametrize("fn_name", EXPECTED_FUNCTIONS)
    def test_registered_function_exists(self, fn_name: str):
        """Each registered pyfunction must be callable on the native module."""
        assert hasattr(_native, fn_name), (
            f"_forge3d.{fn_name} not found -- "
            f"was it removed from wrap_pyfunction in lib.rs?"
        )
        obj = getattr(_native, fn_name)
        assert callable(obj), (
            f"_forge3d.{fn_name} should be callable, got {type(obj)}"
        )


# ===========================================================================
# Section 2: Scene class method contracts
# ===========================================================================
class TestSceneMethodContracts:
    """Verify Scene class exposes the methods that wrappers depend on."""

    # ---- Primary render methods ----

    def test_render_rgba_is_instance_method(self):
        """Scene.render_rgba must exist as an instance method (not static)."""
        assert hasattr(f3d.Scene, "render_rgba"), "Scene.render_rgba not found"
        # On a PyO3 class, methods appear as method descriptors
        attr = getattr(f3d.Scene, "render_rgba")
        assert callable(attr), "Scene.render_rgba must be callable"

    def test_render_png_is_instance_method(self):
        """Scene.render_png must exist as an instance method."""
        assert hasattr(f3d.Scene, "render_png"), "Scene.render_png not found"
        attr = getattr(f3d.Scene, "render_png")
        assert callable(attr), "Scene.render_png must be callable"

    # ---- Configuration methods ----

    def test_set_msaa_samples_is_instance_method(self):
        """Scene.set_msaa_samples must exist as an instance method."""
        assert hasattr(f3d.Scene, "set_msaa_samples"), (
            "Scene.set_msaa_samples not found"
        )
        attr = getattr(f3d.Scene, "set_msaa_samples")
        assert callable(attr), "Scene.set_msaa_samples must be callable"

    def test_set_camera_look_at_exists(self):
        """Scene.set_camera_look_at must exist."""
        assert hasattr(f3d.Scene, "set_camera_look_at"), (
            "Scene.set_camera_look_at not found"
        )

    def test_set_height_from_r32f_exists(self):
        """Scene.set_height_from_r32f must exist."""
        assert hasattr(f3d.Scene, "set_height_from_r32f"), (
            "Scene.set_height_from_r32f not found"
        )

    # ---- AETHER spectral atmosphere ----

    @pytest.mark.parametrize(
        "method_name",
        ("set_atmosphere", "clear_atmosphere", "get_atmosphere_settings"),
    )
    def test_aether_scene_method_exists(self, method_name: str):
        assert hasattr(f3d.Scene, method_name), f"Scene.{method_name} not found"
        assert callable(getattr(f3d.Scene, method_name))

    # ---- OIT methods ----

    def test_enable_oit_exists(self):
        """Scene.enable_oit must exist (P0.1 OIT feature)."""
        assert hasattr(f3d.Scene, "enable_oit"), "Scene.enable_oit not found"

    def test_disable_oit_exists(self):
        """Scene.disable_oit must exist."""
        assert hasattr(f3d.Scene, "disable_oit"), "Scene.disable_oit not found"

    def test_is_oit_enabled_exists(self):
        """Scene.is_oit_enabled must exist."""
        assert hasattr(f3d.Scene, "is_oit_enabled"), (
            "Scene.is_oit_enabled not found"
        )

    def test_get_oit_mode_exists(self):
        """Scene.get_oit_mode must exist."""
        assert hasattr(f3d.Scene, "get_oit_mode"), "Scene.get_oit_mode not found"

    # ---- SSAO methods ----

    def test_ssao_enabled_exists(self):
        """Scene.ssao_enabled must exist."""
        assert hasattr(f3d.Scene, "ssao_enabled"), "Scene.ssao_enabled not found"

    def test_set_ssao_enabled_exists(self):
        """Scene.set_ssao_enabled must exist."""
        assert hasattr(f3d.Scene, "set_ssao_enabled"), (
            "Scene.set_ssao_enabled not found"
        )

    def test_set_ssao_parameters_exists(self):
        """Scene.set_ssao_parameters must exist."""
        assert hasattr(f3d.Scene, "set_ssao_parameters"), (
            "Scene.set_ssao_parameters not found"
        )

    # ---- IBL methods ----

    def test_enable_ibl_exists(self):
        """Scene.enable_ibl must exist."""
        assert hasattr(f3d.Scene, "enable_ibl"), "Scene.enable_ibl not found"

    def test_disable_ibl_exists(self):
        """Scene.disable_ibl must exist."""
        assert hasattr(f3d.Scene, "disable_ibl"), "Scene.disable_ibl not found"

    # ---- Reflections ----

    def test_enable_reflections_exists(self):
        """Scene.enable_reflections must exist."""
        assert hasattr(f3d.Scene, "enable_reflections"), (
            "Scene.enable_reflections not found"
        )

    def test_disable_reflections_exists(self):
        """Scene.disable_reflections must exist."""
        assert hasattr(f3d.Scene, "disable_reflections"), (
            "Scene.disable_reflections not found"
        )

    # ---- DOF ----

    def test_enable_dof_exists(self):
        """Scene.enable_dof must exist."""
        assert hasattr(f3d.Scene, "enable_dof"), "Scene.enable_dof not found"

    def test_disable_dof_exists(self):
        """Scene.disable_dof must exist."""
        assert hasattr(f3d.Scene, "disable_dof"), "Scene.disable_dof not found"

    # ---- Water surface ----

    def test_enable_water_surface_exists(self):
        """Scene.enable_water_surface must exist."""
        assert hasattr(f3d.Scene, "enable_water_surface"), (
            "Scene.enable_water_surface not found"
        )

    # ---- Ground plane ----

    def test_enable_ground_plane_exists(self):
        """Scene.enable_ground_plane must exist."""
        assert hasattr(f3d.Scene, "enable_ground_plane"), (
            "Scene.enable_ground_plane not found"
        )

    # ---- Cloud shadows ----

    def test_enable_cloud_shadows_exists(self):
        """Scene.enable_cloud_shadows must exist."""
        assert hasattr(f3d.Scene, "enable_cloud_shadows"), (
            "Scene.enable_cloud_shadows not found"
        )

    # ---- Point/spot lights ----

    def test_enable_point_spot_lights_exists(self):
        """Scene.enable_point_spot_lights must exist."""
        assert hasattr(f3d.Scene, "enable_point_spot_lights"), (
            "Scene.enable_point_spot_lights not found"
        )

    def test_add_point_light_exists(self):
        """Scene.add_point_light must exist."""
        assert hasattr(f3d.Scene, "add_point_light"), (
            "Scene.add_point_light not found"
        )

    # ---- Text meshes ----

    def test_enable_text_meshes_exists(self):
        """Scene.enable_text_meshes must exist."""
        assert hasattr(f3d.Scene, "enable_text_meshes"), (
            "Scene.enable_text_meshes not found"
        )

    # ---- get_stats ----

    def test_get_stats_exists(self):
        """Scene.get_stats must exist."""
        assert hasattr(f3d.Scene, "get_stats"), "Scene.get_stats not found"


# ===========================================================================
# Section 3: Camera animation target contract (TV17)
# ===========================================================================
class TestCameraAnimationTargetContract:
    """Verify the target-aware camera animation API exposed by TV17."""

    def test_camera_keyframe_constructible(self):
        """CameraKeyframe is constructible and preserves its target tuple."""
        keyframe = _native.CameraKeyframe(
            1.25,
            30.0,
            40.0,
            500.0,
            55.0,
            (10.0, 20.0, 30.0),
        )
        assert keyframe.time == pytest.approx(1.25)
        assert keyframe.phi_deg == pytest.approx(30.0)
        assert keyframe.theta_deg == pytest.approx(40.0)
        assert keyframe.radius == pytest.approx(500.0)
        assert keyframe.fov_deg == pytest.approx(55.0)
        assert keyframe.target == pytest.approx((10.0, 20.0, 30.0))

    def test_camera_animation_target_round_trip(self):
        """Target-bearing keyframes survive add/get/evaluate round-trips."""
        animation = _native.CameraAnimation()
        animation.add_keyframe(0.0, 0.0, 45.0, 1000.0, 55.0, (0.0, 10.0, 0.0))
        animation.add_keyframe(1.0, 90.0, 45.0, 1200.0, 60.0, (100.0, 20.0, 50.0))

        keyframes = animation.get_keyframes()
        assert len(keyframes) == 2
        assert keyframes[0].target == pytest.approx((0.0, 10.0, 0.0))
        assert keyframes[1].target == pytest.approx((100.0, 20.0, 50.0))

        state = animation.evaluate(0.5)
        assert state.target == pytest.approx((50.0, 15.0, 25.0))

    def test_camera_animation_replace_and_clear(self):
        """Editable keyframe APIs accept CameraKeyframe instances directly."""
        animation = _native.CameraAnimation()
        replacement = [
            _native.CameraKeyframe(0.0, 10.0, 30.0, 400.0, 40.0),
            _native.CameraKeyframe(2.0, 20.0, 35.0, 450.0, 45.0, (5.0, 6.0, 7.0)),
        ]

        animation.replace_keyframes(replacement)
        assert animation.keyframe_count == 2
        assert animation.get_keyframes()[1].target == pytest.approx((5.0, 6.0, 7.0))

        animation.clear_keyframes()
        assert animation.keyframe_count == 0


# ===========================================================================
# Section 4: Scene method count snapshot
# ===========================================================================
class TestSceneMethodCount:
    """Guard against accidental bulk removal of Scene methods.

    If someone accidentally removes a large block of #[pymethods], this
    catches it by asserting a minimum method count.
    """

    def test_scene_has_minimum_method_count(self):
        """Scene must expose at least 80 public methods.

        As of baseline (2026-02-19), Scene has ~180 methods. A drop
        below 80 would indicate a serious regression.
        """
        public_methods = [
            name for name in dir(f3d.Scene)
            if not name.startswith("_") and callable(getattr(f3d.Scene, name))
        ]
        assert len(public_methods) >= 80, (
            f"Scene has only {len(public_methods)} public methods; "
            f"expected at least 80. Methods may have been removed accidentally."
        )


# ===========================================================================
# Section 5: Previously-orphaned pyclass types now registered (P0.3)
# ===========================================================================
class TestOrphanedClassesRegistered:
    """Verify that previously-orphaned pyclass types are now registered.

    Frame, SdfPrimitive, SdfScene, and SdfSceneBuilder must be registered
    in the native module and constructible from Python (except Frame, which
    is constructed internally only).
    """

    def test_frame_registered(self):
        """Frame must be registered but not constructible from Python."""
        assert hasattr(_native, "Frame"), (
            "Frame not found on _forge3d -- registration missing"
        )
        assert isinstance(getattr(_native, "Frame"), type)
        with pytest.raises(RuntimeError, match="constructed internally"):
            _native.Frame()

    def test_hdr_frame_registered(self):
        """HdrFrame must be registered but not constructible from Python."""
        assert hasattr(_native, "HdrFrame"), (
            "HdrFrame not found on _forge3d -- registration missing"
        )
        assert isinstance(getattr(_native, "HdrFrame"), type)
        with pytest.raises(RuntimeError, match="constructed internally"):
            _native.HdrFrame()

    def test_sdf_primitive_registered(self):
        """SdfPrimitive must be registered and importable."""
        assert hasattr(_native, "SdfPrimitive"), (
            "SdfPrimitive not found on _forge3d -- P0.3 registration missing"
        )
        assert isinstance(getattr(_native, "SdfPrimitive"), type)

    def test_sdf_scene_registered(self):
        """SdfScene must be registered and importable."""
        assert hasattr(_native, "SdfScene"), (
            "SdfScene not found on _forge3d -- P0.3 registration missing"
        )
        assert isinstance(getattr(_native, "SdfScene"), type)

    def test_sdf_scene_builder_registered(self):
        """SdfSceneBuilder must be registered and importable."""
        assert hasattr(_native, "SdfSceneBuilder"), (
            "SdfSceneBuilder not found on _forge3d -- P0.3 registration missing"
        )
        assert isinstance(getattr(_native, "SdfSceneBuilder"), type)

    def test_sdf_scene_constructible(self):
        """SdfScene() must be constructible with no arguments."""
        scene = _native.SdfScene()
        assert scene.primitive_count() == 0
        assert scene.node_count() == 0

    def test_sdf_primitive_sphere_constructible(self):
        """SdfPrimitive.sphere() classmethod must work."""
        prim = _native.SdfPrimitive.sphere((0.0, 0.0, 0.0), 1.0, 0)
        assert prim.material_id == 0

    def test_sdf_scene_builder_round_trip(self):
        """SdfSceneBuilder can add a sphere and build a scene."""
        builder = _native.SdfSceneBuilder()
        node_id = builder.add_sphere((0.0, 0.0, 0.0), 1.0, 0)
        assert isinstance(node_id, int)
        scene = builder.build()
        assert scene.primitive_count() >= 1


# ===========================================================================
# Section 6: TBN mesh functions exported (P0.4)
# ===========================================================================
class TestTbnFunctionsExported:
    """Verify that P0.4 TBN mesh functions are exported and return correct data.

    These tests validate:
    - Functions are registered on the native module
    - Return dicts have correct keys (vertices, indices, tbn_data)
    - Vertex/index counts match expected geometry
    - The Python wrapper (forge3d.mesh) uses the native path
    """

    def test_mesh_generate_cube_tbn_registered(self):
        """mesh_generate_cube_tbn must be a callable on the native module."""
        assert hasattr(_native, "mesh_generate_cube_tbn"), (
            "mesh_generate_cube_tbn not found -- P0.4 registration missing"
        )
        assert callable(getattr(_native, "mesh_generate_cube_tbn"))

    def test_mesh_generate_plane_tbn_registered(self):
        """mesh_generate_plane_tbn must be a callable on the native module."""
        assert hasattr(_native, "mesh_generate_plane_tbn"), (
            "mesh_generate_plane_tbn not found -- P0.4 registration missing"
        )
        assert callable(getattr(_native, "mesh_generate_plane_tbn"))

    def test_cube_tbn_dict_keys(self):
        """mesh_generate_cube_tbn returns dict with vertices, indices, tbn_data."""
        result = _native.mesh_generate_cube_tbn()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        for key in ("vertices", "indices", "tbn_data"):
            assert key in result, f"Missing key '{key}' in cube TBN result"

    def test_cube_tbn_vertex_count(self):
        """Cube has 24 vertices (6 faces x 4 vertices)."""
        result = _native.mesh_generate_cube_tbn()
        assert len(result["vertices"]) == 24

    def test_cube_tbn_index_count(self):
        """Cube has 36 indices (6 faces x 2 triangles x 3 vertices)."""
        result = _native.mesh_generate_cube_tbn()
        assert len(result["indices"]) == 36

    def test_cube_tbn_data_count(self):
        """Cube has 24 TBN entries (one per vertex)."""
        result = _native.mesh_generate_cube_tbn()
        assert len(result["tbn_data"]) == 24

    def test_cube_vertex_dict_structure(self):
        """Each vertex dict has position (3), normal (3), uv (2)."""
        result = _native.mesh_generate_cube_tbn()
        v = result["vertices"][0]
        assert "position" in v and len(v["position"]) == 3
        assert "normal" in v and len(v["normal"]) == 3
        assert "uv" in v and len(v["uv"]) == 2

    def test_cube_tbn_dict_structure(self):
        """Each TBN dict has tangent (3), bitangent (3), normal (3), handedness."""
        result = _native.mesh_generate_cube_tbn()
        t = result["tbn_data"][0]
        assert "tangent" in t and len(t["tangent"]) == 3
        assert "bitangent" in t and len(t["bitangent"]) == 3
        assert "normal" in t and len(t["normal"]) == 3
        assert "handedness" in t
        assert t["handedness"] in (1.0, -1.0)

    def test_plane_tbn_4x4_vertex_count(self):
        """4x4 plane has 16 vertices."""
        result = _native.mesh_generate_plane_tbn(4, 4)
        assert len(result["vertices"]) == 16

    def test_plane_tbn_4x4_index_count(self):
        """4x4 plane has 54 indices: (4-1)*(4-1)*2 triangles * 3 verts."""
        result = _native.mesh_generate_plane_tbn(4, 4)
        assert len(result["indices"]) == 54

    def test_plane_tbn_4x4_tbn_count(self):
        """4x4 plane has 16 TBN entries (one per vertex)."""
        result = _native.mesh_generate_plane_tbn(4, 4)
        assert len(result["tbn_data"]) == 16

    def test_plane_tbn_rejects_small_dimensions(self):
        """mesh_generate_plane_tbn rejects width or height < 2."""
        with pytest.raises((ValueError, Exception)):
            _native.mesh_generate_plane_tbn(1, 4)
        with pytest.raises((ValueError, Exception)):
            _native.mesh_generate_plane_tbn(4, 1)

    def test_python_wrapper_uses_native_path(self):
        """forge3d.mesh._HAS_TBN is True and generate_cube_tbn uses native."""
        from forge3d.mesh import _HAS_TBN, generate_cube_tbn
        assert _HAS_TBN is True, (
            "_HAS_TBN should be True when native TBN functions are exported"
        )
        verts, indices, tbn = generate_cube_tbn()
        assert len(verts) == 24
        assert len(indices) == 36
        assert len(tbn) == 24

    def test_python_wrapper_plane_native_path(self):
        """forge3d.mesh.generate_plane_tbn uses the native path."""
        from forge3d.mesh import generate_plane_tbn, _HAS_TBN
        assert _HAS_TBN is True, "native TBN path should be active"
        verts, indices, tbn = generate_plane_tbn(3, 3)
        assert len(verts) == 9
        assert len(indices) == 24  # (3-1)*(3-1)*2*3
        assert len(tbn) == 9


# ===========================================================================
# Section 6: Python package-level API contracts
# ===========================================================================
class TestPackageLevelApiContracts:
    """Verify that forge3d package-level re-exports are intact."""

    EXPECTED_PACKAGE_ATTRS = [
        "Scene",
        "Session",
        "TerrainRenderer",
        "TerrainRenderParams",
        "Colormap1D",
        "MaterialSet",
        "IBL",
        "OverlayLayer",
        "HdrFrame",
        "__version__",
        "has_gpu",
        "device_probe",
        "native_import_error",
        "open_viewer",
        "open_viewer_async",
        "render_offscreen_rgba",
        "numpy_to_png",
        "png_to_numpy",
        "MapPlate",
        "Legend",
        "ScaleBar",
        "NorthArrow",
        "save_bundle",
        "load_bundle",
        "set_license_key",
        "LicenseError",
        "render_offline",
        "oidn_available",
        # AEQUITAS: PT-vs-raster adjudication pair
        "render_adjudication_pair",
        # VERITAS: per-pixel cryptographic provenance
        "seal_provenance",
        "verify_provenance",
        # SUTURA: zero-placeholder MapScene contract
        "MapSceneNativeUnavailable",
        "CompiledScenePlan",
        # CARTOGRAPHER-PRIME: bounded-optimal label solve + rationale
        "declutter",
        "declutter_optimal",
        "LabelRationale",
        # CENSOR: native Ed25519 certificate signer
        "sign_render_certificate_digest",
        "begin_render_execution_capture",
        "finish_render_execution_capture",
        "abort_render_execution_capture",
        # CENSOR: certified BRDF pixel renders
        "render_brdf_tile",
        "render_brdf_tile_overrides",
        # DUPLA: verified precision module and package-level convenience calls
        "precision",
        "dd_selftest",
        "dd_harness",
        "dd_jitter_demo",
        "geo",
        "terrain",
    ]

    @pytest.mark.parametrize("attr_name", EXPECTED_PACKAGE_ATTRS)
    def test_package_exports_symbol(self, attr_name: str):
        """forge3d package must re-export key symbols."""
        assert hasattr(f3d, attr_name), (
            f"forge3d.{attr_name} not found in package __init__.py"
        )

    def test_version_is_string(self):
        """forge3d.__version__ must be a non-empty string."""
        assert isinstance(f3d.__version__, str)
        assert len(f3d.__version__) > 0
        # Semver-ish: at least "X.Y.Z"
        parts = f3d.__version__.split(".")
        assert len(parts) >= 3, (
            f"Version '{f3d.__version__}' does not look like semver"
        )

    def test_has_gpu_returns_bool(self):
        """forge3d.has_gpu() must return a boolean."""
        result = f3d.has_gpu()
        assert isinstance(result, bool)

    def test_precision_module_exports_native_proof_calls(self):
        """DUPLA's wrapper module stays thin and points at registered natives."""
        assert f3d.precision.dd_selftest is f3d.dd_selftest
        assert f3d.precision.dd_harness is f3d.dd_harness
        assert f3d.precision.dd_jitter_demo is f3d.dd_jitter_demo
        for name in ("dd_selftest", "dd_harness", "dd_jitter_demo"):
            assert callable(getattr(_native, name))

    def test_legacy_render_api_removed(self):
        """Legacy top-level render helpers stay removed."""
        for attr_name in ("render_raster", "render_polygons", "render_raytrace_mesh"):
            assert not hasattr(f3d, attr_name), (
                f"forge3d.{attr_name} should not be exported"
            )


class TestCartographerPrimeNativeContract:
    """Generic/compatibility declutter surfaces share strict typed inputs."""

    CANDIDATES = [
        (1, 0, (0.0, 0.0, 10.0, 10.0), 6.0, True),
        (2, 0, (5.0, 0.0, 15.0, 10.0), 10.0, True),
        (3, 0, (12.0, 0.0, 22.0, 10.0), 6.0, True),
    ]

    def test_generic_optimal_dispatch_matches_compatibility_alias(self):
        generic = _native.declutter(
            self.CANDIDATES, algorithm="optimal", gap_tolerance=0.0, margin=0.0
        )
        compatibility = _native.declutter_optimal(
            self.CANDIDATES, gap_tolerance=0.0, margin=0.0
        )
        assert generic[0] == compatibility[0] == [(1, 0), (3, 0)]
        assert generic[1] == compatibility[1] == 0.0
        assert generic[2].records() == compatibility[2].records()
        assert isinstance(generic[2], _native.LabelRationale)

    @pytest.mark.parametrize(
        "candidate",
        [
            (1, 0, (0.0, float("nan"), 1.0, 1.0), 1.0, True),
            (1, 0, (0.0, 0.0, 1.0, 1.0), float("inf"), True),
            (1, 0, (0.0, 0.0, 1.0, 1.0), -1.0, True),
        ],
    )
    def test_invalid_candidate_is_rejected(self, candidate):
        with pytest.raises(ValueError, match=r"candidate \(1, 0\)"):
            _native.declutter([candidate])

    def test_duplicate_candidate_identity_is_rejected(self):
        duplicate = (7, 3, (0.0, 0.0, 1.0, 1.0), 1.0, True)
        with pytest.raises(ValueError, match="identity is duplicated"):
            _native.declutter([duplicate, duplicate])

    @pytest.mark.parametrize(
        "bounds",
        [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.01, 1.0),
        ],
    )
    def test_degenerate_or_quantization_collapsed_bounds_are_rejected(self, bounds):
        with pytest.raises(ValueError, match="non-degenerate after quantization"):
            _native.declutter([(1, 0, bounds, 1.0, True)])

    def test_large_weight_aggregate_does_not_overflow(self):
        candidates = [
            (1, 0, (0.0, 0.0, 1.0, 1.0), 5.0e15, True),
            (2, 0, (2.0, 0.0, 3.0, 1.0), 5.0e15, True),
            (3, 0, (4.0, 0.0, 5.0, 1.0), 5.0e15, True),
        ]
        placements, gap, rationale = _native.declutter(
            candidates, gap_tolerance=0.0, margin=0.0
        )
        assert placements == [(1, 0), (2, 0), (3, 0)]
        assert gap == 0.0
        solver = next(record for record in rationale.records() if record["kind"] == "solver")
        assert solver["objective"] == pytest.approx(1.5e16)
        assert solver["objective"] == solver["upper_bound"]

    def test_raw_priority_is_primary_over_cardinality(self):
        unit = 1.0 / 1024.0
        placements, gap, rationale = _native.declutter(
            [
                (1, 0, (0.0, 0.0, 30.0, 10.0), 10.0 * unit, True),
                (2, 0, (0.0, 0.0, 9.0, 10.0), 3.0 * unit, True),
                (3, 0, (10.5, 0.0, 19.5, 10.0), 3.0 * unit, True),
                (4, 0, (21.0, 0.0, 30.0, 10.0), 3.0 * unit, True),
            ],
            gap_tolerance=0.0,
            margin=0.0,
        )
        assert placements == [(1, 0)]
        assert gap == 0.0
        solver = next(record for record in rationale.records() if record["kind"] == "solver")
        assert solver["objective"] == pytest.approx(10.0 * unit)
        assert solver["objective"] == solver["upper_bound"]

    def test_large_box_overlap_area_does_not_overflow(self):
        placements, _, rationale = _native.declutter(
            [
                (1, 0, (0.0, 0.0, 1.0e9, 1.0e9), 2.0, True),
                (2, 0, (0.0, 0.0, 1.0e9, 1.0e9), 1.0, True),
            ],
            gap_tolerance=0.0,
            margin=0.0,
        )
        assert placements == [(1, 0)]
        placed = next(record for record in rationale.records() if record["kind"] == "placed")
        assert placed["displaced"][0]["overlap_area_px"] == pytest.approx(1.0e18)

    def test_visibility_gate_is_solver_only_evidence(self):
        _, _, rationale = _native.declutter(
            [(1, 0, (0.0, 0.0, 1.0, 1.0), 1.0, False)], margin=0.0
        )
        assert rationale.records()[0] == {
            "kind": "visibility_filtered_candidate",
            "label_id": 1,
            "candidate_index": 0,
        }
        rendered = " ".join(rationale.render()).lower()
        for unsupported_claim in ("depth", "silhouette", "occluded"):
            assert unsupported_claim not in rendered

    def test_unsupported_generic_algorithm_is_explicit(self):
        with pytest.raises(ValueError, match="expected 'optimal'"):
            _native.declutter(self.CANDIDATES, algorithm="greedy")

    def test_internal_depth_host_reservation_lifetime_and_double_close(self):
        before = _native.global_memory_metrics()["host_visible_bytes"]
        reservation = _native._reserve_label_depth_host_allocation(
            4096, "labels.depth.api-contract"
        )
        assert type(reservation).__name__ == "_LabelDepthHostAllocation"
        assert reservation.bytes == 4096
        assert reservation.active is True
        assert _native.global_memory_metrics()["host_visible_bytes"] == before + 4096
        assert reservation.close() is True
        assert reservation.close() is False
        assert reservation.active is False
        assert _native.global_memory_metrics()["host_visible_bytes"] == before

    def test_canonical_clippy_aliases_are_single_strings(self):
        from pathlib import Path
        from _toml_compat import load_toml

        config_path = Path(__file__).resolve().parents[1] / ".cargo" / "config.toml"
        aliases = load_toml(config_path)["alias"]
        for name in ("forge3d-clippy", "forge3d-clippy-acceptance"):
            alias = aliases[name]
            assert isinstance(alias, str), f"{name} must use Cargo's string alias form"
            assert alias.startswith("clippy --workspace --all-targets --features ")
            assert alias.endswith(" -- -D warnings")


# ===========================================================================
# Section 7: Geometry free-function contracts
# ===========================================================================
class TestGeometryFunctionContracts:
    """Verify geometry-related native functions are registered."""

    GEOMETRY_FUNCTIONS = [
        "geometry_generate_primitive_py",
        "geometry_generate_tangents_py",
        "geometry_weld_mesh_py",
        "geometry_subdivide_py",
        "geometry_validate_mesh_py",
        "geometry_displace_heightmap_py",
        "geometry_generate_tube_py",
        "geometry_generate_ribbon_py",
        "geometry_generate_thick_polyline_py",
        "geometry_extrude_polygon_py",
        "geometry_simplify_mesh_py",
    ]

    @pytest.mark.parametrize("fn_name", GEOMETRY_FUNCTIONS)
    def test_geometry_function_exists(self, fn_name: str):
        """Geometry functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), (
            f"_forge3d.{fn_name} not found"
        )
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 8: Camera function contracts
# ===========================================================================
class TestCameraFunctionContracts:
    """Verify camera-related native functions are registered."""

    CAMERA_FUNCTIONS = [
        "camera_look_at",
        "camera_perspective",
        "camera_orthographic",
        "camera_view_proj",
        "camera_dof_params",
    ]

    @pytest.mark.parametrize("fn_name", CAMERA_FUNCTIONS)
    def test_camera_function_exists(self, fn_name: str):
        """Camera functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), f"_forge3d.{fn_name} not found"
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 9: IO function contracts
# ===========================================================================
class TestIoFunctionContracts:
    """Verify IO-related native functions are registered."""

    IO_FUNCTIONS = [
        "io_import_obj_py",
        "io_export_obj_py",
        "io_export_stl_py",
        "io_import_gltf_py",
        "io_import_gltf_with_materials_py",
    ]

    @pytest.mark.parametrize("fn_name", IO_FUNCTIONS)
    def test_io_function_exists(self, fn_name: str):
        """IO functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), f"_forge3d.{fn_name} not found"
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 10: Transform function contracts
# ===========================================================================
class TestTransformFunctionContracts:
    """Verify transform-related native functions are registered."""

    TRANSFORM_FUNCTIONS = [
        "translate",
        "rotate_x",
        "rotate_y",
        "rotate_z",
        "scale",
    ]

    @pytest.mark.parametrize("fn_name", TRANSFORM_FUNCTIONS)
    def test_transform_function_exists(self, fn_name: str):
        """Transform functions must be accessible on the native module."""
        assert hasattr(_native, fn_name), f"_forge3d.{fn_name} not found"
        assert callable(getattr(_native, fn_name))


# ===========================================================================
# Section 11: Native module total symbol count guard
# ===========================================================================
class TestNativeModuleSymbolCount:
    """Guard against accidental bulk removal of native symbols."""

    def test_native_module_has_minimum_symbols(self):
        """The native module must export at least 100 symbols.

        As of baseline (2026-02-19), it exports 134 symbols.
        A significant drop indicates accidental removal.
        """
        public_symbols = [
            name for name in dir(_native) if not name.startswith("_")
        ]
        assert len(public_symbols) >= 100, (
            f"Native module has only {len(public_symbols)} public symbols; "
            f"expected at least 100. Symbols may have been removed accidentally."
        )


# ===========================================================================
# Section 12: P0.2 Offscreen render routing contracts
# ===========================================================================
class TestOffscreenRenderRouting:
    """Verify that render_offscreen_rgba routes correctly based on scene type.

    P0.2 fix: The old code probed for a nonexistent module-level
    ``_forge3d.render_rgba`` function, so the native path was never
    reachable.  The corrected code checks whether ``scene`` is a native
    ``Scene`` instance and calls ``scene.render_rgba()`` directly.
    """

    def test_native_scene_detected_correctly(self):
        """_is_native_scene returns True for native Scene instances."""
        from forge3d.helpers.offscreen import _NativeScene
        # _NativeScene should be the native Scene class when extension is loaded
        assert _NativeScene is not None, (
            "_NativeScene is None -- native module failed to load"
        )
        assert _NativeScene is _native.Scene, (
            "_NativeScene should be the same class as _native.Scene"
        )

    def test_non_native_scene_rejected(self):
        """_is_native_scene returns False for non-native objects."""
        from forge3d.helpers.offscreen import _is_native_scene
        assert not _is_native_scene(None)
        assert not _is_native_scene("not a scene")
        assert not _is_native_scene(42)
        assert not _is_native_scene({"fake": "scene"})

    def test_fallback_used_when_no_scene(self):
        """render_offscreen_rgba uses fallback when scene is None."""
        from forge3d.helpers.offscreen import render_offscreen_rgba
        import numpy as np
        # With scene=None, fallback CPU path tracer is used.
        # It returns a valid (H, W, 4) uint8 array.
        result = render_offscreen_rgba(32, 32, scene=None)
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32, 4)
        assert result.dtype == np.uint8

    def test_fallback_used_for_non_native_scene(self):
        """render_offscreen_rgba uses fallback for non-native scene objects."""
        from forge3d.helpers.offscreen import render_offscreen_rgba
        import numpy as np
        # Passing a plain dict (not a native Scene) triggers fallback.
        result = render_offscreen_rgba(16, 16, scene={"type": "fake"})
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, 16, 4)
        assert result.dtype == np.uint8

    def test_render_rgba_is_instance_method_not_module_function(self):
        """render_rgba must be on Scene instances, NOT on the native module.

        This is the root cause of the P0.2 bug: the old offscreen code
        checked ``hasattr(_native, 'render_rgba')`` which was always
        False because render_rgba is a Scene method, not a module-level
        function.
        """
        # Module-level render_rgba must NOT exist
        assert not hasattr(_native, "render_rgba"), (
            "_forge3d.render_rgba should not exist at module level; "
            "it is an instance method on Scene"
        )
        # Instance-level render_rgba MUST exist
        assert hasattr(_native.Scene, "render_rgba"), (
            "Scene.render_rgba must exist as an instance method"
        )


# ===========================================================================
# Section 13: P0.2 MSAA setter routing contracts
# ===========================================================================
class TestMsaaSetterRouting:
    """Verify that set_msaa_samples is NOT a module-level function.

    P0.2 fix: viewer.py previously probed for ``_forge3d.set_msaa_samples``
    at module level, which never existed.  ``set_msaa_samples`` is an
    instance method on ``Scene``.  The dead probe has been removed.
    """

    def test_set_msaa_samples_not_on_module(self):
        """_forge3d must NOT have a module-level set_msaa_samples."""
        assert not hasattr(_native, "set_msaa_samples"), (
            "_forge3d.set_msaa_samples should not exist at module level; "
            "it is an instance method on Scene"
        )

    def test_set_msaa_samples_on_scene_class(self):
        """Scene class must have set_msaa_samples as instance method."""
        assert hasattr(_native.Scene, "set_msaa_samples"), (
            "Scene.set_msaa_samples must exist as an instance method"
        )

    def test_viewer_set_msaa_validates_samples(self):
        """viewer.set_msaa rejects invalid sample counts."""
        from forge3d.viewer import set_msaa
        import pytest as _pt
        with _pt.raises(ValueError, match="Unsupported MSAA"):
            set_msaa(3)
        with _pt.raises(ValueError, match="Unsupported MSAA"):
            set_msaa(16)

    def test_viewer_set_msaa_accepts_valid_samples(self):
        """viewer.set_msaa accepts standard MSAA sample counts."""
        from forge3d.viewer import set_msaa
        for n in (1, 2, 4, 8):
            result = set_msaa(n)
            assert result == n


# ===========================================================================
# Section 13b: BOP-P2-02 TerrainRenderer height streaming API contract
# ===========================================================================
class TestTerrainRendererHeightStreamingContract:
    """Class-level contract for the clipmap height-streaming runtime API.

    BOP-P2-02: ClipmapStreamer + AsyncTileLoader + HeightMosaic are wired to
    TerrainRenderer height-texture updates through these methods. Behavior
    coverage (fly-through, bounded memory, convergence) lives in
    tests/test_terrain_clipmap_streaming.py.
    """

    @pytest.mark.parametrize(
        "method",
        [
            "enable_height_streaming",
            "disable_height_streaming",
            "stream_height_tiles",
            "height_streaming_stats",
        ],
    )
    def test_terrain_renderer_streaming_method_exists(self, method):
        assert hasattr(_native.TerrainRenderer, method), (
            f"TerrainRenderer.{method} missing from native API"
        )


# ===========================================================================
# Section 14: P1.1 SSGI/SSR settings wiring behavior tests
# ===========================================================================
class TestSsgiSsrSettingsWiring:
    """Verify SSGI and SSR settings affect runtime state.

    P1.1: These are behavior tests, not just symbol-existence checks.
    They prove that constructing SSGISettings/SSRSettings with different
    parameters produces observably different state, and that Scene exposes
    the methods needed to apply them.
    """

    # ---- SSGI settings behavior ----

    def test_ssgi_settings_defaults(self):
        """SSGISettings() has documented default values."""
        s = _native.SSGISettings()
        assert s.ray_steps == 24
        assert s.intensity == pytest.approx(1.0)
        assert s.ray_radius == pytest.approx(5.0)

    def test_ssgi_settings_custom_values_differ(self):
        """Constructing SSGISettings with custom params changes stored state."""
        default = _native.SSGISettings()
        custom = _native.SSGISettings(ray_steps=64, intensity=3.5)
        assert custom.ray_steps != default.ray_steps
        assert custom.ray_steps == 64
        assert custom.intensity != default.intensity
        assert custom.intensity == pytest.approx(3.5)

    def test_ssgi_settings_all_fields_accessible(self):
        """All SSGISettings fields are readable after construction."""
        s = _native.SSGISettings(
            ray_steps=32,
            ray_radius=4.0,
            ray_thickness=0.5,
            intensity=2.0,
            temporal_alpha=0.8,
            use_half_res=True,
            ibl_fallback=0.3,
        )
        assert s.ray_steps == 32
        assert s.ray_radius == pytest.approx(4.0)
        assert s.ray_thickness == pytest.approx(0.5)
        assert s.intensity == pytest.approx(2.0)
        assert s.temporal_alpha == pytest.approx(0.8)
        assert s.use_half_res is True
        assert s.ibl_fallback == pytest.approx(0.3)

    # ---- SSR settings behavior ----

    def test_ssr_settings_defaults(self):
        """SSRSettings() has documented default values."""
        s = _native.SSRSettings()
        assert s.max_steps == 48
        assert s.intensity == pytest.approx(1.0)

    def test_ssr_settings_custom_values_differ(self):
        """Constructing SSRSettings with custom params changes stored state."""
        default = _native.SSRSettings()
        custom = _native.SSRSettings(max_steps=128, intensity=0.5)
        assert custom.max_steps != default.max_steps
        assert custom.max_steps == 128
        assert custom.intensity != default.intensity
        assert custom.intensity == pytest.approx(0.5)

    def test_ssr_settings_all_fields_accessible(self):
        """All SSRSettings fields are readable after construction."""
        s = _native.SSRSettings(
            max_steps=96,
            max_distance=200.0,
            thickness=0.3,
            stride=2.0,
            intensity=0.8,
            roughness_fade=0.5,
            edge_fade=0.1,
            temporal_alpha=0.85,
        )
        assert s.max_steps == 96
        assert s.max_distance == pytest.approx(200.0)
        assert s.thickness == pytest.approx(0.3)
        assert s.stride == pytest.approx(2.0)
        assert s.intensity == pytest.approx(0.8)
        assert s.roughness_fade == pytest.approx(0.5)
        assert s.edge_fade == pytest.approx(0.1)
        assert s.temporal_alpha == pytest.approx(0.85)

    # ---- Scene integration methods ----

    def test_scene_has_ssgi_methods(self):
        """Scene must expose enable/disable/query/set/get for SSGI."""
        for method in ("enable_ssgi", "disable_ssgi", "is_ssgi_enabled",
                        "set_ssgi_settings", "get_ssgi_settings"):
            assert hasattr(_native.Scene, method), f"Scene.{method} not found"
            assert callable(getattr(_native.Scene, method))

    def test_scene_has_ssr_methods(self):
        """Scene must expose enable/disable/query/set/get for SSR."""
        for method in ("enable_ssr", "disable_ssr", "is_ssr_enabled",
                        "set_ssr_settings", "get_ssr_settings"):
            assert hasattr(_native.Scene, method), f"Scene.{method} not found"
            assert callable(getattr(_native.Scene, method))

    def test_scene_has_native_text_halo_method(self):
        """Scene must expose native SDF text quads with halo fields."""
        assert hasattr(_native.Scene, "add_native_text_rect_uv_halo")
        assert callable(getattr(_native.Scene, "add_native_text_rect_uv_halo"))

    def test_terrain_params_exposes_screen_space_settings(self):
        """TerrainRenderParams must expose decoded SSAO/SSGI/SSR/TAA settings."""
        assert hasattr(_native.TerrainRenderParams, "screen_space_settings")
        assert callable(getattr(_native.TerrainRenderParams, "screen_space_settings"))


# ===========================================================================
# Section 15: P1.2 Bloom settings wiring behavior tests
# ===========================================================================
class TestBloomSettingsWiring:
    """Verify bloom enable/disable and settings are exposed on Scene.

    P1.2: These are behavior tests proving that the Scene class exposes
    the bloom methods needed to apply bloom settings from Python, and
    that BloomSettings at the Python config level are still properly
    validated and stored.
    """

    # ---- Scene integration methods ----

    def test_scene_has_bloom_methods(self):
        """Scene must expose enable/disable/query/set/get for bloom."""
        for method in ("enable_bloom", "disable_bloom", "is_bloom_enabled",
                        "set_bloom_settings", "get_bloom_settings"):
            assert hasattr(_native.Scene, method), f"Scene.{method} not found"
            assert callable(getattr(_native.Scene, method))

    # ---- BloomSettings Python-level validation ----

    def test_bloom_settings_default_disabled(self):
        """BloomSettings defaults to disabled with conservative values."""
        from forge3d.terrain_params import BloomSettings
        s = BloomSettings()
        assert s.enabled is False
        assert s.threshold == 1.5
        assert s.softness == 0.5
        assert s.intensity == 0.3
        assert s.radius == 1.0

    def test_bloom_settings_enabled_custom(self):
        """BloomSettings with custom params stores them correctly."""
        from forge3d.terrain_params import BloomSettings
        s = BloomSettings(enabled=True, threshold=2.0, softness=0.8,
                          intensity=0.6, radius=1.5)
        assert s.enabled is True
        assert s.threshold == 2.0
        assert s.softness == 0.8
        assert s.intensity == 0.6
        assert s.radius == 1.5

    def test_bloom_settings_rejects_negative_threshold(self):
        """BloomSettings rejects negative threshold."""
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="threshold"):
            BloomSettings(threshold=-1.0)

    def test_bloom_settings_rejects_invalid_softness(self):
        """BloomSettings rejects softness outside [0, 1]."""
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="softness"):
            BloomSettings(softness=1.5)

    def test_bloom_settings_rejects_negative_intensity(self):
        """BloomSettings rejects negative intensity."""
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="intensity"):
            BloomSettings(intensity=-0.1)

    def test_bloom_settings_rejects_zero_radius(self):
        """BloomSettings rejects zero or negative radius."""
        from forge3d.terrain_params import BloomSettings
        with pytest.raises(ValueError, match="radius"):
            BloomSettings(radius=0.0)

    def test_bloom_in_terrain_params(self):
        """BloomSettings integrates correctly into TerrainRenderParams."""
        from forge3d.terrain_params import BloomSettings, make_terrain_params_config
        bloom = BloomSettings(enabled=True, threshold=1.0, intensity=0.5)
        params = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=1000.0,
            msaa_samples=1,
            z_scale=1.0,
            exposure=1.0,
            domain=(1000.0, 2000.0),
            bloom=bloom,
        )
        assert params.bloom is not None
        assert params.bloom.enabled is True
        assert params.bloom.threshold == 1.0
        assert params.bloom.intensity == 0.5


# ===========================================================================
# Section 16: P1.3 Terrain Analysis API (Option A: TerrainSpike methods)
# ===========================================================================
class TestTerrainAnalysisApi:
    """Verify terrain analysis via TerrainSpike methods (Option A).

    Decision: expose analysis through TerrainSpike instance methods rather
    than module-level functions to avoid duplicate implementations.
    The private ``_compute_slope_deg()`` in render.py stays private.
    """

    def test_terrain_spike_analysis_methods_exist_on_class(self):
        """TerrainSpike exposes analysis methods even when runtime GPU is absent."""
        assert hasattr(_native, "TerrainSpike")
        assert callable(getattr(_native.TerrainSpike, "slope_aspect_compute", None))
        assert callable(getattr(_native.TerrainSpike, "contour_extract", None))

    def _make_spike(self):
        if not _TERRAIN_SPIKE_AVAILABLE:
            pytest.skip("TerrainSpike behavior tests require a GPU-backed device")
        return _native.TerrainSpike(64, 64)

    # ---- slope_aspect_compute ----

    @pytest.mark.apple_metal_physical
    def test_slope_flat_surface(self):
        """Flat surface produces zero slope everywhere."""
        import numpy as np
        ts = self._make_spike()
        heights = np.zeros(64 * 64, dtype=np.float32)
        slopes, _ = ts.slope_aspect_compute(heights, 64, 64)
        assert slopes.shape == (64 * 64,)
        assert slopes.max() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.apple_metal_physical
    def test_slope_east_ramp(self):
        """East-facing ramp at 45 degrees produces slope ~45."""
        import numpy as np
        ts = self._make_spike()
        heights = np.zeros(64 * 64, dtype=np.float32)
        for y in range(64):
            for x in range(64):
                heights[y * 64 + x] = float(x)
        slopes, _ = ts.slope_aspect_compute(heights, 64, 64)
        assert slopes.min() == pytest.approx(45.0, abs=0.1)

    @pytest.mark.apple_metal_physical
    def test_slope_rejects_wrong_size(self):
        """slope_aspect_compute rejects array with wrong element count."""
        import numpy as np
        ts = self._make_spike()
        heights = np.zeros(100, dtype=np.float32)  # wrong size for 64x64
        with pytest.raises(Exception):
            ts.slope_aspect_compute(heights, 64, 64)

    @pytest.mark.apple_metal_physical
    def test_slope_rejects_small_grid(self):
        """slope_aspect_compute rejects grids smaller than 3x3."""
        import numpy as np
        ts = self._make_spike()
        heights = np.zeros(4, dtype=np.float32)
        with pytest.raises(Exception):
            ts.slope_aspect_compute(heights, 2, 2)

    # ---- contour_extract ----

    @pytest.mark.apple_metal_physical
    def test_contour_gaussian(self):
        """Gaussian hill produces contours at requested levels."""
        import numpy as np
        import math
        ts = self._make_spike()
        heights = np.zeros(64 * 64, dtype=np.float32)
        cx, cy = 32.0, 32.0
        for y in range(64):
            for x in range(64):
                dx, dy = x - cx, y - cy
                heights[y * 64 + x] = 100.0 * math.exp(-(dx * dx + dy * dy) / 200.0)
        result = ts.contour_extract(heights, 64, 64, levels=[10.0, 50.0, 90.0])
        assert isinstance(result, dict)
        assert result["polyline_count"] > 0
        assert result["total_points"] > 0
        assert len(result["polylines"]) == result["polyline_count"]

    @pytest.mark.apple_metal_physical
    def test_contour_rejects_empty_levels(self):
        """contour_extract rejects empty levels list."""
        import numpy as np
        ts = self._make_spike()
        heights = np.zeros(64 * 64, dtype=np.float32)
        with pytest.raises(Exception):
            ts.contour_extract(heights, 64, 64, levels=[])

    @pytest.mark.apple_metal_physical
    def test_contour_no_crossing(self):
        """Flat surface at 0 with level=50 produces no contours."""
        import numpy as np
        ts = self._make_spike()
        heights = np.zeros(64 * 64, dtype=np.float32)
        result = ts.contour_extract(heights, 64, 64, levels=[50.0])
        assert result["polyline_count"] == 0
        assert result["total_points"] == 0

    # ---- API shape: Option A confirmation ----

    def test_no_module_level_slope_function(self):
        """No module-level slope/contour functions exist (Option A chosen)."""
        assert not hasattr(_native, "compute_slope_aspect_py")
        assert not hasattr(_native, "slope_aspect_compute")
        assert not hasattr(_native, "contour_extract")


# ===========================================================================
# Section 17: Point Cloud Buffer (P2.1)
# ===========================================================================
class TestPointCloudBuffer:
    """P2.1: PointBuffer CPU-to-GPU data interleaving."""

    def test_point_buffer_registered(self):
        """PointBuffer class is accessible from the native module."""
        assert hasattr(_native, "PointBuffer")

    # ---- Compact interleaving (6 floats/point) ----

    def test_create_gpu_buffer_empty(self):
        """Empty PointBuffer produces an empty buffer."""
        import numpy as np
        pb = _native.PointBuffer([], None)
        gpu = pb.create_gpu_buffer()
        assert isinstance(gpu, np.ndarray)
        assert gpu.dtype == np.float32
        assert gpu.shape == (0,)

    def test_create_gpu_buffer_positions_only(self):
        """Positions without colors default to white (1.0, 1.0, 1.0)."""
        positions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        pb = _native.PointBuffer(positions, None)
        gpu = pb.create_gpu_buffer()
        assert gpu.shape == (12,)  # 2 points * 6 floats
        assert gpu[0] == pytest.approx(1.0)
        assert gpu[3] == pytest.approx(1.0)  # r = white
        assert gpu[6] == pytest.approx(4.0)

    def test_create_gpu_buffer_anchored_preserves_earth_scale_offset(self):
        pb = _native.PointBuffer(
            [6378137.00025, 0.0, 0.0, 6378137.00050, 0.0, 0.0],
            None,
        )
        gpu = pb.create_gpu_buffer_anchored((6378137.0, 0.0, 0.0))
        assert gpu[0] == pytest.approx(0.00025, abs=1e-6)
        assert gpu[6] == pytest.approx(0.00050, abs=1e-6)

    def test_create_gpu_buffer_with_colors(self):
        """Positions + colors properly interleaved, colors normalised."""
        positions = [10.0, 20.0, 30.0]
        colors = [255, 0, 128]
        pb = _native.PointBuffer(positions, colors)
        gpu = pb.create_gpu_buffer()
        assert gpu.shape == (6,)
        assert gpu[3] == pytest.approx(1.0)          # r
        assert gpu[4] == pytest.approx(0.0)           # g
        assert gpu[5] == pytest.approx(128 / 255.0, rel=1e-3)  # b

    def test_gpu_byte_size(self):
        """gpu_byte_size matches 6 floats * 4 bytes * point_count."""
        pb = _native.PointBuffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], None)
        assert pb.gpu_byte_size() == 2 * 6 * 4

    # ---- Viewer-compatible interleaving (12 floats/point = 48 bytes) ----

    def test_create_viewer_gpu_buffer_empty(self):
        """Empty buffer returns empty viewer buffer."""
        pb = _native.PointBuffer([], None)
        vgpu = pb.create_viewer_gpu_buffer([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert vgpu.shape == (0,)

    def test_create_viewer_gpu_buffer_layout(self):
        """Viewer buffer has 12 floats/point matching PointInstance3D."""
        positions = [5.0, 100.0, 10.0]  # one point at y=100
        pb = _native.PointBuffer(positions, None)
        vgpu = pb.create_viewer_gpu_buffer([0.0, 0.0, 0.0], [10.0, 200.0, 20.0])
        assert vgpu.shape == (12,)
        # position
        assert vgpu[0] == pytest.approx(5.0)   # x
        assert vgpu[1] == pytest.approx(100.0)  # y
        assert vgpu[2] == pytest.approx(10.0)   # z
        # elevation_norm = (100 - 0) / 200 = 0.5
        assert vgpu[3] == pytest.approx(0.5)
        # rgb (white default)
        assert vgpu[4] == pytest.approx(1.0)
        assert vgpu[5] == pytest.approx(1.0)
        assert vgpu[6] == pytest.approx(1.0)
        # intensity default
        assert vgpu[7] == pytest.approx(0.5)
        # size default
        assert vgpu[8] == pytest.approx(1.0)
        # padding
        assert vgpu[9] == pytest.approx(0.0)
        assert vgpu[10] == pytest.approx(0.0)
        assert vgpu[11] == pytest.approx(0.0)

    def test_create_viewer_gpu_buffer_with_colors(self):
        """Viewer buffer includes normalised colors."""
        pb = _native.PointBuffer([0.0, 50.0, 0.0], [128, 64, 255])
        vgpu = pb.create_viewer_gpu_buffer([0.0, 0.0, 0.0], [0.0, 100.0, 0.0])
        assert vgpu[4] == pytest.approx(128 / 255.0, rel=1e-3)
        assert vgpu[5] == pytest.approx(64 / 255.0, rel=1e-3)
        assert vgpu[6] == pytest.approx(1.0)

    def test_viewer_buffer_byte_count(self):
        """Viewer buffer is 48 bytes (12 floats * 4) per point."""
        pb = _native.PointBuffer([0.0] * 9, None)  # 3 points
        vgpu = pb.create_viewer_gpu_buffer([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert vgpu.shape == (3 * 12,)
        assert vgpu.nbytes == 3 * 48

    # ---- Validation ----

    def test_validation_bad_positions(self):
        """Positions length not divisible by 3 raises ValueError."""
        with pytest.raises(ValueError, match="multiple of 3"):
            _native.PointBuffer([1.0, 2.0], None)

    def test_validation_bad_colors(self):
        """Colors length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            _native.PointBuffer([1.0, 2.0, 3.0], [255, 0])

    def test_point_count(self):
        """point_count property reflects the actual number of points."""
        pb = _native.PointBuffer([0.0] * 9, None)
        assert pb.point_count == 3

    def test_repr(self):
        """PointBuffer has a useful repr."""
        pb = _native.PointBuffer([1.0, 2.0, 3.0], None)
        r = repr(pb)
        assert "PointBuffer" in r
        assert "point_count=1" in r

    # ---- CPU fallback intact ----

    def test_cpu_fallback_intact(self):
        """Pure-Python PointCloudRenderer still importable and usable."""
        from forge3d.pointcloud import PointCloudRenderer
        renderer = PointCloudRenderer()
        assert renderer.point_budget == 5_000_000

    # ---- Fixture check ----

    def test_fixture_exists(self):
        """The MtStHelens.laz fixture file exists on disk."""
        import os
        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets", "lidar", "MtStHelens.laz",
        )
        assert os.path.isfile(fixture), f"Fixture missing: {fixture}"


# ===========================================================================
# Section 18: COPC LAZ Decompression (P2.2)
# ===========================================================================
class TestCopcLazDecompression:
    """P2.2: LAZ decompression feature gate and fixture-based validation."""

    # ---- Feature gate checks ----

    def test_copc_laz_enabled_function_exists(self):
        """copc_laz_enabled() is callable from the native module."""
        assert hasattr(_native, "copc_laz_enabled")
        result = _native.copc_laz_enabled()
        assert isinstance(result, bool)

    def test_copc_laz_feature_flag_is_boolean(self):
        """copc_laz_enabled reports feature state as a strict bool in any build mode."""
        assert isinstance(_native.copc_laz_enabled(), bool)

    def test_read_laz_points_info_function_exists(self):
        """read_laz_points_info() is callable from the native module."""
        assert hasattr(_native, "read_laz_points_info")

    def test_read_laz_point_attributes_function_exists(self):
        """read_laz_point_attributes() is callable from the native module."""
        assert hasattr(_native, "read_laz_point_attributes")

    # ---- Fixture-based LAZ decompression validation ----

    def _fixture_path(self):
        import os
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets", "lidar", "MtStHelens.laz",
        )

    def test_fixture_exists(self):
        """MtStHelens.laz fixture file exists."""
        import os
        assert os.path.isfile(self._fixture_path())

    def test_read_laz_returns_tuple(self):
        """read_laz_points_info returns (point_count, coords, has_rgb)."""
        result = _native.read_laz_points_info(self._fixture_path())
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_decoded_point_count(self):
        """LAZ fixture contains a positive number of points."""
        point_count, _, _ = _native.read_laz_points_info(self._fixture_path())
        assert isinstance(point_count, int)
        assert point_count > 0

    def test_sample_coordinates_valid(self):
        """First decoded points have finite, non-zero coordinates."""
        _, coords, _ = _native.read_laz_points_info(self._fixture_path())
        # At least one point (3 coords)
        assert len(coords) >= 3
        import math
        for c in coords:
            assert math.isfinite(c), f"Non-finite coordinate: {c}"
        # At least one coordinate should be non-zero
        assert any(abs(c) > 0.0 for c in coords)

    def test_sample_coordinates_in_expected_range(self):
        """Mt St Helens coordinates are in a plausible geographic range."""
        _, coords, _ = _native.read_laz_points_info(self._fixture_path())
        x, y, z = coords[0], coords[1], coords[2]
        # The fixture uses NAD83 State Plane Washington South (US feet).
        # X (easting) ~1.2M, Y (northing) ~315K, Z (elevation) ~4K feet
        assert 1_100_000 < x < 1_400_000, f"Unexpected X: {x}"
        assert 200_000 < y < 500_000, f"Unexpected Y: {y}"
        assert 0 < z < 15_000, f"Unexpected Z: {z}"

    def test_multiple_points_decoded(self):
        """At least 3 points are decoded from the fixture."""
        _, coords, _ = _native.read_laz_points_info(self._fixture_path())
        # 3 points * 3 coords = 9 values
        assert len(coords) == 9

    def test_laz_attributes_include_classification_and_intensity(self):
        """Native LAZ samples expose per-point classification and intensity attributes."""
        result = _native.read_laz_point_attributes(self._fixture_path(), 3)
        assert result["point_count"] > 0
        assert len(result["coords"]) == 9
        assert len(result["intensities"]) == 3
        assert len(result["classifications"]) == 3
        assert all(isinstance(v, int) for v in result["classifications"])

    def test_read_laz_nonexistent_file_raises(self):
        """Reading a nonexistent file raises IOError."""
        with pytest.raises(IOError):
            _native.read_laz_points_info("/nonexistent/path.laz")

    def test_read_laz_invalid_file_raises(self):
        """Reading a non-LAZ file raises an error."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as f:
            f.write(b"not a laz file at all")
            tmp = f.name
        try:
            with pytest.raises(Exception):
                _native.read_laz_points_info(tmp)
        finally:
            os.unlink(tmp)


# ===========================================================================
# Section 19: Label Bindings (P2.3)
# ===========================================================================
class TestLabelBindings:
    """P2.3: LabelStyle and LabelFlags PyO3 bindings."""

    def test_label_style_registered(self):
        """LabelStyle class is accessible from the native module."""
        assert hasattr(_native, "LabelStyle")

    def test_label_flags_registered(self):
        """LabelFlags class is accessible from the native module."""
        assert hasattr(_native, "LabelFlags")

    def test_label_style_defaults(self):
        """Default LabelStyle matches Rust LabelStyle::default() values."""
        s = _native.LabelStyle()
        assert s.size == pytest.approx(14.0)
        assert s.color == pytest.approx((0.1, 0.1, 0.1, 1.0))
        assert s.halo_color == pytest.approx((1.0, 1.0, 1.0, 0.8))
        assert s.halo_width == pytest.approx(1.5)
        assert s.priority == 0
        assert s.min_depth == pytest.approx(0.0)
        assert s.max_depth == pytest.approx(1.0)
        assert s.depth_fade == pytest.approx(0.0)
        assert s.min_zoom == pytest.approx(0.0)
        assert s.max_zoom == pytest.approx(3.4028235e38)
        assert s.rotation == pytest.approx(0.0)
        assert s.offset == pytest.approx((0.0, 0.0))
        assert s.horizon_fade_angle == pytest.approx(5.0)

    def test_label_flags_defaults(self):
        """Default LabelFlags are all False."""
        f = _native.LabelFlags()
        assert f.underline is False
        assert f.small_caps is False
        assert f.leader is False

    def test_label_style_set_fields(self):
        """Write and read back each LabelStyle field."""
        s = _native.LabelStyle()
        s.size = 24.0
        assert s.size == pytest.approx(24.0)

        s.color = (1.0, 0.0, 0.0, 0.5)
        assert s.color == pytest.approx((1.0, 0.0, 0.0, 0.5))

        s.halo_color = (0.0, 0.0, 0.0, 1.0)
        assert s.halo_color == pytest.approx((0.0, 0.0, 0.0, 1.0))

        s.halo_width = 3.0
        assert s.halo_width == pytest.approx(3.0)

        s.priority = 10
        assert s.priority == 10

        s.min_depth = 0.1
        assert s.min_depth == pytest.approx(0.1)

        s.max_depth = 0.9
        assert s.max_depth == pytest.approx(0.9)

        s.depth_fade = 0.5
        assert s.depth_fade == pytest.approx(0.5)

        s.min_zoom = 2.0
        assert s.min_zoom == pytest.approx(2.0)

        s.max_zoom = 100.0
        assert s.max_zoom == pytest.approx(100.0)

        s.rotation = 1.57
        assert s.rotation == pytest.approx(1.57)

        s.offset = (10.0, -5.0)
        assert s.offset == pytest.approx((10.0, -5.0))

        s.horizon_fade_angle = 15.0
        assert s.horizon_fade_angle == pytest.approx(15.0)

    def test_label_flags_set(self):
        """Write and read back each LabelFlags field."""
        f = _native.LabelFlags()
        f.underline = True
        assert f.underline is True
        f.small_caps = True
        assert f.small_caps is True
        f.leader = True
        assert f.leader is True

    def test_label_style_color_tuple(self):
        """Color is a 4-element tuple."""
        s = _native.LabelStyle()
        c = s.color
        assert isinstance(c, tuple)
        assert len(c) == 4

    def test_label_style_repr(self):
        """LabelStyle has a useful repr."""
        s = _native.LabelStyle()
        r = repr(s)
        assert "LabelStyle" in r
        assert "size=" in r

    def test_label_style_from_kwargs(self):
        """Construct LabelStyle with keyword arguments."""
        flags = _native.LabelFlags(underline=True, leader=True)
        s = _native.LabelStyle(
            size=20.0,
            color=(1.0, 0.5, 0.0, 1.0),
            halo_width=2.0,
            priority=5,
            rotation=0.5,
            offset=(8.0, 4.0),
            flags=flags,
            horizon_fade_angle=10.0,
        )
        assert s.size == pytest.approx(20.0)
        assert s.color == pytest.approx((1.0, 0.5, 0.0, 1.0))
        assert s.halo_width == pytest.approx(2.0)
        assert s.priority == 5
        assert s.rotation == pytest.approx(0.5)
        assert s.offset == pytest.approx((8.0, 4.0))
        assert s.flags.underline is True
        assert s.flags.small_caps is False
        assert s.flags.leader is True
        assert s.horizon_fade_angle == pytest.approx(10.0)


# ===========================================================================
# Checklist node-ID aliases
# ===========================================================================
# The execution checklist (docs/plans/2026-02-19-api-consolidation-execution-
# checklist.md) references specific test node IDs.  The canonical tests live
# in the classes above; the functions below are thin aliases so that
# ``pytest tests/test_api_contracts.py::test_<node_id>`` resolves.

# ---- P0.2 ----

def test_offscreen_prefers_scene_method():
    """P0.2 alias: offscreen path prefers Scene.render_rgba."""
    TestOffscreenRenderRouting().test_native_scene_detected_correctly()


def test_offscreen_fallback_without_scene():
    """P0.2 alias: offscreen fallback works when no scene provided."""
    TestOffscreenRenderRouting().test_fallback_used_when_no_scene()


def test_viewer_set_msaa_no_module_level_native_dependency():
    """P0.2 alias: set_msaa_samples is NOT a module-level native function."""
    TestMsaaSetterRouting().test_set_msaa_samples_not_on_module()


# ---- P0.3 ----

def test_frame_importable():
    """P0.3 alias: Frame is registered and importable."""
    TestOrphanedClassesRegistered().test_frame_registered()


def test_sdf_classes_importable():
    """P0.3 alias: SDF classes are registered and importable."""
    t = TestOrphanedClassesRegistered()
    t.test_sdf_primitive_registered()
    t.test_sdf_scene_registered()
    t.test_sdf_scene_builder_registered()


# ---- P0.4 ----

def test_mesh_tbn_native_exports():
    """P0.4 alias: mesh TBN functions are exported on the native module."""
    t = TestTbnFunctionsExported()
    t.test_mesh_generate_cube_tbn_registered()
    t.test_mesh_generate_plane_tbn_registered()


def test_mesh_tbn_native_shape_contract():
    """P0.4 alias: mesh TBN returns dict with correct keys and structure."""
    t = TestTbnFunctionsExported()
    t.test_cube_tbn_dict_keys()
    t.test_cube_vertex_dict_structure()
    t.test_cube_tbn_dict_structure()


# ---- P0.5 ----

def test_scene_stub_matches_runtime():
    """P0.5: Verify that __init__.pyi Scene stubs match runtime Scene methods.

    Every ``def <name>`` declared in the Scene class inside __init__.pyi
    must be present as a callable attribute on the native Scene class.
    """
    import pathlib, re

    stub_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "python" / "forge3d" / "__init__.pyi"
    )
    assert stub_path.exists(), f"Stub file not found: {stub_path}"

    text = stub_path.read_text(encoding="utf-8")

    # Extract method names from the Scene class block in the stub.
    # The Scene class starts with "class Scene:" and ends at the next
    # top-level class/def/variable.
    scene_match = re.search(r"^class Scene.*?:", text, re.MULTILINE)
    assert scene_match, "Could not find 'class Scene' in __init__.pyi"

    scene_start = scene_match.end()
    # Find the end of the Scene class: next top-level class or end of file
    next_class = re.search(r"^class \w", text[scene_start:], re.MULTILINE)
    scene_block = text[scene_start: scene_start + next_class.start()] if next_class else text[scene_start:]

    stub_methods = set(re.findall(r"^\s+def (\w+)\(", scene_block, re.MULTILINE))
    # Exclude dunder methods
    stub_methods = {m for m in stub_methods if not m.startswith("_")}

    assert len(stub_methods) >= 10, (
        f"Only found {len(stub_methods)} stub methods -- parsing may be wrong"
    )

    missing = []
    for method_name in sorted(stub_methods):
        if not hasattr(_native.Scene, method_name):
            missing.append(method_name)

    assert not missing, (
        f"Stub declares methods not found on native Scene: {missing}"
    )


# ---- W1B-01 ----

def test_root_stub_declares_material_and_sun_contracts():
    """The root stub mirrors the exact public MaterialSet and sun API shape."""
    stub_path = Path(f3d.__file__).with_name("__init__.pyi")
    module = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
    declarations = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    for class_name in ("MaterialSet", "SunPosition"):
        declarations.update(
            {
                f"{class_name}.{node.name}": node
                for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef)
            }
        )

    def model(node: ast.FunctionDef):
        required = len(node.args.args) - len(node.args.defaults)
        parameters = tuple(
            (
                argument.arg,
                ast.unparse(argument.annotation) if argument.annotation else None,
                index >= required,
            )
            for index, argument in enumerate(node.args.args)
        )
        return (
            parameters,
            ast.unparse(node.returns),
            tuple(ast.unparse(decorator) for decorator in node.decorator_list),
        )

    f = ("float", False)
    fd = ("float", True)
    i = ("int", False)
    expected = {
        "MaterialSet.terrain_default": (
            (("triplanar_scale", *fd), ("normal_strength", *fd), ("blend_sharpness", *fd)),
            "MaterialSet",
            ("staticmethod",),
        ),
        "MaterialSet.custom": (
            (
                ("base_color", "Tuple[float, float, float]", False),
                ("metallic", *f),
                ("roughness", *f),
                ("triplanar_scale", *fd),
                ("normal_strength", *fd),
                ("blend_sharpness", *fd),
            ),
            "MaterialSet",
            ("staticmethod",),
        ),
        "MaterialSet.material_count": ((("self", None, False),), "int", ("property",)),
        "MaterialSet.triplanar_scale": ((("self", None, False),), "float", ("property",)),
        "MaterialSet.normal_strength": ((("self", None, False),), "float", ("property",)),
        "MaterialSet.blend_sharpness": ((("self", None, False),), "float", ("property",)),
        "SunPosition.azimuth": ((("self", None, False),), "float", ("property",)),
        "SunPosition.elevation": ((("self", None, False),), "float", ("property",)),
        "SunPosition.to_direction": (
            (("self", None, False),),
            "Tuple[float, float, float]",
            (),
        ),
        "SunPosition.is_daytime": ((("self", None, False),), "bool", ()),
        "sun_position": (
            (("latitude", *f), ("longitude", *f), ("datetime_utc", "str", False)),
            "SunPosition",
            (),
        ),
        "sun_position_utc": (
            (
                ("latitude", *f),
                ("longitude", *f),
                ("year", *i),
                ("month", *i),
                ("day", *i),
                ("hour", *i),
                ("minute", *i),
                ("second", "int", True),
            ),
            "SunPosition",
            (),
        ),
    }
    assert {name: model(declarations[name]) for name in expected} == expected


def test_root_material_and_sun_runtime_contracts():
    """Root exports retain native identity, exact signatures, and read-only state."""
    public_names = ("MaterialSet", "SunPosition", "sun_position", "sun_position_utc")
    for name in public_names:
        assert name in f3d.__all__
        assert getattr(f3d, name) is getattr(_native, name)

    expected_signatures = {
        f3d.MaterialSet.terrain_default: (
            "(triplanar_scale=6.0, normal_strength=1.0, blend_sharpness=4.0)"
        ),
        f3d.MaterialSet.custom: (
            "(base_color, metallic, roughness, triplanar_scale=1.0, "
            "normal_strength=0.0, blend_sharpness=1.0)"
        ),
        f3d.SunPosition.to_direction: "(self, /)",
        f3d.SunPosition.is_daytime: "(self, /)",
        f3d.sun_position: "(latitude, longitude, datetime_utc)",
        f3d.sun_position_utc: (
            "(latitude, longitude, year, month, day, hour, minute, second=0)"
        ),
    }
    for callable_object, expected in expected_signatures.items():
        assert str(inspect.signature(callable_object)) == expected

    with pytest.raises(TypeError):
        f3d.MaterialSet()
    with pytest.raises(TypeError):
        f3d.SunPosition()

    material = f3d.MaterialSet.custom((0.2, 0.3, 0.4), 0.0, 0.5)
    assert material.material_count == 1
    assert material.triplanar_scale == pytest.approx(1.0)
    assert material.normal_strength == pytest.approx(0.0)
    assert material.blend_sharpness == pytest.approx(1.0)
    with pytest.raises(AttributeError):
        material.material_count = 2

    position = f3d.sun_position_utc(0.0, 0.0, 2024, 1, 1, 12, 0)
    assert isinstance(position, f3d.SunPosition)
    assert isinstance(position.azimuth, float)
    assert isinstance(position.elevation, float)
    assert len(position.to_direction()) == 3
    assert isinstance(position.is_daytime(), bool)
    with pytest.raises(AttributeError):
        position.azimuth = 0.0
