# python/forge3d/__init__.py
# Public Python API for forge3d terrain renderer
"""
forge3d - GPU-accelerated terrain rendering library.

Core API:
    open_viewer_async   - Launch the IPC-controlled interactive viewer
    open_viewer         - Launch the blocking interactive viewer
    TerrainRenderer     - Native GPU terrain renderer
    Renderer            - Fallback CPU renderer
    
Configuration:
    TerrainRenderParams - Terrain rendering parameters
    RendererConfig      - Renderer configuration
    
Utilities:
    numpy_to_png        - Save numpy array as PNG
    png_to_numpy        - Load PNG as numpy array
    has_gpu             - Check GPU availability
"""

__version__ = "1.35.0"
version = __version__

import numpy as np

from . import terrain
from .terrain import VTStore, open_vt_store

from ._png import load_png_rgba as _load_png_rgba
from ._png import save_png as _save_png

# -----------------------------------------------------------------------------
# Native module loading
# -----------------------------------------------------------------------------
from ._native import (
    get_native_module as _get_native_module,
    native_import_error,
)
from ._gpu import (
    enumerate_adapters,
    device_probe,
    has_gpu,
    get_device,
)
from .mem import (
    memory_metrics,
    set_budget_policy,
    get_budget_policy,
    budget_remaining,
    utilization_ratio,
    override_memory_limit,
)

_NATIVE_MODULE = _get_native_module()

# -----------------------------------------------------------------------------
# Native exports (when available)
# -----------------------------------------------------------------------------
_NATIVE_ONLY_EXPORTS = (
        "Scene",
        "Session",
        "Colormap1D",
        "MaterialSet",
        "IBL",
        "OverlayLayer",
        "TerrainRenderParams",
        "TerrainRenderer",
        "Frame",
        "AovFrame",
        "HdrFrame",
        "OfflineBatchResult",
        "OfflineMetrics",
        "Light",
        "Atmosphere",
        "open_viewer",
        "open_terrain_viewer",
        "PickResult",  # Feature B: Picking system (Plan 1)
        "TerrainQueryResult",  # Feature B: Plan 2
        "SelectionStyle",  # Feature B: Plan 2
        "RichPickResult",  # Feature B: Plan 3
        "HighlightStyle",  # Feature B: Plan 3
        "LassoState",  # Feature B: Plan 3
        "HeightfieldHit",  # Feature B: Plan 3
        "CameraKeyframe",  # Feature C: Camera animation keyframe editing
        "CameraAnimation",  # Feature C: Camera animation (Plan 1 MVP)
        "CameraState",  # Feature C: Camera animation (Plan 1 MVP)
        "SunPosition",  # P0.3/M2: Sun ephemeris
        "sun_position",  # P0.3/M2: Sun ephemeris function
        "sun_position_utc",  # P0.3/M2: Sun ephemeris function (components)
        "set_point_shape_mode",
        "set_point_lod_threshold",
        "is_weighted_oit_available",
        "vector_oit_and_pick_demo",
        "vector_render_oit_py",
        "vector_render_oit_edl_py",
        "vector_render_pick_map_py",
        "vector_render_oit_and_pick_py",
        "vector_render_polygons_fill_py",
        "vector_render_analytic_py",
        "vector_coverage_primitives_py",
        "ClipmapConfig",  # P2.1/M5: Clipmap terrain
        "ClipmapMesh",  # P2.1/M5: Clipmap terrain
        "clipmap_generate_py",  # P2.1/M5: Clipmap generation function
        "calculate_triangle_reduction_py",  # P2.1/M5: Triangle reduction calculation
        "PointBuffer",  # P2.1: point cloud GPU buffer
        "copc_laz_enabled",  # P2.2: COPC/LAZ feature gate
        "read_laz_points_info",  # P2.2: LAZ fixture decode info
        "read_laz_point_attributes",  # P2.3: LAZ classification/intensity samples
        "copc_read_node_points",  # P2.3: native COPC node decode
        "render_adjudication_pair",  # AEQUITAS: PT-vs-raster adjudication pair
        "hybrid_render_terrain_reference",  # PROMETHEUS: terrain PT reference
        "hybrid_render_aether_spectral_reference",  # AETHER: stochastic PT acceptance reference
        "AtmosphereLutHandle",  # AETHER: exact typed LUT payload handoff
        "atmosphere_bake_luts",  # AETHER: shipped LUT resolver / offline bake
        "atmosphere_spectral_to_linear_rgb",  # AETHER: spectral conversion
        "atmosphere_generate_environment",  # AETHER: validation environment
        "atmosphere_reference_aerial",  # AETHER: CPU transport diagnostic
        "render_brdf_tile",  # CENSOR: certified BRDF pixel render
        "render_brdf_tile_overrides",  # CENSOR: certified BRDF pixel render
        "seal_provenance",  # VERITAS: Merkle+Ed25519 seal over VT provenance
        "verify_provenance",  # VERITAS: native manifest verification
        "declutter",  # CARTOGRAPHER-PRIME: generic typed label solve
        "declutter_optimal",  # CARTOGRAPHER-PRIME: compatibility alias
        "LabelRationale",  # CARTOGRAPHER-PRIME: grounded solver rationale
        "native_degradations",  # CENSOR: global degradation sink snapshot
        "clear_native_degradations",  # CENSOR: global degradation sink reset
        "terrain_culling_stats",  # TESSELLA: two-phase HZB counters
        "terrain_visibility_stats",  # TESSELLA: visibility resolve counters
        "terrain_vt_stats",  # TESSELLA: virtual-texture residency counters
        "terrain_seam_stats",  # TESSELLA: clipmap seam analysis
        "capabilities",  # CENSOR: negotiated GPU capability report
        "render_execution_report",  # CENSOR: last-render execution certificate JSON
        "begin_render_execution_capture",  # CENSOR: Python-render capture begin
        "finish_render_execution_capture",  # CENSOR: Python-render capture finish
        "abort_render_execution_capture",  # CENSOR: Python-render capture abort
        "sign_render_certificate_digest",  # CENSOR: native Ed25519 signer
        "request_host_visible_allocation_for_test",  # CENSOR: budget-enforce test helper
        "shader_report",  # PROBATUM: WGSL proof report
        "anamnesis_leaf_key",  # ANAMNESIS: native content key primitive
        "anamnesis_pass_key",  # ANAMNESIS: native hermetic pass key
        "anamnesis_engine_fingerprint",  # ANAMNESIS: pinned engine identity
        "anamnesis_store_verify",  # ANAMNESIS: native unified-store verifier
        "anamnesis_store_gc",  # ANAMNESIS: native unified-store LRU
        "anamnesis_store_put_leaf",  # ANAMNESIS: native store interoperability
        "anamnesis_store_get",  # ANAMNESIS: native store interoperability
        "anamnesis_restore_rgba8",  # ANAMNESIS: portable GPU resource restore
        "compress_dem",  # COMPENDIUM: deterministic F3DZ encoder
        "decompress_dem",  # COMPENDIUM: fail-closed F3DZ decoder
        "verify_dem",  # COMPENDIUM: CRC/error-bound verifier
        "encode_bc7_rgba8",  # TESSELLA: deterministic BC7 mode-6 encoder
        "decode_bc7_rgba8",  # TESSELLA: deterministic BC7 mode-6 decoder
        "encode_bc5_rg8",  # TESSELLA: deterministic BC5 encoder
        "decode_bc5_rg8",  # TESSELLA: deterministic BC5 decoder
        "dd_selftest",  # DUPLA: GPU DD exactness canary
        "dd_harness",  # DUPLA: GPU DD bounds proof
        "dd_jitter_demo",  # DUPLA: Everest absolute-coordinate demo
)

if _NATIVE_MODULE is not None:
    for _name in _NATIVE_ONLY_EXPORTS:
        if hasattr(_NATIVE_MODULE, _name):
            globals()[_name] = getattr(_NATIVE_MODULE, _name)


# -----------------------------------------------------------------------------
# CENSOR: typed GPU-error exceptions
# -----------------------------------------------------------------------------
# Prefer the native exception classes so ``except forge3d.MemoryBudgetExceeded``
# catches errors raised across the PyO3 boundary. When the extension is absent
# (pure-Python / certificate-only installs), fall back to RuntimeError
# subclasses so the names still import and ``except`` clauses stay valid.
if _NATIVE_MODULE is not None and hasattr(_NATIVE_MODULE, "MemoryBudgetExceeded"):
    MemoryBudgetExceeded = _NATIVE_MODULE.MemoryBudgetExceeded
else:
    class MemoryBudgetExceeded(RuntimeError):
        """Raised when an operation would exceed the host-visible memory budget."""

if _NATIVE_MODULE is not None and hasattr(_NATIVE_MODULE, "DegradedCapability"):
    DegradedCapability = _NATIVE_MODULE.DegradedCapability
else:
    class DegradedCapability(RuntimeError):
        """Raised when a required GPU capability is unavailable or degraded."""

if _NATIVE_MODULE is not None and hasattr(_NATIVE_MODULE, "TransformFailed"):
    TransformFailed = _NATIVE_MODULE.TransformFailed
else:
    class TransformFailed(RuntimeError):
        """Raised when GIS reprojection cannot transform one or more pixels."""


class _NativeSymbolMissing(AttributeError):
    """Raised when a native-only forge3d symbol is accessed but unavailable.

    Inherits AttributeError so ``hasattr(forge3d, name)`` probes stay False
    instead of erroring; ``from forge3d import Scene`` still surfaces an
    import-flavored failure because the import machinery converts a module
    ``__getattr__`` AttributeError into ImportError. (CPython forbids
    inheriting from both ImportError and AttributeError — their C-level
    instance layouts conflict.)
    """


def __getattr__(name: str):
    if name == "anamnesis":
        import importlib

        module = importlib.import_module(".anamnesis", __name__)
        globals()[name] = module
        return module
    if name in _NATIVE_ONLY_EXPORTS:
        if _NATIVE_MODULE is None:
            cause = native_import_error()
            detail = (
                f"the compiled extension forge3d._forge3d failed to import: {cause!r}"
                if cause is not None
                else "the compiled extension forge3d._forge3d is not built"
            )
            raise _NativeSymbolMissing(
                f"forge3d.{name} requires the native extension, but {detail}. "
                "Reinstall the wheel (pip install --force-reinstall forge3d) or "
                "rebuild from a checkout with: maturin develop --release"
            )
        raise _NativeSymbolMissing(
            f"forge3d.{name} is not provided by this build of forge3d._forge3d "
            "(the extension imported, but the symbol is missing — typically the "
            "wheel was built without the Cargo feature that provides it). "
            "Rebuild with the matching feature enabled, e.g.: "
            "maturin develop --release --features <feature>"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# -----------------------------------------------------------------------------
# Colormaps
# -----------------------------------------------------------------------------
from .colormaps import (
    get as get_colormap,
    available as available_colormaps,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from .config import RendererConfig, load_renderer_config
from .terrain_params import (
    TerrainRenderParams as TerrainRenderParamsConfig,
    LightSettings,
    IblSettings,
    ShadowSettings,
    FogSettings,
    ReflectionSettings,
    WaterSettings,
    CloudSettings,
    HeightAoSettings,
    ScreenSpaceSettings,
    SunVisibilitySettings,
    ProbeSettings,
    ReflectionProbeSettings,
    DetailSettings,
    MaterialNoiseSettings,
    MaterialLayerSettings,
    PomSettings,
    TriplanarSettings,
    LodSettings,
    SamplingSettings,
    ClampSettings,
    DenoiseSettings,
    OfflineQualitySettings,
    VTLayerFamily,
    TerrainVTSettings,
    validate_terrain_vt_support,
    SkySettings,
)
from .offline import OfflineProgress, OfflineResult, render_offline
from .denoise_oidn import oidn_available, oidn_denoise
from . import presets
from . import geo
from . import terrain
from . import animation
from . import gis
from . import thematic
from . import camera_rigs
from . import codec

# -----------------------------------------------------------------------------
# Core rendering API
# -----------------------------------------------------------------------------
from .path_tracing import (
    ExperimentalSyntheticOutput,
    PathTracer,
    hybrid_render_terrain_reference,
    make_camera,
)

# -----------------------------------------------------------------------------
# Interactive Viewer API
# -----------------------------------------------------------------------------
from .viewer import (
    LabelBatchResult,
    NormalizedExtent,
    VectorOverlayVertex,
    ViewerHandle,
    WorldPosition,
    open_viewer,
    open_viewer_async,
)
from . import (
    astro,
    atmosphere,
    viewer_ipc,
    colors,
    interactive,
    datasets,
    widgets,
    sky,
    smoke,
    verify,
)
from .atmosphere import AtmosphereSettings, SUN_ELEVATION_SWEEP_DEG
from .datasets import (
    available as available_datasets,
    bundled as bundled_datasets,
    dataset_info,
    fetch as fetch_dataset,
    fetch_cityjson,
    fetch_copc,
    fetch_dem,
    list_datasets,
    mini_dem,
    mini_dem_path,
    remote as remote_datasets,
    sample_boundaries,
    sample_boundaries_path,
)
from .widgets import ViewerWidget, widgets_available
from ._license import LicenseError, set_license_key
from . import terrain_scatter

# -----------------------------------------------------------------------------
# Fallback Renderer class
# -----------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Mapping
from .certificate import _captured_cpu_render


class Renderer:
    """Fallback CPU renderer for terrain.
    
    Args:
        width: Output image width in pixels
        height: Output image height in pixels
        config: Optional renderer configuration
        **kwargs: Override keywords (brdf, shadows, etc.)
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        config: RendererConfig | Mapping[str, Any] | str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        from .config import split_renderer_overrides
        
        self.width = int(width)
        self.height = int(height)
        overrides, remaining = split_renderer_overrides(dict(kwargs))
        if remaining:
            raise TypeError(f"Unexpected arguments: {', '.join(sorted(str(k) for k in remaining))}")
        self._config = load_renderer_config(config, overrides)
        self._exposure = float(self._config.lighting.exposure)

    def get_config(self) -> dict:
        """Return renderer configuration as dict."""
        return self._config.to_dict()

    def apply_preset(self, name: str, **overrides: Any) -> None:
        """Apply a preset to the renderer configuration."""
        preset_map = presets.get(name)
        self._config = RendererConfig.from_mapping(preset_map, self._config)
        if overrides:
            self._config = load_renderer_config(self._config, overrides)

    @_captured_cpu_render(
        "python.renderer.render_triangle_rgba", "renderer.cpu_triangle", draw_calls=1
    )
    def render_triangle_rgba(
        self, *, certificate: bool | str | Path = False, cache: str | Path | None = None
    ) -> np.ndarray:
        """Render a basic triangle pattern (fallback test method)."""
        _ = cache
        from . import _degradation

        _degradation.record(
            "cpu_fallback",
            "renderer.triangle",
            "fallback Renderer uses the deterministic CPU triangle implementation",
        )
        img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        cx, cy = self.width // 2, self.height // 2
        size = min(self.width, self.height) // 4
        for y in range(self.height):
            for x in range(self.width):
                dx, dy = x - cx, y - cy
                if abs(dx) + abs(dy) < size and y > cy - size // 2:
                    img[y, x] = [128, 64, 32, 255]
                else:
                    img[y, x] = [16, 16, 24, 255]
        return img

    def render_triangle_png(
        self,
        path: str | Path,
        *,
        certificate: bool | str | Path = False,
        cache: str | Path | None = None,
    ) -> None:
        """Render triangle to PNG file."""
        numpy_to_png(path, self.render_triangle_rgba(certificate=certificate, cache=cache))


# -----------------------------------------------------------------------------
# Image I/O utilities
# -----------------------------------------------------------------------------
def numpy_to_png(path, array: np.ndarray) -> None:
    """Save an existing numpy array as PNG.

    Outside CENSOR's render-certificate scope: array-to-PNG I/O executes no
    render.
    """
    path_str = str(path)
    if not path_str.lower().endswith('.png'):
        raise ValueError(f"File must have .png extension, got {path_str}")

    arr = np.ascontiguousarray(array)
    if arr.dtype != np.uint8:
        raise RuntimeError("Array must be uint8")

    _save_png(path, arr)


def png_to_numpy(path) -> np.ndarray:
    """Load an existing PNG into a numpy array.

    Outside CENSOR's render-certificate scope: PNG-to-array I/O executes no
    render.
    """
    return _load_png_rgba(path)


def dem_stats(heightmap: np.ndarray) -> dict:
    """Get DEM statistics."""
    if heightmap.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(heightmap.min()),
        "max": float(heightmap.max()),
        "mean": float(heightmap.mean()),
        "std": float(heightmap.std()),
    }


# -----------------------------------------------------------------------------
# Geometry module
# -----------------------------------------------------------------------------
from . import geometry
from . import io

# -----------------------------------------------------------------------------
# P4: Map Plate / Creator Workflow
# -----------------------------------------------------------------------------
from .map_plate import MapPlate, MapPlateConfig, BBox, PlateRegion
from .legend import Legend, LegendConfig
from .scale_bar import ScaleBar, ScaleBarConfig
from .north_arrow import NorthArrow, NorthArrowConfig
from .graticule import GraticuleSpec, generate_graticule

# -----------------------------------------------------------------------------
# P5-export: Vector Export (SVG/PDF)
# -----------------------------------------------------------------------------
from .export import (
    VectorScene,
    VectorStyle as ExportVectorStyle,
    LabelStyle as ExportLabelStyle,
    Polygon as ExportPolygon,
    Polyline as ExportPolyline,
    Label as ExportLabel,
    Bounds as ExportBounds,
    generate_svg,
    export_svg,
    export_pdf,
    validate_svg,
)

# -----------------------------------------------------------------------------
# Helpers (offscreen rendering, frame dumping)
# -----------------------------------------------------------------------------
from .helpers.offscreen import (
    render_offscreen_rgba,
    save_png_deterministic,
    rgba_to_png_bytes,
)
from .helpers.frame_dump import FrameDumper, dump_frame_sequence

# -----------------------------------------------------------------------------
# Scene Bundle (.forge3d)
# -----------------------------------------------------------------------------
_BUNDLE_EXPORT_NAMES = (
    "save_bundle",
    "load_bundle",
    "is_bundle",
    "BundleManifest",
    "LoadedBundle",
    "CameraBookmark",
    "RasterOverlaySpec",
    "SceneBaseState",
    "ReviewLayer",
    "SceneVariant",
    "SceneState",
    "TerrainMeta",
    "BUNDLE_VERSION",
)
_AVAILABLE_BUNDLE_EXPORTS: list[str] = []
try:
    from . import bundle as _bundle_module
except Exception:
    # Keep unrelated imports working while bundle.py is mid-edit or otherwise
    # unavailable. Direct bundle consumers can still import forge3d.bundle.
    _bundle_module = None
else:
    for _name in _BUNDLE_EXPORT_NAMES:
        if hasattr(_bundle_module, _name):
            globals()[_name] = getattr(_bundle_module, _name)
            _AVAILABLE_BUNDLE_EXPORTS.append(_name)

# -----------------------------------------------------------------------------
# P3-reproject: CRS utilities
# -----------------------------------------------------------------------------
from .crs import (
    proj_available,
    transform_coords,
    reproject_geom,
    crs_to_epsg,
    get_crs_from_rasterio,
    get_crs_from_geopandas,
    body_info,
    areoid_undulation,
    geoid_undulation,
    orthometric_to_ellipsoidal,
    ellipsoidal_to_orthometric,
    geodesic_inverse,
    geodesic_direct,
    wgs84_to_ecef,
    ecef_to_wgs84,
    dem_orthometric_to_ellipsoidal,
)

# -----------------------------------------------------------------------------
# P4: 3D Buildings Pipeline
# -----------------------------------------------------------------------------
from .buildings import (
    Building,
    BuildingLayer,
    BuildingMaterial,
    add_buildings,
    add_buildings_cityjson,
    add_buildings_3dtiles,
    validate_building_layer_support,
    infer_roof_type,
    material_from_tags,
    material_from_name,
)

# -----------------------------------------------------------------------------
# Mapbox Style Spec Import
# -----------------------------------------------------------------------------
from .style import (
    load_style,
    parse_style,
    apply_style,
    parse_color,
    validate_style_support,
    vector_overlay_configs_from_style,
    label_layer_contracts_from_style,
    paint_to_vector_style,
    layout_to_label_style,
    layer_to_vector_style,
    layer_to_label_style,
    StyleSpec,
    StyleLayer,
    VectorStyle as StyleVectorStyle,
    LabelStyle as StyleLabelStyle,
    PaintProps,
    LayoutProps,
)

# -----------------------------------------------------------------------------
# Product diagnostics
# -----------------------------------------------------------------------------
from .diagnostics import (
    Diagnostic,
    LayerSummary,
    P2_FEATURE_DIAGNOSTIC_CODES,
    REQUIRED_DIAGNOSTIC_CODES,
    RenderFailurePolicy,
    SeverityPolicy,
    SupportMatrixEntry,
    ValidationReport,
    crs_mismatch_diagnostic,
    estimated_gpu_memory_diagnostic,
    memory_tracking_completeness_report,
    experimental_feature_diagnostic,
    label_rejection_summary_diagnostic,
    missing_glyphs_diagnostic,
    missing_texture_path_diagnostic,
    missing_uvs_diagnostic,
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
    python_public_3dtiles_incomplete_diagnostic,
    unavailable_cache_lod_stats_diagnostic,
    unsupported_instancing_path_diagnostic,
    unsupported_style_field_diagnostic,
    unsupported_style_layer_type_diagnostic,
    unsupported_texture_format_diagnostic,
    validate_label_support,
    vt_unsupported_family_diagnostic,
)

# -----------------------------------------------------------------------------
# Deterministic label planning
# -----------------------------------------------------------------------------
from .label_plan import (
    AcceptedLabel,
    KeepoutRegion,
    LabelCandidate,
    LabelPlan,
    PriorityClass,
    RejectedLabel,
)

# -----------------------------------------------------------------------------
# Typed MapScene recipe contract
# -----------------------------------------------------------------------------
from .map_scene import (
    CompiledScenePlan,
    FontAtlas,
    FontFallbackRange,
    LabelLayer,
    LightingPreset,
    MapFurnitureLayer,
    MapScene,
    MapSceneNativeUnavailable,
    MapSceneTextLayoutError,
    BuildingLayer as MapSceneBuildingLayer,
    OrbitCamera,
    OutputSpec,
    PointCloudLayer,
    RasterOverlay,
    ReproducibilityProfile,
    SceneRecipe,
    TerrainSource,
    Tiles3DLayer,
    TypographySettings,
    VectorOverlay,
)
from . import recipe_manifest
from .alignment import (
    alignment_report,
    alignment_residual,
    reproject_dem_to_target,
    resample_raster_to_grid,
    transform_features as align_transform_features,
    transform_geometry as align_transform_geometry,
)
from .text_atlas import (
    BakedAtlas,
    bake_atlas,
    default_latin_atlas_paths,
    load_atlas_metrics,
    save_atlas,
    validate_atlas_metrics,
)
from . import text as text
from . import precision as precision
from .precision import dd_harness, dd_jitter_demo, dd_selftest

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    "terrain",
    "VTStore",
    "open_vt_store",
    # Version
    "__version__",
    "version",
    "text",
    "verify",
    "codec",
    "precision",
    "atmosphere",
    "AtmosphereSettings",
    "SUN_ELEVATION_SWEEP_DEG",
    "dd_selftest",
    "dd_harness",
    "dd_jitter_demo",
    # Core rendering
    "Renderer",
    "PathTracer",
    "ExperimentalSyntheticOutput",
    "make_camera",
    # Native types (when available)
    "Scene",
    "Session",
    "Colormap1D",
    "MaterialSet",
    "IBL",
    "OverlayLayer",
    "TerrainRenderParams",
    "TerrainRenderer",
    "Frame",
    "AovFrame",
    "HdrFrame",
    "CameraKeyframe",
    "CameraAnimation",
    "CameraState",
    # P0.3/M2: Sun ephemeris
    "SunPosition",
    "sun_position",
    "sun_position_utc",
    "geo",
    "terrain",
    # AEQUITAS: PT-vs-raster adjudication
    "render_adjudication_pair",
    # PROMETHEUS: GPU terrain path-traced reference
    "hybrid_render_terrain_reference",
    "hybrid_render_aether_spectral_reference",
    # AETHER: spectral atmosphere bake/validation surface
    "AtmosphereLutHandle",
    "atmosphere_bake_luts",
    "atmosphere_spectral_to_linear_rgb",
    "atmosphere_generate_environment",
    "atmosphere_reference_aerial",
    # CENSOR: certified BRDF pixel renders
    "render_brdf_tile",
    "render_brdf_tile_overrides",
    # VERITAS: per-pixel cryptographic provenance
    "seal_provenance",
    "verify_provenance",
    # CARTOGRAPHER-PRIME: bounded-optimal label solve + rationale
    "declutter",
    "declutter_optimal",
    "LabelRationale",
    # CENSOR: global degradation sink
    "native_degradations",
    "clear_native_degradations",
    "terrain_culling_stats",
    "terrain_visibility_stats",
    "terrain_vt_stats",
    "terrain_seam_stats",
    # CENSOR: negotiated GPU capability report
    "capabilities",
    # CENSOR: last-render execution certificate JSON
    "render_execution_report",
    "begin_render_execution_capture",
    "finish_render_execution_capture",
    "abort_render_execution_capture",
    # CENSOR: native Ed25519 certificate signer
    "sign_render_certificate_digest",
    # CENSOR: budget-enforce test helper
    "request_host_visible_allocation_for_test",
    "shader_report",
    "compress_dem",
    "decompress_dem",
    "verify_dem",
    "encode_bc7_rgba8",
    "decode_bc7_rgba8",
    "encode_bc5_rg8",
    "decode_bc5_rg8",
    # CENSOR: typed GPU-error exceptions
    "MemoryBudgetExceeded",
    "DegradedCapability",
    "TransformFailed",
    # Configuration
    "RendererConfig",
    "TerrainRenderParamsConfig",
    "LightSettings",
    "IblSettings",
    "ShadowSettings",
    "FogSettings",
    "ReflectionSettings",
    "WaterSettings",
    "CloudSettings",
    "HeightAoSettings",
    "ScreenSpaceSettings",
    "SunVisibilitySettings",
    "ProbeSettings",
    "ReflectionProbeSettings",
    "DetailSettings",
    "MaterialNoiseSettings",
    "MaterialLayerSettings",
    "PomSettings",
    "TriplanarSettings",
    "LodSettings",
    "SamplingSettings",
    "ClampSettings",
    "DenoiseSettings",
    "OfflineQualitySettings",
    "VTLayerFamily",
    "TerrainVTSettings",
    "validate_terrain_vt_support",
    "SkySettings",
    "OfflineProgress",
    "OfflineResult",
    "render_offline",
    "oidn_available",
    "oidn_denoise",
    "presets",
    # Colormaps
    "get_colormap",
    "available_colormaps",
    # GPU utilities
    "has_gpu",
    "get_device",
    "enumerate_adapters",
    "device_probe",
    "native_import_error",
    "set_point_shape_mode",
    "set_point_lod_threshold",
    "is_weighted_oit_available",
    "vector_oit_and_pick_demo",
    "vector_render_oit_py",
    "vector_render_oit_edl_py",
    "vector_render_pick_map_py",
    "vector_render_oit_and_pick_py",
    "vector_render_polygons_fill_py",
    "vector_render_analytic_py",
    "vector_coverage_primitives_py",
    "memory_metrics",
    "PointBuffer",
    "copc_laz_enabled",
    "read_laz_points_info",
    "read_laz_point_attributes",
    "copc_read_node_points",
    "set_budget_policy",
    "get_budget_policy",
    "budget_remaining",
    "utilization_ratio",
    "override_memory_limit",
    # Image I/O
    "numpy_to_png",
    "png_to_numpy",
    "dem_stats",
    # Helpers
    "render_offscreen_rgba",
    "save_png_deterministic",
    "rgba_to_png_bytes",
    "FrameDumper",
    "dump_frame_sequence",
    # Modules
    "geometry",
    "gis",
    "thematic",
    "io",
    "terrain_scatter",
    "animation",
    "anamnesis",
    "camera_rigs",
    "datasets",
    "widgets",
    "astro",
    "sky",
    # Interactive viewer
    "open_viewer",
    "open_viewer_async",
    "ViewerHandle",
    "LabelBatchResult",
    "WorldPosition",
    "VectorOverlayVertex",
    "NormalizedExtent",
    "ViewerWidget",
    "widgets_available",
    # P4: Map Plate / Creator Workflow
    "MapPlate",
    "MapPlateConfig",
    "BBox",
    "PlateRegion",
    "Legend",
    "LegendConfig",
    "ScaleBar",
    "ScaleBarConfig",
    "NorthArrow",
    "NorthArrowConfig",
    "GraticuleSpec",
    "generate_graticule",
    # Viewer utilities
    "viewer_ipc",
    "colors",
    "interactive",
    # Datasets
    "mini_dem",
    "mini_dem_path",
    "sample_boundaries",
    "sample_boundaries_path",
    "available_datasets",
    "bundled_datasets",
    "remote_datasets",
    "list_datasets",
    "dataset_info",
    "fetch_dataset",
    "fetch_dem",
    "fetch_cityjson",
    "fetch_copc",
    # P5-export: Vector Export (SVG/PDF)
    "VectorScene",
    "ExportVectorStyle",
    "ExportLabelStyle",
    "ExportPolygon",
    "ExportPolyline",
    "ExportLabel",
    "ExportBounds",
    "generate_svg",
    "export_svg",
    "export_pdf",
    "validate_svg",
    # License management
    "set_license_key",
    "LicenseError",
    # Mapbox Style Spec
    "load_style",
    "parse_style",
    "apply_style",
    "parse_color",
    "validate_style_support",
    "vector_overlay_configs_from_style",
    "label_layer_contracts_from_style",
    "paint_to_vector_style",
    "layout_to_label_style",
    "layer_to_vector_style",
    "layer_to_label_style",
    "StyleSpec",
    "StyleLayer",
    "StyleVectorStyle",
    "StyleLabelStyle",
    "PaintProps",
    "LayoutProps",
    # Product diagnostics
    "Diagnostic",
    "LayerSummary",
    "P2_FEATURE_DIAGNOSTIC_CODES",
    "REQUIRED_DIAGNOSTIC_CODES",
    "RenderFailurePolicy",
    "SeverityPolicy",
    "SupportMatrixEntry",
    "ValidationReport",
    "crs_mismatch_diagnostic",
    "estimated_gpu_memory_diagnostic",
    "memory_tracking_completeness_report",
    "experimental_feature_diagnostic",
    "label_rejection_summary_diagnostic",
    "missing_glyphs_diagnostic",
    "missing_texture_path_diagnostic",
    "missing_uvs_diagnostic",
    "placeholder_fallback_diagnostic",
    "pro_gated_path_diagnostic",
    "python_public_3dtiles_incomplete_diagnostic",
    "unavailable_cache_lod_stats_diagnostic",
    "unsupported_instancing_path_diagnostic",
    "unsupported_style_field_diagnostic",
    "unsupported_style_layer_type_diagnostic",
    "unsupported_texture_format_diagnostic",
    "validate_label_support",
    "vt_unsupported_family_diagnostic",
    # Deterministic label planning
    "AcceptedLabel",
    "KeepoutRegion",
    "LabelCandidate",
    "LabelPlan",
    "PriorityClass",
    "RejectedLabel",
    # Typed MapScene recipe contract
    "MapScene",
    "MapSceneNativeUnavailable",
    "MapSceneTextLayoutError",
    "CompiledScenePlan",
    "SceneRecipe",
    "TerrainSource",
    "RasterOverlay",
    "VectorOverlay",
    "FontAtlas",
    "FontFallbackRange",
    "TypographySettings",
    "LabelLayer",
    "PointCloudLayer",
    "Tiles3DLayer",
    "MapSceneBuildingLayer",
    "MapFurnitureLayer",
    "OrbitCamera",
    "LightingPreset",
    "OutputSpec",
    "ReproducibilityProfile",
    "recipe_manifest",
    "alignment_report",
    "alignment_residual",
    "reproject_dem_to_target",
    "resample_raster_to_grid",
    "align_transform_features",
    "align_transform_geometry",
    "BakedAtlas",
    "bake_atlas",
    "default_latin_atlas_paths",
    "load_atlas_metrics",
    "save_atlas",
    "validate_atlas_metrics",
    # P3-reproject: CRS utilities
    "proj_available",
    "transform_coords",
    "reproject_geom",
    "crs_to_epsg",
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
    "get_crs_from_rasterio",
    "get_crs_from_geopandas",
    # P4: 3D Buildings Pipeline
    "Building",
    "BuildingLayer",
    "BuildingMaterial",
    "add_buildings",
    "add_buildings_cityjson",
    "add_buildings_3dtiles",
    "validate_building_layer_support",
    "infer_roof_type",
    "material_from_tags",
    "material_from_name",
]
__all__.extend(_AVAILABLE_BUNDLE_EXPORTS)
