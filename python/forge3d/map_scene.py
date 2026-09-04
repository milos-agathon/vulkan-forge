"""Typed MapScene recipe models for offline map-production workflows."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ._map_scene_common import (
    _has_explicit_crs_policy,
    _json_safe,
    _layer_id,
    _metadata,
    _metadata_dict,
    _same_crs,
    _sequence,
    _stable_hash,
)
from ._map_scene_labels import (
    _atlas_glyph_set,
    _diagnostic_for_layer,
    _feature_geometry,
    _feature_id,
    _feature_properties,
    _has_labels_or_plan,
    _label_plan_from_layer,
    _label_text_from_expression,
    _sample_label_geometry,
    _style_text_expression,
    _transform_label_geometry,
    _valid_geometry,
)
from ._map_scene_render import (
    MapSceneRenderLayerTypes,
    _building_features as _render_building_features,
    _building_fill_color as _render_building_fill_color,
    _building_height as _render_building_height,
    _building_properties as _render_building_properties,
    _building_rings as _render_building_rings,
    _building_roof_type as _render_building_roof_type,
    _dash_segments as _render_dash_segments,
    _color as _render_color,
    _composite_recipe_layers,
    _feature_color as _render_feature_color,
    _feature_number as _render_feature_number,
    _geometry_polygon_rings as _render_geometry_polygon_rings,
    _geometry_points as _render_geometry_points,
    _label_anchor as _render_label_anchor,
    _layout as _render_layout,
    _number as _render_number,
    _paint as _render_paint,
    _point_to_pixel as _render_point_to_pixel,
    _rgb as _render_rgb,
)
from ._map_scene_validation import (
    diagnostic_block,
    required_native_symbols,
    _BUILDING_ASSET_EXTENSIONS,
    _POINT_CLOUD_ASSET_EXTENSIONS,
    _RASTER_ASSET_EXTENSIONS,
    _TERRAIN_ASSET_EXTENSIONS,
    _asset_path_diagnostics,
    _diagnostic_codes_for_layer,
    _dimension_memory_bytes,
    _explicit_memory_bytes,
    _geometry_memory_bytes,
    _has_identity_path_or_metadata,
    _large_scene_resource_summary,
    _merge_report,
    _missing_crs_diagnostic,
    _missing_renderable_data_diagnostic,
    _missing_source_identity_diagnostic,
    _output_memory_bytes,
    _p2_building_texture_diagnostics,
    _p2_resource_availability_diagnostics,
    _point_cloud_memory_bytes,
    _source_kind,
    _source_path,
    _unsupported_feature_diagnostic,
    _unsupported_layer_type_diagnostic,
    _unsupported_output_format_diagnostic,
    _virtual_texture_report_from_metadata,
)
from .diagnostics import (
    Diagnostic,
    LayerSummary,
    RenderFailurePolicy,
    ValidationReport,
    crs_mismatch_diagnostic,
    estimated_gpu_memory_diagnostic,
    experimental_feature_diagnostic,
    missing_external_asset_diagnostic,
    missing_label_field_diagnostic,
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
    python_public_3dtiles_incomplete_diagnostic,
    unicode_coverage_gap_diagnostic,
    unavailable_terrain_sampler_diagnostic,
    unsupported_tile_feature_diagnostic,
    unsupported_tile_format_diagnostic,
    validate_label_support,
)

if TYPE_CHECKING:
    from .graticule import GraticuleSpec


def _render_layer_types() -> MapSceneRenderLayerTypes:
    return MapSceneRenderLayerTypes(
        raster_overlay=RasterOverlay,
        vector_overlay=VectorOverlay,
        label_layer=LabelLayer,
        point_cloud_layer=PointCloudLayer,
        building_layer=BuildingLayer,
    )


def _path_to_str(value: Any | None) -> str | None:
    return None if value is None else str(value)


class MapSceneNativeUnavailable(RuntimeError):
    """A MapScene layer requires native rendering that is unavailable.

    Raised instead of any CPU-placeholder fallback: the scene either renders
    through the native backend or fails with the structured diagnostic blocks
    carried on this exception (see ``_map_scene_validation.diagnostic_block``).
    """

    def __init__(self, blocks: Mapping[str, Any] | Sequence[Mapping[str, Any]]):
        if isinstance(blocks, Mapping):
            blocks = [blocks]
        self.diagnostics: list[dict[str, Any]] = [dict(block) for block in blocks]
        self.diagnostic: dict[str, Any] | None = self.diagnostics[0] if self.diagnostics else None
        summary = "; ".join(
            f"layer={block.get('layer')}: {block.get('reason')} "
            f"(required_native={block.get('required_native')})"
            for block in self.diagnostics
        )
        super().__init__(f"MapScene native rendering unavailable: {summary}")


class MapSceneTextLayoutError(RuntimeError):
    """Native label composition rejected an incomplete shaped-glyph payload."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        self.diagnostic = dict(diagnostic)
        self.diagnostics = [self.diagnostic]
        super().__init__(
            "MapScene text layout failed: "
            f"layer={self.diagnostic.get('layer')} "
            f"label={self.diagnostic.get('object_id')} "
            f"reason={self.diagnostic.get('reason')}"
        )


@dataclass(frozen=True)
class CompiledScenePlan:
    """Frozen output of ``MapScene.compile_plan()``.

    The render phase is a pure reader of this plan: label placement,
    depth-occlusion culling, and decluttering decisions are all resolved at
    compile time from serialized inputs and never mutated during render.
    """

    recipe_hash: str
    camera_terrain_key: str
    label_plans: Mapping[str, Any]
    manifest: Any
    validation_report: "ValidationReport"


def _native_scene_class() -> Any | None:
    try:
        from ._native import get_native_module

        native_module = get_native_module()
    except Exception:
        return None
    return getattr(native_module, "Scene", None)


def _is_native_adapter_unavailable(exc: BaseException) -> bool:
    exc_type = type(exc)
    return (
        exc_type.__module__ == "pyo3_runtime"
        and exc_type.__name__ == "PanicException"
        and "No suitable GPU adapter" in str(exc)
    )


def _terrain_dtype(terrain: "TerrainSource") -> Any:
    import numpy as np

    return np.dtype(terrain.dtype or "float32")


def _as_native_heightmap(data: Any, *, dtype: Any, source_label: str) -> Any:
    import numpy as np

    heightmap = np.asarray(data, dtype=dtype)
    if heightmap.ndim != 2:
        raise ValueError(f"MapScene native/offscreen terrain {source_label} input must be a 2D heightmap")
    return np.ascontiguousarray(heightmap.astype(np.float32, copy=False))


def _load_native_heightmap(terrain: "TerrainSource") -> Any | None:
    dtype = _terrain_dtype(terrain)
    if terrain.data is not None:
        return _as_native_heightmap(terrain.data, dtype=dtype, source_label="array")
    if not terrain.path:
        return None
    path = Path(str(terrain.path))
    suffix = path.suffix.lower()
    if suffix not in {".npy", ".tif", ".tiff"} or not path.exists():
        return None

    import numpy as np

    if suffix == ".npy":
        return _as_native_heightmap(np.load(path), dtype=dtype, source_label=".npy")

    from . import io

    nodata_policy = str(terrain.nodata_policy).lower()
    dem = io.load_dem(
        str(path),
        fill_nodata_values=nodata_policy != "preserve",
        dtype=dtype,
    )
    return _as_native_heightmap(dem.data, dtype=dtype, source_label="GeoTIFF")


def _is_geotiff_path(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _raster_grid_from_path(path: Path, *, target_crs: str | None = None) -> dict[str, Any] | None:
    if not _is_geotiff_path(path) or not path.exists():
        return None
    try:
        import rasterio
    except Exception:
        return None
    if getattr(rasterio, "__forge3d_stub__", False):
        return None
    try:
        with rasterio.open(path) as src:
            if src.crs is None or src.transform is None:
                return None
            if target_crs and not _same_crs(str(src.crs), str(target_crs)):
                return None
            return {
                "crs": str(src.crs),
                "transform": src.transform,
                "width": int(src.width),
                "height": int(src.height),
                "nodata": src.nodata,
            }
    except Exception:
        return None


def _terrain_alignment_grid(
    terrain: "TerrainSource",
    *,
    target_crs: str | None = None,
    fallback_shape: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    metadata = _metadata_dict(terrain.metadata)
    crs = target_crs or metadata.get("target_crs") or metadata.get("crs") or terrain.crs
    transform = metadata.get("transform", metadata.get("geotransform"))
    width = metadata.get("width")
    height = metadata.get("height")
    if (width is None or height is None) and fallback_shape is not None and len(fallback_shape) >= 2:
        height = int(fallback_shape[0])
        width = int(fallback_shape[1])
    if crs and transform is not None and width is not None and height is not None:
        return {
            "crs": str(crs),
            "transform": transform,
            "width": int(width),
            "height": int(height),
            "nodata": metadata.get("nodata"),
        }
    if terrain.path:
        path_grid = _raster_grid_from_path(Path(str(terrain.path)), target_crs=target_crs or terrain.crs)
        if path_grid is not None:
            return path_grid
    return None


def _resize_nearest_rgba(image: Any, target_shape: Sequence[int] | None) -> Any:
    if target_shape is None:
        return image

    import numpy as np

    target_h = int(target_shape[0])
    target_w = int(target_shape[1])
    if target_h <= 0 or target_w <= 0 or image.shape[:2] == (target_h, target_w):
        return image
    src_h, src_w = image.shape[:2]
    sample_y = np.clip(np.arange(target_h) * src_h // target_h, 0, src_h - 1)
    sample_x = np.clip(np.arange(target_w) * src_w // target_w, 0, src_w - 1)
    return np.ascontiguousarray(image[sample_y[:, None], sample_x[None, :]])


def _raster_to_rgba8(data: Any) -> Any:
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("MapScene raster overlay must be 2D grayscale or 3D bands")
    if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[2] not in (1, 3, 4):
        raise ValueError("MapScene raster overlay must have 1, 3, or 4 bands")

    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 255.0):
        lo = float(finite.min())
        hi = float(finite.max())
        arr = (arr - lo) * (255.0 / max(hi - lo, 1.0e-8))
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    rgba = np.empty((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    if arr.shape[2] == 1:
        rgba[..., :3] = arr[..., :1]
        rgba[..., 3] = 255
    elif arr.shape[2] == 3:
        rgba[..., :3] = arr
        rgba[..., 3] = 255
    else:
        rgba[...] = arr
    return np.ascontiguousarray(rgba)


def _load_native_raster_overlay(
    layer: "RasterOverlay",
    target_shape: Sequence[int] | None = None,
    *,
    target_grid: Mapping[str, Any] | None = None,
) -> Any | None:
    if not layer.path:
        return None
    path = Path(str(layer.path))
    suffix = path.suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"} or not path.exists():
        return None

    import numpy as np

    if suffix == ".png":
        from ._png import load_png_rgba

        overlay = np.ascontiguousarray(load_png_rgba(path).astype(np.uint8, copy=False))
    else:
        try:
            import rasterio
        except Exception as exc:
            raise ImportError(
                "rasterio is required for MapScene GeoTIFF raster overlays. "
                "Install with: pip install rasterio (or pip install 'forge3d[raster]')."
            ) from exc
        if getattr(rasterio, "__forge3d_stub__", False):
            raise ImportError(
                "rasterio is required for MapScene GeoTIFF raster overlays. "
                "Install with: pip install rasterio (or pip install 'forge3d[raster]')."
            )
        if target_grid is not None:
            from . import alignment as _alignment

            metadata = _metadata_dict(layer.metadata)
            resampling = str(metadata.get("alignment_resampling") or metadata.get("resampling") or "nearest")
            result = _alignment.resample_raster_to_grid(
                path,
                target_grid,
                resampling=resampling,
                dst_nodata=metadata.get("dst_nodata", metadata.get("nodata")),
            )
            overlay = _raster_to_rgba8(result["array"])
        else:
            with rasterio.open(path) as src:
                overlay = _raster_to_rgba8(src.read())
    return _resize_nearest_rgba(overlay, target_shape)


def _numpy_to_exr_writer() -> Any | None:
    try:
        from ._native import get_native_module

        writer = getattr(get_native_module(), "numpy_to_exr", None)
        if writer is not None:
            return writer
    except Exception:
        pass
    try:
        import forge3d._forge3d as native  # type: ignore[import-not-found]

        return getattr(native, "numpy_to_exr", None)
    except Exception:
        return None


def _write_exr_array(path: Path, array: Any, *, channel_prefix: str) -> None:
    writer = _numpy_to_exr_writer()
    if writer is None:
        raise RuntimeError("EXR output requires the native numpy_to_exr writer")
    import numpy as np

    writer(str(path), np.ascontiguousarray(array, dtype=np.float32), channel_prefix=channel_prefix)


@dataclass
class _MapSceneNativeRenderResult:
    rgba: Any
    aov_frame: Any | None = None
    hdr_frame: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # VERITAS: per-pixel source-id map + contributing-tile records captured
    # at composite time (populated only when provenance emission is on).
    source_map: Any | None = None
    contributing_tiles: list[dict[str, Any]] | None = None


def _aov_arrays_for_rgba(recipe: "SceneRecipe", rgba: Any, requested: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    rgb = np.asarray(rgba[..., :3], dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    denom_x = max(1.0, float(width - 1))
    denom_y = max(1.0, float(height - 1))
    result: dict[str, Any] = {}
    for item in requested:
        name = str(item).lower()
        if name == "albedo":
            result[name] = rgb
        elif name == "normal":
            normal = np.zeros((height, width, 3), dtype=np.float32)
            normal[..., 2] = 1.0
            result[name] = normal
        elif name == "depth":
            result[name] = (yy / denom_y).astype(np.float32)
        elif name == "uv":
            result[name] = np.dstack((xx / denom_x, yy / denom_y)).astype(np.float32)
    return result


def _resize_nearest_array(array: Any, target_shape: Sequence[int]) -> Any:
    import numpy as np

    arr = np.asarray(array)
    target_h = int(target_shape[0])
    target_w = int(target_shape[1])
    if target_h <= 0 or target_w <= 0 or arr.shape[:2] == (target_h, target_w):
        return np.ascontiguousarray(arr)
    src_h, src_w = arr.shape[:2]
    sample_y = np.clip(np.arange(target_h) * src_h // target_h, 0, src_h - 1)
    sample_x = np.clip(np.arange(target_w) * src_w // target_w, 0, src_w - 1)
    return np.ascontiguousarray(arr[sample_y[:, None], sample_x[None, :]])


def _native_aov_array(aov_frame: Any, name: str, target_shape: Sequence[int]) -> Any:
    getter = getattr(aov_frame, name, None)
    if getter is None:
        raise RuntimeError(f"Native terrain AOV frame does not expose {name!r}")
    return _resize_nearest_array(getter(), target_shape)


def _write_mapscene_aovs(
    target_path: Path,
    recipe: "SceneRecipe",
    rgba: Any,
    *,
    aov_frame: Any | None = None,
    require_native: bool = False,
) -> dict[str, str]:
    output = recipe.output
    if output is None or not output.aovs:
        return {}
    if aov_frame is None:
        if require_native:
            raise RuntimeError("MapScene native AOV export requires a renderer AovFrame")
        arrays = _aov_arrays_for_rgba(recipe, rgba, output.aovs)
    else:
        unsupported = sorted(set(output.aovs) - {"albedo", "normal", "depth"})
        if unsupported:
            raise RuntimeError(
                "MapScene native AOV export supports only albedo, normal, and depth; "
                f"unsupported: {', '.join(unsupported)}"
            )
        arrays = {
            name: _native_aov_array(aov_frame, name, rgba.shape[:2])
            for name in output.aovs
        }
    written: dict[str, str] = {}
    stem = target_path.with_suffix("")
    for name, array in arrays.items():
        path = stem.parent / f"{stem.name}_aov-{name}.exr"
        _write_exr_array(path, array, channel_prefix=name)
        written[name] = str(path)
    return written


def _deep_merge_mapping(base: Mapping[str, Any] | None, updates: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base or {}))
    for key, value in dict(updates or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _mapscene_preset_for_name(name: str) -> dict[str, Any] | None:
    if str(name).strip().lower() in {"", "default", "daylight"}:
        return None
    try:
        from . import presets

        return dict(presets.get(name))
    except Exception:
        return None


def _terrain_scene_diagonal(terrain: "TerrainSource") -> float:
    import numpy as np

    metadata = _metadata_dict(terrain.metadata)
    width = float(metadata.get("width") or metadata.get("cols") or 1.0)
    height = float(metadata.get("height") or metadata.get("rows") or 1.0)
    resolution = _metadata_resolution(metadata)
    if resolution is not None:
        return float(max(max(1.0, width) * resolution[0], max(1.0, height) * resolution[1]))
    if terrain.data is not None:
        arr = np.asarray(terrain.data)
        if arr.ndim >= 2:
            return float(max(max(1, arr.shape[1]), max(1, arr.shape[0])))
    return float(max(max(1.0, width), max(1.0, height)))


def _sun_direction_from_preset(sun: Mapping[str, Any]) -> tuple[float, float, float] | None:
    direction = sun.get("direction")
    if isinstance(direction, Sequence) and not isinstance(direction, (str, bytes)) and len(direction) == 3:
        return (float(direction[0]), float(direction[1]), float(direction[2]))
    if "azimuth_deg" not in sun or "elevation_deg" not in sun:
        return None
    azimuth = math.radians(float(sun["azimuth_deg"]))
    elevation = math.radians(float(sun["elevation_deg"]))
    return (
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.cos(azimuth),
    )


def _sun_angles_from_direction(direction: Sequence[float] | None) -> tuple[float, float]:
    if direction is None or len(direction) < 3:
        return (135.0, 35.0)
    x, y, z = (float(direction[0]), float(direction[1]), float(direction[2]))
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1.0e-8:
        return (135.0, 35.0)
    return (
        math.degrees(math.atan2(x, z)),
        math.degrees(math.asin(max(-1.0, min(1.0, y / length)))),
    )


def _heightmap_domain(heightmap: Any) -> tuple[float, float]:
    import numpy as np

    finite = np.asarray(heightmap, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(finite.min())
    hi = float(finite.max())
    if lo == hi:
        hi = lo + 1.0
    return (lo, hi)


def _write_minimal_hdr(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(b"-Y 2 +X 2\n")
        for _ in range(4):
            handle.write(bytes([180, 190, 205, 128]))


def _native_ibl_path(recipe: "SceneRecipe") -> tuple[str, bool]:
    import tempfile

    settings = _metadata_dict(recipe.lighting.settings)
    ibl = settings.get("ibl") if isinstance(settings.get("ibl"), Mapping) else {}
    for key in ("path", "hdr_path", "environment_path"):
        value = ibl.get(key) if isinstance(ibl, Mapping) else None
        if value and Path(str(value)).exists():
            return str(value), False
    builtin = ibl.get("builtin") if isinstance(ibl, Mapping) else None
    if builtin:
        from .terrain_demo import _builtin_ibl_path

        builtin_path = _builtin_ibl_path(builtin, prefer_local_assets=False)
        if builtin_path is not None:
            return str(builtin_path), False
    handle = tempfile.NamedTemporaryFile(suffix=".hdr", delete=False)
    hdr_path = Path(handle.name)
    handle.close()
    _write_minimal_hdr(hdr_path)
    return str(hdr_path), True


def _mapscene_aa_seed(recipe: "SceneRecipe") -> int | None:
    profile = recipe.reproducibility_profile
    return None if profile is None else int(profile.seed)


def _mapscene_denoise_enabled(output: "OutputSpec" | None) -> bool:
    return output is not None and str(output.denoiser).lower() not in {"", "none", "off"}


def _mapscene_aov_settings(
    output: "OutputSpec" | None,
    *,
    force_guidance: bool = False,
) -> Any | None:
    from .terrain_params import AovSettings

    requested = tuple(str(item).lower() for item in (output.aovs if output is not None else ()))
    unsupported = sorted(set(requested) - {"albedo", "normal", "depth"})
    if unsupported:
        raise ValueError(
            "MapScene native terrain AOVs support only albedo, normal, and depth; "
            f"unsupported: {', '.join(unsupported)}"
        )
    enabled = bool(requested) or bool(force_guidance)
    return AovSettings(
        enabled=enabled,
        albedo=("albedo" in requested) or force_guidance,
        normal=("normal" in requested) or force_guidance,
        depth=("depth" in requested) or force_guidance,
        format="exr",
    )


def _mapscene_denoise_settings(output: "OutputSpec" | None) -> Any | None:
    from .terrain_params import DenoiseSettings

    if not _mapscene_denoise_enabled(output):
        return DenoiseSettings(enabled=False, method="none")
    return DenoiseSettings(enabled=True, method=str(output.denoiser).lower())


def _mapscene_offline_settings(output: "OutputSpec") -> Any:
    from .terrain_params import OfflineQualitySettings

    samples = max(1, int(output.samples))
    batch = max(1, min(4, samples))
    return OfflineQualitySettings(
        enabled=True,
        adaptive=False,
        max_samples=samples,
        min_samples=min(4, samples),
        batch_size=batch,
    )


def _mapscene_shadow_settings(shadow_config: Any) -> Any:
    from .terrain_params import ShadowSettings

    settings = ShadowSettings(
        enabled=shadow_config.enabled if shadow_config else True,
        technique=shadow_config.technique.upper() if shadow_config else "PCSS",
        resolution=shadow_config.map_size if shadow_config else 4096,
        cascades=shadow_config.cascades if shadow_config else 3,
        max_distance=4000.0,
        softness=1.5,
        intensity=0.8,
        slope_scale_bias=0.001,
        depth_bias=shadow_config.moment_bias if shadow_config else 0.0005,
        normal_bias=0.0002,
        min_variance=1e-4,
        light_bleed_reduction=0.5,
        evsm_exponent=40.0,
        fade_start=1.0,
        pcss_blocker_radius=shadow_config.pcss_blocker_radius if shadow_config else 6.0,
        pcss_filter_radius=shadow_config.pcss_filter_radius if shadow_config else 4.0,
        light_size=shadow_config.light_size if shadow_config else 1.0,
    )
    settings.validate_for_terrain()
    return settings


def _mapscene_material_settings(recipe: "SceneRecipe") -> Any | None:
    from .terrain_params import MaterialLayerSettings

    metadata = _metadata_dict(recipe.terrain.metadata)
    material_data = metadata.get("material_maps")
    if material_data is None:
        material_data = metadata.get("materials")
    if not isinstance(material_data, Mapping):
        return None

    kwargs: dict[str, str] = {}
    for field_name, alias in (
        ("normal_path", "normal"),
        ("roughness_path", "roughness"),
        ("mask_path", "mask"),
    ):
        value = material_data.get(field_name)
        if value is None:
            value = material_data.get(alias)
        if value is not None:
            kwargs[field_name] = str(value)

    return MaterialLayerSettings(**kwargs) if kwargs else None


def _mapscene_water_settings(recipe: "SceneRecipe") -> Any | None:
    from .terrain_params import WaterSettings

    terrain_metadata = _metadata_dict(recipe.terrain.metadata)
    lighting_settings = _metadata_dict(recipe.lighting.settings)
    data = terrain_metadata.get("water") if isinstance(terrain_metadata.get("water"), Mapping) else None
    if data is None:
        data = lighting_settings.get("water") if isinstance(lighting_settings.get("water"), Mapping) else None
    if data is None:
        return None
    return WaterSettings(
        enabled=bool(data.get("enabled", data.get("auto_mask", data.get("mask_path") is not None))),
        auto_mask=bool(data.get("auto_mask", False)),
        mask_path=(str(data.get("mask_path")) if data.get("mask_path") is not None else None),
        level=(float(data.get("level")) if data.get("level") is not None else None),
        slope_threshold=float(data.get("slope_threshold", 0.02)),
    )


def _mapscene_water_mask(recipe: "SceneRecipe", heightmap: Any) -> Any | None:
    import numpy as np

    settings = _mapscene_water_settings(recipe)
    if settings is None or not bool(settings.enabled):
        return None
    if settings.mask_path:
        path = Path(settings.mask_path)
        if path.suffix.lower() == ".npy":
            return np.ascontiguousarray(np.load(path).astype(np.float32, copy=False))
        from ._png import load_png_rgba

        rgba = load_png_rgba(path)
        return np.ascontiguousarray((rgba[..., 0].astype(np.float32) / 255.0))
    if settings.auto_mask:
        from .gis import derive_water_mask

        return derive_water_mask(
            heightmap,
            level=settings.level,
            slope_threshold=settings.slope_threshold,
        )
    return None


def _mapscene_cloud_settings(recipe: "SceneRecipe") -> Any | None:
    from .terrain_params import CloudSettings

    data = _mapscene_cloud_config(recipe)
    if data is None:
        return None
    shadows_enabled = bool(data.get("shadows_enabled", data.get("shadow_enabled", False)))
    return CloudSettings(
        enabled=bool(data.get("enabled", shadows_enabled)),
        shadows_enabled=shadows_enabled,
        coverage=float(data.get("coverage", 0.5)),
        density=float(data.get("density", 0.5)),
        shadow_strength=float(data.get("shadow_strength", data.get("shadow_intensity", 0.35))),
        quality=str(data.get("quality", "medium")),
    )


def _mapscene_cloud_config(recipe: "SceneRecipe") -> Mapping[str, Any] | None:
    terrain_metadata = _metadata_dict(recipe.terrain.metadata)
    lighting_settings = _metadata_dict(recipe.lighting.settings)
    data = terrain_metadata.get("clouds") if isinstance(terrain_metadata.get("clouds"), Mapping) else None
    if data is None:
        data = lighting_settings.get("clouds") if isinstance(lighting_settings.get("clouds"), Mapping) else None
    if data is None:
        data = lighting_settings.get("cloud") if isinstance(lighting_settings.get("cloud"), Mapping) else None
    if data is None:
        return None
    return data


def _apply_mapscene_cloud_shadow(rgba: Any, recipe: "SceneRecipe") -> tuple[Any, dict[str, Any]]:
    settings = _mapscene_cloud_settings(recipe)
    if settings is None or not bool(settings.enabled) or not bool(settings.shadows_enabled):
        return rgba, {}

    import numpy as np

    out = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8).copy())
    height, width = out.shape[:2]
    cloud_config = _mapscene_cloud_config(recipe) or {}
    offset_x = float(cloud_config.get("shadow_offset_x", cloud_config.get("wind_offset_x", 0.0)))
    offset_y = float(cloud_config.get("shadow_offset_y", cloud_config.get("wind_offset_y", 0.0)))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scale = {"low": 2.0, "medium": 3.0, "high": 4.5, "ultra": 6.0}.get(str(settings.quality), 3.0)
    u = xx / max(1.0, float(width - 1)) + offset_x
    v = yy / max(1.0, float(height - 1)) + offset_y
    field = (
        0.55 * np.sin((u * scale + v * 0.7) * 2.0 * np.pi)
        + 0.30 * np.sin((u * 1.7 - v * scale) * 2.0 * np.pi + 0.6)
        + 0.15 * np.sin((u * 5.1 + v * 4.3) * 2.0 * np.pi + 1.7)
    )
    field = (field - field.min()) / max(float(field.max() - field.min()), 1.0e-6)
    coverage_cutoff = 1.0 - float(settings.coverage)
    cloud = np.clip((field - coverage_cutoff) / max(0.05, float(settings.density)), 0.0, 1.0)
    shadow = 1.0 - cloud * float(settings.shadow_strength)
    rgb = out[..., :3].astype(np.float32) * shadow[..., None]
    out[..., :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return out, {
        "cloud_shadow_backend": "mapscene_numpy_cloud_shadow",
        "cloud_shadow_coverage": float(settings.coverage),
        "cloud_shadow_strength": float(settings.shadow_strength),
        "cloud_shadow_quality": str(settings.quality),
        "cloud_shadow_offset": [offset_x, offset_y],
    }


def _mapscene_screen_space_settings(recipe: "SceneRecipe") -> Any | None:
    from .terrain_params import ScreenSpaceSettings

    settings = _metadata_dict(recipe.lighting.settings)
    data = settings.get("screen_space") if isinstance(settings.get("screen_space"), Mapping) else None
    if data is None:
        data = settings.get("postfx") if isinstance(settings.get("postfx"), Mapping) else None
    if data is None:
        return None

    def child(name: str) -> Mapping[str, Any]:
        value = data.get(name)
        return value if isinstance(value, Mapping) else {}

    ssao = child("ssao")
    ssgi = child("ssgi")
    ssr = child("ssr")
    taa = child("taa")
    enabled = bool(data.get("enabled", False))
    ssao_enabled = bool(ssao.get("enabled", data.get("ssao_enabled", False)))
    ssgi_enabled = bool(ssgi.get("enabled", data.get("ssgi_enabled", False)))
    ssr_enabled = bool(ssr.get("enabled", data.get("ssr_enabled", False)))
    taa_enabled = bool(taa.get("enabled", data.get("taa_enabled", False)))
    return ScreenSpaceSettings(
        enabled=enabled or ssao_enabled or ssgi_enabled or ssr_enabled or taa_enabled,
        ssao_enabled=ssao_enabled,
        ssao_radius=float(ssao.get("radius", data.get("ssao_radius", 1.5))),
        ssao_intensity=float(ssao.get("intensity", data.get("ssao_intensity", 1.0))),
        ssgi_enabled=ssgi_enabled,
        ssgi_intensity=float(ssgi.get("intensity", data.get("ssgi_intensity", 1.0))),
        ssr_enabled=ssr_enabled,
        ssr_intensity=float(ssr.get("intensity", data.get("ssr_intensity", 1.0))),
        taa_enabled=taa_enabled,
        temporal_alpha=float(taa.get("temporal_alpha", data.get("temporal_alpha", 0.1))),
    )


def _apply_mapscene_screen_space(rgba: Any, recipe: "SceneRecipe", heightmap: Any) -> tuple[Any, dict[str, Any]]:
    settings = _mapscene_screen_space_settings(recipe)
    if settings is None or not bool(settings.enabled):
        return rgba, {}

    import numpy as np

    out = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8).copy())
    rgb = out[..., :3].astype(np.float32)
    height, width = out.shape[:2]
    metadata: dict[str, Any] = {"screen_space_backend": "mapscene_numpy_postfx"}
    enabled: list[str] = []

    dem = np.asarray(heightmap, dtype=np.float32)
    if dem.ndim == 2 and dem.size > 0:
        yy = np.linspace(0, dem.shape[0] - 1, height).astype(np.int32)
        xx = np.linspace(0, dem.shape[1] - 1, width).astype(np.int32)
        sampled = dem[np.ix_(yy, xx)].astype(np.float32)
        span = max(float(sampled.max() - sampled.min()), 1.0e-6)
        height_norm = (sampled - float(sampled.min())) / span
    else:
        height_norm = np.zeros((height, width), dtype=np.float32)

    gy, gx = np.gradient(height_norm)
    slope = np.clip(np.sqrt(gx * gx + gy * gy) * max(1.0, float(settings.ssao_radius)), 0.0, 1.0)

    if bool(settings.ssao_enabled):
        occlusion = np.clip((1.0 - height_norm) * 0.55 + slope * 0.45, 0.0, 1.0)
        ao = 1.0 - occlusion * min(0.55, 0.22 * float(settings.ssao_intensity))
        rgb *= ao[..., None]
        enabled.append("ssao")
        metadata["screen_space_ssao_intensity"] = float(settings.ssao_intensity)

    if bool(settings.ssgi_enabled):
        bounce = (1.0 - slope) * height_norm
        warm = np.asarray((1.035, 1.025, 0.985), dtype=np.float32)
        rgb = rgb * (1.0 + bounce[..., None] * min(0.18, 0.06 * float(settings.ssgi_intensity)) * warm)
        enabled.append("ssgi")
        metadata["screen_space_ssgi_intensity"] = float(settings.ssgi_intensity)

    if bool(settings.ssr_enabled):
        water_mask = _mapscene_water_mask(recipe, heightmap)
        if water_mask is not None:
            mask = np.asarray(water_mask, dtype=np.float32)
            if mask.ndim == 2 and mask.size > 0:
                yy = np.linspace(0, mask.shape[0] - 1, height).astype(np.int32)
                xx = np.linspace(0, mask.shape[1] - 1, width).astype(np.int32)
                screen_mask = np.clip(mask[np.ix_(yy, xx)], 0.0, 1.0)
            else:
                screen_mask = np.zeros((height, width), dtype=np.float32)
        else:
            screen_mask = np.clip(1.0 - height_norm * 8.0, 0.0, 1.0)
        reflected = np.flip(rgb, axis=0)
        fresnel = np.linspace(0.25, 0.95, height, dtype=np.float32)[:, None]
        mix = screen_mask * fresnel * min(0.60, 0.32 * float(settings.ssr_intensity))
        rgb = rgb * (1.0 - mix[..., None]) + reflected * mix[..., None]
        enabled.append("ssr")
        metadata["screen_space_ssr_intensity"] = float(settings.ssr_intensity)

    if bool(settings.taa_enabled):
        enabled.append("taa")
        metadata["screen_space_taa_temporal_alpha"] = float(settings.temporal_alpha)

    if not enabled:
        return rgba, {}
    out[..., :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    metadata["screen_space_effects"] = enabled
    return out, metadata


def _mapscene_vt_config(recipe: "SceneRecipe") -> Mapping[str, Any] | None:
    metadata = _metadata_dict(recipe.terrain.metadata)
    config = metadata.get("virtual_texture") or metadata.get("vt")
    return config if isinstance(config, Mapping) else None


def _mapscene_clipmap_config(recipe: "SceneRecipe") -> Mapping[str, Any] | None:
    metadata = _metadata_dict(recipe.terrain.metadata)
    config = metadata.get("terrain_geometry") or metadata.get("geometry") or metadata.get("clipmap")
    if not isinstance(config, Mapping):
        return None
    mode = str(config.get("mode", "clipmap")).lower()
    return config if mode == "clipmap" or bool(config.get("enabled", False)) else None


def _mapscene_clipmap_metadata(recipe: "SceneRecipe", heightmap: Any) -> dict[str, Any]:
    config = _mapscene_clipmap_config(recipe)
    if config is None:
        return {}
    try:
        import numpy as np
        import forge3d as f3d

        arr = np.asarray(heightmap)
        ring_count = int(config.get("levels", config.get("ring_count", 4)))
        ring_resolution = int(config.get("ring_resolution", 64))
        center_resolution = int(config.get("center_resolution", ring_resolution))
        terrain_extent_m = float(config.get("terrain_extent_m", config.get("extent_m", 100_000.0)))
        clip_config = f3d.ClipmapConfig(
            ring_count=ring_count,
            ring_resolution=ring_resolution,
            center_resolution=center_resolution,
            skirt_depth=float(config.get("skirt_depth", 10.0)),
            morph_range=float(config.get("morph_range", 0.3)),
        )
        target = recipe.camera.target or (0.0, 0.0, 0.0)
        center = (float(target[0]), float(target[2] if len(target) > 2 else 0.0))
        mesh = f3d.clipmap_generate_py(clip_config, center, terrain_extent_m)
        resident_height_bytes = ring_count * ring_resolution * ring_resolution * 4
        source_bytes = int(arr.size * arr.dtype.itemsize) if arr.size else 0
        max_resident_bytes = int(config.get("max_resident_height_bytes", resident_height_bytes))
        return {
            "terrain_geometry_backend": "clipmap_indexed_pbr",
            "terrain_geometry_mode": "clipmap",
            "clipmap_ring_count": int(mesh.rings_count),
            "clipmap_ring_resolution": ring_resolution,
            "clipmap_triangle_count": int(mesh.triangle_count),
            "clipmap_vertex_count": int(mesh.vertex_count),
            "clipmap_triangle_reduction_pct": float(mesh.triangle_reduction_percent),
            "clipmap_terrain_extent_m": terrain_extent_m,
            "clipmap_resident_height_bytes": min(resident_height_bytes, max_resident_bytes),
            "clipmap_source_height_bytes": source_bytes,
            "clipmap_bounded_memory": resident_height_bytes <= max_resident_bytes,
        }
    except Exception as exc:
        return {
            "terrain_geometry_backend": "clipmap_planner_unavailable",
            "clipmap_error": f"{type(exc).__name__}: {exc}",
        }


def _mapscene_clipmap_camera_mode(config: Mapping[str, Any] | None) -> str | None:
    if not config:
        return None
    ring_count = int(config.get("ring_count", 4))
    ring_resolution = int(config.get("ring_resolution", 64))
    center_resolution = int(config.get("center_resolution", ring_resolution))
    skirt_depth = float(config.get("skirt_depth", 10.0))
    morph_range = float(config.get("morph_range", 0.3))
    return f"clipmap:{ring_count}:{ring_resolution}:{center_resolution}:{skirt_depth:g}:{morph_range:g}"


def _mapscene_vt_settings(recipe: "SceneRecipe") -> Any | None:
    config = _mapscene_vt_config(recipe)
    if config is None:
        return None

    from .terrain_params import TerrainVTSettings, VTLayerFamily

    families = config.get("families") or config.get("layers") or ("albedo",)
    default_virtual_size = config.get("virtual_size_px")
    if default_virtual_size is None:
        default_virtual_size = (
            config.get("source_size", 512)
            if bool(config.get("procedural_sources", False))
            else (4096, 4096)
        )
    layers = []
    for item in families:
        layer_data = item if isinstance(item, Mapping) else {"family": item}
        virtual_size = layer_data.get("virtual_size_px", default_virtual_size)
        if isinstance(virtual_size, Sequence) and not isinstance(virtual_size, (str, bytes)):
            size_pair = tuple(int(value) for value in list(virtual_size)[:2])
        else:
            size_pair = (int(virtual_size), int(virtual_size))
        family = str(layer_data.get("family", "albedo"))
        fallback = layer_data.get("fallback")
        layers.append(
            VTLayerFamily(
                family=family,
                virtual_size_px=size_pair,  # type: ignore[arg-type]
                tile_size=int(layer_data.get("tile_size", config.get("tile_size", 248))),
                tile_border=int(layer_data.get("tile_border", config.get("tile_border", 4))),
                fallback=(
                    tuple(float(value) for value in fallback)
                    if fallback is not None
                    else None
                ),
            )
        )
    if not layers:
        layers = [VTLayerFamily(family="albedo")]

    return TerrainVTSettings(
        enabled=bool(config.get("enabled", True)),
        layers=layers,
        atlas_size=int(config.get("atlas_size", 4096)),
        residency_budget_mb=float(config.get("residency_budget_mb", 256.0)),
        max_mip_levels=int(config.get("max_mip_levels", 8)),
        use_feedback=bool(config.get("use_feedback", True)),
    )


def _procedural_vt_source(
    size: int | tuple[int, int],
    material_index: int,
    family: str,
    pattern: str,
) -> Any:
    import numpy as np

    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        dimensions = list(size)[:2]
        if len(dimensions) != 2:
            raise ValueError("procedural VT size must contain width and height")
        width, height = (max(16, int(value)) for value in dimensions)
    else:
        width = height = max(16, int(size))
    if family == "normal":
        return np.ascontiguousarray(
            np.broadcast_to(
                np.array([128, 128, 255, 255], dtype=np.uint8),
                (height, width, 4),
            ).copy()
        )
    if family == "mask":
        return np.ascontiguousarray(
            np.full((height, width, 4), 255, dtype=np.uint8)
        )

    coords_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    coords_y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(coords_x, coords_y)
    pattern = str(pattern or "checker").lower()
    if pattern == "stripes":
        modulation = 0.45 + 0.55 * (0.5 + 0.5 * np.sin((xx * 18.0 + material_index) * np.pi))
    else:
        checker = ((np.floor(xx * 12.0) + np.floor(yy * 10.0) + material_index) % 2.0).astype(np.float32)
        wave = 0.5 + 0.5 * np.sin((xx * 9.0 + yy * 13.0 + material_index) * np.pi)
        modulation = 0.25 + 0.75 * (0.65 * checker + 0.35 * wave)
    palette = np.array(
        [
            [0.82, 0.22, 0.12],
            [0.18, 0.64, 0.24],
            [0.18, 0.34, 0.82],
            [0.86, 0.76, 0.20],
        ],
        dtype=np.float32,
    )
    base = palette[int(material_index) % len(palette)]
    rgb = np.clip(base * modulation[..., None] + 0.08 * (1.0 - modulation[..., None]), 0.0, 1.0)
    alpha = np.ones((height, width, 1), dtype=np.float32)
    return np.ascontiguousarray(np.round(np.concatenate([rgb, alpha], axis=-1) * 255.0).astype(np.uint8))


def _mapscene_register_vt_sources(renderer: Any, recipe: "SceneRecipe") -> None:
    config = _mapscene_vt_config(recipe)
    if config is None or not bool(config.get("enabled", True)):
        return
    if not hasattr(renderer, "register_material_vt_source"):
        return
    if hasattr(renderer, "clear_material_vt_sources"):
        renderer.clear_material_vt_sources()

    import numpy as np

    sources = config.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        if bool(config.get("procedural_sources", False)):
            settings = _mapscene_vt_settings(recipe)
            layers = () if settings is None else settings.layers
            configured_families = config.get("families") or config.get("layers") or ()
            explicit_fallbacks = {
                str(item.get("family", "albedo")): item["fallback"]
                for item in configured_families
                if isinstance(item, Mapping) and "fallback" in item
            }
            configured_source_size = config.get("source_size")
            if configured_source_size is not None:
                if isinstance(configured_source_size, Sequence) and not isinstance(
                    configured_source_size, (str, bytes)
                ):
                    source_dimensions = tuple(
                        int(value) for value in list(configured_source_size)[:2]
                    )
                else:
                    source_dimensions = (
                        int(configured_source_size),
                        int(configured_source_size),
                    )
                if len(source_dimensions) != 2 or any(
                    layer.virtual_size_px != source_dimensions for layer in layers
                ):
                    raise ValueError(
                        "procedural VT source_size must match every family's virtual_size_px"
                    )
            sources = [
                {
                    "material_index": int(material_index),
                    "family": layer.family,
                    "pattern": config.get("pattern", "checker"),
                    "size": layer.virtual_size_px,
                    "virtual_size_px": layer.virtual_size_px,
                    **(
                        {"fallback": explicit_fallbacks[layer.family]}
                        if layer.family in explicit_fallbacks
                        else {}
                    ),
                }
                for layer in layers
                for material_index in range(int(config.get("source_count", 4)))
            ]
        else:
            sources = ()

    for source in sources:
        if not isinstance(source, Mapping):
            continue
        material_index = int(source.get("material_index", 0))
        family = str(source.get("family", "albedo"))
        virtual_size = source.get("virtual_size_px", config.get("virtual_size_px", (512, 512)))
        if isinstance(virtual_size, Sequence) and not isinstance(virtual_size, (str, bytes)):
            size_pair = tuple(int(value) for value in list(virtual_size)[:2])
        else:
            size_pair = (int(virtual_size), int(virtual_size))
        if len(size_pair) != 2:
            size_pair = (int(size_pair[0]), int(size_pair[0]))

        path = source.get("path")
        if path is not None:
            path_obj = Path(str(path))
            if path_obj.suffix.lower() == ".npy":
                image = np.asarray(np.load(path_obj), dtype=np.uint8)
            else:
                from ._png import load_png_rgba

                image = np.asarray(load_png_rgba(path_obj), dtype=np.uint8)
        else:
            image = _procedural_vt_source(
                source.get("size", size_pair),
                material_index,
                family,
                str(source.get("pattern", "checker")),
            )

        fallback = source.get("fallback")
        if fallback is None:
            if family == "albedo":
                rgba = np.asarray(image, dtype=np.float32)
                if rgba.ndim == 3 and rgba.shape[-1] >= 3:
                    rgb = rgba[..., :3].mean(axis=(0, 1)) / 255.0
                    fallback = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
                else:
                    fallback = [0.5, 0.5, 0.5, 1.0]
            else:
                fallback = {
                    "normal": [0.5, 0.5, 1.0, 1.0],
                    "mask": [1.0, 1.0, 1.0, 1.0],
                }.get(family, [0.5, 0.5, 0.5, 1.0])
        renderer.register_material_vt_source(
            material_index,
            family,
            np.ascontiguousarray(image),
            size_pair,
            [float(value) for value in list(fallback)[:4]],
        )


def _build_mapscene_terrain_params(
    recipe: "SceneRecipe",
    heightmap: Any,
    render_size: tuple[int, int],
    *,
    emit_source_id: bool = False,
) -> Any | None:
    try:
        import forge3d as f3d
        from .config import load_renderer_config
        from .terrain_params import make_terrain_params_config
    except Exception:
        return None

    if not all(hasattr(f3d, name) for name in ("Colormap1D", "OverlayLayer", "TerrainRenderParams")):
        return None

    domain = _heightmap_domain(heightmap)
    output = recipe.output
    denoise_enabled = _mapscene_denoise_enabled(output)
    settings = _metadata_dict(recipe.lighting.settings)
    preset_name = settings.get("resolved_preset")
    if preset_name:
        from .terrain_demo import _build_colormap

        colormap = _build_colormap(domain, colormap_name=str(settings.get("colormap", "terrain")), heightmap=heightmap)
    else:
        colormap = f3d.Colormap1D.from_stops(
            stops=[
                (domain[0], "#243b2f"),
                ((domain[0] + domain[1]) * 0.5, "#8b7d4d"),
                (domain[1], "#f5f7fb"),
            ],
            domain=domain,
        )
    overlay = f3d.OverlayLayer.from_colormap1d(
        colormap,
        strength=1.0,
        offset=0.0,
        blend_mode="Alpha",
        domain=domain,
    )
    azimuth, elevation = _sun_angles_from_direction(recipe.lighting.sun_direction)
    renderer_config_data = settings.get("renderer_config") if isinstance(settings.get("renderer_config"), Mapping) else None
    renderer_config = load_renderer_config(renderer_config_data)
    ibl = settings.get("ibl") if isinstance(settings.get("ibl"), Mapping) else {}
    sun = settings.get("sun") if isinstance(settings.get("sun"), Mapping) else {}
    camera = settings.get("camera") if isinstance(settings.get("camera"), Mapping) else {}
    cli_params = settings.get("cli_params") if isinstance(settings.get("cli_params"), Mapping) else {}
    terrain_span = max(1.0, _terrain_scene_diagonal(recipe.terrain))
    clip_far = max(6000.0, terrain_span * 1.5)
    preset_albedo = "mix" if preset_name else "colormap"
    preset_colormap_strength = 0.5 if preset_name else 1.0
    camera_mode = str(cli_params.get("camera_mode") or camera.get("camera_mode") or "screen")
    if camera_mode == "screen":
        camera_mode = _mapscene_clipmap_camera_mode(_mapscene_clipmap_config(recipe)) or camera_mode
    config = make_terrain_params_config(
        size_px=render_size,
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=1,
        z_scale=float(settings.get("exaggeration") or 1.0),
        exposure=float(renderer_config.lighting.exposure),
        domain=domain,
        albedo_mode=str(settings.get("albedo_mode") or preset_albedo),
        colormap_strength=float(settings.get("colormap_strength") or preset_colormap_strength),
        ibl_enabled="ibl" in renderer_config.gi.modes,
        light_azimuth_deg=azimuth,
        light_elevation_deg=elevation,
        sun_intensity=float(recipe.lighting.intensity),
        sun_color=sun.get("color"),
        ibl_intensity=float(ibl.get("intensity", 1.0)),
        cam_radius=float(recipe.camera.distance),
        cam_phi_deg=float(recipe.camera.azimuth_deg),
        cam_theta_deg=float(recipe.camera.elevation_deg),
        fov_y_deg=float(recipe.camera.fov_deg),
        camera_mode=camera_mode,
        clip=(0.1, clip_far),
        shadows=_mapscene_shadow_settings(renderer_config.shadows),
        overlays=[overlay],
        aa_samples=max(1, int(output.samples if output is not None else 1)),
        aa_seed=_mapscene_aa_seed(recipe),
        aov=_mapscene_aov_settings(
            output,
            force_guidance=denoise_enabled,
        ),
        denoise=_mapscene_denoise_settings(output),
        screen_space=_mapscene_screen_space_settings(recipe),
        water=_mapscene_water_settings(recipe),
        clouds=_mapscene_cloud_settings(recipe),
        materials=_mapscene_material_settings(recipe),
        vt=_mapscene_vt_settings(recipe),
    )
    if emit_source_id:
        # VERITAS: capture the per-pixel VT source-id map alongside the
        # beauty pass (requires the msaa_samples=1 path used above).
        config.aov.enabled = True
        config.aov.source_id = True
    return f3d.TerrainRenderParams(config)


def _frame_to_rgba(frame: Any, output: "OutputSpec") -> Any:
    import numpy as np

    rgba = np.asarray(frame.to_numpy())
    if rgba.shape[:2] != (int(output.height), int(output.width)):
        rgba = _resize_nearest_rgba(rgba, (int(output.height), int(output.width)))
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8)
    return np.ascontiguousarray(rgba)


@lru_cache(maxsize=1)
def _terrain_renderer_runtime_available() -> bool:
    if (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and platform.system() == "Windows"
        and os.environ.get("FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN") != "1"
    ):
        return False

    try:
        import forge3d as f3d
    except Exception:
        return False

    required = ("Session", "TerrainRenderer", "MaterialSet", "IBL", "TerrainRenderParams")
    if not all(hasattr(f3d, name) for name in required):
        return False
    try:
        if hasattr(f3d, "has_gpu") and not f3d.has_gpu():
            return False
    except Exception:
        return False

    env = os.environ.copy()
    package_parent = Path(__file__).resolve().parents[1]
    repo_root = package_parent.parent
    path_entries = [str(package_parent)]
    if env.get("PYTHONPATH"):
        path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    code = (
        "import forge3d as f3d\n"
        "required=('Session','TerrainRenderer','MaterialSet','IBL','TerrainRenderParams')\n"
        "if not all(hasattr(f3d, n) for n in required): raise SystemExit(1)\n"
    )
    code += "s=f3d.Session(window=False); r=f3d.TerrainRenderer(s)\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_root if (repo_root / "pyproject.toml").exists() else package_parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _render_terrain_renderer_result(
    recipe: "SceneRecipe",
    heightmap: Any,
    *,
    emit_provenance: bool = False,
) -> _MapSceneNativeRenderResult | None:
    try:
        import forge3d as f3d
    except Exception:
        return None

    required = ("Session", "TerrainRenderer", "MaterialSet", "IBL", "TerrainRenderParams")
    if not all(hasattr(f3d, name) for name in required):
        return None
    if not _terrain_renderer_runtime_available():
        return None

    import numpy as np

    output = recipe.output
    assert output is not None
    render_size = (max(64, int(output.width)), max(64, int(output.height)))
    params = _build_mapscene_terrain_params(
        recipe, heightmap, render_size, emit_source_id=emit_provenance
    )
    if params is None:
        return None

    hdr_path, delete_hdr = _native_ibl_path(recipe)
    try:
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        _mapscene_register_vt_sources(renderer, recipe)
        material_set = f3d.MaterialSet.terrain_default()
        env_maps = f3d.IBL.from_hdr(hdr_path, intensity=1.0)
        sample_count = max(1, int(output.samples))
        output_format = str(output.format).lower()
        needs_hdr = output_format == "exr" or bool(output.hdr)
        needs_offline = sample_count > 1 or _mapscene_denoise_enabled(output) or needs_hdr
        water_mask = _mapscene_water_mask(recipe, heightmap)
        building_scatter = (
            _terrain_scatter_building_batches_for_recipe(recipe, heightmap)
            if not needs_offline and hasattr(renderer, "set_scatter_batches")
            else None
        )
        # Label occlusion is already frozen by compile_plan()'s serialized
        # camera/terrain sampler. Never retain a live GPU depth AOV merely
        # because labels request terrain occlusion; render must be a pure
        # consumer of that compiled plan.
        needs_aov = bool(output.aovs) or emit_provenance
        if emit_provenance and needs_offline:
            raise RuntimeError(
                "MapScene provenance emission requires the one-shot render path: "
                "set output.samples=1 and disable denoise/HDR output"
            )
        aov_frame = None
        hdr_frame = None
        metadata: dict[str, Any] = {
            "samples_used": 1,
            "target_samples": sample_count,
            "denoiser_used": "none",
            "adaptive": False,
        }
        metadata.update(_mapscene_clipmap_metadata(recipe, heightmap))
        if building_scatter is not None:
            scatter_batches, scatter_metadata = building_scatter
            renderer.set_scatter_batches(scatter_batches)
            metadata.update(scatter_metadata)
        if needs_offline:
            from .offline import render_offline

            t0 = time.perf_counter()
            offline_result = render_offline(
                renderer,
                material_set,
                env_maps,
                params,
                heightmap,
                settings=_mapscene_offline_settings(output),
                progress_callback=None,
                water_mask=water_mask,
            )
            metadata["offline_accumulation_ms"] = (time.perf_counter() - t0) * 1000.0
            metadata["timing_source"] = "python_perf_counter"
            frame = offline_result.frame
            hdr_frame = offline_result.hdr_frame
            aov_frame = offline_result.aov_frame
            metadata.update(dict(offline_result.metadata or {}))
        elif needs_aov and hasattr(renderer, "render_with_aov"):
            t0 = time.perf_counter()
            frame, aov_frame = renderer.render_with_aov(
                material_set=material_set,
                env_maps=env_maps,
                params=params,
                heightmap=heightmap,
                water_mask=water_mask,
            )
            metadata["terrain_main_pass_ms"] = (time.perf_counter() - t0) * 1000.0
            metadata["timing_source"] = "python_perf_counter"
        else:
            t0 = time.perf_counter()
            frame = renderer.render_terrain_pbr_pom(
                material_set=material_set,
                env_maps=env_maps,
                params=params,
                heightmap=heightmap,
                target=None,
                water_mask=water_mask,
            )
            metadata["terrain_main_pass_ms"] = (time.perf_counter() - t0) * 1000.0
            metadata["timing_source"] = "python_perf_counter"
        rgba = _frame_to_rgba(frame, output)
        source_map = None
        contributing_tiles = None
        if emit_provenance:
            # VERITAS: co-emitted at composite time — the source map and the
            # image describe the same frame. No silent fallback: a missing
            # source-id AOV here is a hard error.
            if aov_frame is None or not getattr(aov_frame, "has_source_id", False):
                raise RuntimeError(
                    "MapScene provenance emission did not produce a source-id AOV; "
                    "the native renderer must support AovSettings.source_id"
                )
            source_map = np.asarray(aov_frame.source_id(), dtype=np.uint32)
            contributing_tiles = list(renderer.read_contributing_tiles())
        if building_scatter is not None and hasattr(renderer, "get_scatter_stats"):
            metadata["building_scatter_stats"] = dict(renderer.get_scatter_stats())
        if hasattr(renderer, "get_material_vt_stats"):
            metadata["material_vt_stats"] = dict(renderer.get_material_vt_stats())
    finally:
        if delete_hdr:
            try:
                Path(hdr_path).unlink(missing_ok=True)
            except Exception:
                pass

    return _MapSceneNativeRenderResult(
        rgba=rgba,
        aov_frame=aov_frame,
        hdr_frame=hdr_frame,
        metadata=metadata,
        source_map=source_map,
        contributing_tiles=contributing_tiles,
    )


def _render_terrain_renderer_rgba(recipe: "SceneRecipe", heightmap: Any) -> Any | None:
    result = _render_terrain_renderer_result(recipe, heightmap)
    return None if result is None else result.rgba


def _rgba01(color: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    return (
        float(color[0]) / 255.0,
        float(color[1]) / 255.0,
        float(color[2]) / 255.0,
        float(color[3]) / 255.0,
    )


def _pixel_to_ndc(point: tuple[int, int], width: int, height: int) -> tuple[float, float]:
    x = -1.0 if width <= 1 else (float(point[0]) / float(width - 1)) * 2.0 - 1.0
    y = 1.0 if height <= 1 else 1.0 - (float(point[1]) / float(height - 1)) * 2.0
    return (max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)))


def _vector_layer_requires_precise_raster(layer: "VectorOverlay") -> bool:
    line_paint = _render_paint(layer, "line")
    line_layout = _render_layout(layer, "line")
    dash_array = getattr(layer, "dash_array", None) or line_paint.get("line-dasharray")
    line_join = str(line_layout.get("line-join") or getattr(layer, "line_join", "round") or "round").lower()
    if dash_array:
        return True
    for feature in layer.features or ():
        geometry = feature.get("geometry") if isinstance(feature, Mapping) else None
        if not isinstance(geometry, Mapping):
            continue
        geometry_type = str(geometry.get("type", "")).lower()
        if "polygon" not in geometry_type and (line_join != "round" or "line-miter-limit" in line_layout):
            if len(_render_geometry_points(geometry)) > 2:
                return True
    return False


def _alpha_composite_rgba(bottom: Any, top: Any) -> Any:
    import numpy as np

    dst = np.asarray(bottom, dtype=np.uint8)
    src = np.asarray(top, dtype=np.uint8)
    if src.ndim != 3 or src.shape != dst.shape or src.shape[2] != 4:
        raise RuntimeError("native vector compositor returned an invalid RGBA image")
    alpha = src[..., 3:4].astype(np.float32) / 255.0
    out = dst.copy()
    out[..., :3] = np.clip(
        dst[..., :3].astype(np.float32) * (1.0 - alpha) + src[..., :3].astype(np.float32) * alpha,
        0.0,
        255.0,
    ).astype(np.uint8)
    out[..., 3] = np.maximum(dst[..., 3], src[..., 3])
    return np.ascontiguousarray(out)


def _native_vector_payload_for_layer(layer: "VectorOverlay", width: int, height: int) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[Any]] | None:
    line_paint = _render_paint(layer, "line")
    from .style import evaluate_color_expr, evaluate_number_expr

    fallback_rgb = _render_rgb(layer.to_dict(), salt="vector")
    line_color_value = line_paint.get("line-color")
    line_color = (
        (*fallback_rgb, 255)
        if isinstance(line_color_value, list)
        else _render_color(line_color_value, (*fallback_rgb, 255))
    )
    line_opacity_value = line_paint.get("line-opacity")
    line_opacity = (
        line_color[3] / 255.0
        if isinstance(line_opacity_value, list)
        else _render_number(line_opacity_value, line_color[3] / 255.0)
    )
    line_rgba = _rgba01((line_color[0], line_color[1], line_color[2], max(0, min(255, int(round(line_opacity * 255.0))))))
    line_width_value = (
        getattr(layer, "width_px", None)
        if getattr(layer, "width_px", None) is not None
        else line_paint.get(
            "line-width",
            getattr(layer, "width_world", None)
            if getattr(layer, "width_world", None) is not None
            else 2.0,
        )
    )
    line_width = max(1.0, 2.0 if isinstance(line_width_value, list) else _render_number(line_width_value, 2.0))
    dash_array = getattr(layer, "dash_array", None) or line_paint.get("line-dasharray")

    points_xy: list[tuple[float, float]] = []
    point_rgba_values: list[tuple[float, float, float, float]] = []
    point_sizes: list[float] = []
    polylines: list[list[tuple[float, float]]] = []
    polyline_rgba: list[tuple[float, float, float, float]] = []
    stroke_width: list[float] = []

    for feature in layer.features or ():
        geometry = feature.get("geometry") if isinstance(feature, Mapping) else None
        if not isinstance(geometry, Mapping):
            continue
        properties = feature.get("properties") if isinstance(feature.get("properties"), Mapping) else {}
        feature_rgba = line_rgba
        if "line-color" in line_paint:
            evaluated = evaluate_color_expr(line_paint.get("line-color"), dict(properties))
            if evaluated is not None:
                feature_rgba = tuple(float(value) for value in evaluated)
        if "line-opacity" in line_paint:
            opacity = evaluate_number_expr(line_paint.get("line-opacity"), dict(properties))
            if opacity is not None:
                feature_rgba = (
                    feature_rgba[0],
                    feature_rgba[1],
                    feature_rgba[2],
                    max(0.0, min(1.0, feature_rgba[3] * float(opacity))),
                )
        feature_width = line_width
        if getattr(layer, "width_px", None) is None and "line-width" in line_paint:
            evaluated_width = evaluate_number_expr(line_paint.get("line-width"), dict(properties))
            if evaluated_width is not None:
                feature_width = max(1.0, float(evaluated_width))
        geometry_type = str(geometry.get("type", "")).lower()
        if "polygon" in geometry_type:
            for polygon_rings in _render_geometry_polygon_rings(geometry):
                for ring in polygon_rings:
                    points = [_render_point_to_pixel(point, width, height) for point in ring]
                    if len(points) < 2:
                        continue
                    if points[0] != points[-1]:
                        points.append(points[0])
                    polylines.append([_pixel_to_ndc(point, width, height) for point in points])
                    polyline_rgba.append(feature_rgba)
                    stroke_width.append(feature_width)
            continue
        points = [_render_point_to_pixel(point, width, height) for point in _render_geometry_points(geometry)]
        if geometry_type == "point" and points:
            points_xy.append(_pixel_to_ndc(points[0], width, height))
            point_rgba_values.append(feature_rgba)
            point_sizes.append(feature_width)
            continue
        if len(points) < 2:
            continue
        segments = _render_dash_segments(points, dash_array) if dash_array else [points]
        for segment in segments:
            if len(segment) < 2:
                continue
            polylines.append([_pixel_to_ndc(point, width, height) for point in segment])
            polyline_rgba.append(feature_rgba)
            stroke_width.append(feature_width)

    return points_xy, point_rgba_values, point_sizes, polylines, polyline_rgba, stroke_width


def _native_polygon_payload_for_layers(
    layers: Sequence["VectorOverlay"],
    width: int,
    height: int,
) -> tuple[list[Any], list[Any], list[tuple[float, float, float, float]]] | None:
    import numpy as np

    exteriors: list[Any] = []
    holes: list[Any] = []
    fill_rgba: list[tuple[float, float, float, float]] = []

    for layer in layers:
        fill_paint = _render_paint(layer, "fill")
        fallback_rgb = _render_rgb(layer.to_dict(), salt="vector")
        fill_color_value = fill_paint.get("fill-color")
        fill_color = (
            (*fallback_rgb, 255)
            if isinstance(fill_color_value, list)
            else _render_color(fill_color_value, (*fallback_rgb, 96))
        )
        fill_opacity_value = fill_paint.get("fill-opacity")
        fill_opacity = (
            fill_color[3] / 255.0
            if isinstance(fill_opacity_value, list)
            else _render_number(fill_opacity_value, fill_color[3] / 255.0)
        )
        for feature in layer.features or ():
            geometry = feature.get("geometry") if isinstance(feature, Mapping) else None
            if not isinstance(geometry, Mapping):
                continue
            geometry_type = str(geometry.get("type", "")).lower()
            if "polygon" not in geometry_type:
                continue
            properties = feature.get("properties") if isinstance(feature.get("properties"), Mapping) else {}
            feature_fill_color = _render_feature_color(fill_color_value, properties, fill_color)
            feature_fill_opacity = _render_feature_number(fill_opacity_value, properties, fill_opacity)
            feature_fill_color = (
                feature_fill_color[0],
                feature_fill_color[1],
                feature_fill_color[2],
                max(0, min(255, int(round(feature_fill_opacity * 255.0)))),
            )
            for polygon_rings in _render_geometry_polygon_rings(geometry):
                if not polygon_rings:
                    continue
                exterior = [_pixel_to_ndc(_render_point_to_pixel(point, width, height), width, height) for point in polygon_rings[0]]
                if len(exterior) < 3:
                    continue
                exteriors.append(np.asarray(exterior, dtype=np.float64))
                hole_set = []
                for ring in polygon_rings[1:]:
                    hole = [_pixel_to_ndc(_render_point_to_pixel(point, width, height), width, height) for point in ring]
                    if len(hole) >= 3:
                        hole_set.append(np.asarray(hole, dtype=np.float64))
                holes.append(hole_set)
                fill_rgba.append(_rgba01(feature_fill_color))

    if not exteriors:
        return None
    return exteriors, holes, fill_rgba


def _composite_native_vector_layers(base: Any, recipe: "SceneRecipe") -> tuple[Any, bool]:
    vector_layers = [layer for layer in recipe.layers if isinstance(layer, VectorOverlay)]
    if not vector_layers:
        return base, False

    import numpy as np

    if any(_vector_layer_requires_precise_raster(layer) for layer in vector_layers):
        rgba = _composite_recipe_layers(
            np.ascontiguousarray(base.copy()),
            recipe,
            {},
            layer_types=_render_layer_types(),
            load_raster_overlay=lambda _layer: None,
            include_raster=False,
            include_vectors=True,
            include_labels=False,
            include_buildings=False,
            include_point_cloud=False,
        )
        return np.ascontiguousarray(rgba.astype(np.uint8, copy=False)), True

    try:
        import forge3d as f3d
    except Exception:
        return base, False

    vector_render = getattr(f3d, "vector_render_oit_py", None)
    polygon_render = getattr(f3d, "vector_render_polygons_fill_py", None)

    height, width = base.shape[:2]
    composited = False

    polygon_payload = _native_polygon_payload_for_layers(vector_layers, int(width), int(height))
    if polygon_payload is not None:
        if polygon_render is None:
            return base, False
        polygon_overlay = polygon_render(
            int(width),
            int(height),
            polygon_payload[0],
            polygon_payload[1],
            fill_rgba_list=polygon_payload[2],
            coordinates_are_ndc=True,
        )
        base = _alpha_composite_rgba(base, np.asarray(polygon_overlay, dtype=np.uint8))
        composited = True

    if vector_render is None:
        return base, composited

    points_xy: list[tuple[float, float]] = []
    point_rgba: list[tuple[float, float, float, float]] = []
    point_size: list[float] = []
    polylines: list[list[tuple[float, float]]] = []
    polyline_rgba: list[tuple[float, float, float, float]] = []
    stroke_width: list[float] = []

    for layer in vector_layers:
        payload = _native_vector_payload_for_layer(layer, int(width), int(height))
        if payload is None:
            return base, False
        layer_points, layer_point_rgba, layer_point_size, layer_lines, layer_line_rgba, layer_widths = payload
        points_xy.extend(layer_points)
        point_rgba.extend(layer_point_rgba)
        point_size.extend(layer_point_size)
        polylines.extend(layer_lines)
        polyline_rgba.extend(layer_line_rgba)
        stroke_width.extend(layer_widths)

    if not points_xy and not polylines:
        return base, composited

    overlay = vector_render(
        int(width),
        int(height),
        points_xy=points_xy or None,
        point_rgba=point_rgba or None,
        point_size=point_size or None,
        polylines=polylines or None,
        polyline_rgba=polyline_rgba or None,
        stroke_width=stroke_width or None,
    )
    rgba = _alpha_composite_rgba(base, np.asarray(overlay, dtype=np.uint8))
    return rgba, True


def _pointcloud_payload_for_layer(
    layer: "PointCloudLayer",
    width: int,
    height: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]], list[float]] | None:
    metadata = _metadata_dict(layer.metadata)
    return _point_payload(
        metadata.get("positions"),
        metadata.get("colors"),
        metadata,
        width,
        height,
        path=Path(layer.path) if layer.path is not None else None,
        point_count=layer.point_count,
    )


def _edl_settings(metadata: Mapping[str, Any] | None) -> tuple[bool, float, float]:
    data = _metadata_dict(metadata)
    enabled = bool(data.get("edl")) or str(data.get("shading", "")).lower() == "edl"
    strength = max(0.0, _render_number(data.get("edl_strength"), 1.5))
    radius = max(1.0, _render_number(data.get("edl_radius_px"), 1.0))
    return enabled, strength, radius


def _point_payload(
    positions: Any,
    colors: Any,
    metadata: dict[str, Any],
    width: int,
    height: int,
    *,
    path: Path | None = None,
    point_count: int | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]], list[float]] | None:
    import numpy as np

    if positions is None:
        if path is None:
            return None
        try:
            from . import pointcloud as pointcloud_mod

            dataset = pointcloud_mod.open_pointcloud(path)
            budget = int(metadata.get("point_budget", point_count or 100_000))
            data = dataset.read_points(budget=budget)
            positions = data.positions
            colors = data.colors
            if metadata.get("bounds") is None and getattr(dataset, "bounds", None) is not None:
                bounds_obj = dataset.bounds
                metadata["bounds"] = (
                    bounds_obj.min[0],
                    bounds_obj.min[1],
                    bounds_obj.max[0],
                    bounds_obj.max[1],
                )
        except Exception:
            return None
    points = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
    if points.size == 0:
        return None
    points_xy = _project_world_xy(points, metadata, width, height)
    if points_xy is None:
        return None
    color = _rgba01(_render_color(metadata.get("color"), (255, 255, 255, 220)))
    if colors is not None:
        color_arr = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
        point_rgba = [(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, color[3]) for r, g, b in color_arr[: len(points_xy)]]
    else:
        point_rgba = [color] * len(points_xy)
    size = max(1.0, _render_number(metadata.get("point_size"), 4.0))
    return points_xy, point_rgba, [size] * len(points_xy)


def _project_world_xy(
    points: Any,
    metadata: Mapping[str, Any],
    width: int,
    height: int,
) -> list[tuple[float, float]] | None:
    """Project one shared f64 scene extent; callers must not normalize per tile."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if points.size == 0 or not np.isfinite(points).all():
        return None
    xy = points[:, :2]
    bounds = metadata.get("bounds")
    if bounds is not None and len(bounds) >= 4:
        x0, y0, x1, y1 = (float(value) for value in list(bounds)[:4])
        span = np.asarray([max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)], dtype=np.float64)
        uv = (xy - np.asarray([x0, y0], dtype=np.float64)) / span
    else:
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        uv = (xy - lo) / np.maximum(hi - lo, 1e-9)
    uv = np.clip(uv, 0.0, 1.0)
    return [_pixel_to_ndc((float(x) * (width - 1), float(y) * (height - 1)), width, height) for x, y in uv]


def _transform_tiles3d_positions(positions: Any, transform: Any) -> Any:
    """Apply a column-major 3D Tiles content transform in f64."""
    import numpy as np

    points = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
    matrix = np.asarray(transform, dtype=np.float64).reshape((4, 4), order="F")
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    transformed = homogeneous @ matrix.T
    w = transformed[:, 3:4]
    if not np.isfinite(transformed).all() or np.any(np.abs(w) <= np.finfo(np.float64).tiny):
        raise ValueError("3D Tiles transform produced a non-finite or zero homogeneous coordinate")
    return transformed[:, :3] / w


def _project_tiles3d_perspective(
    points: Any,
    metadata: Mapping[str, Any],
    width: int,
    height: int,
) -> list[tuple[float, float]] | None:
    """Perspective-project one f64 3D-Tiles scene after scene-anchor subtraction."""
    import numpy as np

    world = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if world.size == 0 or not np.isfinite(world).all():
        return None
    lo = world.min(axis=0)
    hi = world.max(axis=0)
    scene_anchor = (lo + hi) * 0.5
    span = max(float(np.max(hi - lo)), 1.0)
    target = np.asarray(metadata.get("camera_target", scene_anchor), dtype=np.float64).reshape(3)
    if "camera_position" in metadata:
        eye = np.asarray(metadata["camera_position"], dtype=np.float64).reshape(3)
    else:
        eye = scene_anchor + np.asarray([span * 1.5, span * 1.2, span * 1.5])
    forward = target - eye
    forward_length = float(np.linalg.norm(forward))
    if not np.isfinite(forward_length) or forward_length <= np.finfo(np.float64).eps:
        raise ValueError("3D Tiles camera eye and target must be distinct")
    forward /= forward_length
    up_hint = np.asarray(metadata.get("camera_up", (0.0, 1.0, 0.0)), dtype=np.float64).reshape(3)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) <= np.finfo(np.float64).eps:
        up_hint = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Subtract one f64 scene/frame anchor before any display-space narrowing.
    relative = world - eye
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ forward
    fov_y = np.deg2rad(float(metadata.get("fov_y_deg", 45.0)))
    if not 0.0 < fov_y < np.pi:
        raise ValueError("3D Tiles fov_y_deg must be between 0 and 180")
    focal = 1.0 / np.tan(fov_y * 0.5)
    aspect = max(float(width), 1.0) / max(float(height), 1.0)
    visible = camera_z > max(float(metadata.get("near", 1.0e-6)), np.finfo(np.float64).eps)
    safe_z = np.where(visible, camera_z, 1.0)
    ndc_x = camera_x * focal / (safe_z * aspect)
    ndc_y = camera_y * focal / safe_z
    return [
        (float(x), float(y)) if is_visible else (float("nan"), float("nan"))
        for x, y, is_visible in zip(ndc_x, ndc_y, visible)
    ]


def _pick_tiles3d_projected(
    points: Any,
    metadata: Mapping[str, Any],
    width: int,
    height: int,
    screen_ndc: tuple[float, float],
    *,
    tolerance_ndc: float = 0.02,
) -> int | None:
    """Pick the nearest visible 3D-Tiles point in the same shared projection."""
    import numpy as np

    projected = _project_tiles3d_perspective(points, metadata, width, height)
    if projected is None:
        return None
    xy = np.asarray(projected, dtype=np.float64)
    visible = np.isfinite(xy).all(axis=1) & (np.abs(xy[:, 0]) <= 1.0) & (np.abs(xy[:, 1]) <= 1.0)
    if not np.any(visible):
        return None
    delta = xy - np.asarray(screen_ndc, dtype=np.float64)
    distance = np.linalg.norm(delta, axis=1)
    distance[~visible] = np.inf
    index = int(np.argmin(distance))
    return index if distance[index] <= float(tolerance_ndc) else None


def _tiles3d_layer_has_native_geometry(layer: "Tiles3DLayer") -> bool:
    path = _source_path(layer.source)
    return bool(path and str(path).lower().endswith((".pnts", ".b3dm", "tileset.json")))


def _decoded_pnts_payload(
    decoded: Mapping[str, Any],
    metadata: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]], list[float]] | None:
    return _point_payload(
        decoded.get("positions"),
        decoded.get("colors"),
        metadata,
        width,
        height,
        point_count=int(decoded.get("point_count", 0)),
    )


def _tiles3d_position_source_bytes(positions: Any) -> int:
    import numpy as np

    return int(np.asarray(positions, dtype=np.float64).reshape((-1, 3)).nbytes)


def _decoded_b3dm_payload(
    decoded: Mapping[str, Any],
    metadata: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[list[tuple[float, float]]], list[tuple[float, float, float, float]], list[float]] | None:
    import numpy as np

    positions = decoded.get("positions")
    if positions is None:
        return None
    mesh_metadata = dict(metadata)
    mesh_metadata["color"] = metadata.get("mesh_color", metadata.get("color", "#f8fafc"))
    point_payload = _point_payload(positions, None, mesh_metadata, width, height)
    if point_payload is None:
        return None
    vertices = point_payload[0]
    raw_indices = decoded.get("indices")
    if raw_indices is None:
        indices = np.arange(len(vertices), dtype=np.int64)
    else:
        indices = np.asarray(raw_indices, dtype=np.int64).reshape((-1,))
    if len(indices) < 3:
        return None
    triangles = indices[: (len(indices) // 3) * 3].reshape((-1, 3))
    color = _rgba01(_render_color(mesh_metadata.get("color"), (248, 250, 252, 230)))
    width_px = max(1.0, _render_number(metadata.get("mesh_width", metadata.get("line_width", 2.0)), 2.0))
    polylines: list[list[tuple[float, float]]] = []
    for a, b, c in triangles:
        if min(a, b, c) < 0 or max(a, b, c) >= len(vertices):
            continue
        polylines.append([vertices[int(a)], vertices[int(b)], vertices[int(c)], vertices[int(a)]])
    if not polylines:
        return None
    return polylines, [color] * len(polylines), [width_px] * len(polylines)


def _tiles3d_payload_for_content(
    path: Path,
    metadata: dict[str, Any],
    width: int,
    height: int,
    tiles3d: Any,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float, float, float]],
    list[float],
    list[list[tuple[float, float]]],
    list[tuple[float, float, float, float]],
    list[float],
    bool,
    bool,
    int,
] | None:
    suffix = str(path).lower()
    if suffix.endswith(".pnts"):
        decoded = tiles3d.decode_pnts(path.read_bytes())
        payload = _decoded_pnts_payload(decoded, metadata, width, height)
        if payload is None:
            return None
        points, rgba, sizes = payload
        return points, rgba, sizes, [], [], [], True, True, _tiles3d_position_source_bytes(decoded.get("positions"))
    if suffix.endswith(".b3dm"):
        decoded = tiles3d.decode_b3dm(path.read_bytes())
        payload = _decoded_b3dm_payload(decoded, metadata, width, height)
        if payload is None:
            return None
        polylines, rgba, widths = payload
        return [], [], [], polylines, rgba, widths, False, True, _tiles3d_position_source_bytes(decoded.get("positions"))
    return None


def _tiles3d_render_payload_for_layer(
    layer: "Tiles3DLayer",
    width: int,
    height: int,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float, float, float]],
    list[float],
    list[list[tuple[float, float]]],
    list[tuple[float, float, float, float]],
    list[float],
    bool,
    bool,
    int,
] | None:
    path = _source_path(layer.source)
    if not path or not str(path).lower().endswith((".pnts", ".b3dm", "tileset.json")):
        return None
    metadata = _metadata_dict(layer.metadata)
    try:
        from . import tiles3d
    except Exception:
        return None
    if str(path).lower().endswith((".pnts", ".b3dm")):
        try:
            return _tiles3d_payload_for_content(Path(path), metadata, width, height, tiles3d)
        except Exception:
            return None

    try:
        dataset = tiles3d.Tiles3dDataset.from_tileset_json(path)
        camera = metadata.get("camera_position", (0.0, 0.0, 0.0))
        visible = dataset.traverse(
            tuple(float(value) for value in camera),
            sse_threshold=float(metadata.get("sse_threshold", 16.0)),
            max_depth=int(metadata.get("max_depth", 32)),
        )
    except Exception:
        return None

    import numpy as np

    point_chunks: list[tuple[Any, Any]] = []
    mesh_chunks: list[tuple[Any, Any]] = []
    source_bytes = 0
    for tile in visible:
        tile_path = str(tile.get("resolved_path") or tile.get("uri") or "")
        if not tile_path.lower().endswith((".pnts", ".b3dm")):
            continue
        try:
            transform = tile.get("world_transform") or np.eye(4, dtype=np.float64).reshape(16, order="F")
            raw = Path(tile_path).read_bytes()
            if tile_path.lower().endswith(".pnts"):
                decoded = tiles3d.decode_pnts(raw)
                positions = _transform_tiles3d_positions(decoded.get("positions"), transform)
                source_bytes += _tiles3d_position_source_bytes(positions)
                colors = decoded.get("colors")
                if colors is None and decoded.get("colors_rgba") is not None:
                    colors = np.asarray(decoded["colors_rgba"], dtype=np.uint8).reshape((-1, 4))[:, :3]
                point_chunks.append((positions, colors))
            else:
                decoded = tiles3d.decode_b3dm(raw)
                positions = _transform_tiles3d_positions(decoded.get("positions"), transform)
                source_bytes += _tiles3d_position_source_bytes(positions)
                indices = np.asarray(decoded.get("indices"), dtype=np.int64).reshape((-1,))
                mesh_chunks.append((positions, indices))
        except Exception:
            continue
    if not point_chunks and not mesh_chunks:
        return None

    # One scene-wide projection preserves the relative placement of every tile.
    world_chunks = [chunk[0] for chunk in point_chunks] + [chunk[0] for chunk in mesh_chunks]
    world_positions = np.concatenate(world_chunks, axis=0)
    projected = _project_tiles3d_perspective(world_positions, metadata, width, height)
    if projected is None:
        return None

    point_color_default = _rgba01(_render_color(metadata.get("color"), (255, 255, 255, 220)))
    mesh_color = _rgba01(
        _render_color(metadata.get("mesh_color", metadata.get("color")), (248, 250, 252, 230))
    )
    point_size = max(1.0, _render_number(metadata.get("point_size"), 4.0))
    mesh_width = max(1.0, _render_number(metadata.get("mesh_width", metadata.get("line_width", 2.0)), 2.0))
    all_points: list[tuple[float, float]] = []
    all_rgba: list[tuple[float, float, float, float]] = []
    all_sizes: list[float] = []
    all_lines: list[list[tuple[float, float]]] = []
    offset = 0
    for positions, colors in point_chunks:
        count = len(positions)
        chunk_projected = projected[offset:offset + count]
        visible_indices = [
            index
            for index, (x, y) in enumerate(chunk_projected)
            if np.isfinite(x) and np.isfinite(y) and abs(x) <= 1.0 and abs(y) <= 1.0
        ]
        all_points.extend(chunk_projected[index] for index in visible_indices)
        if colors is None:
            all_rgba.extend([point_color_default] * len(visible_indices))
        else:
            rgb = np.asarray(colors, dtype=np.uint8).reshape((-1, 3))
            all_rgba.extend(
                (
                    float(rgb[index, 0]) / 255.0,
                    float(rgb[index, 1]) / 255.0,
                    float(rgb[index, 2]) / 255.0,
                    point_color_default[3],
                )
                for index in visible_indices
            )
        all_sizes.extend([point_size] * len(visible_indices))
        offset += count
    for positions, indices in mesh_chunks:
        count = len(positions)
        vertices = projected[offset:offset + count]
        offset += count
        for triangle in indices[: (len(indices) // 3) * 3].reshape((-1, 3)):
            a, b, c = (int(value) for value in triangle)
            if min(a, b, c) < 0 or max(a, b, c) >= count:
                continue
            if not all(
                np.isfinite(vertices[index][0])
                and np.isfinite(vertices[index][1])
                and abs(vertices[index][0]) <= 1.0
                and abs(vertices[index][1]) <= 1.0
                for index in (a, b, c)
            ):
                continue
            all_lines.append([vertices[a], vertices[b], vertices[c], vertices[a]])
    return (
        all_points,
        all_rgba,
        all_sizes,
        all_lines,
        [mesh_color] * len(all_lines),
        [mesh_width] * len(all_lines),
        bool(point_chunks),
        True,
        source_bytes,
    )


def _composite_native_point_cloud_layers(base: Any, recipe: "SceneRecipe") -> tuple[Any, bool, dict[str, Any]]:
    layers = [layer for layer in recipe.layers if isinstance(layer, (PointCloudLayer, Tiles3DLayer))]
    if not layers:
        return base, False, {}
    try:
        import forge3d as f3d
        import numpy as np
    except Exception:
        return base, False, {}
    edl_requested = False
    edl_strength = 1.5
    edl_radius = 1.0
    for layer in layers:
        enabled, strength, radius = _edl_settings(getattr(layer, "metadata", None))
        if enabled:
            edl_requested = True
            edl_strength = strength
            edl_radius = radius
            break
    vector_render = getattr(f3d, "vector_render_oit_edl_py", None) if edl_requested else getattr(f3d, "vector_render_oit_py", None)
    if vector_render is None:
        return base, False, {}
    height, width = base.shape[:2]
    points_xy: list[tuple[float, float]] = []
    point_rgba: list[tuple[float, float, float, float]] = []
    point_size: list[float] = []
    polylines: list[list[tuple[float, float]]] = []
    polyline_rgba: list[tuple[float, float, float, float]] = []
    stroke_width: list[float] = []
    has_point_clouds = False
    has_tiles = False
    tiles3d_source_bytes = 0
    for layer in layers:
        if isinstance(layer, PointCloudLayer):
            payload = _pointcloud_payload_for_layer(layer, int(width), int(height))
            if payload is not None:
                layer_points, layer_rgba, layer_sizes = payload
                points_xy.extend(layer_points)
                point_rgba.extend(layer_rgba)
                point_size.extend(layer_sizes)
                has_point_clouds = True
        else:
            tiles_payload = _tiles3d_render_payload_for_layer(layer, int(width), int(height))
            if tiles_payload is None:
                continue
            (
                layer_points,
                layer_rgba,
                layer_sizes,
                layer_lines,
                layer_line_rgba,
                layer_widths,
                payload_has_points,
                payload_has_tiles,
                payload_source_bytes,
            ) = tiles_payload
            points_xy.extend(layer_points)
            point_rgba.extend(layer_rgba)
            point_size.extend(layer_sizes)
            polylines.extend(layer_lines)
            polyline_rgba.extend(layer_line_rgba)
            stroke_width.extend(layer_widths)
            has_point_clouds = has_point_clouds or payload_has_points
            has_tiles = has_tiles or payload_has_tiles
            tiles3d_source_bytes += int(payload_source_bytes)
    if not points_xy and not polylines:
        return base, False, {}
    render_kwargs = {
        "points_xy": points_xy or None,
        "point_rgba": point_rgba or None,
        "point_size": point_size or None,
        "polylines": polylines or None,
        "polyline_rgba": polyline_rgba or None,
        "stroke_width": stroke_width or None,
    }
    if edl_requested:
        render_kwargs["edl_strength"] = edl_strength
        render_kwargs["edl_radius_px"] = edl_radius
    overlay = vector_render(int(width), int(height), **render_kwargs)
    metadata: dict[str, Any] = {}
    if has_point_clouds:
        metadata["point_cloud_backend"] = "native_oit_points"
    if has_tiles:
        metadata["tiles3d_backend"] = "native_oit_geometry"
        metadata["tiles3d_source_bytes"] = tiles3d_source_bytes
    if edl_requested:
        metadata["point_cloud_edl_backend"] = "weighted_oit_depth_edl"
    return _alpha_composite_rgba(base, np.asarray(overlay, dtype=np.uint8)), True, metadata


def _composite_native_label_layers(base: Any, recipe: "SceneRecipe", plans: Mapping[str, Any]) -> tuple[Any, bool]:
    label_layers = [layer for layer in recipe.layers if isinstance(layer, LabelLayer)]
    if not label_layers:
        return base, False
    if not any(plans.get(_layer_id(layer, "layer")) and plans[_layer_id(layer, "layer")].accepted for layer in label_layers):
        return base, False

    scene_cls = _native_scene_class()
    if scene_cls is None:
        return base, False

    import numpy as np

    from ._png import load_png_rgba
    from .text_atlas import default_latin_atlas_paths, load_atlas_metrics

    height, width = base.shape[:2]
    required = (
        "set_raster_overlay",
        "set_native_text_atlas",
        "enable_native_text",
        "add_native_text_rect_uv_halo",
        "render_rgba",
    )
    rendered_any = False
    composited = np.ascontiguousarray(base, dtype=np.uint8)
    for layer in label_layers:
        layer_id = _layer_id(layer, "layer")
        plan = plans.get(layer_id)
        if plan is None or not plan.accepted:
            continue

        atlas_png, atlas_json = default_latin_atlas_paths()
        atlas_payload = _metadata_dict(layer.glyph_atlas)
        image_path = atlas_payload.get("image_path")
        metrics_path = atlas_payload.get("metrics_path") or atlas_payload.get("source_path")
        custom_atlas_bound = bool(
            image_path
            and metrics_path
            and Path(str(image_path)).exists()
            and Path(str(metrics_path)).exists()
        )
        if custom_atlas_bound:
            atlas_png, atlas_json = Path(str(image_path)), Path(str(metrics_path))

        atlas = load_png_rgba(atlas_png)
        metrics = load_atlas_metrics(atlas_json)
        glyphs_by_id = metrics.get("glyphs_by_id")
        if not isinstance(glyphs_by_id, Mapping) or not glyphs_by_id:
            raise MapSceneTextLayoutError({
                "status": "diagnostic_block",
                "reason": "shaped_atlas_required",
                "layer": layer_id,
                "object_id": None,
                "required_metadata": "glyphs_by_id",
            })
        atlas_h, atlas_w = atlas.shape[:2]
        atlas_font_size = float(metrics.get("font_size", 1.0))
        if not math.isfinite(atlas_font_size) or atlas_font_size <= 0.0:
            raise MapSceneTextLayoutError({
                "status": "diagnostic_block",
                "reason": "invalid_atlas_font_size",
                "layer": layer_id,
                "object_id": None,
            })
        atlas_hashes = tuple(str(value).lower() for value in metrics.get("font_sha256", ()))
        if not atlas_hashes:
            raise MapSceneTextLayoutError({
                "status": "diagnostic_block",
                "reason": "missing_atlas_font_identity",
                "layer": layer_id,
                "object_id": None,
                "required_metadata": "font_sha256",
            })

        native_scene = scene_cls(int(width), int(height))
        if not all(hasattr(native_scene, name) for name in required):
            return composited, rendered_any
        if hasattr(native_scene, "disable_terrain"):
            native_scene.disable_terrain()
        native_scene.set_raster_overlay(composited, 1.0, None, None)
        native_scene.set_native_text_atlas(atlas, int(metrics.get("channels", 1)), 1.0)
        native_scene.enable_native_text()

        pending_text_rects: list[tuple[float, ...]] = []
        for accepted in plan.accepted:
            typography = dict(getattr(accepted, "typography", None) or {})
            shaped_hashes = tuple(
                str(value).lower() for value in typography.get("font_sha256", ())
            )
            if shaped_hashes != atlas_hashes:
                raise MapSceneTextLayoutError({
                    "status": "diagnostic_block",
                    "reason": "atlas_font_mismatch",
                    "layer": layer_id,
                    "object_id": accepted.label_id,
                    "expected_sha256": list(shaped_hashes),
                    "atlas_sha256": list(atlas_hashes),
                })
            text_color = _rgba01(
                _render_color(typography.get("color") or typography.get("text_color"), (255, 255, 255, 255))
            )
            halo_color = _rgba01(
                _render_color(
                    typography.get("halo_color") or typography.get("text_halo_color"),
                    (0, 0, 0, 190),
                )
            )
            halo_width = _render_number(
                typography.get("halo_width_px")
                if "halo_width_px" in typography
                else typography.get("halo_width", typography.get("text_halo_width")),
                1.0,
            )
            anchor_x, anchor_y = _render_label_anchor(accepted, int(width), int(height))
            # The packaged atlas bake resolution is an implementation detail,
            # not the public default label size. Keep MapScene's default at
            # 12 px there. An explicitly bound custom atlas retains its own
            # declared size unless typography overrides it.
            render_size = _render_number(
                typography.get("font_size", typography.get("size", typography.get("text_size"))),
                atlas_font_size if custom_atlas_bound else 12.0,
            )
            if not math.isfinite(render_size) or render_size <= 0.0:
                raise MapSceneTextLayoutError({
                    "status": "diagnostic_block",
                    "reason": "invalid_text_size",
                    "layer": layer_id,
                    "object_id": accepted.label_id,
                    "font_size": render_size,
                })
            atlas_scale = render_size / atlas_font_size
            positioned = tuple(getattr(accepted, "positioned_glyphs", ()) or ())
            if not positioned:
                raise MapSceneTextLayoutError({
                    "status": "diagnostic_block",
                    "reason": "missing_positioned_glyphs",
                    "layer": layer_id,
                    "object_id": accepted.label_id,
                    "required_mapping": "font_index:glyph_id",
                })
            for positioned_glyph in positioned:
                font_index = int(positioned_glyph["font_index"])
                glyph_id = int(positioned_glyph["glyph_id"])
                identity = f"{font_index}:{glyph_id}"
                glyph = glyphs_by_id.get(identity)
                if glyph is None:
                    raise MapSceneTextLayoutError({
                        "status": "diagnostic_block",
                        "reason": "missing_shaped_glyph",
                        "layer": layer_id,
                        "object_id": accepted.label_id,
                        "font_index": font_index,
                        "glyph_id": glyph_id,
                        "identity": identity,
                    })
                if not bool(positioned_glyph.get("has_outline", True)):
                    continue
                origin = positioned_glyph.get("origin")
                if not isinstance(origin, Sequence) or len(origin) < 2:
                    raise MapSceneTextLayoutError({
                        "status": "diagnostic_block",
                        "reason": "invalid_positioned_glyph",
                        "layer": layer_id,
                        "object_id": accepted.label_id,
                        "identity": identity,
                    })
                try:
                    glyph_x = float(glyph["x"])
                    glyph_y = float(glyph["y"])
                    glyph_w = float(glyph["w"])
                    glyph_h = float(glyph["h"])
                    x = float(anchor_x) + float(origin[0]) * render_size + float(glyph["ox"]) * atlas_scale
                    y = float(anchor_y) + float(origin[1]) * render_size + float(glyph["oy"]) * atlas_scale
                    w = glyph_w * atlas_scale
                    h = glyph_h * atlas_scale
                    u0 = glyph_x / float(atlas_w)
                    v0 = glyph_y / float(atlas_h)
                    u1 = (glyph_x + glyph_w) / float(atlas_w)
                    v1 = (glyph_y + glyph_h) / float(atlas_h)
                    halo_width_px = float(halo_width)
                    rect = (
                        x,
                        y,
                        w,
                        h,
                        u0,
                        v0,
                        u1,
                        v1,
                        *text_color,
                        *halo_color,
                        halo_width_px,
                    )
                    invalid_rect = (
                        not all(math.isfinite(value) for value in rect)
                        or w <= 0.0
                        or h <= 0.0
                        or not (0.0 <= u0 < u1 <= 1.0)
                        or not (0.0 <= v0 < v1 <= 1.0)
                        or halo_width_px < 0.0
                    )
                except (TypeError, ValueError, KeyError):
                    invalid_rect = True
                    rect = ()
                if invalid_rect:
                    raise MapSceneTextLayoutError({
                        "status": "diagnostic_block",
                        "reason": "invalid_text_rect",
                        "layer": layer_id,
                        "object_id": accepted.label_id,
                        "identity": identity,
                    })
                pending_text_rects.append(
                    rect
                )

        if not pending_text_rects:
            raise MapSceneTextLayoutError({
                "status": "diagnostic_block",
                "reason": "empty_positioned_render_stream",
                "layer": layer_id,
                "object_id": None,
            })
        for rect in pending_text_rects:
            native_scene.add_native_text_rect_uv_halo(*rect)
        rgba = np.asarray(native_scene.render_rgba())
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError("MapScene native text compositor returned an invalid RGBA image")
        composited = np.ascontiguousarray(rgba.astype(np.uint8, copy=False))
        rendered_any = True

    return composited, rendered_any


def _needs_native_vector_composite(recipe: "SceneRecipe") -> bool:
    return any(isinstance(layer, VectorOverlay) and bool(layer.features or layer.path) for layer in recipe.layers)


def _needs_native_label_composite(recipe: "SceneRecipe", plans: Mapping[str, Any]) -> bool:
    return any(
        isinstance(layer, LabelLayer)
        and bool(plans.get(_layer_id(layer, "layer")))
        and bool(plans[_layer_id(layer, "layer")].accepted)
        for layer in recipe.layers
    )


def _building_scene_bounds(layers: Sequence["BuildingLayer"]) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for layer in layers:
        for feature in _render_building_features(layer):
            geometry = feature.get("geometry") if isinstance(feature.get("geometry"), Mapping) else {}
            for ring in _render_building_rings(geometry):
                for point in ring:
                    if isinstance(point, Sequence) and len(point) >= 2:
                        xs.append(float(point[0]))
                        ys.append(float(point[1]))
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0
    return (min_x, min_y, max_x, max_y)


def _building_point_to_scene(point: Sequence[Any], bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bounds
    x = (float(point[0]) - min_x) / max(max_x - min_x, 1.0e-9)
    y = (float(point[1]) - min_y) / max(max_y - min_y, 1.0e-9)
    return (x * 1.7 - 0.85, (1.0 - y) * 1.7 - 0.85)


def _append_roof_triangle(
    positions: list[list[float]],
    normals: list[list[float]],
    indices: list[list[int]],
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> None:
    import numpy as np

    pa = np.asarray(a, dtype=np.float32)
    pb = np.asarray(b, dtype=np.float32)
    pc = np.asarray(c, dtype=np.float32)
    normal = np.cross(pb - pa, pc - pa)
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-8:
        normal = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        normal = normal / length
        if normal[1] < 0.0:
            pa, pc = pc, pa
            normal = -normal
    start = len(positions)
    positions.extend([pa.tolist(), pb.tolist(), pc.tolist()])
    normals.extend([normal.tolist(), normal.tolist(), normal.tolist()])
    indices.append([start, start + 1, start + 2])


def _append_roof_geometry(
    positions: list[list[float]],
    normals: list[list[float]],
    indices: list[list[int]],
    footprint: Sequence[Sequence[float]],
    wall_height: float,
    roof_type: str,
) -> None:
    if roof_type == "flat" or len(footprint) < 3:
        return
    xs = [float(point[0]) for point in footprint]
    zs = [float(point[1]) for point in footprint]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    cx = (min_x + max_x) * 0.5
    cz = (min_z + max_z) * 0.5
    roof_h = max(0.05, wall_height * 0.25)
    y0 = wall_height
    y1 = wall_height + roof_h
    corners = [
        [min_x, y0, min_z],
        [max_x, y0, min_z],
        [max_x, y0, max_z],
        [min_x, y0, max_z],
    ]
    if roof_type == "pyramidal":
        apex = [cx, y1, cz]
        for left, right in zip(corners, [*corners[1:], corners[0]]):
            _append_roof_triangle(positions, normals, indices, left, right, apex)
    elif roof_type == "hipped":
        if (max_x - min_x) >= (max_z - min_z):
            ridge = [[min_x * 0.7 + max_x * 0.3, y1, cz], [min_x * 0.3 + max_x * 0.7, y1, cz]]
        else:
            ridge = [[cx, y1, min_z * 0.7 + max_z * 0.3], [cx, y1, min_z * 0.3 + max_z * 0.7]]
        _append_roof_triangle(positions, normals, indices, corners[0], corners[1], ridge[0])
        _append_roof_triangle(positions, normals, indices, corners[1], corners[2], ridge[1])
        _append_roof_triangle(positions, normals, indices, corners[2], corners[3], ridge[1])
        _append_roof_triangle(positions, normals, indices, corners[3], corners[0], ridge[0])
        _append_roof_triangle(positions, normals, indices, ridge[0], corners[1], ridge[1])
        _append_roof_triangle(positions, normals, indices, ridge[0], ridge[1], corners[3])
    elif roof_type == "gabled":
        if (max_x - min_x) >= (max_z - min_z):
            ridge = [[min_x, y1, cz], [max_x, y1, cz]]
            _append_roof_triangle(positions, normals, indices, corners[0], corners[1], ridge[1])
            _append_roof_triangle(positions, normals, indices, corners[0], ridge[1], ridge[0])
            _append_roof_triangle(positions, normals, indices, corners[3], ridge[0], ridge[1])
            _append_roof_triangle(positions, normals, indices, corners[3], ridge[1], corners[2])
            _append_roof_triangle(positions, normals, indices, corners[0], ridge[0], corners[3])
            _append_roof_triangle(positions, normals, indices, corners[1], corners[2], ridge[1])
        else:
            ridge = [[cx, y1, min_z], [cx, y1, max_z]]
            _append_roof_triangle(positions, normals, indices, corners[0], ridge[0], ridge[1])
            _append_roof_triangle(positions, normals, indices, corners[0], ridge[1], corners[3])
            _append_roof_triangle(positions, normals, indices, corners[1], corners[2], ridge[1])
            _append_roof_triangle(positions, normals, indices, corners[1], ridge[1], ridge[0])
            _append_roof_triangle(positions, normals, indices, corners[0], corners[1], ridge[0])
            _append_roof_triangle(positions, normals, indices, corners[3], ridge[1], corners[2])


def _native_building_mesh_batches_for_layers(
    layers: Sequence["BuildingLayer"],
) -> list[dict[str, Any]] | None:
    bounds = _building_scene_bounds(layers)
    if bounds is None:
        return None
    try:
        import numpy as np
        from .geometry import extrude_polygon
    except Exception:
        return None

    batches: list[dict[str, Any]] = []
    feature_counter = 0
    for layer in layers:
        for feature in _render_building_features(layer):
            feature_counter += 1
            geometry = feature.get("geometry") if isinstance(feature.get("geometry"), Mapping) else {}
            properties = _render_building_properties(feature)
            if not properties and isinstance(feature.get("properties"), Mapping):
                properties = feature["properties"]  # type: ignore[assignment]
            fill = _render_building_fill_color(properties)
            color = (fill[0] / 255.0, fill[1] / 255.0, fill[2] / 255.0, fill[3] / 255.0)
            wall_height = max(0.08, min(1.4, _render_building_height(properties) / 45.0))
            roof_type = _render_building_roof_type(properties)
            feature_id = str(
                feature.get("id")
                or properties.get("id")
                or properties.get("source_id")
                or f"{getattr(layer, 'layer_id', 'buildings')}-{feature_counter}"
            )
            all_positions: list[list[float]] = []
            all_normals: list[list[float]] = []
            all_indices: list[list[int]] = []
            for ring in _render_building_rings(geometry):
                if len(ring) < 3:
                    continue
                footprint = [_building_point_to_scene(point, bounds) for point in ring]
                if len(footprint) >= 2 and footprint[0] == footprint[-1]:
                    footprint = footprint[:-1]
                if len(footprint) < 3:
                    continue
                try:
                    mesh = extrude_polygon(np.asarray(footprint, dtype=np.float32), wall_height)
                except Exception:
                    continue
                offset = len(all_positions)
                all_positions.extend(np.asarray(mesh.positions, dtype=np.float32).tolist())
                normals = np.asarray(mesh.normals, dtype=np.float32)
                if normals.shape[:1] != (mesh.positions.shape[0],):
                    normals = np.zeros_like(mesh.positions, dtype=np.float32)
                    normals[:, 1] = 1.0
                all_normals.extend(normals.tolist())
                all_indices.extend((np.asarray(mesh.indices, dtype=np.uint32).reshape(-1, 3) + offset).astype(int).tolist())
                _append_roof_geometry(all_positions, all_normals, all_indices, footprint, wall_height, roof_type)
            if all_positions and all_indices:
                batches.append(
                    {
                        "feature_id": feature_id,
                        "roof_type": roof_type,
                        "wall_height": float(wall_height),
                        "positions": np.ascontiguousarray(all_positions, dtype=np.float32),
                        "indices": np.ascontiguousarray(all_indices, dtype=np.uint32),
                        "normals": np.ascontiguousarray(all_normals, dtype=np.float32),
                        "color": color,
                    }
                )

    return batches or None


def _terrain_scatter_building_batches_for_recipe(
    recipe: "SceneRecipe",
    heightmap: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    building_layers = [layer for layer in recipe.layers if isinstance(layer, BuildingLayer)]
    if not building_layers:
        return None
    batches = _native_building_mesh_batches_for_layers(building_layers)
    if not batches:
        return None

    try:
        import numpy as np
        from .geometry import MeshBuffers
        from .terrain_scatter import (
            TerrainContactSettings,
            TerrainMeshBlendSettings,
            TerrainScatterBatch,
            TerrainScatterLevel,
            TerrainScatterSource,
            make_transform_row_major,
        )
    except Exception:
        return None

    arr = np.asarray(heightmap, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return None
    terrain_width = float(max(arr.shape))
    settings = _metadata_dict(recipe.lighting.settings)
    z_scale = float(settings.get("exaggeration") or 1.0)
    source = TerrainScatterSource(arr, z_scale=z_scale, terrain_width=terrain_width)
    scene_to_contract = terrain_width / 1.7
    scatter_batches: list[dict[str, Any]] = []
    batch_ids: dict[str, int] = {}
    roof_types: dict[str, str] = {}
    contact_distance = max(0.25, terrain_width * 0.015)

    for batch in batches:
        positions = np.asarray(batch["positions"], dtype=np.float32)
        indices = np.asarray(batch["indices"], dtype=np.uint32).reshape(-1, 3)
        normals = np.asarray(batch["normals"], dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            continue
        contract = np.empty_like(positions, dtype=np.float32)
        contract[:, 0] = (positions[:, 0] + 0.85) * scene_to_contract
        contract[:, 1] = positions[:, 1] * scene_to_contract
        contract[:, 2] = (positions[:, 2] + 0.85) * scene_to_contract
        center_x = float((np.min(contract[:, 0]) + np.max(contract[:, 0])) * 0.5)
        center_z = float((np.min(contract[:, 2]) + np.max(contract[:, 2])) * 0.5)
        local = contract.copy()
        local[:, 0] -= center_x
        local[:, 2] -= center_z
        row, col = source.contract_to_pixel(center_x, center_z)
        base_y = source.sample_scaled_height(row, col)
        mesh = MeshBuffers(
            positions=np.ascontiguousarray(local, dtype=np.float32),
            normals=np.ascontiguousarray(normals, dtype=np.float32),
            uvs=np.empty((0, 2), dtype=np.float32),
            indices=np.ascontiguousarray(indices, dtype=np.uint32),
        )
        scatter = TerrainScatterBatch(
            name=f"building:{batch['feature_id']}",
            color=tuple(float(value) for value in batch["color"]),
            transforms=np.asarray(
                [make_transform_row_major((center_x, base_y, center_z))],
                dtype=np.float32,
            ),
            terrain_blend=TerrainMeshBlendSettings(enabled=False),
            terrain_contact=TerrainContactSettings(
                enabled=True,
                distance=contact_distance,
                strength=0.24,
                vertical_weight=0.85,
            ),
            levels=[TerrainScatterLevel(mesh=mesh)],
        )
        feature_id = str(batch["feature_id"])
        batch_ids[feature_id] = len(scatter_batches)
        roof_types[feature_id] = str(batch["roof_type"])
        scatter_batches.append(scatter.to_native_dict())

    if not scatter_batches:
        return None
    return scatter_batches, {
        "building_backend": "terrain_scatter_instanced_mesh",
        "building_batch_count": len(scatter_batches),
        "building_batch_ids": batch_ids,
        "building_roof_types": roof_types,
        "building_shadow_model": "terrain_csm_mesh_cast_receive",
    }


def _native_building_mesh_for_layers(
    layers: Sequence["BuildingLayer"],
) -> tuple[Any, Any, Any, tuple[float, float, float, float]] | None:
    batches = _native_building_mesh_batches_for_layers(layers)
    if not batches:
        return None

    import numpy as np

    all_positions: list[Any] = []
    all_normals: list[Any] = []
    all_indices: list[Any] = []
    color = tuple(float(value) for value in batches[-1]["color"])
    for batch in batches:
        offset = sum(len(np.asarray(item)) for item in all_positions)
        positions = np.asarray(batch["positions"], dtype=np.float32)
        all_positions.append(positions)
        all_normals.append(np.asarray(batch["normals"], dtype=np.float32))
        all_indices.append(np.asarray(batch["indices"], dtype=np.uint32) + offset)
    return (
        np.ascontiguousarray(np.vstack(all_positions), dtype=np.float32),
        np.ascontiguousarray(np.vstack(all_indices), dtype=np.uint32),
        np.ascontiguousarray(np.vstack(all_normals), dtype=np.float32),
        color,  # type: ignore[return-value]
    )


def _native_building_projected_shadow_mesh(
    batches: Sequence[Mapping[str, Any]],
    light_dir: Sequence[float] | None,
) -> tuple[Any, Any, Any] | None:
    try:
        import numpy as np
    except Exception:
        return None
    light_values = list(light_dir or (0.3, 0.7, 0.2))
    while len(light_values) < 3:
        light_values.append(0.0)
    lx, ly, lz = (float(light_values[0]), float(light_values[1]), float(light_values[2]))
    y_denom = max(abs(ly), 0.25)
    positions_out: list[list[float]] = []
    normals_out: list[list[float]] = []
    indices_out: list[list[int]] = []
    for batch in batches:
        positions = np.asarray(batch.get("positions"), dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            continue
        min_x = float(np.min(positions[:, 0]))
        max_x = float(np.max(positions[:, 0]))
        min_z = float(np.min(positions[:, 2]))
        max_z = float(np.max(positions[:, 2]))
        max_y = max(0.0, float(np.max(positions[:, 1])))
        if max_x <= min_x or max_z <= min_z or max_y <= 0.0:
            continue
        shift_x = float(np.clip((lx / y_denom) * max_y * 0.55, -0.55, 0.55))
        shift_z = float(np.clip((lz / y_denom) * max_y * 0.55, -0.55, 0.55))
        shadow_min_x = min(min_x, min_x + shift_x)
        shadow_max_x = max(max_x, max_x + shift_x)
        shadow_min_z = min(min_z, min_z + shift_z)
        shadow_max_z = max(max_z, max_z + shift_z)
        y = 0.012
        quad = [
            [shadow_min_x, y, shadow_min_z],
            [shadow_max_x, y, shadow_min_z],
            [shadow_max_x, y, shadow_max_z],
            [shadow_min_x, y, shadow_max_z],
        ]
        offset = len(positions_out)
        positions_out.extend(quad)
        normals_out.extend([[0.0, 1.0, 0.0]] * 4)
        indices_out.extend([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]])
    if not positions_out:
        return None
    return (
        np.ascontiguousarray(positions_out, dtype=np.float32),
        np.ascontiguousarray(indices_out, dtype=np.uint32),
        np.ascontiguousarray(normals_out, dtype=np.float32),
    )


def _composite_native_building_layers(base: Any, recipe: "SceneRecipe") -> tuple[Any, bool, dict[str, Any]]:
    building_layers = [layer for layer in recipe.layers if isinstance(layer, BuildingLayer)]
    if not building_layers:
        return base, False, {}
    scene_cls = _native_scene_class()
    if scene_cls is None:
        return base, False, {}
    batches = _native_building_mesh_batches_for_layers(building_layers)
    if not batches:
        return base, False, {}

    import numpy as np

    height, width = base.shape[:2]
    native_scene = scene_cls(int(width), int(height))
    required = ("add_instanced_mesh", "render_rgba")
    if not all(hasattr(native_scene, name) for name in required):
        return base, False, {}
    if hasattr(native_scene, "disable_terrain"):
        native_scene.disable_terrain()
    if hasattr(native_scene, "set_camera_look_at"):
        _apply_native_camera(native_scene, recipe.camera, distance_override=3.2, target_override=(0.0, 0.35, 0.0))
    transforms = np.eye(4, dtype=np.float32).reshape(1, 16)
    light_dir = recipe.lighting.sun_direction or (0.3, 0.7, 0.2)
    batch_ids: dict[str, int] = {}
    roof_types: dict[str, str] = {}
    for batch in batches:
        batch_index = native_scene.add_instanced_mesh(
            batch["positions"],
            batch["indices"],
            transforms,
            normals=batch["normals"],
            color=tuple(float(value) for value in batch["color"]),
            light_dir=tuple(float(value) for value in light_dir),
            light_intensity=float(recipe.lighting.intensity),
        )
        feature_id = str(batch["feature_id"])
        batch_ids[feature_id] = int(batch_index)
        roof_types[feature_id] = str(batch["roof_type"])
    mesh_rgba = np.asarray(native_scene.render_rgba())
    if mesh_rgba.ndim != 3 or mesh_rgba.shape[2] != 4:
        raise RuntimeError("MapScene native building compositor returned an invalid RGBA image")

    out = np.ascontiguousarray(base, dtype=np.uint8).copy()
    shadow_mesh = _native_building_projected_shadow_mesh(batches, light_dir)
    if shadow_mesh is not None:
        shadow_positions, shadow_indices, shadow_normals = shadow_mesh
        shadow_scene = scene_cls(int(width), int(height))
        if hasattr(shadow_scene, "disable_terrain"):
            shadow_scene.disable_terrain()
        if hasattr(shadow_scene, "set_camera_look_at"):
            _apply_native_camera(shadow_scene, recipe.camera, distance_override=3.2, target_override=(0.0, 0.35, 0.0))
        shadow_scene.add_instanced_mesh(
            shadow_positions,
            shadow_indices,
            transforms,
            normals=shadow_normals,
            color=(1.0, 1.0, 1.0, 1.0),
            light_dir=(0.0, -1.0, 0.0),
            light_intensity=1.0,
        )
        shadow_rgba = np.asarray(shadow_scene.render_rgba())
        if shadow_rgba.ndim == 3 and shadow_rgba.shape[2] == 4:
            shadow_rgb_max = np.max(shadow_rgba[..., :3].astype(np.float32), axis=2)
            shadow_mask = np.where(shadow_rgb_max > 8.0, shadow_rgb_max / 255.0, 0.0).clip(0.0, 1.0)
            if np.any(shadow_mask > 0.01):
                rgb = out[..., :3].astype(np.float32)
                rgb *= (1.0 - 0.34 * shadow_mask[..., None])
                out[..., :3] = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    mesh_rgb = mesh_rgba[..., :3].astype(np.uint8, copy=False)
    mesh_mask = np.any(mesh_rgb > 8, axis=2)
    if np.any(mesh_mask):
        out[mesh_mask, :3] = mesh_rgb[mesh_mask]
        out[mesh_mask, 3] = 255
    metadata = {
        "building_backend": "native_instanced_mesh",
        "building_batch_count": len(batches),
        "building_batch_ids": batch_ids,
        "building_roof_types": roof_types,
        "building_shadow_model": "projected_native_shadow_mesh",
    }
    return out, True, metadata


def _building_textured_material_intents(layer: "BuildingLayer") -> list[dict[str, Any]]:
    metadata = _metadata_dict(layer.metadata)
    intents = metadata.get("textured_materials")
    if intents is None:
        single = metadata.get("textured_material") or metadata.get("texture_material")
        intents = [single] if isinstance(single, Mapping) else []
    if not isinstance(intents, Sequence) or isinstance(intents, (str, bytes)):
        return []
    return [_metadata(item) for item in intents if isinstance(item, Mapping)]


def _building_gltf_path(layer: "BuildingLayer") -> str | None:
    metadata = _metadata_dict(layer.metadata)
    for key in ("gltf_path", "glb_path", "asset_path"):
        value = metadata.get(key)
        if value:
            return str(value)
    if isinstance(layer.source, Mapping):
        value = layer.source.get("path") or layer.source.get("gltf_path") or layer.source.get("glb_path")
        return str(value) if value else None
    if layer.source is not None:
        return str(layer.source)
    return None


def _screen_rect_pixels(rect: Sequence[float] | None, *, width: int, height: int) -> tuple[int, int, int, int]:
    values = list(rect or (0.30, 0.18, 0.70, 0.72))
    while len(values) < 4:
        values.append(values[-1] if values else 0.0)
    x0, y0, x1, y1 = (float(value) for value in values[:4])
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    left = max(0, min(width - 1, int(round(min(x0, x1)))))
    right = max(left + 1, min(width, int(round(max(x0, x1)))))
    top = max(0, min(height - 1, int(round(min(y0, y1)))))
    bottom = max(top + 1, min(height, int(round(max(y0, y1)))))
    return left, top, right, bottom


def _composite_textured_landmark_layers(base: Any, recipe: "SceneRecipe") -> tuple[Any, bool, dict[str, Any]]:
    building_layers = [layer for layer in recipe.layers if isinstance(layer, BuildingLayer)]
    if not building_layers:
        return base, False, {}

    import numpy as np

    out = np.ascontiguousarray(base, dtype=np.uint8).copy()
    height, width = out.shape[:2]
    rendered = 0
    material_count = 0
    primitive_count = 0
    asset_ids: list[str] = []
    for layer in building_layers:
        intents = _building_textured_material_intents(layer)
        if not intents:
            continue
        gltf_path = _building_gltf_path(layer)
        if not gltf_path:
            continue
        try:
            from . import io

            mesh, materials, primitive_materials = io.import_gltf(gltf_path, with_materials=True)
        except Exception:
            continue
        texture_path = intents[0].get("albedo_texture") or intents[0].get("texture_path")
        if not texture_path:
            continue
        try:
            from ._png import load_png_rgba

            texture = np.asarray(load_png_rgba(texture_path), dtype=np.uint8)
        except Exception:
            continue
        if texture.ndim != 3 or texture.shape[2] != 4 or texture.shape[0] == 0 or texture.shape[1] == 0:
            continue
        rect = _metadata_dict(layer.metadata).get("screen_rect") or _metadata_dict(layer.metadata).get("landmark_screen_rect")
        rect_values = rect if isinstance(rect, Sequence) and not isinstance(rect, (str, bytes)) else None
        left, top, right, bottom = _screen_rect_pixels(rect_values, width=width, height=height)
        target_h = bottom - top
        target_w = right - left
        yy = np.linspace(0, texture.shape[0] - 1, target_h).astype(np.int32)
        xx = np.linspace(0, texture.shape[1] - 1, target_w).astype(np.int32)
        sampled = texture[np.ix_(yy, xx)].astype(np.float32)
        shade = np.linspace(1.08, 0.78, target_h, dtype=np.float32)[:, None]
        sampled[..., :3] *= shade[..., None]
        alpha = (sampled[..., 3:4] / 255.0) * float(intents[0].get("opacity", 1.0))
        region = out[top:bottom, left:right, :3].astype(np.float32)
        region = region * (1.0 - alpha) + sampled[..., :3] * alpha
        out[top:bottom, left:right, :3] = np.clip(region, 0.0, 255.0).astype(np.uint8)
        out[top:bottom, left:right, 3] = 255
        rendered += 1
        material_count += len(materials)
        primitive_count += len(primitive_materials) if primitive_materials else int(np.asarray(mesh.indices).reshape(-1, 3).shape[0])
        asset_ids.append(str(_metadata_dict(layer.metadata).get("source_id") or Path(gltf_path).stem))

    if rendered == 0:
        return base, False, {}
    return out, True, {
        "gltf_textured_backend": "mapscene_textured_landmark",
        "gltf_textured_layer_count": rendered,
        "gltf_material_count": material_count,
        "gltf_primitive_count": primitive_count,
        "gltf_asset_ids": asset_ids,
    }


def _apply_native_camera(
    native_scene: Any,
    camera: "OrbitCamera",
    *,
    distance_override: float | None = None,
    target_override: Sequence[float] | None = None,
) -> None:
    if not hasattr(native_scene, "set_camera_look_at"):
        return

    import math

    target_values = list(target_override if target_override is not None else camera.target or (0.0, 0.0, 0.0))
    while len(target_values) < 3:
        target_values.append(0.0)
    target = tuple(float(value) for value in target_values[:3])
    distance = max(1.0e-3, float(distance_override if distance_override is not None else camera.distance))
    azimuth = math.radians(float(camera.azimuth_deg))
    elevation = math.radians(float(camera.elevation_deg))
    horizontal = distance * math.cos(elevation)
    eye = (
        target[0] + horizontal * math.sin(azimuth),
        target[1] + distance * math.sin(elevation),
        target[2] + horizontal * math.cos(azimuth),
    )
    near = float(camera.near) if camera.near is not None else max(0.01, distance * 0.001)
    far = float(camera.far) if camera.far is not None else max(distance * 4.0, near * 2.0)
    native_scene.set_camera_look_at(eye, target, (0.0, 1.0, 0.0), float(camera.fov_deg), near, far)


def _needs_native_building_composite(recipe: "SceneRecipe") -> bool:
    building_layers = [layer for layer in recipe.layers if isinstance(layer, BuildingLayer)]
    if not building_layers:
        return False
    return _building_scene_bounds(building_layers) is not None


def _native_composite_blocks(
    recipe: "SceneRecipe",
    *,
    native_labels: bool,
    native_vectors: bool,
    native_buildings: bool,
    native_point_clouds: bool,
    plans: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Structured diagnostic blocks for every layer that could not composite natively."""
    blocks: list[dict[str, Any]] = []
    labels_needed = _needs_native_label_composite(recipe, plans) and not native_labels
    vectors_needed = _needs_native_vector_composite(recipe) and not native_vectors
    buildings_needed = _needs_native_building_composite(recipe) and not native_buildings
    for layer in recipe.layers:
        if isinstance(layer, LabelLayer) and labels_needed:
            blocks.append(
                diagnostic_block(
                    layer=_layer_id(layer, "layer"),
                    reason="native label compositing is unavailable or produced no output; "
                    "CPU placeholder label drawing has been removed",
                    required_native="Scene.enable_native_text",
                )
            )
        elif isinstance(layer, VectorOverlay) and vectors_needed:
            blocks.append(
                diagnostic_block(
                    layer=_layer_id(layer, "layer"),
                    reason="native vector OIT compositing is unavailable; "
                    "CPU placeholder vector drawing has been removed",
                    required_native=", ".join(required_native_symbols("vector")),
                )
            )
        elif isinstance(layer, BuildingLayer) and buildings_needed:
            blocks.append(
                diagnostic_block(
                    layer=_layer_id(layer, "layer"),
                    reason="native building mesh compositing is unavailable; "
                    "CPU placeholder building drawing has been removed",
                    required_native="Scene.add_instanced_mesh",
                )
            )
        elif isinstance(layer, PointCloudLayer) and not native_point_clouds:
            blocks.append(
                diagnostic_block(
                    layer=_layer_id(layer, "layer"),
                    reason="native point-cloud OIT compositing is unavailable and the CPU "
                    "fallback compositor does not render point clouds",
                    required_native=", ".join(required_native_symbols("point_cloud")),
                )
            )
        elif (
            isinstance(layer, Tiles3DLayer)
            and _tiles3d_layer_has_native_geometry(layer)
            and not native_point_clouds
        ):
            blocks.append(
                diagnostic_block(
                    layer=_layer_id(layer, "layer"),
                    reason="native 3D Tiles OIT compositing is unavailable and the CPU "
                    "fallback compositor does not render tile geometry",
                    required_native=", ".join(required_native_symbols("tiles3d")),
                )
            )
    return blocks


def _render_native_offscreen_result(
    recipe: "SceneRecipe",
    compiled: "CompiledScenePlan",
    *,
    emit_provenance: bool = False,
) -> _MapSceneNativeRenderResult | None:
    if not isinstance(compiled, CompiledScenePlan):
        raise RuntimeError(
            "MapScene render phase requires a CompiledScenePlan; "
            "call MapScene.compile_plan() before rendering"
        )
    plans = compiled.label_plans
    heightmap = _load_native_heightmap(recipe.terrain)
    if heightmap is None or recipe.output is None:
        return None

    import numpy as np

    try:
        # Keyword passed only when enabled so existing call-compatible test
        # doubles for `_render_terrain_renderer_result` stay valid.
        if emit_provenance:
            result = _render_terrain_renderer_result(recipe, heightmap, emit_provenance=True)
        else:
            result = _render_terrain_renderer_result(recipe, heightmap)
    except BaseException as exc:
        if _is_native_adapter_unavailable(exc):
            return None
        raise
    if result is None:
        return None
    rgba = result.rgba
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise RuntimeError("MapScene terrain renderer returned an invalid RGBA image")
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8)
    base = np.ascontiguousarray(rgba.copy())
    base, cloud_shadow_metadata = _apply_mapscene_cloud_shadow(base, recipe)
    native_buildings = result.metadata.get("building_backend") == "terrain_scatter_instanced_mesh"
    building_metadata: dict[str, Any] = {}
    if not native_buildings:
        base, native_buildings, building_metadata = _composite_native_building_layers(base, recipe)
    base, textured_landmarks, textured_metadata = _composite_textured_landmark_layers(base, recipe)
    base, screen_space_metadata = _apply_mapscene_screen_space(base, recipe, heightmap)
    base, native_labels = _composite_native_label_layers(base, recipe, plans)
    base, native_vectors = _composite_native_vector_layers(base, recipe)
    base, native_point_clouds, point_tile_metadata = _composite_native_point_cloud_layers(base, recipe)
    blocks = _native_composite_blocks(
        recipe,
        native_labels=native_labels,
        native_vectors=native_vectors,
        native_buildings=native_buildings,
        native_point_clouds=native_point_clouds,
        plans=plans,
    )
    if blocks:
        raise MapSceneNativeUnavailable(blocks)
    target_grid = _terrain_alignment_grid(
        recipe.terrain,
        target_crs=recipe.target_crs or recipe.terrain.crs,
        fallback_shape=heightmap.shape,
    )

    raster_overlay_layers: list["RasterOverlay"] = []

    def load_raster_overlay(layer: "RasterOverlay") -> Any | None:
        overlay = _load_native_raster_overlay(layer, target_grid=target_grid)
        if overlay is not None:
            raster_overlay_layers.append(layer)
        return overlay

    composited = _composite_recipe_layers(
        base,
        recipe,
        plans,
        layer_types=_render_layer_types(),
        load_raster_overlay=load_raster_overlay,
        include_raster=True,
        include_vectors=False,
        include_labels=False,
        include_buildings=False,
        include_point_cloud=False,
    )
    metadata = dict(result.metadata)
    metadata.update(cloud_shadow_metadata)
    metadata.update(screen_space_metadata)
    if native_buildings:
        metadata.update(building_metadata)
    if textured_landmarks:
        metadata.update(textured_metadata)
    if native_point_clouds:
        metadata.update(point_tile_metadata)
    if raster_overlay_layers:
        # Honest contract: loaded raster overlays are composited by the
        # deterministic CPU resample compositor, not a native-only path.
        metadata["raster_overlay_backend"] = "python_resample_composite"
        metadata["raster_overlay_layer_count"] = len(raster_overlay_layers)
    vector_layers = [layer for layer in recipe.layers if isinstance(layer, VectorOverlay)]
    if vector_layers and any(
        _vector_layer_requires_precise_raster(layer) for layer in vector_layers
    ):
        # Dashed/mitered precise vectors route through the deterministic CPU
        # raster path, not native OIT.
        metadata["vector_backend"] = "python_precise_raster"
    elif native_vectors:
        metadata["vector_backend"] = "native_oit"
    return _MapSceneNativeRenderResult(
        rgba=composited,
        aov_frame=result.aov_frame,
        hdr_frame=result.hdr_frame,
        metadata=metadata,
        source_map=result.source_map,
        contributing_tiles=result.contributing_tiles,
    )


@dataclass
class TerrainSource:
    path: str | Path | None = None
    data: Any | None = None
    crs: str | None = None
    metadata: Mapping[str, Any] | None = None
    elevation_sampling_available: bool = False
    dtype: str = "float32"
    nodata_policy: str = "fill"

    def __post_init__(self) -> None:
        import numpy as np

        np.dtype(self.dtype)
        if str(self.nodata_policy).lower() not in {"fill", "preserve"}:
            raise ValueError("TerrainSource nodata_policy must be 'fill' or 'preserve'")

    def to_dict(self) -> dict[str, Any]:
        data_summary = None
        if self.data is not None:
            import numpy as np

            arr = np.asarray(self.data)
            data_summary = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        return {
            "kind": "terrain_source",
            "path": _path_to_str(self.path),
            "data": data_summary,
            "crs": self.crs,
            "metadata": _metadata(self.metadata),
            "elevation_sampling_available": bool(self.elevation_sampling_available),
            "dtype": str(self.dtype),
            "nodata_policy": str(self.nodata_policy),
        }


@dataclass
class RasterOverlay:
    layer_id: str
    path: str | Path | None = None
    crs: str | None = None
    opacity: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "raster_overlay",
            "layer_id": str(self.layer_id),
            "path": _path_to_str(self.path),
            "crs": self.crs,
            "opacity": float(self.opacity),
            "metadata": _metadata(self.metadata),
        }


@dataclass
class VectorOverlay:
    layer_id: str
    path: str | Path | None = None
    features: Sequence[Mapping[str, Any]] | None = None
    crs: str | None = None
    style: Mapping[str, Any] | None = None
    width_px: float | None = None
    width_world: float | None = None
    line_join: str = "miter"
    line_cap: str = "butt"
    dash_array: Sequence[float] | None = None
    style_support: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.width_px is not None and float(self.width_px) <= 0.0:
            raise ValueError("VectorOverlay.width_px must be positive")
        if self.width_world is not None and float(self.width_world) <= 0.0:
            raise ValueError("VectorOverlay.width_world must be positive")
        join = str(self.line_join or "miter").lower()
        cap = str(self.line_cap or "butt").lower()
        if join not in {"miter", "bevel", "round"}:
            raise ValueError("VectorOverlay.line_join must be 'miter', 'bevel', or 'round'")
        if cap not in {"butt", "round", "square"}:
            raise ValueError("VectorOverlay.line_cap must be 'butt', 'round', or 'square'")
        self.line_join = join
        self.line_cap = cap
        if self.dash_array is not None:
            values = tuple(float(value) for value in self.dash_array)
            if not values:
                self.dash_array = None
                return
            if any(value <= 0.0 for value in values):
                raise ValueError("VectorOverlay.dash_array must contain positive lengths")
            self.dash_array = values

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_overlay",
            "layer_id": str(self.layer_id),
            "path": _path_to_str(self.path),
            "features": _sequence(self.features),
            "crs": self.crs,
            "style": _metadata(self.style),
            "width_px": self.width_px,
            "width_world": self.width_world,
            "line_join": str(self.line_join),
            "line_cap": str(self.line_cap),
            "dash_array": _sequence(self.dash_array),
            "style_support": _metadata(self.style_support),
            "metadata": _metadata(self.metadata),
        }


@dataclass(frozen=True)
class FontFallbackRange:
    name: str
    start: int
    end: int
    font_family: str

    def __post_init__(self) -> None:
        if int(self.end) < int(self.start):
            raise ValueError("FontFallbackRange end must be greater than or equal to start")

    def covers(self, char: str) -> bool:
        if not char:
            return False
        codepoint = ord(str(char)[0])
        return int(self.start) <= codepoint <= int(self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "start": int(self.start),
            "end": int(self.end),
            "font_family": str(self.font_family),
        }


@dataclass
class FontAtlas:
    glyphs: set[str] = field(default_factory=set)
    font_size: int = 24
    line_height: int = 32
    baseline: int = 24
    coverage: Mapping[str, Any] | None = None
    source_path: str | None = None
    fallbacks: Sequence[FontFallbackRange | Mapping[str, Any]] = field(default_factory=tuple)
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.glyphs = {str(glyph) for glyph in self.glyphs}
        self.coverage = _metadata(self.coverage)
        self.fallbacks = tuple(
            fallback
            if isinstance(fallback, FontFallbackRange)
            else FontFallbackRange(
                str(fallback["name"]),
                int(fallback["start"]),
                int(fallback["end"]),
                str(fallback["font_family"]),
            )
            for fallback in self.fallbacks
        )
        self.fallbacks = tuple(sorted(self.fallbacks, key=lambda item: (item.start, item.end, item.name)))
        self.diagnostics = [
            diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
            for diagnostic in self.diagnostics
        ]

    @classmethod
    def default_latin(
        cls,
        *,
        fallbacks: Sequence[FontFallbackRange | Mapping[str, Any]] | None = None,
    ) -> "FontAtlas":
        from .text_atlas import default_latin_atlas_paths, load_atlas_metrics

        image_path, atlas_path = default_latin_atlas_paths()
        if not image_path.is_file() or not atlas_path.is_file():
            raise FileNotFoundError(
                "packaged default MSDF atlas is incomplete; expected both "
                f"{image_path.name} and {atlas_path.name}"
            )
        payload = load_atlas_metrics(atlas_path)
        glyphs = {chr(int(key)) for key in payload.get("glyphs", {})}
        return cls(
            glyphs=glyphs,
            font_size=int(payload.get("font_size", 24)),
            line_height=int(payload.get("line_height", 32)),
            baseline=int(payload.get("baseline", 24)),
            coverage={
                "start": 32,
                "end": 127,
                "name": "Basic Latin",
                "atlas_kind": payload["kind"],
                "image_path": str(image_path),
                "px_range": payload.get("px_range"),
            },
            source_path=str(atlas_path),
            fallbacks=fallbacks or (),
        )

    @classmethod
    def from_font(
        cls,
        path: str | Path,
        *,
        ranges: Sequence[FontFallbackRange | Mapping[str, Any]] | None = None,
        font_size: int = 24,
        line_height: int | None = None,
    ) -> "FontAtlas":
        font_path = Path(path)
        if not font_path.exists():
            return cls(
                font_size=font_size,
                line_height=line_height or int(round(font_size * 4 / 3)),
                baseline=font_size,
                source_path=str(font_path),
                fallbacks=ranges or (),
                diagnostics=[
                    missing_external_asset_diagnostic(
                        "font_atlas",
                        object_id=str(path),
                        path=str(font_path),
                    )
                ],
            )
        glyphs: set[str] = set()
        for fallback in ranges or ():
            item = fallback if isinstance(fallback, FontFallbackRange) else FontFallbackRange(
                str(fallback["name"]),
                int(fallback["start"]),
                int(fallback["end"]),
                str(fallback["font_family"]),
            )
            glyphs.update(chr(codepoint) for codepoint in range(item.start, item.end + 1))
        if not glyphs:
            glyphs = set("".join(chr(codepoint) for codepoint in range(32, 128)))
        return cls(
            glyphs=glyphs,
            font_size=font_size,
            line_height=line_height or int(round(font_size * 4 / 3)),
            baseline=font_size,
            coverage={"source": "font_file", "path": str(font_path)},
            source_path=str(font_path),
            fallbacks=ranges or (),
        )

    def covers(self, char: str) -> bool:
        return str(char) in self.glyphs or self.fallback_for(str(char)) is not None

    def fallback_for(self, char: str) -> FontFallbackRange | None:
        for fallback in self.fallbacks:
            if fallback.covers(char):
                return fallback
        return None

    def validate_text(self, text: str, *, layer_id: str | None = None, object_id: str | None = None) -> list[Diagnostic]:
        missing = sorted({char for char in str(text) if not self.covers(char)})
        if not missing:
            return []
        return [unicode_coverage_gap_diagnostic(missing, layer_id=layer_id, object_id=object_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "glyphs": sorted(self.glyphs),
            "font_size": int(self.font_size),
            "line_height": int(self.line_height),
            "baseline": int(self.baseline),
            "coverage": _metadata(self.coverage),
            "source_path": self.source_path,
            "fallbacks": [fallback.to_dict() for fallback in self.fallbacks],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


def _font_atlas_payload(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, FontAtlas):
        return value.to_dict()
    return _metadata(value)


@dataclass(frozen=True)
class TypographySettings:
    font_size: int = 24
    kerning: bool = True
    tracking: float = 0.0
    line_height: float | None = None
    multiline: bool = False
    callout: bool = False
    callout_offset: Sequence[float] = (0.0, 0.0)
    halo_width_px: float = 1.0
    halo_color: Sequence[float] | str | None = (1.0, 1.0, 1.0, 0.8)

    def measure_text(self, text: str) -> dict[str, Any]:
        from ._map_scene_render import _text_font_chain
        from .text import shape

        lines = str(text).splitlines() or [""]
        font_chain = _text_font_chain()
        line_widths = []
        kerning_applied = False
        for line in lines:
            width = 0.0
            if line:
                features = {} if self.kerning else {"kern": False}
                shaped = shape(line, font_chain, float(self.font_size), features=features)
                bounds = shaped.outline_bounds()
                if bounds is not None:
                    x0, _y0, x1, _y1 = (float(value) for value in bounds)
                    width = max(0.0, x1 - x0)
                kerning_applied = kerning_applied or (bool(self.kerning) and len(line) > 1)
                width += len(line) * float(self.tracking)
            line_widths.append(max(0.0, width))
        line_height = float(self.line_height if self.line_height is not None else self.font_size * 4 / 3)
        return {
            "width": max(line_widths) if line_widths else 0.0,
            "height": line_height * len(lines),
            "line_count": len(lines),
            "line_height": line_height,
            "kerning_applied": kerning_applied,
            "tracking": float(self.tracking),
        }

    def layout_label(self, text: str, *, anchor: Sequence[float]) -> dict[str, Any]:
        lines = str(text).splitlines() or [""]
        if not self.multiline:
            lines = [" ".join(lines)]
        anchor_values = [float(value) for value in anchor]
        while len(anchor_values) < 3:
            anchor_values.append(0.0)
        offset = [float(value) for value in self.callout_offset[:2]]
        while len(offset) < 2:
            offset.append(0.0)
        label_anchor = [anchor_values[0] + offset[0], anchor_values[1] + offset[1], anchor_values[2]]
        return {
            "lines": lines,
            "metrics": self.measure_text("\n".join(lines)),
            "callout": {
                "enabled": bool(self.callout),
                "anchor": anchor_values[:3],
                "label_anchor": label_anchor,
                "offset": offset[:2],
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "font_size": int(self.font_size),
            "kerning": bool(self.kerning),
            "tracking": float(self.tracking),
            "line_height": self.line_height,
            "multiline": bool(self.multiline),
            "callout": bool(self.callout),
            "callout_offset": _sequence(self.callout_offset),
            "halo_width_px": float(self.halo_width_px),
            "halo_color": self.halo_color
            if self.halo_color is None or isinstance(self.halo_color, str)
            else _sequence(self.halo_color),
        }


@dataclass
class LabelLayer:
    layer_id: str
    labels: Sequence[Mapping[str, Any]] | None = None
    glyph_atlas: Mapping[str, Any] | None = None
    atlas: FontAtlas | Mapping[str, Any] | None = None
    typography: Mapping[str, Any] | None = None
    occlusion: str = "terrain"
    priority_rules: Sequence[Any] | None = None
    plan: Any | None = None
    metadata: Mapping[str, Any] | None = None
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        value = str(self.occlusion or "terrain").lower()
        if value not in {"none", "terrain"}:
            raise ValueError("LabelLayer.occlusion must be 'none' or 'terrain'")
        self.occlusion = value
        if self.glyph_atlas is None and self.atlas is not None:
            self.glyph_atlas = _font_atlas_payload(self.atlas)

    @classmethod
    def from_features(
        cls,
        features: Sequence[Mapping[str, Any]],
        *,
        text: Any = "name",
        crs: str | None = None,
        target_crs: str | None = None,
        terrain_sampling: str = "auto",
        terrain_sampler: Any | None = None,
        typography: Mapping[str, Any] | None = None,
        occlusion: str = "terrain",
        layer_id: str = "labels",
        glyph_atlas: Mapping[str, Any] | None = None,
        atlas: FontAtlas | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LabelLayer":
        labels: list[dict[str, Any]] = []
        diagnostics: list[Diagnostic] = []
        for index, feature in enumerate(features or ()):
            feature_id = _feature_id(feature, index)
            geometry = _feature_geometry(feature)
            geometry_type, placement_kind = _valid_geometry(geometry)
            if geometry_type is None:
                diagnostics.append(
                    placeholder_fallback_diagnostic(
                        "label invalid geometry",
                        layer_id=layer_id,
                        object_id=feature_id,
                    )
                )
                continue
            if placement_kind is None:
                diagnostics.append(
                    _unsupported_feature_diagnostic(
                        f"label geometry type {geometry_type}",
                        layer_id=layer_id,
                        object_id=feature_id,
                    )
                )
                continue

            transformed_geometry = _transform_label_geometry(geometry, from_crs=crs, to_crs=target_crs)
            sampled_geometry = transformed_geometry
            terrain_sample: Mapping[str, Any] = {}
            if terrain_sampling == "required" and terrain_sampler is not None:
                sampled_geometry, terrain_sample = _sample_label_geometry(
                    transformed_geometry,
                    terrain_sampler=terrain_sampler,
                )

            properties = _feature_properties(feature)
            label_text, missing_field = _label_text_from_expression(text, properties)
            if missing_field is not None:
                diagnostics.append(
                    missing_label_field_diagnostic(
                        missing_field,
                        layer_id=layer_id,
                        object_id=feature_id,
                    )
                )
                continue

            if terrain_sampling == "required" and terrain_sampler is None:
                diagnostics.append(unavailable_terrain_sampler_diagnostic(layer_id=layer_id, object_id=feature_id))

            label_record = {
                "id": feature_id,
                "source_id": feature_id,
                "text": label_text,
                "geometry": _json_safe(sampled_geometry),
                "geometry_type": geometry_type,
                "placement_kind": placement_kind,
                "properties": _metadata(properties),
                "terrain_mode": "required" if terrain_sampling == "required" else terrain_sampling,
            }
            if terrain_sample:
                label_record["terrain_sample"] = _metadata(terrain_sample)
            labels.append(label_record)

        labels.sort(
            key=lambda label: (
                str(label.get("id", "")),
                str(label.get("geometry_type", "")),
                str(label.get("text", "")),
            )
        )
        layer_metadata = _metadata(metadata)
        if crs is not None:
            layer_metadata["crs"] = target_crs or crs
            if target_crs is not None and not _same_crs(crs, target_crs):
                layer_metadata["source_crs"] = crs
        layer_metadata["terrain_sampling"] = terrain_sampling
        return cls(
            layer_id=layer_id,
            labels=labels,
            glyph_atlas=glyph_atlas,
            atlas=atlas,
            typography=typography,
            occlusion=occlusion,
            metadata=layer_metadata,
            diagnostics=diagnostics,
        )

    @classmethod
    def from_geodataframe(
        cls,
        gdf: Any,
        *,
        text: Any = "name",
        crs: str | None = None,
        target_crs: str | None = None,
        terrain_sampling: str = "auto",
        terrain_sampler: Any | None = None,
        typography: Mapping[str, Any] | None = None,
        occlusion: str = "terrain",
        layer_id: str = "labels",
        glyph_atlas: Mapping[str, Any] | None = None,
        atlas: FontAtlas | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LabelLayer":
        if not hasattr(gdf, "iterrows"):
            raise TypeError("LabelLayer.from_geodataframe requires a GeoDataFrame-like object")
        features: list[dict[str, Any]] = []
        for index, row in gdf.iterrows():
            properties = {
                str(key): row[key]
                for key in getattr(gdf, "columns", ())
                if str(key) != "geometry"
            }
            geometry = row.get("geometry") if hasattr(row, "get") else getattr(row, "geometry", None)
            if hasattr(geometry, "__geo_interface__"):
                geometry_payload = geometry.__geo_interface__
            else:
                geometry_payload = geometry
            features.append(
                {
                    "type": "Feature",
                    "id": str(index),
                    "properties": properties,
                    "geometry": geometry_payload,
                }
            )
        effective_crs = crs or str(getattr(gdf, "crs", "") or "") or None
        return cls.from_features(
            features,
            text=text,
            crs=effective_crs,
            target_crs=target_crs,
            terrain_sampling=terrain_sampling,
            terrain_sampler=terrain_sampler,
            typography=typography,
            occlusion=occlusion,
            layer_id=layer_id,
            glyph_atlas=glyph_atlas,
            atlas=atlas,
            metadata=metadata,
        )

    @classmethod
    def from_style_layer(
        cls,
        features: Sequence[Mapping[str, Any]],
        style_layer: Any,
        *,
        crs: str | None = None,
        target_crs: str | None = None,
        terrain_sampling: str = "auto",
        terrain_sampler: Any | None = None,
        occlusion: str = "terrain",
        layer_id: str = "labels",
        glyph_atlas: Mapping[str, Any] | None = None,
        atlas: FontAtlas | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LabelLayer":
        return cls.from_features(
            features,
            text=_style_text_expression(style_layer) or "name",
            crs=crs,
            target_crs=target_crs,
            terrain_sampling=terrain_sampling,
            terrain_sampler=terrain_sampler,
            occlusion=occlusion,
            layer_id=layer_id,
            glyph_atlas=glyph_atlas,
            atlas=atlas,
            metadata=metadata,
        )

    def compile_labels(self, camera: Any, viewport: Any, terrain: Any | None = None) -> Any:
        from .label_plan import LabelPlan

        return LabelPlan.compile(
            labels=self.labels or (),
            camera=camera,
            viewport=viewport,
            terrain=terrain,
            priority_rules=self.priority_rules or (),
            typography=self.typography or {},
            glyph_atlas=self.glyph_atlas,
            seed=int(_metadata_dict(self.metadata).get("seed", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "label_layer",
            "layer_id": str(self.layer_id),
            "labels": _sequence(self.labels),
            "glyph_atlas": _metadata(self.glyph_atlas),
            "atlas": _font_atlas_payload(self.atlas),
            "typography": _metadata(self.typography),
            "occlusion": str(self.occlusion),
            "priority_rules": _sequence(self.priority_rules),
            "plan": _json_safe(self.plan) if self.plan is not None else None,
            "metadata": _metadata(self.metadata),
            "diagnostics": [
                diagnostic.to_dict() if isinstance(diagnostic, Diagnostic) else _json_safe(diagnostic)
                for diagnostic in (self.diagnostics or ())
            ],
        }


@dataclass
class PointCloudLayer:
    layer_id: str
    path: str | Path | None = None
    crs: str | None = None
    point_count: int | None = None
    support_level: str = "native-required"
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "point_cloud_layer",
            "layer_id": str(self.layer_id),
            "path": _path_to_str(self.path),
            "crs": self.crs,
            "point_count": self.point_count,
            "support_level": str(self.support_level),
            "metadata": _metadata(self.metadata),
        }


@dataclass
class BuildingLayer:
    layer_id: str
    source: str | Mapping[str, Any] | None = None
    support_level: str = "underdeveloped"
    geometry_count: int | None = None
    bounds: Sequence[float] | None = None
    material_status: str | None = None
    features: Sequence[Mapping[str, Any]] | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        **options: Any,
    ) -> "BuildingLayer":
        features = options.pop("features", None)
        metadata = _metadata(options.pop("metadata", None))
        metadata.update(_metadata(options))
        metadata.setdefault("source_format", "geojson")
        path_obj = Path(path)
        if features is None and path_obj.exists():
            try:
                payload = json.loads(path_obj.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping) and isinstance(payload.get("features"), Sequence):
                    features = [item for item in payload.get("features", ()) if isinstance(item, Mapping)]
            except Exception:
                features = None
        return cls(
            layer_id=str(metadata.pop("layer_id", None) or path_obj.stem or "buildings"),
            source={"path": str(path), "source_format": "geojson"},
            support_level=str(metadata.pop("support_level", "supported")),
            geometry_count=(
                metadata.pop("geometry_count")
                if "geometry_count" in metadata
                else (len(features) if features is not None else None)
            ),
            bounds=metadata.pop("bounds", None),
            material_status=str(metadata.pop("material_status", "scalar_pbr_underdeveloped")),
            features=features,
            metadata=metadata,
        )

    @classmethod
    def from_cityjson(
        cls,
        path: str | Path,
        **options: Any,
    ) -> "BuildingLayer":
        metadata = _metadata(options.pop("metadata", None))
        metadata.update(_metadata(options))
        metadata.setdefault("source_format", "cityjson")
        geometry_count = metadata.pop("geometry_count", None)
        path_obj = Path(path)
        if geometry_count is None and path_obj.exists():
            try:
                payload = json.loads(path_obj.read_text(encoding="utf-8"))
                city_objects = payload.get("CityObjects") if isinstance(payload, Mapping) else None
                if isinstance(city_objects, Mapping):
                    geometry_count = len(city_objects)
            except Exception:
                geometry_count = None
        return cls(
            layer_id=str(metadata.pop("layer_id", None) or path_obj.stem or "buildings"),
            source={"path": str(path), "source_format": "cityjson"},
            support_level=str(metadata.pop("support_level", "underdeveloped")),
            geometry_count=geometry_count,
            bounds=metadata.pop("bounds", None),
            material_status=str(metadata.pop("material_status", "scalar_pbr_underdeveloped")),
            features=metadata.pop("features", None),
            metadata=metadata,
        )

    @classmethod
    def from_mesh(
        cls,
        path: str | Path | None = None,
        **options: Any,
    ) -> "BuildingLayer":
        metadata = _metadata(options.pop("metadata", None))
        metadata.update(_metadata(options))
        metadata.setdefault("source_format", "mesh")
        return cls(
            layer_id=str(metadata.pop("layer_id", None) or (Path(path).stem if path else "buildings.mesh")),
            source={"path": str(path), "source_format": "mesh"} if path else {"source_format": "mesh"},
            support_level=str(metadata.pop("support_level", "unsupported")),
            geometry_count=metadata.pop("geometry_count", None),
            bounds=metadata.pop("bounds", None),
            material_status=str(metadata.pop("material_status", "unsupported")),
            features=metadata.pop("features", None),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "building_layer",
            "layer_id": str(self.layer_id),
            "source": _json_safe(self.source),
            "support_level": self.support_level,
            "geometry_count": self.geometry_count,
            "bounds": _sequence(self.bounds),
            "material_status": self.material_status,
            "features": _sequence(self.features),
            "metadata": _metadata(self.metadata),
        }


MapSceneBuildingLayer = BuildingLayer


@dataclass
class Tiles3DLayer:
    layer_id: str
    source: str | Mapping[str, Any] | None = None
    support_level: str = "underdeveloped"
    lod: Mapping[str, Any] | None = None
    cache_budget: int | None = None
    cache_stats: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] | None = None

    @classmethod
    def from_tileset_json(
        cls,
        path: str | Path,
        *,
        lod: Mapping[str, Any] | None = None,
        cache_budget: int | None = None,
        cache_stats: Mapping[str, Any] | None = None,
        layer_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Tiles3DLayer":
        data = _metadata(metadata)
        data.setdefault("source_format", "tileset.json")
        return cls(
            layer_id=layer_id or Path(path).stem or "tiles3d",
            source={"path": str(path), "source_format": "tileset.json"},
            lod=lod,
            cache_budget=cache_budget,
            cache_stats=cache_stats,
            metadata=data,
        )

    @classmethod
    def from_b3dm(
        cls,
        path: str | Path,
        *,
        lod: Mapping[str, Any] | None = None,
        cache_budget: int | None = None,
        cache_stats: Mapping[str, Any] | None = None,
        layer_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Tiles3DLayer":
        data = _metadata(metadata)
        data.setdefault("source_format", "b3dm")
        return cls(
            layer_id=layer_id or Path(path).stem or "tiles3d",
            source={"path": str(path), "source_format": "b3dm"},
            lod=lod,
            cache_budget=cache_budget,
            cache_stats=cache_stats,
            metadata=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "tiles3d_layer",
            "layer_id": str(self.layer_id),
            "source": _json_safe(self.source),
            "support_level": self.support_level,
            "lod": _metadata(self.lod),
            "cache_budget": self.cache_budget,
            "cache_stats": _metadata(self.cache_stats),
            "metadata": _metadata(self.metadata),
            "diagnostics": [
                diagnostic.to_dict() if isinstance(diagnostic, Diagnostic) else _json_safe(diagnostic)
                for diagnostic in (self.diagnostics or ())
            ],
        }


@dataclass
class MapFurnitureLayer:
    title: str | None = None
    legend: Mapping[str, Any] | None = None
    scale_bar: Mapping[str, Any] | None = None
    north_arrow: Mapping[str, Any] | None = None
    graticule: "GraticuleSpec | Mapping[str, Any] | None" = None
    keepouts: Sequence[Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "map_furniture_layer",
            "title": self.title,
            "legend": _metadata(self.legend) if self.legend is not None else None,
            "scale_bar": _metadata(self.scale_bar) if self.scale_bar is not None else None,
            "north_arrow": _metadata(self.north_arrow) if self.north_arrow is not None else None,
            "graticule": _json_safe(self.graticule) if self.graticule is not None else None,
            "keepouts": _sequence(self.keepouts),
            "metadata": _metadata(self.metadata),
        }


@dataclass
class OrbitCamera:
    target: Sequence[float] = (0.0, 0.0, 0.0)
    distance: float = 1.0
    azimuth_deg: float = 0.0
    elevation_deg: float = 45.0
    fov_deg: float = 45.0
    near: float | None = None
    far: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "orbit_camera",
            "target": _sequence(self.target),
            "distance": float(self.distance),
            "azimuth_deg": float(self.azimuth_deg),
            "elevation_deg": float(self.elevation_deg),
            "fov_deg": float(self.fov_deg),
            "near": self.near,
            "far": self.far,
        }


@dataclass
class LightingPreset:
    name: str = "default"
    sun_direction: Sequence[float] | None = None
    intensity: float = 1.0
    settings: Mapping[str, Any] | None = None
    overrides: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "lighting_preset",
            "name": str(self.name),
            "sun_direction": _sequence(self.sun_direction),
            "intensity": float(self.intensity),
            "settings": _metadata(self.settings),
            "overrides": _metadata(self.overrides),
        }


@dataclass
class OutputSpec:
    width: int
    height: int
    format: str = "png"
    path: str | Path | None = None
    samples: int = 1
    denoiser: str = "none"
    aovs: Sequence[str] = field(default_factory=tuple)
    hdr: bool = False
    bit_depth: int = 8
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("OutputSpec width and height must be positive")
        if int(self.samples) <= 0:
            raise ValueError("OutputSpec samples must be positive")
        if int(self.bit_depth) not in {8, 16}:
            raise ValueError("OutputSpec bit_depth must be 8 or 16")
        self.bit_depth = int(self.bit_depth)
        denoiser = str(self.denoiser).lower()
        if denoiser not in {"none", "off", "atrous", "oidn"}:
            raise ValueError("OutputSpec denoiser must be one of: none, off, atrous, oidn")
        self.denoiser = "none" if denoiser == "off" else denoiser
        allowed_aovs = {"albedo", "normal", "depth"}
        normalized_aovs = tuple(str(item).lower() for item in self.aovs or ())
        unknown = sorted(set(normalized_aovs) - allowed_aovs)
        if unknown:
            raise ValueError(f"Unsupported OutputSpec AOV(s): {', '.join(unknown)}")
        self.aovs = normalized_aovs

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "output_spec",
            "width": int(self.width),
            "height": int(self.height),
            "format": str(self.format),
            "path": _path_to_str(self.path),
            "samples": int(self.samples),
            "denoiser": str(self.denoiser),
            "aovs": list(self.aovs),
            "hdr": bool(self.hdr),
            "bit_depth": int(self.bit_depth),
            "metadata": _metadata(self.metadata),
        }


@dataclass
class ReproducibilityProfile:
    seed: int = 0
    camera: Mapping[str, Any] | None = None
    output_size: Sequence[int] | None = None
    terrain_transform: Mapping[str, Any] | None = None
    style_hashes: Mapping[str, str] | None = None
    asset_hashes_or_ids: Mapping[str, str] | None = None
    renderer_backend: str | None = None
    pixel_tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "reproducibility_profile",
            "seed": int(self.seed),
            "camera": _metadata(self.camera),
            "output_size": _sequence(self.output_size),
            "terrain_transform": _metadata(self.terrain_transform),
            "style_hashes": _metadata(self.style_hashes),
            "asset_hashes_or_ids": _metadata(self.asset_hashes_or_ids),
            "renderer_backend": self.renderer_backend,
            "pixel_tolerance": self.pixel_tolerance,
        }


@dataclass
class SceneRecipe:
    terrain: TerrainSource
    camera: OrbitCamera
    lighting: LightingPreset
    layers: Sequence[Any] = field(default_factory=tuple)
    output: OutputSpec | None = None
    target_crs: str | None = None
    map_furniture: MapFurnitureLayer | None = None
    render_policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING
    diagnostics_policy: Mapping[str, Any] | None = None
    reproducibility_profile: ReproducibilityProfile | None = None

    def __post_init__(self) -> None:
        self.render_policy = RenderFailurePolicy.validate(self.render_policy)
        self.layers = tuple(self.layers or ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "scene_recipe",
            "terrain": _json_safe(self.terrain),
            "camera": _json_safe(self.camera),
            "lighting": _json_safe(self.lighting),
            "layers": _sequence(self.layers),
            "output": _json_safe(self.output) if self.output is not None else None,
            "target_crs": self.target_crs,
            "map_furniture": _json_safe(self.map_furniture) if self.map_furniture is not None else None,
            "render_policy": self.render_policy,
            "diagnostics_policy": _metadata(self.diagnostics_policy),
            "reproducibility_profile": (
                _json_safe(self.reproducibility_profile)
                if self.reproducibility_profile is not None
                else None
            ),
        }


def _camera_from_preset(current: OrbitCamera, terrain: TerrainSource, camera_data: Mapping[str, Any]) -> OrbitCamera:
    radius_scale = camera_data.get("radius_scale")
    distance = camera_data.get("distance")
    if distance is None and radius_scale is not None:
        distance = _terrain_scene_diagonal(terrain) * float(radius_scale)
    if distance is None:
        distance = current.distance
    return OrbitCamera(
        target=camera_data.get("target", current.target),
        distance=float(distance),
        azimuth_deg=float(camera_data.get("azimuth_deg", current.azimuth_deg)),
        elevation_deg=float(camera_data.get("elevation_deg", current.elevation_deg)),
        fov_deg=float(camera_data.get("fov_deg", current.fov_deg)),
        near=camera_data.get("near", current.near),
        far=camera_data.get("far", current.far),
    )


def _lighting_from_preset(current: LightingPreset, preset_data: Mapping[str, Any]) -> LightingPreset:
    sun_data = dict(preset_data.get("sun") or {})
    renderer_lighting = dict(preset_data.get("lighting") or {})
    lights = renderer_lighting.get("lights") or ()
    first_light = next((light for light in lights if isinstance(light, Mapping)), {})
    direction = (
        tuple(float(value) for value in current.sun_direction)
        if current.sun_direction is not None
        else _sun_direction_from_preset(sun_data)
        or tuple(float(value) for value in first_light.get("direction", (0.0, 1.0, 0.0)))
    )
    if current.intensity != 1.0:
        intensity = float(current.intensity)
    elif "intensity" in sun_data:
        intensity = float(sun_data["intensity"])
    else:
        intensity = float(first_light.get("intensity", current.intensity))

    renderer_config = {
        key: copy.deepcopy(value)
        for key, value in preset_data.items()
        if key in {"lighting", "shading", "shadows", "gi", "atmosphere", "ibl", "brdf_override"}
    }
    settings = _deep_merge_mapping(
        {
            "resolved_preset": str(current.name),
            "renderer_config": renderer_config,
            "sun": sun_data,
            "ibl": preset_data.get("ibl") or {},
            "camera": preset_data.get("camera") or {},
            "cli_params": preset_data.get("cli_params") or {},
            "exaggeration": preset_data.get("exaggeration"),
        },
        current.settings,
    )
    if current.overrides:
        settings["preset_overrides"] = sorted(str(key) for key in current.overrides.keys())

    return LightingPreset(
        name=str(current.name),
        sun_direction=direction,
        intensity=intensity,
        settings=settings,
        overrides=current.overrides,
    )


def _reproducibility_from_preset(
    current: ReproducibilityProfile | None,
    preset_data: Mapping[str, Any],
) -> ReproducibilityProfile | None:
    repro_data = preset_data.get("reproducibility")
    if not isinstance(repro_data, Mapping):
        return current
    if current is not None:
        return current
    return ReproducibilityProfile(
        seed=int(repro_data.get("seed", 0)),
        renderer_backend=repro_data.get("renderer_backend"),
        pixel_tolerance=repro_data.get("pixel_tolerance"),
    )


def _apply_mapscene_lighting_preset(recipe: SceneRecipe) -> SceneRecipe:
    preset = _mapscene_preset_for_name(recipe.lighting.name)
    if preset is None:
        return recipe
    override_data = dict(recipe.lighting.overrides or {})
    resolved = _deep_merge_mapping(preset, override_data)
    camera = recipe.camera
    if isinstance(resolved.get("camera"), Mapping):
        camera = _camera_from_preset(recipe.camera, recipe.terrain, resolved["camera"])
    lighting = _lighting_from_preset(recipe.lighting, resolved)
    reproducibility_profile = _reproducibility_from_preset(recipe.reproducibility_profile, resolved)
    return SceneRecipe(
        terrain=recipe.terrain,
        camera=camera,
        lighting=lighting,
        layers=recipe.layers,
        output=recipe.output,
        target_crs=recipe.target_crs,
        map_furniture=recipe.map_furniture,
        render_policy=recipe.render_policy,
        diagnostics_policy=recipe.diagnostics_policy,
        reproducibility_profile=reproducibility_profile,
    )


def _alignment_transform_diagnostic(layer_id: str, from_crs: str, to_crs: str) -> Diagnostic:
    return Diagnostic(
        code="alignment_transform_applied",
        severity="info",
        message="Scene asset coordinates were transformed into the MapScene target CRS.",
        remediation="No action required; source_crs and target_crs are recorded in layer metadata.",
        support_level="supported",
        layer_id=layer_id,
        details={"source_crs": from_crs, "target_crs": to_crs},
    )


def _alignment_residual_diagnostic(layer_id: str, residual: Mapping[str, Any]) -> Diagnostic:
    max_residual = float(residual.get("max", residual.get("max_error", 0.0)) or 0.0)
    threshold = float(residual.get("threshold", residual.get("tolerance", 1.0)) or 1.0)
    return Diagnostic(
        code="alignment_residual",
        severity="warning",
        message="Layer alignment residual exceeds the configured tolerance.",
        remediation="Check source CRS, geotransform metadata, and control points before rendering.",
        support_level="underdeveloped",
        layer_id=layer_id,
        details={
            "max": max_residual,
            "threshold": threshold,
            "units": residual.get("units"),
            "source_crs": residual.get("source_crs"),
            "target_crs": residual.get("target_crs"),
        },
    )


def _metadata_resolution(metadata: Mapping[str, Any] | None) -> tuple[float, float] | None:
    data = _metadata_dict(metadata)
    value = data.get("resolution", data.get("pixel_size", data.get("spacing")))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return abs(float(value[0])), abs(float(value[1]))
    if isinstance(value, (int, float)):
        scalar = abs(float(value))
        return scalar, scalar
    if "resolution_x" in data and "resolution_y" in data:
        return abs(float(data["resolution_x"])), abs(float(data["resolution_y"]))
    if "width" in data and "height" in data and "bounds" in data:
        bounds = data.get("bounds")
        if isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)) and len(bounds) == 4:
            width = max(1.0, float(data["width"]))
            height = max(1.0, float(data["height"]))
            return abs(float(bounds[2]) - float(bounds[0])) / width, abs(float(bounds[3]) - float(bounds[1])) / height
    return None


def _resolution_mismatch_diagnostic(
    layer_id: str,
    terrain_resolution: tuple[float, float],
    layer_resolution: tuple[float, float],
) -> Diagnostic:
    return Diagnostic(
        code="resolution_mismatch",
        severity="warning",
        message="Layer resolution differs from the terrain grid resolution.",
        remediation="Resample the layer onto the terrain grid before rendering if exact registration is required.",
        support_level="supported",
        layer_id=layer_id,
        details={
            "terrain_resolution": list(terrain_resolution),
            "layer_resolution": list(layer_resolution),
        },
    )


def _aligned_metadata(metadata: Mapping[str, Any] | None, from_crs: str, to_crs: str) -> dict[str, Any]:
    result = _metadata(metadata)
    result["source_crs"] = from_crs
    result["target_crs"] = to_crs
    result["alignment_transform_applied"] = True
    return result


def _align_label_records(labels: Sequence[Mapping[str, Any]] | None, from_crs: str, to_crs: str) -> list[dict[str, Any]]:
    from .alignment import transform_geometry

    aligned = []
    for label in labels or ():
        item = dict(label)
        geometry = item.get("geometry")
        if isinstance(geometry, Mapping):
            item["geometry"] = transform_geometry(geometry, from_crs, to_crs)
        aligned.append(item)
    return aligned


def _terrain_source_file_crs(path: Path) -> str | None:
    grid = _raster_grid_from_path(path)
    if grid is None:
        return None
    crs = grid.get("crs")
    return None if crs is None else str(crs)


def _align_terrain_to_target(terrain: TerrainSource, target_crs: str) -> tuple[TerrainSource, bool]:
    if terrain.data is not None or not terrain.path:
        return terrain, False
    path = Path(str(terrain.path))
    if not _is_geotiff_path(path) or not path.exists():
        return terrain, False
    source_crs = _terrain_source_file_crs(path) or terrain.crs
    if not source_crs or _same_crs(str(source_crs), str(target_crs)):
        return terrain, False

    from .alignment import reproject_dem_to_target

    result = reproject_dem_to_target(path, str(target_crs), resampling="bilinear")
    metadata = _metadata(terrain.metadata)
    result_metadata = dict(result["metadata"])
    result_metadata["alignment_transform_applied"] = True
    result_metadata["alignment_kind"] = "terrain_reproject"
    result_metadata["source_path"] = str(path)
    metadata.update(result_metadata)
    return (
        TerrainSource(
            path=terrain.path,
            data=result["array"],
            crs=str(target_crs),
            metadata=metadata,
            elevation_sampling_available=terrain.elevation_sampling_available,
            dtype=str(result["array"].dtype),
            nodata_policy=terrain.nodata_policy,
        ),
        True,
    )


def _apply_scene_alignment(recipe: SceneRecipe) -> SceneRecipe:
    target_crs = recipe.target_crs or recipe.terrain.crs
    if not target_crs:
        return recipe
    from .alignment import transform_features

    terrain, terrain_changed = _align_terrain_to_target(recipe.terrain, str(target_crs))
    aligned_layers: list[Any] = []
    changed = terrain_changed
    for layer in recipe.layers:
        layer_crs = getattr(layer, "crs", None)
        if isinstance(layer, VectorOverlay) and layer_crs and not _same_crs(str(layer_crs), str(target_crs)):
            aligned_layers.append(
                VectorOverlay(
                    layer_id=layer.layer_id,
                    path=layer.path,
                    features=transform_features(layer.features or (), str(layer_crs), str(target_crs)),
                    crs=str(target_crs),
                    style=layer.style,
                    width_px=layer.width_px,
                    width_world=layer.width_world,
                    line_join=layer.line_join,
                    line_cap=layer.line_cap,
                    dash_array=layer.dash_array,
                    style_support=layer.style_support,
                    metadata=_aligned_metadata(layer.metadata, str(layer_crs), str(target_crs)),
                )
            )
            changed = True
            continue
        if isinstance(layer, LabelLayer):
            metadata = _metadata_dict(layer.metadata)
            label_crs = metadata.get("crs")
            if label_crs and not _same_crs(str(label_crs), str(target_crs)):
                aligned_layers.append(
                    LabelLayer(
                        layer_id=layer.layer_id,
                        labels=_align_label_records(layer.labels, str(label_crs), str(target_crs)),
                        glyph_atlas=layer.glyph_atlas,
                        atlas=layer.atlas,
                        typography=layer.typography,
                        occlusion=layer.occlusion,
                        priority_rules=layer.priority_rules,
                        plan=layer.plan,
                        metadata=_aligned_metadata(layer.metadata, str(label_crs), str(target_crs)),
                        diagnostics=layer.diagnostics,
                    )
                )
                changed = True
                continue
        aligned_layers.append(layer)
    if not changed and recipe.target_crs is None:
        return recipe
    return SceneRecipe(
        terrain=terrain,
        camera=recipe.camera,
        lighting=recipe.lighting,
        layers=tuple(aligned_layers),
        output=recipe.output,
        target_crs=str(target_crs),
        map_furniture=recipe.map_furniture,
        render_policy=recipe.render_policy,
        diagnostics_policy=recipe.diagnostics_policy,
        reproducibility_profile=recipe.reproducibility_profile,
    )


def _preset_override_diagnostic(lighting: LightingPreset) -> Diagnostic | None:
    if not lighting.overrides:
        return None
    return Diagnostic(
        code="preset_override",
        severity="info",
        message="MapScene lighting preset fields were explicitly overridden.",
        remediation="Review LightingPreset.overrides if the resolved camera or lighting differs from the named preset.",
        support_level="supported",
        layer_id="scene",
        details={"preset": str(lighting.name), "fields": sorted(str(key) for key in lighting.overrides.keys())},
    )


class MapScene:
    def __init__(
        self,
        recipe: SceneRecipe | None = None,
        *,
        terrain: TerrainSource | None = None,
        camera: OrbitCamera | None = None,
        lighting: LightingPreset | None = None,
        layers: Sequence[Any] | None = None,
        output: OutputSpec | None = None,
        target_crs: str | None = None,
        map_furniture: MapFurnitureLayer | None = None,
        render_policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING,
        diagnostics_policy: Mapping[str, Any] | None = None,
        reproducibility_profile: ReproducibilityProfile | None = None,
    ) -> None:
        if recipe is not None and any(
            value is not None
            for value in (
                terrain,
                camera,
                lighting,
                layers,
                output,
                target_crs,
                map_furniture,
                diagnostics_policy,
                reproducibility_profile,
            )
        ):
            raise TypeError("Pass either recipe or recipe keyword components, not both")
        if recipe is None:
            if terrain is None or lighting is None or output is None:
                raise TypeError("terrain, lighting, and output are required when recipe is not provided")
            recipe = SceneRecipe(
                terrain=terrain,
                camera=camera or OrbitCamera(),
                lighting=lighting,
                layers=layers or (),
                output=output,
                target_crs=target_crs,
                map_furniture=map_furniture,
                render_policy=render_policy,
                diagnostics_policy=diagnostics_policy,
                reproducibility_profile=reproducibility_profile,
            )
        recipe = _apply_mapscene_lighting_preset(recipe)
        recipe = _apply_scene_alignment(recipe)
        self.recipe = recipe
        self.render_policy = recipe.render_policy
        self.reproducibility_profile = recipe.reproducibility_profile
        self.last_validation_report: ValidationReport | None = None
        self.compiled_label_plans: dict[str, Any] = {}
        self.compiled_plan: CompiledScenePlan | None = None
        self.last_render_path: str | None = None
        self.last_render_backend: str | None = None
        self.last_render_metadata: dict[str, Any] | None = None
        self.last_bundle_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "map_scene", "recipe": self.recipe.to_dict()}

    def alignment_report(self) -> dict[str, Any]:
        from .alignment import alignment_report

        return alignment_report(self)

    def material_vt_stats(self) -> dict[str, float] | None:
        """Terrain virtual-texture residency stats from the most recent render.

        Returns the native ``get_material_vt_stats()`` dict captured in
        ``last_render_metadata``, including the per-family fields
        (``resident_tiles_albedo``, ``resident_bytes_normal``,
        ``budget_bytes_mask``, ...), or ``None`` when no render has run or the
        native renderer was not used.
        """
        metadata = self.last_render_metadata or {}
        stats = metadata.get("material_vt_stats")
        if not isinstance(stats, Mapping):
            return None
        return {str(key): float(value) for key, value in stats.items()}

    @staticmethod
    def _layer_from_dict(data: Mapping[str, Any]) -> Any:
        kind = str(data.get("kind", ""))
        if kind == "raster_overlay":
            return RasterOverlay(
                layer_id=str(data["layer_id"]),
                path=data.get("path"),
                crs=data.get("crs"),
                opacity=float(data.get("opacity", 1.0)),
                metadata=data.get("metadata") or {},
            )
        if kind == "vector_overlay":
            return VectorOverlay(
                layer_id=str(data["layer_id"]),
                path=data.get("path"),
                    features=data.get("features") or (),
                    crs=data.get("crs"),
                    style=data.get("style") or {},
                    width_px=data.get("width_px"),
                    width_world=data.get("width_world"),
                    line_join=str(data.get("line_join", "miter")),
                    line_cap=str(data.get("line_cap", "butt")),
                    dash_array=data.get("dash_array"),
                    style_support=data.get("style_support") or {},
                    metadata=data.get("metadata") or {},
                )
        if kind == "label_layer":
                return LabelLayer(
                    layer_id=str(data["layer_id"]),
                    labels=data.get("labels") or (),
                    glyph_atlas=data.get("glyph_atlas") or {},
                    atlas=data.get("atlas"),
                    typography=data.get("typography") or {},
                    occlusion=str(data.get("occlusion", "terrain")),
                    priority_rules=data.get("priority_rules") or (),
                    plan=data.get("plan"),
                    metadata=data.get("metadata") or {},
                diagnostics=data.get("diagnostics") or (),
            )
        if kind == "point_cloud_layer":
            return PointCloudLayer(
                layer_id=str(data["layer_id"]),
                path=data.get("path"),
                crs=data.get("crs"),
                point_count=data.get("point_count"),
                support_level=str(data.get("support_level", "native-required")),
                metadata=data.get("metadata") or {},
            )
        if kind == "building_layer":
            return BuildingLayer(
                layer_id=str(data["layer_id"]),
                source=data.get("source"),
                support_level=str(data.get("support_level", "underdeveloped")),
                geometry_count=data.get("geometry_count"),
                bounds=data.get("bounds") or None,
                material_status=data.get("material_status"),
                features=data.get("features") or (),
                metadata=data.get("metadata") or {},
            )
        if kind == "tiles3d_layer":
            return Tiles3DLayer(
                layer_id=str(data["layer_id"]),
                source=data.get("source"),
                support_level=str(data.get("support_level", "underdeveloped")),
                lod=data.get("lod") or {},
                cache_budget=data.get("cache_budget"),
                cache_stats=data.get("cache_stats") or {},
                metadata=data.get("metadata") or {},
                diagnostics=data.get("diagnostics") or (),
            )
        return dict(data)

    @classmethod
    def _recipe_from_dict(cls, data: Mapping[str, Any]) -> SceneRecipe:
        terrain_data = data.get("terrain") or {}
        camera_data = data.get("camera") or {}
        lighting_data = data.get("lighting") or {}
        output_data = data.get("output") or {}
        furniture_data = data.get("map_furniture")
        reproducibility_data = data.get("reproducibility_profile")
        return SceneRecipe(
            terrain=TerrainSource(
                path=terrain_data.get("path"),
                crs=terrain_data.get("crs"),
                metadata=terrain_data.get("metadata") or {},
                elevation_sampling_available=bool(terrain_data.get("elevation_sampling_available", False)),
                dtype=str(terrain_data.get("dtype", "float32")),
                nodata_policy=str(terrain_data.get("nodata_policy", "fill")),
            ),
            camera=OrbitCamera(
                target=camera_data.get("target") or (0.0, 0.0, 0.0),
                distance=float(camera_data.get("distance", 1.0)),
                azimuth_deg=float(camera_data.get("azimuth_deg", 0.0)),
                elevation_deg=float(camera_data.get("elevation_deg", 45.0)),
                fov_deg=float(camera_data.get("fov_deg", 45.0)),
                near=camera_data.get("near"),
                far=camera_data.get("far"),
            ),
            lighting=LightingPreset(
                name=str(lighting_data.get("name", "default")),
                sun_direction=lighting_data.get("sun_direction"),
                intensity=float(lighting_data.get("intensity", 1.0)),
                settings=lighting_data.get("settings") or {},
                overrides=lighting_data.get("overrides") or {},
            ),
            layers=tuple(cls._layer_from_dict(layer) for layer in data.get("layers") or ()),
            output=OutputSpec(
                width=int(output_data.get("width", 1)),
                height=int(output_data.get("height", 1)),
                format=str(output_data.get("format", "png")),
                path=output_data.get("path"),
                samples=int(output_data.get("samples", 1)),
                denoiser=str(output_data.get("denoiser", "none")),
                aovs=tuple(output_data.get("aovs") or ()),
                hdr=bool(output_data.get("hdr", False)),
                bit_depth=int(output_data.get("bit_depth", 8)),
                metadata=output_data.get("metadata") or {},
            ),
            target_crs=data.get("target_crs"),
            map_furniture=(
                MapFurnitureLayer(
                    title=furniture_data.get("title"),
                    legend=furniture_data.get("legend"),
                    scale_bar=furniture_data.get("scale_bar"),
                    north_arrow=furniture_data.get("north_arrow"),
                    graticule=furniture_data.get("graticule"),
                    keepouts=furniture_data.get("keepouts") or (),
                    metadata=furniture_data.get("metadata") or {},
                )
                if isinstance(furniture_data, Mapping)
                else None
            ),
            render_policy=str(data.get("render_policy", RenderFailurePolicy.CONTINUE_ON_WARNING)),
            diagnostics_policy=data.get("diagnostics_policy") or {},
            reproducibility_profile=(
                ReproducibilityProfile(
                    seed=int(reproducibility_data.get("seed", 0)),
                    camera=reproducibility_data.get("camera") or {},
                    output_size=reproducibility_data.get("output_size") or (),
                    terrain_transform=reproducibility_data.get("terrain_transform") or {},
                    style_hashes=reproducibility_data.get("style_hashes") or {},
                    asset_hashes_or_ids=reproducibility_data.get("asset_hashes_or_ids") or {},
                    renderer_backend=reproducibility_data.get("renderer_backend"),
                    pixel_tolerance=reproducibility_data.get("pixel_tolerance"),
                )
                if isinstance(reproducibility_data, Mapping)
                else None
            ),
        )

    @classmethod
    def load_bundle(cls, path: str | Path) -> "MapScene":
        bundle_path = Path(path)
        manifest_path = bundle_path / "manifest.json"
        if manifest_path.exists():
            from .bundle import BundleManifest

            # Enforces the bundle version contract: version > BUNDLE_VERSION
            # raises ValueError instead of silently loading a future schema.
            BundleManifest.load(manifest_path)
        recipe_path = bundle_path / "scene" / "mapscene_recipe.json"
        if not recipe_path.exists():
            raise FileNotFoundError(f"MapScene bundle recipe not found: {recipe_path}")
        with recipe_path.open("r", encoding="utf-8") as handle:
            recipe_payload = json.load(handle)
        scene = cls(recipe=cls._recipe_from_dict(recipe_payload))
        scene.last_bundle_path = str(bundle_path)
        compiled_path = bundle_path / "scene" / "compiled_plan.json"
        if compiled_path.exists():
            from .recipe_manifest import load_manifest

            scene._rehydrate_compiled_plan(load_manifest(compiled_path))
        else:
            # v2 read path (BUNDLE_VERSION < 3): no frozen compiled plan on
            # disk — recompile once from the serialized recipe.
            scene.compile_plan()
        state_path = bundle_path / "scene" / "state.json"
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as handle:
                state_payload = json.load(handle)
            report_payload = state_payload.get("validation_report")
            if isinstance(report_payload, Mapping):
                scene.last_validation_report = ValidationReport.from_dict(report_payload)
        return scene

    def validate(self) -> ValidationReport:
        self.compiled_label_plans = {}
        diagnostics: list[Diagnostic] = []
        layer_summaries: list[LayerSummary] = []
        supported_features: dict[str, str] = {
            "mapscene.recipe": "underdeveloped",
            "mapscene.validation": "supported",
        }
        unsupported_features: dict[str, str] = {}
        total_memory = 0
        memory_known = False

        preset_override = _preset_override_diagnostic(self.recipe.lighting)
        if preset_override is not None:
            diagnostics.append(preset_override)
        lighting_settings = _metadata_dict(self.recipe.lighting.settings)
        if lighting_settings.get("resolved_preset"):
            supported_features["mapscene.presets"] = "supported"
            supported_features["mapscene.lighting_preset"] = "supported"

        output = self.recipe.output
        if output is not None and str(output.format).lower() not in {"png", "exr"}:
            diagnostics.append(_unsupported_output_format_diagnostic(str(output.format).lower()))
            unsupported_features["output.format"] = "unsupported"
        elif output is not None:
            supported_features[f"output.format.{str(output.format).lower()}"] = "supported"
            supported_features[f"output.bit_depth.{int(output.bit_depth)}"] = "supported"
            if int(output.samples) > 1:
                supported_features["mapscene.offline_accumulation"] = "supported"
            if output.aovs:
                supported_features["mapscene.aov_export"] = "supported"
            if bool(output.hdr) or str(output.format).lower() == "exr":
                supported_features["mapscene.hdr_output"] = "supported"

        furniture = self.recipe.map_furniture
        if furniture is not None:
            supported_features["mapscene.furniture"] = "supported"
            if furniture.graticule is not None:
                supported_features["mapscene.graticule"] = "supported"
            if furniture.scale_bar is not None:
                supported_features["mapscene.scale_bar"] = "supported"
                scale_bar_options = _metadata_dict(furniture.scale_bar)
                if bool(scale_bar_options.get("geodesic", True)):
                    supported_features["mapscene.scale_bar.geodesic"] = "supported"
            if furniture.north_arrow is not None:
                supported_features["mapscene.north_arrow"] = "supported"
            if furniture.legend is not None:
                supported_features["mapscene.legend"] = "underdeveloped"

        output_bytes = _output_memory_bytes(output)
        if output_bytes is not None:
            total_memory += output_bytes
            memory_known = True

        terrain = self.recipe.terrain
        scene_crs = self.recipe.target_crs or terrain.crs
        terrain_metadata = _metadata_dict(terrain.metadata)
        terrain_memory = _dimension_memory_bytes(terrain.metadata)
        if terrain_memory is not None:
            total_memory += terrain_memory
            memory_known = True

        vt_report = _virtual_texture_report_from_metadata(terrain.metadata)
        if vt_report is not None:
            _merge_report(
                vt_report,
                diagnostics=diagnostics,
                layer_summaries=layer_summaries,
                supported_features=supported_features,
                unsupported_features=unsupported_features,
            )

        terrain_support_level = "supported"
        terrain_details = {
            "crs": scene_crs,
            "elevation_sampling_available": bool(terrain.elevation_sampling_available),
            "path": _path_to_str(terrain.path),
        }
        terrain_resource_diagnostics, terrain_resource_details, terrain_resource_support = (
            _p2_resource_availability_diagnostics(
                terrain.metadata,
                layer_id="terrain",
                layer_type="terrain_source",
            )
        )
        if terrain_resource_diagnostics:
            diagnostics.extend(terrain_resource_diagnostics)
            terrain_details.update(terrain_resource_details)
            if any(diagnostic.code == "unavailable_cache_lod_stats" for diagnostic in terrain_resource_diagnostics):
                unsupported_features["terrain.cache_lod_stats"] = "underdeveloped"
            if terrain_resource_support == "underdeveloped":
                terrain_support_level = "underdeveloped"
        if not _has_identity_path_or_metadata(terrain.path, terrain.metadata):
            diagnostics.append(
                _missing_source_identity_diagnostic(
                    "terrain_source",
                    layer_id="terrain",
                    source_fields=("path", "metadata"),
                )
            )
            unsupported_features["terrain.source_identity"] = "unsupported"
            terrain_support_level = "unsupported"
        else:
            terrain_asset_diagnostics = _asset_path_diagnostics(
                "terrain_source",
                layer_id="terrain",
                path=terrain.path,
                metadata=terrain.metadata,
                supported_extensions=_TERRAIN_ASSET_EXTENSIONS,
            )
            if terrain_asset_diagnostics:
                diagnostics.extend(terrain_asset_diagnostics)
                unsupported_features["terrain.asset"] = "unsupported"
                terrain_support_level = "unsupported"
        if not scene_crs:
            diagnostics.append(_missing_crs_diagnostic(None, layer_id="terrain"))
            unsupported_features["terrain.crs"] = "unsupported"
            terrain_support_level = "unsupported"
        if terrain_metadata.get("alignment_transform_applied"):
            source_crs = str(terrain_metadata.get("source_crs") or terrain.crs or "")
            target = str(terrain_metadata.get("target_crs") or scene_crs or "")
            diagnostics.append(_alignment_transform_diagnostic("terrain", source_crs, target))
            supported_features["mapscene.alignment"] = "supported"

        layer_summaries.append(
            LayerSummary(
                layer_id="terrain",
                layer_type="terrain_source",
                support_level=terrain_support_level,
                diagnostic_codes=_diagnostic_codes_for_layer(diagnostics, "terrain"),
                memory_estimate_bytes=terrain_memory,
                details=terrain_details,
            )
        )
        supported_features["layer.terrain"] = "supported"
        if self.recipe.target_crs:
            supported_features["mapscene.alignment"] = "supported"

        for index, layer in enumerate(self.recipe.layers):
            layer_id = _layer_id(layer, f"layer_{index}")
            layer_diagnostics: list[Diagnostic] = []
            layer_memory: int | None = None
            support_level = "supported"
            layer_type = type(layer).__name__
            object_count: int | None = None
            details: dict[str, Any] = {}
            layer_metadata = _metadata_dict(getattr(layer, "metadata", None))
            terrain_resolution = _metadata_resolution(terrain.metadata)
            layer_resolution = _metadata_resolution(layer_metadata)
            if terrain_resolution is not None and layer_resolution is not None:
                rel_x = abs(layer_resolution[0] - terrain_resolution[0]) / max(abs(terrain_resolution[0]), 1.0e-12)
                rel_y = abs(layer_resolution[1] - terrain_resolution[1]) / max(abs(terrain_resolution[1]), 1.0e-12)
                if max(rel_x, rel_y) > 1.0e-6:
                    layer_diagnostics.append(
                        _resolution_mismatch_diagnostic(layer_id, terrain_resolution, layer_resolution)
                    )
            residual = layer_metadata.get("alignment_residual")
            if isinstance(residual, Mapping):
                max_residual = float(residual.get("max", residual.get("max_error", 0.0)) or 0.0)
                threshold = float(residual.get("threshold", residual.get("tolerance", 1.0)) or 1.0)
                if max_residual > threshold:
                    layer_diagnostics.append(_alignment_residual_diagnostic(layer_id, residual))
            if layer_metadata.get("alignment_transform_applied"):
                source_crs = str(layer_metadata.get("source_crs"))
                target = str(layer_metadata.get("target_crs") or scene_crs)
                layer_diagnostics.append(_alignment_transform_diagnostic(layer_id, source_crs, target))
                supported_features["mapscene.alignment"] = "supported"

            layer_crs = getattr(layer, "crs", None)
            has_crs_field = hasattr(layer, "crs")
            if scene_crs and has_crs_field and layer_crs is None and not _has_explicit_crs_policy(layer):
                layer_diagnostics.append(_missing_crs_diagnostic(str(scene_crs), layer_id=layer_id))
                support_level = "unsupported"
            elif scene_crs and layer_crs and not _same_crs(scene_crs, str(layer_crs)) and not _has_explicit_crs_policy(layer):
                layer_diagnostics.append(
                    crs_mismatch_diagnostic(
                        str(scene_crs),
                        str(layer_crs),
                        layer_id=layer_id,
                    )
                )
                support_level = "unsupported"

            if isinstance(layer, RasterOverlay):
                layer_type = "raster_overlay"
                layer_memory = _dimension_memory_bytes(layer.metadata)
                details = {"path": _path_to_str(layer.path), "crs": layer.crs, "opacity": float(layer.opacity)}
                supported_features["layer.raster_overlay"] = "supported"
                if not _has_identity_path_or_metadata(layer.path, layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("path", "metadata"),
                        )
                    )
                    unsupported_features["raster.source_identity"] = "unsupported"
                    support_level = "unsupported"
                else:
                    layer_diagnostics.extend(
                        _asset_path_diagnostics(
                            layer_type,
                            layer_id=layer_id,
                            path=layer.path,
                            metadata=layer.metadata,
                            supported_extensions=_RASTER_ASSET_EXTENSIONS,
                        )
                    )
                    if any(
                        diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                        for diagnostic in layer_diagnostics
                    ):
                        unsupported_features["raster.asset"] = "unsupported"
                        support_level = "unsupported"
            elif isinstance(layer, VectorOverlay):
                layer_type = "vector_overlay"
                object_count = len(layer.features or ())
                details = {
                    "path": _path_to_str(layer.path),
                    "crs": layer.crs,
                    "feature_count": object_count,
                    "width_px": layer.width_px,
                    "width_world": layer.width_world,
                    "line_join": layer.line_join,
                    "line_cap": layer.line_cap,
                    "dash_array": _sequence(layer.dash_array),
                }
                supported_features["layer.vector_overlay"] = "supported"
                supported_features["vector.stroke.joins_caps"] = "supported"
                if layer.dash_array:
                    supported_features["vector.stroke.dashes"] = "supported"
                if layer.width_px is not None:
                    supported_features["vector.stroke.width_px"] = "supported"
                if layer.width_world is not None:
                    supported_features["vector.stroke.width_world"] = "supported"
                if not layer.path and not layer.features:
                    layer_diagnostics.append(
                        _missing_renderable_data_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            required_any_of=("path", "features"),
                        )
                    )
                    unsupported_features["vector.renderable_data"] = "unsupported"
                    support_level = "unsupported"
                elif layer.path and not layer.features:
                    layer_diagnostics.append(placeholder_fallback_diagnostic("vector path loader", layer_id=layer_id))
                    unsupported_features["vector.path_loader"] = "placeholder/fallback"
                    support_level = "placeholder/fallback"
                if layer.style:
                    for style_layer in (layer.style.get("layers", ()) if isinstance(layer.style, Mapping) else ()):
                        if not isinstance(style_layer, Mapping) or str(style_layer.get("type", "")).lower() != "line":
                            continue
                        paint = style_layer.get("paint") if isinstance(style_layer.get("paint"), Mapping) else {}
                        layout = style_layer.get("layout") if isinstance(style_layer.get("layout"), Mapping) else {}
                        if paint.get("line-dasharray"):
                            supported_features["vector.stroke.dashes"] = "supported"
                        if layout.get("line-cap") or layout.get("line-join"):
                            supported_features["vector.stroke.joins_caps"] = "supported"
                    from .style import validate_style_support

                    style_report = validate_style_support(dict(layer.style))
                    _merge_report(
                        style_report,
                        diagnostics=diagnostics,
                        layer_summaries=layer_summaries,
                        supported_features=supported_features,
                        unsupported_features=unsupported_features,
                    )
                    if style_report.status in {"error", "fatal"}:
                        support_level = "unsupported"
                    elif style_report.status == "warning" and support_level == "supported":
                        support_level = "underdeveloped"
            elif isinstance(layer, LabelLayer):
                layer_type = "label_layer"
                labels = list(layer.labels or ())
                object_count = len(labels)
                details = {"label_count": object_count, "occlusion": str(layer.occlusion)}
                supported_features["layer.label_layer"] = "supported"
                if layer.occlusion == "terrain":
                    supported_features["labels.occlusion.terrain"] = "supported"
                    layer_metadata = _metadata_dict(layer.metadata)
                    terrain_metadata = _metadata_dict(terrain.metadata)
                    if any(
                        key in layer_metadata or key in terrain_metadata
                        for key in ("depth_occlusion", "depth_aov", "depth_buffer", "depth_image")
                    ):
                        supported_features["labels.occlusion.depth_aov"] = "supported"
                else:
                    supported_features["labels.occlusion.none"] = "supported"
                layer_diagnostics.extend(
                    diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
                    for diagnostic in (layer.diagnostics or ())
                )
                if not _has_labels_or_plan(layer):
                    layer_diagnostics.append(
                        _missing_renderable_data_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            required_any_of=("labels", "plan"),
                        )
                    )
                    unsupported_features["labels.renderable_data"] = "unsupported"
                    support_level = "unsupported"
                label_report = validate_label_support(
                    labels,
                    atlas_glyphs=_atlas_glyph_set(layer.glyph_atlas),
                    layer_id=layer_id,
                )
                _merge_report(
                    label_report,
                    diagnostics=diagnostics,
                    layer_summaries=layer_summaries,
                    supported_features=supported_features,
                    unsupported_features=unsupported_features,
                )
                if _has_labels_or_plan(layer):
                    plan = _label_plan_from_layer(layer, recipe=self.recipe, terrain=terrain)
                    self.compiled_label_plans[layer_id] = plan
                    plan_diagnostics = [_diagnostic_for_layer(diagnostic, layer_id) for diagnostic in plan.diagnostics]
                    existing_diagnostic_keys = {
                        (diagnostic.code, diagnostic.layer_id, diagnostic.object_id)
                        for diagnostic in diagnostics
                    }
                    plan_diagnostics = [
                        diagnostic
                        for diagnostic in plan_diagnostics
                        if (diagnostic.code, diagnostic.layer_id, diagnostic.object_id) not in existing_diagnostic_keys
                    ]
                    diagnostics.extend(plan_diagnostics)
                    details["compiled_label_plan"] = {
                        "accepted_count": len(plan.accepted),
                        "rejected_count": len(plan.rejected),
                        "diagnostic_codes": sorted({diagnostic.code for diagnostic in plan_diagnostics}),
                        "seed": plan.seed,
                    }
                    if any(diagnostic.severity in {"error", "fatal"} for diagnostic in plan_diagnostics):
                        support_level = "underdeveloped"
                    elif support_level == "supported":
                        support_level = "supported"
            elif isinstance(layer, PointCloudLayer):
                layer_type = "point_cloud_layer"
                object_count = layer.point_count
                layer_memory = _point_cloud_memory_bytes(layer.point_count)
                details = {"path": _path_to_str(layer.path), "crs": layer.crs, "point_count": layer.point_count}
                has_renderable_points = "positions" in _metadata_dict(layer.metadata) or bool(layer.path)
                supported_features["layer.point_cloud"] = "supported" if has_renderable_points else "underdeveloped"
                if has_renderable_points:
                    supported_features["point_cloud.mapscene_render"] = "supported"
                support_level = "supported" if has_renderable_points else "underdeveloped"
                if not layer.path and layer.point_count is None and not _metadata_dict(layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("path", "point_count", "metadata"),
                        )
                    )
                    unsupported_features["point_cloud.source_identity"] = "unsupported"
                    support_level = "unsupported"
                else:
                    layer_diagnostics.extend(
                        _asset_path_diagnostics(
                            layer_type,
                            layer_id=layer_id,
                            path=layer.path,
                            metadata=layer.metadata,
                            supported_extensions=_POINT_CLOUD_ASSET_EXTENSIONS,
                        )
                    )
                    if any(
                        diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                        for diagnostic in layer_diagnostics
                    ):
                        unsupported_features["point_cloud.asset"] = "unsupported"
                        support_level = "unsupported"
                if support_level == "underdeveloped":
                    layer_diagnostics.append(
                        placeholder_fallback_diagnostic("point cloud MapScene render path", layer_id=layer_id)
                    )
                    unsupported_features["point_cloud.mapscene_render"] = "placeholder/fallback"
                    support_level = "placeholder/fallback"
            elif isinstance(layer, Tiles3DLayer):
                layer_type = "tiles3d_layer"
                object_count = None
                layer_memory = _explicit_memory_bytes(layer.metadata)
                source_path = _source_path(layer.source)
                source_kind = _source_kind(layer.source, layer.metadata)
                details = {
                    "source_kind": source_kind,
                    "cache_budget": layer.cache_budget,
                    "cache_stats": _metadata(layer.cache_stats),
                    "lod": _metadata(layer.lod),
                }
                supported_features["layer.tiles3d_intent"] = "underdeveloped"
                support_level = layer.support_level
                layer_diagnostics.extend(
                    diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
                    for diagnostic in (layer.diagnostics or ())
                )
                if layer.source is None and not _metadata_dict(layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("source", "metadata"),
                        )
                    )
                    unsupported_features["tiles3d.source_identity"] = "unsupported"
                    support_level = "unsupported"
                else:
                    if source_path:
                        if not source_path.lower().endswith(("tileset.json", ".b3dm", ".pnts")):
                            layer_diagnostics.append(
                                unsupported_tile_format_diagnostic(
                                    Path(source_path).suffix.lstrip(".") or source_kind or "unknown",
                                    layer_id=layer_id,
                                    object_id=source_path,
                                )
                            )
                            unsupported_features["tiles3d.format"] = "unsupported"
                            support_level = "unsupported"
                        layer_diagnostics.extend(
                            _asset_path_diagnostics(
                                layer_type,
                                layer_id=layer_id,
                                path=source_path,
                                metadata=layer.metadata,
                                supported_extensions=("tileset.json", ".b3dm", ".pnts"),
                            )
                        )
                    for feature in _metadata_dict(layer.metadata).get("unsupported_features", ()) or ():
                        layer_diagnostics.append(
                            unsupported_tile_feature_diagnostic(str(feature), layer_id=layer_id)
                        )
                        unsupported_features["tiles3d.feature"] = "unsupported"
                        support_level = "unsupported"
                    if support_level != "unsupported" and _tiles3d_layer_has_native_geometry(layer):
                        supported_features["layer.tiles3d_intent"] = "supported"
                        supported_features["tiles3d.mapscene_render"] = "supported"
                        support_level = "supported"
                    elif support_level != "unsupported":
                        layer_diagnostics.append(python_public_3dtiles_incomplete_diagnostic(layer_id=layer_id))
                        unsupported_features["tiles3d.public_python_render"] = "underdeveloped"
                        support_level = "underdeveloped"
            elif isinstance(layer, BuildingLayer):
                layer_type = "building_layer"
                object_count = layer.geometry_count
                layer_memory = _geometry_memory_bytes(layer.geometry_count)
                source_kind = _source_kind(layer.source, layer.metadata)
                support_level = layer.support_level
                details = {
                    "geometry_count": layer.geometry_count,
                    "material_status": layer.material_status,
                    "source_kind": source_kind,
                }
                scalar_building = str(layer.material_status or "").lower().startswith("scalar")
                supported_features["layer.building_intent"] = "supported" if scalar_building else "underdeveloped"
                if scalar_building:
                    supported_features["buildings.scalar_materials"] = "supported"
                    supported_features["buildings.mapscene_render"] = "supported"
                if layer.source is None and not _metadata_dict(layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("source", "metadata"),
                        )
                    )
                    unsupported_features["buildings.source_identity"] = "unsupported"
                    support_level = "unsupported"
                elif (
                    layer.support_level == "supported"
                    or (layer.support_level == "underdeveloped" and bool(layer.features))
                ) and scalar_building:
                    layer_diagnostics.extend(
                        _asset_path_diagnostics(
                            layer_type,
                            layer_id=layer_id,
                            path=_source_path(layer.source),
                            metadata=layer.metadata,
                            supported_extensions=_BUILDING_ASSET_EXTENSIONS,
                        )
                    )
                    if any(
                        diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                        for diagnostic in layer_diagnostics
                    ):
                        unsupported_features["buildings.asset"] = "unsupported"
                        support_level = "unsupported"
                    else:
                        support_level = "supported"
                if source_kind == "3dtiles":
                    layer_diagnostics.append(python_public_3dtiles_incomplete_diagnostic(layer_id=layer_id))
                    unsupported_features["tiles3d.public_python_render"] = "underdeveloped"
                    support_level = "underdeveloped"
                if layer.support_level == "Pro-gated":
                    layer_diagnostics.append(pro_gated_path_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.pro_gated_path"] = "Pro-gated"
                if str(layer.material_status or "").lower() in {"textured_pbr_unsupported", "textured pbr unsupported"}:
                    layer_diagnostics.append(
                        _unsupported_feature_diagnostic("building textured PBR", layer_id=layer_id)
                    )
                    unsupported_features["buildings.textured_pbr"] = "unsupported"
                elif layer.support_level == "placeholder/fallback" or (
                    layer.geometry_count == 0 and layer.support_level not in {"experimental", "underdeveloped"}
                ):
                    layer_diagnostics.append(placeholder_fallback_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.placeholder_fallback"] = "placeholder/fallback"
                    support_level = "placeholder/fallback"
                elif layer.support_level == "experimental":
                    layer_diagnostics.append(experimental_feature_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.experimental"] = "experimental"
                elif layer.support_level == "unsupported":
                    layer_diagnostics.append(_unsupported_feature_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.unsupported"] = "unsupported"
                texture_diagnostics, texture_details, texture_support = _p2_building_texture_diagnostics(
                    layer.metadata,
                    layer_id=layer_id,
                )
                if texture_details:
                    details.update(texture_details)
                if texture_diagnostics:
                    layer_diagnostics.extend(texture_diagnostics)
                    if any(diagnostic.code == "missing_texture_path" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.missing_texture_path"] = "unsupported"
                    if any(diagnostic.code == "missing_uvs" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.missing_uvs"] = "unsupported"
                    if any(diagnostic.code == "unsupported_texture_format" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.unsupported_texture_format"] = "unsupported"
                    if any(diagnostic.code == "placeholder_fallback" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.textured_material_fallback"] = "placeholder/fallback"
                    if any(diagnostic.code == "unsupported_feature" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.textured_pbr"] = "unsupported"
                    if texture_support == "unsupported":
                        support_level = "unsupported"
                    elif texture_support == "placeholder/fallback" and support_level not in {"unsupported", "Pro-gated"}:
                        support_level = "placeholder/fallback"
                elif texture_support == "supported":
                    supported_features["buildings.textured_pbr"] = "supported"
            else:
                layer_diagnostics.append(_unsupported_layer_type_diagnostic(layer, layer_id=layer_id))
                unsupported_features["layer.unknown"] = "unsupported"
                support_level = "unsupported"
                details = {"python_type": type(layer).__name__}

            resource_diagnostics, resource_details, resource_support = _p2_resource_availability_diagnostics(
                getattr(layer, "metadata", None),
                layer_id=layer_id,
                layer_type=layer_type,
            )
            if resource_diagnostics:
                layer_diagnostics.extend(resource_diagnostics)
                details.update(resource_details)
                if any(diagnostic.code == "unavailable_cache_lod_stats" for diagnostic in resource_diagnostics):
                    unsupported_features[f"{layer_type}.cache_lod_stats"] = "underdeveloped"
                if any(diagnostic.code == "unsupported_instancing_path" for diagnostic in resource_diagnostics):
                    unsupported_features[f"{layer_type}.instancing"] = "unsupported"
                if resource_support == "unsupported":
                    support_level = "unsupported"
                elif resource_support == "underdeveloped" and support_level == "supported":
                    support_level = "underdeveloped"

            if layer_memory is not None:
                total_memory += layer_memory
                memory_known = True
            if any(diagnostic.code in {"crs_mismatch", "missing_crs"} for diagnostic in layer_diagnostics):
                support_level = "unsupported"

            diagnostics.extend(layer_diagnostics)
            layer_summaries.append(
                LayerSummary(
                    layer_id=layer_id,
                    layer_type=layer_type,
                    support_level=support_level,
                    diagnostic_codes=_diagnostic_codes_for_layer(diagnostics, layer_id),
                    object_count=object_count,
                    bounds=getattr(layer, "bounds", None),
                    memory_estimate_bytes=layer_memory,
                    details=details,
                )
            )

        estimated_memory = total_memory if memory_known else None
        budget = None
        diagnostics_policy = _metadata_dict(self.recipe.diagnostics_policy)
        if "gpu_memory_budget_bytes" in diagnostics_policy:
            try:
                budget = int(diagnostics_policy["gpu_memory_budget_bytes"])
            except (TypeError, ValueError):
                budget = None
        if estimated_memory is not None and budget is not None:
            diagnostics.append(
                estimated_gpu_memory_diagnostic(
                    estimated_memory,
                    budget,
                    layer_id="scene",
                )
            )

        if bool(diagnostics_policy.get("large_scene_summary")):
            layer_summaries.append(
                _large_scene_resource_summary(
                    layer_summaries,
                    estimated_memory=estimated_memory,
                    diagnostics=diagnostics,
                )
            )

        report = ValidationReport(
            diagnostics=diagnostics,
            layer_summaries=layer_summaries,
            estimated_gpu_memory_bytes=estimated_memory,
            supported_features=supported_features,
            unsupported_features=unsupported_features,
        )
        self.last_validation_report = report
        return report

    def compile_plan(self) -> CompiledScenePlan:
        """Resolve and freeze the render plan from serialized inputs only.

        Runs BEFORE any drawing and BEFORE serialization: label placements
        are resolved, depth-occlusion culling is evaluated against the
        deterministic CPU camera/terrain proxy (never a live GPU frame), and
        the resulting label set, decluttering decisions, and per-label
        visibility flags are frozen into a ``RecipeManifest``. The output is
        a total function of the serialized recipe, so a reloaded bundle
        reproduces the identical cull.
        """
        from types import MappingProxyType

        from .recipe_manifest import RecipeManifest

        report = self.validate()
        recipe_payload = self.recipe.to_dict()
        recipe_hash = _stable_hash(recipe_payload)
        camera_terrain_key = _stable_hash(
            {
                "camera": self.recipe.camera.to_dict(),
                "terrain": _json_safe(self.recipe.terrain),
                "output": self.recipe.output.to_dict() if self.recipe.output is not None else None,
            }
        )
        label_plans = dict(self.compiled_label_plans)
        compiled_label_plans = {
            str(layer_id): plan.to_dict() for layer_id, plan in sorted(label_plans.items())
        }
        depth_cull_layers: dict[str, Any] = {}
        for layer_id, plan in sorted(label_plans.items()):
            visibility: dict[str, Any] = {}
            for visible, labels_for_state in (
                (True, plan.accepted),
                (False, plan.rejected),
            ):
                for label in labels_for_state:
                    instance_id = "label-instance:" + _stable_hash(
                        {
                            "layer_id": str(layer_id),
                            "label_id": str(label.label_id),
                            "source_id": str(label.source_id),
                            "ordering_key": str(label.ordering_key or ""),
                        }
                    )
                    entry = {
                        "instance_id": instance_id,
                        "label_id": str(label.label_id),
                        "source_id": str(label.source_id),
                        "visible": visible,
                    }
                    if not visible:
                        entry["reason"] = str(label.reason)
                    visibility[instance_id] = entry
            depth_cull_layers[str(layer_id)] = {
                "accepted": [str(label.label_id) for label in plan.accepted],
                "rejected": [
                    {"label_id": str(label.label_id), "reason": str(label.reason)}
                    for label in plan.rejected
                ],
                # Public label ids are presentation metadata and need not be
                # unique.  Stable instance/source identity prevents duplicate
                # ids from overwriting each other's frozen visibility state.
                "visibility": visibility,
            }
        depth_authorities = sorted(
            {
                str(sample["depth_authority"])
                for plan in label_plans.values()
                for sample in (
                    *(
                        dict(candidate.terrain_sample or {})
                        for label in plan.accepted
                        for candidate in label.candidates
                    ),
                    *(
                        dict(label.details.get("terrain_sample") or {})
                        for label in plan.rejected
                    ),
                )
                if sample.get("depth_authority")
            }
        )
        manifest = RecipeManifest(
            recipe_family="mapscene_showcases",
            recipe_id=f"mapscene-compiled-{recipe_hash[:16]}",
            status="proven_in_forge3d",
            camera_defaults=self.recipe.camera.to_dict(),
            lighting_defaults=self.recipe.lighting.to_dict(),
            render_export_defaults=(
                self.recipe.output.to_dict() if self.recipe.output is not None else {}
            ),
            compiled_label_plans=compiled_label_plans,
            depth_cull={
                "source": "compile_phase",
                "depth_inputs": depth_authorities,
                "depth_proxy": (
                    "pre_supplied_authoritative"
                    if "pre_supplied_authoritative" in depth_authorities
                    else "deterministic_terrain_proxy"
                ),
                "camera_terrain_key": camera_terrain_key,
                "layers": depth_cull_layers,
            },
        )
        compiled = CompiledScenePlan(
            recipe_hash=recipe_hash,
            camera_terrain_key=camera_terrain_key,
            label_plans=MappingProxyType(label_plans),
            manifest=manifest,
            validation_report=report,
        )
        self.compiled_plan = compiled
        return compiled

    def _rehydrate_compiled_plan(self, manifest: Any) -> CompiledScenePlan:
        """Restore a frozen compiled plan verbatim from a bundle manifest."""
        from types import MappingProxyType

        from .label_plan import LabelPlan

        report = self.validate()
        label_plans = {
            str(layer_id): LabelPlan.from_dict(payload)
            for layer_id, payload in dict(manifest.compiled_label_plans or {}).items()
        }
        self.compiled_label_plans = dict(label_plans)
        depth_cull = dict(manifest.depth_cull or {})
        compiled = CompiledScenePlan(
            recipe_hash=_stable_hash(self.recipe.to_dict()),
            camera_terrain_key=str(depth_cull.get("camera_terrain_key") or ""),
            label_plans=MappingProxyType(label_plans),
            manifest=manifest,
            validation_report=report,
        )
        self.compiled_plan = compiled
        return compiled

    def _compiled_plan_for_current_recipe(self) -> CompiledScenePlan:
        """Reuse the compiled plan only when it still matches the recipe.

        A compiled plan is a total function of the serialized recipe; after
        any recipe mutation it is stale and must be recompiled once, or a
        render/bundle would carry frozen state from a recipe that no longer
        exists.
        """
        current_hash = _stable_hash(self.recipe.to_dict())
        compiled = self.compiled_plan
        if compiled is None or compiled.recipe_hash != current_hash:
            return self.compile_plan()
        return compiled

    def _report_with_feature(self, report: ValidationReport, feature: str, support_level: str) -> ValidationReport:
        payload = report.to_dict()
        supported = dict(payload.get("supported_features") or {})
        supported[feature] = support_level
        payload["supported_features"] = supported
        updated = ValidationReport.from_dict(payload)
        self.last_validation_report = updated
        return updated

    def render(
        self,
        path: str | None = None,
        *,
        emit_provenance: bool = False,
        provenance_signing_key: bytes | None = None,
        certificate: "bool | str | os.PathLike[str]" = False,
        cache: "str | os.PathLike[str] | None" = None,
    ) -> ValidationReport:
        output = self.recipe.output
        target = path or (output.path if output is not None else None)
        cache_eligible = bool(
            cache is not None
            and target
            and not emit_provenance
            and provenance_signing_key is None
            and not certificate
            and output is not None
            and str(output.format).lower() == "png"
            and int(output.bit_depth) == 8
            and not output.aovs
            and not output.hdr
        )
        if cache_eligible:
            from .anamnesis import render_sequence
            from ._canonical_json import canonical_json_bytes

            rendered: dict[str, ValidationReport] = {}

            def render_frame(_recipe: Mapping[str, Any], _frame: int) -> bytes:
                report = self._render_impl(
                    str(target),
                    emit_provenance=False,
                    provenance_signing_key=None,
                )
                rendered["report"] = report
                return Path(str(target)).read_bytes()

            sequence = render_sequence(
                self.to_dict(),
                frames=[0],
                cache=cache,
                render_frame=render_frame,
                render_frame_fingerprint=b"forge3d.python.mapscene.render/v1",
                render_frame_context=canonical_json_bytes(
                    self.to_dict(),
                    error_context="MapScene ANAMNESIS callback context",
                ),
            )
            report = rendered.get("report")
            if report is None:
                Path(str(target)).parent.mkdir(parents=True, exist_ok=True)
                Path(str(target)).write_bytes(sequence.frame_blobs[0])
                report = self._compiled_plan_for_current_recipe().validation_report
                self.last_render_path = str(target)
                self.last_render_backend = "anamnesis-cache"
                self.last_render_metadata = {
                    "cache_hit": True,
                    "cache_report": sequence.cache_report.to_dict(),
                    "frame_sha256": sequence.frame_hashes[0],
                }
            else:
                self.last_render_metadata = dict(self.last_render_metadata or {})
                self.last_render_metadata["cache_hit"] = False
                self.last_render_metadata["cache_report"] = sequence.cache_report.to_dict()
                self.last_render_metadata["frame_sha256"] = sequence.frame_hashes[0]
            return report

        from . import certificate as _certificate

        with _certificate._render_capture(
            "python.map_scene.render", "mapscene.finalize", draw_calls=1
        ):
            report = self._render_impl(
                path,
                emit_provenance=emit_provenance,
                provenance_signing_key=provenance_signing_key,
            )
        if certificate:
            sha = _certificate.emit_render_certificate(certificate)
            if sha is not None:
                self.last_render_metadata["certificate_payload_sha256"] = sha
        return report

    def _render_impl(
        self,
        path: str | None = None,
        *,
        emit_provenance: bool = False,
        provenance_signing_key: bytes | None = None,
    ) -> ValidationReport:
        output = self.recipe.output
        target = path or (output.path if output is not None else None)
        if not target:
            raise ValueError("MapScene.render requires a render path or OutputSpec.path")
        if emit_provenance:
            # VERITAS: opt-in emission of the (source_map.npy, provenance.json)
            # sibling artifacts next to the rendered PNG.
            if provenance_signing_key is None or len(provenance_signing_key) != 32:
                raise ValueError(
                    "emit_provenance=True requires provenance_signing_key: a "
                    "32-byte Ed25519 seed"
                )
        elif provenance_signing_key is not None:
            raise ValueError("provenance_signing_key requires emit_provenance=True")

        compiled = self._compiled_plan_for_current_recipe()
        report = compiled.validation_report
        self.last_validation_report = report
        if report.render_blocked(self.render_policy):
            if report.status == "warning" and self.render_policy == RenderFailurePolicy.FAIL_ON_WARNING:
                raise RuntimeError("MapScene.render blocked by warning diagnostics")
            raise RuntimeError("MapScene.render blocked by blocking diagnostics")

        from .helpers.offscreen import save_png_deterministic

        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        output = self.recipe.output
        sample_count = int(output.samples if output is not None else 1)
        native_result = _render_native_offscreen_result(
            self.recipe, compiled, emit_provenance=emit_provenance
        )
        if native_result is None:
            raise MapSceneNativeUnavailable(
                diagnostic_block(
                    layer="terrain",
                    reason="MapScene rendering requires the native offscreen terrain backend "
                    "(no renderable heightmap or no compatible GPU adapter); the deterministic "
                    "CPU placeholder output has been removed",
                    required_native=", ".join(required_native_symbols("terrain")),
                )
            )
        rgba = native_result.rgba
        backend = "gpu_terrain"
        native_metadata = dict(native_result.metadata)
        aov_frame = native_result.aov_frame
        hdr_frame = native_result.hdr_frame
        aov_paths = _write_mapscene_aovs(
            target_path,
            self.recipe,
            rgba,
            aov_frame=aov_frame,
            require_native=backend == "gpu_terrain",
        )
        output_format = str(output.format if output is not None else "png").lower()
        bit_depth = int(output.bit_depth if output is not None else 8)
        if output_format == "exr" or bool(output.hdr if output is not None else False):
            hdr_path = target_path if output_format == "exr" else target_path.with_suffix(".exr")
            if hdr_frame is not None:
                hdr_frame.save(str(hdr_path))
            else:
                _write_exr_array(hdr_path, rgba.astype(np.float32) / 255.0, channel_prefix="beauty")
            if output_format != "exr":
                save_png_deterministic(target_path, rgba, bit_depth=bit_depth)
                self.last_render_path = str(target_path)
            else:
                self.last_render_path = str(hdr_path)
        else:
            save_png_deterministic(target_path, rgba, bit_depth=bit_depth)
            self.last_render_path = str(target_path)
        self.last_render_backend = backend
        provenance_paths: dict[str, str] = {}
        if emit_provenance:
            # VERITAS: seal the frame's provenance and write the sibling
            # artifacts next to the rendered image. No silent fallback — a
            # render that could not produce the source map already raised.
            if native_result.source_map is None or native_result.contributing_tiles is None:
                raise RuntimeError(
                    "MapScene provenance emission produced no source map; "
                    "the native render path did not capture provenance"
                )
            import numpy as np

            import forge3d as f3d

            if not hasattr(f3d, "seal_provenance"):
                raise RuntimeError(
                    "MapScene provenance emission requires the native "
                    "forge3d.seal_provenance entrypoint"
                )
            manifest_bytes = bytes(
                f3d.seal_provenance(
                    native_result.source_map,
                    native_result.contributing_tiles,
                    provenance_signing_key,
                )
            )
            source_map_path = target_path.with_name(f"{target_path.stem}.source_map.npy")
            manifest_path = target_path.with_name(f"{target_path.stem}.provenance.json")
            np.save(source_map_path, native_result.source_map)
            manifest_path.write_bytes(manifest_bytes)
            provenance_paths = {
                "source_map": str(source_map_path),
                "manifest": str(manifest_path),
            }
        metadata = {
            "samples_used": int(native_metadata.get("samples_used", 1)),
            "target_samples": int(native_metadata.get("target_samples", sample_count)),
            "denoiser_used": str(native_metadata.get("denoiser_used", "none")),
            "aov_paths": aov_paths,
            "hdr": bool(output.hdr if output is not None else False),
            "format": output_format,
            "bit_depth": bit_depth,
            "aa_seed": _mapscene_aa_seed(self.recipe),
        }
        for key in ("final_p95_delta", "converged_ratio", "adaptive"):
            if key in native_metadata:
                metadata[key] = native_metadata[key]
        for key in (
            "building_backend",
            "building_batch_count",
            "building_batch_ids",
            "building_roof_types",
            "building_shadow_model",
            "building_scatter_stats",
            "gltf_textured_backend",
            "gltf_textured_layer_count",
            "gltf_material_count",
            "gltf_primitive_count",
            "gltf_asset_ids",
            "terrain_geometry_backend",
            "terrain_geometry_mode",
            "clipmap_ring_count",
            "clipmap_ring_resolution",
            "clipmap_triangle_count",
            "clipmap_vertex_count",
            "clipmap_triangle_reduction_pct",
            "clipmap_terrain_extent_m",
            "clipmap_resident_height_bytes",
            "clipmap_source_height_bytes",
            "clipmap_bounded_memory",
            "clipmap_error",
            "material_vt_stats",
            "raster_overlay_backend",
            "raster_overlay_layer_count",
            "vector_backend",
            "point_cloud_backend",
            "point_cloud_edl_backend",
            "tiles3d_backend",
            "tiles3d_source_bytes",
            "cloud_shadow_backend",
            "cloud_shadow_coverage",
            "cloud_shadow_strength",
            "cloud_shadow_quality",
            "cloud_shadow_offset",
            "screen_space_backend",
            "screen_space_effects",
            "screen_space_ssao_intensity",
            "screen_space_ssgi_intensity",
            "screen_space_ssr_intensity",
            "screen_space_taa_temporal_alpha",
            "terrain_main_pass_ms",
            "offline_accumulation_ms",
            "timing_source",
        ):
            if key in native_metadata:
                metadata[key] = native_metadata[key]
        self.last_render_metadata = metadata
        report = self._report_with_feature(report, "mapscene.render_backend", "supported")
        water_settings = _mapscene_water_settings(self.recipe)
        if water_settings is not None and bool(water_settings.enabled):
            report = self._report_with_feature(report, "mapscene.water_mask", "supported")
        if int(metadata["samples_used"]) > 1 or sample_count > 1:
            report = self._report_with_feature(report, "mapscene.offline_accumulation", "supported")
        if aov_paths:
            report = self._report_with_feature(report, "mapscene.aov_export", "supported")
        if output_format == "exr" or bool(output.hdr if output is not None else False):
            report = self._report_with_feature(report, "mapscene.hdr_output", "supported")
        if bit_depth == 16 and output_format == "png":
            report = self._report_with_feature(report, "mapscene.render_png_16bit", "supported")
        if any(isinstance(layer, VectorOverlay) for layer in self.recipe.layers):
            report = self._report_with_feature(report, "mapscene.vector_composite", "supported")
        if metadata.get("raster_overlay_backend") == "python_resample_composite":
            report = self._report_with_feature(report, "mapscene.raster_overlay_composite", "supported")
        if metadata.get("vector_backend") == "python_precise_raster":
            report = self._report_with_feature(report, "mapscene.vector_precise_raster_composite", "supported")
        if any(isinstance(layer, LabelLayer) for layer in self.recipe.layers):
            report = self._report_with_feature(report, "mapscene.label_composite", "supported")
        if any(isinstance(layer, BuildingLayer) for layer in self.recipe.layers):
            report = self._report_with_feature(report, "mapscene.building_composite", "supported")
        if metadata.get("building_backend") in {"native_instanced_mesh", "terrain_scatter_instanced_mesh"}:
            report = self._report_with_feature(report, "mapscene.building_gpu_mesh_composite", "supported")
        if metadata.get("gltf_textured_backend") == "mapscene_textured_landmark":
            report = self._report_with_feature(report, "gltf.textured_mapscene_render", "supported")
            report = self._report_with_feature(report, "buildings.textured_pbr", "supported")
        if metadata.get("terrain_geometry_backend") == "clipmap_indexed_pbr":
            report = self._report_with_feature(report, "terrain.clipmap_indexed", "supported")
            report = self._report_with_feature(report, "terrain.clipmap_planner", "supported")
            report = self._report_with_feature(report, "terrain.clipmap_bounded_memory", "supported")
        if metadata.get("point_cloud_backend") == "native_oit_points":
            report = self._report_with_feature(report, "point_cloud.mapscene_render", "supported")
        if metadata.get("point_cloud_edl_backend") == "weighted_oit_depth_edl":
            report = self._report_with_feature(report, "point_cloud.edl", "supported")
        if metadata.get("tiles3d_backend") == "native_oit_geometry":
            report = self._report_with_feature(report, "tiles3d.mapscene_render", "supported")
        if metadata.get("cloud_shadow_backend") == "mapscene_numpy_cloud_shadow":
            report = self._report_with_feature(report, "mapscene.cloud_shadows", "supported")
        screen_space_effects = set(metadata.get("screen_space_effects") or ())
        if screen_space_effects:
            report = self._report_with_feature(report, "mapscene.screen_space", "supported")
        if "ssao" in screen_space_effects:
            report = self._report_with_feature(report, "mapscene.ssao", "supported")
        if "ssgi" in screen_space_effects:
            report = self._report_with_feature(report, "mapscene.ssgi", "supported")
        if "ssr" in screen_space_effects:
            report = self._report_with_feature(report, "mapscene.ssr", "supported")
        if "taa" in screen_space_effects:
            report = self._report_with_feature(report, "mapscene.taa", "supported")
        if self.recipe.map_furniture is not None:
            report = self._report_with_feature(report, "mapscene.furniture_composite", "supported")
        render_feature = "mapscene.render_exr" if output_format == "exr" else "mapscene.render_png"
        return self._report_with_feature(report, render_feature, "supported")

    def _bundle_path(self, path: str | Path) -> Path:
        bundle_path = Path(path)
        if not bundle_path.suffix:
            bundle_path = bundle_path.with_suffix(".forge3d")
        return bundle_path

    def _bundle_manifest(self, checksums: Mapping[str, str]) -> Any:
        from .bundle import BUNDLE_VERSION, BundleManifest

        manifest = BundleManifest(
            version=BUNDLE_VERSION,
            name="mapscene_review",
            created_at="1970-01-01T00:00:00+00:00",
            description="Deterministic MapScene review bundle",
            checksums={key: checksums[key] for key in sorted(checksums)},
        )
        return manifest

    def _write_bundle_json(self, bundle_path: Path, rel_path: str, payload: Any, checksums: dict[str, str]) -> None:
        file_path = bundle_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        checksums[rel_path.replace("\\", "/")] = hashlib.sha256(file_path.read_bytes()).hexdigest()

    def _label_source_payload(self, layer: LabelLayer, layer_id: str) -> dict[str, Any]:
        metadata = _metadata_dict(layer.metadata)
        source_reference = {
            "layer_id": layer_id,
            "source_id": str(metadata.get("source_id") or layer_id),
            "source_hash": _stable_hash(layer.labels or layer.plan or {}),
        }
        return {
            "kind": "mapscene_label_source",
            "layer_id": layer_id,
            "source_reference": source_reference,
            "labels": _sequence(layer.labels),
            "metadata": _metadata(layer.metadata),
        }

    def _source_payload(self, layer: Any, layer_id: str, layer_type: str) -> dict[str, Any]:
        if hasattr(layer, "to_dict") and callable(layer.to_dict):
            payload = layer.to_dict()
        else:
            payload = _json_safe(layer)
        metadata = _metadata_dict(getattr(layer, "metadata", None))
        source_value = getattr(layer, "path", None)
        if source_value is None:
            source_value = getattr(layer, "source", None)
        source_id = metadata.get("source_id") or source_value or layer_id
        result = {
            "kind": "mapscene_layer_source",
            "layer_id": layer_id,
            "layer_type": layer_type,
            "source_reference": {
                "layer_id": layer_id,
                "source_id": str(source_id),
                "source_hash": _stable_hash(payload),
            },
            "payload": payload,
            "metadata": _metadata(metadata),
        }
        if isinstance(layer, VectorOverlay):
            result["features"] = _sequence(layer.features)
            result["style"] = _metadata(layer.style)
        return result

    def save_bundle(self, path: str | Path) -> ValidationReport:
        compiled = self._compiled_plan_for_current_recipe()
        report = self._report_with_feature(compiled.validation_report, "mapscene.save_bundle", "supported")
        bundle_path = self._bundle_path(path)
        bundle_path.mkdir(parents=True, exist_ok=True)
        checksums: dict[str, str] = {}

        recipe_payload = self.recipe.to_dict()
        self._write_bundle_json(bundle_path, "scene/mapscene_recipe.json", recipe_payload, checksums)

        from .recipe_manifest import manifest_to_json

        self._write_bundle_json(
            bundle_path,
            "scene/compiled_plan.json",
            json.loads(manifest_to_json(compiled.manifest)),
            checksums,
        )

        renderable = not report.render_blocked(self.render_policy)
        review_payload = {
            "kind": "mapscene_review_bundle",
            "schema": "forge3d.mapscene.review.v1",
            "renderable": renderable,
            "render_status": "ready_for_render" if renderable else "blocked_by_diagnostics",
            "recipe_hash": _stable_hash(recipe_payload),
            "validation_report_hash": _stable_hash(report.to_dict()),
            "camera": self.recipe.camera.to_dict(),
            "lighting": self.recipe.lighting.to_dict(),
            "output": self.recipe.output.to_dict() if self.recipe.output is not None else None,
            "map_furniture": self.recipe.map_furniture.to_dict() if self.recipe.map_furniture is not None else None,
            "supported_features": dict(report.supported_features or {}),
            "unsupported_features": dict(report.unsupported_features or {}),
            "supported_export_settings": {
                "bundle_schema": "forge3d.mapscene.review.v1",
                "label_plan_persistence": True,
                "output_formats": ["exr", "png"],
            },
            "compiled_label_plan_ids": sorted(compiled.label_plans),
            "source_layer_ids": [],
            "last_render_path": self.last_render_path,
            "last_render_backend": self.last_render_backend,
            "last_render_metadata": _metadata(self.last_render_metadata),
        }

        layer_source_ids = {"terrain"}
        self._write_bundle_json(
            bundle_path,
            "scene/layer_sources/terrain.json",
            self._source_payload(self.recipe.terrain, "terrain", "terrain_source"),
            checksums,
        )

        for layer in self.recipe.layers:
            layer_source_ids.add(_layer_id(layer, "layer"))
            layer_type = str(layer.to_dict().get("kind")) if hasattr(layer, "to_dict") else type(layer).__name__
            self._write_bundle_json(
                bundle_path,
                f"scene/layer_sources/{_layer_id(layer, 'layer')}.json",
                self._source_payload(layer, _layer_id(layer, "layer"), layer_type),
                checksums,
            )

        review_payload["source_layer_ids"] = sorted(layer_source_ids)
        self._write_bundle_json(bundle_path, "scene/mapscene_review.json", review_payload, checksums)

        for layer in self.recipe.layers:
            if isinstance(layer, LabelLayer) and _has_labels_or_plan(layer):
                layer_id = _layer_id(layer, "labels")
                plan = compiled.label_plans.get(layer_id)
                if plan is not None:
                    self._write_bundle_json(
                        bundle_path,
                        f"scene/label_plans/{layer_id}.json",
                        plan.to_dict(),
                        checksums,
                    )
                self._write_bundle_json(
                    bundle_path,
                    f"scene/label_sources/{layer_id}.json",
                    self._label_source_payload(layer, layer_id),
                    checksums,
                )

        from .bundle import SceneState

        state = SceneState(validation_report=report)
        self._write_bundle_json(bundle_path, "scene/state.json", state.to_dict(), checksums)

        manifest = self._bundle_manifest(checksums)
        with (bundle_path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

        self.last_bundle_path = str(bundle_path)
        return report


__all__ = [
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
    "BuildingLayer",
    "MapSceneBuildingLayer",
    "MapFurnitureLayer",
    "OrbitCamera",
    "LightingPreset",
    "OutputSpec",
    "ReproducibilityProfile",
]
