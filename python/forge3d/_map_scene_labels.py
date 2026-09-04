"""Label planning helpers for private MapScene orchestration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._map_scene_common import _metadata, _metadata_dict, _same_crs
from .diagnostics import (
    Diagnostic,
    missing_external_asset_diagnostic,
    missing_label_field_diagnostic,
    unavailable_terrain_sampler_diagnostic,
    unicode_coverage_gap_diagnostic,
)


def _has_labels_or_plan(layer: "LabelLayer") -> bool:
    return bool(layer.labels) or layer.plan is not None


def _label_plan_seed(recipe: "SceneRecipe", layer: "LabelLayer") -> int:
    metadata = _metadata_dict(layer.metadata)
    if "seed" in metadata:
        try:
            return int(metadata["seed"])
        except (TypeError, ValueError):
            pass
    if recipe.reproducibility_profile is not None:
        return int(recipe.reproducibility_profile.seed)
    return 0


def _terrain_heightmap_array(terrain: "TerrainSource") -> Any | None:
    import numpy as np

    data = getattr(terrain, "data", None)
    if data is not None:
        array = np.asarray(data, dtype=np.float32)
        return array if array.ndim == 2 else None

    path_value = getattr(terrain, "path", None)
    if not path_value:
        return None
    path = Path(path_value)
    try:
        if path.suffix.lower() == ".npy":
            array = np.asarray(np.load(path), dtype=np.float32)
        else:
            from .io import load_dem

            dem = load_dem(path)
            array = np.asarray(dem.data, dtype=np.float32)
    except Exception:
        return None
    return array if array.ndim == 2 else None


def _recipe_output_size(recipe: "SceneRecipe") -> tuple[float, float]:
    output = getattr(recipe, "output", None)
    width = float(getattr(output, "width", 1) or 1)
    height = float(getattr(output, "height", 1) or 1)
    return max(1.0, width), max(1.0, height)


def _coordinate_has_explicit_z(record: Mapping[str, Any]) -> bool:
    geometry = record.get("geometry") if isinstance(record.get("geometry"), Mapping) else {}
    coordinates = geometry.get("coordinates")
    if _is_coordinate(coordinates) and len(coordinates) > 2:
        return True
    for key in ("position", "world_pos"):
        value = record.get(key)
        if _is_coordinate(value) and len(value) > 2:
            return True
    return False


def _axis_to_height_index(value: float, viewport_extent: float, height_extent: int) -> int:
    if height_extent <= 1:
        return 0
    if 0.0 <= value <= 1.0 and viewport_extent > 1.0:
        unit = value
    else:
        unit = value / max(viewport_extent - 1.0, 1.0)
    unit = max(0.0, min(1.0, unit))
    return int(round(unit * float(height_extent - 1)))


class _TerrainOcclusionSampler:
    def __init__(
        self,
        heightmap: Any,
        *,
        viewport_size: tuple[float, float],
        bias: float = 0.0,
    ) -> None:
        if (
            len(viewport_size) != 2
            or not all(math.isfinite(float(value)) and float(value) > 0.0 for value in viewport_size)
        ):
            raise ValueError("Terrain occlusion viewport dimensions must be finite and positive")
        if not math.isfinite(float(bias)):
            raise ValueError("Terrain occlusion bias must be finite")
        self.heightmap = heightmap
        self.viewport_width, self.viewport_height = (
            float(viewport_size[0]),
            float(viewport_size[1]),
        )
        self.bias = float(bias)

    def sample_label(
        self,
        coords: Sequence[float],
        *,
        record: Mapping[str, Any],
        label_id: str,
    ) -> Mapping[str, Any]:
        del label_id
        row_count, col_count = self.heightmap.shape
        col = _axis_to_height_index(float(coords[0]), self.viewport_width, col_count)
        row = _axis_to_height_index(float(coords[1]), self.viewport_height, row_count)
        elevation = float(self.heightmap[row, col])
        explicit_z = _coordinate_has_explicit_z(record)
        z = float(coords[2]) if len(coords) > 2 else elevation
        height_tested = bool(explicit_z and abs(z) > 1.0e-6)
        visible = True if not height_tested else (z + self.bias) >= elevation
        return {
            "elevation": elevation,
            "source": "mapscene_terrain_heightmap",
            "visible": bool(visible),
            "occlusion": "terrain",
            "bias": self.bias,
            "sample_pixel": [int(col), int(row)],
            "explicit_z": bool(explicit_z),
            "height_tested": height_tested,
            "depth_authority": "deterministic_terrain_proxy",
        }


class _DepthOcclusionSampler:
    def __init__(
        self,
        depth_image: Any,
        *,
        viewport_size: tuple[float, float],
        bias: float = 0.0,
        source: str = "mapscene_depth_aov",
        authoritative: bool = True,
        depth_convention: str | None = None,
        depth_domain: Sequence[float] | None = None,
    ) -> None:
        import numpy as np

        from ._native import get_native_module

        self._tracked_bytes = 0
        self._closed = True
        self._native_reservation = None
        estimated_bytes = _depth_input_bytes(depth_image)
        if estimated_bytes <= 0:
            raise ValueError("Depth occlusion image must be nonempty")
        native = get_native_module()
        reserve = (
            getattr(native, "_reserve_label_depth_host_allocation", None)
            if native is not None
            else None
        )
        if not callable(reserve):
            raise RuntimeError(
                "Authoritative native label-depth memory tracking is unavailable"
            )
        self._native_reservation = reserve(
            int(estimated_bytes), "label_plan.compile.depth_visibility"
        )
        try:
            depth = np.array(depth_image, dtype=np.float32, order="C", copy=True)
            if depth.ndim != 2:
                raise ValueError("Depth occlusion image must be a 2D array")
            if depth.shape[0] == 0 or depth.shape[1] == 0:
                raise ValueError("Depth occlusion image must be nonempty")
            if not np.isfinite(depth).all():
                raise ValueError("Depth occlusion image must contain only finite values")
            convention = str(depth_convention or "").strip().lower()
            if convention not in {
                "normalized_device_depth",
                "reverse_normalized_device_depth",
                "linear_eye_depth",
            }:
                raise ValueError(
                    "Depth occlusion requires an explicit supported depth_convention"
                )
            if isinstance(depth_domain, (str, bytes)):
                raise ValueError(
                    "Depth occlusion requires explicit depth_domain=[minimum, maximum]"
                )
            try:
                if len(depth_domain) != 2:  # type: ignore[arg-type]
                    raise ValueError
                domain = (float(depth_domain[0]), float(depth_domain[1]))  # type: ignore[index]
            except (TypeError, ValueError, IndexError):
                raise ValueError(
                    "Depth occlusion requires explicit depth_domain=[minimum, maximum]"
                ) from None
            if not all(np.isfinite(value) for value in domain) or domain[0] >= domain[1]:
                raise ValueError("Depth occlusion depth_domain must be finite and increasing")
            if float(depth.min()) < domain[0] or float(depth.max()) > domain[1]:
                raise ValueError("Depth occlusion values must lie within depth_domain")
            resolved_bias = float(bias)
            if not np.isfinite(resolved_bias):
                raise ValueError("Depth occlusion bias must be finite")
            try:
                if len(viewport_size) != 2:
                    raise ValueError
                resolved_viewport = (
                    float(viewport_size[0]),
                    float(viewport_size[1]),
                )
            except (TypeError, ValueError, IndexError):
                raise ValueError(
                    "Depth occlusion viewport dimensions must be finite and positive"
                ) from None
            if not all(
                np.isfinite(value) and value > 0.0 for value in resolved_viewport
            ):
                raise ValueError(
                    "Depth occlusion viewport dimensions must be finite and positive"
                )
            self.depth_image = depth
            self._tracked_bytes = int(self.depth_image.nbytes)
            if self._tracked_bytes != estimated_bytes:
                raise ValueError(
                    "Depth occlusion image must be a rectangular 2D float32-compatible array"
                )
        except Exception:
            self._native_reservation.close()
            self._native_reservation = None
            raise
        self._closed = False
        self.viewport_width, self.viewport_height = resolved_viewport
        self.bias = resolved_bias
        self.source = str(source or "mapscene_depth_aov")
        self.authoritative = bool(authoritative)
        self.depth_convention = convention
        self.depth_domain = domain
        self.requires_projected_anchor = True

    def close(self) -> bool:
        if self._closed:
            return False
        # Disable sampling and discard the owned array while the native
        # reservation still covers its lifetime. Release the reservation last.
        self._closed = True
        self.depth_image = None
        reservation = self._native_reservation
        self._native_reservation = None
        return bool(reservation.close()) if reservation is not None else False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def sample_label(
        self,
        coords: Sequence[float],
        *,
        record: Mapping[str, Any],
        label_id: str,
    ) -> Mapping[str, Any]:
        del label_id
        if self._closed or self.depth_image is None:
            raise RuntimeError("Depth occlusion sampler is closed")
        projection_authority = str(record.get("projection_authority", "")).lower()
        projected_convention = str(record.get("projected_depth_convention", "")).lower()
        projected_domain = record.get("projected_depth_domain")
        projected_depth = record.get(
            "projected_depth", record.get("screen_depth", record.get("anchor_depth"))
        )
        if projected_depth is None and projection_authority in {
            "deterministic",
            "authoritative",
            "deterministic_camera_projection",
            "serialized_projected_anchor",
        }:
            projected_depth = coords[2] if len(coords) > 2 else None
        try:
            label_depth = float(projected_depth)
        except (TypeError, ValueError):
            label_depth = float("nan")
        if (
            projection_authority
            not in {
                "deterministic",
                "authoritative",
                "deterministic_camera_projection",
                "serialized_projected_anchor",
            }
            or not all(math.isfinite(float(value)) for value in coords[:3])
            or not math.isfinite(label_depth)
        ):
            return {
                "source": self.source,
                "visible": False,
                "occlusion": "depth_aov",
                "depth_tested": False,
                "projection_authority_missing": True,
                "depth_authority": (
                    "pre_supplied_authoritative"
                    if self.authoritative
                    else "deterministic_depth_proxy"
                ),
            }
        try:
            normalized_projected_domain = tuple(float(value) for value in projected_domain)
        except (TypeError, ValueError):
            normalized_projected_domain = ()
        if (
            projected_convention != self.depth_convention
            or normalized_projected_domain != self.depth_domain
            or not (self.depth_domain[0] <= label_depth <= self.depth_domain[1])
        ):
            return {
                "source": self.source,
                "visible": False,
                "occlusion": "depth_aov",
                "depth_tested": False,
                "depth_convention_incompatible": True,
                "depth_convention": self.depth_convention,
                "depth_domain": list(self.depth_domain),
                "projected_depth_convention": projected_convention or None,
                "projected_depth_domain": list(normalized_projected_domain),
                "depth_authority": (
                    "pre_supplied_authoritative"
                    if self.authoritative
                    else "deterministic_depth_proxy"
                ),
            }
        row_count, col_count = self.depth_image.shape
        col = _axis_to_height_index(float(coords[0]), self.viewport_width, col_count)
        row = _axis_to_height_index(float(coords[1]), self.viewport_height, row_count)
        scene_depth = float(self.depth_image[row, col])
        explicit_z = _coordinate_has_explicit_z(record)
        depth_tested = True
        if self.depth_convention == "reverse_normalized_device_depth":
            visible = label_depth >= scene_depth - self.bias
        else:
            visible = label_depth <= scene_depth + self.bias
        result: dict[str, Any] = {
            "scene_depth": scene_depth,
            "label_depth": label_depth,
            "source": self.source,
            "visible": bool(visible),
            "occlusion": "depth_aov",
            "bias": self.bias,
            "sample_pixel": [int(col), int(row)],
            "explicit_z": bool(explicit_z),
            "depth_tested": depth_tested,
            "depth_convention": self.depth_convention,
            "depth_domain": list(self.depth_domain),
            "projection_authority": projection_authority or None,
            "depth_authority": (
                "pre_supplied_authoritative"
                if self.authoritative
                else "deterministic_depth_proxy"
            ),
        }
        if not explicit_z:
            result["elevation"] = scene_depth
        return result


def _label_occlusion_bias(layer: "LabelLayer") -> float:
    metadata = _metadata_dict(getattr(layer, "metadata", None))
    value = metadata.get("terrain_occlusion_bias", metadata.get("occlusion_bias", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _depth_occlusion_config(layer: "LabelLayer", terrain: "TerrainSource") -> Mapping[str, Any] | None:
    for metadata in (
        _metadata_dict(getattr(layer, "metadata", None)),
        _metadata_dict(getattr(terrain, "metadata", None)),
    ):
        for key in (
            "depth_occlusion",
            "depth_aov",
            "depth_buffer",
            "depth_image",
            "silhouette_depth",
            "silhouette",
        ):
            value = metadata.get(key)
            if value is None:
                continue
            if isinstance(value, Mapping):
                return dict(value)
            return {"image": value}
    return None


def _depth_image_from_config(config: Mapping[str, Any]) -> Any | None:
    for key in ("image", "data", "values", "depth"):
        if key in config:
            return config[key]
    return None


def _depth_input_bytes(value: Any) -> int:
    """Exact byte envelope of the float32 compile-time depth allocation."""
    shape = getattr(value, "shape", None)
    if isinstance(shape, Sequence) and len(shape) == 2:
        return max(0, int(shape[0])) * max(0, int(shape[1])) * 4
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = len(value)
        cols = 0
        if rows:
            first = value[0]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                cols = len(first)
        return rows * cols * 4
    return 0


def _terrain_occlusion_sampler(
    layer: "LabelLayer",
    *,
    recipe: "SceneRecipe",
    terrain: "TerrainSource",
) -> Any | None:
    if str(getattr(layer, "occlusion", "terrain")).lower() != "terrain":
        return None
    depth_config = _depth_occlusion_config(layer, terrain)
    if depth_config is not None:
        depth_image = _depth_image_from_config(depth_config)
        if depth_image is not None:
            bias_value = depth_config.get("bias", _label_occlusion_bias(layer))
            return _DepthOcclusionSampler(
                depth_image,
                viewport_size=_recipe_output_size(recipe),
                bias=bias_value,
                source=str(depth_config.get("source") or "mapscene_depth_aov"),
                authoritative=bool(depth_config.get("authoritative", True)),
                depth_convention=depth_config.get("depth_convention"),
                depth_domain=depth_config.get("depth_domain"),
            )
    heightmap = _terrain_heightmap_array(terrain)
    if heightmap is None:
        return None
    return _TerrainOcclusionSampler(
        heightmap,
        viewport_size=_recipe_output_size(recipe),
        bias=_label_occlusion_bias(layer),
    )


def _labels_with_terrain_occlusion(
    labels: Sequence[Mapping[str, Any]] | None,
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    result = []
    for label in labels or ():
        item = dict(label)
        if enabled:
            item["requires_terrain"] = True
            item["terrain_mode"] = "terrain"
            item["occlusion"] = "terrain"
        result.append(item)
    return result


def _label_plan_from_layer(
    layer: "LabelLayer",
    *,
    recipe: "SceneRecipe",
    terrain: "TerrainSource",
) -> Any:
    from .label_plan import LabelPlan

    if layer.plan is not None:
        if isinstance(layer.plan, LabelPlan):
            return layer.plan
        if isinstance(layer.plan, Mapping):
            return LabelPlan.from_dict(layer.plan)
        if hasattr(layer.plan, "to_dict") and callable(layer.plan.to_dict):
            return LabelPlan.from_dict(layer.plan.to_dict())
        raise TypeError("LabelLayer.plan must be a LabelPlan or LabelPlan-compatible mapping")

    keepouts = tuple(recipe.map_furniture.keepouts or ()) if recipe.map_furniture is not None else ()
    terrain_sampler = _terrain_occlusion_sampler(layer, recipe=recipe, terrain=terrain)
    labels = _labels_with_terrain_occlusion(
        layer.labels or (),
        enabled=terrain_sampler is not None,
    )
    try:
        return LabelPlan.compile(
            labels=labels,
            camera=recipe.camera,
            viewport=recipe.output,
            terrain=terrain_sampler if terrain_sampler is not None else terrain,
            keepouts=keepouts,
            priority_rules=layer.priority_rules or (),
            typography=layer.typography or {},
            glyph_atlas=layer.glyph_atlas,
            seed=_label_plan_seed(recipe, layer),
        )
    finally:
        close = getattr(terrain_sampler, "close", None)
        if callable(close):
            close()


def _diagnostic_for_layer(diagnostic: Diagnostic, layer_id: str) -> Diagnostic:
    if diagnostic.layer_id in {None, "labels", layer_id}:
        payload = diagnostic.to_dict()
        payload["layer_id"] = layer_id
        return Diagnostic.from_dict(payload)
    return diagnostic


def _feature_geometry(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    geometry = feature.get("geometry")
    if isinstance(geometry, Mapping):
        return geometry
    geometry_type = feature.get("geometry_type") or feature.get("type")
    coordinates = feature.get("coordinates") or feature.get("position") or feature.get("world_pos")
    if geometry_type and coordinates is not None:
        return {"type": str(geometry_type), "coordinates": coordinates}
    return {}


def _is_coordinate(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    if len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _valid_geometry(geometry: Mapping[str, Any]) -> tuple[str | None, str | None]:
    geometry_type = str(geometry.get("type", "") or "")
    coordinates = geometry.get("coordinates")
    geometry_key = geometry_type.lower()
    if geometry_key == "point":
        return ("Point", "point") if _is_coordinate(coordinates) else (None, None)
    if geometry_key == "linestring":
        if (
            isinstance(coordinates, Sequence)
            and not isinstance(coordinates, (str, bytes))
            and len(coordinates) >= 2
            and all(_is_coordinate(point) for point in coordinates)
        ):
            return "LineString", "line"
        return None, None
    if geometry_key == "polygon":
        if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes)) or not coordinates:
            return None, None
        outer = coordinates[0]
        if (
            isinstance(outer, Sequence)
            and not isinstance(outer, (str, bytes))
            and len(outer) >= 4
            and all(_is_coordinate(point) for point in outer)
        ):
            return "Polygon", "polygon"
        return None, None
    if geometry_type:
        return geometry_type, None
    return None, None


def _feature_properties(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    properties = feature.get("properties")
    if isinstance(properties, Mapping):
        return properties
    return {str(key): value for key, value in feature.items() if key not in {"geometry", "coordinates", "type"}}


def _feature_id(feature: Mapping[str, Any], index: int) -> str:
    return str(feature.get("id") or feature.get("feature_id") or feature.get("source_id") or f"feature-{index}")


def _style_text_expression(style_layer: Any) -> Any:
    if isinstance(style_layer, Mapping):
        layout = style_layer.get("layout")
        if isinstance(layout, Mapping):
            return layout.get("text-field")
        return style_layer.get("text") or style_layer.get("text_field")
    layout = getattr(style_layer, "layout", None)
    if layout is not None:
        return getattr(layout, "text_field", None)
    return getattr(style_layer, "text", None) or getattr(style_layer, "text_field", None)


def _label_text_from_expression(expression: Any, properties: Mapping[str, Any]) -> tuple[str, str | None]:
    if expression is None:
        expression = "name"
    if isinstance(expression, str):
        if expression.startswith("{") and expression.endswith("}") and len(expression) > 2:
            field = expression[1:-1]
            if field not in properties:
                return "", field
            return str(properties[field]), None
        if expression in properties:
            return str(properties[expression]), None
        return str(expression), None
    try:
        from .style_expressions import EvalContext, evaluate

        value = evaluate(expression, EvalContext(properties=dict(properties)))
    except Exception:
        return "", str(expression)
    if value is None:
        if isinstance(expression, Sequence) and not isinstance(expression, (str, bytes)) and len(expression) >= 2:
            return "", str(expression[1])
        return "", str(expression)
    return str(value), None


def _transform_label_geometry(
    geometry: Mapping[str, Any],
    *,
    from_crs: str | None,
    to_crs: str | None,
) -> Mapping[str, Any]:
    if not from_crs or not to_crs or _same_crs(from_crs, to_crs):
        return geometry

    import numpy as np

    from .crs import transform_coords

    geometry_type = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates")
    if geometry_type == "Point" and _is_coordinate(coordinates):
        xy = np.asarray([[float(coordinates[0]), float(coordinates[1])]], dtype=np.float64)
        transformed = transform_coords(xy, from_crs, to_crs)[0]
        updated = [float(transformed[0]), float(transformed[1])]
        if len(coordinates) > 2:
            updated.append(float(coordinates[2]))
        return {"type": geometry_type, "coordinates": updated}
    if geometry_type == "LineString" and isinstance(coordinates, Sequence) and not isinstance(coordinates, (str, bytes)):
        points = [point for point in coordinates if _is_coordinate(point)]
        transformed = transform_coords(
            np.asarray([[float(point[0]), float(point[1])] for point in points], dtype=np.float64),
            from_crs,
            to_crs,
        )
        updated = []
        for point, xy in zip(points, transformed):
            new_point = [float(xy[0]), float(xy[1])]
            if len(point) > 2:
                new_point.append(float(point[2]))
            updated.append(new_point)
        return {"type": geometry_type, "coordinates": updated}
    if geometry_type == "Polygon" and isinstance(coordinates, Sequence) and not isinstance(coordinates, (str, bytes)):
        rings = []
        for ring in coordinates:
            if not isinstance(ring, Sequence) or isinstance(ring, (str, bytes)):
                rings.append(ring)
                continue
            points = [point for point in ring if _is_coordinate(point)]
            if not points:
                rings.append([])
                continue
            transformed = transform_coords(
                np.asarray([[float(point[0]), float(point[1])] for point in points], dtype=np.float64),
                from_crs,
                to_crs,
            )
            updated_ring = []
            for point, xy in zip(points, transformed):
                new_point = [float(xy[0]), float(xy[1])]
                if len(point) > 2:
                    new_point.append(float(point[2]))
                updated_ring.append(new_point)
            rings.append(updated_ring)
        return {"type": geometry_type, "coordinates": rings}
    return geometry


def _call_label_terrain_sampler(sampler: Any, coordinates: Sequence[Any]) -> Mapping[str, Any]:
    if sampler is None or not callable(sampler):
        return {}
    x = float(coordinates[0])
    y = float(coordinates[1])
    z = float(coordinates[2]) if len(coordinates) > 2 else 0.0
    for args in ((x, y, z), (x, y), (coordinates,)):
        try:
            result = sampler(*args)
        except TypeError:
            continue
        if isinstance(result, Mapping):
            return _metadata(result)
        if result is not None:
            return {"elevation": float(result), "source": type(sampler).__name__, "visible": True}
    return {}


def _sample_label_geometry(
    geometry: Mapping[str, Any],
    *,
    terrain_sampler: Any | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    coordinates = geometry.get("coordinates")
    if str(geometry.get("type")) != "Point" or not _is_coordinate(coordinates):
        return geometry, {}
    sample = _call_label_terrain_sampler(terrain_sampler, coordinates)
    if "elevation" not in sample:
        return geometry, sample
    updated = [float(coordinates[0]), float(coordinates[1]), float(sample["elevation"])]
    return {"type": "Point", "coordinates": updated}, sample


def _atlas_glyph_set(glyph_atlas: Any) -> set[str] | None:
    if glyph_atlas is None:
        return None
    if hasattr(glyph_atlas, "glyphs"):
        return {str(glyph) for glyph in getattr(glyph_atlas, "glyphs")}
    if isinstance(glyph_atlas, Mapping):
        glyphs = glyph_atlas.get("glyphs")
        if isinstance(glyphs, Mapping):
            result = set()
            for key in glyphs:
                try:
                    result.add(chr(int(key)))
                except (TypeError, ValueError):
                    result.add(str(key))
            return result
        if isinstance(glyphs, (set, frozenset, list, tuple)):
            return {chr(glyph) if isinstance(glyph, int) else str(glyph) for glyph in glyphs}
    if isinstance(glyph_atlas, (set, frozenset, list, tuple)):
        return {chr(glyph) if isinstance(glyph, int) else str(glyph) for glyph in glyph_atlas}
    return None
