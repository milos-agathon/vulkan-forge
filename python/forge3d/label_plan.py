"""Deterministic label-plan compiler contract for offline map rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .diagnostics import (
    Diagnostic,
    label_rejection_summary_diagnostic,
    missing_glyphs_diagnostic,
    placeholder_fallback_diagnostic,
)


PAYLOAD_VERSION = 2
SUPPORTED_PAYLOAD_VERSIONS = (1, PAYLOAD_VERSION)
MAX_LABEL_RECORDS = 100_000
REJECTION_REASONS = (
    "collision",
    "outside_view",
    "missing_glyph",
    "priority_lost",
    "keepout_region",
    "terrain_occluded",
    "invalid_geometry",
    "unsupported_geometry_type",
    "empty_text",
    "font_chain_required",
    "malformed_font",
    "shaping_failed",
    "missing_geometry_authority",
    "missing_projection_authority",
    "no_eligible_candidate",
    "incompatible_depth_convention",
)

CARTOGRAPHIC_PRIORITY_PRESET = (
    {"name": "annotations", "rank": 10},
    {"name": "roads", "rank": 20},
    {"name": "rivers", "rank": 30},
    {"name": "peaks", "rank": 40},
    {"name": "cities", "rank": 50},
    {"name": "capitals", "rank": 60},
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"LabelPlan payload value is not JSON-serializable: {type(value).__name__}")


def _stable_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_version(value: Any) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported label plan payload_version: {value!r}") from error
    if version not in SUPPORTED_PAYLOAD_VERSIONS:
        raise ValueError(f"unsupported label plan payload_version: {version}")
    return version


def _migrate_payload_v1_to_v2(data: Mapping[str, Any]) -> dict[str, Any]:
    migrated = dict(data)
    migrated["payload_version"] = PAYLOAD_VERSION
    accepted = []
    for raw_label in migrated.get("accepted") or ():
        label = dict(raw_label)
        label.setdefault("positioned_glyphs", [])
        typography = dict(label.get("typography") or {})
        if not label["positioned_glyphs"]:
            typography["render_mapping"] = "legacy_codepoints_not_renderable"
        label["typography"] = typography
        accepted.append(label)
    migrated["accepted"] = accepted
    migrated.setdefault("rationale", [])
    return migrated


def _stable_unit_interval(*parts: Any) -> float:
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _number(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coordinates(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    coords = [_number(item, default=float("nan")) for item in value]
    if len(coords) < 2 or any(coord != coord for coord in coords):
        return None
    while len(coords) < 3:
        coords.append(0.0)
    return coords[:3]


def _viewport_size(viewport: Any) -> tuple[float, float] | None:
    if isinstance(viewport, Mapping):
        if "width" in viewport and "height" in viewport:
            return (_number(viewport["width"]), _number(viewport["height"]))
    if isinstance(viewport, Sequence) and not isinstance(viewport, (str, bytes)) and len(viewport) >= 2:
        return (_number(viewport[0]), _number(viewport[1]))
    width = getattr(viewport, "width", None)
    height = getattr(viewport, "height", None)
    if width is not None and height is not None:
        return (_number(width), _number(height))
    return None


def _strict_projected_anchor(value: Any) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 3
    ):
        return None
    anchor = [_number(item, default=float("nan")) for item in value[:3]]
    if not all(math.isfinite(item) for item in anchor):
        return None
    return anchor


def _project_anchor(
    record: Mapping[str, Any],
    camera: Any,
    viewport: Any,
    world_anchor: Sequence[float],
) -> tuple[list[float] | None, str | None]:
    """Resolve the serialized/projected screen anchor used for visibility.

    Callers may serialize ``projected_anchor`` directly, or supply a camera
    object with a deterministic ``project`` method. Legacy screen-coordinate
    inputs remain unchanged.
    """
    projected = _strict_projected_anchor(record.get("projected_anchor"))
    if projected is not None:
        if "projected_depth" in record:
            projected[2] = _number(record["projected_depth"], default=projected[2])
        if math.isfinite(projected[2]):
            return projected, "serialized_projected_anchor"
        return None, None
    projector = getattr(camera, "project", None)
    declared = str(getattr(camera, "projection_authority", "")).lower()
    if callable(projector) and declared in {"deterministic", "authoritative"}:
        try:
            value = projector(tuple(float(item) for item in world_anchor), viewport=viewport)
        except TypeError:
            value = projector(tuple(float(item) for item in world_anchor))
        projected = _strict_projected_anchor(value)
        if projected is None:
            raise ValueError("camera.project must return a finite screen x/y/depth anchor")
        return projected, "deterministic_camera_projection"
    return None, None


def _projection_diagnostic(label_id: str) -> Diagnostic:
    return Diagnostic(
        code="label_projection_authority_missing",
        severity="error",
        message="Authoritative depth visibility requires a projected screen anchor and depth.",
        remediation=(
            "Serialize projected_anchor=[x, y, depth] for this label or provide a "
            "camera.project implementation declared with projection_authority='deterministic'."
        ),
        support_level="unsupported",
        layer_id="labels",
        object_id=label_id,
        details={"required": "finite projected_anchor[x,y,depth]"},
    )


def _depth_convention_diagnostic(label_id: str) -> Diagnostic:
    return Diagnostic(
        code="label_depth_convention_incompatible",
        severity="error",
        message="Projected label depth is not compatible with the authoritative depth input.",
        remediation=(
            "Use the same explicit depth_convention and finite increasing depth_domain "
            "for projected anchors and the serialized depth image."
        ),
        support_level="unsupported",
        layer_id="labels",
        object_id=label_id,
    )


def _iter_label_records(labels: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(labels, Mapping):
        for key in sorted(labels.keys(), key=str):
            value = labels[key]
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("id", str(key))
                yield str(key), record
        return
    for index, value in enumerate(labels or ()):
        if isinstance(value, Mapping):
            yield str(index), dict(value)


def _glyph_set(glyph_atlas: Any) -> set[str] | None:
    if glyph_atlas is None:
        return None
    if isinstance(glyph_atlas, Mapping):
        glyphs = glyph_atlas.get("glyphs")
        if glyphs is not None:
            return {str(glyph) for glyph in glyphs}
    if isinstance(glyph_atlas, (set, frozenset, list, tuple)):
        return {str(glyph) for glyph in glyph_atlas}
    return None


def _font_paths_from_glyph_atlas(glyph_atlas: Any) -> list[str]:
    if not isinstance(glyph_atlas, Mapping):
        return []
    paths: list[str] = []
    for key in ("font_chain", "font_paths", "font_sources"):
        values = glyph_atlas.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for value in values:
                path = Path(str(value))
                if path.suffix.lower() in {".ttf", ".otf", ".ttc"} and path.exists():
                    paths.append(str(path))
    for key in ("font_path", "source_font_path", "font_file", "source_path"):
        value = glyph_atlas.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.suffix.lower() in {".ttf", ".otf", ".ttc"} and path.exists():
            paths.append(str(path))
    coverage = glyph_atlas.get("coverage")
    if isinstance(coverage, Mapping):
        value = coverage.get("path") or coverage.get("font_path")
        if value:
            path = Path(str(value))
            if path.suffix.lower() in {".ttf", ".otf", ".ttc"} and path.exists():
                paths.append(str(path))
    return list(dict.fromkeys(paths))


def _packaged_latin_font_path() -> str:
    return str(Path(__file__).resolve().parent / "data" / "fonts" / "NotoSansLatin-subset.ttf")


def _stable_font_source(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _native_shape_label_glyphs(
    text: str,
    glyph_atlas: Any,
    line_ranges: Sequence[Sequence[int]] | None = None,
) -> tuple[list[str] | None, Mapping[str, Any]] | None:
    font_paths = _font_paths_from_glyph_atlas(glyph_atlas)
    if not font_paths:
        return None
    try:
        from .text import shape

        shaped = shape(text, font_paths, 1.0)
    except Exception as error:
        diagnostics = list(getattr(error, "diagnostics", ()))
        native_reason = str(diagnostics[0].get("reason", "shaping_failed")) if diagnostics else "shaping_failed"
        return None, {
            "shaping": "littera_error",
            "native_reason": native_reason,
            "diagnostics": diagnostics,
        }
    resolved_ranges = (
        [tuple(int(value) for value in item) for item in line_ranges]
        if line_ranges is not None
        else None
    )
    payload = shaped.to_dict(resolved_ranges)
    glyph_records = [glyph for run in payload["runs"] for glyph in run["glyphs"]]
    positioned = [dict(glyph) for glyph in payload.get("positioned_glyphs", ())]
    line_ranges = [list(item) for item in payload.get("line_ranges", ())]
    glyphs = list(text)
    if not glyphs:
        return None
    details = {
        "shaping": "littera",
        "engine": "littera",
        "direction": payload["runs"][0]["direction"] if payload["runs"] else "ltr",
        "glyph_ids": [glyph["glyph_id"] for glyph in glyph_records],
        "font_indices": [glyph["font_index"] for glyph in glyph_records],
        "clusters": [glyph["cluster"] for glyph in glyph_records],
        "advances": [glyph["x_advance"] for glyph in glyph_records],
        "line_ranges": line_ranges,
        "positioned_glyphs": positioned,
        "render_mapping": "positioned_glyphs_by_id",
        "shaped_runs": payload["runs"],
        "compositor": "native_analytic_coverage",
        "font_chain": [
            _stable_font_source(path) for path in payload.get("font_sources", font_paths)
        ],
        "font_sha256": list(payload.get("font_sha256", ())),
    }
    return glyphs, details


def _rect_bounds(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    bounds = [_number(item, default=float("nan")) for item in value[:4]]
    if any(coord != coord for coord in bounds):
        return None
    x0, y0, x1, y1 = bounds
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _rects_intersect(left: Sequence[float] | None, right: Sequence[float] | None) -> bool:
    left_bounds = _rect_bounds(left)
    right_bounds = _rect_bounds(right)
    if left_bounds is None or right_bounds is None:
        return False
    return (
        left_bounds[0] <= right_bounds[2]
        and left_bounds[2] >= right_bounds[0]
        and left_bounds[1] <= right_bounds[3]
        and left_bounds[3] >= right_bounds[1]
    )


def _requires_terrain(record: Mapping[str, Any]) -> bool:
    mode = str(record.get("terrain_mode", "")).lower()
    return bool(record.get("requires_terrain")) or mode in {"required", "sample", "terrain"}


def _requires_complex_shaping(text: str) -> bool:
    return any(
        ("\u0590" <= char <= "\u08ff")
        or ("\u0900" <= char <= "\u0dff")
        or ("\ufb50" <= char <= "\ufdff")
        or ("\ufe70" <= char <= "\ufeff")
        for char in text
    )


def _shape_label_glyphs(
    text: str,
    glyph_atlas: Any | None = None,
    line_ranges: Sequence[Sequence[int]] | None = None,
) -> tuple[list[str] | None, Mapping[str, Any]]:
    native_shaped = _native_shape_label_glyphs(text, glyph_atlas, line_ranges)
    if native_shaped is not None:
        return native_shaped
    if not _requires_complex_shaping(text):
        packaged = _native_shape_label_glyphs(
            text, {"font_path": _packaged_latin_font_path()}, line_ranges
        )
        if packaged is not None:
            glyphs, details = packaged
            if glyphs is not None:
                return glyphs, details
            diagnostics = list(details.get("diagnostics", ()))
            if diagnostics and diagnostics[0].get("reason") != "native_text_unavailable":
                return glyphs, details
        # Planning remains usable without the extension, but this compatibility
        # payload is explicitly non-renderable by the production compositor.
        return list(text), {
            "shaping": "legacy_metadata_only",
            "render_mapping": "legacy_codepoints_not_renderable",
            "positioned_glyphs": [],
        }
    return None, {
        "shaping": "font_chain_required",
        "diagnostics": [{"status": "diagnostic_block", "reason": "font_chain_required"}],
    }


def _call_terrain_sampler(terrain: Any, coords: Sequence[float]) -> Mapping[str, Any]:
    sampler = getattr(terrain, "sample", None) or (terrain if callable(terrain) else None)
    if sampler is None:
        return {}
    x, y, z = coords
    for args in ((x, y, z), (x, y), (coords,)):
        try:
            result = sampler(*args)
        except TypeError:
            continue
        if isinstance(result, Mapping):
            return dict(result)
        if result is not None:
            return {"elevation": _number(result), "source": type(terrain).__name__, "visible": True}
    return {"source": type(terrain).__name__, "unavailable": True, "visible": False}


def _terrain_sample(
    record: Mapping[str, Any],
    terrain: Any,
    label_id: str,
    coords: Sequence[float] | None = None,
    candidate_id: str | None = None,
) -> Mapping[str, Any]:
    for key in ("candidate_terrain_samples", "terrain_samples"):
        indexed = record.get(key)
        if (
            candidate_id is not None
            and isinstance(indexed, Mapping)
            and isinstance(indexed.get(candidate_id), Mapping)
        ):
            return dict(indexed[candidate_id])
    sample = record.get("terrain_sample")
    if (
        candidate_id is not None
        and isinstance(sample, Mapping)
        and str(sample.get("candidate_id", "")) == candidate_id
        and isinstance(sample.get("sample"), Mapping)
    ):
        return dict(sample["sample"])
    if isinstance(terrain, Mapping):
        samples = terrain.get("samples")
        label_samples = samples.get(label_id) if isinstance(samples, Mapping) else None
        if (
            candidate_id is not None
            and isinstance(label_samples, Mapping)
            and isinstance(label_samples.get(candidate_id), Mapping)
        ):
            return dict(label_samples[candidate_id])
    if coords is not None and _requires_terrain(record):
        if terrain is None:
            return {"source": "terrain_sampler", "unavailable": True, "visible": False}
        sample_label = getattr(terrain, "sample_label", None)
        if callable(sample_label):
            try:
                result = sample_label(coords, record=record, label_id=label_id)
            except TypeError:
                result = sample_label(coords)
            if isinstance(result, Mapping):
                return dict(result)
            if result is not None:
                return {"elevation": _number(result), "source": type(terrain).__name__, "visible": True}
        return _call_terrain_sampler(terrain, coords)
    return {}


def _native_declutter_optimal() -> Any | None:
    """Return the native bounded-optimal declutter solver, or ``None``."""
    try:
        from ._native import get_native_module

        native = get_native_module()
    except Exception:
        return None
    if native is None:
        return None
    return getattr(native, "declutter_optimal", None)


_SUPPORTED_DEPTH_CONVENTIONS = frozenset(
    {
        "normalized_device_depth",
        "reverse_normalized_device_depth",
        "linear_eye_depth",
    }
)


def _depth_domain(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) != 2:
            return None
        domain = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(item) for item in domain) or domain[0] >= domain[1]:
        return None
    return domain


def _derive_authoritative_depth_sample(
    record: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
    sample_anchor: Sequence[float],
) -> dict[str, Any]:
    """Validate depth evidence and derive visibility inside the compiler.

    The sampler owns the scene-depth measurement.  The compiler owns the
    projected label depth and comparison semantics, so a caller-supplied
    ``visible`` flag is intentionally ignored and overwritten.
    """
    sample = dict(raw_sample)
    projection_authority = str(record.get("projection_authority", "")).lower()
    if projection_authority not in {
        "deterministic",
        "authoritative",
        "deterministic_camera_projection",
        "serialized_projected_anchor",
    }:
        sample.update(
            {
                "visible": False,
                "depth_tested": False,
                "projection_authority_missing": True,
            }
        )
        return sample

    sample_convention = str(sample.get("depth_convention", "")).lower()
    projected_convention = str(
        record.get("projected_depth_convention", "")
    ).lower()
    sample_domain = _depth_domain(sample.get("depth_domain"))
    projected_domain = _depth_domain(record.get("projected_depth_domain"))
    if (
        sample.get("depth_convention_incompatible") is True
        or sample_convention not in _SUPPORTED_DEPTH_CONVENTIONS
        or sample_convention != projected_convention
        or sample_domain is None
        or sample_domain != projected_domain
    ):
        sample.update(
            {
                "visible": False,
                "depth_tested": False,
                "depth_convention_incompatible": True,
                "depth_convention": sample_convention or None,
                "depth_domain": (
                    list(sample_domain) if sample_domain is not None else None
                ),
            }
        )
        return sample

    try:
        scene_depth = float(sample["scene_depth"])
        label_depth = float(sample_anchor[2])
        bias = float(sample.get("bias", 0.0))
    except (KeyError, TypeError, ValueError, IndexError):
        scene_depth = label_depth = bias = float("nan")
    numeric_values = (scene_depth, label_depth, bias)
    if (
        not all(math.isfinite(value) for value in numeric_values)
        or not (sample_domain[0] <= scene_depth <= sample_domain[1])
        or not (sample_domain[0] <= label_depth <= sample_domain[1])
    ):
        sample.update(
            {
                "scene_depth": scene_depth if math.isfinite(scene_depth) else None,
                "label_depth": label_depth if math.isfinite(label_depth) else None,
                "bias": bias if math.isfinite(bias) else None,
                "visible": False,
                "depth_tested": False,
                "depth_sample_invalid": True,
                "depth_convention": sample_convention,
                "depth_domain": list(sample_domain),
            }
        )
        return sample

    if sample_convention == "reverse_normalized_device_depth":
        visible = label_depth >= scene_depth - bias
        comparison = "reverse_greater_equal"
    else:
        visible = label_depth <= scene_depth + bias
        comparison = "forward_less_equal"
    sample.update(
        {
            "scene_depth": scene_depth,
            "label_depth": label_depth,
            "bias": bias,
            "visible": bool(visible),
            "depth_tested": True,
            "depth_convention": sample_convention,
            "depth_domain": list(sample_domain),
            "depth_comparison": comparison,
            "visibility_authority": "label_plan.compile",
        }
    )
    return sample


def _candidate_visibility_records(
    record: Mapping[str, Any],
    terrain: Any,
    label_id: str,
    source_id: str,
    candidates: Sequence["LabelCandidate"],
) -> list[dict[str, Any]]:
    """Compile-time silhouette/depth visibility gate over candidate anchors.

    Samples the terrain depth/silhouette proxy at every candidate anchor and
    marks occluded anchors ``visible=False`` so they contribute zero
    placements; each occluded anchor yields a grounded rationale record
    citing the sampled depth versus the anchor depth.
    """
    records: list[dict[str, Any]] = []
    if not _requires_terrain(record):
        return records
    for candidate in candidates:
        sample_anchor = list(candidate.anchor)
        projected_depth = record.get("projected_depth")
        geometry = record.get("geometry") if isinstance(record.get("geometry"), Mapping) else {}
        world_anchor = _coordinates(
            geometry.get("coordinates", record.get("position", record.get("world_pos")))
        )
        if projected_depth is not None:
            sample_anchor[2] = _number(projected_depth, default=sample_anchor[2])
        elif world_anchor is not None:
            sample_anchor[2] = world_anchor[2]
        sample = _terrain_sample(
            record,
            terrain,
            label_id,
            sample_anchor,
            candidate_id=candidate.candidate_id,
        )
        depth_authority = str(sample.get("depth_authority", ""))
        is_authoritative_depth = (
            "scene_depth" in sample
            or depth_authority
            in {"pre_supplied_authoritative", "deterministic_depth_proxy"}
            or str(sample.get("occlusion", "")).lower()
            in {"depth", "depth_aov", "depth_silhouette"}
        )
        if is_authoritative_depth:
            sample = _derive_authoritative_depth_sample(
                record, sample, sample_anchor
            )
        if (
            sample.get("visible") is not False
            and not sample.get("depth_tested", False)
            and "elevation" in sample
        ):
            elevation = _number(sample["elevation"], default=float("nan"))
            if math.isfinite(elevation):
                candidate.anchor = (
                    float(candidate.anchor[0]),
                    float(candidate.anchor[1]),
                    elevation,
                )
            else:
                sample = {
                    **dict(sample),
                    "visible": False,
                    "terrain_sample_invalid": True,
                }
        candidate.terrain_sample = _json_safe(dict(sample))
        details = dict(candidate.details or {})
        if sample.get("visible") is False:
            details["visible"] = False
            gates = list(details.get("visibility_gates") or ())
            gates.append(
                {
                    "kind": "occlusion",
                    "terrain_sample": _json_safe(dict(sample)),
                }
            )
            details["visibility_gates"] = gates
        else:
            details.setdefault("visible", True)
        candidate.details = _json_safe(details)
        if sample.get("visible") is not False:
            continue
        records.append(
            {
                "kind": "occluded_anchor",
                "label_id": label_id,
                "source_id": source_id,
                "candidate_id": candidate.candidate_id,
                "terrain_sample": _json_safe(dict(sample)),
            }
        )
    return records


def _candidate_constraint_records(
    *,
    label_id: str,
    source_id: str,
    candidates: Sequence["LabelCandidate"],
    viewport_size: tuple[float, float] | None,
    keepouts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        details = dict(candidate.details or {})
        gates = list(details.get("visibility_gates") or ())
        bounds = _rect_bounds(candidate.bounds)
        if viewport_size is not None and bounds is not None:
            width, height = viewport_size
            if (
                bounds[0] < 0.0
                or bounds[1] < 0.0
                or bounds[2] > width
                or bounds[3] > height
            ):
                gate = {
                    "kind": "viewport",
                    "candidate_bounds": bounds,
                    "viewport": [width, height],
                }
                gates.append(gate)
                records.append(
                    {
                        **gate,
                        "kind": "candidate_ineligible",
                        "gate": "viewport",
                        "label_id": label_id,
                        "source_id": source_id,
                        "candidate_id": candidate.candidate_id,
                    }
                )
        for keepout in keepouts:
            if not _rects_intersect(candidate.bounds, keepout.get("bounds")):
                continue
            gate = {
                "kind": "keepout",
                "candidate_bounds": list(bounds or ()),
                "keepout_bounds": list(keepout["bounds"]),
                "keepout_kind": keepout["kind"],
                "keepout_region_id": keepout["region_id"],
            }
            gates.append(gate)
            records.append(
                {
                    **gate,
                    "kind": "candidate_ineligible",
                    "gate": "keepout",
                    "label_id": label_id,
                    "source_id": source_id,
                    "candidate_id": candidate.candidate_id,
                }
            )
        if gates:
            details["visible"] = False
            details["visibility_gates"] = gates
            candidate.details = _json_safe(details)
    return records


def _ineligible_rejection_reason(candidates: Sequence["LabelCandidate"]) -> str:
    gate_sets = [
        {str(gate.get("kind")) for gate in (candidate.details or {}).get("visibility_gates", ())}
        for candidate in candidates
    ]
    if gate_sets and all("occlusion" in gates for gates in gate_sets):
        return "terrain_occluded"
    if gate_sets and all("viewport" in gates for gates in gate_sets):
        return "outside_view"
    if gate_sets and all("keepout" in gates for gates in gate_sets):
        return "keepout_region"
    return "no_eligible_candidate"


def _label_screen_size(
    record: Mapping[str, Any],
    text: str,
    shaping_details: Mapping[str, Any],
    typography: Mapping[str, Any] | None,
) -> tuple[float, float]:
    """Return a deterministic, non-degenerate screen footprint.

    An explicit ``screen_bounds``/``label_size`` is authoritative. Otherwise
    the already-shaped advances are used; this sizes the solver box without
    laying out glyphs a second time.
    """
    explicit = _rect_bounds(record.get("screen_bounds"))
    if explicit is not None and explicit[2] > explicit[0] and explicit[3] > explicit[1]:
        return explicit[2] - explicit[0], explicit[3] - explicit[1]
    size = record.get("label_size")
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)) and len(size) >= 2:
        width, height = _number(size[0]), _number(size[1])
        if width > 0.0 and height > 0.0:
            return width, height
    normalized = _normalize_typography(typography or record.get("typography") or {})
    font_size = _number(
        normalized.get("font_size", normalized.get("size", normalized.get("text_size", 12.0))),
        default=12.0,
    )
    if not math.isfinite(font_size) or font_size <= 0.0:
        font_size = 12.0
    positioned = shaping_details.get("positioned_glyphs")
    extents: list[float] = []
    if isinstance(positioned, Sequence) and not isinstance(positioned, (str, bytes)):
        for glyph in positioned:
            if not isinstance(glyph, Mapping):
                continue
            origin = glyph.get("origin")
            glyph_advance = glyph.get("advance")
            if (
                isinstance(origin, Sequence)
                and isinstance(glyph_advance, Sequence)
                and len(origin) >= 1
                and len(glyph_advance) >= 1
            ):
                left = _number(origin[0])
                extents.extend((left, left + _number(glyph_advance[0])))
    if extents:
        advance = max(extents) - min(extents)
    else:
        advances = shaping_details.get("advances")
        if isinstance(advances, Sequence) and not isinstance(advances, (str, bytes)):
            # HarfBuzz advances use the packaged font's 26.6 fixed-point scale.
            advance = sum(abs(_number(value)) for value in advances) / 64.0
        else:
            advance = max(1.0, len(text) * 0.6)
    return max(1.0, advance * font_size), max(1.0, font_size)


def _bounds_at_anchor(anchor: Sequence[float], size: tuple[float, float]) -> list[float]:
    width, height = size
    return [
        float(anchor[0]) - width * 0.5,
        float(anchor[1]) - height * 0.5,
        float(anchor[0]) + width * 0.5,
        float(anchor[1]) + height * 0.5,
    ]


def _ensure_candidate_bounds(
    candidates: Sequence["LabelCandidate"], size: tuple[float, float]
) -> None:
    for candidate in candidates:
        bounds = _rect_bounds(candidate.bounds)
        if bounds is None or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            candidate.bounds = tuple(_bounds_at_anchor(candidate.anchor, size))
        else:
            candidate.bounds = tuple(bounds)


def _validated_authority_glyphs(value: Any) -> list[dict[str, Any]] | None:
    """Validate and normalize one geometry-authority glyph stream.

    Authority geometry is render input, not advisory metadata.  Validate every
    numeric value before it can enter a candidate or the canonical plan, while
    preserving authority-provided fields verbatim apart from numeric
    normalization of the required compositor fields.
    """
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        return None

    validated: list[dict[str, Any]] = []
    for glyph in value:
        if not isinstance(glyph, Mapping):
            return None
        try:
            item = _json_safe(dict(glyph))
        except TypeError:
            return None

        def all_finite(payload: Any) -> bool:
            if isinstance(payload, Mapping):
                return all(all_finite(child) for child in payload.values())
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
                return all(all_finite(child) for child in payload)
            return not isinstance(payload, float) or math.isfinite(payload)

        if not all_finite(item):
            return None

        for key in ("glyph_id", "font_index"):
            raw_integer = item.get(key)
            if isinstance(raw_integer, bool) or not isinstance(raw_integer, int):
                return None
            if raw_integer < 0:
                return None

        origin = item.get("origin")
        if (
            not isinstance(origin, Sequence)
            or isinstance(origin, (str, bytes))
            or len(origin) < 2
        ):
            return None
        normalized_origin = [
            _number(component, default=float("nan")) for component in origin
        ]
        if not all(math.isfinite(component) for component in normalized_origin):
            return None

        rotation = _number(item.get("rotation"), default=float("nan"))
        if not math.isfinite(rotation):
            return None

        if "advance" in item:
            advance = item["advance"]
            if (
                not isinstance(advance, Sequence)
                or isinstance(advance, (str, bytes))
                or len(advance) < 2
            ):
                return None
            normalized_advance = [
                _number(component, default=float("nan")) for component in advance
            ]
            if not all(math.isfinite(component) for component in normalized_advance):
                return None
            item["advance"] = normalized_advance

        if "scale" in item:
            scale = _number(item["scale"], default=float("nan"))
            if not math.isfinite(scale) or scale <= 0.0:
                return None
            item["scale"] = scale

        for key in ("cluster", "line_index"):
            if key not in item:
                continue
            raw_integer = item[key]
            if isinstance(raw_integer, bool) or not isinstance(raw_integer, int):
                return None
            if raw_integer < 0:
                return None
        if "has_outline" in item and not isinstance(item["has_outline"], bool):
            return None

        item["origin"] = normalized_origin
        item["rotation"] = rotation
        validated.append(item)
    return validated


def _authority_candidates(
    record: Mapping[str, Any],
    *,
    label_id: str,
    score: float,
    ordering_key: str,
    terrain_sample: Mapping[str, Any],
) -> tuple[list["LabelCandidate"], Sequence[Mapping[str, Any]]] | None:
    """Decode geometry emitted by the curved/line geometry authorities.

    The compiler consumes candidate anchors, bounds, and optional positioned
    glyphs verbatim. It deliberately performs no curve or line re-layout.
    """
    authority = record.get("geometry_authority")
    if not isinstance(authority, Mapping):
        return None
    raw_candidates = authority.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        return None
    source = str(authority.get("source", ""))
    if source not in {"layout_curved_text", "compute_line_label_placement"}:
        return None

    shared_positioned = _validated_authority_glyphs(
        authority.get("positioned_glyphs")
    )
    if shared_positioned is None:
        return None

    candidates: list[LabelCandidate] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            return None
        anchor = _strict_projected_anchor(raw.get("anchor"))
        bounds = _rect_bounds(raw.get("bounds"))
        if anchor is None or bounds is None or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return None
        details = dict(raw.get("details") or {})
        details["geometry_authority"] = source
        if "visible" in raw:
            details["visible"] = bool(raw["visible"])
        if "positioned_glyphs" in raw:
            positioned = _validated_authority_glyphs(raw["positioned_glyphs"])
            if positioned is None:
                return None
        else:
            positioned = shared_positioned
        # Every candidate carries only geometry that has already passed the
        # complete authority-stream validation.  This prevents an unselected
        # malformed stream from entering the canonical plan/hash, and lets the
        # native choice reconstruct the exact candidate-specific glyphs.
        details["positioned_glyphs"] = positioned
        candidates.append(
            LabelCandidate(
                candidate_id=str(raw.get("candidate_id") or f"{label_id}:authority-{index}"),
                candidate_type=str(raw.get("candidate_type") or "geometry_authority"),
                anchor=anchor,
                score=max(
                    0.0,
                    _number(raw.get("priority", raw.get("score", score)), default=score),
                ),
                bounds=bounds,
                terrain_sample=terrain_sample,
                details=details,
                ordering_key=f"{ordering_key}:{index:04d}:{source}",
            )
        )
    if not candidates:
        return None
    return candidates, shared_positioned


def _candidate_policy(record: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = record.get("candidate_policy")
    return dict(policy) if isinstance(policy, Mapping) else {}


def _priority_score(record: Mapping[str, Any], priority_ranks: Mapping[str, int]) -> float:
    priority_class = str(record.get("priority_class", "default"))
    rank = int(priority_ranks.get(priority_class, 0))
    local_priority = _number(record.get("priority", 0))
    # A label with the default public priority still has positive placement
    # utility.  Without this unit baseline every default candidate has zero
    # objective weight after non-negative clamping, so an optimal solver may
    # select an arbitrary alternative or drop a conflict-free label entirely.
    return 1.0 + max(0.0, (rank * 1_000_000.0) + local_priority)


def _point_label_candidates(
    *,
    label_id: str,
    coords: Sequence[float],
    score: float,
    ordering_key: str,
    record: Mapping[str, Any],
    seed: int,
    terrain_sample: Mapping[str, Any],
) -> list[LabelCandidate]:
    x, y, z = coords
    policy = _candidate_policy(record)
    offset = _number(policy.get("offset_px", record.get("candidate_offset_px", 12.0)), default=12.0)
    radial_radius = _number(
        policy.get("radial_radius_px", record.get("radial_radius_px", offset * 1.5)),
        default=offset * 1.5,
    )
    radial_count = max(0, int(_number(policy.get("radial_count", record.get("radial_count", 4)), default=4.0)))
    radial_jitter_deg = max(
        0.0,
        _number(policy.get("radial_jitter_deg", record.get("radial_jitter_deg", 0.0)), default=0.0),
    )

    candidates: list[LabelCandidate] = []

    def add_candidate(
        suffix: str,
        candidate_type: str,
        anchor: Sequence[float],
        order: int,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        candidates.append(
            LabelCandidate(
                candidate_id=f"{label_id}:{suffix}",
                candidate_type=candidate_type,
                anchor=anchor,
                score=max(0.0, score - (order * 0.001)),
                bounds=[anchor[0], anchor[1], anchor[0], anchor[1]],
                terrain_sample=terrain_sample,
                details=details or {},
                ordering_key=f"{ordering_key}:{order:02d}:{suffix}",
            )
        )

    add_candidate("center", "center", [x, y, z], 0)
    add_candidate("above", "above", [x, y - offset, z], 1, details={"offset_px": offset})
    add_candidate("below", "below", [x, y + offset, z], 2, details={"offset_px": offset})
    add_candidate("left", "left", [x - offset, y, z], 3, details={"offset_px": offset})
    add_candidate("right", "right", [x + offset, y, z], 4, details={"offset_px": offset})

    for index in range(radial_count):
        base_angle = (360.0 / radial_count) * index if radial_count else 0.0
        jitter = (_stable_unit_interval(seed, label_id, index, "radial") - 0.5) * 2.0 * radial_jitter_deg
        angle = (base_angle + jitter) % 360.0
        radians = math.radians(angle)
        anchor = [
            x + math.cos(radians) * radial_radius,
            y + math.sin(radians) * radial_radius,
            z,
        ]
        add_candidate(
            f"radial-{index}",
            "radial",
            anchor,
            5 + index,
            details={
                "angle_deg": round(angle, 6),
                "jitter_deg": round(jitter, 6),
                "radial_index": index,
                "radius_px": radial_radius,
            },
        )

    if bool(record.get("leader_line")) or str(record.get("placement_preset", "")).lower() in {"callout", "leader"}:
        leader_anchor = [x + offset, y - offset, z]
        candidates.insert(
            0,
            LabelCandidate(
                candidate_id=f"{label_id}:leader",
                candidate_type="leader_line",
                anchor=leader_anchor,
                score=score + 0.01,
                bounds=[leader_anchor[0], leader_anchor[1], leader_anchor[0], leader_anchor[1]],
                terrain_sample=terrain_sample,
                details={
                    "leader_line": True,
                    "placement_preset": str(record.get("placement_preset", "callout")),
                    "anchor": [x, y, z],
                    "offset_px": offset,
                },
                ordering_key=f"{ordering_key}:00:leader",
            ),
        )

    return candidates


def _polygon_ring(geometry: Mapping[str, Any]) -> list[list[float]] | None:
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes)) or not coordinates:
        return None
    ring_data = coordinates[0]
    if not isinstance(ring_data, Sequence) or isinstance(ring_data, (str, bytes)):
        return None
    ring: list[list[float]] = []
    for point in ring_data:
        coords = _coordinates(point)
        if coords is None:
            return None
        ring.append(coords)
    if len(ring) < 4:
        return None
    if ring[0][:2] != ring[-1][:2]:
        ring.append(list(ring[0]))
    unique_xy = {(point[0], point[1]) for point in ring[:-1]}
    if len(unique_xy) < 3:
        return None
    return ring


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for left, right in zip(ring, ring[1:]):
        total += (left[0] * right[1]) - (right[0] * left[1])
    return total * 0.5


def _polygon_centroid(ring: Sequence[Sequence[float]], area: float) -> list[float]:
    cx = 0.0
    cy = 0.0
    for left, right in zip(ring, ring[1:]):
        cross = (left[0] * right[1]) - (right[0] * left[1])
        cx += (left[0] + right[0]) * cross
        cy += (left[1] + right[1]) * cross
    factor = 1.0 / (6.0 * area)
    return [cx * factor, cy * factor, 0.0]


def _point_in_polygon(x: float, y: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    for left, right in zip(ring, ring[1:]):
        x0, y0 = left[0], left[1]
        x1, y1 = right[0], right[1]
        intersects = (y0 > y) != (y1 > y)
        if intersects:
            x_at_y = ((x1 - x0) * (y - y0) / (y1 - y0)) + x0
            if x < x_at_y:
                inside = not inside
    return inside


def _polygon_visual_center(ring: Sequence[Sequence[float]], fallback: Sequence[float]) -> list[float]:
    xs = [point[0] for point in ring[:-1]]
    ys = [point[1] for point in ring[:-1]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x or min_y == max_y:
        return [fallback[0], fallback[1], 0.0]

    best: tuple[float, float, float] | None = None
    steps = 12
    for ix in range(steps):
        x = min_x + ((ix + 0.5) * (max_x - min_x) / steps)
        for iy in range(steps):
            y = min_y + ((iy + 0.5) * (max_y - min_y) / steps)
            if not _point_in_polygon(x, y, ring):
                continue
            distance = min((x - point[0]) ** 2 + (y - point[1]) ** 2 for point in ring[:-1])
            candidate = (distance, x, y)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return [fallback[0], fallback[1], 0.0]
    return [best[1], best[2], 0.0]


def _polygon_label_candidates(
    *,
    label_id: str,
    geometry: Mapping[str, Any],
    score: float,
    ordering_key: str,
    terrain_sample: Mapping[str, Any],
) -> tuple[LabelCandidate, list[LabelCandidate]] | None:
    ring = _polygon_ring(geometry)
    if ring is None:
        return None
    area = _ring_area(ring)
    if abs(area) < 1.0e-9:
        return None

    centroid = _polygon_centroid(ring, area)
    centroid_inside = _point_in_polygon(centroid[0], centroid[1], ring)
    visual_center = _polygon_visual_center(ring, centroid)

    centroid_candidate = LabelCandidate(
        candidate_id=f"{label_id}:centroid",
        candidate_type="centroid",
        anchor=centroid,
        score=score,
        bounds=[centroid[0], centroid[1], centroid[0], centroid[1]],
        terrain_sample=terrain_sample,
        details={
            "area": abs(area),
            "inside_polygon": centroid_inside,
            "visible": centroid_inside,
            **(
                {}
                if centroid_inside
                else {
                    "visibility_gates": [
                        {
                            "kind": "geometry",
                            "reason": "centroid_outside_polygon",
                        }
                    ]
                }
            ),
        },
        ordering_key=f"{ordering_key}:00:centroid",
    )
    visual_candidate = LabelCandidate(
        candidate_id=f"{label_id}:visual-center",
        candidate_type="visual_center",
        anchor=visual_center,
        score=max(0.0, score - 0.001),
        bounds=[visual_center[0], visual_center[1], visual_center[0], visual_center[1]],
        terrain_sample=terrain_sample,
        details={"area": abs(area), "fallback_for": "centroid"},
        ordering_key=f"{ordering_key}:01:visual-center",
    )
    selected = centroid_candidate if centroid_inside else visual_candidate
    return selected, [centroid_candidate, visual_candidate]


def _label_sort_key(record: Mapping[str, Any], fallback_key: str) -> tuple[str, str, str]:
    label_id = str(record.get("id", fallback_key))
    text = str(record.get("text", ""))
    geometry = record.get("geometry") if isinstance(record.get("geometry"), Mapping) else {}
    geometry_type = str(geometry.get("type", record.get("geometry_type", "")))
    return (label_id, geometry_type, text)


def _normalize_typography(value: Mapping[str, Any] | None) -> dict[str, Any]:
    typography = dict(value or {})
    if "halo_width_px" not in typography:
        for key in ("halo_width", "text_halo_width"):
            if key in typography:
                typography["halo_width_px"] = typography[key]
                break
    if "halo_width_px" in typography:
        typography.setdefault("halo_width", typography["halo_width_px"])
        typography.setdefault("text_halo_width", typography["halo_width_px"])
    if "halo_color" not in typography and "text_halo_color" in typography:
        typography["halo_color"] = typography["text_halo_color"]
    if "halo_color" in typography:
        typography.setdefault("text_halo_color", typography["halo_color"])
    return typography


@dataclass
class LabelCandidate:
    candidate_id: str
    candidate_type: str
    anchor: Sequence[float]
    score: float = 0.0
    bounds: Sequence[float] | None = None
    terrain_sample: Mapping[str, Any] | None = None
    details: Mapping[str, Any] | None = None
    ordering_key: str | None = None

    def __post_init__(self) -> None:
        self.anchor = tuple(float(value) for value in self.anchor)
        self.bounds = tuple(float(value) for value in self.bounds) if self.bounds is not None else None
        self.terrain_sample = _json_safe(dict(self.terrain_sample or {}))
        self.details = _json_safe(dict(self.details or {}))
        self.ordering_key = self.ordering_key or self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "anchor": list(self.anchor),
            "score": float(self.score),
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "terrain_sample": dict(self.terrain_sample or {}),
            "details": dict(self.details or {}),
            "ordering_key": self.ordering_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelCandidate":
        return cls(
            candidate_id=str(data["candidate_id"]),
            candidate_type=str(data["candidate_type"]),
            anchor=data["anchor"],
            score=float(data.get("score", 0.0)),
            bounds=data.get("bounds"),
            terrain_sample=data.get("terrain_sample") or {},
            details=data.get("details") or {},
            ordering_key=data.get("ordering_key"),
        )


@dataclass
class AcceptedLabel:
    label_id: str
    source_id: str
    text: str
    geometry_type: str
    candidate: LabelCandidate | Mapping[str, Any]
    candidates: Sequence[LabelCandidate | Mapping[str, Any]] = field(default_factory=tuple)
    priority_class: str = "default"
    screen_bounds: Sequence[float] | None = None
    world_bounds: Sequence[float] | None = None
    typography: Mapping[str, Any] | None = None
    glyphs: Sequence[str] = field(default_factory=tuple)
    line_ranges: Sequence[Sequence[int]] = field(default_factory=tuple)
    positioned_glyphs: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    ordering_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, LabelCandidate):
            self.candidate = LabelCandidate.from_dict(self.candidate)
        candidate_items = self.candidates or (self.candidate,)
        self.candidates = sorted(
            (
                candidate if isinstance(candidate, LabelCandidate) else LabelCandidate.from_dict(candidate)
                for candidate in candidate_items
            ),
            key=lambda candidate: candidate.ordering_key or candidate.candidate_id,
        )
        if not any(candidate.candidate_id == self.candidate.candidate_id for candidate in self.candidates):
            self.candidates.insert(0, self.candidate)
        self.screen_bounds = (
            tuple(float(value) for value in self.screen_bounds) if self.screen_bounds is not None else None
        )
        self.world_bounds = (
            tuple(float(value) for value in self.world_bounds) if self.world_bounds is not None else None
        )
        self.typography = _json_safe(_normalize_typography(self.typography))
        self.glyphs = tuple(str(glyph) for glyph in self.glyphs)
        self.line_ranges = tuple(
            (int(item[0]), int(item[1])) for item in self.line_ranges
        )
        self.positioned_glyphs = tuple(_json_safe(dict(glyph)) for glyph in self.positioned_glyphs)
        self.ordering_key = self.ordering_key or self.label_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "source_id": self.source_id,
            "text": self.text,
            "geometry_type": self.geometry_type,
            "candidate": self.candidate.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "priority_class": self.priority_class,
            "screen_bounds": list(self.screen_bounds) if self.screen_bounds is not None else None,
            "world_bounds": list(self.world_bounds) if self.world_bounds is not None else None,
            "typography": dict(self.typography or {}),
            "glyphs": list(self.glyphs),
            "line_ranges": [list(item) for item in self.line_ranges],
            "positioned_glyphs": [dict(glyph) for glyph in self.positioned_glyphs],
            "ordering_key": self.ordering_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AcceptedLabel":
        return cls(
            label_id=str(data["label_id"]),
            source_id=str(data["source_id"]),
            text=str(data["text"]),
            geometry_type=str(data["geometry_type"]),
            candidate=data["candidate"],
            candidates=data.get("candidates") or (data["candidate"],),
            priority_class=str(data.get("priority_class", "default")),
            screen_bounds=data.get("screen_bounds"),
            world_bounds=data.get("world_bounds"),
            typography=data.get("typography") or {},
            glyphs=data.get("glyphs") or (),
            line_ranges=data.get("line_ranges") or (),
            positioned_glyphs=data.get("positioned_glyphs") or (),
            ordering_key=data.get("ordering_key"),
        )


@dataclass
class RejectedLabel:
    label_id: str
    source_id: str
    reason: str
    candidate_id: str | None = None
    diagnostic_refs: Sequence[str] = field(default_factory=tuple)
    ordering_key: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.reason not in REJECTION_REASONS:
            raise ValueError(f"Unknown label rejection reason: {self.reason!r}")
        self.diagnostic_refs = tuple(str(ref) for ref in self.diagnostic_refs)
        self.details = _json_safe(dict(self.details or {}))
        self.ordering_key = self.ordering_key or f"{self.label_id}:{self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "diagnostic_refs": list(self.diagnostic_refs),
            "ordering_key": self.ordering_key,
            "details": dict(self.details or {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RejectedLabel":
        return cls(
            label_id=str(data["label_id"]),
            source_id=str(data["source_id"]),
            reason=str(data["reason"]),
            candidate_id=data.get("candidate_id"),
            diagnostic_refs=data.get("diagnostic_refs") or (),
            ordering_key=data.get("ordering_key"),
            details=data.get("details") or {},
        )


@dataclass
class KeepoutRegion:
    region_id: str
    kind: str
    bounds: Sequence[float]
    priority: int = 0

    def __post_init__(self) -> None:
        self.bounds = tuple(float(value) for value in self.bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "bounds": list(self.bounds),
            "priority": int(self.priority),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KeepoutRegion":
        return cls(
            region_id=str(data["region_id"]),
            kind=str(data["kind"]),
            bounds=data["bounds"],
            priority=int(data.get("priority", 0)),
        )


@dataclass
class PriorityClass:
    name: str
    rank: int = 0
    tie_break_policy: str = "stable_ordering_key"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rank": int(self.rank),
            "tie_break_policy": self.tie_break_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorityClass":
        return cls(
            name=str(data["name"]),
            rank=int(data.get("rank", 0)),
            tie_break_policy=str(data.get("tie_break_policy", "stable_ordering_key")),
        )


def _priority_payload(priority_rules: Sequence[PriorityClass | Mapping[str, Any]] | str | None) -> list[dict[str, Any]]:
    if priority_rules is None:
        return []
    if isinstance(priority_rules, str):
        if priority_rules == "cartographic":
            return [dict(item) for item in CARTOGRAPHIC_PRIORITY_PRESET]
        raise ValueError(f"Unknown priority preset: {priority_rules!r}")
    return [
        item.to_dict() if isinstance(item, PriorityClass) else PriorityClass.from_dict(item).to_dict()
        for item in priority_rules
    ]


@dataclass
class LabelPlan:
    accepted: Sequence[AcceptedLabel | Mapping[str, Any]]
    rejected: Sequence[RejectedLabel | Mapping[str, Any]]
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] = field(default_factory=tuple)
    bounds: Mapping[str, Any] | None = None
    seed: int = 0
    payload_version: int = PAYLOAD_VERSION
    rationale: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.rationale = tuple(_json_safe(dict(record)) for record in self.rationale or ())
        self.accepted = sorted(
            (
                label if isinstance(label, AcceptedLabel) else AcceptedLabel.from_dict(label)
                for label in self.accepted
            ),
            key=lambda label: label.ordering_key or label.label_id,
        )
        self.rejected = sorted(
            (
                label if isinstance(label, RejectedLabel) else RejectedLabel.from_dict(label)
                for label in self.rejected
            ),
            key=lambda label: label.ordering_key or label.label_id,
        )
        self.diagnostics = sorted(
            (
                diagnostic
                if isinstance(diagnostic, Diagnostic)
                else Diagnostic.from_dict(diagnostic)
                for diagnostic in self.diagnostics
            ),
            key=lambda diagnostic: diagnostic.sort_key(),
        )
        self.bounds = _json_safe(dict(self.bounds or {"screen": None, "world": None}))
        self.seed = int(self.seed)
        self.payload_version = int(self.payload_version)

    @classmethod
    def compile(
        cls,
        *,
        labels: Any,
        camera: Any,
        viewport: Any,
        terrain: Any | None = None,
        keepouts: Sequence[KeepoutRegion | Mapping[str, Any]] = (),
        priority_rules: Sequence[PriorityClass | Mapping[str, Any]] | None = None,
        typography: Mapping[str, Any] | None = None,
        glyph_atlas: Any | None = None,
        seed: int = 0,
        declutter: str = "optimal",
        gap_tolerance: float = 0.02,
        declutter_node_budget: int = 200_000,
    ) -> "LabelPlan":
        if declutter not in {"optimal", "greedy"}:
            raise ValueError("LabelPlan.compile declutter must be 'optimal' or 'greedy'")
        try:
            label_count = len(labels)
        except TypeError:
            label_count = None
        if label_count is not None and label_count > MAX_LABEL_RECORDS:
            raise ValueError(
                f"LabelPlan.compile label count {label_count} exceeds the safe limit "
                f"of {MAX_LABEL_RECORDS}"
            )
        viewport_size = _viewport_size(viewport)
        if (
            viewport_size is None
            or not all(math.isfinite(value) and value > 0.0 for value in viewport_size)
        ):
            raise ValueError("LabelPlan.compile viewport must be finite and positive")
        keepout_payload = [
            region.to_dict() if isinstance(region, KeepoutRegion) else KeepoutRegion.from_dict(region).to_dict()
            for region in keepouts
        ]
        priority_payload = _priority_payload(priority_rules)
        priority_ranks = {str(item["name"]): int(item["rank"]) for item in priority_payload}
        atlas_glyphs = _glyph_set(glyph_atlas)
        accepted: list[AcceptedLabel] = []
        rejected: list[RejectedLabel] = []
        diagnostics: list[Diagnostic] = []
        missing_by_label: dict[str, list[str]] = {}
        rationale_records: list[dict[str, Any]] = []

        records = sorted(
            _iter_label_records(labels),
            key=lambda item: (
                _label_sort_key(item[1], item[0]),
                _stable_json(item[1]),
            ),
        )
        instance_occurrences: dict[str, int] = {}
        for fallback_key, raw_record in records:
            # Compile against an owned copy because projection metadata is
            # resolved below.  A deterministic occurrence ordinal below keeps
            # byte-identical duplicate public/source ids distinct without
            # making list input order affect otherwise different records.
            record = dict(raw_record)
            label_id = str(record.get("id", fallback_key))
            source_id = str(record.get("source_id", label_id))
            text = str(record.get("text", ""))
            instance_base = f"{label_id}:{source_id}:{_stable_json(record)}"
            instance_ordinal = instance_occurrences.get(instance_base, 0)
            instance_occurrences[instance_base] = instance_ordinal + 1
            ordering_key = f"{instance_base}:{instance_ordinal:08d}"
            glyph_sequence, shaping_details = _shape_label_glyphs(
                text, glyph_atlas, record.get("line_ranges")
            )

            if not text.strip():
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="empty_text",
                        ordering_key=ordering_key,
                    )
                )
                continue

            if glyph_sequence is None:
                shaping_diagnostics = list(shaping_details.get("diagnostics", ()))
                reason = str(
                    shaping_diagnostics[0].get("reason", "shaping_failed")
                    if shaping_diagnostics
                    else "shaping_failed"
                )
                native_reason = str(shaping_details.get("native_reason", reason))
                if reason not in REJECTION_REASONS:
                    reason = "shaping_failed"
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason=reason,
                        ordering_key=ordering_key,
                        details={**dict(shaping_details), "native_reason": native_reason},
                    )
                )
                continue

            missing = sorted({char for char in glyph_sequence if atlas_glyphs is not None and char not in atlas_glyphs})
            if missing:
                missing_by_label[label_id] = missing
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="missing_glyph",
                        diagnostic_refs=["missing_glyphs"],
                        ordering_key=ordering_key,
                        details={"missing_glyphs": missing},
                    )
                )
                continue

            geometry = record.get("geometry") if isinstance(record.get("geometry"), Mapping) else {}
            geometry_type = str(geometry.get("type", record.get("geometry_type", "Point")))
            geometry_type_key = geometry_type.lower()
            terrain_sample: Mapping[str, Any] = {}
            score = _priority_score(record, priority_ranks)
            requires_projection = bool(
                getattr(terrain, "requires_projected_anchor", False)
            )
            authority = _authority_candidates(
                record,
                label_id=label_id,
                score=score,
                ordering_key=ordering_key,
                terrain_sample=terrain_sample,
            )
            is_line = geometry_type_key == "linestring"
            is_curved = bool(record.get("curved_text")) or str(
                record.get("placement_preset", "")
            ).lower() == "curved"
            required_authority = (
                "layout_curved_text" if is_curved else "compute_line_label_placement"
            )
            authority_payload = record.get("geometry_authority")
            authority_source = (
                str(authority_payload.get("source", ""))
                if isinstance(authority_payload, Mapping)
                else ""
            )
            if (is_line or is_curved) and (
                authority is None or authority_source != required_authority
            ):
                diagnostics.append(
                    Diagnostic(
                        code="label_geometry_authority_missing",
                        severity="error",
                        message="Line and curved labels require authoritative positioned glyph geometry.",
                        remediation=(
                            f"Provide geometry_authority.source={required_authority!r} with "
                            "nonempty candidates and positioned_glyphs."
                        ),
                        support_level="unsupported",
                        layer_id="labels",
                        object_id=label_id,
                        details={"required_authority": required_authority},
                    )
                )
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="missing_geometry_authority",
                        diagnostic_refs=["label_geometry_authority_missing"],
                        ordering_key=ordering_key,
                        details={"required_authority": required_authority},
                    )
                )
                continue
            if authority is not None and requires_projection:
                projection_declaration = str(
                    authority_payload.get("projection_authority", "")
                    if isinstance(authority_payload, Mapping)
                    else ""
                ).lower()
                if projection_declaration not in {"deterministic", "authoritative"}:
                    diagnostics.append(_projection_diagnostic(label_id))
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="missing_projection_authority",
                            diagnostic_refs=["label_projection_authority_missing"],
                            ordering_key=ordering_key,
                            details={
                                "required": "geometry_authority.projection_authority"
                            },
                        )
                    )
                    continue
                record["projection_authority"] = projection_declaration

            authority_positioned: Sequence[Mapping[str, Any]] = ()
            if authority is not None:
                candidates, authority_positioned = authority
                candidate = candidates[0]
                authority_positioned = tuple(
                    (candidate.details or {}).get("positioned_glyphs") or ()
                )
                x, y, z = candidate.anchor
                screen_bounds = list(candidate.bounds or ())
                world_coords = _coordinates(
                    geometry.get("coordinates", record.get("position", record.get("world_pos")))
                )
                if world_coords is None or geometry_type_key != "point":
                    world_bounds = [x, y, z, x, y, z]
                else:
                    world_bounds = [*world_coords, *world_coords]
            elif geometry_type_key == "point":
                world_coords = _coordinates(
                    geometry.get(
                        "coordinates", record.get("position", record.get("world_pos"))
                    )
                )
                coords = world_coords
                if coords is None:
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="invalid_geometry",
                            ordering_key=ordering_key,
                        )
                    )
                    continue

                projected, projection_authority = _project_anchor(
                    record, camera, viewport, coords
                )
                if requires_projection and projected is None:
                    diagnostics.append(_projection_diagnostic(label_id))
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="missing_projection_authority",
                            diagnostic_refs=["label_projection_authority_missing"],
                            ordering_key=ordering_key,
                            details={"required": "finite projected_anchor[x,y,depth]"},
                        )
                    )
                    continue
                coords = projected if projected is not None else list(coords)
                if projected is not None:
                    record["projected_depth"] = coords[2]
                    record["projection_authority"] = projection_authority

                x, y, z = coords
                screen_bounds = [x, y, x, y]
                if projected is not None:
                    assert world_coords is not None
                    world_bounds = [*world_coords, *world_coords]
                else:
                    world_bounds = [x, y, z, x, y, z]
                candidates = _point_label_candidates(
                    label_id=label_id,
                    coords=coords,
                    score=score,
                    ordering_key=ordering_key,
                    record=record,
                    seed=int(seed),
                    terrain_sample=terrain_sample,
                )
                candidate = candidates[0]
            elif geometry_type_key == "polygon":
                if requires_projection:
                    diagnostics.append(_projection_diagnostic(label_id))
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="missing_projection_authority",
                            diagnostic_refs=["label_projection_authority_missing"],
                            ordering_key=ordering_key,
                            details={"required": "projected polygon candidates"},
                        )
                    )
                    continue
                polygon_candidates = _polygon_label_candidates(
                    label_id=label_id,
                    geometry=geometry,
                    score=score,
                    ordering_key=ordering_key,
                    terrain_sample=terrain_sample,
                )
                if polygon_candidates is None:
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="invalid_geometry",
                            ordering_key=ordering_key,
                        )
                    )
                    continue
                candidate, candidates = polygon_candidates
                x, y, z = candidate.anchor
                screen_bounds = list(candidate.bounds or [x, y, x, y])
                world_bounds = [x, y, z, x, y, z]
            else:
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="unsupported_geometry_type",
                        ordering_key=ordering_key,
                        details={"geometry_type": geometry_type},
                    )
                )
                continue

            _ensure_candidate_bounds(
                candidates,
                _label_screen_size(record, text, shaping_details, typography),
            )
            candidate = next(
                item for item in candidates if item.candidate_id == candidate.candidate_id
            )
            explicit_bounds = _rect_bounds(record.get("screen_bounds"))
            if authority is None and explicit_bounds is not None:
                primary_x, primary_y = candidate.anchor[:2]
                for item in candidates:
                    dx = float(item.anchor[0]) - float(primary_x)
                    dy = float(item.anchor[1]) - float(primary_y)
                    item.bounds = tuple(
                        (
                            explicit_bounds[0] + dx,
                            explicit_bounds[1] + dy,
                            explicit_bounds[2] + dx,
                            explicit_bounds[3] + dy,
                        )
                    )
            screen_bounds = list(candidate.bounds or screen_bounds)

            visibility_records = _candidate_visibility_records(
                record, terrain, label_id, source_id, candidates
            )
            terrain_sample = dict(candidate.terrain_sample or {})
            if (
                authority is None
                and geometry_type_key == "point"
                and not record.get("projection_authority")
                and world_coords is not None
                and terrain_sample.get("visible") is not False
                and not terrain_sample.get("depth_tested", False)
                and "elevation" in terrain_sample
            ):
                grounded_z = float(candidate.anchor[2])
                world_bounds = [
                    float(world_coords[0]),
                    float(world_coords[1]),
                    grounded_z,
                    float(world_coords[0]),
                    float(world_coords[1]),
                    grounded_z,
                ]
            rationale_records.extend(visibility_records)
            rationale_records.extend(
                _candidate_constraint_records(
                    label_id=label_id,
                    source_id=source_id,
                    candidates=candidates,
                    viewport_size=viewport_size,
                    keepouts=keepout_payload,
                )
            )
            visible_candidates = [
                item
                for item in candidates
                if (item.details or {}).get("visible") is not False
                and (item.terrain_sample or {}).get("visible") is not False
            ]
            if not visible_candidates:
                diagnostic_refs = ["label_rejection_summary"]
                projection_missing = any(
                    bool((item.terrain_sample or {}).get("projection_authority_missing"))
                    for item in candidates
                )
                if projection_missing:
                    diagnostics.append(_projection_diagnostic(label_id))
                    diagnostic_refs.append("label_projection_authority_missing")
                depth_incompatible = any(
                    bool((item.terrain_sample or {}).get("depth_convention_incompatible"))
                    for item in candidates
                )
                if depth_incompatible:
                    diagnostics.append(_depth_convention_diagnostic(label_id))
                    diagnostic_refs.append("label_depth_convention_incompatible")
                if terrain_sample.get("unavailable") is True:
                    diagnostics.append(
                        placeholder_fallback_diagnostic(
                            "terrain_sampler",
                            layer_id="labels",
                            object_id=label_id,
                        )
                    )
                    diagnostic_refs.append("placeholder_fallback")
                reason = (
                    "missing_projection_authority"
                    if projection_missing
                    else (
                        "incompatible_depth_convention"
                        if depth_incompatible
                        else _ineligible_rejection_reason(candidates)
                    )
                )
                gate_details = {
                    item.candidate_id: list(
                        (item.details or {}).get("visibility_gates") or ()
                    )
                    for item in candidates
                }
                rejection_details: dict[str, Any] = {
                    "terrain_sample": terrain_sample,
                    "candidate_gates": gate_details,
                }
                if reason == "keepout_region":
                    keepout_gate = next(
                        (
                            gate
                            for item in candidates
                            for gate in (item.details or {}).get(
                                "visibility_gates", ()
                            )
                            if gate.get("kind") == "keepout"
                        ),
                        None,
                    )
                    if keepout_gate is not None:
                        rejection_details.update(
                            {
                                key: keepout_gate[key]
                                for key in (
                                    "keepout_bounds",
                                    "keepout_kind",
                                    "keepout_region_id",
                                )
                            }
                        )
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason=reason,
                        candidate_id=candidate.candidate_id,
                        diagnostic_refs=diagnostic_refs,
                        ordering_key=ordering_key,
                        details=rejection_details,
                    )
                )
                continue
            accepted.append(
                AcceptedLabel(
                    label_id=label_id,
                    source_id=source_id,
                    text=text,
                    geometry_type=geometry_type,
                    candidate=candidate,
                    candidates=candidates,
                    priority_class=str(record.get("priority_class", "default")),
                    screen_bounds=screen_bounds,
                    world_bounds=world_bounds,
                    typography={
                        **_normalize_typography(typography or record.get("typography") or {}),
                        **{
                            key: value
                            for key, value in shaping_details.items()
                            if key not in {"line_ranges", "positioned_glyphs"}
                        },
                    },
                    glyphs=list(glyph_sequence),
                    line_ranges=shaping_details.get("line_ranges", ()),
                    positioned_glyphs=(
                        authority_positioned
                        if authority_positioned
                        else shaping_details.get("positioned_glyphs", ())
                    ),
                    ordering_key=ordering_key,
                )
            )

        accepted, collision_rejections, solve_records = _resolve_label_placements(
            accepted,
            declutter=declutter,
            gap_tolerance=float(gap_tolerance),
            node_budget=int(declutter_node_budget),
            diagnostics=diagnostics,
        )
        rejected.extend(collision_rejections)
        rationale_records.extend(solve_records)

        for label_id, glyphs in sorted(missing_by_label.items()):
            diagnostics.append(missing_glyphs_diagnostic(glyphs, layer_id="labels", object_id=label_id))

        if rejected:
            counts: dict[str, int] = {}
            for item in rejected:
                counts[item.reason] = counts.get(item.reason, 0) + 1
            diagnostics.append(label_rejection_summary_diagnostic(counts, layer_id="labels"))

        bounds = _plan_bounds(accepted)
        bounds["keepouts"] = sorted(keepout_payload, key=lambda item: (item["priority"], item["kind"], item["region_id"]))
        bounds["priority_rules"] = sorted(priority_payload, key=lambda item: (item["rank"], item["name"]))
        return cls(
            accepted=accepted,
            rejected=rejected,
            diagnostics=diagnostics,
            bounds=bounds,
            seed=seed,
            rationale=rationale_records,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_version": self.payload_version,
            "seed": self.seed,
            "accepted": [label.to_dict() for label in self.accepted],
            "rejected": [label.to_dict() for label in self.rejected],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "bounds": _json_safe(dict(self.bounds or {})),
            "rationale": [dict(record) for record in self.rationale],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelPlan":
        version = _payload_version(data.get("payload_version", PAYLOAD_VERSION))
        if version == 1:
            data = _migrate_payload_v1_to_v2(data)
            version = PAYLOAD_VERSION
        return cls(
            accepted=data.get("accepted") or (),
            rejected=data.get("rejected") or (),
            diagnostics=data.get("diagnostics") or (),
            bounds=data.get("bounds") or {},
            seed=int(data.get("seed", 0)),
            payload_version=version,
            rationale=data.get("rationale") or (),
        )

    def render_rationale(self) -> list[str]:
        """Human-readable design rationale derived solely from the recorded
        solver decisions — every line cites the actual geometry (overlap
        areas, displaced label ids, sampled depths) captured at solve time.
        """
        return [_render_rationale_record(record) for record in self.rationale]

    def canonical_bytes(self) -> bytes:
        """Return the public canonical byte representation of this plan."""
        from ._canonical_json import canonical_json_bytes

        return canonical_json_bytes(
            self.to_dict(), error_context="LabelPlan canonical serialization"
        )

    def plan_hash(self) -> str:
        """Return SHA-256 over :meth:`canonical_bytes`."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def _payload_with_backend(
        self,
        *,
        kind: str,
        backend: str | None,
        supported_backends: set[str],
    ) -> dict[str, Any]:
        backend_name = backend or "label_plan"
        payload = self.to_dict()
        payload["kind"] = kind
        payload["backend"] = backend_name
        payload["supported"] = backend_name in supported_backends
        if not payload["supported"]:
            payload["diagnostics"] = [
                *payload["diagnostics"],
                placeholder_fallback_diagnostic(
                    f"{kind}:{backend_name}",
                    layer_id="labels",
                ).to_dict(),
            ]
        return payload

    def to_render_payload(self, *, backend: str | None = None) -> dict[str, Any]:
        return self._payload_with_backend(
            kind="label_plan_render_payload",
            backend=backend,
            supported_backends={"default", "label_plan", "software"},
        )

    def to_export_payload(self, *, backend: str | None = None) -> dict[str, Any]:
        return self._payload_with_backend(
            kind="label_plan_export_payload",
            backend=backend,
            supported_backends={"default", "json", "label_plan"},
        )


def _plan_bounds(accepted: Sequence[AcceptedLabel]) -> dict[str, Any]:
    if not accepted:
        return {"screen": None, "world": None}

    screen_values = [label.screen_bounds for label in accepted if label.screen_bounds is not None]
    world_values = [label.world_bounds for label in accepted if label.world_bounds is not None]
    screen = None
    world = None
    if screen_values:
        screen = [
            min(bounds[0] for bounds in screen_values),
            min(bounds[1] for bounds in screen_values),
            max(bounds[2] for bounds in screen_values),
            max(bounds[3] for bounds in screen_values),
        ]
    if world_values:
        world = [
            min(bounds[0] for bounds in world_values),
            min(bounds[1] for bounds in world_values),
            min(bounds[2] for bounds in world_values),
            max(bounds[3] for bounds in world_values),
            max(bounds[4] for bounds in world_values),
            max(bounds[5] for bounds in world_values),
        ]
    return {"screen": screen, "world": world}


def _resolve_label_collisions(
    accepted: Sequence[AcceptedLabel],
) -> tuple[list[AcceptedLabel], list[RejectedLabel]]:
    winners: list[AcceptedLabel] = []
    rejected: list[RejectedLabel] = []
    solve_order = sorted(
        accepted,
        key=lambda label: (
            -float(label.candidate.score),
            label.ordering_key or label.label_id,
            label.label_id,
        ),
    )

    for label in solve_order:
        winner = next(
            (
                accepted_label
                for accepted_label in winners
                if _rects_intersect(label.screen_bounds, accepted_label.screen_bounds)
            ),
            None,
        )
        if winner is None:
            winners.append(label)
            continue

        label_score = float(label.candidate.score)
        winner_score = float(winner.candidate.score)
        reason = "priority_lost" if label_score < winner_score else "collision"
        rejected.append(
            RejectedLabel(
                label_id=label.label_id,
                source_id=label.source_id,
                reason=reason,
                candidate_id=label.candidate.candidate_id,
                diagnostic_refs=["label_rejection_summary"],
                ordering_key=label.ordering_key,
                details={
                    "collides_with": winner.label_id,
                    "collides_with_source_id": winner.source_id,
                    "candidate_bounds": list(label.screen_bounds or ()),
                    "winner_bounds": list(winner.screen_bounds or ()),
                    "candidate_priority": label_score,
                    "candidate_priority_class": label.priority_class,
                    "winner_priority": winner_score,
                    "winner_priority_class": winner.priority_class,
                },
            )
        )

    return winners, rejected


def _collision_rejection(label: AcceptedLabel, winner: AcceptedLabel) -> RejectedLabel:
    label_score = float(label.candidate.score)
    winner_score = float(winner.candidate.score)
    reason = "priority_lost" if label_score < winner_score else "collision"
    return RejectedLabel(
        label_id=label.label_id,
        source_id=label.source_id,
        reason=reason,
        candidate_id=label.candidate.candidate_id,
        diagnostic_refs=["label_rejection_summary"],
        ordering_key=label.ordering_key,
        details={
            "collides_with": winner.label_id,
            "collides_with_source_id": winner.source_id,
            "candidate_bounds": list(label.screen_bounds or ()),
            "winner_bounds": list(winner.screen_bounds or ()),
            "candidate_priority": label_score,
            "candidate_priority_class": label.priority_class,
            "winner_priority": winner_score,
            "winner_priority_class": winner.priority_class,
        },
    )


def _translate_native_rationale(
    native_rationale: Any,
    ordered: Sequence[AcceptedLabel],
    candidate_lookup: Mapping[tuple[int, int], LabelCandidate],
) -> list[dict[str, Any]]:
    """Map native solver records (index-keyed) back to plan label ids."""
    records: list[dict[str, Any]] = []
    for raw in native_rationale.records():
        record = dict(raw)
        kind = str(record.get("kind", ""))
        if kind in {
            "placed",
            "dropped",
            "occluded_candidate",
            "visibility_filtered_candidate",
        }:
            label_index = int(record.pop("label_id"))
            candidate_index = int(record.pop("candidate_index", 0))
            label = ordered[label_index]
            candidate = candidate_lookup[(label_index, candidate_index)]
            visibility_gates = list(
                (candidate.details or {}).get("visibility_gates", ())
            )
            gates = {str(gate.get("kind")) for gate in visibility_gates}
            if kind == "occluded_candidate":
                # The compiler emits ``occluded_anchor`` with the actual
                # sampled depths before solving; the native record is only a
                # duplicate visibility-gate echo.
                continue
            if kind == "visibility_filtered_candidate":
                # Compile-time visibility already emitted a more specific,
                # grounded Python record for these gates.  Suppress the
                # solver's generic echo so public rationale has one decision
                # per eligibility fact.
                if gates.intersection({"occlusion", "viewport", "keepout"}):
                    continue
                record["candidate_index"] = candidate_index
                record["candidate_bounds"] = list(candidate.bounds or ())
                record["visibility_gates"] = _json_safe(visibility_gates)
                if visibility_gates:
                    record["visibility_reason"] = ",".join(
                        sorted(
                            f"{gate.get('kind')}:{gate.get('reason', 'ineligible')}"
                            for gate in visibility_gates
                        )
                    )
                elif (candidate.details or {}).get("geometry_authority"):
                    record["visibility_reason"] = "geometry_authority_visible_false"
                else:
                    record["visibility_reason"] = "compiled_visibility_gate"
            record["label_id"] = label.label_id
            record["source_id"] = label.source_id
            if kind != "visibility_filtered_candidate":
                record["solver_label_index"] = label_index
            record["candidate_id"] = candidate.candidate_id
            for key in ("displaced", "blocking"):
                if key not in record:
                    continue
                entries = []
                for entry in record[key]:
                    other_label_index = int(entry["label_id"])
                    other_candidate_index = int(entry.get("candidate_index", 0))
                    other = ordered[other_label_index]
                    other_candidate = candidate_lookup[
                        (other_label_index, other_candidate_index)
                    ]
                    entries.append(
                        {
                            "label_id": other.label_id,
                            "source_id": other.source_id,
                            "solver_label_index": other_label_index,
                            "candidate_id": other_candidate.candidate_id,
                            "overlap_area_px": float(entry.get("overlap_area_px", 0.0)),
                        }
                    )
                record[key] = entries
        elif kind == "solver":
            record["algorithm"] = "optimal"
        records.append(_json_safe(record))
    return records


def _resolve_label_placements_optimal(
    accepted: Sequence[AcceptedLabel],
    solver: Any,
    *,
    gap_tolerance: float,
    node_budget: int,
) -> tuple[list[AcceptedLabel], list[RejectedLabel], list[dict[str, Any]]]:
    """Bounded-optimal select-or-drop placement over the compiled candidates.

    Every stable candidate is forwarded with its true per-label index,
    authoritative non-degenerate bounds, priority, and visibility gate. The
    native choice is then reconstructed exactly on the compiled label.
    """
    ordered = sorted(
        accepted,
        key=lambda label: (label.ordering_key or label.label_id, label.label_id),
    )
    solver_input = []
    candidate_lookup: dict[tuple[int, int], LabelCandidate] = {}
    for index, label in enumerate(ordered):
        for candidate_index, candidate in enumerate(label.candidates):
            bounds = _rect_bounds(candidate.bounds)
            if bounds is None or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(
                    f"label candidate {candidate.candidate_id!r} lacks non-degenerate "
                    "screen-space bounds"
                )
            visible = (
                (candidate.details or {}).get("visible") is not False
                and (candidate.terrain_sample or {}).get("visible") is not False
            )
            candidate_lookup[(index, candidate_index)] = candidate
            solver_input.append(
                (
                    index,
                    candidate_index,
                    (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])),
                    float(candidate.score),
                    bool(visible),
                )
            )
    placements, _gap, native_rationale = solver(
        solver_input,
        gap_tolerance=float(gap_tolerance),
        node_budget=int(node_budget),
        margin=0.0,
    )
    placed_choices = {(int(index), int(candidate)) for index, candidate in placements}
    placed_by_label = {
        label_index: candidate_index for label_index, candidate_index in placed_choices
    }

    winners: list[AcceptedLabel] = []
    rejections: list[RejectedLabel] = []
    placed_labels: list[AcceptedLabel] = []
    placed_by_index: dict[int, AcceptedLabel] = {}
    for label_index, candidate_index in sorted(placed_choices):
        label = ordered[label_index]
        candidate = candidate_lookup[(label_index, candidate_index)]
        positioned = (candidate.details or {}).get("positioned_glyphs")
        placed = replace(
                label,
                candidate=candidate,
                screen_bounds=tuple(candidate.bounds or ()),
                positioned_glyphs=(positioned if positioned is not None else label.positioned_glyphs),
            )
        placed_labels.append(placed)
        placed_by_index[label_index] = placed
    for index, label in enumerate(ordered):
        if index in placed_by_label:
            winners.append(placed_by_index[index])
            continue
        conflicts = [
            (candidate, placed)
            for candidate in label.candidates
            if (candidate.details or {}).get("visible") is not False
            and (candidate.terrain_sample or {}).get("visible") is not False
            for placed in placed_labels
            if _rects_intersect(candidate.bounds, placed.screen_bounds)
        ]
        blockers = sorted(
            (placed for _candidate, placed in conflicts),
            key=lambda placed: (
                -float(placed.candidate.score),
                placed.ordering_key or placed.label_id,
                placed.label_id,
            ),
        )
        if not blockers:
            # Native may legitimately drop a non-positive candidate without a
            # geometric blocker. Preserve that exact solve outcome.
            rejections.append(
                RejectedLabel(
                    label_id=label.label_id,
                    source_id=label.source_id,
                    reason="priority_lost",
                    candidate_id=label.candidate.candidate_id,
                    diagnostic_refs=["label_rejection_summary"],
                    ordering_key=label.ordering_key,
                    details={"candidate_priority": float(label.candidate.score)},
                )
            )
            continue
        rejected_candidate = next(
            candidate for candidate, placed in conflicts if placed is blockers[0]
        )
        rejections.append(
            _collision_rejection(
                replace(
                    label,
                    candidate=rejected_candidate,
                    screen_bounds=tuple(rejected_candidate.bounds or ()),
                ),
                blockers[0],
            )
        )

    return winners, rejections, _translate_native_rationale(
        native_rationale, ordered, candidate_lookup
    )


def _resolve_label_placements(
    accepted: Sequence[AcceptedLabel],
    *,
    declutter: str,
    gap_tolerance: float,
    node_budget: int,
    diagnostics: list[Diagnostic],
) -> tuple[list[AcceptedLabel], list[RejectedLabel], list[dict[str, Any]]]:
    """Resolve final placements with the requested declutter engine.

    ``optimal`` uses the native bounded-optimal solver; when the native
    module is unavailable the greedy engine runs instead and the plan
    carries an explicit placeholder-fallback diagnostic — never a silent
    downgrade.
    """
    solver = _native_declutter_optimal() if declutter == "optimal" else None
    if declutter == "optimal" and solver is not None:
        return _resolve_label_placements_optimal(
            accepted,
            solver,
            gap_tolerance=gap_tolerance,
            node_budget=node_budget,
        )
    if declutter == "optimal":
        diagnostics.append(
            placeholder_fallback_diagnostic("optimal_declutter", layer_id="labels")
        )
    visible_fallback: list[AcceptedLabel] = []
    for label in accepted:
        candidate = next(
            (
                item
                for item in label.candidates
                if (item.details or {}).get("visible") is not False
                and (item.terrain_sample or {}).get("visible") is not False
            ),
            None,
        )
        if candidate is None:
            continue
        positioned = (candidate.details or {}).get("positioned_glyphs")
        visible_fallback.append(
            replace(
                label,
                candidate=candidate,
                screen_bounds=tuple(candidate.bounds or ()),
                positioned_glyphs=(
                    positioned
                    if positioned is not None
                    else label.positioned_glyphs
                ),
            )
        )
    winners, rejections = _resolve_label_collisions(visible_fallback)
    records: list[dict[str, Any]] = []
    for rejection in rejections:
        details = dict(rejection.details or {})
        records.append(
            {
                "kind": "dropped",
                "label_id": rejection.label_id,
                "source_id": rejection.source_id,
                "candidate_id": rejection.candidate_id,
                "priority_lost": rejection.reason == "priority_lost",
                "blocking": [
                    {
                        "label_id": details.get("collides_with"),
                        "source_id": details.get("collides_with_source_id"),
                    }
                ],
            }
        )
    records.append(
        {
            "kind": "solver",
            "algorithm": "greedy",
            "gap": None,
            "certified": False,
            "nodes_explored": len(visible_fallback),
            "gap_tolerance": float(gap_tolerance),
        }
    )
    return winners, rejections, records


def _conflict_text(entries: Sequence[Mapping[str, Any]] | None) -> str:
    parts = []
    for entry in entries or ():
        label = entry.get("label_id")
        area = entry.get("overlap_area_px")
        if area is None:
            parts.append(f"label {label!r}")
        else:
            parts.append(f"label {label!r} (overlap {float(area):.2f} px^2)")
    return ", ".join(parts)


def _render_rationale_record(record: Mapping[str, Any]) -> str:
    kind = str(record.get("kind", ""))
    if kind == "placed":
        line = (
            f"placed {record.get('label_id')!r} at candidate "
            f"{record.get('candidate_id')!r} (weight {float(record.get('weight', 0.0)):.3f})"
        )
        if record.get("displaced"):
            line += "; displaced " + _conflict_text(record.get("displaced"))
        return line
    if kind == "dropped":
        reason = "priority_lost" if record.get("priority_lost") else "collision"
        return (
            f"dropped {record.get('label_id')!r} ({reason}): blocked by "
            + _conflict_text(record.get("blocking"))
        )
    if kind in {"occluded_anchor", "occluded_candidate"}:
        sample = record.get("terrain_sample") or {}
        scene_depth = sample.get("scene_depth", sample.get("elevation"))
        label_depth = sample.get("label_depth")
        anchor = record.get("candidate_id") or record.get("label_id")
        if scene_depth is not None and label_depth is not None:
            return (
                f"occluded anchor {anchor!r}: terrain depth {float(scene_depth):.3f} "
                f"nearer than anchor depth {float(label_depth):.3f}"
            )
        if scene_depth is not None:
            return f"occluded anchor {anchor!r}: terrain elevation {float(scene_depth):.3f} occludes anchor"
        return f"occluded anchor {anchor!r}: silhouette/depth visibility gate"
    if kind == "visibility_filtered_candidate":
        return (
            f"filtered label {record.get('label_id')!r} "
            f"(source {record.get('source_id')!r}) candidate "
            f"{record.get('candidate_id')!r} at index "
            f"{int(record.get('candidate_index', 0))}: "
            f"{record.get('visibility_reason')}; bounds "
            f"{record.get('candidate_bounds')}"
        )
    if kind == "candidate_ineligible":
        candidate = record.get("candidate_id")
        gate = record.get("gate")
        if gate == "viewport":
            return (
                f"ineligible candidate {candidate!r}: bounds "
                f"{record.get('candidate_bounds')} outside viewport {record.get('viewport')}"
            )
        if gate == "keepout":
            return (
                f"ineligible candidate {candidate!r}: overlaps keepout "
                f"{record.get('keepout_region_id')!r} at {record.get('keepout_bounds')}"
            )
        return f"ineligible candidate {candidate!r}: {gate} visibility gate"
    if kind == "solver":
        gap = record.get("gap")
        gap_text = "n/a" if gap is None else f"{float(gap):.6f}"
        return (
            f"solver[{record.get('algorithm', 'optimal')}]: "
            f"{record.get('nodes_explored', 0)} nodes, "
            f"certified={bool(record.get('certified'))}, gap={gap_text}"
        )
    return f"record[{kind}]"


__all__ = [
    "AcceptedLabel",
    "KeepoutRegion",
    "LabelCandidate",
    "LabelPlan",
    "PAYLOAD_VERSION",
    "SUPPORTED_PAYLOAD_VERSIONS",
    "PriorityClass",
    "REJECTION_REASONS",
    "RejectedLabel",
]
