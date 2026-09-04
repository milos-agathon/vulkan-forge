"""Structured diagnostics for offline map-rendering validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence


SEVERITIES = ("info", "warning", "error", "fatal")
SUPPORT_LEVELS = (
    "supported",
    "underdeveloped",
    "missing",
    "Pro-gated",
    "placeholder/fallback",
    "experimental",
    "unsupported",
    "non-goal",
)
REQUIRED_DIAGNOSTIC_CODES = frozenset(
    {
        "crs_mismatch",
        "missing_glyphs",
        "unsupported_style_field",
        "unsupported_style_layer_type",
        "pro_gated_path",
        "placeholder_fallback",
        "experimental_feature",
        "vt_unsupported_family",
        "python_public_3dtiles_incomplete",
        "estimated_gpu_memory",
        "label_rejection_summary",
    }
)
P1_FEATURE_DIAGNOSTIC_CODES = frozenset(
    {
        "missing_label_field",
        "unicode_coverage_gap",
        "unsupported_tile_format",
        "unsupported_tile_feature",
        "missing_external_asset",
        "unavailable_terrain_sampler",
    }
)
P2_FEATURE_DIAGNOSTIC_CODES = frozenset(
    {
        "missing_texture_path",
        "missing_uvs",
        "unsupported_texture_format",
        "unavailable_cache_lod_stats",
        "unsupported_instancing_path",
    }
)

_STATUS_RANK = {"ok": 0, "info": 0, "warning": 1, "error": 2, "fatal": 3}
_DIAGNOSTIC_SORT_RANK = {"fatal": 0, "error": 1, "warning": 2, "info": 3}


class RenderFailurePolicy:
    """Render blocking policy for warning-level diagnostics."""

    CONTINUE_ON_WARNING = "continue_on_warning"
    FAIL_ON_WARNING = "fail_on_warning"

    _VALUES = (CONTINUE_ON_WARNING, FAIL_ON_WARNING)

    @classmethod
    def validate(cls, policy: str) -> str:
        if policy not in cls._VALUES:
            raise ValueError(f"Unknown render failure policy: {policy!r}")
        return policy


class SeverityPolicy:
    """Severity validation and report status behavior."""

    @staticmethod
    def validate(severity: str) -> str:
        if severity not in SEVERITIES:
            raise ValueError(f"Unknown diagnostic severity: {severity!r}")
        return severity

    @staticmethod
    def status_for(severities: Sequence[str]) -> str:
        status = "ok"
        for severity in severities:
            SeverityPolicy.validate(severity)
            if _STATUS_RANK[severity] > _STATUS_RANK[status]:
                status = severity
        return status

    @staticmethod
    def render_blocked(status: str, policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING) -> bool:
        RenderFailurePolicy.validate(policy)
        if status not in ("ok", "warning", "error", "fatal"):
            raise ValueError(f"Unknown validation status: {status!r}")
        if status in ("error", "fatal"):
            return True
        return status == "warning" and policy == RenderFailurePolicy.FAIL_ON_WARNING


def _validate_support_level(support_level: str | None) -> str | None:
    if support_level is not None and support_level not in SUPPORT_LEVELS:
        raise ValueError(f"Unknown support level: {support_level!r}")
    return support_level


def _normalize_support_summary(summary: Mapping[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in sorted(dict(summary or {}).items()):
        normalized[str(key)] = _validate_support_level(str(value)) or str(value)
    return normalized


def _json_safe(value: Any, *, context: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value.keys(), key=str):
            if not isinstance(key, str):
                raise TypeError(f"{context} must use string mapping keys")
            normalized[key] = _json_safe(value[key], context=context)
        return normalized
    if isinstance(value, tuple):
        return [_json_safe(item, context=context) for item in value]
    if isinstance(value, list):
        return [_json_safe(item, context=context) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{context} must be JSON-serializable")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    remediation: str
    support_level: str | None = None
    layer_id: str | None = None
    object_id: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.severity = SeverityPolicy.validate(str(self.severity))
        self.support_level = _validate_support_level(self.support_level)
        self.details = _json_safe(dict(self.details or {}), context="details")
        _stable_json(self.details)

    def sort_key(self) -> tuple[int, str, str, str, str, str]:
        return (
            _DIAGNOSTIC_SORT_RANK[self.severity],
            self.code,
            self.layer_id or "",
            self.object_id or "",
            self.message,
            _stable_json(self.details),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "remediation": self.remediation,
            "support_level": self.support_level,
            "layer_id": self.layer_id,
            "object_id": self.object_id,
            "details": _json_safe(dict(self.details or {}), context="details"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Diagnostic":
        return cls(
            code=str(data["code"]),
            severity=str(data["severity"]),
            message=str(data["message"]),
            remediation=str(data["remediation"]),
            support_level=data.get("support_level"),
            layer_id=data.get("layer_id"),
            object_id=data.get("object_id"),
            details=data.get("details") or {},
        )


def crs_mismatch_diagnostic(
    scene_crs: str,
    layer_crs: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="crs_mismatch",
        severity="error",
        message="Layer CRS differs from the scene or terrain CRS and no transform was provided.",
        remediation="Use matching CRS metadata or provide an explicit transform before rendering.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"layer_crs": layer_crs, "scene_crs": scene_crs},
    )


def missing_glyphs_diagnostic(
    missing_glyphs: Sequence[str],
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    glyphs = sorted(str(glyph) for glyph in missing_glyphs)
    return Diagnostic(
        code="missing_glyphs",
        severity="warning",
        message=f"{len(glyphs)} glyphs are missing from the active atlas.",
        remediation="Load an atlas with the missing glyphs or change label text before rendering.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
        details={"count": len(glyphs), "missing_glyphs": glyphs},
    )


def unsupported_style_field_diagnostic(
    layer_id: str,
    fields: Sequence[str],
    *,
    section: str | None = None,
) -> Diagnostic:
    unsupported_fields = sorted(str(field) for field in fields)
    details: dict[str, Any] = {"fields": unsupported_fields}
    if section:
        details["section"] = section
    return Diagnostic(
        code="unsupported_style_field",
        severity="warning",
        message="Style layer uses paint or layout fields that forge3d does not support.",
        remediation="Remove unsupported fields or use the documented local feature styling subset.",
        support_level="unsupported",
        layer_id=layer_id,
        details=details,
    )


def unsupported_style_layer_type_diagnostic(layer_id: str, layer_type: str) -> Diagnostic:
    return Diagnostic(
        code="unsupported_style_layer_type",
        severity="error",
        message="Style layer type is not supported by forge3d offline feature styling.",
        remediation="Use a supported local feature layer type such as fill, line, or circle.",
        support_level="unsupported",
        layer_id=layer_id,
        details={"layer_type": layer_type},
    )


def pro_gated_path_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="pro_gated_path",
        severity="error",
        message="Requested workflow requires a Pro-gated native path.",
        remediation="Enable the required Pro/native capability or choose a supported public path.",
        support_level="Pro-gated",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": feature},
    )


def placeholder_fallback_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="placeholder_fallback",
        severity="error",
        message="Requested workflow would use placeholder or non-renderable fallback output.",
        remediation="Use a renderable supported path or keep the workflow blocked before render.",
        support_level="placeholder/fallback",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": feature},
    )


def experimental_feature_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="experimental_feature",
        severity="warning",
        message="Requested workflow uses a feature that is not production-stable.",
        remediation="Treat this path as experimental or use a documented supported alternative.",
        support_level="experimental",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": feature},
    )


def vt_unsupported_family_diagnostic(
    family: str,
    *,
    supported_family: str = "albedo, mask, normal",
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="vt_unsupported_family",
        severity="error",
        message="Requested terrain virtual-texturing family is not paged by the native runtime.",
        remediation="Use one of the native terrain VT families: albedo, normal, or mask.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"family": family, "supported_family": supported_family},
    )


def python_public_3dtiles_incomplete_diagnostic(
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="python_public_3dtiles_incomplete",
        severity="error",
        message="Public Python 3D Tiles workflow cannot complete the requested render path.",
        remediation="Use supported local fixtures only for validation, or wait for public MapScene integration.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
    )


def estimated_gpu_memory_diagnostic(
    estimated_bytes: int,
    budget_bytes: int | None,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    exceeds_budget = budget_bytes is not None and estimated_bytes > budget_bytes
    return Diagnostic(
        code="estimated_gpu_memory",
        severity="warning" if exceeds_budget else "info",
        message=(
            "Estimated GPU memory use exceeds the configured budget."
            if exceeds_budget
            else "Estimated GPU memory use is available for review."
        ),
        remediation=(
            "Reduce layer resolution, simplify inputs, or increase the configured GPU memory budget."
            if exceeds_budget
            else "No action is required unless other diagnostics indicate risk."
        ),
        support_level="supported",
        layer_id=layer_id,
        object_id=object_id,
        details={
            "budget_bytes": budget_bytes,
            "estimated_bytes": int(estimated_bytes),
        },
    )


def memory_budget_validation_report(
    metrics: Mapping[str, Any] | None = None,
) -> "ValidationReport":
    """Build a diagnostics report from memory-budget telemetry."""
    if metrics is None:
        from . import mem

        metrics = mem.memory_metrics()

    snapshot = dict(metrics)
    host_visible_bytes = int(snapshot.get("host_visible_bytes", 0))
    budget_bytes_raw = snapshot.get("limit_bytes")
    budget_bytes = int(budget_bytes_raw) if budget_bytes_raw is not None else None
    budget_policy = str(snapshot.get("budget_policy", "enforce"))
    within_budget = bool(snapshot.get("within_budget", True))
    diagnostic = estimated_gpu_memory_diagnostic(host_visible_bytes, budget_bytes)
    details = dict(diagnostic.details or {})
    details.update(
        {
            "budget_policy": budget_policy,
            "buffer_bytes": int(snapshot.get("buffer_bytes", 0)),
            "texture_bytes": int(snapshot.get("texture_bytes", 0)),
            "within_budget": within_budget,
        }
    )
    return ValidationReport(
        diagnostics=(
            Diagnostic(
                code=diagnostic.code,
                severity="warning" if not within_budget else diagnostic.severity,
                message=diagnostic.message,
                remediation=diagnostic.remediation,
                support_level=diagnostic.support_level,
                details=details,
            ),
        ),
        estimated_gpu_memory_bytes=host_visible_bytes,
    )


def memory_tracking_completeness_report(
    expected_bytes: int,
    metrics: Mapping[str, Any] | None = None,
    *,
    min_coverage: float = 0.95,
) -> "ValidationReport":
    """Report whether tracked memory accounts for an expected allocation envelope."""
    if metrics is None:
        from . import mem

        metrics = mem.memory_metrics()
    expected = max(0, int(expected_bytes))
    tracked = int(dict(metrics).get("host_visible_bytes", 0))
    coverage = 1.0 if expected == 0 else tracked / float(expected)
    ok = coverage >= float(min_coverage)
    diagnostic = Diagnostic(
        code="memory_tracking_completeness",
        severity="info" if ok else "warning",
        message=(
            "Tracked memory coverage meets the expected allocation envelope."
            if ok
            else "Tracked memory coverage is below the expected allocation envelope."
        ),
        remediation=(
            "No action is required."
            if ok
            else "Route the missing allocation path through tracked constructors or update the estimate."
        ),
        support_level="supported" if ok else "underdeveloped",
        details={
            "expected_bytes": expected,
            "tracked_bytes": tracked,
            "coverage_ratio": coverage,
            "min_coverage": float(min_coverage),
        },
    )
    return ValidationReport(
        diagnostics=(diagnostic,),
        estimated_gpu_memory_bytes=expected,
        supported_features={"memory.tracking_completeness": "supported" if ok else "underdeveloped"},
    )


def capabilities() -> dict[str, Any]:
    """Report the negotiated GPU capability set for the active device.

    Returns a dict with ``requested`` (features forge3d asks for),
    ``granted`` (the subset the adapter actually provides), and ``limits``
    (a subset of the device limits). Requesting a capability never hard-fails
    the device: anything absent is recorded as a degradation instead.

    Raises ``RuntimeError`` when the native extension is unavailable or no GPU
    adapter can be acquired.
    """
    from ._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "capabilities"):
        raise RuntimeError(
            "forge3d native extension is unavailable; capabilities() requires the "
            "compiled _forge3d module (build with `maturin develop`)."
        )
    return native.capabilities()


def culling_stats() -> dict[str, Any]:
    """Return GPU-read two-phase terrain culling counters for the last frame."""
    from ._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "terrain_culling_stats"):
        raise RuntimeError(
            "forge3d native extension is unavailable; culling_stats() requires "
            "the compiled _forge3d module (build with `maturin develop`)."
        )
    return dict(native.terrain_culling_stats())

def visibility_stats() -> dict[str, Any]:
    """Return one-record-per-visible-pixel counters for the last frame."""
    from ._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "terrain_visibility_stats"):
        raise RuntimeError(
            "forge3d native extension is unavailable; visibility_stats() "
            "requires the compiled _forge3d module."
        )
    return dict(native.terrain_visibility_stats())


def vt_stats() -> dict[str, Any]:
    """Return live terrain virtual-texture residency and footprint counters."""
    from ._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "terrain_vt_stats"):
        raise RuntimeError(
            "forge3d native extension is unavailable; vt_stats() requires "
            "the compiled _forge3d module."
        )
    return dict(native.terrain_vt_stats())


def seam_stats() -> dict[str, Any]:
    """Return live clipmap seam-analysis counters for the last geometry build."""
    from ._native import get_native_module

    native = get_native_module()
    if native is None or not hasattr(native, "terrain_seam_stats"):
        raise RuntimeError(
            "forge3d native extension is unavailable; seam_stats() requires "
            "the compiled _forge3d module."
        )
    return dict(native.terrain_seam_stats())


def render_certificate(sign: bool = True) -> dict[str, Any]:
    """Assemble a RenderCertificate for the LAST completed native render.

    Loads the native, unsigned execution report
    (``forge3d._forge3d.render_execution_report``), then merges the
    Python-side degradation sink (:mod:`forge3d._degradation`) into the
    ``degradations`` list — native entries win on ``(kind, name)`` collisions —
    and re-sorts the merged list by ``(kind, name)``. When ``sign`` is true the
    certificate is sealed with :func:`forge3d.certificate.sign_certificate`.

    Raises ``RuntimeError`` when the native extension is unavailable, and
    propagates the native error when no render has completed in this process.
    """
    from ._native import get_native_module
    from . import _degradation
    from . import certificate as _certificate

    native = get_native_module()
    if native is None or not hasattr(native, "render_execution_report"):
        raise RuntimeError(
            "forge3d native extension is unavailable; render_certificate() requires "
            "the compiled _forge3d module (build with `maturin develop`) and a "
            "completed native render."
        )

    cert: dict[str, Any] = json.loads(native.render_execution_report())

    degradations = [dict(entry) for entry in (cert.get("degradations") or [])]
    present = {(entry.get("kind"), entry.get("name")) for entry in degradations}
    for entry in _degradation.capture_snapshot():
        key = (entry.get("kind"), entry.get("name"))
        if key not in present:
            degradations.append(dict(entry))
            present.add(key)
    degradations.sort(key=lambda entry: (str(entry.get("kind", "")), str(entry.get("name", ""))))
    cert["degradations"] = degradations

    if sign:
        cert = _certificate.sign_certificate(cert)
    return cert


def label_rejection_summary_diagnostic(
    rejection_counts: Mapping[str, int],
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    counts = {str(key): int(value) for key, value in sorted(dict(rejection_counts).items())}
    total = sum(counts.values())
    return Diagnostic(
        code="label_rejection_summary",
        severity="warning",
        message=f"{total} label candidates were rejected during placement.",
        remediation="Inspect rejection reasons and adjust priorities, keepouts, glyph coverage, or geometry.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
        details={"rejection_counts": counts, "total": total},
    )


def missing_label_field_diagnostic(
    field: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="missing_label_field",
        severity="error",
        message="Label text expression references a missing feature field.",
        remediation="Provide the referenced property or change the label text expression.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"field": str(field)},
    )


def unicode_coverage_gap_diagnostic(
    missing_glyphs: Sequence[str],
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    glyphs = sorted(str(glyph) for glyph in missing_glyphs)
    return Diagnostic(
        code="unicode_coverage_gap",
        severity="warning",
        message="Label text contains Unicode code points outside configured atlas coverage.",
        remediation="Load an atlas or fallback range covering the missing code points before rendering.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
        details={"count": len(glyphs), "missing_glyphs": glyphs},
    )


def unsupported_tile_format_diagnostic(
    tile_format: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
    supported_formats: Sequence[str] = ("tileset.json", "b3dm"),
) -> Diagnostic:
    return Diagnostic(
        code="unsupported_tile_format",
        severity="error",
        message="3D Tiles source uses a format that is not supported by the public MapScene workflow.",
        remediation="Use a supported local tileset JSON or B3DM fixture, or keep the layer diagnostic-only.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={
            "format": str(tile_format),
            "supported_formats": sorted(str(value) for value in supported_formats),
        },
    )


def unsupported_tile_feature_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="unsupported_tile_feature",
        severity="error",
        message="3D Tiles content requires a feature not supported by the public MapScene workflow.",
        remediation="Remove the unsupported tile feature or use a documented supported local fixture.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": str(feature)},
    )


def missing_external_asset_diagnostic(
    layer_type: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
    path: str,
) -> Diagnostic:
    return Diagnostic(
        code="missing_external_asset",
        severity="error",
        message="Scene or bundle references an external asset that cannot be found.",
        remediation="Provide the referenced asset or update the scene/bundle to point at an available file.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"layer_type": str(layer_type), "path": str(path)},
    )


def missing_texture_path_diagnostic(
    path: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
    material_id: str | None = None,
) -> Diagnostic:
    details: dict[str, Any] = {"path": str(path)}
    if material_id is not None:
        details["material_id"] = str(material_id)
    return Diagnostic(
        code="missing_texture_path",
        severity="error",
        message="Building material references a texture path that is missing or unreadable.",
        remediation="Provide the referenced texture asset or remove the textured-material intent before rendering.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details=details,
    )


def missing_uvs_diagnostic(
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
    material_id: str | None = None,
) -> Diagnostic:
    details: dict[str, Any] = {}
    if material_id is not None:
        details["material_id"] = str(material_id)
    return Diagnostic(
        code="missing_uvs",
        severity="error",
        message="Building material requests a texture but the affected geometry has no usable UVs.",
        remediation="Provide UV coordinates for the textured geometry or use an explicit scalar-material fallback.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details=details,
    )


def unsupported_texture_format_diagnostic(
    texture_format: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
    path: str | None = None,
    supported_formats: Sequence[str] = ("jpg", "jpeg", "png", "tif", "tiff"),
) -> Diagnostic:
    details: dict[str, Any] = {
        "format": str(texture_format).lower().lstrip("."),
        "supported_formats": sorted(str(value).lower().lstrip(".") for value in supported_formats),
    }
    if path is not None:
        details["path"] = str(path)
    return Diagnostic(
        code="unsupported_texture_format",
        severity="error",
        message="Building material texture uses a format that is not supported by the current MapScene workflow.",
        remediation="Use a documented supported texture format or keep the textured material diagnostic-only.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details=details,
    )


def unavailable_cache_lod_stats_diagnostic(
    layer_type: str,
    unavailable_stats: Sequence[str],
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    stats = sorted(str(stat) for stat in unavailable_stats)
    return Diagnostic(
        code="unavailable_cache_lod_stats",
        severity="warning",
        message="Requested cache or LOD statistics are not available for this layer.",
        remediation="Use available layer metadata only, or add a renderer/stat source before relying on these stats.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
        details={"layer_type": str(layer_type), "unavailable_stats": stats},
    )


def unsupported_instancing_path_diagnostic(
    path: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="unsupported_instancing_path",
        severity="error",
        message="Requested MapScene instancing workflow is not supported in this configuration.",
        remediation="Use a supported non-instanced workflow or enable a documented instancing path with tests.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"path": str(path)},
    )


def unavailable_terrain_sampler_diagnostic(
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="unavailable_terrain_sampler",
        severity="warning",
        message="Terrain-height sampling was requested but no terrain sampler is available.",
        remediation="Provide a terrain sampler or choose a label terrain policy that does not require sampling.",
        support_level="underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
    )


def validate_label_support(
    labels: Sequence[Mapping[str, Any]],
    *,
    atlas_glyphs: set[str] | frozenset[str] | None = None,
    layer_id: str | None = None,
) -> "ValidationReport":
    """Validate label support boundaries without requiring raw viewer IPC.

    This helper does not compile or render labels. It reports PRD-scoped
    diagnostics for experimental line/curved label paths and known missing
    glyphs so callers can include those findings in validation reports before
    render.
    """
    diagnostics: list[Diagnostic] = []
    glyphs = set(atlas_glyphs) if atlas_glyphs is not None else None

    for index, label in enumerate(labels):
        object_id = str(label.get("id", f"label_{index}"))
        kind = str(label.get("kind", label.get("placement_kind", "point")))
        text = str(label.get("text", ""))

        if kind in {"line", "curved"}:
            diagnostics.append(
                experimental_feature_diagnostic(
                    f"{kind} labels",
                    layer_id=layer_id,
                    object_id=object_id,
                )
            )

        if glyphs is not None:
            missing = sorted({char for char in text if char not in glyphs})
            if missing:
                diagnostics.append(
                    missing_glyphs_diagnostic(
                        missing,
                        layer_id=layer_id,
                        object_id=object_id,
                    )
                )

    return ValidationReport(
        diagnostics=diagnostics,
        supported_features={"labels.point": "underdeveloped"},
        unsupported_features={
            "labels.curved.production": "experimental",
            "labels.line.production": "experimental",
        },
    )


@dataclass
class LayerSummary:
    layer_id: str
    layer_type: str
    support_level: str
    diagnostic_codes: Sequence[str] = field(default_factory=tuple)
    object_count: int | None = None
    bounds: Sequence[float] | None = None
    memory_estimate_bytes: int | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.support_level = _validate_support_level(self.support_level) or self.support_level
        self.diagnostic_codes = tuple(sorted(str(code) for code in self.diagnostic_codes))
        self.bounds = tuple(float(value) for value in self.bounds) if self.bounds is not None else None
        self.details = _json_safe(dict(self.details or {}), context="details")

    def sort_key(self) -> tuple[str, str, str]:
        return (self.layer_id, self.layer_type, self.support_level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "support_level": self.support_level,
            "diagnostic_codes": list(self.diagnostic_codes),
            "object_count": self.object_count,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "memory_estimate_bytes": self.memory_estimate_bytes,
            "details": _json_safe(dict(self.details or {}), context="details"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerSummary":
        return cls(
            layer_id=str(data["layer_id"]),
            layer_type=str(data["layer_type"]),
            support_level=str(data["support_level"]),
            diagnostic_codes=data.get("diagnostic_codes") or (),
            object_count=data.get("object_count"),
            bounds=data.get("bounds"),
            memory_estimate_bytes=data.get("memory_estimate_bytes"),
            details=data.get("details") or {},
        )


@dataclass
class SupportMatrixEntry:
    area: str
    capability: str
    support_level: str
    scope: str
    limitations: Sequence[str] = field(default_factory=tuple)
    diagnostic_codes: Sequence[str] = field(default_factory=tuple)
    remediation: str = ""
    evidence: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.support_level = _validate_support_level(self.support_level) or self.support_level
        self.limitations = tuple(sorted(str(item) for item in self.limitations))
        self.diagnostic_codes = tuple(sorted(str(code) for code in self.diagnostic_codes))
        self.evidence = tuple(sorted(str(item) for item in self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "capability": self.capability,
            "support_level": self.support_level,
            "scope": self.scope,
            "limitations": list(self.limitations),
            "diagnostic_codes": list(self.diagnostic_codes),
            "remediation": self.remediation,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SupportMatrixEntry":
        return cls(
            area=str(data["area"]),
            capability=str(data["capability"]),
            support_level=str(data["support_level"]),
            scope=str(data["scope"]),
            limitations=data.get("limitations") or (),
            diagnostic_codes=data.get("diagnostic_codes") or (),
            remediation=str(data.get("remediation") or ""),
            evidence=data.get("evidence") or (),
        )


@dataclass
class ValidationReport:
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] = field(default_factory=tuple)
    layer_summaries: Sequence[LayerSummary | Mapping[str, Any]] = field(default_factory=tuple)
    estimated_gpu_memory_bytes: int | None = None
    supported_features: Mapping[str, str] | None = None
    unsupported_features: Mapping[str, str] | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        self.diagnostics = tuple(
            sorted(
                (
                    diag if isinstance(diag, Diagnostic) else Diagnostic.from_dict(diag)
                    for diag in self.diagnostics
                ),
                key=lambda diag: diag.sort_key(),
            )
        )
        self.layer_summaries = tuple(
            sorted(
                (
                    summary if isinstance(summary, LayerSummary) else LayerSummary.from_dict(summary)
                    for summary in self.layer_summaries
                ),
                key=lambda summary: summary.sort_key(),
            )
        )
        derived_status = SeverityPolicy.status_for([diag.severity for diag in self.diagnostics])
        if self.status is not None:
            if self.status not in ("ok", "warning", "error", "fatal"):
                raise ValueError(f"Unknown validation status: {self.status!r}")
            if _STATUS_RANK[self.status] > _STATUS_RANK[derived_status]:
                derived_status = self.status
        self.status = derived_status
        self.supported_features = _normalize_support_summary(self.supported_features)
        self.unsupported_features = _normalize_support_summary(self.unsupported_features)

    @property
    def has_errors(self) -> bool:
        return self.status in ("error", "fatal")

    def render_blocked(self, policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING) -> bool:
        return SeverityPolicy.render_blocked(self.status or "ok", policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "diagnostics": [diag.to_dict() for diag in self.diagnostics],
            "layer_summaries": [summary.to_dict() for summary in self.layer_summaries],
            "estimated_gpu_memory_bytes": self.estimated_gpu_memory_bytes,
            "supported_features": dict(self.supported_features or {}),
            "unsupported_features": dict(self.unsupported_features or {}),
            "render_blocked": self.render_blocked(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        return cls(
            diagnostics=data.get("diagnostics") or (),
            layer_summaries=data.get("layer_summaries") or (),
            estimated_gpu_memory_bytes=data.get("estimated_gpu_memory_bytes"),
            supported_features=data.get("supported_features") or {},
            unsupported_features=data.get("unsupported_features") or {},
            status=data.get("status"),
        )


__all__ = [
    "Diagnostic",
    "LayerSummary",
    "P1_FEATURE_DIAGNOSTIC_CODES",
    "P2_FEATURE_DIAGNOSTIC_CODES",
    "REQUIRED_DIAGNOSTIC_CODES",
    "RenderFailurePolicy",
    "SeverityPolicy",
    "SupportMatrixEntry",
    "ValidationReport",
    "SEVERITIES",
    "SUPPORT_LEVELS",
    "capabilities",
    "culling_stats",
    "vt_stats",
    "visibility_stats",
    "seam_stats",
    "crs_mismatch_diagnostic",
    "estimated_gpu_memory_diagnostic",
    "memory_budget_validation_report",
    "memory_tracking_completeness_report",
    "experimental_feature_diagnostic",
    "label_rejection_summary_diagnostic",
    "missing_external_asset_diagnostic",
    "missing_glyphs_diagnostic",
    "missing_label_field_diagnostic",
    "missing_texture_path_diagnostic",
    "missing_uvs_diagnostic",
    "placeholder_fallback_diagnostic",
    "pro_gated_path_diagnostic",
    "python_public_3dtiles_incomplete_diagnostic",
    "render_certificate",
    "unicode_coverage_gap_diagnostic",
    "unavailable_cache_lod_stats_diagnostic",
    "unavailable_terrain_sampler_diagnostic",
    "unsupported_instancing_path_diagnostic",
    "unsupported_style_field_diagnostic",
    "unsupported_style_layer_type_diagnostic",
    "unsupported_texture_format_diagnostic",
    "unsupported_tile_feature_diagnostic",
    "unsupported_tile_format_diagnostic",
    "validate_label_support",
    "vt_unsupported_family_diagnostic",
]
