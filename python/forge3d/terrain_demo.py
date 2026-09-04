from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import math
import os

import numpy as np

import forge3d as f3d
from . import io as _io
from .config import load_renderer_config
from .terrain_params import (
    ShadowSettings as TerrainShadowSettings,
    FogSettings as TerrainFogSettings,
    ReflectionSettings as TerrainReflectionSettings,
    TriplanarSettings,
    load_height_curve_lut,
    make_terrain_params_config,
)
from .colormaps.core import (
    interpolate_hex_colors as _cm_interpolate_hex_colors,
    elevation_stops_from_hex_colors as _cm_elevation_stops,
)
from .lighting import sun_direction_from_angles as _sun_direction_from_angles


DEFAULT_DEM = Path("assets/tif/Gore_Range_Albers_1m.tif")
DEFAULT_HDR = Path("assets/hdri/snow_field_4k.hdr")
DEFAULT_OUTPUT = Path("examples/out/terrain_demo.png")
DEFAULT_SIZE = (1920, 1080)
DEFAULT_DOMAIN = (200.0, 2200.0)
DEFAULT_CAM_RADIUS = 1000.0
DEFAULT_CAM_PHI = 135.0
DEFAULT_CAM_THETA = 45.0
DEFAULT_CAM_FOV = 55.0
DEFAULT_CAMERA_MODE = "screen"
DEFAULT_COLORMAP_STOPS: Sequence[tuple[float, str]] = (
    (200.0, "#00aa00"),   # Low elevation: Vibrant green (valleys)
    (800.0, "#80ff00"),   # Mid-low: Bright lime (foothills)
    (1200.0, "#ffff00"),  # Mid: Pure yellow (slopes)
    (1600.0, "#ff8000"),  # Mid-high: Vivid orange (rocky terrain)
    (2000.0, "#ff0000"),  # High: Pure red (peaks)
    (2200.0, "#800000"),  # Highest: Dark red (summits)
)

QUANTILE_DEFAULT_LO = 0.0
QUANTILE_DEFAULT_HI = 1.0


def _require_attributes(attr_names: Iterable[str]) -> None:
    missing = [name for name in attr_names if not hasattr(f3d, name)]
    if missing:
        raise SystemExit(
            "Required forge3d attributes are missing in this build: " + ", ".join(missing)
        )


def _normalize_preset_name(name: str) -> str:
    return "".join(c for c in str(name).strip().lower() if c not in {"-", "_", " ", "."})


def _arg_was_explicit(args: Any, name: str) -> bool:
    explicit = getattr(args, "_explicit_cli_args", None)
    return bool(explicit is not None and str(name) in explicit)


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1.0e-9
    except (TypeError, ValueError):
        return left == right


def _set_arg_default(args: Any, name: str, value: Any, default: Any = None) -> None:
    if value is None or _arg_was_explicit(args, name):
        return
    current = getattr(args, name, default)
    if current is None or _values_equal(current, default):
        setattr(args, name, value)


def _preset_payload(name: str) -> Mapping[str, Any]:
    try:
        from . import presets as _presets

        return dict(_presets.get(str(name)))
    except Exception:
        return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _builtin_ibl_path(name: Any, *, prefer_local_assets: bool = True) -> Path | None:
    """Resolve a named demo IBL.

    ``prefer_local_assets=False`` selects the packaged deterministic fallback
    even when optional, untracked high-resolution demo assets exist in a
    checkout. Signed MapScene recipes use that hermetic mode.
    """
    key = _normalize_preset_name(str(name or ""))
    if key not in {"clearsky", "clear"}:
        return None
    if prefer_local_assets:
        root = Path(__file__).resolve().parents[2]
        for relative in (
            Path("assets") / "hdri" / "venice_sunrise_4k.hdr",
            Path("assets") / "hdri" / "brown_photostudio_02_4k.hdr",
        ):
            candidate = root / relative
            if candidate.exists():
                return candidate

    # The named source is a real built-in, not an optional local asset. Keep a
    # tiny deterministic Radiance environment available in clean installations.
    import tempfile

    generated = Path(tempfile.gettempdir()) / "forge3d-clear-sky-v1.hdr"
    expected = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n\n"
        b"-Y 2 +X 2\n"
        + bytes([180, 190, 205, 128]) * 4
    )
    if not generated.exists() or generated.read_bytes() != expected:
        generated.write_bytes(expected)
    return generated


def check_camera_sun_alignment(
    cam_phi_deg: float,
    cam_theta_deg: float,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> float:
    """Check alignment between camera view direction and sun direction.
    
    Returns the dot product of view and light directions.
    Values > 0.7 indicate the sun is nearly behind the camera, causing flat lighting.
    Values < 0.3 indicate good cross-lighting for dramatic shadows.
    
    Args:
        cam_phi_deg: Camera azimuth angle in degrees
        cam_theta_deg: Camera elevation angle in degrees  
        sun_azimuth_deg: Sun azimuth angle in degrees
        sun_elevation_deg: Sun elevation angle in degrees
        
    Returns:
        Dot product in range [-1, 1]. Higher = more aligned = flatter lighting.
    """
    # Convert to radians
    cam_phi = math.radians(cam_phi_deg)
    cam_theta = math.radians(cam_theta_deg)
    sun_az = math.radians(sun_azimuth_deg)
    sun_el = math.radians(sun_elevation_deg)
    
    # Camera view direction (from camera toward scene center)
    view_x = math.cos(cam_theta) * math.cos(cam_phi)
    view_y = math.sin(cam_theta)
    view_z = math.cos(cam_theta) * math.sin(cam_phi)
    
    # Sun light direction (direction light comes FROM)
    light_x = math.cos(sun_el) * math.sin(sun_az)
    light_y = math.sin(sun_el)
    light_z = math.cos(sun_el) * math.cos(sun_az)
    
    # Dot product
    dot = view_x * light_x + view_y * light_y + view_z * light_z
    return dot


def _load_dem(path: Path):
    if path.suffix.lower() == ".npy":
        dem = _io.load_dem_from_array(np.load(path))
    else:
        dem = _io.load_dem(str(path), fill_nodata_values=True)
    data = getattr(dem, "data", None)
    if data is None:
        raise SystemExit("DEM object does not expose a .data array")
    # Match legacy terrain_demo orientation
    dem.data = np.flipud(np.asarray(data, dtype=np.float32)).copy()
    return dem


def _dem_spacing_info(dem: Any) -> tuple[float, float, float, str]:
    """Return (dx_m, dy_m, terrain_span, note) for logging and scaling."""
    res = getattr(dem, "resolution", (1.0, 1.0)) or (1.0, 1.0)
    try:
        dx_raw = float(res[0] or 1.0)
        dy_raw = float(res[1] or 1.0)
    except Exception:
        dx_raw, dy_raw = 1.0, 1.0

    crs = getattr(dem, "crs", None)
    bounds = getattr(dem, "bounds", None)
    is_geographic = False
    if crs:
        crs_str = str(crs).lower()
        if "4326" in crs_str or "wgs84" in crs_str:
            is_geographic = True

    dx_m, dy_m = dx_raw, dy_raw
    spacing_note = "projected/unspecified"
    if is_geographic:
        lat_deg = 0.0
        if bounds and len(bounds) == 4:
            try:
                lat_deg = 0.5 * (float(bounds[1]) + float(bounds[3]))
            except Exception:
                lat_deg = 0.0
        lat_rad = math.radians(lat_deg)
        # Approx meters per degree using a common ellipsoid approximation
        meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        meters_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
        dx_m = dx_raw * meters_per_deg_lon
        dy_m = dy_raw * meters_per_deg_lat
        spacing_note = f"geographic(lat={lat_deg:.3f})"

    dx_m = max(dx_m, 1e-6)
    dy_m = max(dy_m, 1e-6)
    h, w = np.asarray(dem.data).shape[:2]
    span_x = dx_m * float(w)
    span_y = dy_m * float(h)
    terrain_span = float(max(span_x, span_y))
    return dx_m, dy_m, terrain_span, spacing_note


def _build_renderer_config(args: Any):
    overrides: dict[str, Any] = {}
    if getattr(args, "light", None):
        overrides["lights"] = [_parse_light_spec(spec) for spec in args.light]
    overrides["exposure"] = float(args.exposure)
    if getattr(args, "brdf", None):
        overrides["brdf"] = args.brdf
    # P0.2/M3: Handle shadow_technique (new) or shadows (legacy alias)
    shadow_tech = getattr(args, "shadow_technique", None) or getattr(args, "shadows", None)
    if shadow_tech:
        overrides["shadows"] = shadow_tech
    if getattr(args, "shadow_map_res", None) is not None:
        overrides["shadow_map_res"] = int(args.shadow_map_res)
    if getattr(args, "cascades", None) is not None:
        overrides["cascades"] = int(args.cascades)
    if getattr(args, "pcss_blocker_radius", None) is not None:
        overrides["pcss_blocker_radius"] = float(args.pcss_blocker_radius)
    if getattr(args, "pcss_filter_radius", None) is not None:
        overrides["pcss_filter_radius"] = float(args.pcss_filter_radius)
    if getattr(args, "shadow_light_size", None) is not None:
        overrides["light_size"] = float(args.shadow_light_size)
    if getattr(args, "shadow_moment_bias", None) is not None:
        overrides["moment_bias"] = float(args.shadow_moment_bias)
    if getattr(args, "gi", None):
        modes = [item.strip() for item in str(args.gi).split(",") if item.strip()]
        overrides["gi"] = modes
    if getattr(args, "sky", None):
        overrides["sky"] = args.sky
    if getattr(args, "hdr", None):
        overrides["hdr"] = args.hdr
    if getattr(args, "volumetric", None):
        overrides["volumetric"] = _parse_volumetric_spec(str(args.volumetric))
    # P0.1/M1: Wire OIT mode to renderer config
    oit_mode = getattr(args, "oit", None)
    if oit_mode is not None:
        overrides["oit_mode"] = oit_mode
    # P1.4: Wire TAA settings to renderer config
    if getattr(args, "taa", False):
        overrides["taa_enabled"] = True
    taa_weight = getattr(args, "taa_history_weight", None)
    if taa_weight is not None:
        overrides["taa_history_weight"] = float(taa_weight)
    if getattr(args, "preset", None):
        try:
            from . import presets as _presets

            preset_map = dict(_presets.get(str(args.preset)))
            preset_map.pop("cli_params", None)
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Unknown or unavailable preset '{args.preset}': {exc}")
        return load_renderer_config(preset_map, overrides)
    return load_renderer_config(None, overrides)


def _split_key_value_string(spec: str) -> dict[str, str]:
    tokens = spec.split(",")
    result: dict[str, str] = {}
    current_key: str | None = None
    for raw in tokens:
        segment = raw.strip()
        if not segment:
            continue
        if "=" in segment:
            key, value = segment.split("=", 1)
            current_key = key.strip().lower()
            result[current_key] = value.strip()
        elif current_key is not None:
            result[current_key] = f"{result[current_key]},{segment}"
        else:
            raise ValueError(f"Invalid segment '{segment}' in specification '{spec}'")
    return result


def _parse_float_list(value: str, length: int, label: str) -> tuple[float, ...]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != length:
        raise ValueError(f"{label} requires exactly {length} comma-separated floats")
    return tuple(parts)


def _parse_light_spec(spec: str) -> dict[str, Any]:
    mapping = _split_key_value_string(spec)
    out: dict[str, Any] = {}
    for key, val in mapping.items():
        if key in {"type", "light"}:
            out["type"] = val
        elif key in {"dir", "direction"}:
            out["direction"] = _parse_float_list(val, 3, "direction")
        elif key in {"pos", "position"}:
            out["position"] = _parse_float_list(val, 3, "position")
        elif key in {"intensity", "power"}:
            out["intensity"] = float(val)
        elif key in {"color", "rgb"}:
            out["color"] = _parse_float_list(val, 3, "color")
        elif key in {"cone", "cone_angle", "angle"}:
            out["cone_angle"] = float(val)
        elif key in {"area", "extent", "area_extent"}:
            out["area_extent"] = _parse_float_list(val, 2, "area extent")
        elif key in {"hdr", "hdr_path"}:
            out["hdr_path"] = val
    return out


def _parse_volumetric_spec(spec: str) -> dict[str, Any]:
    mapping = _split_key_value_string(spec)
    out: dict[str, Any] = {}
    if "density" in mapping:
        out["density"] = float(mapping["density"])
    if "phase" in mapping:
        out["phase"] = mapping["phase"]
    if "g" in mapping:
        out["g"] = float(mapping["g"])
    if "anisotropy" in mapping:
        out["anisotropy"] = float(mapping["anisotropy"])
    return out


def _make_terrain_shadow_settings(shadow_config) -> TerrainShadowSettings:
    """Create ShadowSettings for terrain rendering with early validation.
    
    Validates that the requested technique is supported by the terrain shader.
    
    Raises:
        ValueError: If the technique is not supported for terrain.
    """
    settings = TerrainShadowSettings(
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
    # Validate that technique is implemented in terrain shader
    settings.validate_for_terrain()
    return settings


def _build_colormap(
    domain: tuple[float, float],
    colormap_name: str = "viridis",
    *,
    heightmap=None,
    q_lo: float = QUANTILE_DEFAULT_LO,
    q_hi: float = QUANTILE_DEFAULT_HI,
    interpolate_custom: bool = False,
    custom_size: int = 256,
):
    """Build a Colormap1D for the DEM elevation domain."""

    def quantile_stops(colors: list[str]):
        if heightmap is None:
            return None
        h = np.asarray(heightmap, dtype=np.float32).reshape(-1)
        h = h[np.isfinite(h)]
        if h.size == 0:
            return None
        qs = np.linspace(q_lo, q_hi, len(colors))
        elevs = np.quantile(h, qs)
        return list(zip([float(e) for e in elevs], colors))

    # Custom hex palette
    if colormap_name and "," in colormap_name:
        hex_colors = [c.strip() for c in colormap_name.split(",")]

        import re

        hex_pattern = re.compile(r"^[0-9A-Fa-f]{6}$")
        invalid_colors = [c for c in hex_colors if not hex_pattern.search(c.lstrip("#"))]
        if invalid_colors:
            print(f"Warning: Invalid hex color format: {invalid_colors}")
            print("Expected format: #RRGGBB (e.g., #ff0000)")
            print("Falling back to viridis colormap")
            colormap_name = "viridis"
        else:
            if interpolate_custom and len(hex_colors) >= 2:
                hex_colors = _cm_interpolate_hex_colors(hex_colors, size=custom_size)

            stops = _cm_elevation_stops(
                domain,
                hex_colors,
                heightmap=heightmap,
                q_lo=q_lo,
                q_hi=q_hi,
            )
            print(f"Custom colormap created with {len(hex_colors)} colors")
            return f3d.Colormap1D.from_stops(stops=stops, domain=domain)

    # Named palette (viridis, magma, ...)
    if colormap_name and colormap_name != "terrain":
        try:
            cmap = f3d.get_colormap(f"forge3d:{colormap_name}")
            rgba_array = cmap.rgba  # (N,4)
            n_samples = 1024
            indices = np.linspace(0, len(rgba_array) - 1, n_samples, dtype=int)
            colors_hex: list[str] = []
            for idx in indices:
                rgba = rgba_array[idx]
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                colors_hex.append(f"#{r:02x}{g:02x}{b:02x}")

            stops = _cm_elevation_stops(
                domain,
                colors_hex,
                heightmap=heightmap,
                q_lo=q_lo,
                q_hi=q_hi,
            )
            return f3d.Colormap1D.from_stops(stops=stops, domain=domain)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: Failed to load colormap '{colormap_name}': {exc}")
            print("Falling back to terrain colormap (earth tones)")
            colormap_name = "terrain"

    # Fallback: terrain colormap using DEFAULT_COLORMAP_STOPS
    original_min = DEFAULT_COLORMAP_STOPS[0][0]
    original_max = DEFAULT_COLORMAP_STOPS[-1][0]
    original_range = original_max - original_min

    new_min, new_max = float(domain[0]), float(domain[1])
    new_range = new_max - new_min

    stops = []
    for value, color in DEFAULT_COLORMAP_STOPS:
        t = (value - original_min) / original_range
        mapped_value = new_min + t * new_range
        stops.append((mapped_value, color))

    return f3d.Colormap1D.from_stops(stops=stops, domain=domain)


def _build_params(
    size: tuple[int, int],
    render_scale: float,
    terrain_span: float,
    msaa: int,
    z_scale: float,
    exposure: float,
    domain: tuple[float, float],
    colormap,
    albedo_mode: str = "mix",
    colormap_strength: float = 0.5,
    ibl_enabled: bool = True,
    pom_enabled: bool | None = None,
    light_azimuth_deg: float = 135.0,
    light_elevation_deg: float = 35.0,
    sun_intensity: float = 3.0,
    sun_color: Sequence[float] | None = None,
    ibl_intensity: float = 1.0,
    cam_radius: float = 1200.0,
    cam_phi_deg: float = 135.0,
    cam_theta_deg: float = 45.0,
    fov_y_deg: float = 55.0,
    camera_mode: str = "screen",
    debug_mode: int = 0,
    clip: tuple[float, float] | None = None,
    height_curve_mode: str = "linear",
    height_curve_strength: float = 0.0,
    height_curve_power: float = 1.0,
    height_curve_lut=None,
    shadow_config=None,  # Optional ShadowParams from CLI
    fog_config=None,  # P2: Optional FogSettings from CLI
    reflection_config=None,  # P4: Optional ReflectionSettings from CLI
    triplanar_config=None,  # P5-N: Optional TriplanarSettings for normal_strength
    lambert_contrast: float = 0.0,  # P5-L: Lambert contrast curve strength
    colormap_srgb: bool = False,  # P6.1: Use Rgba8UnormSrgb for colormap texture
    output_srgb_eotf: bool = False,  # P6.1: Use exact linear_to_srgb() output encoding
):
    overlays = [
        f3d.OverlayLayer.from_colormap1d(
            colormap,
            strength=1.0,
            offset=0.0,
            blend_mode="Alpha",
            domain=domain,
        )
    ]

    config = make_terrain_params_config(
        size_px=size,
        render_scale=render_scale,
        msaa_samples=msaa,
        z_scale=z_scale,
        exposure=exposure,
        domain=domain,
        albedo_mode=albedo_mode,
        colormap_strength=colormap_strength,
        ibl_enabled=ibl_enabled,
        light_azimuth_deg=light_azimuth_deg,
        light_elevation_deg=light_elevation_deg,
        sun_intensity=sun_intensity,
        sun_color=sun_color,
        ibl_intensity=ibl_intensity,
        cam_radius=cam_radius,
        cam_phi_deg=cam_phi_deg,
        cam_theta_deg=cam_theta_deg,
        fov_y_deg=fov_y_deg,
        terrain_span=terrain_span,
        camera_mode=camera_mode,
        debug_mode=debug_mode,
        clip=clip,
        height_curve_mode=height_curve_mode,
        height_curve_strength=height_curve_strength,
        height_curve_power=height_curve_power,
        height_curve_lut=height_curve_lut,
        shadows=_make_terrain_shadow_settings(shadow_config),
        overlays=overlays,
        fog=fog_config,  # P2: Pass fog config (None = disabled)
        reflection=reflection_config,  # P4: Pass reflection config (None = disabled)
        triplanar=triplanar_config,  # P5-N: Pass triplanar config for normal_strength
        lambert_contrast=lambert_contrast,  # P5-L: Pass lambert contrast
    )
    
    # P6.1: Set color space correctness toggles
    config.colormap_srgb = colormap_srgb
    config.output_srgb_eotf = output_srgb_eotf

    if pom_enabled is not None:
        try:
            config.pom.enabled = bool(pom_enabled)
        except AttributeError:
            pass

    return f3d.TerrainRenderParams(config)


def _save_image(img, path: Path) -> None:
    try:
        from PIL import Image  # type: ignore[import]

        Image.fromarray(img, mode="RGBA").save(str(path))
    except Exception:
        try:
            f3d.numpy_to_png(str(path), img)
        except Exception:
            np.save(str(path).replace(".png", ".npy"), img)
            print("  Warning: Saved as .npy (no PNG writer available)")


def _apply_luminance_unsharp(frame, strength: float):
    """P5-US: Apply luminance unsharp mask for gradient enhancement.
    
    Enhances local contrast by boosting high-frequency luminance detail.
    Works on the luminance channel only to preserve color relationships.
    
    Style Match Fix: Use multi-scale unsharp for terrain texture visibility.
    
    Args:
        frame: Rendered frame (forge3d.Frame or numpy array)
        strength: Unsharp strength k in [0.1, 1.0]. Higher = more contrast.
    Returns:
        Modified frame with enhanced gradients
    """
    from scipy.ndimage import gaussian_filter
    
    # Get numpy array from frame
    if hasattr(frame, "to_numpy"):
        img = frame.to_numpy()
    elif hasattr(frame, "__array__"):
        img = np.asarray(frame)
    else:
        img = frame
    
    # Convert to float [0,1] if needed
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
        was_uint8 = True
    else:
        img = img.astype(np.float32)
        was_uint8 = False
    
    # Extract RGB (ignore alpha if present)
    rgb = img[..., :3]
    alpha = img[..., 3:4] if img.shape[-1] == 4 else None
    
    # Convert to luminance (Rec. 709)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    
    # Style Match: Multi-scale unsharp for terrain microstructure
    # Fine scale (sigma=1.5) captures micro-ridges
    # Coarse scale (sigma=4.0) captures macro-terrain features
    blurred_fine = gaussian_filter(luma, sigma=1.5)
    blurred_coarse = gaussian_filter(luma, sigma=4.0)
    
    detail_fine = luma - blurred_fine
    detail_coarse = luma - blurred_coarse
    
    # Combine multi-scale detail (fine gets more weight for terrain texture)
    combined_detail = detail_fine * 0.7 + detail_coarse * 0.3
    
    # Moderate shadow protection (less aggressive to allow shadow detail)
    shadow_protect = np.power(np.maximum(luma, 0.05), 0.3)
    
    # Apply stronger unsharp for terrain edge visibility
    luma_enhanced = np.clip(luma + strength * 2.0 * combined_detail * shadow_protect, 0.0, 1.0)
    
    # Scale RGB to match new luminance (preserve color ratios)
    # Avoid division by zero
    luma_safe = np.maximum(luma, 1e-6)
    scale = luma_enhanced / luma_safe
    rgb_enhanced = np.clip(rgb * scale[..., np.newaxis], 0.0, 1.0)
    
    # Reconstruct with alpha
    if alpha is not None:
        result = np.concatenate([rgb_enhanced, alpha], axis=-1)
    else:
        result = rgb_enhanced
    
    # Convert back to uint8 if needed
    if was_uint8:
        result = (result * 255.0).astype(np.uint8)
    
    # Wrap back in Frame if needed
    if hasattr(frame, "from_numpy"):
        return frame.from_numpy(result)
    elif hasattr(f3d, "Frame"):
        return f3d.Frame.from_numpy(result)
    else:
        # Return a simple wrapper that has .save()
        class _FrameWrapper:
            def __init__(self, data):
                self._data = data
            def save(self, path):
                from PIL import Image
                if self._data.dtype != np.uint8:
                    self._data = (self._data * 255).astype(np.uint8)
                Image.fromarray(self._data).save(path)
        return _FrameWrapper(result)


def _apply_preset_cli_defaults(args: Any) -> None:
    """Apply preset payload defaults to CLI args without overriding explicit flags."""
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        return

    preset = _preset_payload(str(preset_name))
    if not preset:
        return

    cli_params = _mapping(preset.get("cli_params"))
    camera = _mapping(preset.get("camera"))
    sun = _mapping(preset.get("sun"))
    ibl = _mapping(preset.get("ibl"))

    _set_arg_default(args, "camera_mode", cli_params.get("camera_mode") or camera.get("camera_mode"), DEFAULT_CAMERA_MODE)
    _set_arg_default(args, "cam_radius", cli_params.get("cam_radius") or camera.get("radius"), DEFAULT_CAM_RADIUS)
    _set_arg_default(args, "cam_theta", cli_params.get("cam_theta") or camera.get("theta_deg") or camera.get("elevation_deg"), DEFAULT_CAM_THETA)
    _set_arg_default(args, "cam_phi", cli_params.get("cam_phi") or camera.get("azimuth_deg"), DEFAULT_CAM_PHI)
    _set_arg_default(args, "cam_fov", cli_params.get("cam_fov") or camera.get("fov_deg"), DEFAULT_CAM_FOV)

    _set_arg_default(args, "sun_azimuth", sun.get("azimuth_deg"), None)
    _set_arg_default(args, "sun_elevation", sun.get("elevation_deg"), None)
    _set_arg_default(args, "sun_intensity", sun.get("intensity"), None)
    _set_arg_default(args, "sun_color", sun.get("color"), None)

    _set_arg_default(args, "ibl_intensity", ibl.get("intensity"), 1.0)
    _set_arg_default(args, "hdr", ibl.get("path") or ibl.get("hdr_path") or _builtin_ibl_path(ibl.get("builtin")), DEFAULT_HDR)
    _set_arg_default(args, "z_scale", preset.get("exaggeration"), 2.0)


def _apply_preset_dem_defaults(args: Any, terrain_span: float) -> None:
    """Apply preset defaults that depend on DEM dimensions."""
    preset_name = getattr(args, "preset", None)
    if not preset_name or _arg_was_explicit(args, "cam_radius"):
        return

    preset = _preset_payload(str(preset_name))
    camera = _mapping(preset.get("camera"))
    radius_scale = camera.get("radius_scale")
    if radius_scale is None:
        return
    current = getattr(args, "cam_radius", DEFAULT_CAM_RADIUS)
    if current is None or _values_equal(current, DEFAULT_CAM_RADIUS):
        setattr(args, "cam_radius", max(1.0, float(terrain_span)) * float(radius_scale))


def render_sunrise_to_noon_sequence(
    *,
    dem_path: Path,
    hdr_path: Path,
    output_dir: Path,
    size: tuple[int, int] = (320, 180),
    steps: int = 4,
    certificate: bool | str | Path = False,
    cache: str | Path | None = None,
) -> list[Path]:
    """Render a short sunrise-to-noon sequence using RendererConfig."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(certificate, (str, Path)):
        Path(certificate).mkdir(parents=True, exist_ok=True)

    use_native = bool(f3d.has_gpu()) and all(
        hasattr(f3d, name)
        for name in ("Session", "TerrainRenderer", "MaterialSet", "IBL", "TerrainRenderParams")
    )
    if not use_native:
        raise RuntimeError(
            "render_sunrise_to_noon_sequence requires the native terrain renderer; "
            "the fallback triangle placeholder has been removed"
        )

    dem = _load_dem(dem_path)
    heightmap_array = getattr(dem, "data", None)
    if heightmap_array is None:
        raise SystemExit("DEM object does not expose a .data array")

    domain_meta = _io.infer_dem_domain(dem, fallback=DEFAULT_DOMAIN)
    domain = _io.robust_dem_domain(
        heightmap_array,
        q_lo=QUANTILE_DEFAULT_LO,
        q_hi=QUANTILE_DEFAULT_HI,
        fallback=(float(domain_meta[0]), float(domain_meta[1])),
    )
    domain = (float(domain[0]), float(domain[1]))
    colormap = _build_colormap(domain, colormap_name="terrain", heightmap=heightmap_array)

    base_overrides: dict[str, Any] = {
        "sky": "hosek-wilkie",
        "hdr": str(hdr_path),
        "gi": ["ibl"],
        "volumetric": {
            "density": 0.03,
            "phase": "hg",
            "g": 0.7,
            "mode": "raymarch",
            "max_steps": 48,
        },
    }

    sess = None
    renderer_native = None
    materials = None
    ibl = None

    if use_native:
        sess = f3d.Session(window=False)
        renderer_native = f3d.TerrainRenderer(sess)
        materials = f3d.MaterialSet.terrain_default(
            triplanar_scale=6.0,
            normal_strength=1.0,
            blend_sharpness=4.0,
        )
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
        ibl.set_base_resolution(256)

    if steps <= 1:
        sun_elevations: list[float] = [30.0]
    else:
        sun_elevations = np.linspace(5.0, 60.0, int(steps), dtype=float).tolist()

    azimuth_deg = 135.0
    width, height = int(size[0]), int(size[1])
    outputs: list[Path] = []

    for idx, elev in enumerate(sun_elevations):
        sun_dir = _sun_direction_from_angles(azimuth_deg, float(elev))

        overrides = dict(base_overrides)
        overrides["light"] = [
            {
                "type": "directional",
                "direction": list(sun_dir),
                "intensity": 5.0,
                "color": [1.0, 0.97, 0.94],
            }
        ]

        cfg = load_renderer_config(None, overrides)

        out_path = output_dir / f"terrain_sunrise_{idx:02d}.png"
        if sess is not None and renderer_native is not None and materials is not None and ibl is not None:
            ibl_enabled = "ibl" in cfg.gi.modes
            clip_far = max(6000.0, terrain_span * 1.5)
            params = _build_params(
                size=(width, height),
                render_scale=1.0,
                terrain_span=terrain_span,
                msaa=1,
                z_scale=1.0,
                exposure=float(cfg.lighting.exposure),
                domain=domain,
                colormap=colormap,
                albedo_mode="mix",
                colormap_strength=0.5,
                ibl_enabled=ibl_enabled,
                light_azimuth_deg=azimuth_deg,
                light_elevation_deg=elev,
                shadow_config=cfg.shadows,  # Pass shadow config from renderer config
                clip=(0.1, clip_far),
            )

            # Native binding currently expects a non-None IBL handle; turning IBL on/off
            # is controlled via params (ibl_enabled/intensity), not by omitting env_maps.
            frame = renderer_native.render_terrain_pbr_pom(
                material_set=materials,
                env_maps=ibl,
                params=params,
                heightmap=heightmap_array,
                target=None,
                certificate=(
                    Path(certificate) / f"terrain_sunrise_{idx:02d}.certificate.json"
                    if isinstance(certificate, (str, Path))
                    else out_path.with_suffix(".certificate.json")
                    if certificate
                    else False
                ),
                cache=cache,
            )
            rgba = frame.to_numpy()
        else:
            raise RuntimeError("native terrain renderer initialization was incomplete")

        _save_image(rgba, out_path)
        outputs.append(out_path)

    return outputs


def run(args: Any) -> int:
    """Execute the terrain demo render using an argparse-style args object."""

    height_curve_lut = None
    if getattr(args, "height_curve_mode", None) == "lut":
        if getattr(args, "height_curve_lut", None) is None:
            raise SystemExit(
                "Error: --height-curve-lut is required when --height-curve-mode=lut"
            )
        height_curve_lut = load_height_curve_lut(args.height_curve_lut)

    _require_attributes(
        (
            "Session",
            "TerrainRenderer",
            "MaterialSet",
            "IBL",
            "Colormap1D",
            "TerrainRenderParams",
            "LightSettings",
            "IblSettings",
            "ShadowSettings",
            "TriplanarSettings",
            "PomSettings",
            "LodSettings",
            "SamplingSettings",
            "ClampSettings",
            "OverlayLayer",
        )
    )

    _apply_preset_cli_defaults(args)

    renderer_config = _build_renderer_config(args)

    if args.output.exists() and not getattr(args, "overwrite", False):
        raise SystemExit(
            f"Output file already exists: {args.output}. Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sess = f3d.Session(window=bool(getattr(args, "window", False)))

    dem = _load_dem(args.dem)
    dem_dx_m, dem_dy_m, terrain_span, spacing_note = _dem_spacing_info(dem)
    _apply_preset_dem_defaults(args, terrain_span)
    heightmap_array = dem.data

    if getattr(args, "colormap_domain", None) is not None:
        domain_meta = tuple(args.colormap_domain)  # type: ignore[arg-type]
    else:
        domain_meta = _io.infer_dem_domain(dem, fallback=DEFAULT_DOMAIN)

    domain = _io.robust_dem_domain(
        heightmap_array,
        q_lo=QUANTILE_DEFAULT_LO,
        q_hi=QUANTILE_DEFAULT_HI,
        fallback=(float(domain_meta[0]), float(domain_meta[1])),
    )
    domain = (float(domain[0]), float(domain[1]))
    print(
        f"[DEM] spacing_m=({dem_dx_m:.3f},{dem_dy_m:.3f}) span={terrain_span:.2f} source={spacing_note}"
    )

    colormap = _build_colormap(
        domain,
        colormap_name=args.colormap,
        heightmap=heightmap_array,
        interpolate_custom=bool(getattr(args, "colormap_interpolate", False)),
        custom_size=int(getattr(args, "colormap_size", 256)),
    )

    # P5-N: Get normal_strength from CLI (default 1.0, range 0.25-4.0)
    normal_strength = float(getattr(args, "normal_strength", 1.0))
    triplanar_config = TriplanarSettings(
        scale=6.0,
        blend_sharpness=4.0,
        normal_strength=normal_strength,
    )
    materials = f3d.MaterialSet.terrain_default(
        triplanar_scale=6.0,
        normal_strength=1.0,  # MaterialSet normal_strength is for PBR textures, not terrain normals
        blend_sharpness=4.0,
    )

    hdr_path_value: Path | str = args.hdr
    if renderer_config.atmosphere.hdr_path is not None:
        hdr_path_value = Path(renderer_config.atmosphere.hdr_path)
    else:
        env_hdr = next(
            (
                light.hdr_path
                for light in renderer_config.lighting.lights
                if light.type == "environment" and light.hdr_path is not None
            ),
            None,
        )
        if env_hdr is not None:
            hdr_path_value = Path(env_hdr)

    ibl = f3d.IBL.from_hdr(
        str(hdr_path_value),
        intensity=float(args.ibl_intensity),
        rotate_deg=0.0,
    )
    if args.ibl_res <= 0:
        raise SystemExit("Error: --ibl-res must be greater than zero.")
    ibl.set_base_resolution(int(args.ibl_res))
    if getattr(args, "ibl_cache", None) is not None:
        ibl.set_cache_dir(str(args.ibl_cache))

    # Respect GI configuration: enable IBL only when 'ibl' is present in gi.modes.
    # This mirrors render_sunrise_to_noon_sequence(), where cfg.gi.modes controls
    # whether the native TerrainRenderer receives env_maps or None.
    ibl_enabled = "ibl" in renderer_config.gi.modes

    sun_azimuth_deg = float(args.sun_azimuth) if getattr(args, "sun_azimuth", None) is not None else 135.0
    sun_elevation_deg = float(args.sun_elevation) if getattr(args, "sun_elevation", None) is not None else 35.0
    sun_intensity = float(args.sun_intensity) if getattr(args, "sun_intensity", None) is not None else 3.0
    sun_color = tuple(args.sun_color) if getattr(args, "sun_color", None) is not None else None

    # Check for camera-sun alignment that causes flat lighting
    cam_phi_deg = float(args.cam_phi)
    cam_theta_deg = float(args.cam_theta)
    fov_y_deg = float(args.cam_fov)
    
    # PHASE 0: Debug camera logging (FORGE3D_DEBUG_CAMERA=1)
    # Proves whether the engine is honoring the requested low-angle lighting and camera orientation.
    # cam_theta is polar angle from +Y axis: theta=0 = top-down, theta=90 = horizon
    # elevation_above_horizon = 90 - theta
    # camera_pitch = elevation (angle looking up from horizon)
    if os.environ.get("FORGE3D_DEBUG_CAMERA") == "1":
        cam_radius = float(args.cam_radius)
        theta_rad = math.radians(cam_theta_deg)
        phi_rad = math.radians(cam_phi_deg)
        eye_x = cam_radius * math.sin(theta_rad) * math.cos(phi_rad)
        eye_y = cam_radius * math.cos(theta_rad)
        eye_z = cam_radius * math.sin(theta_rad) * math.sin(phi_rad)
        view_len = math.sqrt(eye_x**2 + eye_y**2 + eye_z**2) or 1.0
        view_dir = (-eye_x / view_len, -eye_y / view_len, -eye_z / view_len)
        sun_dir = _sun_direction_from_angles(sun_azimuth_deg, sun_elevation_deg)
        view_light_dot = view_dir[0] * sun_dir[0] + view_dir[1] * sun_dir[1] + view_dir[2] * sun_dir[2]
        render_mode = str(getattr(args, "camera_mode", DEFAULT_CAMERA_MODE)).lower()
        render_path = "mesh_perspective_grid" if render_mode == "mesh" else "screen_fullscreen_triangle"
        print(
            "[FORGE3D_DEBUG_CAMERA]"
            f" render_mode={render_mode} path={render_path}"
            f" eye=({eye_x:.2f},{eye_y:.2f},{eye_z:.2f}) target=(0.0,0.0,0.0)"
            f" view_dir=({view_dir[0]:.4f},{view_dir[1]:.4f},{view_dir[2]:.4f})"
            f" sun_dir=({sun_dir[0]:.4f},{sun_dir[1]:.4f},{sun_dir[2]:.4f})"
            f" dot={view_light_dot:.4f}"
            f" cli_phi_deg={cam_phi_deg:.2f} cli_theta_deg={cam_theta_deg:.2f} cli_fov_y_deg={fov_y_deg:.2f}"
            f" sun_azimuth_deg={sun_azimuth_deg:.2f} sun_elevation_deg={sun_elevation_deg:.2f}"
            f" dem_texel_m=({dem_dx_m:.3f},{dem_dy_m:.3f}) terrain_span={terrain_span:.2f}"
            f" z_scale={float(args.z_scale):.3f} spacing_note={spacing_note}"
        )

    alignment_dot = check_camera_sun_alignment(
        cam_phi_deg, cam_theta_deg, sun_azimuth_deg, sun_elevation_deg
    )
    if alignment_dot > 0.85:
        print(
            f"[WARNING] Sun is nearly aligned with camera (dot={alignment_dot:.2f}). "
            f"Shadows will appear flat. Adjust sun or camera angles to introduce cross-lighting."
        )

    # P2: Parse fog parameters from CLI
    fog_density = float(getattr(args, "fog_density", 0.0) or 0.0)
    fog_height_falloff = float(getattr(args, "fog_height_falloff", 0.0) or 0.0)
    fog_inscatter_str = getattr(args, "fog_inscatter", "1.0,1.0,1.0") or "1.0,1.0,1.0"
    try:
        fog_inscatter = tuple(float(x.strip()) for x in fog_inscatter_str.split(","))[:3]
        if len(fog_inscatter) < 3:
            fog_inscatter = (1.0, 1.0, 1.0)
    except (ValueError, AttributeError):
        fog_inscatter = (1.0, 1.0, 1.0)
    
    # Only create FogSettings if density > 0 (otherwise leave as None for no-op)
    fog_config = None
    if fog_density > 0.0:
        fog_config = TerrainFogSettings(
            density=fog_density,
            height_falloff=fog_height_falloff,
            inscatter=fog_inscatter,
        )

    # P4: Parse reflection parameters from CLI
    water_reflections = bool(getattr(args, "water_reflections", False))
    reflection_intensity = float(getattr(args, "reflection_intensity", 0.8) or 0.8)
    reflection_fresnel_power = float(getattr(args, "reflection_fresnel_power", 5.0) or 5.0)
    reflection_wave_strength = float(getattr(args, "reflection_wave_strength", 0.02) or 0.02)
    reflection_shore_atten = float(getattr(args, "reflection_shore_atten", 0.3) or 0.3)
    reflection_plane_height = float(getattr(args, "reflection_plane_height", 0.0) or 0.0)
    
    # Only create ReflectionSettings if enabled (otherwise leave as None for no-op)
    reflection_config = None
    if water_reflections:
        reflection_config = TerrainReflectionSettings(
            enabled=True,
            intensity=reflection_intensity,
            fresnel_power=reflection_fresnel_power,
            wave_strength=reflection_wave_strength,
            shore_atten_width=reflection_shore_atten,
            water_plane_height=reflection_plane_height,
        )

    # P6: Parse detail normal parameters from CLI
    render_mode = getattr(args, "render", None)
    detail_normal_path = getattr(args, "detail_normals", None)
    detail_strength = float(getattr(args, "detail_strength", 0.0) or 0.0)
    detail_sigma_px = float(getattr(args, "detail_sigma_px", 3.0) or 3.0)
    
    # P6 render mode preset: auto-generate detail normals if not provided
    if render_mode == "p6":
        if detail_normal_path is None:
            # Auto-generate detail normals from DEM
            generated_path = Path("assets/generated") / f"detail_normal_sigma{detail_sigma_px:.1f}.png"
            if not generated_path.exists():
                print(f"[P6] Generating detail normal map (sigma={detail_sigma_px} px)...")
                import subprocess
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).parent.parent.parent / "tools" / "detail_normals.py"),
                        "--dem", str(args.dem),
                        "--sigma-px", str(detail_sigma_px),
                        "--output", str(generated_path),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"[P6] Warning: Failed to generate detail normals: {result.stderr}")
                else:
                    print(f"[P6] Generated: {generated_path}")
            detail_normal_path = generated_path
        # P6 preset: enable detail normals with default strength
        if detail_strength <= 0.0:
            detail_strength = 0.5  # P6 default strength
    
    # Log P6 detail normal configuration
    if detail_normal_path is not None and detail_strength > 0.0:
        print(f"[P6] Detail normals: {detail_normal_path} (strength={detail_strength:.2f})")

    # Log shadow technique selection (P6.2 acceptance criterion B)
    shadow_cfg = renderer_config.shadows
    print(f"[SHADOW] technique={shadow_cfg.technique.upper()}, cascades={shadow_cfg.cascades}, res={shadow_cfg.map_size}")

    clip_far = max(6000.0, terrain_span * 1.5)
    params = _build_params(
        size=(int(args.size[0]), int(args.size[1])),
        render_scale=float(args.render_scale),
        terrain_span=terrain_span,
        msaa=int(args.msaa),
        z_scale=float(args.z_scale),
        exposure=float(renderer_config.lighting.exposure),
        domain=domain,
        colormap=colormap,
        albedo_mode=args.albedo_mode,
        colormap_strength=float(args.colormap_strength),
        ibl_enabled=ibl_enabled,
        pom_enabled=not bool(getattr(args, "pom_disabled", False)),
        light_azimuth_deg=sun_azimuth_deg,
        light_elevation_deg=sun_elevation_deg,
        sun_intensity=sun_intensity,
        sun_color=sun_color,
        ibl_intensity=float(args.ibl_intensity),
        cam_radius=float(args.cam_radius),
        cam_phi_deg=float(args.cam_phi),
        cam_theta_deg=float(args.cam_theta),
        fov_y_deg=float(args.cam_fov),
        camera_mode=str(getattr(args, "camera_mode", "screen")),
        debug_mode=int(getattr(args, "debug_mode", 0)),
        clip=(0.1, clip_far),
        height_curve_mode=str(args.height_curve_mode),
        height_curve_strength=float(args.height_curve_strength),
        height_curve_power=float(args.height_curve_power),
        height_curve_lut=height_curve_lut,
        shadow_config=renderer_config.shadows,  # Pass CLI shadow config
        fog_config=fog_config,  # P2: Pass fog config
        reflection_config=reflection_config,  # P4: Pass reflection config
        triplanar_config=triplanar_config,  # P5-N: Pass triplanar config for normal_strength
        lambert_contrast=float(getattr(args, "lambert_contrast", 0.0)),  # P5-L
        colormap_srgb=bool(getattr(args, "colormap_srgb", False)),  # P6.1
        output_srgb_eotf=bool(getattr(args, "output_srgb_eotf", False)),  # P6.1
    )

    renderer = f3d.TerrainRenderer(sess)

    if renderer_config.lighting.lights:
        light_dicts = []
        for light in renderer_config.lighting.lights:
            d: dict[str, Any] = {"type": light.type}
            if getattr(light, "azimuth", None) is not None:
                d["azimuth"] = light.azimuth
            if getattr(light, "elevation", None) is not None:
                d["elevation"] = light.elevation
            if getattr(light, "position", None) is not None:
                d["position"] = light.position
            if getattr(light, "direction", None) is not None:
                d["direction"] = light.direction
            if getattr(light, "intensity", None) is not None:
                d["intensity"] = light.intensity
            if getattr(light, "color", None) is not None:
                try:
                    d["color"] = list(light.color)
                except TypeError:
                    d["color"] = light.color
            if getattr(light, "range", None) is not None:
                d["range"] = light.range
            if getattr(light, "cone_angle", None) is not None:
                d["cone_angle"] = light.cone_angle
            if getattr(light, "inner_angle", None) is not None:
                d["inner_angle"] = light.inner_angle
            if getattr(light, "outer_angle", None) is not None:
                d["outer_angle"] = light.outer_angle
            if getattr(light, "area_extent", None) is not None:
                d["area_extent"] = light.area_extent
            if getattr(light, "radius", None) is not None:
                d["radius"] = light.radius
            light_dicts.append(d)

        try:
            renderer.set_lights(light_dicts)
            print(f"[P1-09] Uploaded {len(light_dicts)} light(s) to GPU")
            if getattr(args, "debug_lights", False):
                print("\n" + "=" * 60)
                print("Light Buffer Debug Info:")
                print("=" * 60)
                print(renderer.light_debug_info())
                print("=" * 60 + "\n")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: Failed to upload lights: {exc}")

    frame = renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=ibl,
        params=params,
        heightmap=heightmap_array,
        target=None,
    )

    # P5-US: Apply luminance unsharp mask for gradient enhancement
    unsharp_strength = float(getattr(args, "unsharp_strength", 0.0))
    # P6: Check render mode preset for unsharp override
    render_mode = getattr(args, "render", None)
    if render_mode == "p5":
        # P5 preset: may use unsharp for gradients (legacy behavior)
        pass
    elif render_mode == "p6":
        # P6 preset: detail normals handle gradients, disable unsharp
        # (P6 must meet gradient targets without unsharp)
        unsharp_strength = 0.0
    
    if unsharp_strength > 0.0:
        frame = _apply_luminance_unsharp(frame, unsharp_strength)

    frame.save(str(args.output))
    print(f"Wrote {args.output}")

    if getattr(args, "viewer", False):
        viewer_cls = getattr(f3d, "Viewer", None)
        if viewer_cls is None:
            print("forge3d.Viewer is not available in this build. Skipping interactive viewer.")
        else:
            if getattr(args, "sky", None):
                os.environ["FORGE3D_SKY_MODEL"] = str(args.sky)
            with viewer_cls(sess, renderer, heightmap_array, materials, ibl, params) as view:
                view.run()

    return 0
