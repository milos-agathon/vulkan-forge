# python/forge3d/terrain_params.py
# Typed dataclasses describing terrain renderer configuration
# Exists to gather all tunable terrain parameters in one validated place
# RELEVANT FILES: python/forge3d/__init__.py, tests/test_terrain_params.py, src/session.rs, src/colormap1d.rs
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral
from typing import TYPE_CHECKING, List, Optional, Tuple, Sequence

import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from . import AtmosphereLutHandle


_UNSET = object()


def _as_f32(value: object) -> float:
    """Normalize public scalar settings exactly as their native f32 seam."""

    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.asarray(value, dtype=np.float32).item())


@dataclass
class LightSettings:
    """Directional, point, or spot light configuration."""

    light_type: str  # "Directional", "Point", "Spot"
    azimuth_deg: float
    elevation_deg: float
    intensity: float
    color: List[float]  # [R, G, B]

    def __post_init__(self) -> None:
        valid_types = {"Directional", "Point", "Spot"}
        if self.light_type not in valid_types:
            raise ValueError(f"Invalid light_type: {self.light_type}")

        if len(self.color) != 3:
            raise ValueError("color must be [R, G, B]")

        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")


@dataclass
class IblSettings:
    """Image based lighting configuration."""

    enabled: bool
    intensity: float
    rotation_deg: float

    def __post_init__(self) -> None:
        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")


@dataclass
class ShadowSettings:
    """Shadow mapping configuration."""

    enabled: bool
    technique: str  # "HARD", "PCF", "PCSS", "CSM" (terrain-supported moment techniques included)
    resolution: int
    cascades: int
    max_distance: float
    softness: float
    intensity: float
    slope_scale_bias: float
    depth_bias: float
    normal_bias: float
    min_variance: float
    light_bleed_reduction: float
    evsm_exponent: float
    fade_start: float
    # Legacy PCSS light radius in world units. When non-zero it takes precedence
    # over light_size and is converted per cascade using that cascade's texel size.
    pcss_light_radius: float = 0.0
    # PCSS search radius, maximum adaptive filter radius, and area-light radius
    # in shadow-map texels.
    pcss_blocker_radius: float = 6.0
    pcss_filter_radius: float = 4.0
    light_size: float = 1.0

    # Shadow technique constants matching Rust ShadowTechnique enum
    # ALL_TECHNIQUES: Full set recognized by config layer
    ALL_TECHNIQUES = {"NONE", "HARD", "PCF", "PCSS", "VSM", "EVSM", "MSM"}
    # P0.2/M3: TERRAIN_SUPPORTED_TECHNIQUES now includes VSM/EVSM/MSM
    # Note: CSM is the pipeline, not a filter - use HARD/PCF/PCSS/VSM/EVSM/MSM as the technique
    TERRAIN_SUPPORTED_TECHNIQUES = {"NONE", "HARD", "PCF", "PCSS", "VSM", "EVSM", "MSM"}
    # Alias for backwards compatibility
    SUPPORTED_TECHNIQUES = ALL_TECHNIQUES
    # Memory budget: 512 MiB host-visible heap (AGENTS.md constraint)
    MAX_SHADOW_MEMORY_BYTES = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        # Normalize technique to uppercase for consistent validation
        self.technique = self.technique.upper()
        
        # Validate technique against full set first (catch typos)
        if self.technique not in self.ALL_TECHNIQUES:
            supported_list = ", ".join(sorted(self.ALL_TECHNIQUES - {"NONE"}))
            raise ValueError(
                f"Unsupported shadow technique: {self.technique!r}. "
                f"Supported techniques: {supported_list}. "
                f"Use 'NONE' to disable shadows."
            )
        
        # When technique is NONE, shadows are disabled
        if self.technique == "NONE":
            self.enabled = False

        valid_resolutions = {512, 1024, 2048, 4096, 8192}
        if self.resolution not in valid_resolutions:
            raise ValueError("resolution must be power of 2 between 512-8192")
        if not 1 <= self.cascades <= 4:
            raise ValueError("cascades must be 1-4")

        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")

        if self.softness < 0.0:
            raise ValueError("softness must be >= 0")

        for name, value, allow_zero in (
            ("pcss_light_radius", self.pcss_light_radius, True),
            ("pcss_blocker_radius", self.pcss_blocker_radius, True),
            ("pcss_filter_radius", self.pcss_filter_radius, True),
            ("light_size", self.light_size, False),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0 or (not allow_zero and value == 0.0):
                operator = ">=" if allow_zero else ">"
                raise ValueError(f"{name} must be {operator} 0")

        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")

        if self.min_variance < 0.0:
            raise ValueError("min_variance must be >= 0")

        if self.light_bleed_reduction < 0.0:
            raise ValueError("light_bleed_reduction must be >= 0")

        if self.evsm_exponent <= 0.0:
            raise ValueError("evsm_exponent must be > 0")

        # Memory budget check (AGENTS.md: ≤512 MiB host-visible heap)
        mem_bytes = self._estimate_memory_bytes()
        if mem_bytes > self.MAX_SHADOW_MEMORY_BYTES:
            mem_mib = mem_bytes / (1024 * 1024)
            max_mib = self.MAX_SHADOW_MEMORY_BYTES / (1024 * 1024)
            raise ValueError(
                f"Shadow resources exceed memory budget: {mem_mib:.1f} MiB > {max_mib:.0f} MiB. "
                f"Reduce resolution ({self.resolution}) or cascades ({self.cascades})."
            )

    def _estimate_memory_bytes(self) -> int:
        """Estimate GPU memory for shadow resources."""
        pixels = self.resolution * self.resolution * self.cascades
        depth_bytes = pixels * 4  # Depth32Float
        # All moment techniques use an Rgba16Float atlas and an equally-sized
        # persistent intermediate for the separable blur.
        if self.technique in {"VSM", "EVSM", "MSM"}:
            moment_bytes = pixels * 16  # atlas (8) + blur intermediate (8)
        else:
            moment_bytes = 0
        return depth_bytes + moment_bytes

    def validate_for_terrain(self) -> None:
        """Validate that this shadow technique is implemented in the terrain pipeline.

        Raises ValueError with a clear message if the technique is not supported.
        Terrain supports HARD/PCF/PCSS plus moment-map variants VSM/EVSM/MSM.
        """
        if self.technique not in self.TERRAIN_SUPPORTED_TECHNIQUES:
            terrain_list = ", ".join(sorted(self.TERRAIN_SUPPORTED_TECHNIQUES - {"NONE"}))
            raise ValueError(
                f"Shadow technique {self.technique!r} is not supported for terrain rendering. "
                f"Terrain-supported techniques: {terrain_list}. "
                f"Use 'NONE' to disable shadows."
            )


@dataclass
class FogSettings:
    """P2: Atmospheric fog configuration.
    
    Height-based exponential fog applied after PBR, before tonemap.
    When density = 0.0, fog is disabled (no-op for P1 compatibility).
    
    base_height: World-space Z coordinate below which fog is at full density.
                 Should be set to the minimum terrain elevation (in world units).
                 If None, will be auto-computed from terrain bounds.
    """

    density: float = 0.0  # 0.0 = disabled
    height_falloff: float = 0.0
    base_height: Optional[float] = None  # None = auto from terrain min height
    inscatter: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.density < 0.0:
            raise ValueError("density must be >= 0")
        if self.height_falloff < 0.0:
            raise ValueError("height_falloff must be >= 0")
        if len(self.inscatter) != 3:
            raise ValueError("inscatter must be (R, G, B)")
        for c in self.inscatter:
            if not 0.0 <= c <= 1.0:
                raise ValueError("inscatter components must be in [0, 1]")


@dataclass
class ReflectionSettings:
    """P4: Water planar reflection configuration.
    
    When enabled=False, reflections are disabled (no-op for P3 compatibility).
    Reflections sample a half-resolution render of the scene mirrored across
    the water plane, with wave-based UV distortion and Fresnel mixing.
    """

    enabled: bool = False  # Disabled by default (P3 compatibility)
    intensity: float = 0.8  # Reflection intensity (0.0-1.0)
    fresnel_power: float = 5.0  # Fresnel falloff exponent
    wave_strength: float = 0.02  # Wave-based UV distortion strength
    shore_atten_width: float = 0.3  # Reduce reflections near land
    water_plane_height: float = 0.0  # Water plane height in world space

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError("intensity must be in [0, 1]")
        if self.fresnel_power < 0.0:
            raise ValueError("fresnel_power must be >= 0")
        if self.wave_strength < 0.0:
            raise ValueError("wave_strength must be >= 0")
        if self.shore_atten_width < 0.0:
            raise ValueError("shore_atten_width must be >= 0")


@dataclass
class WaterSettings:
    """Terrain water-mask settings for MapScene/TerrainRenderer."""

    enabled: bool = False
    auto_mask: bool = False
    mask_path: Optional[str] = None
    level: Optional[float] = None
    slope_threshold: float = 0.02

    def __post_init__(self) -> None:
        if self.slope_threshold < 0.0:
            raise ValueError("slope_threshold must be >= 0")


@dataclass
class CloudSettings:
    """Terrain cloud-shadow settings for MapScene/TerrainRenderer."""

    enabled: bool = False
    shadows_enabled: bool = False
    coverage: float = 0.5
    density: float = 0.5
    shadow_strength: float = 0.35
    quality: str = "medium"

    def __post_init__(self) -> None:
        for name in ("coverage", "density", "shadow_strength"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if str(self.quality) not in {"low", "medium", "high", "ultra"}:
            raise ValueError("quality must be one of: low, medium, high, ultra")


@dataclass
class BloomSettings:
    """M2: Bloom post-processing configuration.
    
    When enabled=False, bloom is disabled (identical output for backward compatibility).
    Bloom extracts bright pixels above threshold and applies Gaussian blur,
    then composites the result back onto the original image.
    """

    enabled: bool = False  # Disabled by default for backward compatibility
    threshold: float = 1.5  # Brightness threshold (1.5 = HDR only)
    softness: float = 0.5  # Threshold transition softness (0.0-1.0)
    intensity: float = 0.3  # Bloom intensity when compositing
    radius: float = 1.0  # Blur radius multiplier

    def __post_init__(self) -> None:
        if self.threshold < 0.0:
            raise ValueError("threshold must be >= 0")
        if not 0.0 <= self.softness <= 1.0:
            raise ValueError("softness must be in [0, 1]")
        if self.intensity < 0.0:
            raise ValueError("intensity must be >= 0")
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")


@dataclass
class ScreenSpaceSettings:
    """Screen-space effect settings passed to terrain/MapScene renders."""

    enabled: bool = False
    ssao_enabled: bool = False
    ssao_radius: float = 1.5
    ssao_intensity: float = 1.0
    ssgi_enabled: bool = False
    ssgi_intensity: float = 1.0
    ssr_enabled: bool = False
    ssr_intensity: float = 1.0
    taa_enabled: bool = False
    temporal_alpha: float = 0.1

    def __post_init__(self) -> None:
        for name in ("ssao_radius", "ssao_intensity", "ssgi_intensity", "ssr_intensity"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= float(self.temporal_alpha) <= 1.0:
            raise ValueError("temporal_alpha must be in [0, 1]")


@dataclass
class HeightAoSettings:
    """Heightfield ray-traced ambient occlusion configuration.
    
    Computes AO by ray-marching the heightfield in multiple directions.
    When enabled=False, AO is disabled (default for backward compatibility).
    """

    enabled: bool = False
    resolution_scale: float = 0.5  # Render at half resolution for performance
    directions: int = 6  # Number of horizon directions to sample
    steps: int = 16  # Steps per direction
    max_distance: float = 200.0  # Max ray distance in world units
    strength: float = 1.0  # AO intensity multiplier
    blur: bool = False  # Optional bilateral blur

    def __post_init__(self) -> None:
        if not 0.1 <= self.resolution_scale <= 1.0:
            raise ValueError("resolution_scale must be in [0.1, 1.0]")
        if not 1 <= self.directions <= 16:
            raise ValueError("directions must be in [1, 16]")
        if not 1 <= self.steps <= 64:
            raise ValueError("steps must be in [1, 64]")
        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError("strength must be in [0.0, 2.0]")


@dataclass
class SunVisibilitySettings:
    """Heightfield ray-traced sun visibility / soft shadows configuration.
    
    Computes sun visibility by ray-marching toward the sun direction.
    When enabled=False, sun visibility is disabled (default for backward compatibility).
    """

    enabled: bool = False
    mode: str = "hard"  # "hard" or "soft"
    resolution_scale: float = 0.5  # Render at half resolution for performance
    samples: int = 4  # Number of jittered samples for soft shadows
    steps: int = 24  # Steps per ray
    max_distance: float = 400.0  # Max ray distance in world units
    softness: float = 1.0  # Penumbra softness multiplier
    bias: float = 0.01  # Self-shadowing bias

    def __post_init__(self) -> None:
        valid_modes = {"hard", "soft"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode!r}")
        if not 0.1 <= self.resolution_scale <= 1.0:
            raise ValueError("resolution_scale must be in [0.1, 1.0]")
        if not 1 <= self.samples <= 16:
            raise ValueError("samples must be in [1, 16]")
        if not 1 <= self.steps <= 64:
            raise ValueError("steps must be in [1, 64]")
        if self.max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")
        if self.softness < 0.0:
            raise ValueError("softness must be >= 0")
        if self.bias < 0.0:
            raise ValueError("bias must be >= 0")


@dataclass
class ProbeSettings:
    """TV5: Irradiance probe configuration for terrain scenes."""

    enabled: bool = False
    grid_dims: Tuple[int, int] = (8, 8)
    origin: Optional[Tuple[float, float]] = None
    spacing: Optional[Tuple[float, float]] = None
    height_offset: float = 5.0
    ray_count: int = 64
    fallback_blend_distance: Optional[float] = None
    sky_color: Tuple[float, float, float] = (0.6, 0.75, 1.0)
    sky_intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.enabled:
            cols, rows = self.grid_dims
            if cols < 1 or rows < 1:
                raise ValueError("grid_dims must be >= (1, 1)")
            if cols * rows > 4096:
                raise ValueError("grid_dims product must be <= 4096 (probe count limit)")
            if self.ray_count < 1:
                raise ValueError("ray_count must be >= 1")
            if self.spacing is not None and (self.spacing[0] <= 0.0 or self.spacing[1] <= 0.0):
                raise ValueError("spacing must be > 0 when provided")
            if self.fallback_blend_distance is not None and self.fallback_blend_distance < 0.0:
                raise ValueError("fallback_blend_distance must be >= 0")
            if len(self.sky_color) != 3:
                raise ValueError("sky_color must be (R, G, B)")
            if self.sky_intensity < 0.0:
                raise ValueError("sky_intensity must be >= 0")


@dataclass
class ReflectionProbeSettings:
    """Local reflection probe configuration for terrain scenes."""

    enabled: bool = False
    grid_dims: Tuple[int, int] = (4, 4)
    origin: Optional[Tuple[float, float]] = None
    spacing: Optional[Tuple[float, float]] = None
    height_offset: float = 5.0
    resolution: int = 16
    ray_count: int = 64
    trace_steps: int = 192
    trace_refine_steps: int = 5
    fallback_blend_distance: Optional[float | Tuple[float, float]] = None
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.enabled:
            cols, rows = self.grid_dims
            if cols < 1 or rows < 1:
                raise ValueError("grid_dims must be >= (1, 1)")
            if cols * rows > 64:
                raise ValueError("reflection probe count limit is 64")
            if self.resolution < 4 or (self.resolution & (self.resolution - 1)) != 0:
                raise ValueError("resolution must be a power of two >= 4")
            if self.ray_count < 1:
                raise ValueError("ray_count must be >= 1")
            if self.trace_steps < 8:
                raise ValueError("trace_steps must be >= 8")
            if self.trace_refine_steps < 0:
                raise ValueError("trace_refine_steps must be >= 0")
            if self.spacing is not None and (self.spacing[0] <= 0.0 or self.spacing[1] <= 0.0):
                raise ValueError("spacing must be > 0 when provided")
            if self.fallback_blend_distance is not None:
                value = self.fallback_blend_distance
                if isinstance(value, tuple):
                    if value[0] < 0.0 or value[1] < 0.0:
                        raise ValueError("fallback_blend_distance tuple must be >= 0")
                elif value < 0.0:
                    raise ValueError("fallback_blend_distance must be >= 0")
            if not 0.0 <= self.strength <= 1.0:
                raise ValueError("strength must be in [0, 1]")


@dataclass
class DetailSettings:
    """P6 micro-detail configuration for close-range surface enhancement.

    When ``enabled`` is false, micro-detail is disabled. When enabled, the
    renderer adds triplanar detail normals and procedural albedo noise that fade
    with distance to prevent LOD popping and shimmer.

    Gradient-match fields:
    ``detail_normal_path`` points to an optional DEM-derived detail-normal
    texture, ``detail_sigma_px`` records the Gaussian sigma used to generate it,
    and ``detail_strength`` controls how strongly it blends with the procedural
    detail path.
    """

    enabled: bool = False  # Disabled by default (P5 compatibility)
    detail_scale: float = 2.0  # 2 meter repeat interval
    normal_strength: float = 0.3  # Detail normal blending strength
    albedo_noise: float = 0.1  # ±10% brightness variation
    fade_start: float = 50.0  # Start fading at 50 units
    fade_end: float = 200.0  # Fully faded at 200 units
    # P6 Gradient Match: DEM-derived detail normal map
    detail_normal_path: Optional[str] = None  # Path to detail normal texture
    detail_sigma_px: float = 3.0  # Gaussian sigma used to generate detail normals
    detail_strength: float = 0.0  # DEM-derived detail normal strength (0=off)

    def __post_init__(self) -> None:
        if self.detail_scale <= 0.0:
            raise ValueError("detail_scale must be > 0")
        if not 0.0 <= self.normal_strength <= 1.0:
            raise ValueError("normal_strength must be in [0, 1]")
        if not 0.0 <= self.albedo_noise <= 0.5:
            raise ValueError("albedo_noise must be in [0, 0.5]")
        if self.fade_start < 0.0:
            raise ValueError("fade_start must be >= 0")
        if self.fade_end <= self.fade_start:
            raise ValueError("fade_end must be > fade_start")
        if self.detail_sigma_px <= 0.0:
            raise ValueError("detail_sigma_px must be > 0")
        if not 0.0 <= self.detail_strength <= 1.0:
            raise ValueError("detail_strength must be in [0, 1]")


@dataclass
class MaterialNoiseSettings:
    """TV4: Bounded procedural variation controls for terrain material layers.

    Coordinates are evaluated in normalized terrain UV space so the controls remain
    stable across DEMs with very different real-world extents.

    ``macro_scale`` controls low-frequency breakup, ``detail_scale`` controls
    higher-frequency breakup, and ``octaves`` bounds FBM/ridged FBM cost.
    Per-layer amplitudes default to zero, which preserves the pre-TV4 material output.
    """

    macro_scale: float = 3.5
    detail_scale: float = 18.0
    octaves: int = 4
    snow_macro_amplitude: float = 0.0
    snow_detail_amplitude: float = 0.0
    rock_macro_amplitude: float = 0.0
    rock_detail_amplitude: float = 0.0
    wetness_macro_amplitude: float = 0.0
    wetness_detail_amplitude: float = 0.0

    def __post_init__(self) -> None:
        if self.macro_scale <= 0.0:
            raise ValueError("macro_scale must be > 0")
        if self.detail_scale <= 0.0:
            raise ValueError("detail_scale must be > 0")
        if not 1 <= int(self.octaves) <= 8:
            raise ValueError("octaves must be in [1, 8]")
        self.octaves = int(self.octaves)

        for name, value in [
            ("snow_macro_amplitude", self.snow_macro_amplitude),
            ("snow_detail_amplitude", self.snow_detail_amplitude),
            ("rock_macro_amplitude", self.rock_macro_amplitude),
            ("rock_detail_amplitude", self.rock_detail_amplitude),
            ("wetness_macro_amplitude", self.wetness_macro_amplitude),
            ("wetness_detail_amplitude", self.wetness_detail_amplitude),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class MaterialLayerSettings:
    """M4: Terrain material layering configuration.

    Provides slope/aspect/altitude-driven material blending for realistic terrain:
    - Snow: deposits on high-altitude, low-slope areas (south-facing receives less)
    - Rock: exposed on steep slopes (>45°)
    - Wetness: darkening in concave areas (placeholder: based on slope curvature)

    TV4 extends this with bounded procedural variation controls. Those controls live
    under ``variation`` and default to zero amplitudes so the existing material
    layering output remains unchanged until explicitly enabled.

    ``normal_path``, ``roughness_path``, and ``mask_path`` describe optional
    per-texel material maps for the terrain shader/VT path. They are inert when
    unset and serialize with the rest of the render parameters.
    """

    normal_path: Optional[str] = None
    roughness_path: Optional[str] = None
    mask_path: Optional[str] = None

    # Snow layer settings
    snow_enabled: bool = False
    snow_altitude_min: float = 2000.0  # Minimum altitude for snow (world units)
    snow_altitude_blend: float = 500.0  # Altitude blend range
    snow_slope_max: float = 45.0  # Maximum slope angle (degrees) for snow
    snow_slope_blend: float = 15.0  # Slope blend range (degrees)
    snow_aspect_influence: float = 0.3  # 0=no aspect effect, 1=full (south-facing less snow)
    snow_color: Tuple[float, float, float] = (0.95, 0.95, 0.98)  # Snow albedo
    snow_roughness: float = 0.4  # Snow surface roughness
    snow_subsurface_strength: float = 0.0  # TV10: terrain SSS response for snow/ice-like layers
    snow_subsurface_tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Rock layer settings
    rock_enabled: bool = False
    rock_slope_min: float = 45.0  # Minimum slope angle (degrees) for rock exposure
    rock_slope_blend: float = 10.0  # Slope blend range (degrees)
    rock_color: Tuple[float, float, float] = (0.35, 0.32, 0.28)  # Rock albedo
    rock_roughness: float = 0.8  # Rock surface roughness
    rock_subsurface_strength: float = 0.0
    rock_subsurface_tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Wetness layer settings (darkening in concave areas)
    wetness_enabled: bool = False
    wetness_strength: float = 0.3  # Darkening strength (0-1)
    wetness_slope_influence: float = 0.5  # How much slope affects wetness
    wetness_subsurface_strength: float = 0.0
    wetness_subsurface_tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # TV4: Procedural variation controls shared across snow/rock/wetness.
    variation: MaterialNoiseSettings = field(default_factory=MaterialNoiseSettings)

    def __post_init__(self) -> None:
        if self.snow_altitude_blend <= 0.0:
            raise ValueError("snow_altitude_blend must be > 0")
        if not 0.0 <= self.snow_slope_max <= 90.0:
            raise ValueError("snow_slope_max must be in [0, 90]")
        if self.snow_slope_blend <= 0.0:
            raise ValueError("snow_slope_blend must be > 0")
        if not 0.0 <= self.snow_aspect_influence <= 1.0:
            raise ValueError("snow_aspect_influence must be in [0, 1]")
        if len(self.snow_color) != 3:
            raise ValueError("snow_color must be (R, G, B)")
        if not 0.0 <= self.snow_roughness <= 1.0:
            raise ValueError("snow_roughness must be in [0, 1]")
        if not 0.0 <= self.snow_subsurface_strength <= 1.0:
            raise ValueError("snow_subsurface_strength must be in [0, 1]")
        if len(self.snow_subsurface_tint) != 3:
            raise ValueError("snow_subsurface_tint must be (R, G, B)")
        for component in self.snow_subsurface_tint:
            if not 0.0 <= component <= 1.0:
                raise ValueError("snow_subsurface_tint components must be in [0, 1]")
        
        if not 0.0 <= self.rock_slope_min <= 90.0:
            raise ValueError("rock_slope_min must be in [0, 90]")
        if self.rock_slope_blend <= 0.0:
            raise ValueError("rock_slope_blend must be > 0")
        if len(self.rock_color) != 3:
            raise ValueError("rock_color must be (R, G, B)")
        if not 0.0 <= self.rock_roughness <= 1.0:
            raise ValueError("rock_roughness must be in [0, 1]")
        if not 0.0 <= self.rock_subsurface_strength <= 1.0:
            raise ValueError("rock_subsurface_strength must be in [0, 1]")
        if len(self.rock_subsurface_tint) != 3:
            raise ValueError("rock_subsurface_tint must be (R, G, B)")
        for component in self.rock_subsurface_tint:
            if not 0.0 <= component <= 1.0:
                raise ValueError("rock_subsurface_tint components must be in [0, 1]")
        
        if not 0.0 <= self.wetness_strength <= 1.0:
            raise ValueError("wetness_strength must be in [0, 1]")
        if not 0.0 <= self.wetness_slope_influence <= 1.0:
            raise ValueError("wetness_slope_influence must be in [0, 1]")
        if not 0.0 <= self.wetness_subsurface_strength <= 1.0:
            raise ValueError("wetness_subsurface_strength must be in [0, 1]")
        if len(self.wetness_subsurface_tint) != 3:
            raise ValueError("wetness_subsurface_tint must be (R, G, B)")
        for component in self.wetness_subsurface_tint:
            if not 0.0 <= component <= 1.0:
                raise ValueError("wetness_subsurface_tint components must be in [0, 1]")
        if not isinstance(self.variation, MaterialNoiseSettings):
            raise ValueError("variation must be a MaterialNoiseSettings instance")
        for name, value in [
            ("normal_path", self.normal_path),
            ("roughness_path", self.roughness_path),
            ("mask_path", self.mask_path),
        ]:
            if value is not None and not str(value):
                raise ValueError(f"{name} must be a non-empty path when provided")


@dataclass
class VectorOverlaySettings:
    """M5: Vector overlay configuration for depth-correct rendering and halos.
    
    Controls how vector overlays (lines, polygons) interact with terrain:
    - depth_test: When True, vectors are occluded by terrain ridges
    - halo: Adds outline/shadow for improved readability over terrain
    
    When depth_test=False (default), output is identical to baseline.
    """
    
    # Depth testing
    depth_test: bool = False  # When True, vectors hidden behind terrain
    depth_bias: float = 0.001  # Depth offset to prevent z-fighting (smaller = closer)
    depth_bias_slope: float = 1.0  # Slope-scaled bias for grazing angles
    
    # Halo/outline for readability
    halo_enabled: bool = False
    halo_width: float = 2.0  # Halo width in pixels
    halo_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5)  # RGBA
    halo_blur: float = 1.0  # Blur/softness of halo edge
    
    # Contour rendering (ink-like effect)
    contour_enabled: bool = False
    contour_width: float = 1.0  # Contour line width in pixels
    contour_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.8)

    def __post_init__(self) -> None:
        if self.depth_bias < 0.0:
            raise ValueError("depth_bias must be >= 0")
        if self.depth_bias_slope < 0.0:
            raise ValueError("depth_bias_slope must be >= 0")
        if self.halo_width < 0.0:
            raise ValueError("halo_width must be >= 0")
        if len(self.halo_color) != 4:
            raise ValueError("halo_color must be (R, G, B, A)")
        if self.halo_blur < 0.0:
            raise ValueError("halo_blur must be >= 0")
        if self.contour_width < 0.0:
            raise ValueError("contour_width must be >= 0")
        if len(self.contour_color) != 4:
            raise ValueError("contour_color must be (R, G, B, A)")


@dataclass
class TonemapSettings:
    """M6: Tonemap configuration for HDR to SDR conversion.
    
    Controls tone mapping operator selection, 3D LUT application, and white balance.
    
    Operators:
    - 'reinhard': Simple Reinhard (default)
    - 'reinhard_extended': Extended Reinhard with white point
    - 'aces': ACES filmic (cinematic look)
    - 'uncharted2': Uncharted 2 filmic
    - 'exposure': Simple exposure mapping
    
    White balance uses temperature (Kelvin) and tint (green-magenta).
    """
    
    # Tonemap operator selection
    operator: str = "aces"  # reinhard, reinhard_extended, aces, uncharted2, exposure
    
    # White point for extended operators
    white_point: float = 4.0
    
    # 3D LUT support (cube format)
    lut_enabled: bool = False
    lut_path: Optional[str] = None  # Path to .cube LUT file
    lut_strength: float = 1.0  # Blend strength 0-1
    
    # White balance (temperature/tint)
    white_balance_enabled: bool = False
    temperature: float = 6500.0  # Color temperature in Kelvin (2000-12000)
    tint: float = 0.0  # Green-magenta tint (-1.0 to 1.0)

    def __post_init__(self) -> None:
        valid_operators = {"reinhard", "reinhard_extended", "aces", "uncharted2", "exposure"}
        if self.operator not in valid_operators:
            raise ValueError(f"operator must be one of {valid_operators}, got '{self.operator}'")
        if self.white_point <= 0.0:
            raise ValueError("white_point must be > 0")
        if self.lut_strength < 0.0 or self.lut_strength > 1.0:
            raise ValueError("lut_strength must be in range [0, 1]")
        if self.temperature < 2000.0 or self.temperature > 12000.0:
            raise ValueError("temperature must be in range [2000, 12000] Kelvin")
        if self.tint < -1.0 or self.tint > 1.0:
            raise ValueError("tint must be in range [-1, 1]")


@dataclass
class AovSettings:
    """M1: AOV (Arbitrary Output Variable) export configuration.
    
    Controls which auxiliary render outputs are captured alongside the beauty pass.
    When enabled=False, no AOVs are exported (default for backward compatibility).
    
    Supported AOVs for M1:
    - albedo: Base color before lighting (Rgba8Unorm)
    - normal: World-space normals remapped to [0,1] (Rgba8Unorm)
    - depth: Linear depth normalized to [near, far] (R32Float or Rgba8Unorm)
    
    Future milestones will add: roughness, metallic, AO, sun_vis, mask/ID
    """
    
    enabled: bool = False  # Disabled by default (backward compatibility)
    albedo: bool = True    # Export albedo AOV when enabled
    normal: bool = True    # Export world-space normal AOV when enabled
    depth: bool = True     # Export linear depth AOV when enabled
    # VERITAS: per-pixel VT source-id map (uint32; 0 == SOURCE_ID_NONE).
    # Requires msaa_samples=1 and render_scale=1.0. Off by default.
    source_id: bool = False
    output_dir: Optional[str] = None  # Directory for AOV output (None = same as beauty)
    format: str = "png"    # Output format: "png" or "exr" (M2)
    
    def __post_init__(self) -> None:
        valid_formats = {"png", "exr", "raw"}
        if self.format not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}, got '{self.format}'")
    
    @property
    def any_enabled(self) -> bool:
        """Returns True if AOV export is enabled and at least one AOV is selected."""
        return self.enabled and (self.albedo or self.normal or self.depth)


@dataclass
class DofSettings:
    """M3: Depth of Field configuration with tilt-shift support.
    
    Controls camera depth of field blur effect. When enabled=False, DoF is disabled
    (default for backward compatibility).
    
    Standard DoF parameters:
    - f_stop: Aperture f-number (e.g., 2.8, 5.6, 11). Lower = more blur.
    - focus_distance: Distance to focus plane in world units.
    - focal_length: Camera focal length in mm (default 50mm).
    
    Tilt-shift parameters (Scheimpflug effect):
    - tilt_pitch: Tilt around horizontal axis in degrees. Creates diagonal focus plane.
    - tilt_yaw: Tilt around vertical axis in degrees.
    
    Quality settings:
    - method: "gather" (quality) or "separable" (performance)
    - quality: "low", "medium", "high", "ultra"
    """
    
    enabled: bool = False  # Disabled by default (backward compatibility)
    f_stop: float = 5.6    # Aperture f-number (2.8 = shallow DoF, 16 = deep DoF)
    focus_distance: float = 100.0  # Focus distance in world units
    focal_length: float = 50.0     # Focal length in mm
    
    # M3: Tilt-shift parameters (Scheimpflug effect)
    tilt_pitch: float = 0.0  # Tilt around horizontal axis (degrees)
    tilt_yaw: float = 0.0    # Tilt around vertical axis (degrees)
    
    # Quality settings
    method: str = "gather"   # "gather" or "separable"
    quality: str = "medium"  # "low", "medium", "high", "ultra"
    
    # Debug/visualization
    show_coc: bool = False   # Overlay circle-of-confusion visualization
    debug_mode: int = 0      # 0=normal, 1=CoC grayscale, 2=field zones
    
    def __post_init__(self) -> None:
        if self.f_stop <= 0:
            raise ValueError("f_stop must be > 0")
        if self.focus_distance <= 0:
            raise ValueError("focus_distance must be > 0")
        if self.focal_length <= 0:
            raise ValueError("focal_length must be > 0")
        
        valid_methods = {"gather", "separable"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{self.method}'")
        
        valid_qualities = {"low", "medium", "high", "ultra"}
        if self.quality not in valid_qualities:
            raise ValueError(f"quality must be one of {valid_qualities}, got '{self.quality}'")
    
    @property
    def aperture(self) -> float:
        """Convert f-stop to aperture value (1/f_stop)."""
        return 1.0 / self.f_stop
    
    @property
    def tilt_pitch_rad(self) -> float:
        """Tilt pitch in radians."""
        import math
        return math.radians(self.tilt_pitch)
    
    @property
    def tilt_yaw_rad(self) -> float:
        """Tilt yaw in radians."""
        import math
        return math.radians(self.tilt_yaw)
    
    @property
    def has_tilt(self) -> bool:
        """Returns True if tilt-shift is active."""
        return abs(self.tilt_pitch) > 0.01 or abs(self.tilt_yaw) > 0.01


@dataclass
class MotionBlurSettings:
    """M4: Motion blur configuration for camera shutter accumulation.
    
    Simulates motion blur by accumulating multiple sub-frames across a shutter
    interval. Camera position/rotation is interpolated between frames.
    
    Note: Object motion blur is NOT supported in this implementation.
    Only camera motion blur via shutter accumulation is available.
    
    Shutter timing:
    - shutter_open: When shutter opens relative to frame (0.0 = start of frame)
    - shutter_close: When shutter closes relative to frame (1.0 = end of frame)
    - For 180° shutter: shutter_open=0.0, shutter_close=0.5
    - For 360° shutter: shutter_open=0.0, shutter_close=1.0
    
    Camera interpolation:
    - cam_phi_delta: Change in camera azimuth (degrees) over shutter interval
    - cam_theta_delta: Change in camera elevation (degrees) over shutter interval
    - cam_radius_delta: Change in camera distance over shutter interval
    """
    
    enabled: bool = False  # Disabled by default (backward compatibility)
    samples: int = 8       # Number of sub-frames to accumulate (1-64)
    shutter_open: float = 0.0   # Shutter open time (0.0 = frame start)
    shutter_close: float = 0.5  # Shutter close time (1.0 = frame end)
    
    # Camera motion deltas over shutter interval
    cam_phi_delta: float = 0.0      # Azimuth change (degrees)
    cam_theta_delta: float = 0.0    # Elevation change (degrees)
    cam_radius_delta: float = 0.0   # Distance change (world units)
    
    # Determinism
    seed: Optional[int] = None  # Seed for deterministic sampling (None = default)
    
    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be >= 1")
        if self.samples > 64:
            raise ValueError("samples must be <= 64 (performance limit)")
        if self.shutter_open < 0.0 or self.shutter_open > 1.0:
            raise ValueError("shutter_open must be in [0.0, 1.0]")
        if self.shutter_close < 0.0 or self.shutter_close > 1.0:
            raise ValueError("shutter_close must be in [0.0, 1.0]")
        if self.shutter_close <= self.shutter_open:
            raise ValueError("shutter_close must be > shutter_open")
    
    @property
    def shutter_angle(self) -> float:
        """Shutter angle in degrees (360° = full frame exposure)."""
        return (self.shutter_close - self.shutter_open) * 360.0
    
    @property
    def has_camera_motion(self) -> bool:
        """Returns True if any camera motion is configured."""
        return (abs(self.cam_phi_delta) > 0.001 or 
                abs(self.cam_theta_delta) > 0.001 or 
                abs(self.cam_radius_delta) > 0.001)


@dataclass
class LensEffectsSettings:
    """M5: Lens and sensor effects for post-processing.
    
    Simulates optical imperfections and sensor characteristics:
    - Barrel/pincushion distortion
    - Chromatic aberration (color fringing)
    - Vignetting (corner darkening)
    
    Applied after tonemapping, before final output.
    """
    
    enabled: bool = False  # Disabled by default (backward compatibility)
    
    # Lens distortion (barrel/pincushion)
    # Positive = barrel, Negative = pincushion, 0 = none
    distortion: float = 0.0
    
    # Chromatic aberration (lateral color fringing)
    # Controls RGB channel separation at edges
    chromatic_aberration: float = 0.0
    
    # Vignette (corner darkening)
    vignette_strength: float = 0.0   # 0 = none, 1 = strong
    vignette_radius: float = 0.7     # Start radius (0-1, center to corner)
    vignette_softness: float = 0.3   # Falloff softness
    
    def __post_init__(self) -> None:
        if self.vignette_strength < 0.0:
            raise ValueError("vignette_strength must be >= 0")
        if self.vignette_radius < 0.0 or self.vignette_radius > 1.0:
            raise ValueError("vignette_radius must be in [0.0, 1.0]")
        if self.vignette_softness < 0.0:
            raise ValueError("vignette_softness must be >= 0")
    
    @property
    def has_distortion(self) -> bool:
        """Returns True if lens distortion is active."""
        return abs(self.distortion) > 0.001
    
    @property
    def has_chromatic_aberration(self) -> bool:
        """Returns True if chromatic aberration is active."""
        return abs(self.chromatic_aberration) > 0.001
    
    @property
    def has_vignette(self) -> bool:
        """Returns True if vignetting is active."""
        return self.vignette_strength > 0.001
    
    @property
    def has_any_effect(self) -> bool:
        """Returns True if any lens effect is active."""
        return self.has_distortion or self.has_chromatic_aberration or self.has_vignette


@dataclass
class DenoiseSettings:
    """M5: Denoising configuration for noise reduction.
    
    Supports CPU-based A-trous wavelet denoising for:
    - Final rendered images
    - AOV buffers (AO, sun visibility, etc.)

    Methods:
    - 'atrous': A-trous wavelet transform (edge-preserving)
    - 'oidn': Intel Open Image Denoise (runtime optional)
    - 'none': No denoising
    """
    
    enabled: bool = False  # Disabled by default
    method: str = "atrous"  # 'atrous', 'oidn', 'none'
    iterations: int = 3     # Number of filter passes (1-10)
    
    # A-trous parameters
    sigma_color: float = 0.1   # Color similarity weight
    sigma_normal: float = 0.1  # Normal similarity weight (if guidance available)
    sigma_depth: float = 0.1   # Depth similarity weight (if guidance available)
    
    # Edge preservation
    edge_stopping: float = 1.0  # Edge-stopping strength (0 = none, 1 = strong)
    
    def __post_init__(self) -> None:
        valid_methods = ("atrous", "oidn", "none")
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.iterations > 10:
            raise ValueError("iterations must be <= 10 (quality/performance limit)")
        if self.sigma_color < 0.0:
            raise ValueError("sigma_color must be >= 0")
        if self.sigma_normal < 0.0:
            raise ValueError("sigma_normal must be >= 0")
        if self.sigma_depth < 0.0:
            raise ValueError("sigma_depth must be >= 0")
        if self.edge_stopping < 0.0:
            raise ValueError("edge_stopping must be >= 0")
    
    @property
    def uses_guidance(self) -> bool:
        """Returns True if denoiser uses normal/depth guidance."""
        return (self.sigma_normal > 0.001 or self.sigma_depth > 0.001) and self.method == "atrous"


@dataclass
class OfflineQualitySettings:
    """Offline accumulation and adaptive sampling policy."""

    enabled: bool = False  # Explicit opt-in required by render_offline()
    adaptive: bool = False
    target_variance: float = 0.001
    max_samples: int = 64
    min_samples: int = 4
    batch_size: int = 4
    tile_size: int = 16
    convergence_ratio: float = 0.95

    def __post_init__(self) -> None:
        if self.target_variance < 0.0:
            raise ValueError("target_variance must be >= 0")
        if self.max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if self.min_samples > self.max_samples:
            raise ValueError("min_samples must be <= max_samples")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        if not 0.0 <= self.convergence_ratio <= 1.0:
            raise ValueError("convergence_ratio must be in [0, 1]")


@dataclass
class DensityVolumeSettings:
    """TV6: Bounded heterogeneous density volume for terrain viewer volumetrics.

    Coordinates are expressed in terrain-viewer world units:
    - ``center.x`` / ``center.z`` live in the terrain XY plane
    - ``center.y`` lives in the exaggerated terrain height space used by the viewer

    The preset controls how the 3D density texture is generated:
    - ``valley_fog`` hugs terrain and fills low areas inside the volume bounds
    - ``plume`` creates a rising, wind-tilted column suitable for smoke or ash
    - ``localized_haze`` creates a soft ellipsoidal atmospheric pocket
    """

    preset: str = "valley_fog"
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (128.0, 64.0, 128.0)
    resolution: Tuple[int, int, int] = (64, 32, 64)
    density_scale: float = 1.0
    edge_softness: float = 0.25
    noise_strength: float = 0.35
    floor_offset: float = 0.0
    ceiling: float = 0.4
    plume_spread: float = 0.35
    wind: Tuple[float, float, float] = (0.25, 1.0, 0.0)
    seed: int = 0

    VALID_PRESETS = {"valley_fog", "plume", "localized_haze"}
    MAX_ACTIVE_VOLUMES = 4
    MAX_RESOLUTION_AXIS = 96

    def __post_init__(self) -> None:
        if self.preset not in self.VALID_PRESETS:
            raise ValueError(f"preset must be one of {sorted(self.VALID_PRESETS)}")
        if len(self.center) != 3:
            raise ValueError("center must be (x, y, z)")
        if len(self.size) != 3:
            raise ValueError("size must be (x, y, z)")
        if len(self.resolution) != 3:
            raise ValueError("resolution must be (x, y, z)")
        if len(self.wind) != 3:
            raise ValueError("wind must be (x, y, z)")

        for axis, value in zip(("x", "y", "z"), self.size):
            if value <= 0.0:
                raise ValueError(f"size.{axis} must be > 0")

        for axis, value in zip(("x", "y", "z"), self.resolution):
            if not 8 <= int(value) <= self.MAX_RESOLUTION_AXIS:
                raise ValueError(
                    f"resolution.{axis} must be in [8, {self.MAX_RESOLUTION_AXIS}]"
                )

        if self.density_scale < 0.0:
            raise ValueError("density_scale must be >= 0")
        if not 0.0 <= self.edge_softness <= 1.0:
            raise ValueError("edge_softness must be in [0.0, 1.0]")
        if not 0.0 <= self.noise_strength <= 1.0:
            raise ValueError("noise_strength must be in [0.0, 1.0]")
        if not 0.0 <= self.ceiling <= 1.0:
            raise ValueError("ceiling must be in [0.0, 1.0]")
        if not 0.05 <= self.plume_spread <= 2.0:
            raise ValueError("plume_spread must be in [0.05, 2.0]")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


def valley_fog_volume(
    *,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    resolution: Tuple[int, int, int] = (64, 32, 64),
    density_scale: float = 1.0,
    edge_softness: float = 0.25,
    noise_strength: float = 0.35,
    floor_offset: float = 4.0,
    ceiling: float = 0.42,
    seed: int = 0,
) -> DensityVolumeSettings:
    return DensityVolumeSettings(
        preset="valley_fog",
        center=center,
        size=size,
        resolution=resolution,
        density_scale=density_scale,
        edge_softness=edge_softness,
        noise_strength=noise_strength,
        floor_offset=floor_offset,
        ceiling=ceiling,
        seed=seed,
    )


def plume_volume(
    *,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    resolution: Tuple[int, int, int] = (48, 80, 48),
    density_scale: float = 1.0,
    edge_softness: float = 0.18,
    noise_strength: float = 0.5,
    plume_spread: float = 0.45,
    wind: Tuple[float, float, float] = (0.35, 1.0, -0.1),
    seed: int = 0,
) -> DensityVolumeSettings:
    return DensityVolumeSettings(
        preset="plume",
        center=center,
        size=size,
        resolution=resolution,
        density_scale=density_scale,
        edge_softness=edge_softness,
        noise_strength=noise_strength,
        plume_spread=plume_spread,
        wind=wind,
        seed=seed,
    )


def localized_haze_volume(
    *,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    resolution: Tuple[int, int, int] = (48, 32, 48),
    density_scale: float = 0.8,
    edge_softness: float = 0.35,
    noise_strength: float = 0.25,
    ceiling: float = 0.65,
    seed: int = 0,
) -> DensityVolumeSettings:
    return DensityVolumeSettings(
        preset="localized_haze",
        center=center,
        size=size,
        resolution=resolution,
        density_scale=density_scale,
        edge_softness=edge_softness,
        noise_strength=noise_strength,
        ceiling=ceiling,
        seed=seed,
    )


@dataclass
class VolumetricsSettings:
    """M6: Volumetric fog and light shafts configuration.
    
    Simulates atmospheric scattering effects:
    - Volumetric fog with density falloff
    - Light shafts (god rays) from sun
    - Shadow-aware volumetric lighting
    
    Applied after depth, before tonemapping.

    TV6 extends this with optional bounded 3D density volumes. When
    ``density_volumes`` is populated the viewer samples those localized density
    fields in the same volumetric pass used by legacy fog modes.
    """
    
    enabled: bool = False  # Disabled by default
    mode: str = "uniform"  # 'uniform', 'height', 'exponential'
    density: float = 0.01  # Global fog density
    
    # Height-based fog parameters
    height_falloff: float = 0.1   # Density falloff with altitude
    base_height: float = 0.0      # Fog base height in world units
    
    # Scattering parameters
    scattering: float = 0.5       # In-scatter amount [0-1]
    absorption: float = 0.1       # Light absorption [0-1]
    phase_g: float = 0.0          # Henyey-Greenstein phase (-1=back, 0=iso, 1=forward)
    
    # Light shafts
    light_shafts: bool = False    # Enable god rays
    shaft_intensity: float = 1.0  # Light shaft brightness
    shaft_samples: int = 32       # Ray march samples [8-128]
    
    # Performance
    use_shadows: bool = True      # Use shadow map for volumetrics
    half_res: bool = False        # Render at half resolution
    density_volumes: Tuple[DensityVolumeSettings, ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        valid_modes = ("uniform", "height", "exponential")
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        if self.density < 0.0:
            raise ValueError("density must be >= 0")
        if self.scattering < 0.0 or self.scattering > 1.0:
            raise ValueError("scattering must be in [0.0, 1.0]")
        if self.absorption < 0.0 or self.absorption > 1.0:
            raise ValueError("absorption must be in [0.0, 1.0]")
        if self.phase_g < -1.0 or self.phase_g > 1.0:
            raise ValueError("phase_g must be in [-1.0, 1.0]")
        if self.shaft_samples < 8 or self.shaft_samples > 128:
            raise ValueError("shaft_samples must be in [8, 128]")
        if len(self.density_volumes) > DensityVolumeSettings.MAX_ACTIVE_VOLUMES:
            raise ValueError(
                f"density_volumes supports at most {DensityVolumeSettings.MAX_ACTIVE_VOLUMES} active entries"
            )
    
    @property
    def has_light_shafts(self) -> bool:
        """Returns True if light shafts are enabled."""
        return self.light_shafts and self.shaft_intensity > 0.001

    @property
    def uses_density_volumes(self) -> bool:
        """Returns True when localized 3D density volumes are configured."""
        return len(self.density_volumes) > 0

    def to_viewer_dict(self) -> dict:
        """Convert to the terrain-viewer IPC payload shape."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "density": self.density,
            "height_falloff": self.height_falloff,
            "scattering": self.scattering,
            "absorption": self.absorption,
            "light_shafts": self.light_shafts,
            "shaft_intensity": self.shaft_intensity,
            "steps": self.shaft_samples,
            "half_res": self.half_res,
            "density_volumes": [asdict(volume) for volume in self.density_volumes],
        }


@dataclass
class SkySettings:
    """Procedural or AETHER spectral sky and aerial-perspective configuration.
    
    Renders procedural sky with:
    - Sun disc rendering
    - Hosek-Wilkie RGB coefficient-table sky model
    - Optional Preetham or legacy approximate gradients for migration
    - AETHER shipped spectral LUTs with explicit ozone and Mie anisotropy
    - Aerial perspective for distant terrain using the same sky tint path
    
    Applied as a rendered sky background in mesh-mode terrain views and sampled
    by the terrain atmosphere path for aerial perspective and fog inscatter tint.
    """
    
    enabled: bool = False  # Disabled by default
    model: str = "hosek-wilkie"  # includes the spectral "aether" model
    
    # Sky model parameters
    # Private sentinel defaults let the handle distinguish omitted values from
    # explicit conflicting values while leaving normalized instances as floats.
    turbidity: float = field(default=_UNSET)       # type: ignore[assignment]
    ground_albedo: float = field(default=_UNSET)   # type: ignore[assignment]
    ozone_du: float = field(default=_UNSET)        # type: ignore[assignment]
    mie_g: float = field(default=_UNSET)           # type: ignore[assignment]
    lut_handle: AtmosphereLutHandle | None = None
    
    # Sun parameters (uses global sun direction if not overridden)
    sun_intensity: float = 1.0    # Sun disc brightness multiplier
    sun_size: float = 1.0         # Sun disc angular size multiplier
    
    # Aerial perspective
    aerial_perspective: bool = True  # Apply atmospheric scattering to terrain
    aerial_density: float = 1.0      # Aerial perspective strength [0.0-10.0]
    
    # Exposure
    sky_exposure: float = 1.0     # Sky brightness adjustment
    
    def __post_init__(self) -> None:
        valid_models = ("hosek-wilkie", "preetham", "approximate", "aether")
        if self.model not in valid_models:
            raise ValueError(f"model must be one of {valid_models}")

        physical_defaults = {
            "turbidity": 2.0,
            "ground_albedo": 0.3,
            "ozone_du": 300.0,
            "mie_g": 0.8,
        }
        if self.lut_handle is None:
            for name, default in physical_defaults.items():
                value = getattr(self, name)
                setattr(self, name, float(default if value is _UNSET else value))
        else:
            if self.model != "aether":
                raise ValueError("lut_handle requires model='aether'")
            try:
                handle_values = {
                    name: _as_f32(getattr(self.lut_handle, name))
                    for name in physical_defaults
                }
            except (AttributeError, OverflowError, TypeError, ValueError) as error:
                raise TypeError(
                    "lut_handle must be an AtmosphereLutHandle returned by "
                    "forge3d.atmosphere_bake_luts()"
                ) from error
            for name, expected in handle_values.items():
                supplied = getattr(self, name)
                if supplied is not _UNSET and _as_f32(supplied) != expected:
                    raise ValueError(
                        f"{name}={supplied} does not match lut_handle.{name}={expected}"
                    )
                setattr(self, name, expected)

        if not math.isfinite(self.turbidity) or not 1.0 <= self.turbidity <= 10.0:
            raise ValueError("turbidity must be in [1.0, 10.0] and finite")
        if not math.isfinite(self.ground_albedo) or not 0.0 <= self.ground_albedo <= 1.0:
            raise ValueError("ground_albedo must be in [0.0, 1.0] and finite")
        if not math.isfinite(self.ozone_du) or self.ozone_du < 0.0 or self.ozone_du > 600.0:
            raise ValueError("ozone_du must be finite and in [0.0, 600.0]")
        if not math.isfinite(self.mie_g) or not 0.0 <= self.mie_g <= 0.99:
            raise ValueError("mie_g must be finite and in [0.0, 0.99]")
        if not math.isfinite(self.sun_intensity) or self.sun_intensity < 0.0:
            raise ValueError("sun_intensity must be finite and >= 0")
        if not math.isfinite(self.sun_size) or self.sun_size < 0.0:
            raise ValueError("sun_size must be finite and >= 0")
        if (
            not math.isfinite(self.aerial_density)
            or not 0.0 <= self.aerial_density <= 10.0
        ):
            raise ValueError("aerial_density must be finite and in [0.0, 10.0]")
        if not math.isfinite(self.sky_exposure) or self.sky_exposure < 0.0:
            raise ValueError("sky_exposure must be finite and >= 0")
    
    @property
    def has_aerial_perspective(self) -> bool:
        """Returns True if aerial perspective is active."""
        return self.aerial_perspective and self.aerial_density > 0.001


@dataclass
class VTLayerFamily:
    """Describes one paged terrain material family.

    The native runtime supports ``albedo``, ``normal``, and ``mask`` terrain
    material families in the same residency pass. Albedo feeds material color;
    normal feeds terrain normal perturbation; mask gates per-texel material-map
    effects.
    """
    family: str                        # "albedo" | "normal" | "mask"
    virtual_size_px: Tuple[int, int] = (4096, 4096)  # family-wide invariant
    tile_size: int = 248               # content pixels per tile edge
    tile_border: int = 4               # gutter pixels per tile edge (slot_size = 256)
    fallback: Optional[Tuple[float, ...]] = None  # family-safe last-resort value

    def __post_init__(self) -> None:
        valid_families = ("albedo", "normal", "mask")
        if self.family not in valid_families:
            raise ValueError(f"family must be one of {valid_families}")
        if isinstance(self.tile_size, bool) or not isinstance(self.tile_size, Integral):
            raise ValueError("tile_size must be an integer")
        if self.tile_size < 16:
            raise ValueError("tile_size must be >= 16")
        if isinstance(self.tile_border, bool) or not isinstance(self.tile_border, Integral):
            raise ValueError("tile_border must be an integer")
        if self.tile_border < 0:
            raise ValueError("tile_border must be >= 0")
        if (
            not isinstance(self.virtual_size_px, Sequence)
            or isinstance(self.virtual_size_px, (str, bytes))
            or len(self.virtual_size_px) != 2
        ):
            raise ValueError("virtual_size_px must contain exactly two dimensions")
        w, h = self.virtual_size_px
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (w, h)):
            raise ValueError("virtual_size_px dimensions must be integers")
        if w < self.tile_size or h < self.tile_size:
            raise ValueError("virtual_size_px must be >= tile_size in both dimensions")
        default_fallbacks = {
            "albedo": (0.5, 0.5, 0.5, 1.0),
            "normal": (0.5, 0.5, 1.0, 1.0),
            "mask": (1.0, 1.0, 1.0, 1.0),
        }
        if self.fallback is None:
            self.fallback = default_fallbacks[self.family]
        if (
            not isinstance(self.fallback, Sequence)
            or isinstance(self.fallback, (str, bytes))
            or len(self.fallback) != 4
        ):
            raise ValueError("fallback must contain exactly four channels")
        fallback = tuple(float(value) for value in self.fallback)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in fallback):
            raise ValueError("fallback channels must be finite values in [0, 1]")
        self.fallback = fallback

    @property
    def slot_size(self) -> int:
        """Physical atlas slot size = content + 2 * border."""
        return self.tile_size + 2 * self.tile_border

    @property
    def pages_x0(self) -> int:
        """Finest-level page count X."""
        import math
        return math.ceil(self.virtual_size_px[0] / self.tile_size)

    @property
    def pages_y0(self) -> int:
        """Finest-level page count Y."""
        import math
        return math.ceil(self.virtual_size_px[1] / self.tile_size)

    @property
    def full_pyramid_levels(self) -> int:
        """Maximum mip levels the virtual extent can support.
        Derived from finest page counts, not raw pixel ratio."""
        import math
        return int(math.floor(math.log2(max(self.pages_x0, self.pages_y0)))) + 1

    def pages_at_mip(self, mip: int) -> Tuple[int, int]:
        """Page count at a given mip level (ceil-div, min 1)."""
        import math
        return (
            max(1, math.ceil(self.pages_x0 / (2 ** mip))),
            max(1, math.ceil(self.pages_y0 / (2 ** mip))),
        )


@dataclass
class TerrainVTSettings:
    """Terrain material virtual texturing configuration.

    Supports feedback-driven paging for terrain material families. Albedo,
    normal, and mask families share one page-table layout and must use matching
    virtual size and tile geometry when enabled together.
    """
    enabled: bool = False
    layers: List[VTLayerFamily] = field(default_factory=lambda: [
        VTLayerFamily(family="albedo")
    ])
    atlas_size: int = 4096
    residency_budget_mb: float = 256.0
    max_mip_levels: int = 8
    use_feedback: bool = True

    @property
    def families(self) -> Tuple[str, ...]:
        """Requested family names, in layer order.

        These are the families propagated end-to-end into the native render;
        each requested family must have a registered VT source or the native
        renderer raises a fatal diagnostic instead of degrading silently.
        """
        return tuple(layer.family for layer in self.layers)

    def __post_init__(self) -> None:
        if self.enabled and not self.layers:
            raise ValueError("enabled terrain VT requires at least one family")
        families = [l.family for l in self.layers]
        if len(families) != len(set(families)):
            raise ValueError("duplicate family in layers")
        if self.atlas_size < 256:
            raise ValueError("atlas_size must be >= 256")
        if not math.isfinite(float(self.residency_budget_mb)) or self.residency_budget_mb <= 0:
            raise ValueError("residency_budget_mb must be a positive finite value")
        if self.residency_budget_mb > 512.0:
            raise ValueError("residency_budget_mb must not exceed the 512 MiB host-visible limit")
        if self.max_mip_levels < 1:
            raise ValueError("max_mip_levels must be >= 1")
        for layer in self.layers:
            if self.atlas_size % layer.slot_size != 0:
                raise ValueError(
                    f"atlas_size ({self.atlas_size}) must be divisible by "
                    f"slot_size ({layer.slot_size}) for family '{layer.family}'"
                )
        if self.layers:
            geometry = {
                (layer.virtual_size_px, layer.tile_size, layer.tile_border)
                for layer in self.layers
            }
            if len(geometry) != 1:
                raise ValueError(
                    "enabled terrain VT families must share virtual_size_px, "
                    "tile_size, and tile_border"
                )
            # The runtime splits the total budget evenly. Reject a setting
            # that cannot hold one raw RGBA logical slot for every requested
            # family; silently rounding each share up would exceed the budget.
            slot_size = self.layers[0].slot_size
            atlas_slots = (self.atlas_size // slot_size) ** 2
            if atlas_slots < len(self.layers):
                raise ValueError(
                    "atlas_size must provide at least one physical slot per "
                    f"enabled family ({len(self.layers)} required, {atlas_slots} available)"
                )
            minimum_bytes = len(self.layers) * slot_size * slot_size * 4
            budget_bytes = int(self.residency_budget_mb * 1024.0 * 1024.0)
            if budget_bytes < minimum_bytes:
                minimum_mb = minimum_bytes / (1024.0 * 1024.0)
                raise ValueError(
                    "residency_budget_mb must hold at least one logical tile "
                    f"per enabled family ({minimum_mb:.6g} MiB required)"
                )

    def actual_mip_count(self, family: str = "albedo") -> int:
        """Effective mip count: min(requested, full pyramid levels)."""
        layer = next(l for l in self.layers if l.family == family)
        return min(self.max_mip_levels, layer.full_pyramid_levels)


def validate_terrain_vt_support(
    settings: TerrainVTSettings,
    *,
    layer_id: str | None = None,
):
    """Validate terrain VT family support against the current native runtime."""
    from .diagnostics import LayerSummary, ValidationReport

    effective_layer_id = layer_id or "terrain.vt"
    diagnostics = []
    families = sorted(layer.family for layer in settings.layers)
    return ValidationReport(
        diagnostics=diagnostics,
        layer_summaries=[
            LayerSummary(
                layer_id=effective_layer_id,
                layer_type="terrain.virtual_texture",
                support_level="supported",
                diagnostic_codes=[diag.code for diag in diagnostics],
                details={
                    "enabled": settings.enabled,
                    "families": families,
                    "native_supported_families": families,
                },
            )
        ],
        supported_features={f"vt.{family}": "supported" for family in families},
        unsupported_features={},
    )


@dataclass
class OverlayBlendMode:
    """Blend mode constants for overlay layers."""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    OVERLAY = "overlay"


@dataclass
class OverlayLayerConfig:
    """Configuration for a single terrain overlay layer.
    
    Overlays are textures draped onto terrain surface, sampled in the fragment
    shader and blended into albedo before lighting. This means overlays are
    fully lit and shadowed by the sun, just like the terrain itself.
    
    Attributes:
        name: Unique identifier for this layer
        source: Path to image file (PNG, JPEG, etc.) or RGBA numpy array
        extent: Extent in terrain UV space [u_min, v_min, u_max, v_max].
                None means full terrain coverage [0, 0, 1, 1]
        opacity: Overlay opacity (0.0 = transparent, 1.0 = opaque)
        blend_mode: How to blend with terrain albedo ("normal", "multiply", "overlay")
        visible: Whether this layer is rendered
        z_order: Stacking order (lower = behind, higher = in front)
    """
    
    name: str
    source: str  # Path to image file, or np.ndarray for raw RGBA
    extent: Optional[Tuple[float, float, float, float]] = None  # [u_min, v_min, u_max, v_max]
    opacity: float = 1.0
    blend_mode: str = "normal"  # "normal", "multiply", "overlay"
    visible: bool = True
    z_order: int = 0
    
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0.0, 1.0]")
        valid_blend_modes = {"normal", "multiply", "overlay"}
        if self.blend_mode not in valid_blend_modes:
            raise ValueError(f"blend_mode must be one of {valid_blend_modes}, got '{self.blend_mode}'")
        if self.extent is not None:
            if len(self.extent) != 4:
                raise ValueError("extent must be (u_min, v_min, u_max, v_max)")
            u_min, v_min, u_max, v_max = self.extent
            if u_min >= u_max or v_min >= v_max:
                raise ValueError("extent must have u_min < u_max and v_min < v_max")


@dataclass
class OverlaySettings:
    """Terrain overlay system configuration.
    
    When enabled=False, the overlay system is disabled and output is identical
    to rendering without overlays (default off for backward compatibility).
    
    Overlays modify terrain albedo before lighting, meaning they:
    - Are lit by sun (diffuse term includes overlay color)
    - Are shadowed (shadow_term multiplies diffuse result)  
    - Receive ambient occlusion (height_ao multiplies ambient term)
    - Do NOT affect specular (specular depends on roughness, not albedo)
    
    Attributes:
        enabled: Enable the overlay system (default: False)
        global_opacity: Global opacity multiplier for all layers (0.0-1.0)
        layers: List of OverlayLayerConfig for individual overlay layers
        resolution_scale: Composite texture resolution relative to terrain
                         (1.0 = terrain resolution, 0.5 = half resolution)
    """
    
    enabled: bool = False  # Disabled by default for backward compatibility
    global_opacity: float = 1.0  # Global opacity multiplier
    layers: Optional[List[OverlayLayerConfig]] = None  # Overlay layer configs
    resolution_scale: float = 1.0  # Composite texture resolution scale
    
    def __post_init__(self) -> None:
        if not 0.0 <= self.global_opacity <= 1.0:
            raise ValueError("global_opacity must be in [0.0, 1.0]")
        if not 0.1 <= self.resolution_scale <= 2.0:
            raise ValueError("resolution_scale must be in [0.1, 2.0]")
        if self.layers is None:
            self.layers = []
    
    @property
    def has_visible_layers(self) -> bool:
        """Returns True if any layers are visible with non-zero opacity."""
        if not self.layers:
            return False
        return any(
            layer.visible and layer.opacity > 0.001 
            for layer in self.layers
        )
    
    @property
    def layer_count(self) -> int:
        """Returns the number of configured overlay layers."""
        return len(self.layers) if self.layers else 0


from enum import Enum


class PrimitiveType(Enum):
    """Primitive type for vector overlay geometry."""
    POINTS = "points"
    LINES = "lines"
    LINE_STRIP = "line_strip"
    TRIANGLES = "triangles"
    TRIANGLE_STRIP = "triangle_strip"


@dataclass
class VectorVertex:
    """Single vertex for vector overlay geometry.
    
    Vector overlays are GPU geometry (points/lines/polygons) rendered in world space,
    optionally draped onto terrain heightfield, with proper lighting and shadowing.
    
    Attributes:
        x: World X coordinate
        y: World Y coordinate (or 0 if draping - will be computed from terrain)
        z: World Z coordinate
        r: Red color component (0.0-1.0)
        g: Green color component (0.0-1.0)
        b: Blue color component (0.0-1.0)
        a: Alpha component (0.0-1.0)
        feature_id: Feature ID for picking (default 0)
    """
    x: float
    y: float
    z: float
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0
    feature_id: int = 0
    
    def __post_init__(self) -> None:
        for name, val in [("r", self.r), ("g", self.g), ("b", self.b), ("a", self.a)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0]")
    
    def to_array(self) -> List[float]:
        """Convert to [x, y, z, r, g, b, a, feature_id] array for IPC."""
        return [self.x, self.y, self.z, self.r, self.g, self.b, self.a, self.feature_id]


@dataclass
class VectorOverlayConfig:
    """Configuration for a vector overlay layer.
    
    Vector overlays render GPU geometry (points/lines/polygons) in world space,
    optionally draped onto terrain heightfield, with proper lighting and shadows.
    
    The overlay shader uses the same lighting model as terrain:
    - Diffuse: albedo * sun_color * NdotL * sun_intensity * shadow_term
    - Shadow lookup: Sample sun_vis_tex at terrain UV for shadow factor
    - Ambient: albedo * ambient
    
    This ensures overlays receive identical lighting and shadows as terrain.
    
    Attributes:
        name: Unique identifier for this overlay layer
        vertices: List of VectorVertex defining geometry
        indices: List of indices for indexed drawing
        primitive: Primitive type (points, lines, triangles, etc.)
        drape: If True, drape vertices onto terrain surface
        drape_offset: Height offset above terrain when draped (meters)
        opacity: Layer opacity (0.0-1.0)
        depth_bias: Z-fighting prevention offset (0.01-1.0)
        line_width: Line width (world units for triangle quads, pixels for GPU lines)
        point_size: Point size (world units for markers, pixels for GPU points)
        visible: Whether this layer is rendered
        z_order: Stacking order (lower = behind, higher = in front)
    
    Example:
        # Simple red triangle
        config = VectorOverlayConfig(
            name="marker",
            vertices=[
                VectorVertex(100, 0, 100, r=1, g=0, b=0),
                VectorVertex(200, 0, 100, r=0, g=1, b=0),
                VectorVertex(150, 0, 200, r=0, g=0, b=1),
            ],
            indices=[0, 1, 2],
            primitive=PrimitiveType.TRIANGLES,
            drape=True,
            drape_offset=1.0,
        )
    """
    
    name: str
    vertices: List[VectorVertex]
    indices: List[int]
    primitive: PrimitiveType = PrimitiveType.TRIANGLES
    drape: bool = False
    drape_offset: float = 0.5
    opacity: float = 1.0
    depth_bias: float = 0.1
    line_width: float = 2.0
    point_size: float = 5.0
    visible: bool = True
    z_order: int = 0
    
    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0.0, 1.0]")
        if not 0.01 <= self.depth_bias <= 1.0:
            raise ValueError("depth_bias must be in [0.01, 1.0]")
        if self.line_width < 0.1:
            raise ValueError("line_width must be >= 0.1")
        if self.point_size < 0.1:
            raise ValueError("point_size must be >= 0.1")
        if not isinstance(self.primitive, PrimitiveType):
            raise ValueError("primitive must be a PrimitiveType enum value")
    
    def to_ipc_dict(self) -> dict:
        """Convert to IPC request dictionary format."""
        return {
            "cmd": "add_vector_overlay",
            "name": self.name,
            "vertices": [v.to_array() for v in self.vertices],
            "indices": self.indices,
            "primitive": self.primitive.value,
            "drape": self.drape,
            "drape_offset": self.drape_offset,
            "opacity": self.opacity,
            "depth_bias": self.depth_bias,
            "line_width": self.line_width,
            "point_size": self.point_size,
            "z_order": self.z_order,
        }
    
    @property
    def vertex_count(self) -> int:
        """Number of vertices in this overlay."""
        return len(self.vertices)
    
    @property
    def index_count(self) -> int:
        """Number of indices in this overlay."""
        return len(self.indices)


@dataclass
class TriplanarSettings:
    """Triplanar texture mapping configuration."""

    scale: float
    blend_sharpness: float
    normal_strength: float

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be > 0")

        if self.blend_sharpness <= 0.0:
            raise ValueError("blend_sharpness must be > 0")

        if self.normal_strength < 0.0:
            raise ValueError("normal_strength must be >= 0")


@dataclass
class PomSettings:
    """Parallax occlusion mapping configuration."""

    enabled: bool
    mode: str  # "Occlusion", "Relief", "Parallax"
    scale: float
    min_steps: int
    max_steps: int
    refine_steps: int
    shadow: bool
    occlusion: bool

    def __post_init__(self) -> None:
        valid_modes = {"Occlusion", "Relief", "Parallax"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.scale < 0.0:
            raise ValueError("scale must be >= 0")

        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")

        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")

        if self.max_steps > 100:
            raise ValueError("max_steps must be <= 100")

        if self.refine_steps < 0:
            raise ValueError("refine_steps must be >= 0")


@dataclass
class LodSettings:
    """Level of detail configuration."""

    level: int
    bias: float
    lod0_bias: float

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("level must be >= 0")


@dataclass
class SamplingSettings:
    """Texture sampling configuration."""

    mag_filter: str  # "Linear", "Nearest"
    min_filter: str
    mip_filter: str
    anisotropy: int
    address_u: str  # "Repeat", "ClampToEdge", "MirrorRepeat"
    address_v: str
    address_w: str

    def __post_init__(self) -> None:
        valid_filters = {"Linear", "Nearest"}
        if self.mag_filter not in valid_filters:
            raise ValueError(f"Invalid mag_filter: {self.mag_filter}")

        if self.min_filter not in valid_filters:
            raise ValueError(f"Invalid min_filter: {self.min_filter}")

        if self.mip_filter not in valid_filters:
            raise ValueError(f"Invalid mip_filter: {self.mip_filter}")

        valid_address = {"Repeat", "ClampToEdge", "MirrorRepeat"}
        for name, value in [
            ("address_u", self.address_u),
            ("address_v", self.address_v),
            ("address_w", self.address_w),
        ]:
            if value not in valid_address:
                raise ValueError(f"Invalid {name}: {value}")

        if not 1 <= self.anisotropy <= 16:
            raise ValueError("anisotropy must be 1-16")


@dataclass
class ClampSettings:
    """Value clamping configuration."""

    height_range: Tuple[float, float]
    slope_range: Tuple[float, float]
    ambient_range: Tuple[float, float]
    shadow_range: Tuple[float, float]
    occlusion_range: Tuple[float, float]

    def __post_init__(self) -> None:
        for name, (min_val, max_val) in [
            ("height_range", self.height_range),
            ("slope_range", self.slope_range),
            ("ambient_range", self.ambient_range),
            ("shadow_range", self.shadow_range),
            ("occlusion_range", self.occlusion_range),
        ]:
            if min_val >= max_val:
                raise ValueError(f"{name}: min must be < max")


@dataclass
class TerrainRenderParams:
    """Master terrain rendering parameter container."""

    size_px: Tuple[int, int]
    render_scale: float
    # Physical span of the terrain in world units. Used to scale UVs to world XY.
    terrain_span: float
    msaa_samples: int
    z_scale: float
    cam_target: List[float]
    cam_radius: float
    cam_phi_deg: float
    cam_theta_deg: float
    cam_gamma_deg: float
    fov_y_deg: float
    clip: Tuple[float, float]
    light: LightSettings
    ibl: IblSettings
    shadows: ShadowSettings
    triplanar: TriplanarSettings
    pom: PomSettings
    lod: LodSettings
    sampling: SamplingSettings
    clamp: ClampSettings
    overlays: List  # forward reference to overlay types
    exposure: float
    gamma: float
    albedo_mode: str
    colormap_strength: float
    height_curve_mode: str = "linear"
    height_curve_strength: float = 0.0
    height_curve_power: float = 1.0
    height_curve_lut: Optional[np.ndarray] = None
    # P5-L: Lambert contrast parameter [0,1] for gradient enhancement
    lambert_contrast: float = 0.0
    # P2: Atmospheric fog (defaults to disabled for P1 compatibility)
    fog: Optional[FogSettings] = None
    # P4: Water planar reflections (defaults to disabled for P3 compatibility)
    reflection: Optional[ReflectionSettings] = None
    # Terrain water-mask settings for GPU water shading.
    water: Optional[WaterSettings] = None
    # Terrain cloud-shadow settings, defaults disabled for compatibility.
    clouds: Optional[CloudSettings] = None
    # P5: AO weight/multiplier (0.0 = no AO effect, 1.0 = full AO). Default 0.0 for P4 compatibility.
    ao_weight: float = 0.0
    # P6: Micro-detail (defaults to disabled for P5 compatibility)
    detail: Optional[DetailSettings] = None
    # Heightfield ray-traced AO (defaults to disabled for backward compatibility)
    height_ao: Optional[HeightAoSettings] = None
    # Heightfield ray-traced sun visibility (defaults to disabled for backward compatibility)
    sun_visibility: Optional[SunVisibilitySettings] = None
    # TV5: Local irradiance probes (defaults to disabled for backward compatibility)
    probes: Optional[ProbeSettings] = None
    # Local reflection probes (defaults to disabled for backward compatibility)
    reflection_probes: Optional[ReflectionProbeSettings] = None
    # P6.1: Color space correctness toggles (defaults to False for P5 compatibility)
    colormap_srgb: bool = False  # Use Rgba8UnormSrgb for colormap texture (correct sampling)
    output_srgb_eotf: bool = False  # Use exact linear_to_srgb() instead of pow-gamma
    # P7: Camera projection mode ("screen" = fullscreen triangle, "mesh" = perspective grid).
    # "mesh:zup" selects the Z-up orbit camera: mesh terrain lives in the world
    # XY plane with heights along +Z, so the legacy Y-up orbit renders oblique
    # views rolled (its up vector is a grid axis). With zup, cam_theta_deg is
    # the polar angle from +Z (0 = top-down, 90 = horizon) and cam_phi_deg the
    # azimuth within the terrain plane; plain "mesh" keeps legacy output.
    camera_mode: str = "screen"
    # Terrain submission policy. HZB mode is conservative and only active for clipmap/MSAA1.
    culling: str = "frustum"
    # Material evaluation path. "visibility" performs a depth/ID prepass and
    # one full-screen barycentric material resolve per visible pixel.
    shading: str = "forward"
    # Optional disk-backed store returned by forge3d.terrain.open_vt_store().
    vt_store: Optional[object] = None
    # One-frame camera-velocity extrapolation horizon for lower-priority pages.
    prefetch_horizon_ms: float = 100.0
    # Hard per-frame page upload cap.
    vt_upload_budget_bytes: int = 16 * 1024 * 1024
    # P7: Debug mode for projection probes (0=normal, 40=view-depth, 41=NDC depth, 42=view-pos XYZ)
    debug_mode: int = 0
    # M1: Accumulation AA sample count (1 = no AA, 16/64/256 typical for offline)
    aa_samples: int = 1
    # M1: Accumulation AA seed for deterministic jitter (None = default sequence)
    aa_seed: Optional[int] = None
    # M2: Bloom post-processing (defaults to disabled for backward compatibility)
    bloom: Optional[BloomSettings] = None
    # Screen-space effects bridge (SSAO/SSGI/SSR/TAA), defaults disabled.
    screen_space: Optional[ScreenSpaceSettings] = None
    # M4: Material layering (snow/rock/wetness, defaults to disabled for backward compatibility)
    materials: Optional[MaterialLayerSettings] = None
    # M5: Vector overlay settings (depth test, halos)
    vector_overlay: Optional[VectorOverlaySettings] = None
    # M6: Tonemap settings (operator, LUT, white balance)
    tonemap: Optional[TonemapSettings] = None
    # M1: AOV export settings
    aov: Optional[AovSettings] = None
    # M3: Depth of Field settings
    dof: Optional[DofSettings] = None
    # M4: Motion blur settings
    motion_blur: Optional[MotionBlurSettings] = None
    # M5: Lens effects settings
    lens_effects: Optional[LensEffectsSettings] = None
    # M5: Denoise settings
    denoise: Optional[DenoiseSettings] = None
    # M6: Volumetrics settings
    volumetrics: Optional[VolumetricsSettings] = None
    # M6: Sky settings
    sky: Optional[SkySettings] = None
    # TV20: Terrain material virtual texturing (defaults to disabled for backward compatibility)
    # Terrain material virtual texturing (defaults to disabled for backward compatibility)
    vt: Optional[TerrainVTSettings] = None
    # Overlay system settings (lit texture overlays draped on terrain)
    overlay: Optional[OverlaySettings] = None
    # P3-reproject: Terrain CRS for auto-reprojection of vector overlays
    terrain_crs: Optional[str] = None  # e.g., "EPSG:4326", "EPSG:32654"
    # Optional caller-provided terrain revision/checksum used for cache invalidation.
    # When set, terrain probe prep can reuse this key instead of hashing every DEM sample.
    terrain_data_revision: Optional[int] = None
    # Slope/elevation hue rotation. 0.0 preserves the pre-lighting albedo palette.
    hue_variation_strength: float = 0.08

    def __post_init__(self) -> None:
        # Default fog to disabled if not provided
        if self.fog is None:
            self.fog = FogSettings()
        # Default reflection to disabled if not provided
        if self.reflection is None:
            self.reflection = ReflectionSettings()
        if self.water is None:
            self.water = WaterSettings()
        if self.clouds is None:
            self.clouds = CloudSettings()
        # Default detail to disabled if not provided
        if self.detail is None:
            self.detail = DetailSettings()
        # Default height_ao to disabled if not provided
        if self.height_ao is None:
            self.height_ao = HeightAoSettings()
        # Default sun_visibility to disabled if not provided
        if self.sun_visibility is None:
            self.sun_visibility = SunVisibilitySettings()
        # TV5: Default probes to disabled if not provided
        if self.probes is None:
            self.probes = ProbeSettings()
        # Default reflection probes to disabled if not provided
        if self.reflection_probes is None:
            self.reflection_probes = ReflectionProbeSettings()
        # M2: Default bloom to disabled if not provided
        if self.bloom is None:
            self.bloom = BloomSettings()
        # M4: Default materials to disabled if not provided
        if self.materials is None:
            self.materials = MaterialLayerSettings()
        # M5: Default vector overlay to disabled if not provided
        if self.vector_overlay is None:
            self.vector_overlay = VectorOverlaySettings()
        # M6: Default tonemap to ACES if not provided
        if self.tonemap is None:
            self.tonemap = TonemapSettings()
        # M1: Default AOV to disabled if not provided
        if self.aov is None:
            self.aov = AovSettings()
        # M3: Default DoF to disabled if not provided
        if self.dof is None:
            self.dof = DofSettings()
        # M4: Default motion blur to disabled if not provided
        if self.motion_blur is None:
            self.motion_blur = MotionBlurSettings()
        # M5: Default lens effects to disabled if not provided
        if self.lens_effects is None:
            self.lens_effects = LensEffectsSettings()
        # M5: Default denoise to disabled if not provided
        if self.denoise is None:
            self.denoise = DenoiseSettings()
        # M6: Default volumetrics to disabled if not provided
        if self.volumetrics is None:
            self.volumetrics = VolumetricsSettings()
        # M6: Default sky to disabled if not provided
        if self.sky is None:
            self.sky = SkySettings()
        # Default overlay to disabled if not provided
        if self.overlay is None:
            self.overlay = OverlaySettings()
        width, height = self.size_px
        if width < 64 or height < 64:
            raise ValueError("size_px must be >= 64x64")

        if width > 8192 or height > 8192:
            raise ValueError("size_px must be <= 8192x8192")

        if self.terrain_span <= 0.0:
            raise ValueError("terrain_span must be > 0")

        if not 0.25 <= self.render_scale <= 4.0:
            raise ValueError("render_scale must be 0.25-4.0")

        if self.msaa_samples not in {1, 2, 4, 8}:
            raise ValueError("msaa_samples must be 1, 2, 4, or 8")

        if not 0.1 <= self.z_scale <= 50.0:
            raise ValueError("z_scale must be 0.1-50.0")

        if len(self.cam_target) != 3:
            raise ValueError("cam_target must be [x, y, z]")

        if self.cam_radius <= 0.0:
            raise ValueError("cam_radius must be > 0")

        if not 0.0 <= self.fov_y_deg <= 180.0:
            raise ValueError("fov_y_deg must be 0-180")

        near, far = self.clip
        if near <= 0.0 or near >= far:
            raise ValueError("Invalid clip planes")

        if self.albedo_mode not in {"colormap", "mix", "material"}:
            raise ValueError(f"Invalid albedo_mode: {self.albedo_mode}")

        if not 0.0 <= self.colormap_strength <= 1.0:
            raise ValueError("colormap_strength must be 0-1")

        self.hue_variation_strength = float(self.hue_variation_strength)
        if not np.isfinite(self.hue_variation_strength):
            raise ValueError("hue_variation_strength must be finite")
        self.hue_variation_strength = min(max(self.hue_variation_strength, 0.0), 0.2)

        valid_curve_modes = {"linear", "pow", "smoothstep", "lut"}
        if self.height_curve_mode not in valid_curve_modes:
            raise ValueError(
                f"height_curve_mode must be one of {sorted(valid_curve_modes)}, "
                f"got {self.height_curve_mode}"
            )

        if not 0.0 <= self.height_curve_strength <= 1.0:
            raise ValueError("height_curve_strength must be in [0, 1]")

        if self.height_curve_power <= 0.0:
            raise ValueError("height_curve_power must be > 0")

        if self.height_curve_mode == "lut":
            if self.height_curve_lut is None:
                raise ValueError("height_curve_lut is required when height_curve_mode='lut'")

            lut = np.asarray(self.height_curve_lut, dtype=np.float32)
            if lut.ndim != 1 or lut.shape[0] != 256:
                raise ValueError("height_curve_lut must be a 1D float32 array of length 256")
            if not np.isfinite(lut).all():
                raise ValueError("height_curve_lut must contain finite values")
            if np.any(lut < 0.0) or np.any(lut > 1.0):
                raise ValueError("height_curve_lut values must be within [0, 1]")

            # Store normalized LUT back on the instance for downstream consumption
            self.height_curve_lut = lut

        # P5: Validate ao_weight
        if not 0.0 <= self.ao_weight <= 1.0:
            raise ValueError("ao_weight must be 0.0-1.0")

        # M1: Validate aa_samples (must be >= 1)
        if self.aa_samples < 1:
            raise ValueError("aa_samples must be >= 1")
        if self.aa_samples > 4096:
            raise ValueError("aa_samples must be <= 4096 (practical limit for offline rendering)")

        if self.terrain_data_revision is not None:
            if isinstance(self.terrain_data_revision, bool) or not isinstance(
                self.terrain_data_revision, Integral
            ):
                raise ValueError("terrain_data_revision must be an integer or None")
            if self.terrain_data_revision < 0:
                raise ValueError("terrain_data_revision must be >= 0")
            if self.terrain_data_revision > 0xFFFF_FFFF_FFFF_FFFF:
                raise ValueError("terrain_data_revision must fit in u64")
        if self.culling not in {"none", "frustum", "hzb_two_phase"}:
            raise ValueError("culling must be one of: none, frustum, hzb_two_phase")
        if self.shading not in {"forward", "visibility"}:
            raise ValueError("shading must be one of: forward, visibility")
        if not np.isfinite(self.prefetch_horizon_ms) or self.prefetch_horizon_ms < 0.0:
            raise ValueError("prefetch_horizon_ms must be finite and >= 0")
        if (
            isinstance(self.vt_upload_budget_bytes, bool)
            or not isinstance(self.vt_upload_budget_bytes, Integral)
            or self.vt_upload_budget_bytes <= 0
        ):
            raise ValueError("vt_upload_budget_bytes must be a positive integer")
        if self.vt_store is not None and not (
            hasattr(self.vt_store, "path") or isinstance(self.vt_store, (str, Path))
        ):
            raise ValueError("vt_store must be a path or forge3d.terrain.VTStore")


def load_height_curve_lut(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Height curve LUT not found: {p}")

    try:
        data = np.load(p)
    except Exception:
        try:
            data = np.loadtxt(p)
        except Exception as exc:
            raise ValueError(f"Failed to load height curve LUT from {p}: {exc}")

    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if data.shape[0] != 256:
        raise ValueError(f"height_curve_lut must contain 256 entries, found {data.shape[0]} in {p}")
    if not np.isfinite(data).all():
        raise ValueError("height_curve_lut must contain finite values")
    if np.any(data < 0.0) or np.any(data > 1.0):
        raise ValueError("height_curve_lut values must lie within [0, 1]")
    return data


def make_terrain_params_config(
    *,
    size_px: Tuple[int, int],
    render_scale: float,
    terrain_span: float,
    msaa_samples: int,
    z_scale: float,
    exposure: float,
    domain: Tuple[float, float],
    albedo_mode: str = "mix",
    colormap_strength: float = 0.5,
    hue_variation_strength: float = 0.08,
    ibl_enabled: bool = True,
    light_azimuth_deg: float = 135.0,
    light_elevation_deg: float = 35.0,
    sun_intensity: float = 3.0,
    sun_color: Optional[Sequence[float]] = None,
    ibl_intensity: float = 1.0,
    cam_radius: float = 1200.0,
    cam_phi_deg: float = 135.0,
    cam_theta_deg: float = 45.0,
    cam_target: Sequence[float] = (0.0, 0.0, 0.0),
    fov_y_deg: float = 55.0,
    camera_mode: str = "screen",  # "screen", "mesh", or "mesh:zup" (Z-up orbit, see TerrainParams)
    culling: str = "frustum",  # "none", "frustum", or "hzb_two_phase"
    shading: str = "forward",  # "forward" or "visibility"
    vt_store: Optional[object] = None,
    prefetch_horizon_ms: float = 100.0,
    vt_upload_budget_bytes: int = 16 * 1024 * 1024,
    debug_mode: int = 0,  # 0=normal, 40=view-depth probe, 41=NDC depth, 42=view-pos XYZ
    clip: Optional[Tuple[float, float]] = None,
    height_curve_mode: str = "linear",
    height_curve_strength: float = 0.0,
    height_curve_power: float = 1.0,
    height_curve_lut: Optional[np.ndarray] = None,
    lambert_contrast: float = 0.0,  # P5-L: Lambert contrast [0,1]
    shadows: Optional[ShadowSettings] = None,
    triplanar: Optional[TriplanarSettings] = None,
    pom: Optional[PomSettings] = None,
    lod: Optional[LodSettings] = None,
    sampling: Optional[SamplingSettings] = None,
    clamp: Optional[ClampSettings] = None,
    overlays: Optional[list] = None,
    fog: Optional[FogSettings] = None,
    reflection: Optional[ReflectionSettings] = None,
    water: Optional[WaterSettings] = None,
    clouds: Optional[CloudSettings] = None,
    ao_weight: float = 0.0,
    detail: Optional[DetailSettings] = None,
    height_ao: Optional[HeightAoSettings] = None,
    sun_visibility: Optional[SunVisibilitySettings] = None,
    probes: Optional[ProbeSettings] = None,
    reflection_probes: Optional[ReflectionProbeSettings] = None,
    aa_samples: int = 1,  # M1: Accumulation AA sample count (1 = no AA)
    aa_seed: Optional[int] = None,  # M1: Accumulation AA seed for determinism
    bloom: Optional[BloomSettings] = None,  # M2: Bloom post-processing
    screen_space: Optional[ScreenSpaceSettings] = None,  # Screen-space effects bridge
    materials: Optional[MaterialLayerSettings] = None,  # M4: Material layering
    vector_overlay: Optional[VectorOverlaySettings] = None,  # M5: Vector overlay settings
    tonemap: Optional[TonemapSettings] = None,  # M6: Tonemap settings
    aov: Optional[AovSettings] = None,  # M1: AOV export settings
    dof: Optional[DofSettings] = None,  # M3: Depth of Field settings
    motion_blur: Optional[MotionBlurSettings] = None,  # M4: Motion blur settings
    lens_effects: Optional[LensEffectsSettings] = None,  # M5: Lens effects settings
    denoise: Optional[DenoiseSettings] = None,  # M5: Denoise settings
    volumetrics: Optional[VolumetricsSettings] = None,  # M6: Volumetrics settings
    sky: Optional[SkySettings] = None,  # M6: Sky settings
    vt: Optional[TerrainVTSettings] = None,  # TV20: Terrain material virtual texturing
    overlay: Optional[OverlaySettings] = None,  # Overlay settings (lit texture overlays)
    terrain_crs: Optional[str] = None,  # P3-reproject: Terrain CRS for auto-reprojection
    terrain_data_revision: Optional[int] = None,
) -> TerrainRenderParams:
    light_color = [1.0, 1.0, 1.0]
    if sun_color is not None:
        try:
            light_color = [
                float(sun_color[0]),
                float(sun_color[1]),
                float(sun_color[2]),
            ]
        except (TypeError, IndexError, ValueError):
            light_color = [1.0, 1.0, 1.0]
    light_intensity = float(sun_intensity)

    if shadows is None:
        # Default shadow settings with small bias values for proper depth comparison.
        # Large bias values (e.g., 0.5) cause all fragments to appear lit (no shadows).
        shadows = ShadowSettings(
            enabled=True,
            technique="PCSS",
            resolution=4096,
            cascades=3,
            max_distance=4000.0,
            softness=1.5,
            pcss_light_radius=0.0,
            intensity=0.8,
            slope_scale_bias=0.001,   # Slope-scaled bias for grazing angles
            depth_bias=0.0005,        # Base depth bias
            normal_bias=0.0002,       # Peter-panning offset (bias along normal)
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        )

    if triplanar is None:
        triplanar = TriplanarSettings(
            scale=6.0,
            blend_sharpness=4.0,
            normal_strength=1.0,
        )

    if pom is None:
        pom = PomSettings(
            enabled=True,
            mode="Occlusion",
            scale=0.04,
            min_steps=12,
            max_steps=40,
            refine_steps=4,
            shadow=True,
            occlusion=True,
        )

    if lod is None:
        lod = LodSettings(level=0, bias=0.0, lod0_bias=-0.5)

    if sampling is None:
        sampling = SamplingSettings(
            mag_filter="Linear",
            min_filter="Linear",
            mip_filter="Linear",
            anisotropy=8,
            address_u="Repeat",
            address_v="Repeat",
            address_w="Repeat",
        )

    if clamp is None:
        clamp = ClampSettings(
            height_range=(float(domain[0]), float(domain[1])),
            slope_range=(0.04, 1.0),
            ambient_range=(0.22, 0.38),  # P2-S1: ambient floor in [0.22, 0.38]
            shadow_range=(0.30, 1.0),    # P2-S3: shadow factor in [0.30, 1.0]
            occlusion_range=(0.65, 1.0), # P2-S2: AO capped at 35% darkening (min 0.65)
        )

    if overlays is None:
        overlays = []

    if clip is None:
        clip = (0.1, 6000.0)

    return TerrainRenderParams(
        size_px=size_px,
        render_scale=render_scale,
        terrain_span=float(terrain_span),
        msaa_samples=msaa_samples,
        z_scale=z_scale,
        cam_target=[float(value) for value in cam_target],
        cam_radius=float(cam_radius),
        cam_phi_deg=float(cam_phi_deg),
        cam_theta_deg=float(cam_theta_deg),
        cam_gamma_deg=0.0,
        fov_y_deg=float(fov_y_deg),
        clip=clip,
        light=LightSettings(
            light_type="Directional",
            azimuth_deg=float(light_azimuth_deg),
            elevation_deg=float(light_elevation_deg),
            intensity=light_intensity,
            color=light_color,
        ),
        ibl=IblSettings(
            enabled=ibl_enabled,
            intensity=float(ibl_intensity),
            rotation_deg=0.0,
        ),
        shadows=shadows,
        triplanar=triplanar,
        pom=pom,
        lod=lod,
        sampling=sampling,
        clamp=clamp,
        overlays=overlays,
        exposure=exposure,
        gamma=2.2,
        albedo_mode=albedo_mode,
        colormap_strength=colormap_strength,
        hue_variation_strength=hue_variation_strength,
        height_curve_mode=height_curve_mode,
        height_curve_strength=height_curve_strength,
        height_curve_power=height_curve_power,
        height_curve_lut=height_curve_lut,
        lambert_contrast=lambert_contrast,
        fog=fog,
        reflection=reflection,
        water=water,
        clouds=clouds,
        ao_weight=ao_weight,
        detail=detail,
        height_ao=height_ao,
        sun_visibility=sun_visibility,
        probes=probes,
        reflection_probes=reflection_probes,
        camera_mode=str(camera_mode),
        culling=str(culling),
        shading=str(shading),
        vt_store=vt_store,
        prefetch_horizon_ms=float(prefetch_horizon_ms),
        vt_upload_budget_bytes=int(vt_upload_budget_bytes),
        debug_mode=int(debug_mode),
        aa_samples=int(aa_samples),
        aa_seed=aa_seed,
        bloom=bloom,
        screen_space=screen_space,
        materials=materials,
        vector_overlay=vector_overlay,
        tonemap=tonemap,
        aov=aov,
        dof=dof,
        motion_blur=motion_blur,
        lens_effects=lens_effects,
        denoise=denoise,
        volumetrics=volumetrics,
        sky=sky,
        vt=vt,
        overlay=overlay,
        terrain_crs=terrain_crs,
        terrain_data_revision=terrain_data_revision,
    )


__all__ = [
    "LightSettings",
    "IblSettings",
    "ShadowSettings",
    "FogSettings",
    "ReflectionSettings",
    "WaterSettings",
    "CloudSettings",
    "BloomSettings",
    "ScreenSpaceSettings",
    "HeightAoSettings",
    "SunVisibilitySettings",
    "ProbeSettings",
    "ReflectionProbeSettings",
    "DetailSettings",
    "MaterialNoiseSettings",
    "MaterialLayerSettings",
    "VectorOverlaySettings",
    "TonemapSettings",
    "AovSettings",
    "DofSettings",
    "MotionBlurSettings",
    "LensEffectsSettings",
    "DenoiseSettings",
    "OfflineQualitySettings",
    "DensityVolumeSettings",
    "VolumetricsSettings",
    "SkySettings",
    "valley_fog_volume",
    "plume_volume",
    "localized_haze_volume",
    "VTLayerFamily",
    "TerrainVTSettings",
    "validate_terrain_vt_support",
    "OverlayBlendMode",
    "OverlayLayerConfig",
    "OverlaySettings",
    "TriplanarSettings",
    "PomSettings",
    "LodSettings",
    "SamplingSettings",
    "ClampSettings",
    "TerrainRenderParams",
]
