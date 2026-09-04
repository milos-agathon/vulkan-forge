# python/forge3d/config.py
# Renderer configuration parsing utilities for lighting and shading
# Exists to keep Python and Rust renderer config structures aligned
# RELEVANT FILES: python/forge3d/__init__.py, src/render/params.rs, examples/terrain_demo.py, tests/test_renderer_config.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence, Union, Optional, Tuple, Dict, List

ConfigSource = Union["RendererConfig", Mapping[str, Any], str, Path, None]

_LIGHT_TYPES: Dict[str, str] = {
    "directional": "directional",
    "dir": "directional",
    "sun": "directional",
    "point": "point",
    "pointlight": "point",
    "spot": "spot",
    "spotlight": "spot",
    "arearect": "area-rect",
    "rect": "area-rect",
    "rectlight": "area-rect",
    "areadisk": "area-disk",
    "disk": "area-disk",
    "disklight": "area-disk",
    "areasphere": "area-sphere",
    "sphere": "area-sphere",
    "spherelight": "area-sphere",
    "environment": "environment",
    "env": "environment",
    "hdri": "environment",
}

_BRDF_MODELS: Dict[str, str] = {
    "lambert": "lambert",
    "phong": "phong",
    "blinnphong": "blinn-phong",
    "blinn-phong": "blinn-phong",
    "orennayar": "oren-nayar",
    "oren-nayar": "oren-nayar",
    "cooktorranceggx": "cooktorrance-ggx",
    "cooktorrance-ggx": "cooktorrance-ggx",
    "ggx": "cooktorrance-ggx",
    "cooktorrancebeckmann": "cooktorrance-beckmann",
    "cooktorrance-beckmann": "cooktorrance-beckmann",
    "beckmann": "cooktorrance-beckmann",
    "disneyprincipled": "disney-principled",
    "disney-principled": "disney-principled",
    "disney": "disney-principled",
    "ashikhminshirley": "ashikhmin-shirley",
    "ashikhmin-shirley": "ashikhmin-shirley",
    "ward": "ward",
    "toon": "toon",
    "minnaert": "minnaert",
    "subsurface": "subsurface",
    "sss": "subsurface",
    "hair": "hair",
    "kajiyakay": "hair",
    "kajiya-kay": "hair",
}

# Supported shadow techniques for terrain rendering
# P0.2/M3: VSM/EVSM/MSM now fully supported with moment-based shadow maps
_SHADOW_TECHNIQUES: Dict[str, str] = {
    "none": "none",  # P0: Disable shadows entirely
    "hard": "hard",  # Single-sample, hard-edged shadows
    "pcf": "pcf",    # Percentage-closer filtering (soft edges)
    "pcss": "pcss",  # Percentage-closer soft shadows (variable penumbra)
    "vsm": "vsm",    # Variance shadow maps (soft shadows, potential light leaks)
    "evsm": "evsm",  # Exponential variance shadow maps (reduced light leaks)
    "msm": "msm",    # Moment shadow maps (high quality, 4 moments)
}

# CSM is a pipeline, not a filter technique - provide clear error message
_UNSUPPORTED_SHADOW_TECHNIQUES: set = {"csm"}

def validate_shadow_technique(technique: str) -> str:
    """Validate shadow technique and provide clear error for unsupported ones."""
    key = _normalize_key(technique)
    if key in _UNSUPPORTED_SHADOW_TECHNIQUES:
        supported = ", ".join(sorted(_SHADOW_TECHNIQUES.keys()))
        raise ValueError(
            f"Shadow technique 'csm' is not a valid filter option. "
            f"CSM (Cascaded Shadow Maps) is the underlying pipeline used by all techniques. "
            f"Use --shadow-technique with one of: {supported}"
        )
    return _normalize_choice(technique, _SHADOW_TECHNIQUES, "shadow technique")


def _plain_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(value))

_GI_MODES: Dict[str, str] = {
    "none": "none",
    "ibl": "ibl",
    "irradianceprobes": "irradiance-probes",
    "irradiance-probes": "irradiance-probes",
    "probes": "irradiance-probes",
    "ddgi": "ddgi",
    "voxelconetracing": "voxel-cone-tracing",
    "voxel-cone-tracing": "voxel-cone-tracing",
    "vct": "voxel-cone-tracing",
    "ssao": "ssao",
    "gtao": "gtao",
    "ssgi": "ssgi",
    "ssr": "ssr",
}

_SKY_MODELS: Dict[str, str] = {
    "hosekwilkie": "hosek-wilkie",
    "hosek-wilkie": "hosek-wilkie",
    "preetham": "preetham",
    "hdri": "hdri",
    "environment": "hdri",
    "envmap": "hdri",
}

_PHASE_FUNCTIONS: Dict[str, str] = {
    "isotropic": "isotropic",
    "henyeygreenstein": "henyey-greenstein",
    "henyey-greenstein": "henyey-greenstein",
    "hg": "henyey-greenstein",
}


def _normalize_key(value: Any) -> str:
    return "".join(
        c
        for c in str(value).strip().lower()
        if c not in {"-", "_", " ", "."}
    )


def _normalize_choice(value: Any, mapping: Mapping[str, str], label: str) -> str:
    key = _normalize_key(value)
    if key not in mapping:
        raise ValueError(f"Unknown {label}: {value!r}")
    return mapping[key]


def _to_float3(value: Any, label: str) -> Tuple[float, float, float]:
    if value is None:
        raise ValueError(f"{label} requires three floats")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError(f"{label} must be a sequence of three numeric values")


def _to_float2(value: Any, label: str) -> Tuple[float, float]:
    if value is None:
        raise ValueError(f"{label} requires two floats")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"{label} must be a sequence of two numeric values")


def _maybe_float3(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    return _to_float3(value, "vector")


def _maybe_float2(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    return _to_float2(value, "vector")


@dataclass
class LightConfig:
    type: str = "directional"
    intensity: float = 5.0
    color: Tuple[float, float, float] = (1.0, 0.97, 0.94)
    direction: Optional[Tuple[float, float, float]] = (-0.35, -1.0, -0.25)
    position: Optional[Tuple[float, float, float]] = None
    cone_angle: Optional[float] = None
    area_extent: Optional[Tuple[float, float]] = None
    hdr_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "intensity": self.intensity,
            "color": list(self.color),
            "direction": list(self.direction) if self.direction is not None else None,
            "position": list(self.position) if self.position is not None else None,
            "cone_angle": self.cone_angle,
            "area_extent": list(self.area_extent) if self.area_extent is not None else None,
            "hdr_path": self.hdr_path,
        }

    def validate(self, index: int) -> None:
        label = f"lights[{index}]"
        if self.type == "directional":
            if self.direction is None:
                raise ValueError(f"{label}.direction required for directional lights")
        if self.type in {"point", "spot", "area-rect", "area-disk", "area-sphere"}:
            if self.position is None:
                raise ValueError(f"{label}.position required for {self.type} lights")
        if self.type == "environment":
            if self.hdr_path is None:
                raise ValueError(f"{label}.hdr_path required for environment lights")
        if self.cone_angle is not None and not (0.0 <= float(self.cone_angle) <= 180.0):
            raise ValueError(f"{label}.cone_angle must be within [0, 180]")
        if self.area_extent is not None:
            if self.area_extent[0] <= 0.0 or self.area_extent[1] <= 0.0:
                raise ValueError(f"{label}.area_extent entries must be positive")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["LightConfig"] = None) -> "LightConfig":
        base = copy.deepcopy(default) if default is not None else cls()
        if "type" in data:
            base.type = _normalize_choice(data["type"], _LIGHT_TYPES, "light type")
        if "intensity" in data:
            base.intensity = float(data["intensity"])
        if "color" in data:
            base.color = _to_float3(data["color"], "color")
        if "direction" in data:
            base.direction = _maybe_float3(data["direction"])
        if "position" in data:
            base.position = _maybe_float3(data["position"])
        if "cone_angle" in data:
            base.cone_angle = None if data["cone_angle"] is None else float(data["cone_angle"])
        if "area_extent" in data:
            base.area_extent = _maybe_float2(data["area_extent"])
        if "hdr" in data and "hdr_path" not in data:
            base.hdr_path = data.get("hdr")
        if "hdr_path" in data:
            base.hdr_path = None if data["hdr_path"] is None else str(data["hdr_path"])
        if default is None and "direction" not in data and base.type == "directional":
            base.direction = None
        return base


@dataclass
class LightingParams:
    lights: List[LightConfig] = field(default_factory=lambda: [LightConfig()])
    exposure: float = 1.0

    def to_dict(self) -> dict:
        return {
            "lights": [light.to_dict() for light in self.lights],
            "exposure": self.exposure,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["LightingParams"] = None) -> "LightingParams":
        base = copy.deepcopy(default) if default is not None else cls()
        if "exposure" in data:
            base.exposure = float(data["exposure"])
        lights_value = data.get("lights", data.get("light"))
        if lights_value is not None:
            lights_list: List[LightConfig] = []
            items = lights_value if isinstance(lights_value, Sequence) and not isinstance(lights_value, (str, bytes)) else [lights_value]
            for item in items:
                if isinstance(item, Mapping):
                    lights_list.append(LightConfig.from_mapping(item))
                else:
                    raise TypeError("lighting.lights entries must be mappings")
            if not lights_list:
                raise ValueError("lighting.lights requires at least one light definition")
            base.lights = lights_list
        return base


@dataclass
class ShadingParams:
    brdf: str = "cooktorrance-ggx"
    normal_maps: bool = True
    metallic: float = 0.0
    roughness: float = 0.5
    sheen: float = 0.0
    clearcoat: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "brdf": self.brdf,
            "normal_maps": self.normal_maps,
            "metallic": self.metallic,
            "roughness": self.roughness,
            "sheen": self.sheen,
            "clearcoat": self.clearcoat,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["ShadingParams"] = None) -> "ShadingParams":
        base = copy.deepcopy(default) if default is not None else cls()
        if "brdf" in data:
            base.brdf = _normalize_choice(data["brdf"], _BRDF_MODELS, "BRDF model")
        if "normal_maps" in data:
            base.normal_maps = bool(data["normal_maps"])
        if "metallic" in data:
            base.metallic = float(data["metallic"])
        if "roughness" in data:
            base.roughness = float(data["roughness"])
        if "sheen" in data:
            base.sheen = float(data["sheen"])
        if "clearcoat" in data:
            base.clearcoat = float(data["clearcoat"])
        return base

@dataclass
class VolumetricParams:
    density: float = 0.02
    phase: str = "isotropic"
    anisotropy: float = 0.0
    mode: str = "raymarch"  # or "froxels"
    max_steps: int = 64
    height_falloff: float = 0.0
    start_distance: float = 0.0
    max_distance: float = 1000.0
    absorption: float = 0.0
    scattering_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    temporal_alpha: float = 0.2
    use_shadows: bool = False
    jitter_strength: float = 0.25
    preset: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "density": self.density,
            "phase": self.phase,
            "anisotropy": self.anisotropy,
            "mode": self.mode,
            "max_steps": self.max_steps,
            "height_falloff": self.height_falloff,
            "start_distance": self.start_distance,
            "max_distance": self.max_distance,
            "absorption": self.absorption,
            "scattering_color": list(self.scattering_color),
            "ambient_color": list(self.ambient_color),
            "temporal_alpha": self.temporal_alpha,
            "use_shadows": self.use_shadows,
            "jitter_strength": self.jitter_strength,
            "preset": self.preset,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["VolumetricParams"] = None) -> "VolumetricParams":
        base = copy.deepcopy(default) if default is not None else cls()
        if "density" in data:
            base.density = float(data["density"])
        if "phase" in data:
            base.phase = _normalize_choice(data["phase"], _PHASE_FUNCTIONS, "volumetric phase")
        if "anisotropy" in data:
            base.anisotropy = float(data["anisotropy"])
        if "g" in data:
            base.anisotropy = float(data["g"])
        if "mode" in data:
            m = str(data["mode"]).strip().lower()
            if m in {"raymarch", "rm", "0"}:
                base.mode = "raymarch"
            elif m in {"froxels", "fx", "1"}:
                base.mode = "froxels"
        if "max_steps" in data:
            base.max_steps = int(data["max_steps"])
        if "height_falloff" in data:
            base.height_falloff = float(data["height_falloff"])
        if "start_distance" in data:
            base.start_distance = float(data["start_distance"])
        if "max_distance" in data:
            base.max_distance = float(data["max_distance"])
        if "absorption" in data:
            base.absorption = float(data["absorption"])
        if "scattering_color" in data:
            base.scattering_color = _to_float3(data["scattering_color"], "scattering_color")
        if "ambient_color" in data:
            base.ambient_color = _to_float3(data["ambient_color"], "ambient_color")
        if "temporal_alpha" in data:
            base.temporal_alpha = float(data["temporal_alpha"])
        if "use_shadows" in data:
            base.use_shadows = bool(data["use_shadows"])
        if "jitter_strength" in data:
            base.jitter_strength = float(data["jitter_strength"])
        return base

@dataclass
class AtmosphereParams:
    enabled: bool = True
    sky: str = "hosek-wilkie"
    hdr_path: Optional[str] = None
    volumetric: Optional[VolumetricParams] = None

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "sky": self.sky,
            "hdr_path": self.hdr_path,
            "volumetric": self.volumetric.to_dict() if self.volumetric is not None else None,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["AtmosphereParams"] = None) -> "AtmosphereParams":
        base = copy.deepcopy(default) if default is not None else cls()
        if "enabled" in data:
            base.enabled = bool(data["enabled"])
        if "sky" in data:
            base.sky = _normalize_choice(data["sky"], _SKY_MODELS, "sky model")
        if "hdr_path" in data:
            base.hdr_path = None if data["hdr_path"] is None else str(data["hdr_path"])
        if "hdr" in data and "hdr_path" not in data:
            base.hdr_path = None if data["hdr"] is None else str(data["hdr"])
        if "volumetric" in data:
            if data["volumetric"] is None:
                base.volumetric = None
            elif isinstance(data["volumetric"], Mapping):
                base.volumetric = VolumetricParams.from_mapping(data["volumetric"])
            else:
                raise TypeError("atmosphere.volumetric must be a mapping or None")
        return base


@dataclass
class ShadowParams:
    enabled: bool = True
    technique: str = "pcf"
    map_size: int = 2048
    cascades: int = 1
    contact_hardening: bool = False
    pcss_blocker_radius: float = 6.0
    pcss_filter_radius: float = 4.0
    light_size: float = 1.0
    moment_bias: float = 5e-4

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "technique": self.technique,
            "map_size": self.map_size,
            "cascades": self.cascades,
            "contact_hardening": self.contact_hardening,
            "pcss_blocker_radius": self.pcss_blocker_radius,
            "pcss_filter_radius": self.pcss_filter_radius,
            "light_size": self.light_size,
            "moment_bias": self.moment_bias,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["ShadowParams"] = None) -> "ShadowParams":
        base = copy.deepcopy(default) if default is not None else cls()
        if "enabled" in data:
            base.enabled = bool(data["enabled"])
        if "technique" in data:
            base.technique = validate_shadow_technique(data["technique"])
            # P0: 'none' technique means shadows disabled
            if base.technique == "none":
                base.enabled = False
        if "map_size" in data:
            base.map_size = int(data["map_size"])
        if "cascades" in data:
            base.cascades = int(data["cascades"])
        if "contact_hardening" in data:
            base.contact_hardening = bool(data["contact_hardening"])
        if "pcss_blocker_radius" in data:
            base.pcss_blocker_radius = float(data["pcss_blocker_radius"])
        if "pcss_filter_radius" in data:
            base.pcss_filter_radius = float(data["pcss_filter_radius"])
        if "light_size" in data:
            base.light_size = float(data["light_size"])
        if "moment_bias" in data:
            base.moment_bias = float(data["moment_bias"])
        return base

    def requires_moments(self) -> bool:
        """Return True if technique requires moment-based shadow maps (VSM/EVSM/MSM)."""
        return self.technique in {"vsm", "evsm", "msm"}

    def atlas_memory_bytes(self) -> int:
        # Depth32Float (4) plus the Rgba16Float moment atlas (8) and its
        # persistent separable-blur intermediate (8).
        bpp = 20 if self.requires_moments() else 4
        return int(self.map_size) * int(self.map_size) * bpp * max(1, int(self.cascades))


@dataclass
class GiParams:
    modes: List[str] = field(default_factory=list)
    ambient_occlusion_strength: float = 0.0

    def to_dict(self) -> dict:
        return {
            "modes": list(self.modes),
            "ambient_occlusion_strength": self.ambient_occlusion_strength,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["GiParams"] = None) -> "GiParams":
        base = copy.deepcopy(default) if default is not None else cls()
        if "modes" in data:
            raw = data["modes"]
            items = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else [raw]
            out: List[str] = []
            for item in items:
                if item is None:
                    continue
                m = _normalize_choice(item, _GI_MODES, "gi mode")
                if m != "none" and m not in out:
                    out.append(m)
            base.modes = out
        if "ambient_occlusion_strength" in data:
            base.ambient_occlusion_strength = float(data["ambient_occlusion_strength"])
        return base


@dataclass
class RendererConfig:
    lighting: LightingParams = field(default_factory=LightingParams)
    shading: ShadingParams = field(default_factory=ShadingParams)
    shadows: ShadowParams = field(default_factory=ShadowParams)
    gi: GiParams = field(default_factory=GiParams)
    atmosphere: AtmosphereParams = field(default_factory=AtmosphereParams)
    camera: Dict[str, Any] = field(default_factory=dict)
    sun: Dict[str, Any] = field(default_factory=dict)
    ibl: Dict[str, Any] = field(default_factory=dict)
    exaggeration: Optional[float] = None
    brdf_override: Optional[str] = None

    def to_dict(self) -> dict:
        data = {
            "lighting": self.lighting.to_dict(),
            "shading": self.shading.to_dict(),
            "shadows": self.shadows.to_dict(),
            "gi": self.gi.to_dict(),
            "atmosphere": self.atmosphere.to_dict(),
        }
        if self.camera:
            data["camera"] = copy.deepcopy(self.camera)
        if self.sun:
            data["sun"] = copy.deepcopy(self.sun)
        if self.ibl:
            data["ibl"] = copy.deepcopy(self.ibl)
        if self.exaggeration is not None:
            data["exaggeration"] = float(self.exaggeration)
        if self.brdf_override is not None:
            data["brdf_override"] = self.brdf_override
        return data

    def copy(self) -> "RendererConfig":
        return copy.deepcopy(self)

    def validate(self) -> None:
        for idx, light in enumerate(self.lighting.lights):
            light.validate(idx)
        if self.shadows.enabled:
            if self.shadows.map_size <= 0:
                raise ValueError(
                    "shadows.map_size must be greater than zero when shadows are enabled"
                )
            if self.shadows.map_size & (self.shadows.map_size - 1):
                raise ValueError("shadows.map_size must be a power of two")
            if (
                self.shadows.technique in {"pcss", "pcf", "vsm", "evsm", "msm"}
                and self.shadows.map_size < 256
            ):
                raise ValueError(
                    "shadows.map_size should be at least 256 for filtered techniques"
                )
            if not (1 <= self.shadows.cascades <= 4):
                raise ValueError("shadows.cascades must be within [1, 4]")
            if self.shadows.technique == "pcss":
                if self.shadows.pcss_blocker_radius < 0.0:
                    raise ValueError("shadows.pcss_blocker_radius must be non-negative")
                if self.shadows.pcss_filter_radius < 0.0:
                    raise ValueError("shadows.pcss_filter_radius must be non-negative")
                if self.shadows.light_size <= 0.0:
                    raise ValueError("shadows.light_size must be positive for PCSS")
            if self.shadows.requires_moments() and self.shadows.moment_bias <= 0.0:
                raise ValueError("shadows.moment_bias must be positive for moment-based techniques")
            max_bytes = 256 * 1024 * 1024
            if self.shadows.atlas_memory_bytes() > max_bytes:
                raise ValueError(
                    f"shadow atlas exceeds 256 MiB budget (map_size={self.shadows.map_size}, cascades={self.shadows.cascades})"
                )
        if self.atmosphere.enabled and self.atmosphere.sky == "hdri":
            if self.atmosphere.hdr_path is None and not any(light.type == "environment" and light.hdr_path for light in self.lighting.lights):
                raise ValueError("atmosphere.sky=hdri requires atmosphere.hdr_path or an environment light with hdr_path")
        if self.atmosphere.volumetric is not None:
            if self.atmosphere.volumetric.density < 0.0:
                raise ValueError("atmosphere.volumetric.density must be non-negative")
            if self.atmosphere.volumetric.phase == "henyey-greenstein":
                if not (-0.999 <= self.atmosphere.volumetric.anisotropy <= 0.999):
                    raise ValueError("atmosphere.volumetric.anisotropy must be within [-0.999, 0.999] for Henyey-Greenstein")
        for mode in self.gi.modes:
            if mode == "ibl":
                has_env = any(light.type == "environment" and light.hdr_path for light in self.lighting.lights)
                has_ibl_path = any(self.ibl.get(key) for key in ("path", "hdr_path", "environment_path"))
                builtin = str(self.ibl.get("builtin") or "").lower().replace("_", "").replace("-", "")
                has_builtin = builtin in {"clear", "clearsky"}
                if builtin and not has_builtin:
                    raise ValueError(f"unknown built-in IBL source: {self.ibl['builtin']!r}")
                if not has_env and self.atmosphere.hdr_path is None and not has_ibl_path and not has_builtin:
                    raise ValueError(
                        "gi mode 'ibl' requires an environment light, atmosphere.hdr_path, "
                        "ibl path, or recognized built-in IBL source"
                    )
        if self.exaggeration is not None and float(self.exaggeration) <= 0.0:
            raise ValueError("exaggeration must be positive")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], default: Optional["RendererConfig"] = None) -> "RendererConfig":
        base = copy.deepcopy(default) if default is not None else cls()
        if "lighting" in data:
            if isinstance(data["lighting"], Mapping):
                base.lighting = LightingParams.from_mapping(data["lighting"], base.lighting)
            else:
                raise TypeError("lighting must be a mapping")
        if "shading" in data:
            if isinstance(data["shading"], Mapping):
                base.shading = ShadingParams.from_mapping(data["shading"], base.shading)
            else:
                raise TypeError("shading must be a mapping")
        if "shadows" in data:
            if isinstance(data["shadows"], Mapping):
                base.shadows = ShadowParams.from_mapping(data["shadows"], base.shadows)
            else:
                raise TypeError("shadows must be a mapping")
        if "gi" in data:
            if isinstance(data["gi"], Mapping):
                base.gi = GiParams.from_mapping(data["gi"], base.gi)
            else:
                base.gi = GiParams.from_mapping({"modes": data["gi"]}, base.gi)
        if "atmosphere" in data:
            if isinstance(data["atmosphere"], Mapping):
                base.atmosphere = AtmosphereParams.from_mapping(data["atmosphere"], base.atmosphere)
            else:
                raise TypeError("atmosphere must be a mapping")
        if "camera" in data:
            base.camera = _plain_mapping(data["camera"], "camera")
        if "sun" in data:
            base.sun = _plain_mapping(data["sun"], "sun")
        if "ibl" in data:
            base.ibl = _plain_mapping(data["ibl"], "ibl")
        if "exaggeration" in data:
            value = data["exaggeration"]
            base.exaggeration = None if value is None else float(value)
        if "brdf_override" in data:
            value = data["brdf_override"]
            base.brdf_override = None if value is None else _normalize_choice(value, _BRDF_MODELS, "BRDF model")
        return base


def _load_from_path(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".json", ""}:
        return json.loads(text)
    raise ValueError(f"Unsupported renderer config file format: {path}")


def _build_override_mapping(overrides: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in overrides.items():
        if key in {"light", "lights"}:
            out.setdefault("lighting", {})["lights"] = value
        elif key == "exposure":
            out.setdefault("lighting", {})["exposure"] = value
        elif key == "brdf":
            out.setdefault("shading", {})["brdf"] = value
        elif key in {"shadows", "shadow_technique"}:
            out.setdefault("shadows", {})["technique"] = value
        elif key in {"shadow_map_res", "shadow_map_resolution", "shadow_map_size"}:
            out.setdefault("shadows", {})["map_size"] = value
        elif key == "cascades":
            out.setdefault("shadows", {})["cascades"] = value
        elif key in {"contact_hardening", "shadow_contact_hardening"}:
            out.setdefault("shadows", {})["contact_hardening"] = value
        elif key in {"pcss_blocker_radius", "pcss_search_radius"}:
            out.setdefault("shadows", {})["pcss_blocker_radius"] = value
        elif key in {"pcss_filter_radius", "pcss_filter_size"}:
            out.setdefault("shadows", {})["pcss_filter_radius"] = value
        elif key in {"pcss_light_size", "light_size", "shadow_light_size"}:
            out.setdefault("shadows", {})["light_size"] = value
        elif key in {"moment_bias", "shadow_moment_bias"}:
            out.setdefault("shadows", {})["moment_bias"] = value
        elif key in {"gi", "gi_modes"}:
            out.setdefault("gi", {})["modes"] = value
        elif key == "ambient_occlusion_strength":
            out.setdefault("gi", {})["ambient_occlusion_strength"] = value
        elif key == "sky":
            out.setdefault("atmosphere", {})["sky"] = value
        elif key in {"hdr", "hdr_path"}:
            out.setdefault("atmosphere", {})["hdr_path"] = value
        elif key == "volumetric":
            # CLI-style volumetric overrides come in two forms:
            # - mapping: nested atmosphere.volumetric config (P6).
            # - string: opaque spec parsed elsewhere (e.g., terrain_demo).
            #
            # Only the mapping form participates in AtmosphereParams.from_mapping;
            # string forms are accepted but ignored here so they do not cause
            # type errors during config validation.
            if isinstance(value, Mapping):
                out.setdefault("atmosphere", {})["volumetric"] = value
        elif key == "atmosphere":
            if isinstance(value, Mapping):
                out.setdefault("atmosphere", {}).update(value)
            else:
                raise TypeError("atmosphere override must be a mapping")
        elif key in {"camera", "sun", "ibl"}:
            if isinstance(value, Mapping):
                out.setdefault(key, {}).update(value)
            else:
                raise TypeError(f"{key} override must be a mapping")
        elif key == "exaggeration":
            out["exaggeration"] = value
        elif key == "brdf_override":
            out["brdf_override"] = value
        else:
            # Unknown keys are handled by caller.
            pass
    return out


def load_renderer_config(config: ConfigSource = None, overrides: Optional[Mapping[str, Any]] = None) -> RendererConfig:
    if isinstance(config, RendererConfig):
        cfg = config.copy()
    elif isinstance(config, Mapping):
        cfg = RendererConfig.from_mapping(config)
    elif isinstance(config, (str, Path)):
        cfg = RendererConfig.from_mapping(_load_from_path(Path(config)))
    elif config is None:
        cfg = RendererConfig()
    else:
        raise TypeError("config must be RendererConfig, mapping, path, or None")

    if overrides:
        merged = _build_override_mapping(overrides)
        if merged:
            cfg = RendererConfig.from_mapping(merged, cfg)
    cfg.validate()
    return cfg


def split_renderer_overrides(kwargs: MutableMapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    overrides: Dict[str, Any] = {}
    remaining: Dict[str, Any] = {}
    recognized = {
        "light",
        "lights",
        "exposure",
        "brdf",
        "shadows",
        "shadow_technique",
        "shadow_map_res",
        "shadow_map_resolution",
        "shadow_map_size",
        "cascades",
        "contact_hardening",
        "pcss_blocker_radius",
        "pcss_search_radius",
        "pcss_filter_radius",
        "pcss_filter_size",
        "light_size",
        "shadow_light_size",
        "pcss_light_size",
        "moment_bias",
        "shadow_moment_bias",
        "gi",
        "gi_modes",
        "ambient_occlusion_strength",
        "sky",
        "hdr",
        "hdr_path",
        "volumetric",
        "atmosphere",
        "camera",
        "sun",
        "ibl",
        "exaggeration",
        "brdf_override",
    }
    for key, value in list(kwargs.items()):
        if key in recognized:
            overrides[key] = kwargs.pop(key)
        else:
            remaining[key] = value
    return overrides, remaining
