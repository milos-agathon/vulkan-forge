"""
python/forge3d/presets.py
High-level rendering presets for Python UX polish (P7-01).

Each preset returns a plain dict compatible with
python/forge3d/config.py::RendererConfig.from_mapping(), covering
lighting/shading/shadows/gi/atmosphere keys. Values are chosen to be
validation-friendly (e.g., GI modes empty by default to avoid HDRI
requirements; shadow map sizes are powers of two; CSM cascades 2..4).

Example
-------
>>> from forge3d import Renderer
>>> from forge3d import presets
>>> r = Renderer(1280, 720)
>>> cfg = presets.get("outdoor_sun")
>>> # Optionally add overrides (e.g., enable GI IBL once HDR is provided)
>>> cfg["gi"] = {"modes": ["ibl"]}
>>> cfg["atmosphere"]["hdr_path"] = "assets/sky.hdr"
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _dir_light(
    *,
    direction: tuple[float, float, float],
    intensity: float = 5.0,
    color: tuple[float, float, float] = (1.0, 0.97, 0.94),
) -> Dict[str, Any]:
    """Build a directional light mapping compatible with LightConfig.from_mapping()."""
    return {
        "type": "directional",
        "direction": [float(direction[0]), float(direction[1]), float(direction[2])],
        "intensity": float(intensity),
        "color": [float(color[0]), float(color[1]), float(color[2])],
    }


def _normalize_name(name: str) -> str:
    return "".join(c for c in str(name).strip().lower() if c not in {"-", "_", " ", "."})


# -----------------------------------------------------------------------------
# Preset definitions (schema-aligned with python/forge3d/config.py)
# -----------------------------------------------------------------------------

def studio_pbr() -> Dict[str, Any]:
    """Studio preset: directional key, Disney BRDF, PCF shadows.

    Notes
    -----
    - GI modes are left empty to avoid HDR asset requirements by default.
    - Atmosphere is disabled; users may enable HDR sky via overrides later.
    """
    return {
        "lighting": {
            "exposure": 1.0,
            "lights": [
                _dir_light(direction=(-0.30, -0.95, -0.20), intensity=6.0, color=(1.0, 0.98, 0.95)),
            ],
        },
        "shading": {
            "brdf": "disney-principled",
            "roughness": 0.35,
            "metallic": 0.0,
            "normal_maps": True,
        },
        "shadows": {
            "enabled": True,
            "technique": "pcf",
            "map_size": 2048,
            "cascades": 1,
        },
        "gi": {
            "modes": [],
        },
        "atmosphere": {
            "enabled": False,
        },
    }


def outdoor_sun() -> Dict[str, Any]:
    """Outdoor preset: Hosek–Wilkie sky, sun as directional light, CSM, GGX."""
    return {
        "lighting": {
            "exposure": 1.0,
            "lights": [
                _dir_light(direction=(-0.35, -1.00, -0.25), intensity=5.0, color=(1.0, 0.97, 0.92)),
            ],
        },
        "shading": {
            "brdf": "cooktorrance-ggx",
            "roughness": 0.5,
            "metallic": 0.0,
            "normal_maps": True,
        },
        "shadows": {
            "enabled": True,
            "technique": "pcf",  # PCF is the standard soft shadow filter on CSM pipeline
            "map_size": 2048,
            "cascades": 3,  # Valid range [2,4] for CSM; pick 3 for balance
        },
        "gi": {
            "modes": [],  # Users can enable ["ibl"] if they provide an HDR path
        },
        "atmosphere": {
            "enabled": True,
            "sky": "hosek-wilkie",
            # "hdr_path": None  # provide via overrides if using GI IBL
        },
    }


def toon_viz() -> Dict[str, Any]:
    """Toon visualization: toon BRDF, hard shadows, no GI, flat background."""
    return {
        "lighting": {
            "exposure": 1.0,
            "lights": [
                _dir_light(direction=(-0.40, -0.90, -0.10), intensity=4.0, color=(1.0, 1.0, 1.0)),
            ],
        },
        "shading": {
            "brdf": "toon",
            "normal_maps": False,
        },
        "shadows": {
            "enabled": True,
            "technique": "hard",
            "map_size": 1024,
            "cascades": 1,
        },
        "gi": {
            "modes": [],
        },
        "atmosphere": {
            "enabled": False,
        },
    }


def rainier_showcase() -> Dict[str, Any]:
    """Terrain showcase preset optimized for dramatic mountain lighting.

    Designed to avoid the "flat Rainier" problem by:
    - Offsetting sun azimuth ~90° from typical camera angles for cross-lighting
    - Using lower sun elevation (25°) for longer shadows
    - Reducing IBL intensity to 0.3 to let shadows dominate
    - Increasing sun intensity to 4.0 to compensate for reduced fill
    - Using 4 CSM cascades at high resolution for detailed shadows

    Includes camera, sun, IBL, and terrain exaggeration values so MapScene can
    consume the preset without hidden command-line parameters.
    """
    return {
        "lighting": {
            "exposure": 1.0,
            "lights": [
                # Sun direction: azimuth ~135°, elevation 25° for long shadows
                # Direction vector for sun at (az=135°, el=25°): 
                # x = cos(25°)*sin(135°) ≈ 0.64
                # y = sin(25°) ≈ 0.42  
                # z = cos(25°)*cos(135°) ≈ -0.64
                _dir_light(direction=(0.64, 0.42, -0.64), intensity=4.0, color=(1.0, 0.95, 0.90)),
            ],
        },
        "shading": {
            "brdf": "cooktorrance-ggx",
            "roughness": 0.6,
            "metallic": 0.0,
            "normal_maps": True,
        },
        "shadows": {
            "enabled": True,
            "technique": "pcss",  # PCSS for variable penumbra shadows
            "map_size": 4096,
            "cascades": 4,
        },
        "gi": {
            "modes": ["ibl", "ssao"],
            "ambient_occlusion_strength": 0.35,
        },
        "atmosphere": {
            "enabled": True,
            "sky": "hosek-wilkie",
        },
        "camera": {
            "target": [0.0, 0.0, 0.0],
            "radius_scale": 2.4,
            "azimuth_deg": 135.0,
            "elevation_deg": 45.0,
            "fov_deg": 55.0,
        },
        "sun": {
            "azimuth_deg": 135.0,
            "elevation_deg": 25.0,
            "intensity": 4.0,
            "color": [1.0, 0.95, 0.90],
            "direction": [0.64, 0.42, -0.64],
        },
        "ibl": {
            "builtin": "clear_sky",
            "intensity": 0.3,
        },
        "exaggeration": 1.35,
        "height_sampling": "Nearest",
        "reproducibility": {
            "seed": 1350,
            "renderer_backend": "gpu_terrain",
            "pixel_tolerance": 0.005,
        },
    }


def rainier_relief() -> Dict[str, Any]:
    """Rainier relief preset: low sun + perspective mesh for strong terrain relief.
    
    Constraints:
    1. Sun elevation is low (18 deg < 30) for long shadows.
    2. Camera defaults to a perspective mesh view (theta 65 deg, phi 45 deg, fov 55).
    3. Relief comes from low-angle sun plus high-resolution PCSS shadows.
    4. Camera, sun, IBL, and terrain exaggeration values are included in the
       returned mapping for MapScene and terrain-demo parity.
    """
    import math

    sun_az_rad = math.radians(225.0)
    sun_el_rad = math.radians(18.0)
    sun_x = math.cos(sun_el_rad) * math.sin(sun_az_rad)
    sun_y = math.sin(sun_el_rad)
    sun_z = math.cos(sun_el_rad) * math.cos(sun_az_rad)

    return {
        "lighting": {
            "exposure": 1.2,
            "lights": [
                _dir_light(
                    direction=(sun_x, sun_y, sun_z),
                    intensity=5.0,
                    color=(1.0, 0.92, 0.85),
                ),
            ],
        },
        "shading": {
            "brdf": "cooktorrance-ggx",
            "roughness": 0.55,
            "metallic": 0.0,
            "normal_maps": True,
        },
        "shadows": {
            "enabled": True,
            "technique": "pcss",
            "map_size": 4096,
            "cascades": 4,
            "light_size": 2.0,
        },
        "gi": {
            "modes": ["ibl", "ssao"],
            "ambient_occlusion_strength": 0.45,
        },
        "atmosphere": {
            "enabled": True,
            "sky": "hosek-wilkie",
        },
        "camera": {
            "target": [0.0, 0.0, 0.0],
            "radius_scale": 2.1,
            "azimuth_deg": 45.0,
            "elevation_deg": 65.0,
            "fov_deg": 55.0,
        },
        "sun": {
            "azimuth_deg": 225.0,
            "elevation_deg": 18.0,
            "intensity": 5.0,
            "color": [1.0, 0.92, 0.85],
            "direction": [sun_x, sun_y, sun_z],
        },
        "ibl": {
            "builtin": "clear_sky",
            "intensity": 0.25,
        },
        "exaggeration": 1.5,
        "reproducibility": {
            "seed": 1818,
            "renderer_backend": "gpu_terrain",
            "pixel_tolerance": 0.005,
        },
        "cli_params": {
            "camera_mode": "mesh",
            "cam_theta": 65.0,
            "cam_phi": 45.0,
            "cam_fov": 55.0,
        },
    }



# -----------------------------------------------------------------------------
# Registry and lookup helpers
# -----------------------------------------------------------------------------

_PRESETS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "studiopbr": studio_pbr,
    "outdoorsun": outdoor_sun,
    "toonviz": toon_viz,
    "rainiershowcase": rainier_showcase,
    "rainierrelief": rainier_relief,
}

_ALIASES: Dict[str, str] = {
    "studio": "studiopbr",
    "pbr": "studiopbr",
    "sun": "outdoorsun",
    "outdoor": "outdoorsun",
    "toon": "toonviz",
    "rainier": "rainiershowcase",
    "showcase": "rainiershowcase",
    "terrain": "rainiershowcase",
    "relief": "rainierrelief",
    "lowangle": "rainierrelief",
}


def available() -> List[str]:
    """List available preset names."""
    return sorted(_PRESETS.keys())


def get(name: str) -> Dict[str, Any]:
    """Resolve a preset by name (case-insensitive; supports common aliases).

    Raises
    ------
    ValueError
        If the preset name is unknown.
    """
    key = _normalize_name(name)
    if key in _ALIASES:
        key = _ALIASES[key]
    if key not in _PRESETS:
        raise ValueError(f"Unknown preset: {name!r}. Available: {', '.join(available())}")
    # Return a shallow copy to avoid accidental mutation of registry entries
    out = _PRESETS[key]()
    assert isinstance(out, dict)
    return copy.deepcopy(out)


__all__ = [
    "studio_pbr",
    "outdoor_sun",
    "toon_viz",
    "rainier_showcase",
    "rainier_relief",
    "available",
    "get",
]
