from __future__ import annotations

import os
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import forge3d as f3d

from _terrain_runtime import terrain_rendering_available
from tests._golden_variants import (
    assert_nvidia_vulkan_golden_adapter,
    nvidia_vulkan_golden_selected,
    selected_golden_variant,
)
from tests._ssim import ssim


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "recipes"
CERT_DIR = ROOT / "tests" / "golden" / "certificates"
SIGNING_PUB_PATH = CERT_DIR / "signing.pub"
def _update_goldens_enabled() -> bool:
    """Read baseline-update mode from the environment at CALL time.

    This must never be an import-time constant: the negative control
    (test_recipe_golden_gate_rejects_pixel_regression) disables update mode via
    monkeypatch.delenv, which can only work if every consumer re-reads the
    environment when it runs. An import-time constant silently turned the
    rejecting gate into a copy-over-the-golden write during baseline refreshes.
    """
    return os.environ.get("FORGE3D_UPDATE_RECIPE_GOLDENS") == "1"


def _update_certificates_enabled() -> bool:
    """Permit an explicit signed-certificate rotation without touching pixels."""
    return os.environ.get("FORGE3D_UPDATE_RECIPE_CERTIFICATES") == "1"


def _production_signing_required() -> bool:
    return os.environ.get("FORGE3D_REQUIRE_PRODUCTION_SIGNING") == "1"
ARTIFACT_DIR = (
    Path(os.environ["FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR"])
    if os.environ.get("FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR")
    else None
)
SSIM_MIN = 0.995
MEAN_ABS_MAX = 2.0


def _recipe_golden_variant() -> str | None:
    """Return the explicitly selected backend baseline variant, if any."""
    return selected_golden_variant(
        "FORGE3D_RECIPE_GOLDEN_VARIANT", implicit_metal=False
    )


@dataclass(frozen=True)
class RecipeGolden:
    scene_id: str
    family: str
    build: Callable[[Path], f3d.MapScene]
    expected_features: tuple[str, ...]
    bit_depth: int = 8
    ssim_min: float = SSIM_MIN
    mean_abs_max: float = MEAN_ABS_MAX

    @property
    def canonical_golden_path(self) -> Path:
        return GOLDEN_DIR / f"{self.scene_id}.png"

    @property
    def golden_path(self) -> Path:
        variant = _recipe_golden_variant()
        if variant is None:
            return self.canonical_golden_path
        return GOLDEN_DIR / f"{self.scene_id}.{variant}.png"

    @property
    def command(self) -> str:
        return f"pytest tests/test_recipe_goldens.py -k {self.scene_id}"


def _heightmap(size: int = 8) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return (0.25 * xx + 0.75 * yy).astype(np.float32)


def _water_heightmap(size: int = 8) -> np.ndarray:
    dem = np.ones((size, size), dtype=np.float32)
    dem[2 : size - 2, 2 : size - 2] = 0.0
    return dem


def _write_rgba_png(path: Path, rgba: np.ndarray) -> Path:
    f3d.numpy_to_png(path, np.ascontiguousarray(rgba, dtype=np.uint8))
    return path


def _arabic_font_path() -> Path:
    path = ROOT / "assets" / "fonts" / "NotoSansArabic-subset.ttf"
    assert path.is_file(), f"Packaged Arabic test font is missing: {path}"
    return path


def _pad4(data: bytes, pad: bytes = b" ") -> bytes:
    return data + pad * ((4 - (len(data) % 4)) % 4)


def _write_pnts_fixture(path: Path) -> Path:
    positions_array = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [6.0, 3.0, 0.0],
            [8.0, 4.0, 0.0],
            [10.0, 5.0, 0.0],
            [3.0, 6.0, 0.0],
            [5.0, 7.0, 0.0],
            [7.0, 8.0, 0.0],
        ],
        dtype="<f4",
    )
    colors = np.asarray(
        [
            [244, 63, 94],
            [249, 115, 22],
            [234, 179, 8],
            [34, 197, 94],
            [20, 184, 166],
            [14, 165, 233],
            [99, 102, 241],
            [168, 85, 247],
            [236, 72, 153],
        ],
        dtype=np.uint8,
    )
    positions = positions_array.tobytes()
    feature_table = _pad4(
        json.dumps(
            {
                "POINTS_LENGTH": int(len(positions_array)),
                "POSITION": {"byteOffset": 0},
                "RGB": {"byteOffset": len(positions)},
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        b" ",
    )
    feature_binary = _pad4(positions + colors.tobytes(), b"\x00")
    byte_length = 28 + len(feature_table) + len(feature_binary)
    path.write_bytes(
        b"".join(
            [
                b"pnts",
                struct.pack("<IIIIII", 1, byte_length, len(feature_table), len(feature_binary), 0, 0),
                feature_table,
                feature_binary,
            ]
        )
    )
    return path


def _material_map_assets(tmp_path: Path) -> dict[str, str]:
    size = 64
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)

    normal = np.zeros((size, size, 4), dtype=np.uint8)
    normal[..., 0] = np.clip(128.0 + 92.0 * np.sin(xx * np.pi * 10.0), 0.0, 255.0).astype(np.uint8)
    normal[..., 1] = np.clip(128.0 + 56.0 * np.cos(yy * np.pi * 8.0), 0.0, 255.0).astype(np.uint8)
    normal[..., 2] = 208
    normal[..., 3] = 255

    roughness = np.zeros((size, size, 4), dtype=np.uint8)
    rough = np.clip(54.0 + 174.0 * (0.5 + 0.5 * np.sin((xx + yy) * np.pi * 7.0)), 0.0, 255.0).astype(np.uint8)
    roughness[..., :3] = rough[..., None]
    roughness[..., 3] = 255

    mask = np.zeros((size, size, 4), dtype=np.uint8)
    rings = ((np.floor(xx * 8.0) + np.floor(yy * 8.0)) % 2.0).astype(np.uint8) * 255
    mask[..., :3] = rings[..., None]
    mask[..., 3] = 255

    return {
        "normal_path": str(_write_rgba_png(tmp_path / "material-normal.png", normal)),
        "roughness_path": str(_write_rgba_png(tmp_path / "material-roughness.png", roughness)),
        "mask_path": str(_write_rgba_png(tmp_path / "material-mask.png", mask)),
    }


def _base_scene(
    tmp_path: Path,
    scene_id: str,
    *,
    layers: list[object] | None = None,
    width: int = 96,
    height: int = 64,
    samples: int = 1,
    aovs: tuple[str, ...] = (),
    hdr: bool = False,
    bit_depth: int = 8,
    map_furniture: f3d.MapFurnitureLayer | None = None,
    terrain_metadata: dict[str, object] | None = None,
    lighting_settings: dict[str, object] | None = None,
    heightmap: np.ndarray | None = None,
) -> f3d.MapScene:
    data = _heightmap() if heightmap is None else np.asarray(heightmap, dtype=np.float32)
    metadata = {
        "source_id": f"{scene_id}-dem",
        "width": int(data.shape[1]),
        "height": int(data.shape[0]),
        "asset_status": "fixture",
        "bounds": (-122.5, 46.6, -121.9, 47.0),
    }
    if terrain_metadata:
        metadata.update(terrain_metadata)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=data,
            crs="EPSG:32610",
            metadata=metadata,
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=800.0, azimuth_deg=35.0),
        lighting=f3d.LightingPreset(name="rainier_showcase", intensity=1.15, settings=lighting_settings),
        output=f3d.OutputSpec(
            width=width,
            height=height,
            format="png",
            path=str(tmp_path / f"{scene_id}.png"),
            samples=samples,
            aovs=aovs,
            hdr=hdr,
            bit_depth=bit_depth,
        ),
        layers=layers or [],
        map_furniture=map_furniture,
        reproducibility_profile=f3d.ReproducibilityProfile(seed=2026),
    )


def _terrain_raster(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_terrain_raster",
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.tif",
                crs="EPSG:32610",
                opacity=0.72,
                metadata={"source_id": "ortho-fixture", "width": 8, "height": 8, "asset_status": "fixture"},
            )
        ],
    )


def _vector_labels(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_vector_labels",
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {"id": "a", "geometry": {"type": "LineString", "coordinates": [(0.1, 0.2), (0.9, 0.75)]}},
                    {"id": "b", "geometry": {"type": "LineString", "coordinates": [(0.12, 0.78), (0.88, 0.28)]}},
                ],
                width_px=4,
                line_cap="round",
                line_join="round",
                dash_array=[10, 5],
                style={"version": 8, "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#f9fafb"}}]},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {"id": "summit", "text": "Summit", "geometry": {"type": "Point", "coordinates": (34.0, 20.0, 0.0)}},
                    {"id": "trail", "text": "Trail", "geometry": {"type": "Point", "coordinates": (68.0, 44.0, 0.0)}},
                ],
                glyph_atlas={"glyphs": sorted(set("SummitTrail"))},
            ),
        ],
    )


def _label_halo_depth(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_label_halo_depth",
        width=128,
        height=80,
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "front",
                        "text": "Front",
                        "geometry": {"type": "Point", "coordinates": (28.0, 26.0, 0.25)},
                        "typography": {
                            "color": [1.0, 1.0, 1.0, 1.0],
                            "halo_color": [0.02, 0.02, 0.02, 0.92],
                            "halo_width_px": 3.0,
                        },
                    },
                    {
                        "id": "summit",
                        "text": "Summit",
                        "geometry": {"type": "Point", "coordinates": (72.0, 50.0, 0.20)},
                        "typography": {
                            "color": [0.12, 0.16, 0.18, 1.0],
                            "halo_color": [1.0, 1.0, 1.0, 0.88],
                            "halo_width_px": 2.0,
                        },
                    },
                    {
                        "id": "behind",
                        "text": "Behind",
                        "geometry": {"type": "Point", "coordinates": (28.0, 26.0, 0.85)},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("FrontSummitBehind"))},
                occlusion="terrain",
                metadata={
                    "depth_occlusion": {
                        "image": np.full((8, 8), 0.5, dtype=np.float32).tolist(),
                        "source": "recipe_depth_aov",
                        "bias": 0.0,
                    }
                },
            )
        ],
    )


def _label_arabic_joining(tmp_path: Path) -> f3d.MapScene:
    from forge3d.text_atlas import BakedAtlas, save_atlas

    shaped_glyphs = ["\ufe8e", "\ufe92", "\ufea3", "\ufeae", "\ufee3"]
    charset = sorted(set("مرحبا" + "".join(shaped_glyphs)))
    font_path = _arabic_font_path()
    shaped = f3d.text.shape("مرحبا", [str(font_path)], 34.0)
    baked = f3d.text.bake_msdf_atlas([str(font_path)], shaped, 34.0, 8.0, 4)
    metrics = dict(baked["metrics"])
    metrics["font_source"] = str(font_path)
    metrics["font_sources"] = [str(font_path)]
    atlas = BakedAtlas(image=np.asarray(baked["image"], dtype=np.uint8), metrics=metrics)
    atlas_png, atlas_json = save_atlas(
        atlas,
        tmp_path / "arabic_joining_atlas.png",
        tmp_path / "arabic_joining_atlas.json",
    )
    glyph_atlas = {
        "glyphs": charset,
        "image_path": str(atlas_png),
        "metrics_path": str(atlas_json),
        "source_path": str(atlas_json),
    }
    glyph_atlas["font_path"] = str(font_path)
    return _base_scene(
        tmp_path,
        "mapscene_label_arabic_joining",
        width=128,
        height=80,
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "arabic-city",
                        "text": "مرحبا",
                        "geometry": {"type": "Point", "coordinates": (52.0, 34.0, 0.0)},
                        "typography": {
                            "color": [1.0, 1.0, 1.0, 1.0],
                            "halo_color": [0.0, 0.0, 0.0, 0.9],
                            "halo_width_px": 3.0,
                        },
                    }
                ],
                glyph_atlas=glyph_atlas,
            )
        ],
    )


def _label_occlusion_ridge(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_label_occlusion_ridge",
        width=128,
        height=80,
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "front",
                        "text": "Front",
                        "geometry": {"type": "Point", "coordinates": (34.0, 26.0, 0.0)},
                        "typography": {
                            "color": [1.0, 1.0, 1.0, 1.0],
                            "halo_color": [0.02, 0.02, 0.02, 0.92],
                            "halo_width_px": 3.0,
                        },
                    },
                    {
                        "id": "behind-ridge",
                        "text": "Hidden",
                        "geometry": {"type": "Point", "coordinates": (34.0, 26.0, 0.95)},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("FrontHidden"))},
                occlusion="terrain",
                # SUTURA: depth occlusion culls against a depth source that is
                # serialized with the recipe, never a live GPU frame, so the
                # compiled plan (and this golden) reproduce after a bundle
                # round-trip.
                metadata={
                    "depth_occlusion": {
                        "image": np.full((16, 16), 0.5, dtype=np.float32).tolist(),
                        "source": "serialized_depth_proxy",
                        "bias": 0.0,
                    }
                },
            )
        ],
    )


def _vector_stroke_quality(
    tmp_path: Path,
    *,
    scene_id: str = "mapscene_vector_stroke_quality",
    width: int = 128,
    height: int = 80,
) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        scene_id,
        width=width,
        height=height,
        layers=[
            f3d.VectorOverlay(
                layer_id="cartography",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "hairpin",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [(0.06, 0.74), (0.30, 0.18), (0.52, 0.74), (0.74, 0.22), (0.94, 0.74)],
                        },
                    },
                    {
                        "id": "dashed-boundary",
                        "geometry": {"type": "LineString", "coordinates": [(0.08, 0.10), (0.92, 0.10)]},
                    },
                    {
                        "id": "park-with-hole",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [(0.10, 0.32), (0.38, 0.32), (0.38, 0.62), (0.10, 0.62), (0.10, 0.32)],
                                [(0.19, 0.41), (0.30, 0.41), (0.30, 0.53), (0.19, 0.53), (0.19, 0.41)],
                            ],
                        },
                    },
                ],
                width_px=6,
                line_cap="round",
                line_join="round",
                dash_array=[12, 7],
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "cartography",
                            "type": "line",
                            "paint": {"line-color": "#f8fafc", "line-width": 6, "fill-color": "#2563eb"},
                        }
                    ],
                },
            )
        ],
    )


def _vector_stroke_quality_4x(tmp_path: Path) -> f3d.MapScene:
    return _vector_stroke_quality(
        tmp_path,
        scene_id="mapscene_vector_stroke_quality_4x",
        width=256,
        height=160,
    )


def _choropleth(tmp_path: Path) -> f3d.MapScene:
    values = np.asarray([12.0, 28.0, 57.0, 83.0], dtype=np.float32)
    result = f3d.thematic.classify(values, scheme="quantile", k=4)
    classes = result["classes"]
    palette = {
        1: "#edf8fb",
        2: "#b2e2e2",
        3: "#66c2a4",
        4: "#238b45",
    }
    features = []
    for idx, cls in enumerate(classes.tolist()):
        x0 = 0.10 + (idx % 2) * 0.42
        y0 = 0.14 + (idx // 2) * 0.38
        x1 = x0 + 0.32
        y1 = y0 + 0.28
        features.append(
            {
                "id": f"zone-{idx}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]],
                },
                "properties": {"class": int(cls), "value": float(values[idx])},
            }
        )
    return _base_scene(
        tmp_path,
        "mapscene_thematic_choropleth",
        width=128,
        height=88,
        layers=[
            f3d.VectorOverlay(
                layer_id="classified-zones",
                crs="EPSG:32610",
                features=features,
                width_px=2,
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "zones-fill",
                            "type": "fill",
                            "paint": {
                                "fill-color": [
                                    "match",
                                    ["get", "class"],
                                    1,
                                    palette[1],
                                    2,
                                    palette[2],
                                    3,
                                    palette[3],
                                    palette[4],
                                ],
                                "fill-opacity": 0.84,
                            },
                        },
                        {
                            "id": "zones-outline",
                            "type": "line",
                            "paint": {"line-color": "#0f172a", "line-width": 2},
                        },
                    ],
                },
            )
        ],
    )


def _offline_aovs(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_offline_aovs",
        samples=4,
        aovs=("albedo", "normal", "depth"),
        hdr=True,
    )


def _buildings(tmp_path: Path) -> f3d.MapScene:
    roof_types = ("flat", "gabled", "hipped", "pyramidal")
    features = []
    for idx, roof_type in enumerate(roof_types):
        x0 = 0.08 + idx * 0.22
        x1 = x0 + 0.15
        y0 = 0.24 if idx % 2 == 0 else 0.34
        y1 = y0 + 0.30
        features.append(
            {
                "id": f"b-{roof_type}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]],
                },
                "properties": {
                    "height": 22.0 + idx * 7.0,
                    "roof:shape": roof_type,
                    "building:material": "brick" if idx % 2 else "concrete",
                },
            }
        )
    building = f3d.MapSceneBuildingLayer(
        layer_id="buildings",
        source={"source_id": "inline-buildings", "asset_status": "fixture"},
        support_level="supported",
        geometry_count=len(features),
        material_status="scalar_pbr_underdeveloped",
        features=features,
        metadata={"source_id": "inline-buildings", "asset_status": "fixture"},
    )
    return _base_scene(tmp_path, "mapscene_buildings", layers=[building], width=128, height=88)


def _screen_space_contact(tmp_path: Path) -> f3d.MapScene:
    scene = _buildings(tmp_path)
    scene.recipe.lighting = f3d.LightingPreset(
        name="outdoor_sun",
        intensity=1.1,
        settings={
            "screen_space": {
                "ssao": {"enabled": True, "radius": 2.6, "intensity": 1.35},
                "ssgi": {"enabled": True, "intensity": 0.45},
                "taa": {"enabled": True, "temporal_alpha": 0.18},
            }
        },
    )
    scene.recipe.output.path = str(tmp_path / "mapscene_screen_space_contact.png")
    return scene


def _screen_space_reflection(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_screen_space_reflection",
        width=128,
        height=80,
        heightmap=_water_heightmap(),
        terrain_metadata={"water": {"enabled": True, "auto_mask": True, "level": 0.1, "slope_threshold": 1.0}},
        lighting_settings={
            "water": {"enabled": True, "auto_mask": True, "level": 0.1, "slope_threshold": 1.0},
            "screen_space": {
                "ssr": {"enabled": True, "intensity": 0.85},
            },
        },
    )


def _textured_gltf_landmark(tmp_path: Path) -> f3d.MapScene:
    from tests.test_gltf_import import _write_material_glb

    gltf_path = tmp_path / "textured-landmark.glb"
    _write_material_glb(gltf_path)
    texture = np.zeros((16, 16, 4), dtype=np.uint8)
    texture[..., 0] = np.linspace(40, 230, 16, dtype=np.uint8)[None, :]
    texture[..., 1] = np.linspace(230, 60, 16, dtype=np.uint8)[:, None]
    texture[..., 2] = 120
    texture[..., 3] = 255
    texture[::2, :, 2] = 220
    texture[:, ::2, 0] = 245
    texture_path = _write_rgba_png(tmp_path / "textured-landmark-albedo.png", texture)
    layer = f3d.MapSceneBuildingLayer(
        layer_id="textured-landmark",
        source={"path": str(gltf_path), "source_format": "gltf"},
        support_level="supported",
        geometry_count=1,
        material_status="textured_pbr",
        metadata={
            "source_id": "textured-landmark",
            "gltf_path": str(gltf_path),
            "screen_rect": [0.34, 0.16, 0.68, 0.70],
            "textured_materials": [
                {
                    "material_id": "mat_red",
                    "object_id": "landmark",
                    "albedo_texture": str(texture_path),
                    "texture_format": "png",
                    "uv_available": True,
                }
            ],
        },
    )
    return _base_scene(
        tmp_path,
        "mapscene_textured_gltf_landmark",
        layers=[layer],
        width=128,
        height=88,
        lighting_settings={"screen_space": {"ssao": {"enabled": True, "radius": 1.8, "intensity": 0.65}}},
    )


def _furniture(tmp_path: Path) -> f3d.MapScene:
    furniture = f3d.MapFurnitureLayer(
        title="Recipe Golden",
        legend={"items": [{"label": "Forest", "color": "#2f855a"}, {"label": "Snow", "color": "#f8fafc"}]},
        scale_bar={"length_m": 1000, "units": "km", "location": "lower_left", "geodesic": True},
        north_arrow={"location": "upper_right", "size": 34},
        graticule={
            "bounds": (-122.5, 46.6, -121.9, 47.0),
            "projected_bounds": (-122.5, 46.6, -121.9, 47.0),
            "target_crs": "EPSG:4326",
            "interval_deg": 0.2,
            "include_labels": True,
        },
    )
    return _base_scene(tmp_path, "mapscene_furniture_graticule", map_furniture=furniture, width=128, height=88)


def _alignment(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_alignment_utm",
        layers=[
            f3d.VectorOverlay(
                layer_id="aligned-boundary",
                crs="EPSG:4326",
                features=[{"id": "bbox", "geometry": {"type": "LineString", "coordinates": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]}}],
                metadata={"crs_policy": "explicit_transform", "crs_transform": "fixture-transform"},
                width_px=3,
            )
        ],
    )


def _material_maps(tmp_path: Path) -> f3d.MapScene:
    material_maps = _material_map_assets(tmp_path)
    return _base_scene(
        tmp_path,
        "mapscene_material_maps",
        width=128,
        height=80,
        terrain_metadata={"material_maps": material_maps},
        lighting_settings={
            "albedo_mode": "material",
            "colormap_strength": 0.0,
            "exaggeration": 1.35,
        },
    )


def _clipmap_large_region(tmp_path: Path) -> f3d.MapScene:
    size = 32
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dem = (0.35 * np.sin(xx * np.pi * 2.0) + 0.22 * np.cos(yy * np.pi * 3.0)).astype(np.float32)
    return _base_scene(
        tmp_path,
        "mapscene_clipmap_large_region",
        width=128,
        height=80,
        heightmap=dem,
        terrain_metadata={
            "clipmap": {
                "enabled": True,
                "levels": 4,
                "ring_resolution": 32,
                "terrain_extent_m": 100_000.0,
                "max_resident_height_bytes": 4 * 32 * 32 * 4,
            }
        },
        lighting_settings={"exaggeration": 1.2},
    )


def _auto_water(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_auto_water",
        width=128,
        height=80,
        heightmap=_water_heightmap(),
        terrain_metadata={"water": {"enabled": True, "auto_mask": True, "level": 0.1, "slope_threshold": 1.0}},
        lighting_settings={"water": {"enabled": True, "auto_mask": True, "level": 0.1, "slope_threshold": 1.0}},
    )


def _cloud_shadows(tmp_path: Path) -> f3d.MapScene:
    dem = np.zeros((16, 16), dtype=np.float32)
    dem[5:11, 5:11] = 0.35
    return _base_scene(
        tmp_path,
        "mapscene_cloud_shadows",
        width=128,
        height=80,
        heightmap=dem,
        terrain_metadata={
            "width": 16,
            "height": 16,
            "source_id": "cloud-shadow-dem",
            "clouds": {
                "enabled": True,
                "shadows_enabled": True,
                "coverage": 0.72,
                "density": 0.48,
                "shadow_strength": 0.38,
                "quality": "high",
            },
        },
    )


def _tiles3d_points(tmp_path: Path) -> f3d.MapScene:
    pnts_path = _write_pnts_fixture(tmp_path / "points.pnts")
    tileset_path = tmp_path / "tileset.json"
    tileset_path.write_text(
        json.dumps(
            {
                "asset": {"version": "1.0"},
                "geometricError": 0.0,
                "root": {
                    "boundingVolume": {"sphere": [5.0, 4.0, 0.0, 8.0]},
                    "geometricError": 0.0,
                    "content": {"uri": pnts_path.name},
                },
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return _base_scene(
        tmp_path,
        "mapscene_tiles3d_points",
        width=128,
        height=80,
        layers=[
            f3d.Tiles3DLayer.from_tileset_json(
                tileset_path,
                layer_id="fixture-pnts-tileset",
                metadata={
                    "bounds": [0.0, 0.0, 10.0, 8.0],
                    "point_size": 5.0,
                    "camera_position": [5.0, 4.0, 25.0],
                    "shading": "edl",
                    "edl_strength": 2.0,
                    "edl_radius_px": 2.0,
                },
            )
        ],
    )


def _copc_points(tmp_path: Path) -> f3d.MapScene:
    from tests.test_pointcloud_gpu_integration import _write_tiny_copc

    copc_path = tmp_path / "tiny.copc.laz"
    _write_tiny_copc(str(copc_path))
    return _base_scene(
        tmp_path,
        "mapscene_copc_points",
        width=128,
        height=80,
        layers=[
            f3d.PointCloudLayer(
                layer_id="fixture-copc-points",
                path=copc_path,
                crs="EPSG:32610",
                point_count=2,
                metadata={
                    "bounds": [101.0, 202.0, 101.1, 202.1],
                    "point_budget": 2,
                    "point_size": 6.0,
                    "shading": "edl",
                    "edl_strength": 2.0,
                    "edl_radius_px": 2.0,
                },
            )
        ],
    )


def _png16_color(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(tmp_path, "mapscene_png16_color", bit_depth=16, width=80, height=48)


RECIPE_GOLDENS = (
    RecipeGolden("mapscene_terrain_raster", "terrain_raster", _terrain_raster, ("mapscene.render_png", "mapscene.render_backend")),
    RecipeGolden("mapscene_vector_labels", "labels_vectors", _vector_labels, ("mapscene.vector_composite", "mapscene.label_composite")),
    RecipeGolden("mapscene_label_halo_depth", "labels_depth_occlusion", _label_halo_depth, ("mapscene.label_composite",)),
    RecipeGolden(
        "mapscene_label_arabic_joining",
        "labels_complex_scripts",
        _label_arabic_joining,
        ("mapscene.label_composite",),
        ssim_min=0.990,
    ),
    RecipeGolden("mapscene_label_occlusion_ridge", "labels_depth_occlusion", _label_occlusion_ridge, ("mapscene.label_composite",)),
    RecipeGolden("mapscene_vector_stroke_quality", "vector_stroke_quality", _vector_stroke_quality, ("mapscene.vector_composite",)),
    RecipeGolden("mapscene_vector_stroke_quality_4x", "vector_stroke_quality", _vector_stroke_quality_4x, ("mapscene.vector_composite",)),
    RecipeGolden("mapscene_thematic_choropleth", "thematic_choropleth", _choropleth, ("mapscene.vector_composite",)),
    RecipeGolden("mapscene_offline_aovs", "offline_accumulation", _offline_aovs, ("mapscene.offline_accumulation", "mapscene.aov_export")),
    RecipeGolden("mapscene_buildings", "buildings", _buildings, ("mapscene.building_composite", "mapscene.building_gpu_mesh_composite")),
    RecipeGolden("mapscene_screen_space_contact", "screen_space_effects", _screen_space_contact, ("mapscene.screen_space", "mapscene.ssao", "mapscene.ssgi", "mapscene.taa")),
    RecipeGolden("mapscene_screen_space_reflection", "screen_space_effects", _screen_space_reflection, ("mapscene.screen_space", "mapscene.ssr")),
    RecipeGolden("mapscene_textured_gltf_landmark", "gltf_textured_assets", _textured_gltf_landmark, ("gltf.textured_mapscene_render", "buildings.textured_pbr")),
    RecipeGolden("mapscene_furniture_graticule", "map_furniture", _furniture, ("mapscene.furniture_composite",)),
    RecipeGolden("mapscene_alignment_utm", "alignment_crs", _alignment, ("mapscene.vector_composite",)),
    RecipeGolden("mapscene_material_maps", "terrain_materials", _material_maps, ("mapscene.render_png",)),
    RecipeGolden("mapscene_clipmap_large_region", "clipmap_large_region", _clipmap_large_region, ("terrain.clipmap_planner", "terrain.clipmap_indexed", "terrain.clipmap_bounded_memory")),
    RecipeGolden("mapscene_auto_water", "water_masks", _auto_water, ("mapscene.render_png", "mapscene.water_mask")),
    RecipeGolden("mapscene_cloud_shadows", "cloud_shadows", _cloud_shadows, ("mapscene.render_png", "mapscene.cloud_shadows")),
    RecipeGolden(
        "mapscene_tiles3d_points",
        "point_cloud_tiles",
        _tiles3d_points,
        ("point_cloud.mapscene_render", "tiles3d.mapscene_render", "point_cloud.edl"),
    ),
    RecipeGolden(
        "mapscene_copc_points",
        "point_cloud_tiles",
        _copc_points,
        ("point_cloud.mapscene_render", "point_cloud.edl"),
    ),
    RecipeGolden("mapscene_png16_color", "output_color", _png16_color, ("mapscene.render_png_16bit",), bit_depth=16),
)


def _write_failure_artifacts(spec: RecipeGolden, actual: np.ndarray, expected: np.ndarray) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    diff = np.abs(actual[..., :3].astype(np.int16) - expected[..., :3].astype(np.int16)).astype(np.uint8)
    diff_rgba = np.concatenate([diff, np.full(diff.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    f3d.numpy_to_png(ARTIFACT_DIR / f"{spec.scene_id}_actual.png", actual)
    f3d.numpy_to_png(ARTIFACT_DIR / f"{spec.scene_id}_expected.png", expected)
    f3d.numpy_to_png(ARTIFACT_DIR / f"{spec.scene_id}_diff.png", diff_rgba)


def _assert_matches_golden(spec: RecipeGolden, actual_path: Path) -> None:
    if _update_goldens_enabled():
        spec.golden_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual_path, spec.golden_path)
        return

    assert spec.golden_path.exists(), (
        f"Missing recipe golden {spec.golden_path}. "
        "Regenerate with FORGE3D_UPDATE_RECIPE_GOLDENS=1."
    )
    actual = f3d.png_to_numpy(actual_path)
    expected = f3d.png_to_numpy(spec.golden_path)
    assert actual.shape == expected.shape
    mean_abs = float(np.mean(np.abs(actual[..., :3].astype(np.float32) - expected[..., :3].astype(np.float32))))
    score = ssim(actual[..., :3], expected[..., :3], data_range=255.0)
    if score < spec.ssim_min or mean_abs > spec.mean_abs_max:
        _write_failure_artifacts(spec, actual, expected)
    assert score >= spec.ssim_min, f"{spec.scene_id} SSIM too low: {score:.6f}"
    assert mean_abs <= spec.mean_abs_max, f"{spec.scene_id} mean absolute difference too high: {mean_abs:.4f}"


def _certificate_refresh_without_backend_baseline(spec: RecipeGolden) -> bool:
    """Whether a certificate-only rotation lacks this backend's pixel fixture.

    The protected pixel gate runs before this path.  Certificate rotation still
    renders every catalog scene, but cannot compare an invented backend PNG.
    """
    return _update_certificates_enabled() and not spec.golden_path.exists()


def _committed_cert_path(spec: RecipeGolden) -> Path:
    return CERT_DIR / f"{spec.scene_id}.json"


def _assert_certificate_is_clean(cert: dict) -> None:
    assert cert.get("degradations") == [], (
        "Certificate refresh must be degradation-free; resolve the missing "
        f"capability or render-path fallback before signing: {cert.get('degradations')!r}"
    )


def _clear_degradation_sinks() -> None:
    """Isolate each scene's certificate: reset the process-global native and
    Python degradation sinks so a per-scene cert reflects only that scene."""
    from forge3d import _degradation

    _degradation.clear()
    try:
        from forge3d._forge3d import clear_native_degradations

        clear_native_degradations()
    except Exception:
        # Native sink reset is best-effort; the Python sink is the one that
        # would otherwise accumulate across scenes in one pytest process.
        pass


def _emit_or_verify_certificate(spec: RecipeGolden) -> None:
    """Emit (UPDATE mode) or verify (normal GPU mode) the committed signed
    certificate for ``spec``.

    The certificate reflects the LAST in-process native render, so this must be
    called immediately after this scene's ``render()`` and golden match.

    UPDATE mode: sign a fresh certificate with the configured seed, write it to
    ``tests/golden/certificates/<scene_id>.json``, and (first scene only) write
    the dev public key hex to ``signing.pub``.

    Normal mode: assert the committed cert exists and verifies against
    ``signing.pub``, that its ``degradations`` are empty, and that the FRESH
    render's ``wgsl_module_hashes`` match the committed cert's — the load-bearing
    golden-gate check that ties committed certs to the current WGSL sources.
    """
    from forge3d import certificate as _certificate
    from forge3d.diagnostics import render_certificate

    cert = render_certificate()  # signed; reflects the last completed render
    cert_path = _committed_cert_path(spec)

    if _update_certificates_enabled():
        _assert_certificate_is_clean(cert)
        CERT_DIR.mkdir(parents=True, exist_ok=True)
        _certificate.write_certificate(cert, cert_path)
        pubkey_hex = cert["signature"]["pubkey"]
        if not SIGNING_PUB_PATH.exists():
            SIGNING_PUB_PATH.write_text(pubkey_hex + "\n", encoding="utf-8")
        return

    assert cert_path.exists(), (
        f"Missing recipe golden certificate {cert_path}. "
        "Regenerate with FORGE3D_UPDATE_RECIPE_GOLDENS=1."
    )
    assert SIGNING_PUB_PATH.exists(), (
        f"Missing signing public key {SIGNING_PUB_PATH}. "
        "Regenerate with FORGE3D_UPDATE_RECIPE_GOLDENS=1."
    )
    pubkey = SIGNING_PUB_PATH.read_text(encoding="utf-8").strip()
    committed = json.loads(cert_path.read_text(encoding="utf-8"))
    assert (committed.get("signature") or {}).get("pubkey") == pubkey, (
        f"{cert_path} was not signed by the pinned production public key"
    )
    assert _certificate.verify(cert_path, pubkey) is True, (
        f"Committed certificate {cert_path} failed Ed25519 verification against "
        f"{SIGNING_PUB_PATH}."
    )

    if _production_signing_required():
        assert os.environ.get("FORGE3D_CERT_SIGNING_KEY"), (
            "protected golden lane requires FORGE3D_CERT_SIGNING_KEY"
        )
        assert (cert.get("signature") or {}).get("pubkey") == pubkey, (
            "fresh golden certificate was not signed by the pinned production key"
        )
        assert _certificate.verify(cert, pubkey) is True
    assert committed.get("degradations") == [], (
        f"{spec.scene_id} committed certificate records degradations "
        f"{committed.get('degradations')!r}; a clean golden must degrade nothing. "
        "Investigate the fallback before regenerating."
    )

    fresh_hashes = (cert.get("engine") or {}).get("wgsl_module_hashes") or {}
    committed_hashes = (committed.get("engine") or {}).get("wgsl_module_hashes") or {}
    # The shader-hash registry is process-lifetime: other GPU tests running
    # earlier in the same pytest process may register EXTRA modules (e.g. the
    # minimal terrain pipeline). The tamper gate therefore checks that every
    # module recorded in the committed certificate still exists with an
    # identical hash — extra fresh entries are fine, changed or missing ones
    # are not.
    missing = sorted(set(committed_hashes) - set(fresh_hashes))
    assert not missing, (
        f"WGSL modules {missing} named in the committed certificate were never "
        "compiled by this render; regenerate with FORGE3D_UPDATE_RECIPE_GOLDENS=1 "
        "after verifying the pixel goldens"
    )
    changed = {
        label: (committed_hashes[label], fresh_hashes[label])
        for label in committed_hashes
        if fresh_hashes[label] != committed_hashes[label]
    }
    assert not changed, (
        "WGSL source changed since golden certificates were generated; regenerate "
        "with FORGE3D_UPDATE_RECIPE_GOLDENS=1 after verifying the pixel goldens: "
        f"{changed}"
    )


def test_recipe_golden_manifest_catalog_has_required_coverage() -> None:
    families = {spec.family for spec in RECIPE_GOLDENS}
    assert len(RECIPE_GOLDENS) >= 8
    assert len(families) >= 5
    assert {"terrain_raster", "labels_vectors", "offline_accumulation", "buildings", "map_furniture"} <= families


def test_recipe_golden_catalog_links_docs_gallery() -> None:
    gallery = (ROOT / "docs" / "gallery" / "index.md").read_text(encoding="utf-8")
    for spec in RECIPE_GOLDENS:
        assert spec.scene_id in gallery
        assert str(spec.canonical_golden_path.relative_to(ROOT)).replace("\\", "/") in gallery
        assert spec.command in gallery


def test_arabic_golden_recipe_accepts_packaged_font_and_rasterizes_coverage(
    tmp_path: Path,
) -> None:
    scene = _label_arabic_joining(tmp_path)
    compiled = scene.compile_plan()
    plan = compiled.label_plans["labels"]
    assert not plan.rejected
    assert len(plan.accepted) == 1
    assert plan.accepted[0].typography["glyph_ids"]
    shaped = f3d.text.shape("مرحبا", [str(_arabic_font_path())], 34.0)
    mask = f3d.text.rasterize_shaped_run(shaped, 128, 80, origin=(40.0, 48.0))
    assert np.count_nonzero(mask > 0.0) > 20
    assert np.any((mask > 0.0) & (mask < 1.0))


def test_update_mode_is_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Baseline-update mode must be a call-time decision, not an import-time
    constant, or the negative control's delenv guard is dead code."""
    monkeypatch.setenv("FORGE3D_UPDATE_RECIPE_GOLDENS", "1")
    assert _update_goldens_enabled() is True
    monkeypatch.delenv("FORGE3D_UPDATE_RECIPE_GOLDENS")
    assert _update_goldens_enabled() is False


def test_certificate_update_mode_never_enables_pixel_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE3D_UPDATE_RECIPE_CERTIFICATES", "1")
    assert _update_certificates_enabled() is True
    assert _update_goldens_enabled() is False


def test_certificate_refresh_requires_a_real_render_but_not_an_absent_backend_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optional Metal diagnostic must never fabricate or write baselines."""
    monkeypatch.setenv("FORGE3D_UPDATE_RECIPE_CERTIFICATES", "1")
    monkeypatch.setenv("WGPU_BACKEND", "metal")
    monkeypatch.setenv("FORGE3D_RECIPE_GOLDEN_VARIANT", "metal")
    assert _certificate_refresh_without_backend_baseline(RECIPE_GOLDENS[0]) is False
    assert _certificate_refresh_without_backend_baseline(RECIPE_GOLDENS[1]) is True
    monkeypatch.delenv("FORGE3D_UPDATE_RECIPE_CERTIFICATES")
    assert _certificate_refresh_without_backend_baseline(RECIPE_GOLDENS[1]) is False


def test_certificate_refresh_rejects_capability_degradation() -> None:
    with pytest.raises(AssertionError, match="must be degradation-free"):
        _assert_certificate_is_clean(
            {"degradations": [{"kind": "capability_absent", "name": "timestamp_query"}]}
        )


def test_recipe_golden_variant_uses_an_explicit_backend_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = RECIPE_GOLDENS[0]
    monkeypatch.delenv("FORGE3D_RECIPE_GOLDEN_VARIANT", raising=False)
    assert spec.golden_path == spec.canonical_golden_path
    monkeypatch.setenv("WGPU_BACKEND", "metal")
    monkeypatch.setenv("FORGE3D_RECIPE_GOLDEN_VARIANT", "metal")
    assert spec.golden_path == GOLDEN_DIR / "mapscene_terrain_raster.metal.png"
    monkeypatch.setenv("WGPU_BACKEND", "vulkan")
    monkeypatch.setenv("FORGE3D_RECIPE_GOLDEN_VARIANT", "nvidia-vulkan")
    assert spec.golden_path == GOLDEN_DIR / "mapscene_terrain_raster.nvidia-vulkan.png"


def test_recipe_golden_gate_rejects_pixel_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This negative control must remain a rejecting gate even during an
    # intentional baseline-refresh run: simulate a refresh run by setting the
    # env first, then prove the control's delenv actually disables update mode
    # (call-time read) so the corrupted image is compared, not copied over the
    # committed golden.
    monkeypatch.setenv("FORGE3D_UPDATE_RECIPE_GOLDENS", "1")
    monkeypatch.delenv("FORGE3D_UPDATE_RECIPE_GOLDENS", raising=False)
    assert _update_goldens_enabled() is False
    spec = RECIPE_GOLDENS[0]
    assert spec.golden_path.exists()
    golden_bytes_before = spec.golden_path.read_bytes()
    corrupted = f3d.png_to_numpy(spec.golden_path).copy()
    corrupted[: corrupted.shape[0] // 2, : corrupted.shape[1] // 2, :3] = 255 - corrupted[
        : corrupted.shape[0] // 2,
        : corrupted.shape[1] // 2,
        :3,
    ]
    actual_path = tmp_path / "corrupted_recipe_golden.png"
    f3d.numpy_to_png(actual_path, corrupted)

    with pytest.raises(AssertionError):
        _assert_matches_golden(spec, actual_path)

    assert spec.golden_path.read_bytes() == golden_bytes_before, (
        "negative control must never write over the committed golden"
    )


def _render_recipe_golden_pixels(tmp_path: Path, spec: RecipeGolden) -> None:
    if not terrain_rendering_available():
        import pytest

        pytest.skip("Recipe goldens require a terrain-capable hardware-backed forge3d runtime")

    scene = spec.build(tmp_path)
    manifest = f3d.recipe_manifest(
        scene,
        golden_fixture_intent={
            "scene_id": spec.scene_id,
            "family": spec.family,
            "golden_path": str(spec.golden_path.relative_to(ROOT)).replace("\\", "/"),
            "command": spec.command,
            "backend": "gpu_terrain",
            "tolerance": {"ssim_min": spec.ssim_min, "mean_abs_max": spec.mean_abs_max},
        },
    )
    intent = manifest["golden_fixture_intent"]
    assert intent["scene_id"] == spec.scene_id
    assert intent["family"] == spec.family
    assert intent["backend"] == "gpu_terrain"

    _clear_degradation_sinks()
    report = scene.render()
    if nvidia_vulkan_golden_selected("FORGE3D_RECIPE_GOLDEN_VARIANT"):
        adapter_probe = f3d.device_probe("vulkan")
        assert_nvidia_vulkan_golden_adapter(
            "FORGE3D_RECIPE_GOLDEN_VARIANT", adapter_probe
        )
        if ARTIFACT_DIR is not None:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            (ARTIFACT_DIR / "recipe-render-adapter.json").write_text(
                json.dumps(adapter_probe, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    assert scene.last_render_backend == "gpu_terrain"
    for feature in spec.expected_features:
        assert report.supported_features[feature] == "supported"
    if spec.family == "buildings":
        metadata = scene.last_render_metadata or {}
        assert metadata["building_backend"] == "terrain_scatter_instanced_mesh"
        assert metadata["building_shadow_model"] == "terrain_csm_mesh_cast_receive"
    output_path = Path(scene.last_render_path or scene.recipe.output.path or "")
    assert output_path.exists()
    if spec.scene_id == "mapscene_label_arabic_joining":
        rendered = f3d.png_to_numpy(output_path)
        assert np.count_nonzero(np.max(rendered[..., :3], axis=-1) > 245) > 20
    if _certificate_refresh_without_backend_baseline(spec):
        assert spec.canonical_golden_path.exists(), (
            f"Certificate refresh requires the canonical baseline for {spec.scene_id}; "
            "it must never create a backend-specific pixel golden."
        )
    else:
        _assert_matches_golden(spec, output_path)


@pytest.mark.parametrize("spec", RECIPE_GOLDENS, ids=lambda item: item.scene_id)
def test_recipe_goldens_render_and_match(tmp_path, spec: RecipeGolden) -> None:
    """Render pixels and enforce the protected signed-certificate contract."""
    _render_recipe_golden_pixels(tmp_path, spec)
    _emit_or_verify_certificate(spec)


def test_nvidia_vulkan_recipe_pixel_golden_render_and_match(tmp_path: Path) -> None:
    """Verify the required NVIDIA pixel baseline without rotating certificates.

    Production certificate refresh is deliberately main-only. A candidate PR
    can prove its backend-specific pixels on physical hardware, but it must not
    bypass or rewrite the separately protected signed-certificate contract.
    """
    if not nvidia_vulkan_golden_selected("FORGE3D_RECIPE_GOLDEN_VARIANT"):
        pytest.skip("NVIDIA/Vulkan recipe pixel proof was not selected")
    spec = next(
        item for item in RECIPE_GOLDENS if item.scene_id == "mapscene_terrain_raster"
    )
    _render_recipe_golden_pixels(tmp_path, spec)
