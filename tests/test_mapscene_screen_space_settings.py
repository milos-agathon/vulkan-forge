from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
from forge3d.gis import derive_water_mask
from forge3d.terrain_params import CloudSettings, ScreenSpaceSettings, make_terrain_params_config


ROOT = Path(__file__).resolve().parents[1]


def _load_water_cloud_example():
    path = ROOT / "examples" / "mapscene_water_clouds.py"
    spec = importlib.util.spec_from_file_location("mapscene_water_clouds_example", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_terrain_render_params_decodes_screen_space_settings():
    config = make_terrain_params_config(
        size_px=(64, 64),
        render_scale=1.0,
        terrain_span=1.0,
        msaa_samples=1,
        z_scale=1.0,
        exposure=1.0,
        domain=(0.0, 1.0),
        screen_space=ScreenSpaceSettings(
            ssao_enabled=True,
            ssao_radius=2.5,
            ssao_intensity=1.25,
            ssgi_enabled=True,
            ssgi_intensity=0.75,
            ssr_enabled=True,
            ssr_intensity=0.5,
            taa_enabled=True,
            temporal_alpha=0.2,
        ),
    )

    params = f3d.TerrainRenderParams(config)
    got = params.screen_space_settings()

    assert got["enabled"] is True
    assert got["ssao_enabled"] is True
    assert got["ssao_radius"] == pytest.approx(2.5)
    assert got["ssao_intensity"] == pytest.approx(1.25)
    assert got["ssgi_enabled"] is True
    assert got["ssgi_intensity"] == pytest.approx(0.75)
    assert got["ssr_enabled"] is True
    assert got["ssr_intensity"] == pytest.approx(0.5)
    assert got["taa_enabled"] is True
    assert got["temporal_alpha"] == pytest.approx(0.2)


def test_mapscene_lighting_settings_reach_terrain_screen_space_params(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(
            name="outdoor_sun",
            settings={
                "screen_space": {
                    "ssao": {"enabled": True, "radius": 3.0, "intensity": 1.4},
                    "ssr": {"enabled": True, "intensity": 0.35},
                    "taa": {"enabled": True, "temporal_alpha": 0.25},
                }
            },
        ),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )
    calls: dict[str, object] = {}

    class FakeSession:
        def __init__(self, *, window=False):
            calls["session_window"] = window

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(_path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            return "ibl"

    class FakeParams:
        def __init__(self, config):
            calls["screen_space"] = config.screen_space

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, _session):
            pass

        def render_terrain_pbr_pom(self, **_kwargs):
            return FakeFrame()

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    scene.render(str(tmp_path / "screen-space-settings.png"))

    screen_space = calls["screen_space"]
    assert isinstance(screen_space, ScreenSpaceSettings)
    assert screen_space.enabled is True
    assert screen_space.ssao_enabled is True
    assert screen_space.ssao_radius == pytest.approx(3.0)
    assert screen_space.ssao_intensity == pytest.approx(1.4)
    assert screen_space.ssr_enabled is True
    assert screen_space.ssr_intensity == pytest.approx(0.35)
    assert screen_space.taa_enabled is True
    assert screen_space.temporal_alpha == pytest.approx(0.25)
    assert scene.last_render_metadata["terrain_main_pass_ms"] >= 0.0
    assert scene.last_render_metadata["timing_source"] == "python_perf_counter"


def test_mapscene_screen_space_postfx_modulates_output():
    heightmap = np.ones((8, 8), dtype=np.float32)
    heightmap[2:6, 2:6] = 0.0
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=heightmap,
            crs="EPSG:32610",
            metadata={
                "width": 8,
                "height": 8,
                "source_id": "inline-dem",
                "water": {"enabled": True, "auto_mask": True, "level": 0.1, "slope_threshold": 1.0},
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(
            name="daylight",
            settings={
                "screen_space": {
                    "ssao": {"enabled": True, "radius": 2.0, "intensity": 1.25},
                    "ssgi": {"enabled": True, "intensity": 0.7},
                    "ssr": {"enabled": True, "intensity": 0.8},
                    "taa": {"enabled": True, "temporal_alpha": 0.2},
                }
            },
        ),
        output=f3d.OutputSpec(width=48, height=48, format="png"),
    )
    base = np.full((48, 48, 4), 180, dtype=np.uint8)
    base[..., 3] = 255

    rendered, metadata = map_scene._apply_mapscene_screen_space(base, scene.recipe, heightmap)

    assert metadata["screen_space_backend"] == "mapscene_numpy_postfx"
    assert metadata["screen_space_effects"] == ["ssao", "ssgi", "ssr", "taa"]
    assert metadata["screen_space_ssao_intensity"] == pytest.approx(1.25)
    assert metadata["screen_space_ssr_intensity"] == pytest.approx(0.8)
    assert metadata["screen_space_taa_temporal_alpha"] == pytest.approx(0.2)
    assert not np.array_equal(rendered[..., :3], base[..., :3])


def test_mapscene_clipmap_planner_metadata_reaches_render_report(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((32, 32), dtype=np.float32),
            crs="EPSG:32610",
            metadata={
                "width": 32,
                "height": 32,
                "source_id": "large-region-dem",
                "clipmap": {
                    "enabled": True,
                    "levels": 4,
                    "ring_resolution": 32,
                    "terrain_extent_m": 100_000.0,
                    "max_resident_height_bytes": 4 * 32 * 32 * 4,
                },
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100_000.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )

    class FakeSession:
        def __init__(self, *, window=False):
            pass

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(_path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            return "ibl"

    class FakeParams:
        def __init__(self, _config):
            pass

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, _session):
            pass

        def render_terrain_pbr_pom(self, **_kwargs):
            return FakeFrame()

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    scene.render(str(tmp_path / "clipmap.png"))

    metadata = scene.last_render_metadata
    assert metadata["terrain_geometry_backend"] == "clipmap_indexed_pbr"
    assert metadata["clipmap_ring_count"] == 4
    assert metadata["clipmap_terrain_extent_m"] == pytest.approx(100_000.0)
    assert metadata["clipmap_bounded_memory"] is True
    assert metadata["clipmap_triangle_count"] > 0
    assert scene.last_validation_report.supported_features["terrain.clipmap_planner"] == "supported"
    assert scene.last_validation_report.supported_features["terrain.clipmap_indexed"] == "supported"


def test_mapscene_records_material_vt_stats_metadata(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )

    class FakeSession:
        def __init__(self, *, window=False):
            pass

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(_path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            return "ibl"

    class FakeParams:
        def __init__(self, _config):
            pass

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, _session):
            pass

        def render_terrain_pbr_pom(self, **_kwargs):
            return FakeFrame()

        def get_material_vt_stats(self):
            return {"avg_upload_ms": 0.25, "feedback_requests": 3.0}

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    scene.render(str(tmp_path / "vt-stats.png"))

    assert scene.last_render_metadata["material_vt_stats"] == {
        "avg_upload_ms": 0.25,
        "feedback_requests": 3.0,
    }


def test_mapscene_routes_virtual_texture_metadata_to_renderer(tmp_path, monkeypatch):
    captured = {"sources": []}
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={
                "width": 8,
                "height": 8,
                "source_id": "inline-dem",
                "virtual_texture": {
                    "enabled": True,
                    "families": [
                        {
                            "family": "albedo",
                            "virtual_size_px": [512, 512],
                            "tile_size": 120,
                            "tile_border": 4,
                        },
                        {
                            "family": "normal",
                            "virtual_size_px": [512, 512],
                            "tile_size": 120,
                            "tile_border": 4,
                        },
                        {
                            "family": "mask",
                            "virtual_size_px": [512, 512],
                            "tile_size": 120,
                            "tile_border": 4,
                        },
                    ],
                    "atlas_size": 1024,
                    "residency_budget_mb": 16.0,
                    "max_mip_levels": 4,
                    "use_feedback": True,
                    "procedural_sources": True,
                    "source_count": 2,
                    "source_size": 512,
                },
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(
            name="daylight",
            settings={"albedo_mode": "material", "colormap_strength": 0.0},
        ),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )

    class FakeSession:
        def __init__(self, *, window=False):
            pass

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(_path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            return "ibl"

    class FakeColormap:
        @staticmethod
        def from_stops(*, stops, domain):
            return (stops, domain)

    class FakeOverlay:
        @staticmethod
        def from_colormap1d(*_args, **_kwargs):
            return "overlay"

    class FakeParams:
        def __init__(self, config):
            captured["vt"] = config.vt

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, _session):
            pass

        def clear_material_vt_sources(self):
            captured["cleared"] = True

        def register_material_vt_source(self, material_index, family, image, virtual_size_px, fallback):
            captured["sources"].append(
                {
                    "material_index": material_index,
                    "family": family,
                    "shape": image.shape,
                    "virtual_size_px": virtual_size_px,
                    "fallback": fallback,
                }
            )

        def render_terrain_pbr_pom(self, **_kwargs):
            return FakeFrame()

        def get_material_vt_stats(self):
            return {"source_count": float(len(captured["sources"])), "avg_upload_ms": 0.1}

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "Colormap1D", FakeColormap)
    monkeypatch.setattr(f3d, "OverlayLayer", FakeOverlay)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)
    monkeypatch.setattr(map_scene, "_terrain_renderer_runtime_available", lambda: True)

    scene.render(str(tmp_path / "vt-metadata.png"))

    assert captured["cleared"] is True
    assert captured["vt"].enabled is True
    assert captured["vt"].atlas_size == 1024
    assert captured["vt"].layers[0].virtual_size_px == (512, 512)
    assert len(captured["sources"]) == 6
    assert {source["family"] for source in captured["sources"]} == {
        "albedo",
        "normal",
        "mask",
    }
    assert captured["sources"][0]["shape"] == (512, 512, 4)
    assert scene.last_render_metadata["material_vt_stats"]["source_count"] == 6.0


def test_derive_water_mask_detects_low_flat_regions():
    dem = np.ones((8, 8), dtype=np.float32)
    dem[2:6, 2:6] = 0.0

    mask = derive_water_mask(dem, level=0.1, slope_threshold=1.0)

    assert mask.dtype == np.float32
    assert mask.shape == dem.shape
    assert mask[3:5, 3:5].min() == pytest.approx(1.0)
    assert mask[0, 0] == pytest.approx(0.0)


def test_derive_water_mask_matches_hand_labeled_fixture():
    dem = np.ones((12, 12), dtype=np.float32)
    dem[3:9, 4:8] = 0.0
    dem[2, 4:8] = 0.2
    dem[9, 4:8] = 0.2
    expected = np.zeros_like(dem, dtype=bool)
    expected[3:9, 4:8] = True

    mask = derive_water_mask(dem, level=0.05, slope_threshold=0.6) > 0.5
    intersection = np.logical_and(mask, expected).sum()
    union = np.logical_or(mask, expected).sum()

    assert union > 0
    assert intersection / union >= 0.9


def test_mapscene_cloud_shadow_modulates_terrain_output(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={
                "width": 8,
                "height": 8,
                "source_id": "inline-dem",
                "clouds": {
                    "enabled": True,
                    "shadows_enabled": True,
                    "coverage": 0.7,
                    "density": 0.5,
                    "shadow_strength": 0.4,
                    "quality": "high",
                    "shadow_offset_x": 0.25,
                    "shadow_offset_y": 0.10,
                },
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )

    class FakeSession:
        def __init__(self, *, window=False):
            pass

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(_path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            return "ibl"

    class FakeParams:
        def __init__(self, _config):
            pass

    class FakeFrame:
        def to_numpy(self):
            rgba = np.full((64, 64, 4), 220, dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, _session):
            pass

        def render_terrain_pbr_pom(self, **_kwargs):
            return FakeFrame()

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    scene.render(str(tmp_path / "cloud-shadows.png"))

    assert scene.last_render_metadata["cloud_shadow_backend"] == "mapscene_numpy_cloud_shadow"
    assert scene.last_render_metadata["cloud_shadow_strength"] == pytest.approx(0.4)
    assert scene.last_render_metadata["cloud_shadow_offset"] == pytest.approx([0.25, 0.10])
    assert scene.last_validation_report.supported_features["mapscene.cloud_shadows"] == "supported"
    rendered = f3d.png_to_numpy(str(tmp_path / "cloud-shadows.png"))
    assert rendered[..., :3].min() < 220
    assert rendered[..., :3].max() == 220


def test_mapscene_cloud_shadow_offset_changes_pattern():
    base = np.full((48, 48, 4), 220, dtype=np.uint8)
    base[..., 3] = 255

    def scene_for_offset(offset_x: float) -> f3d.MapScene:
        return f3d.MapScene(
            terrain=f3d.TerrainSource(
                data=np.zeros((8, 8), dtype=np.float32),
                crs="EPSG:32610",
                metadata={
                    "width": 8,
                    "height": 8,
                    "source_id": "inline-dem",
                    "clouds": {
                        "enabled": True,
                        "shadows_enabled": True,
                        "coverage": 0.7,
                        "density": 0.5,
                        "shadow_strength": 0.4,
                        "quality": "high",
                        "shadow_offset_x": offset_x,
                    },
                },
            ),
            camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
            lighting=f3d.LightingPreset(name="daylight"),
            output=f3d.OutputSpec(width=48, height=48, format="png"),
        )

    first, first_metadata = map_scene._apply_mapscene_cloud_shadow(base, scene_for_offset(0.0).recipe)
    second, second_metadata = map_scene._apply_mapscene_cloud_shadow(base, scene_for_offset(0.33).recipe)

    assert first_metadata["cloud_shadow_offset"] == pytest.approx([0.0, 0.0])
    assert second_metadata["cloud_shadow_offset"] == pytest.approx([0.33, 0.0])
    assert not np.array_equal(first[..., :3], second[..., :3])


def test_mapscene_water_cloud_example_uses_engine_settings(tmp_path):
    example = _load_water_cloud_example()

    first = example.build_scene(tmp_path / "first.png", frame_index=0, total_frames=4, width=64, height=36)
    later = example.build_scene(tmp_path / "later.png", frame_index=2, total_frames=4, width=64, height=36)

    water = first.recipe.terrain.metadata["water"]
    first_clouds = first.recipe.terrain.metadata["clouds"]
    later_clouds = later.recipe.terrain.metadata["clouds"]
    assert water["enabled"] is True
    assert water["auto_mask"] is True
    assert first_clouds["enabled"] is True
    assert first_clouds["shadows_enabled"] is True
    assert first_clouds["shadow_offset_x"] != later_clouds["shadow_offset_x"]


def test_mapscene_auto_water_mask_reaches_terrain_renderer(tmp_path, monkeypatch):
    heightmap = np.ones((8, 8), dtype=np.float32)
    heightmap[2:6, 2:6] = 0.0
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=heightmap,
            crs="EPSG:32610",
            metadata={
                "width": 8,
                "height": 8,
                "source_id": "inline-dem",
                "water": {"enabled": True, "auto_mask": True, "level": 0.1, "slope_threshold": 1.0},
            },
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )
    calls: dict[str, object] = {}

    class FakeSession:
        def __init__(self, *, window=False):
            calls["session_window"] = window

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(_path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            return "ibl"

    class FakeParams:
        def __init__(self, config):
            calls["water_settings"] = config.water

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, _session):
            pass

        def render_terrain_pbr_pom(self, **kwargs):
            calls["water_mask"] = kwargs["water_mask"]
            return FakeFrame()

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    scene.render(str(tmp_path / "water-mask.png"))

    assert calls["water_settings"].auto_mask is True
    water_mask = calls["water_mask"]
    assert water_mask.shape == heightmap.shape
    assert water_mask.dtype == np.float32
    assert water_mask[3:5, 3:5].min() == pytest.approx(1.0)
    assert water_mask[0, 0] == pytest.approx(0.0)


def test_mapscene_cloud_settings_reach_terrain_params(tmp_path, monkeypatch):
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=100.0),
        lighting=f3d.LightingPreset(
            name="cloud-shadow",
            settings={
                "clouds": {
                    "enabled": True,
                    "shadows_enabled": True,
                    "coverage": 0.65,
                    "density": 0.55,
                    "shadow_strength": 0.4,
                    "quality": "high",
                }
            },
        ),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
    )
    calls: dict[str, object] = {}

    class FakeSession:
        def __init__(self, *, window=False):
            calls["session_window"] = window

    class FakeMaterialSet:
        @staticmethod
        def terrain_default():
            return "material-set"

    class FakeIbl:
        @staticmethod
        def from_hdr(_path, intensity=1.0, rotate_deg=0.0, quality="auto"):
            return "ibl"

    class FakeParams:
        def __init__(self, config):
            calls["clouds"] = config.clouds

    class FakeFrame:
        def to_numpy(self):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            return rgba

    class FakeTerrainRenderer:
        def __init__(self, _session):
            pass

        def render_terrain_pbr_pom(self, **_kwargs):
            return FakeFrame()

    monkeypatch.setattr(f3d, "Session", FakeSession)
    monkeypatch.setattr(f3d, "MaterialSet", FakeMaterialSet)
    monkeypatch.setattr(f3d, "IBL", FakeIbl)
    monkeypatch.setattr(f3d, "TerrainRenderParams", FakeParams)
    monkeypatch.setattr(f3d, "TerrainRenderer", FakeTerrainRenderer)

    scene.render(str(tmp_path / "cloud-settings.png"))

    clouds = calls["clouds"]
    assert isinstance(clouds, CloudSettings)
    assert clouds.enabled is True
    assert clouds.shadows_enabled is True
    assert clouds.coverage == pytest.approx(0.65)
    assert clouds.density == pytest.approx(0.55)
    assert clouds.shadow_strength == pytest.approx(0.4)
    assert clouds.quality == "high"
