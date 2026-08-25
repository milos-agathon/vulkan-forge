from __future__ import annotations

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
from forge3d._map_scene_render import _dash_segments, _draw_polygon_fill, _draw_polyline, _resolve_line_width_px


def test_dash_segments_split_polyline_by_arc_length() -> None:
    segments = _dash_segments([(0, 0), (10, 0)], [4, 2])

    assert segments == [
        ((0.0, 0.0), (4.0, 0.0)),
        ((6.0, 0.0), (10.0, 0.0)),
    ]


def test_draw_polyline_dash_leaves_screen_space_gaps() -> None:
    image = np.zeros((16, 64, 4), dtype=np.uint8)

    _draw_polyline(
        image,
        [(4, 8), (60, 8)],
        (255, 0, 0, 255),
        width_px=3,
        cap="butt",
        join="miter",
        dash_array=[6, 6],
    )

    assert image[8, 6, 0] > 200
    assert image[8, 14, 0] == 0
    assert image[8, 20, 0] > 200


def test_draw_polyline_edges_use_fractional_antialiasing() -> None:
    image = np.zeros((16, 24, 4), dtype=np.uint8)

    _draw_polyline(image, [(4, 8), (20, 8)], (255, 255, 255, 255), width_px=4, cap="butt")

    assert image[8, 10, 3] == 255
    assert 0 < image[6, 10, 3] < 255


def test_miter_and_bevel_joins_produce_distinct_geometry() -> None:
    miter = np.zeros((40, 40, 4), dtype=np.uint8)
    bevel = np.zeros((40, 40, 4), dtype=np.uint8)
    limited = np.zeros((40, 40, 4), dtype=np.uint8)
    points = [(8, 24), (24, 24), (24, 8)]

    _draw_polyline(miter, points, (255, 255, 255, 255), width_px=8, cap="butt", join="miter", miter_limit=4.0)
    _draw_polyline(bevel, points, (255, 255, 255, 255), width_px=8, cap="butt", join="bevel", miter_limit=4.0)
    _draw_polyline(limited, points, (255, 255, 255, 255), width_px=8, cap="butt", join="miter", miter_limit=1.0)

    assert miter[27, 27, 3] > 200
    assert bevel[27, 27, 3] == 0
    assert limited[27, 27, 3] == 0


def test_polygon_fill_antialiases_edges_and_preserves_holes() -> None:
    image = np.zeros((24, 24, 4), dtype=np.uint8)

    _draw_polygon_fill(
        image,
        [
            [(4, 4), (20, 4), (20, 20), (4, 20)],
            [(9, 9), (15, 9), (15, 15), (9, 15)],
        ],
        (0, 0, 255, 255),
    )

    assert image[8, 8, 3] == 255
    assert image[12, 12, 3] == 0
    assert 0 < image[4, 12, 3] < 255


def test_width_world_resolves_from_terrain_bounds() -> None:
    scene = f3d.SceneRecipe(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            metadata={"bounds": [100.0, 200.0, 300.0, 400.0]},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=200, height=100),
    )
    layer = f3d.VectorOverlay(layer_id="roads", width_world=20.0)

    assert _resolve_line_width_px(layer, {}, scene, 200, 100) == 15.0


def test_round_cap_extends_beyond_butt_cap() -> None:
    butt = np.zeros((16, 24, 4), dtype=np.uint8)
    rounded = np.zeros((16, 24, 4), dtype=np.uint8)

    _draw_polyline(butt, [(8, 8), (16, 8)], (0, 255, 0, 255), width_px=6, cap="butt")
    _draw_polyline(rounded, [(8, 8), (16, 8)], (0, 255, 0, 255), width_px=6, cap="round")

    assert butt[8, 5, 1] == 0
    assert rounded[8, 5, 1] > 200


def test_vector_overlay_stroke_style_serializes_and_reports_support() -> None:
    layer = f3d.VectorOverlay(
        layer_id="roads",
        features=[
            {
                "id": "road",
                "geometry": {"type": "LineString", "coordinates": [(0.1, 0.5), (0.9, 0.5)]},
            }
        ],
        width_px=5,
        line_join="round",
        line_cap="square",
        dash_array=[8, 4],
    )
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            metadata={"source_id": "flat-dem", "width": 4, "height": 4},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=32),
        layers=[layer],
    )

    report = scene.validate()
    payload = layer.to_dict()

    assert payload["width_px"] == 5
    assert payload["line_join"] == "round"
    assert payload["line_cap"] == "square"
    assert payload["dash_array"] == [8.0, 4.0]
    assert report.supported_features["vector.stroke.joins_caps"] == "supported"
    assert report.supported_features["vector.stroke.dashes"] == "supported"
    assert report.supported_features["vector.stroke.width_px"] == "supported"


def test_styled_vector_layers_route_to_precise_raster_path(monkeypatch, tmp_path) -> None:
    def fail_oit(*_args, **_kwargs):
        raise AssertionError("styled vectors must not use the simplified OIT bridge")

    monkeypatch.setattr(f3d, "vector_render_oit_py", fail_oit, raising=False)
    base = np.zeros((48, 64, 4), dtype=np.uint8)
    scene = f3d.SceneRecipe(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"source_id": "flat-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=48),
        layers=[
            f3d.VectorOverlay(
                layer_id="styled",
                crs="EPSG:32610",
                metadata={"source_id": "styled-roads"},
                features=[
                    {
                        "id": "turn",
                        "geometry": {"type": "LineString", "coordinates": [(0.1, 0.8), (0.4, 0.2), (0.7, 0.8)]},
                    },
                    {
                        "id": "park",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[(0.15, 0.35), (0.45, 0.35), (0.45, 0.65), (0.15, 0.65), (0.15, 0.35)]],
                        },
                    },
                ],
                width_px=6,
                line_cap="butt",
                line_join="miter",
                dash_array=[6, 4],
            )
        ],
    )

    rgba, composited = map_scene._composite_native_vector_layers(base, scene)

    assert composited is True
    assert int(np.count_nonzero(rgba[..., 3])) > 0

    # Through public MapScene.render() the precise route must not be hidden as
    # native OIT: the metadata and support report name the deterministic CPU
    # precise raster compositor explicitly.
    def fake_terrain(_recipe, _heightmap, **_kwargs):
        frame = np.zeros((48, 64, 4), dtype=np.uint8)
        frame[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=frame)

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)
    map_scene_obj = f3d.MapScene(recipe=scene)
    report = map_scene_obj.render(str(tmp_path / "styled.png"))

    assert map_scene_obj.last_render_metadata["vector_backend"] == "python_precise_raster"
    assert map_scene_obj.last_render_metadata["vector_backend"] != "native_oit"
    assert report.supported_features["mapscene.vector_precise_raster_composite"] == "supported"


def test_style_parser_accepts_line_dash_cap_join() -> None:
    from forge3d.style import validate_style_support

    report = validate_style_support(
        {
            "version": 8,
            "layers": [
                {
                    "id": "roads",
                    "type": "line",
                    "layout": {"line-cap": "round", "line-join": "bevel"},
                    "paint": {"line-color": "#ffffff", "line-width": 2, "line-dasharray": [4, 2]},
                }
            ],
        }
    )

    assert report.status == "ok"
    assert not report.diagnostics


def _center_pixel(image: np.ndarray) -> list[int]:
    return np.asarray(image)[image.shape[0] // 2, image.shape[1] // 2].tolist()


@pytest.mark.apple_metal_physical
def test_native_vector_oit_line_and_point_preserve_exact_rgba() -> None:
    if not hasattr(f3d, "vector_render_oit_py") or not f3d.has_gpu():
        return

    line_image = f3d.vector_render_oit_py(
        64,
        48,
        polylines=[[(-0.8, 0.0), (0.8, 0.0)]],
        polyline_rgba=[(0.0, 1.0, 0.0, 1.0)],
        stroke_width=[8.0],
    )
    point_image = f3d.vector_render_oit_py(
        64,
        48,
        points_xy=[(0.0, 0.0)],
        point_rgba=[(1.0, 0.0, 0.0, 1.0)],
        point_size=[16.0],
    )
    translucent = f3d.vector_render_oit_py(
        64,
        48,
        points_xy=[(0.0, 0.0)],
        point_rgba=[(0.0, 0.0, 1.0, 0.2)],
        point_size=[16.0],
    )

    assert line_image.shape == (48, 64, 4)
    np.testing.assert_allclose(_center_pixel(line_image), [0, 255, 0, 255], atol=1)
    assert 0 < np.count_nonzero(line_image[..., 3]) < line_image.shape[0] * line_image.shape[1] // 2
    assert point_image.shape == (48, 64, 4)
    np.testing.assert_allclose(_center_pixel(point_image), [255, 0, 0, 255], atol=1)
    np.testing.assert_allclose(_center_pixel(translucent), [0, 0, 255, 51], atol=1)
    assert np.count_nonzero(point_image[..., 3]) > 0


@pytest.mark.apple_metal_physical
def test_native_vector_oit_overlap_is_order_independent_with_exact_alpha() -> None:
    if not hasattr(f3d, "vector_render_oit_py") or not f3d.has_gpu():
        return

    kwargs = {
        "width": 64,
        "height": 48,
        "points_xy": [(0.0, 0.0), (0.0, 0.0)],
        "point_size": [16.0, 16.0],
    }
    red_then_blue = f3d.vector_render_oit_py(
        **kwargs,
        point_rgba=[(1.0, 0.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)],
    )
    blue_then_red = f3d.vector_render_oit_py(
        **kwargs,
        point_rgba=[(0.0, 0.0, 1.0, 0.5), (1.0, 0.0, 0.0, 0.5)],
    )

    assert np.array_equal(red_then_blue, blue_then_red)
    np.testing.assert_allclose(_center_pixel(red_then_blue), [128, 0, 128, 191], atol=1)


@pytest.mark.apple_metal_physical
def test_native_vector_oit_zero_alpha_writes_no_pixels() -> None:
    if not hasattr(f3d, "vector_render_oit_py") or not f3d.has_gpu():
        return

    image = f3d.vector_render_oit_py(
        64,
        48,
        points_xy=[(0.0, 0.0)],
        point_rgba=[(1.0, 0.0, 0.0, 0.0)],
        point_size=[16.0],
    )

    assert np.count_nonzero(image) == 0


@pytest.mark.apple_metal_physical
def test_native_vector_oit_repeated_frames_are_bitexact() -> None:
    if not hasattr(f3d, "vector_render_oit_py") or not f3d.has_gpu():
        return

    frames = [
        np.asarray(
            f3d.vector_render_oit_py(
                32,
                24,
                points_xy=[(0.0, 0.0)],
                point_rgba=[(1.0, 0.5, 0.0, 1.0)],
                point_size=[6.0],
            )
        ).copy()
        for _ in range(4)
    ]

    assert all(np.array_equal(frames[0], frame) for frame in frames[1:])
    assert 0 < np.count_nonzero(frames[0][..., 3]) < frames[0].shape[0] * frames[0].shape[1]
    assert frames[0][0, 0].tolist() == [0, 0, 0, 0]


@pytest.mark.apple_metal_physical
def test_native_vector_oit_edl_output_has_visible_alpha() -> None:
    if not hasattr(f3d, "vector_render_oit_edl_py") or not f3d.has_gpu():
        return

    image = f3d.vector_render_oit_edl_py(
        64,
        48,
        points_xy=[(-0.2, -0.2), (0.2, 0.2)],
        point_rgba=[(1.0, 0.2, 0.1, 1.0), (0.2, 0.8, 1.0, 1.0)],
        point_size=[12.0, 12.0],
        edl_strength=2.0,
        edl_radius_px=2.0,
    )

    assert image.shape == (48, 64, 4)
    assert int(np.max(image[..., :3])) > 0
    assert int(np.max(image[..., 3])) > 0
    assert np.count_nonzero(image[..., 3]) > 0
