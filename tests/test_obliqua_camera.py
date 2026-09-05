from __future__ import annotations

import numpy as np
import pytest

import forge3d.path_tracing as pt

from _obliqua_gpu_guard import require_qualifying_gpu


class _CaptureNative:
    def hybrid_render_terrain_reference(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def test_reference_wrapper_forwards_obliqua_camera_contract(monkeypatch):
    monkeypatch.setattr(pt, "_NATIVE", _CaptureNative())
    camera = {
        "origin": (0.0, 3.0, 8.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "model": "orthographic",
        "half_height": 2.5,
    }

    result = pt.hybrid_render_terrain_reference(
        np.zeros((2, 2), dtype=np.float32),
        64,
        32,
        camera,
        sensor_rect=(0.25, 0.0, 0.75, 0.5),
        full_width=128,
        full_height=64,
        pixel_offset=(32, 0),
        min_frames=1,
        max_frames=1,
    )

    kwargs = result["kwargs"]
    assert kwargs["camera_model"] == "orthographic"
    assert kwargs["sensor_rect"] == (0.25, 0.0, 0.75, 0.5)
    assert kwargs["full_width"] == 128
    assert kwargs["full_height"] == 64
    assert kwargs["pixel_offset"] == (32, 0)


@pytest.mark.parametrize("model", ["fisheye", "", "ORTHOGRAPHIC"])
def test_reference_wrapper_rejects_unknown_camera_model(monkeypatch, model):
    monkeypatch.setattr(pt, "_NATIVE", _CaptureNative())
    with pytest.raises(ValueError, match="camera model"):
        pt.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            8,
            8,
            {"model": model},
            min_frames=1,
            max_frames=1,
        )


def test_reference_wrapper_rejects_explicit_empty_camera_model(monkeypatch):
    monkeypatch.setattr(pt, "_NATIVE", _CaptureNative())
    with pytest.raises(ValueError, match="camera model"):
        pt.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            8,
            8,
            {},
            camera_model="",
            min_frames=1,
            max_frames=1,
        )


def test_reference_wrapper_rejects_underspecified_crop(monkeypatch):
    monkeypatch.setattr(pt, "_NATIVE", _CaptureNative())
    with pytest.raises(ValueError, match="full_width and full_height"):
        pt.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            8,
            8,
            {},
            pixel_offset=(1, 0),
            min_frames=1,
            max_frames=1,
        )
    with pytest.raises(ValueError, match="pixel_offset"):
        pt.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            8,
            8,
            {},
            pixel_offset=(1.9, 0),
            min_frames=1,
            max_frames=1,
        )
    with pytest.raises(ValueError, match="full_width and full_height"):
        pt.hybrid_render_terrain_reference(
            np.zeros((2, 2), dtype=np.float32),
            8,
            8,
            {},
            sensor_rect=(0.0, 0.0, 0.5, 1.0),
            min_frames=1,
            max_frames=1,
        )


def test_orthographic_tile_is_byte_exact_full_frame_crop():
    require_qualifying_gpu("OBLIQUA camera crop proof")
    dem = np.arange(64, dtype=np.float32).reshape(8, 8) * 0.02
    camera = {
        "origin": (0.0, 6.0, 7.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "model": "orthographic",
        "half_height": 4.0,
        "fov_y": None,
    }
    options = {
        "min_frames": 32,
        "max_frames": 32,
        "variance_threshold": 1e9,
        "seed": 19,
        "full_width": 16,
        "full_height": 8,
    }
    full = pt.hybrid_render_terrain_reference(dem, 16, 8, camera, **options)["rgba"]
    left = pt.hybrid_render_terrain_reference(
        dem,
        8,
        8,
        camera,
        pixel_offset=(0, 0),
        **options,
    )["rgba"]
    right = pt.hybrid_render_terrain_reference(
        dem,
        8,
        8,
        camera,
        pixel_offset=(8, 0),
        **options,
    )["rgba"]

    assert np.array_equal(full, np.concatenate([left, right], axis=1))


def test_default_equivalent_pinhole_metadata_is_byte_exact():
    require_qualifying_gpu("OBLIQUA camera crop proof")
    dem = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.02
    camera = {
        "origin": (0.0, 3.0, 8.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_y": 45.0,
    }
    options = {
        "min_frames": 4,
        "max_frames": 4,
        "variance_threshold": 1e9,
        "seed": 23,
    }
    legacy = pt.hybrid_render_terrain_reference(dem, 8, 8, camera, **options)["rgba"]
    explicit = pt.hybrid_render_terrain_reference(
        dem,
        8,
        8,
        {**camera, "model": "pinhole"},
        **options,
    )["rgba"]
    assert np.array_equal(legacy, explicit)
    identity_rect = pt.hybrid_render_terrain_reference(
        dem,
        8,
        8,
        camera,
        sensor_rect=(0.0, 0.0, 1.0, 1.0),
        **options,
    )["rgba"]
    assert np.array_equal(legacy, identity_rect)
    nullable_metadata = pt.hybrid_render_terrain_reference(
        dem,
        8,
        8,
        {
            **camera,
            "model": None,
            "sensor_rect": None,
            "half_height": None,
        },
        **options,
    )["rgba"]
    assert np.array_equal(legacy, nullable_metadata)


def test_pinhole_tile_is_byte_exact_full_frame_crop():
    require_qualifying_gpu("OBLIQUA camera crop proof")
    dem = np.arange(64, dtype=np.float32).reshape(8, 8) * 0.02
    camera = {
        "origin": (0.0, 6.0, 7.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "model": "pinhole",
        "fov_y": 35.0,
    }
    options = {
        "min_frames": 32,
        "max_frames": 32,
        "variance_threshold": 1e9,
        "seed": 29,
        "full_width": 16,
        "full_height": 8,
    }
    full = pt.hybrid_render_terrain_reference(dem, 16, 8, camera, **options)["rgba"]
    halves = [
        pt.hybrid_render_terrain_reference(
            dem,
            8,
            8,
            camera,
            pixel_offset=(start, 0),
            **options,
        )["rgba"]
        for start in (0, 8)
    ]
    assert np.array_equal(full, np.concatenate(halves, axis=1))
    explicit_right = pt.hybrid_render_terrain_reference(
        dem,
        8,
        8,
        camera,
        sensor_rect=(0.5, 0.0, 1.0, 1.0),
        **options,
    )["rgba"]
    assert np.array_equal(full[:, 8:], explicit_right)
    for bad_offset in ((0, 0), (1, 0)):
        with pytest.raises(ValueError, match="pixel_offset"):
            pt.hybrid_render_terrain_reference(
                dem,
                8,
                8,
                camera,
                sensor_rect=(0.5, 0.0, 1.0, 1.0),
                pixel_offset=bad_offset,
                **options,
            )
    with pytest.raises(ValueError, match="pixel-aligned"):
        pt.hybrid_render_terrain_reference(
            dem,
            8,
            8,
            camera,
            sensor_rect=(2.5 / 16, 0.0, 10.5 / 16, 1.0),
            **options,
        )


def test_native_camera_dict_sensor_rect_matches_top_level():
    require_qualifying_gpu("OBLIQUA camera crop proof")
    dem = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.02
    camera = {
        "origin": (0.0, 3.0, 8.0),
        "look_at": (0.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "model": "off_axis",
        "fov_y": 45.0,
    }
    options = {
        "min_frames": 4,
        "max_frames": 4,
        "variance_threshold": 1e9,
        "seed": 31,
    }
    from_dict = pt._NATIVE.hybrid_render_terrain_reference(
        dem,
        4,
        4,
        {**camera, "sensor_rect": (0.25, 0.25, 0.75, 0.75)},
        full_width=8,
        full_height=8,
        **options,
    )["rgba"]
    from_kwarg = pt._NATIVE.hybrid_render_terrain_reference(
        dem,
        4,
        4,
        camera,
        sensor_rect=(0.25, 0.25, 0.75, 0.75),
        full_width=8,
        full_height=8,
        **options,
    )["rgba"]
    assert np.array_equal(from_dict, from_kwarg)
