from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from pyproj import Geod
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

import forge3d as f3d
from forge3d.terrain import viewshed


gpu_required = pytest.mark.skipif(not f3d.has_gpu(), reason="viewshed requires a GPU")

SWISS_BOUNDS = (7.0, 46.4, 8.0, 47.2)
SWISS_SHAPE = (64, 64)
SWISS_OBSERVER_CELL = (55, 49)


def _switzerland_dem() -> np.ndarray:
    source = Path(__file__).parents[1] / "assets" / "tif" / "switzerland_dem.tif"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "d09d229fa265749720a6b4bd40c440799f43286bf2d401d732ea77f89d0bd478"
    )
    with rasterio.open(source) as dataset:
        dem = dataset.read(
            1,
            window=from_bounds(*SWISS_BOUNDS, dataset.transform),
            out_shape=SWISS_SHAPE,
            resampling=Resampling.bilinear,
            masked=True,
        )
    assert dem.count() == dem.size, "committed HELIOS crop must contain no nodata"
    return np.asarray(dem, dtype=np.float32)


def _switzerland_observer() -> tuple[float, float]:
    row, column = SWISS_OBSERVER_CELL
    height, width = SWISS_SHAPE
    left, bottom, right, top = SWISS_BOUNDS
    return (
        top - (row + 0.5) * (top - bottom) / height,
        left + (column + 0.5) * (right - left) / width,
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.count_nonzero(left & right) / np.count_nonzero(left | right))


@gpu_required
def test_viewshed_returns_deterministic_visibility_and_physics_arrays() -> None:
    dem = np.zeros((33, 33), dtype=np.float32)
    dem[:, 16] = 600.0
    kwargs = dict(
        observer=(0.5, 0.1),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        observer_height=100.0,
        target_height=0.0,
        max_distance=120_000.0,
        earth_model="ellipsoid",
        refraction_model="bennett",
        return_diagnostics=True,
    )

    first = viewshed(dem, **kwargs)
    second = viewshed(dem, **kwargs)

    assert first["visibility"].dtype == np.bool_
    assert first["visibility"].shape == dem.shape
    assert np.array_equal(first["visibility"], second["visibility"])
    assert np.array_equal(first["curvature_drop_m"], second["curvature_drop_m"])
    assert first["visibility"][16, 10]
    assert not first["visibility"][16, 24]
    assert first["curvature_drop_m"][16, 32] > 0.0
    assert first["refraction_gain_m"][16, 32] > 0.0
    assert first["horizon_distance_m"][16, 32] > 0.0


@gpu_required
def test_viewshed_default_return_is_boolean_array() -> None:
    result = viewshed(
        np.zeros((8, 8), dtype=np.float32),
        observer=(0.5, 0.5),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.dtype == np.bool_
    assert result.all()


def test_viewshed_rejects_unsupported_models() -> None:
    with pytest.raises(ValueError, match="unsupported earth_model"):
        viewshed(
            np.zeros((4, 4), dtype=np.float32),
            observer=(0.5, 0.5),
            bounds=(0.0, 0.0, 1.0, 1.0),
            height_system="ellipsoidal",
            earth_model="cube",
        )


def test_viewshed_and_shadow_mask_reject_flat_refraction() -> None:
    from forge3d.geo import SolarTime
    from forge3d.terrain import shadow_mask

    dem = np.zeros((2, 2), dtype=np.float32)
    common = dict(
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="bennett",
    )
    with pytest.raises(RuntimeError, match="flat earth only supports"):
        viewshed(dem, (0.5, 0.5), **common)
    with pytest.raises(RuntimeError, match="flat earth only supports"):
        shadow_mask(
            dem,
            SolarTime(
                utc=(2003, 10, 17, 17, 0, 0),
                observer_lat=0.5,
                observer_lon=0.5,
            ),
            **common,
        )


def test_viewshed_rejects_nonlocal_global_bounds() -> None:
    with pytest.raises(ValueError, match="spanning less than 180 degrees"):
        viewshed(
            np.zeros((2, 2), dtype=np.float32),
            observer=(0.0, 0.0),
            bounds=(-180.0, -1.0, 180.0, 1.0),
            height_system="ellipsoidal",
            earth_model="flat",
            refraction_model="none",
        )


@gpu_required
def test_viewshed_accepts_local_antimeridian_bounds() -> None:
    result = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        observer=(-9.5, 180.0),
        bounds=(179.8, -10.0, -179.8, -9.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.shape == (3, 3)


@gpu_required
def test_viewshed_uses_geodesic_distance_at_high_latitude() -> None:
    diagnostics = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        observer=(75.0, 0.0),
        bounds=(0.0, 70.0, 10.0, 80.0),
        height_system="ellipsoidal",
        earth_model="ellipsoid",
        refraction_model="none",
        return_diagnostics=True,
    )
    inverse = f3d.geodesic_inverse(75.0, 0.0, 78.33333333333333, 8.333333333333334)
    latitude = np.deg2rad(75.0)
    azimuth = np.deg2rad(inverse["azi1"])
    eccentricity_squared = 6.694_379_990_141_316_5e-3
    w = np.sqrt(1.0 - eccentricity_squared * np.sin(latitude) ** 2)
    meridional = 6_378_137.0 * (1.0 - eccentricity_squared) / w**3
    prime_vertical = 6_378_137.0 / w
    radius = 1.0 / (
        np.cos(azimuth) ** 2 / meridional + np.sin(azimuth) ** 2 / prime_vertical
    )
    expected_drop = inverse["s12"] ** 2 / (2.0 * radius)
    assert diagnostics["curvature_drop_m"][0, 2] == pytest.approx(
        expected_drop, rel=2e-5
    )


@gpu_required
def test_viewshed_marches_blockers_on_the_geodesic() -> None:
    dem = np.zeros((11, 11), dtype=np.float32)
    dem[2, 4] = 1_500.0
    visibility = viewshed(
        dem,
        observer=(75.0, 0.0),
        bounds=(0.0, 70.0, 10.0, 80.0),
        height_system="ellipsoidal",
        observer_height=1_000.0,
        earth_model="flat",
        refraction_model="none",
    )
    assert not visibility[0, 10]


@gpu_required
def test_continuous_leaf_detects_between_half_cell_samples() -> None:
    # Along the diagonal, the bilinear surface is below the 25 m sightline at
    # s=0, 0.5, and 1, but rises to 25.5 m at s=0.75. The former half-cell
    # point sampler therefore missed it; exact leaf traversal must block it.
    dem = np.array([[16.5, 28.5], [28.5, 24.5]], dtype=np.float32)
    visibility = viewshed(
        dem,
        observer=(0.0015, 0.0005),
        bounds=(0.0, 0.0, 0.002, 0.002),
        height_system="ellipsoidal",
        observer_height=8.5,
        target_height=0.5,
        earth_model="flat",
        refraction_model="none",
    )
    assert not visibility[1, 1]


@gpu_required
def test_horizon_diagnostic_includes_terrain_elevation() -> None:
    dem = np.zeros((3, 3), dtype=np.float32)
    dem[1, 2] = 1_000.0
    diagnostics = viewshed(
        dem,
        observer=(0.5, 0.5),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        observer_height=2.0,
        earth_model="sphere",
        refraction_model="none",
        return_diagnostics=True,
    )
    assert (
        diagnostics["horizon_distance_m"][1, 2]
        > diagnostics["horizon_distance_m"][0, 2]
    )


@gpu_required
def test_spherical_viewshed_scales_with_configured_radius() -> None:
    kwargs = dict(
        observer=(0.5, 0.5),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="sphere",
        refraction_model="none",
        return_diagnostics=True,
    )
    small = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        sphere_radius_m=3_000_000.0,
        **kwargs,
    )
    large = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        sphere_radius_m=6_000_000.0,
        **kwargs,
    )
    assert large["curvature_drop_m"][0, 2] == pytest.approx(
        2.0 * small["curvature_drop_m"][0, 2], rel=2e-5
    )


@gpu_required
def test_viewshed_preserves_observer_at_raster_edge() -> None:
    result = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        observer=(0.5, 0.0),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.shape == (3, 3)


@gpu_required
def test_viewshed_rejects_geodesics_outside_dem_footprint() -> None:
    with pytest.raises(RuntimeError, match="geodesic leaves the DEM footprint"):
        viewshed(
            np.zeros((3, 3), dtype=np.float32),
            observer=(74.1666666667, -26.6666666667),
            bounds=(-40.0, 70.0, 40.0, 75.0),
            height_system="ellipsoidal",
            earth_model="flat",
            refraction_model="none",
        )


def test_viewshed_shader_has_no_fixed_los_step_cap() -> None:
    shader = (
        Path(__file__).parents[1] / "src" / "shaders" / "terrain_viewshed.wgsl"
    ).read_text(encoding="utf-8")
    assert "half_cell_steps" not in shader
    assert "fn terrain_slab_xz(" in shader
    assert "fn terrain_leaf_occluded(" in shader
    assert "fn terrain_trace_segment(" in shader
    assert shader.count("terrain_trace_segment(") == 3
    assert "fn minmax_may_exceed(" not in shader
    assert "north_crossing_m" in shader
    assert "east_crossing_m" in shader
    assert "shadow_step_m(segment_latitude, azimuth)" in shader
    assert "geodesic_positions_m: array<vec2<f32>>" in shader
    assert "shadow_result: array<atomic<u32>>" in shader
    assert "minmax_texture: texture_2d<f32>" in shader
    assert "textureNumLevels(minmax_texture)" in shader
    assert "var<storage, read> heights" not in shader


def test_viewshed_reports_missing_native_lazily(monkeypatch) -> None:
    import forge3d.terrain as terrain

    monkeypatch.setattr(terrain, "get_native_module", lambda: None)
    monkeypatch.setattr(terrain, "native_import_error", lambda: ImportError("missing test module"))
    with pytest.raises(RuntimeError, match="missing test module"):
        terrain.viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
            height_system="ellipsoidal",
        )


def test_viewshed_reports_stale_native_extension(monkeypatch) -> None:
    import forge3d.terrain as terrain

    monkeypatch.setattr(terrain, "get_native_module", object)
    with pytest.raises(RuntimeError, match="does not provide terrain_viewshed"):
        terrain.viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
            height_system="ellipsoidal",
        )


@gpu_required
def test_viewshed_height_system_is_explicit_and_strict() -> None:
    with pytest.raises(TypeError, match="height_system"):
        viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="unsupported height_system"):
        viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
            height_system="unknown",
        )
    result = viewshed(
        np.zeros((2, 2), dtype=np.float32),
        (0.0, 0.0),
        bounds=(-1.0, -1.0, 1.0, 1.0),
        height_system="orthometric_egm96",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.shape == (2, 2)


@gpu_required
def test_real_dem_curvature_and_refraction_are_load_bearing() -> None:
    dem = _switzerland_dem()
    observer = _switzerland_observer()
    east_west_m = f3d.geodesic_inverse(
        observer[0], SWISS_BOUNDS[0], observer[0], SWISS_BOUNDS[2]
    )["s12"]
    north_south_m = f3d.geodesic_inverse(
        SWISS_BOUNDS[1], observer[1], SWISS_BOUNDS[3], observer[1]
    )["s12"]
    assert min(east_west_m, north_south_m) >= 60_000.0

    common = dict(
        observer=observer,
        bounds=SWISS_BOUNDS,
        height_system="orthometric_egm96",
        observer_height=2.0,
        target_height=0.0,
    )
    flat = viewshed(dem, earth_model="flat", refraction_model="none", **common)
    curved = viewshed(
        dem, earth_model="ellipsoid", refraction_model="bennett", **common
    )
    iou = _iou(flat, curved)
    flipped = int(np.count_nonzero(flat ^ curved))
    print(
        f"HELIOS load-bearing: IoU={iou:.9f}, flipped={flipped}, "
        f"extent={east_west_m:.1f}x{north_south_m:.1f} m"
    )
    assert iou <= 0.96
    assert flipped > 0


@gpu_required
def test_viewshed_matches_committed_curved_whitebox_reference() -> None:
    curved_reference_path = (
        Path(__file__).parent
        / "golden"
        / "viewshed"
        / "whitebox_curved_analytic_256.png"
    )
    flat_reference_path = curved_reference_path.with_name(
        "whitebox_flat_analytic_256.png"
    )
    assert (
        hashlib.sha256(curved_reference_path.read_bytes()).hexdigest()
        == "b40d22f15fafd71965382cdbf0a7b544069542e7ea5a510488a803828163055d"
    )
    assert (
        hashlib.sha256(flat_reference_path.read_bytes()).hexdigest()
        == "cb11c310d80215f00aad6d08e30bfb00e4ac0849dc5314722bd4636cf4611622"
    )
    curved_reference = f3d.png_to_numpy(curved_reference_path)[..., 0] > 0
    dem = np.zeros((256, 256), dtype=np.float32)
    observer = (
        0.5 - 127.5 / 256.0,
        -0.5 + 127.5 / 256.0,
    )
    common = dict(
        dem=dem,
        observer=observer,
        bounds=(-0.5, -0.5, 0.5, 0.5),
        height_system="ellipsoidal",
        observer_height=250.0,
        target_height=0.0,
    )
    actual = viewshed(
        earth_model="ellipsoid",
        refraction_model="effective_radius",
        refraction_k=0.13,
        **common,
    )
    flat_negative = viewshed(
        earth_model="flat",
        refraction_model="none",
        **common,
    )
    iou = _iou(actual, curved_reference)
    flipped = int(np.count_nonzero(actual ^ curved_reference))
    flat_negative_iou = _iou(flat_negative, curved_reference)
    flat_negative_flipped = int(np.count_nonzero(flat_negative ^ curved_reference))
    print(
        f"HELIOS curved Whitebox reference: IoU={iou:.9f}, flipped={flipped}, "
        f"flat_negative_iou={flat_negative_iou:.9f}, "
        f"flat_negative_flipped={flat_negative_flipped}"
    )
    assert iou >= 0.98
    assert flat_negative_iou < 0.98


def test_whitebox_reference_workload_rejects_flat_negative_control() -> None:
    directory = Path(__file__).parent / "golden" / "viewshed"
    curved_path = directory / "whitebox_curved_analytic_256.png"
    flat_path = directory / "whitebox_flat_analytic_256.png"
    assert (
        hashlib.sha256(curved_path.read_bytes()).hexdigest()
        == "b40d22f15fafd71965382cdbf0a7b544069542e7ea5a510488a803828163055d"
    )
    assert (
        hashlib.sha256(flat_path.read_bytes()).hexdigest()
        == "cb11c310d80215f00aad6d08e30bfb00e4ac0849dc5314722bd4636cf4611622"
    )
    curved = f3d.png_to_numpy(curved_path)[..., 0] > 0
    flat = f3d.png_to_numpy(flat_path)[..., 0] > 0
    iou = _iou(flat, curved)
    flipped = int(np.count_nonzero(flat ^ curved))
    assert int(np.count_nonzero(curved)) == 57_957
    assert int(np.count_nonzero(flat)) == 65_536
    assert iou == pytest.approx(0.8843536376953125, abs=1e-15)
    assert flipped == 7_579
    assert iou < 0.98

    rows, columns = np.indices((256, 256))
    latitudes = 0.5 - (rows + 0.5) / 256.0
    longitudes = -0.5 + (columns + 0.5) / 256.0
    observer_latitude = 0.5 - 127.5 / 256.0
    observer_longitude = -0.5 + 127.5 / 256.0
    azimuth_deg, _, distance_m = Geod(ellps="WGS84").inv(
        np.full_like(longitudes, observer_longitude),
        np.full_like(latitudes, observer_latitude),
        longitudes,
        latitudes,
    )
    latitude_rad = np.deg2rad(observer_latitude)
    eccentricity_squared = 6.694_379_990_141_316_5e-3
    w = np.sqrt(1.0 - eccentricity_squared * np.sin(latitude_rad) ** 2)
    meridional_m = 6_378_137.0 * (1.0 - eccentricity_squared) / w**3
    prime_vertical_m = 6_378_137.0 / w
    azimuth_rad = np.deg2rad(azimuth_deg)
    quadratic = 0.5 * 0.87 * (
        np.cos(azimuth_rad) ** 2 / meridional_m
        + np.sin(azimuth_rad) ** 2 / prime_vertical_m
    )
    linear = np.divide(
        -quadratic * distance_m**2 - 250.0,
        distance_m,
        out=np.zeros_like(distance_m),
        where=distance_m > 0.0,
    )
    vertex_m = np.divide(
        -linear,
        2.0 * quadratic,
        out=np.zeros_like(distance_m),
        where=quadratic > 0.0,
    )
    vertex_m = np.clip(vertex_m, 0.0, distance_m)
    minimum_height_m = 250.0 + linear * vertex_m + quadratic * vertex_m**2
    analytic = minimum_height_m >= -0.001
    analytic[127, 127] = True
    analytic_iou = _iou(analytic, curved)
    analytic_flipped = int(np.count_nonzero(analytic ^ curved))
    assert int(np.count_nonzero(analytic)) == 57_813
    assert analytic_iou == pytest.approx(0.9975153993477923, abs=1e-15)
    assert analytic_flipped == 144
    assert analytic_iou >= 0.98


@gpu_required
def test_viewshed_and_shadow_mask_report_exact_tracked_host_memory() -> None:
    probe = r"""
import json
import sys

import numpy as np
import forge3d as f3d
from forge3d.geo import SolarTime
from forge3d.terrain import shadow_mask, viewshed

mode = sys.argv[1]
dem = np.zeros((64, 64), dtype=np.float32)
common = dict(
    bounds=(0.0, 0.0, 1.0, 1.0),
    height_system="ellipsoidal",
    earth_model="ellipsoid",
    refraction_model="effective_radius",
    refraction_k=0.13,
)
if mode == "viewshed":
    output = viewshed(dem, (0.5, 0.5), observer_height=2.0, **common)
    expected_readback_bytes = dem.size * 16
else:
    output = shadow_mask(
        dem,
        SolarTime(
            utc=(2003, 10, 17, 12, 0, 0),
            observer_lat=0.5,
            observer_lon=0.5,
        ),
        **common,
    )
    expected_readback_bytes = ((dem.size + 31) // 32) * 4
assert output.shape == dem.shape
metrics = f3d.memory_metrics()
print(json.dumps({
    "current_host_visible_bytes": int(metrics["host_visible_bytes"]),
    "expected_readback_bytes": int(expected_readback_bytes),
    "peak_host_visible_bytes": int(metrics["peak_host_visible_bytes"]),
}, sort_keys=True))
"""

    measurements: dict[str, dict[str, int]] = {}
    environment = dict(os.environ)
    environment["FORGE3D_NO_BOOTSTRAP"] = "1"
    for mode in ("viewshed", "shadow_mask"):
        completed = subprocess.run(
            [sys.executable, "-c", probe, mode],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        measurements[mode] = json.loads(completed.stdout.splitlines()[-1])

    expected = {"viewshed": 64 * 64 * 16, "shadow_mask": 128 * 4}
    for mode, expected_readback_bytes in expected.items():
        assert measurements[mode] == {
            "current_host_visible_bytes": 0,
            "expected_readback_bytes": expected_readback_bytes,
            "peak_host_visible_bytes": expected_readback_bytes,
        }
    print("HELIOS path memory: " + json.dumps(measurements, sort_keys=True))
