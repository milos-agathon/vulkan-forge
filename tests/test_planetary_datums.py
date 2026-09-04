"""SELENE planetary-datum CPU gates."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from forge3d import crs, gis


ROOT = Path(__file__).resolve().parents[1]


def _longitude_error(actual, expected):
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def _areoid_reference_points():
    path = Path(__file__).parent / "data" / "mars_areoid_reference.txt"
    return [
        (float(lat), float(lon), float(value), source)
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
        for lat, lon, value, source in [line.split()]
    ]


def _mars_geodesic_reference_points():
    path = Path(__file__).parent / "data" / "geodtest_mars.dat"
    return [
        tuple(map(float, line.split()))
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _load_moon_example():
    path = ROOT / "examples" / "moon_south_pole.py"
    spec = importlib.util.spec_from_file_location("moon_south_pole", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_body_info_reports_reference_surfaces_and_units():
    earth = crs.body_info("earth")
    moon = crs.body_info("MOON")
    mars = crs.body_info("Mars")

    assert earth["name"] == "Earth"
    assert earth["semi_major_m"] == 6_378_137.0
    assert earth["gravity_surface"] == "EGM96"
    assert moon["name"] == "Moon"
    assert moon["semi_major_m"] == 1_737_400.0
    assert moon["flattening"] == 0.0
    assert moon["gravity_surface"] is None
    assert mars["name"] == "Mars"
    assert mars["semi_major_m"] == 3_396_190.0
    assert 1.0 / mars["flattening"] == pytest.approx(169.894_447_223_612)
    assert mars["gravity_surface"] == "Mars areoid"
    assert set(mars) == {
        "name",
        "semi_major_m",
        "semi_minor_m",
        "flattening",
        "prime_meridian_w0_deg",
        "rotation_rate_deg_per_day",
        "gravity_surface",
    }


def test_body_info_rejects_unknown_body_without_earth_fallback():
    with pytest.raises(ValueError, match="unsupported body 'ceres'.*earth, moon, or mars"):
        crs.body_info("ceres")


def test_mars_geodesics_match_committed_geographiclib_oracle():
    points = _mars_geodesic_reference_points()
    assert len(points) == 100
    worst_distance = 0.0
    worst_azimuth = 0.0
    for lat1, lon1, azi1, lat2, lon2, azi2, distance, *_ in points:
        result = crs.geodesic_inverse(
            lat1, lon1, lat2, lon2, body="mars"
        )
        worst_distance = max(worst_distance, abs(result["s12"] - distance))
        worst_azimuth = max(
            worst_azimuth,
            _longitude_error(result["azi1"], azi1),
            _longitude_error(result["azi2"], azi2),
        )
    assert worst_distance < 1e-3
    assert worst_azimuth < 1e-9


def test_moon_geodesics_match_exact_great_circle_and_close():
    radius = crs.body_info("moon")["semi_major_m"]
    worst_relative = 0.0
    for index in range(100):
        lat1 = -80.0 + index * 1.4
        lon1 = -175.0 + (index * 73.0) % 350.0
        lat2 = 75.0 - index * 1.3
        lon2 = -170.0 + (index * 47.0) % 340.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        central_angle = math.atan2(
            math.hypot(
                math.cos(phi2) * math.sin(delta_lon),
                math.cos(phi1) * math.sin(phi2)
                - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon),
            ),
            math.sin(phi1) * math.sin(phi2)
            + math.cos(phi1) * math.cos(phi2) * math.cos(delta_lon),
        )
        result = crs.geodesic_inverse(
            lat1, lon1, lat2, lon2, body="moon"
        )
        expected = radius * central_angle
        worst_relative = max(
            worst_relative, abs(result["s12"] - expected) / expected
        )
        destination = crs.geodesic_direct(
            lat1, lon1, result["azi1"], result["s12"], body="moon"
        )
        assert abs(destination["lat2"] - lat2) < 1e-12
        assert _longitude_error(destination["lon2"], lon2) < 1e-12
    assert worst_relative < 1e-12


def test_geodesic_body_defaults_to_earth_and_rejects_unknown_body():
    omitted = crs.geodesic_inverse(10.0, 20.0, -30.0, 40.0)
    explicit = crs.geodesic_inverse(10.0, 20.0, -30.0, 40.0, body="earth")
    assert omitted == explicit
    with pytest.raises(ValueError, match="unsupported body 'ceres'"):
        crs.geodesic_inverse(10.0, 20.0, -30.0, 40.0, body="ceres")


def test_olympus_mons_wrong_earth_datum_moves_position_over_2900_km():
    lon, lat, height = 226.2, 18.65, 21_900.0
    mars = np.asarray(
        gis.CrsTransform.from_crs(
            "IAU:49902", "FORGE3D:499"
        ).transform_point3(lon, lat, height)
    )
    deliberately_wrong_wgs84 = np.asarray(crs.wgs84_to_ecef(lon, lat, height))
    assert np.linalg.norm(mars - deliberately_wrong_wgs84) > 2_900_000.0


def test_lola_scene_asset_is_typed_small_provenanced_and_tracked(tmp_path):
    asset = ROOT / "assets" / "tif" / "moon_south_pole_lola.tif"
    manifest_path = asset.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert asset.stat().st_size <= 5 * 1024 * 1024
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == manifest["asset_sha256"]

    info = gis.read_raster_info(asset)
    assert info.crs_authority == {"name": "IAU", "code": "30110"}
    assert (info.width, info.height) == (480, 480)
    assert info.height_system == "ellipsoidal"
    assert info.resolution == pytest.approx(
        (1895.2094015093426, 1895.2094015093426)
    )

    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "assets/tif/moon_south_pole_lola.tif",
            "assets/tif/moon_south_pole_lola.manifest.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    scene = _load_moon_example().build_scene(tmp_path / "moon.png")
    assert scene.recipe.terrain.crs == "IAU:30110"
    assert scene.recipe.target_crs == "IAU:30110"
    assert scene.recipe.terrain.metadata["body"] == "Moon"
    assert scene.recipe.terrain.metadata["body_radius_m"] == 1_737_400.0
    assert scene.validate().status == "ok"


def test_mars_areoid_matches_committed_pds_gmm3_map_below_half_metre():
    points = _areoid_reference_points()
    assert len(points) >= 20
    errors = [
        abs(crs.areoid_undulation(lat, lon) - expected)
        for lat, lon, expected, _source in points
    ]
    assert max(errors) < 0.5


@pytest.mark.parametrize(
    "lat,lon",
    [(90.1, 0.0), (-90.1, 0.0), (float("nan"), 0.0), (0.0, float("inf"))],
)
def test_mars_areoid_rejects_invalid_coordinates(lat, lon):
    with pytest.raises(ValueError):
        crs.areoid_undulation(lat, lon)


@pytest.mark.parametrize(
    "geographic,projected",
    [
        ("IAU:30100", "IAU:30110"),
        ("IAU:49900", "IAU:49910"),
        ("IAU:49902", "IAU:49912"),
    ],
)
def test_iau_equirectangular_global_grid_closes_below_nanodegree(
    geographic, projected
):
    forward = gis.CrsTransform.from_crs(geographic, projected)
    inverse = gis.CrsTransform.from_crs(projected, geographic)
    worst = 0.0
    for row in range(25):
        lat = -88.0 + 176.0 * row / 24.0
        for col in range(40):
            lon = -175.5 + 351.0 * col / 39.0
            x, y = forward.transform_point(lon, lat)
            recovered_lon, recovered_lat = inverse.transform_point(x, y)
            worst = max(
                worst,
                _longitude_error(recovered_lon, lon),
                abs(recovered_lat - lat),
            )
    assert worst < 1e-9


@pytest.mark.parametrize(
    "geographic,body_fixed",
    [("IAU:30100", "FORGE3D:301"), ("IAU:49902", "FORGE3D:499")],
)
def test_planetocentric_body_fixed_global_grid_closes_below_nanodegree(
    geographic, body_fixed
):
    forward = gis.CrsTransform.from_crs(geographic, body_fixed)
    inverse = gis.CrsTransform.from_crs(body_fixed, geographic)
    worst_degrees = 0.0
    worst_height = 0.0
    for row in range(25):
        lat = -88.0 + 176.0 * row / 24.0
        for col in range(40):
            lon = -175.5 + 351.0 * col / 39.0
            xyz = forward.transform_point3(lon, lat, 1234.5)
            recovered_lon, recovered_lat, recovered_height = inverse.transform_point3(
                *xyz
            )
            worst_degrees = max(
                worst_degrees,
                _longitude_error(recovered_lon, lon),
                abs(recovered_lat - lat),
            )
            worst_height = max(worst_height, abs(recovered_height - 1234.5))
    assert worst_degrees < 1e-9
    assert worst_height < 1e-6


def test_mars_spherical_and_ellipsoidal_iau_crs_keep_distinct_surfaces():
    sphere = gis.CrsTransform.from_crs("IAU:49900", "FORGE3D:499")
    ellipsoid = gis.CrsTransform.from_crs("IAU:49902", "FORGE3D:499")
    sphere_radius = np.linalg.norm(sphere.transform_point3(0.0, 45.0, 0.0))
    ellipsoid_radius = np.linalg.norm(ellipsoid.transform_point3(0.0, 45.0, 0.0))
    assert sphere_radius == pytest.approx(3_396_190.0, abs=1e-9)
    assert sphere_radius - ellipsoid_radius > 10_000.0


def test_mars_ocentric_projections_match_proj_iau_2015_oracle():
    equirectangular = gis.CrsTransform.from_crs("IAU:49902", "IAU:49912")
    polar = gis.CrsTransform.from_crs("IAU:49902", "IAU:49932")
    assert equirectangular.transform_point(10.0, 20.0) == pytest.approx(
        (592_746.9752330622, 1_198_439.5701681552), abs=1e-6
    )
    assert polar.transform_point(10.0, 80.0) == pytest.approx(
        (102_584.23432793329, -581_784.1031234111), abs=1e-6
    )


def test_python_custom_planetocentric_projection_dicts_are_routable():
    mars_eqc = {
        "method": "planetocentric_eqc",
        "a": 3_396_190.0,
        "inv_f": 169.894447223612,
        "lat0": 0.0,
        "lon0": 0.0,
        "lat_ts": 0.0,
        "false_easting": 0.0,
        "false_northing": 0.0,
    }
    forward = gis.CrsTransform.from_crs("IAU:49902", mars_eqc)
    inverse = gis.CrsTransform.from_crs(mars_eqc, "IAU:49902")
    projected = forward.transform_point(10.0, 20.0)
    assert projected == pytest.approx(
        (592_746.9752330622, 1_198_439.5701681552), abs=1e-6
    )
    assert inverse.transform_point(*projected) == pytest.approx((10.0, 20.0), abs=1e-12)

    rounded_mars = {**mars_eqc, "a": 3_396_190.000001, "inv_f": 169.8944472}
    gis.CrsTransform.from_crs("IAU:49902", rounded_mars)

    moon_eqc = {
        **mars_eqc,
        "method": "planetocentric_eqc",
        "a": 1_737_400.0,
        "inv_f": float("inf"),
    }
    moon = gis.CrsTransform.from_crs("IAU:30100", moon_eqc)
    assert moon.transform_point(10.0, 20.0) == pytest.approx(
        (303_233.5042414948, 606_467.0084829896), abs=1e-6
    )

    ambiguous_mars = {**mars_eqc, "method": "eqc", "inv_f": float("inf")}
    with pytest.raises(ValueError, match="planetocentric|custom"):
        gis.CrsTransform.from_crs("IAU:49900", ambiguous_mars)


@pytest.mark.parametrize(
    "geographic,north,south",
    [
        ("IAU:30100", "IAU:30130", "IAU:30135"),
        ("IAU:49900", "IAU:49930", "IAU:49935"),
    ],
)
def test_iau_polar_stereographic_round_trips(geographic, north, south):
    for code, lon, lat in [(north, 44.0, 80.0), (south, -73.0, -80.0)]:
        forward = gis.CrsTransform.from_crs(geographic, code)
        inverse = gis.CrsTransform.from_crs(code, geographic)
        x, y = forward.transform_point(lon, lat)
        recovered_lon, recovered_lat = inverse.transform_point(x, y)
        assert _longitude_error(recovered_lon, lon) < 1e-9
        assert abs(recovered_lat - lat) < 1e-9


def test_iau_dispatch_rejects_unknown_cross_body_and_mars_ographic_codes():
    with pytest.raises(ValueError, match=r"unsupported IAU code 30199"):
        gis.CrsTransform.from_crs("IAU:30100", "IAU:30199")
    with pytest.raises(ValueError, match=r"Moon.*Mars|Mars.*Moon"):
        gis.CrsTransform.from_crs("IAU:30100", "IAU:49910")
    with pytest.raises(ValueError, match=r"planetographic|ographic.*planetocentric"):
        gis.CrsTransform.from_crs("IAU:49901", "IAU:49910")


def test_unknown_iau_geotiff_requires_and_preserves_defining_wkt(tmp_path):
    path = tmp_path / "unknown_lunar_crs.tif"
    crs_spec = {
        "name": "IAU",
        "code": "30199",
        "wkt": 'PROJCRS["Unknown lunar projection"]',
    }
    written = gis.write_raster(
        path,
        np.arange(4, dtype=np.int16).reshape(2, 2),
        crs=crs_spec,
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )
    reread = gis.read_raster_info(path)
    expected_authority = {"name": "IAU", "code": "30199"}
    assert written.crs_authority == expected_authority
    assert reread.crs_authority == expected_authority
    assert reread.crs_wkt == crs_spec["wkt"]


def test_iau_wkt_body_consistency_and_like_path_equivalence(tmp_path):
    source = tmp_path / "moon_with_wkt.tif"
    moon_crs = {
        "name": "IAU",
        "code": "30115",
        "wkt": 'PROJCRS["Moon"]',
    }
    gis.write_raster(
        source,
        np.arange(4, dtype=np.int16).reshape(2, 2),
        crs=moon_crs,
        transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )
    copied = gis.write_raster(
        tmp_path / "copied.tif",
        np.arange(4, dtype=np.int16).reshape(2, 2),
        crs="IAU:30115",
        like_path=source,
    )
    assert copied.crs_authority == {"name": "IAU", "code": "30115"}

    with pytest.raises(ValueError, match="different planetary body"):
        gis.write_raster(
            tmp_path / "contradictory.tif",
            np.arange(4, dtype=np.int16).reshape(2, 2),
            crs={
                "name": "IAU",
                "code": "30115",
                "wkt": 'PROJCRS["Mars"]',
            },
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
    with pytest.raises(ValueError, match="different planetary body"):
        gis.write_raster(
            tmp_path / "earth_wkt.tif",
            np.arange(4, dtype=np.int16).reshape(2, 2),
            crs={
                "name": "IAU",
                "code": "30115",
                "wkt": 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984"]]',
            },
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
    with pytest.raises(ValueError, match="WKT kind"):
        gis.write_raster(
            tmp_path / "wrong_kind.tif",
            np.arange(4, dtype=np.int16).reshape(2, 2),
            crs={
                "name": "IAU",
                "code": "30115",
                "wkt": 'GEOGCRS["Moon"]',
            },
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )


def test_iau_raster_authority_survives_geotiff_round_trip(tmp_path):
    path = tmp_path / "moon_iau.tif"
    written = gis.write_raster(
        path,
        np.arange(4, dtype=np.int16).reshape(2, 2),
        crs="IAU:30115",
        transform=(0.5, 0.0, -180.0, 0.0, -0.5, -80.0),
    )
    reread = gis.read_raster_info(path)
    expected = {"name": "IAU", "code": "30115"}
    assert written.crs_authority == expected
    assert reread.crs_authority == expected
