"""Regenerate the offline HELIOS WhiteboxTools reference raster."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import forge3d
import numpy as np
from pyproj import Geod
import rasterio
from rasterio.transform import from_origin
import shapefile


BOUNDS = (-0.5, -0.5, 0.5, 0.5)
SIZE = 256
REFRACTION_K = 0.13


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitebox-tools", required=True, type=Path)
    parser.add_argument(
        "--earth-model",
        required=True,
        choices=("flat", "effective-radius"),
    )
    parser.add_argument("--observer-height", required=True, type=float)
    parser.add_argument("--observer-row", required=True, type=int)
    parser.add_argument("--observer-column", required=True, type=int)
    parser.add_argument(
        "--output",
        type=Path,
    )
    args = parser.parse_args()
    output = args.output or Path(__file__).with_name(
        "whitebox_curved_analytic_256.png"
        if args.earth_model == "effective-radius"
        else "whitebox_flat_analytic_256.png"
    )
    ellipsoidal = np.zeros((SIZE, SIZE), dtype=np.float32)
    row, column = args.observer_row, args.observer_column
    if not (0 <= row < SIZE and 0 <= column < SIZE):
        raise ValueError("observer row/column must be inside the reference raster")
    latitude = BOUNDS[3] - (row + 0.5) * (BOUNDS[3] - BOUNDS[1]) / SIZE
    longitude = BOUNDS[0] + (column + 0.5) * (BOUNDS[2] - BOUNDS[0]) / SIZE
    longitude_step = (BOUNDS[2] - BOUNDS[0]) / SIZE
    latitude_step = (BOUNDS[3] - BOUNDS[1]) / SIZE
    geod = Geod(ellps="WGS84")
    _, _, cell_width_m = geod.inv(
        longitude,
        latitude,
        longitude + longitude_step,
        latitude,
    )
    _, _, cell_height_m = geod.inv(
        longitude,
        latitude,
        longitude,
        latitude + latitude_step,
    )
    target_width = target_height = SIZE
    target_transform = from_origin(
        -(column + 0.5) * cell_width_m,
        (row + 0.5) * cell_height_m,
        cell_width_m,
        cell_height_m,
    )
    target_crs = rasterio.crs.CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} "
        "+datum=WGS84 +units=m +no_defs"
    )
    projected = ellipsoidal.copy()

    # Whitebox Viewshed is a flat projected LOS implementation. Feeding it
    # h_eff makes its independent horizon algorithm an EffectiveRadius(k)
    # reference without claiming that Whitebox itself models curvature.
    target_rows, target_columns = np.indices((SIZE, SIZE))
    target_lon = BOUNDS[0] + (target_columns.ravel() + 0.5) * longitude_step
    target_lat = BOUNDS[3] - (target_rows.ravel() + 0.5) * latitude_step
    azimuth_deg, _, distance_m = geod.inv(
        np.full(target_width * target_height, longitude),
        np.full(target_width * target_height, latitude),
        np.asarray(target_lon),
        np.asarray(target_lat),
    )
    latitude_rad = np.deg2rad(latitude)
    azimuth_rad = np.deg2rad(azimuth_deg)
    eccentricity_squared = 6.694_379_990_141_316_5e-3
    semi_major_m = 6_378_137.0
    w = np.sqrt(1.0 - eccentricity_squared * np.sin(latitude_rad) ** 2)
    meridional_m = semi_major_m * (1.0 - eccentricity_squared) / w**3
    prime_vertical_m = semi_major_m / w
    inverse_radius = (
        np.cos(azimuth_rad) ** 2 / meridional_m
        + np.sin(azimuth_rad) ** 2 / prime_vertical_m
    )
    if args.earth_model == "effective-radius":
        effective_drop_m = (
            0.5
            * (1.0 - REFRACTION_K)
            * inverse_radius
            * np.asarray(distance_m) ** 2
        ).reshape((target_height, target_width))
        projected -= effective_drop_m.astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="forge3d-helios-whitebox-") as directory:
        work = Path(directory)
        with rasterio.open(
            work / "dem.tif",
            "w",
            driver="GTiff",
            width=target_width,
            height=target_height,
            count=1,
            dtype="float32",
            crs=target_crs,
            transform=target_transform,
        ) as dataset:
            dataset.write(projected, 1)
        writer = shapefile.Writer(str(work / "station"))
        writer.field("id", "N")
        writer.point(0.0, 0.0)
        writer.record(1)
        writer.close()
        (work / "station.prj").write_text(
            target_crs.to_wkt(), encoding="utf-8"
        )
        command = [
            str(args.whitebox_tools.resolve()),
            "--run=Viewshed",
            f"--wd={work}",
            "--dem=dem.tif",
            "--stations=station.shp",
            "--output=whitebox.tif",
            f"--height={args.observer_height}",
            "--compress_rasters=False",
        ]
        print(subprocess.list2cmdline(command))
        subprocess.run(command, check=True)
        with rasterio.open(work / "whitebox.tif") as dataset:
            reference = dataset.read(1)

    gray = (reference > 0).astype(np.uint8) * 255
    rgba = np.dstack((gray, gray, gray, np.full_like(gray, 255)))
    output.parent.mkdir(parents=True, exist_ok=True)
    forge3d.numpy_to_png(output, rgba)


if __name__ == "__main__":
    main()
