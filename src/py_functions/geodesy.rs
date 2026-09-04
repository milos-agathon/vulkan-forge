// src/py_functions/geodesy.rs
// MENSURA Python surface: EGM96 geoid undulation, Karney geodesics, and the
// full-f64 geodetic ⇄ geocentric (ECEF) conversion.
// RELEVANT FILES: src/geo/geoid.rs, src/geo/geodesic.rs, src/geo/projections/geocentric.rs

use super::super::*;

#[cfg(feature = "extension-module")]
use numpy::{PyArray2, PyReadonlyArray2};

#[cfg(feature = "extension-module")]
fn solar_time_field<'py>(value: &Bound<'py, PyAny>, name: &str) -> Option<Bound<'py, PyAny>> {
    if let Ok(dict) = value.downcast::<PyDict>() {
        return dict.get_item(name).ok().flatten();
    }
    value
        .getattr(name)
        .ok()
        .or_else(|| value.get_item(name).ok())
}

#[cfg(feature = "extension-module")]
fn solar_time_from_dict(value: &Bound<'_, PyAny>) -> PyResult<crate::geo::solar::SolarTime> {
    let required = |name: &str| {
        solar_time_field(value, name)
            .ok_or_else(|| PyValueError::new_err(format!("solar_time is missing {name:?}")))
    };
    let (year, month, day, hour, minute, second) = required("utc")?
        .extract::<(i32, u32, u32, u32, u32, f64)>()
        .map_err(|_| {
            PyValueError::new_err(
                "solar_time['utc'] must be (year, month, day, hour, minute, second)",
            )
        })?;
    let delta_t_seconds = match solar_time_field(value, "delta_t") {
        Some(alias) if !alias.is_none() => alias.extract()?,
        _ => {
            solar_time_field(value, "delta_t_seconds").map_or(Ok(69.0), |value| value.extract())?
        }
    };
    Ok(crate::geo::solar::SolarTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
        tz_offset_hours: solar_time_field(value, "tz_offset_hours")
            .map_or(Ok(0.0), |value| value.extract())?,
        delta_t_seconds,
        latitude_deg: required("observer_lat")?.extract()?,
        longitude_deg: required("observer_lon")?.extract()?,
        elevation_m: solar_time_field(value, "observer_elev_m")
            .map_or(Ok(0.0), |value| value.extract())?,
        pressure_mbar: solar_time_field(value, "pressure_mbar")
            .map_or(Ok(1013.25), |value| value.extract())?,
        temperature_c: solar_time_field(value, "temperature_c")
            .map_or(Ok(15.0), |value| value.extract())?,
    })
}

#[cfg(feature = "extension-module")]
fn terrain_grid_heights(
    dem: &PyReadonlyArray2<'_, f32>,
    bounds: (f64, f64, f64, f64),
    height_system: &str,
) -> PyResult<(Vec<f32>, usize, usize, f64, f64, f64, f64)> {
    let (left, bottom, right, top) = bounds;
    let right_unwrapped = if right <= left { right + 360.0 } else { right };
    let longitude_span = right_unwrapped - left;
    if !(-180.0..=180.0).contains(&left)
        || !(-180.0..=180.0).contains(&right)
        || !(-90.0..=90.0).contains(&bottom)
        || !(-90.0..=90.0).contains(&top)
        || !(longitude_span > 0.0 && longitude_span < 180.0 && top > bottom)
    {
        return Err(PyValueError::new_err(
            "bounds must be a local EPSG:4326 extent spanning less than 180 degrees",
        ));
    }
    let array = dem.as_array();
    let (height, width) = (array.nrows(), array.ncols());
    if width < 2 || height < 2 || width > u32::MAX as usize || height > u32::MAX as usize {
        return Err(PyValueError::new_err("dem must be at least 2x2"));
    }
    if array.iter().any(|height| !height.is_finite()) {
        return Err(PyValueError::new_err("dem heights must be finite"));
    }
    let longitude_step = longitude_span / width as f64;
    let latitude_step = (top - bottom) / height as f64;
    let heights = match height_system {
        "ellipsoidal" => array.iter().copied().collect(),
        "orthometric_egm96" => {
            let mut converted = Vec::with_capacity(width * height);
            for row in 0..height {
                let latitude = top - (row as f64 + 0.5) * latitude_step;
                for column in 0..width {
                    let longitude_unwrapped =
                        left + (column as f64 + 0.5) * longitude_step;
                    let longitude = (longitude_unwrapped + 180.0).rem_euclid(360.0) - 180.0;
                    converted.push(
                        array[(row, column)]
                            + crate::geo::geoid::undulation_deg(latitude, longitude) as f32,
                    );
                }
            }
            converted
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported height_system {height_system:?}; expected 'ellipsoidal' or 'orthometric_egm96'"
            )))
        }
    };
    Ok((
        heights,
        width,
        height,
        right_unwrapped,
        longitude_step,
        latitude_step,
        top,
    ))
}

/// NREL SPA topocentric solar vector.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    utc,
    lat,
    lon,
    elev_m = 0.0,
    *,
    tz_offset_hours = 0.0,
    delta_t_seconds = 69.0,
    pressure_mbar = 1013.25,
    temperature_c = 15.0
))]
pub(crate) fn solar_position(
    py: Python<'_>,
    utc: &Bound<'_, PyAny>,
    lat: f64,
    lon: f64,
    elev_m: f64,
    tz_offset_hours: f64,
    delta_t_seconds: f64,
    pressure_mbar: f64,
    temperature_c: f64,
) -> PyResult<PyObject> {
    let (year, month, day, hour, minute, second) = utc
        .extract::<(i32, u32, u32, u32, u32, f64)>()
        .map_err(|_| {
            PyValueError::new_err("utc must be (year, month, day, hour, minute, second)")
        })?;
    let vector = crate::geo::solar::solar_position(&crate::geo::solar::SolarTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
        tz_offset_hours,
        delta_t_seconds,
        latitude_deg: lat,
        longitude_deg: lon,
        elevation_m: elev_m,
        pressure_mbar,
        temperature_c,
    })
    .map_err(PyValueError::new_err)?;
    let dict = PyDict::new_bound(py);
    dict.set_item("zenith_deg", vector.zenith_deg)?;
    dict.set_item("azimuth_deg", vector.azimuth_deg)?;
    dict.set_item("apparent_elevation_deg", vector.apparent_elevation_deg)?;
    dict.set_item("true_elevation_deg", vector.true_elevation_deg)?;
    dict.set_item("distance_au", vector.distance_au)?;
    dict.set_item("equation_of_time_min", vector.equation_of_time_min)?;
    Ok(dict.into_py(py))
}

/// Curvature- and refraction-aware GPU viewshed over a north-up EPSG:4326 DEM.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    dem,
    observer,
    bounds,
    height_system,
    *,
    observer_height = 1.7,
    target_height = 0.0,
    max_distance = None,
    earth_model = "ellipsoid",
    sphere_radius_m = 6_371_008.8,
    refraction_model = "bennett",
    refraction_k = 0.13,
    pressure_mbar = 1013.25,
    temperature_c = 15.0
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn terrain_viewshed<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    observer: (f64, f64),
    bounds: (f64, f64, f64, f64),
    height_system: &str,
    observer_height: f64,
    target_height: f64,
    max_distance: Option<f64>,
    earth_model: &str,
    sphere_radius_m: f64,
    refraction_model: &str,
    refraction_k: f64,
    pressure_mbar: f64,
    temperature_c: f64,
) -> PyResult<PyObject> {
    let (observer_lat, observer_lon) = observer;
    let (left, bottom, right, top) = bounds;
    let right_unwrapped = if right <= left { right + 360.0 } else { right };
    let longitude_span = right_unwrapped - left;
    let mut observer_lon_unwrapped = observer_lon;
    if observer_lon_unwrapped < left {
        observer_lon_unwrapped += 360.0;
    }
    if !(-90.0..=90.0).contains(&observer_lat)
        || !(-180.0..=180.0).contains(&observer_lon)
        || !(-180.0..=180.0).contains(&left)
        || !(-180.0..=180.0).contains(&right)
        || !(longitude_span > 0.0 && top > bottom)
        || longitude_span >= 180.0
        || top - bottom >= 180.0
        || !(-90.0..=90.0).contains(&bottom)
        || !(-90.0..=90.0).contains(&top)
        || observer_lon_unwrapped < left
        || observer_lon_unwrapped > right_unwrapped
        || observer_lat < bottom
        || observer_lat > top
    {
        return Err(PyValueError::new_err(
            "observer must be finite and inside local EPSG:4326 bounds spanning less than 180 degrees",
        ));
    }
    let array = dem.as_array();
    let (height, width) = (array.nrows(), array.ncols());
    if width < 2 || height < 2 || width > u32::MAX as usize || height > u32::MAX as usize {
        return Err(PyValueError::new_err(
            "dem must be a two-dimensional array at least 2x2",
        ));
    }
    let geodesic = crate::geo::geodesic::Geodesic::wgs84();
    let longitude_step = longitude_span / width as f64;
    let latitude_step = (top - bottom) / height as f64;
    let normalized_observer_lon = (observer_lon + 180.0).rem_euclid(360.0) - 180.0;
    let earth =
        crate::geo::refraction::EarthModel::from_name(earth_model, observer_lat, sphere_radius_m)
            .map_err(PyValueError::new_err)?;
    let mut positions_m = Vec::with_capacity(width * height);
    let mut farthest_m = 0.0_f64;
    for row in 0..height {
        let latitude = top - (row as f64 + 0.5) * latitude_step;
        for column in 0..width {
            let longitude_unwrapped = left + (column as f64 + 0.5) * longitude_step;
            let longitude = (longitude_unwrapped + 180.0).rem_euclid(360.0) - 180.0;
            let (distance_m, azimuth) = match earth {
                crate::geo::refraction::EarthModel::Sphere { radius_m } => {
                    let latitude_1 = observer_lat.to_radians();
                    let latitude_2 = latitude.to_radians();
                    let longitude_delta = (longitude - normalized_observer_lon).to_radians();
                    let central_angle = (latitude_1.sin() * latitude_2.sin()
                        + latitude_1.cos() * latitude_2.cos() * longitude_delta.cos())
                    .clamp(-1.0, 1.0)
                    .acos();
                    let azimuth = (longitude_delta.sin() * latitude_2.cos()).atan2(
                        latitude_1.cos() * latitude_2.sin()
                            - latitude_1.sin() * latitude_2.cos() * longitude_delta.cos(),
                    );
                    (radius_m * central_angle, azimuth)
                }
                _ => {
                    let inverse = geodesic.inverse(
                        observer_lat,
                        normalized_observer_lon,
                        latitude,
                        longitude,
                    );
                    (inverse.s12, inverse.azi1.to_radians())
                }
            };
            positions_m.push([
                (distance_m * azimuth.sin()) as f32,
                (distance_m * azimuth.cos()) as f32,
            ]);
            farthest_m = farthest_m.max(distance_m);
        }
    }
    let max_distance_m = max_distance.unwrap_or(farthest_m);
    let refraction = crate::geo::refraction::RefractionModel::from_name(
        refraction_model,
        pressure_mbar,
        temperature_c,
        refraction_k,
    )
    .map_err(PyValueError::new_err)?;
    let options = crate::terrain::analysis::viewshed::ViewshedOptions {
        width: width as u32,
        height: height as u32,
        observer_x: ((observer_lon_unwrapped - left) / longitude_step - 0.5) as f32,
        observer_y: ((top - observer_lat) / latitude_step - 0.5) as f32,
        observer_height_m: observer_height as f32,
        target_height_m: target_height as f32,
        max_distance_m: max_distance_m as f32,
        observer_latitude_rad: observer_lat.to_radians() as f32,
        observer_longitude_rad: normalized_observer_lon.to_radians() as f32,
        left_unwrapped_deg: left as f32,
        top_deg: top as f32,
        longitude_step_deg: longitude_step as f32,
        latitude_step_deg: latitude_step as f32,
        geodesic_sphere_radius_m: match earth {
            crate::geo::refraction::EarthModel::Sphere { radius_m } => radius_m as f32,
            _ => 0.0,
        },
        earth_model: earth,
        refraction_model: refraction,
    };
    let heights: Vec<f32> = match height_system {
        "ellipsoidal" => array.iter().copied().collect(),
        "orthometric_egm96" => {
            let mut converted = Vec::with_capacity(width * height);
            for row in 0..height {
                let latitude = top - (row as f64 + 0.5) * latitude_step;
                for column in 0..width {
                    let longitude_unwrapped =
                        left + (column as f64 + 0.5) * longitude_step;
                    let longitude = (longitude_unwrapped + 180.0).rem_euclid(360.0) - 180.0;
                    converted.push(
                        array[(row, column)]
                            + crate::geo::geoid::undulation_deg(latitude, longitude) as f32,
                    );
                }
            }
            converted
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported height_system {height_system:?}; expected 'ellipsoidal' or 'orthometric_egm96'"
            )))
        }
    };
    let output =
        crate::terrain::analysis::viewshed::compute_viewshed(&heights, &positions_m, &options)
            .map_err(|error| PyRuntimeError::new_err(format!("viewshed failed: {error}")))?;
    let shape = (height, width);
    let visibility = ndarray::Array2::from_shape_vec(
        shape,
        output.visibility.into_iter().map(u8::from).collect(),
    )
    .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let curvature = ndarray::Array2::from_shape_vec(shape, output.curvature_drop_m)
        .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let refraction = ndarray::Array2::from_shape_vec(shape, output.refraction_gain_m)
        .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let horizon = ndarray::Array2::from_shape_vec(shape, output.horizon_distance_m)
        .map_err(|error| PyRuntimeError::new_err(format!("viewshed shape failed: {error}")))?;
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "visibility",
        PyArray2::from_owned_array_bound(py, visibility),
    )?;
    dict.set_item(
        "curvature_drop_m",
        PyArray2::from_owned_array_bound(py, curvature),
    )?;
    dict.set_item(
        "refraction_gain_m",
        PyArray2::from_owned_array_bound(py, refraction),
    )?;
    dict.set_item(
        "horizon_distance_m",
        PyArray2::from_owned_array_bound(py, horizon),
    )?;
    Ok(dict.into_py(py))
}

/// Direct curvature-aware terrain-to-sun visibility; true pixels are lit.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    dem,
    solar_time,
    bounds,
    height_system,
    *,
    earth_model = "ellipsoid",
    sphere_radius_m = 6_371_008.8,
    refraction_model = "bennett",
    refraction_k = 0.13
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn terrain_shadow_mask<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f32>,
    solar_time: &Bound<'py, PyAny>,
    bounds: (f64, f64, f64, f64),
    height_system: &str,
    earth_model: &str,
    sphere_radius_m: f64,
    refraction_model: &str,
    refraction_k: f64,
) -> PyResult<PyObject> {
    let time = solar_time_from_dict(solar_time)?;
    let (heights, width, height, right_unwrapped, longitude_step, latitude_step, top) =
        terrain_grid_heights(&dem, bounds, height_system)?;
    let (left, bottom, _, _) = bounds;
    let earth = crate::geo::refraction::EarthModel::from_name(
        earth_model,
        time.latitude_deg,
        sphere_radius_m,
    )
    .map_err(PyValueError::new_err)?;
    let refraction = crate::geo::refraction::RefractionModel::from_name(
        refraction_model,
        time.pressure_mbar,
        time.temperature_c,
        refraction_k,
    )
    .map_err(PyValueError::new_err)?;

    let mut geodetic_positions_and_sun = Vec::with_capacity(width * height);
    for row in 0..height {
        let latitude = top - (row as f64 + 0.5) * latitude_step;
        for column in 0..width {
            let longitude_unwrapped = left + (column as f64 + 0.5) * longitude_step;
            let longitude = (longitude_unwrapped + 180.0).rem_euclid(360.0) - 180.0;
            let solar = crate::geo::solar::solar_position(&crate::geo::solar::SolarTime {
                latitude_deg: latitude,
                longitude_deg: longitude,
                elevation_m: heights[row * width + column] as f64,
                ..time
            })
            .map_err(PyValueError::new_err)?;
            let launch_elevation_deg =
                if matches!(refraction, crate::geo::refraction::RefractionModel::None) {
                    solar.true_elevation_deg
                } else {
                    // HELIOS defines refracted shadows with SPA's apparent launch
                    // angle; EffectiveRadius supplies the distance correction.
                    solar.apparent_elevation_deg
                };
            geodetic_positions_and_sun.push([
                latitude.to_radians() as f32,
                longitude.to_radians() as f32,
                solar.azimuth_deg.to_radians() as f32,
                launch_elevation_deg.to_radians() as f32,
            ]);
        }
    }
    let right = (right_unwrapped + 180.0).rem_euclid(360.0) - 180.0;
    let diagonal_m = match earth {
        crate::geo::refraction::EarthModel::Sphere { radius_m } => {
            let central_angle = |lat_1: f64, lon_1: f64, lat_2: f64, lon_2: f64| {
                let (lat_1, lat_2) = (lat_1.to_radians(), lat_2.to_radians());
                let longitude_delta = (lon_2 - lon_1).to_radians();
                (lat_1.sin() * lat_2.sin() + lat_1.cos() * lat_2.cos() * longitude_delta.cos())
                    .clamp(-1.0, 1.0)
                    .acos()
            };
            radius_m
                * central_angle(top, left, bottom, right)
                    .max(central_angle(top, right, bottom, left))
        }
        _ => crate::geo::geodesic::Geodesic::wgs84()
            .inverse(top, left, bottom, right)
            .s12
            .max(
                crate::geo::geodesic::Geodesic::wgs84()
                    .inverse(top, right, bottom, left)
                    .s12,
            ),
    };
    let options = crate::terrain::analysis::viewshed::ViewshedOptions {
        width: width as u32,
        height: height as u32,
        observer_x: 0.0,
        observer_y: 0.0,
        observer_height_m: 0.0,
        target_height_m: 0.0,
        max_distance_m: (diagonal_m * 1.01) as f32,
        observer_latitude_rad: time.latitude_deg.to_radians() as f32,
        observer_longitude_rad: time.longitude_deg.to_radians() as f32,
        left_unwrapped_deg: left as f32,
        top_deg: top as f32,
        longitude_step_deg: longitude_step as f32,
        latitude_step_deg: latitude_step as f32,
        geodesic_sphere_radius_m: match earth {
            crate::geo::refraction::EarthModel::Sphere { radius_m } => radius_m as f32,
            _ => 0.0,
        },
        earth_model: earth,
        refraction_model: refraction,
    };
    let output = crate::terrain::analysis::viewshed::compute_shadow_mask(
        &heights,
        &geodetic_positions_and_sun,
        &options,
    )
    .map_err(|error| PyRuntimeError::new_err(format!("shadow mask failed: {error}")))?;
    let mask = ndarray::Array2::from_shape_vec(
        (height, width),
        output.into_iter().map(u8::from).collect(),
    )
    .map_err(|error| PyRuntimeError::new_err(format!("shadow mask shape failed: {error}")))?;
    Ok(PyArray2::from_owned_array_bound(py, mask).into_py(py))
}

/// Closed-form curved-Earth mountain shadow terminus.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (
    dem,
    peak_lat,
    peak_lon,
    solar_time,
    bounds,
    height_system,
    *,
    earth_model = "ellipsoid",
    sphere_radius_m = 6_371_008.8,
    refraction_model = "bennett",
    refraction_k = 0.13
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn terrain_shadow_tip(
    py: Python<'_>,
    dem: PyReadonlyArray2<'_, f32>,
    peak_lat: f64,
    peak_lon: f64,
    solar_time: &Bound<'_, PyAny>,
    bounds: (f64, f64, f64, f64),
    height_system: &str,
    earth_model: &str,
    sphere_radius_m: f64,
    refraction_model: &str,
    refraction_k: f64,
) -> PyResult<PyObject> {
    let time = solar_time_from_dict(solar_time)?;
    let (heights, width, height, right_unwrapped, longitude_step, latitude_step, top) =
        terrain_grid_heights(&dem, bounds, height_system)?;
    let (left, bottom, _, _) = bounds;
    let mut peak_lon_unwrapped = peak_lon;
    if peak_lon_unwrapped < left {
        peak_lon_unwrapped += 360.0;
    }
    if !peak_lat.is_finite()
        || !peak_lon.is_finite()
        || peak_lat < bottom
        || peak_lat > top
        || peak_lon_unwrapped < left
        || peak_lon_unwrapped > right_unwrapped
    {
        return Err(PyValueError::new_err(
            "peak must be finite and inside bounds",
        ));
    }
    let pixel_x = (peak_lon_unwrapped - left) / longitude_step - 0.5;
    let pixel_y = (top - peak_lat) / latitude_step - 0.5;
    let x0 = pixel_x.floor().clamp(0.0, (width - 1) as f64) as usize;
    let y0 = pixel_y.floor().clamp(0.0, (height - 1) as f64) as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = (pixel_x - x0 as f64).clamp(0.0, 1.0);
    let fy = (pixel_y - y0 as f64).clamp(0.0, 1.0);
    let at = |row: usize, column: usize| f64::from(heights[row * width + column]);
    let peak_height_m = (at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx) * (1.0 - fy)
        + (at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx) * fy;
    if peak_height_m <= 0.0 {
        return Err(PyValueError::new_err(
            "peak elevation must be positive in the selected height system",
        ));
    }
    let solar = crate::geo::solar::solar_position(&crate::geo::solar::SolarTime {
        latitude_deg: peak_lat,
        longitude_deg: peak_lon,
        elevation_m: peak_height_m,
        ..time
    })
    .map_err(PyValueError::new_err)?;
    let bearing_deg = (solar.azimuth_deg + 180.0).rem_euclid(360.0);
    let earth =
        crate::geo::refraction::EarthModel::from_name(earth_model, peak_lat, sphere_radius_m)
            .map_err(PyValueError::new_err)?;
    let refraction = crate::geo::refraction::RefractionModel::from_name(
        refraction_model,
        time.pressure_mbar,
        time.temperature_c,
        refraction_k,
    )
    .map_err(PyValueError::new_err)?;
    let launch_elevation_deg =
        if matches!(refraction, crate::geo::refraction::RefractionModel::None) {
            solar.true_elevation_deg
        } else {
            // The HELIOS DoD defines refracted L using α_app together with R′.
            solar.apparent_elevation_deg
        };
    if launch_elevation_deg <= 0.0 {
        return Err(PyValueError::new_err(
            "shadow tip is undefined when the selected solar launch angle is at or below the horizon",
        ));
    }
    let effective_radius_m =
        crate::geo::refraction::effective_radius_m(earth, refraction, bearing_deg)
            .map_err(PyValueError::new_err)?;
    let tangent = launch_elevation_deg.to_radians().tan();
    let length_m = if effective_radius_m.is_infinite() {
        peak_height_m / tangent
    } else {
        let discriminant = tangent * tangent - 2.0 * peak_height_m / effective_radius_m;
        if discriminant <= 0.0 {
            return Err(PyValueError::new_err(
                "shadow ray does not re-intersect the effective Earth before the horizon",
            ));
        }
        2.0 * peak_height_m / (tangent + discriminant.sqrt())
    };
    let (tip_lat, tip_lon) = match earth {
        crate::geo::refraction::EarthModel::Sphere { radius_m } => {
            let angular = length_m / radius_m;
            let latitude = peak_lat.to_radians();
            let longitude = peak_lon.to_radians();
            let bearing = bearing_deg.to_radians();
            let tip_latitude = (latitude.sin() * angular.cos()
                + latitude.cos() * angular.sin() * bearing.cos())
            .clamp(-1.0, 1.0)
            .asin();
            let tip_longitude = longitude
                + (bearing.sin() * angular.sin() * latitude.cos())
                    .atan2(angular.cos() - latitude.sin() * tip_latitude.sin());
            (
                tip_latitude.to_degrees(),
                (tip_longitude.to_degrees() + 180.0).rem_euclid(360.0) - 180.0,
            )
        }
        _ => {
            let direct = crate::geo::geodesic::Geodesic::wgs84().direct(
                peak_lat,
                peak_lon,
                bearing_deg,
                length_m,
            );
            (direct.lat2, direct.lon2)
        }
    };
    let dict = PyDict::new_bound(py);
    dict.set_item("bearing_deg", bearing_deg)?;
    dict.set_item("length_m", length_m)?;
    dict.set_item("tip_lat", tip_lat)?;
    dict.set_item("tip_lon", tip_lon)?;
    dict.set_item("peak_height_m", peak_height_m)?;
    dict.set_item("effective_radius_m", effective_radius_m)?;
    dict.set_item("solar_azimuth_deg", solar.azimuth_deg)?;
    dict.set_item("solar_apparent_elevation_deg", solar.apparent_elevation_deg)?;
    dict.set_item("solar_launch_elevation_deg", launch_elevation_deg)?;
    Ok(dict.into_py(py))
}

/// Return the registered datum constants for Earth, Moon, or Mars.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (name))]
pub(crate) fn body_info(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let body = crate::geo::body::body(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("name", body.name)?;
    dict.set_item("semi_major_m", body.ellipsoid.a)?;
    dict.set_item("semi_minor_m", body.ellipsoid.b())?;
    dict.set_item("flattening", body.ellipsoid.f)?;
    dict.set_item("prime_meridian_w0_deg", body.prime_meridian_w0)?;
    dict.set_item("rotation_rate_deg_per_day", body.rotation_rate)?;
    dict.set_item(
        "gravity_surface",
        body.gravity_surface.map(|surface| surface.name()),
    )?;
    Ok(dict.into_py(py))
}

/// EGM96 geoid undulation N(lat, lon) in metres (degree/order 120 synthesis,
/// NGA F477 convention, WGS84 ellipsoid).
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat, lon))]
pub(crate) fn geoid_undulation(lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "invalid_argument: latitude must be in [-90, 90] and longitude finite, got ({lat}, {lon})"
        )));
    }
    Ok(crate::geo::geoid::undulation_deg(lat, lon))
}

/// GMM3 Mars areoid undulation above its reference ellipsoid, metres.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat, lon))]
pub(crate) fn areoid_undulation(lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "invalid_argument: latitude must be in [-90, 90] and longitude finite, got ({lat}, {lon})"
        )));
    }
    Ok(crate::geo::geoid::areoid_undulation_deg(lat, lon))
}

/// Convert an orthometric (EGM96) height to an ellipsoidal height:
/// h = H + N(lat, lon). Returns metres.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (h_orthometric, lat, lon))]
pub(crate) fn orthometric_to_ellipsoidal(h_orthometric: f64, lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() || !h_orthometric.is_finite() {
        return Err(PyValueError::new_err(
            "invalid_argument: height must be finite, latitude in [-90, 90], longitude finite"
                .to_string(),
        ));
    }
    use crate::geo::units::{Angle, Height};
    Ok(crate::geo::geoid::orthometric_to_ellipsoidal(
        Height::new(h_orthometric),
        Angle::new(lat),
        Angle::new(lon),
    )
    .metres())
}

/// Convert an ellipsoidal height to an orthometric (EGM96) height:
/// H = h − N(lat, lon). Returns metres.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (h_ellipsoidal, lat, lon))]
pub(crate) fn ellipsoidal_to_orthometric(h_ellipsoidal: f64, lat: f64, lon: f64) -> PyResult<f64> {
    if !(-90.0..=90.0).contains(&lat) || !lon.is_finite() || !h_ellipsoidal.is_finite() {
        return Err(PyValueError::new_err(
            "invalid_argument: height must be finite, latitude in [-90, 90], longitude finite"
                .to_string(),
        ));
    }
    use crate::geo::units::{Angle, Height};
    Ok(crate::geo::geoid::ellipsoidal_to_orthometric(
        Height::new(h_ellipsoidal),
        Angle::new(lat),
        Angle::new(lon),
    )
    .metres())
}

/// Karney inverse geodesic on a registered planetary ellipsoid: distance and
/// azimuths between two points. Returns
/// {"s12": m, "azi1": deg, "azi2": deg, "a12": deg}.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat1, lon1, lat2, lon2, *, body = "earth"))]
pub(crate) fn geodesic_inverse(
    py: Python<'_>,
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    body: &str,
) -> PyResult<PyObject> {
    for (name, lat) in [("lat1", lat1), ("lat2", lat2)] {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(PyValueError::new_err(format!(
                "invalid_argument: {name} must be in [-90, 90], got {lat}"
            )));
        }
    }
    if !lon1.is_finite() || !lon2.is_finite() {
        return Err(PyValueError::new_err(
            "invalid_argument: longitudes must be finite".to_string(),
        ));
    }
    let body = crate::geo::body::body(body).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let g = crate::geo::geodesic::Geodesic::new(&body.ellipsoid);
    let r = g.inverse(lat1, lon1, lat2, lon2);
    let dict = PyDict::new_bound(py);
    dict.set_item("s12", r.s12)?;
    dict.set_item("azi1", r.azi1)?;
    dict.set_item("azi2", r.azi2)?;
    dict.set_item("a12", r.a12)?;
    Ok(dict.into_py(py))
}

/// Karney direct geodesic on a registered planetary ellipsoid: destination
/// from start point, azimuth, and distance. Returns
/// {"lat2": deg, "lon2": deg, "azi2": deg, "a12": deg}.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lat1, lon1, azi1, s12, *, body = "earth"))]
pub(crate) fn geodesic_direct(
    py: Python<'_>,
    lat1: f64,
    lon1: f64,
    azi1: f64,
    s12: f64,
    body: &str,
) -> PyResult<PyObject> {
    if !(-90.0..=90.0).contains(&lat1) || !lon1.is_finite() || !azi1.is_finite() || !s12.is_finite()
    {
        return Err(PyValueError::new_err(
            "invalid_argument: lat1 must be in [-90, 90]; lon1/azi1/s12 must be finite".to_string(),
        ));
    }
    let body = crate::geo::body::body(body).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let g = crate::geo::geodesic::Geodesic::new(&body.ellipsoid);
    let r = g.direct(lat1, lon1, azi1, s12);
    let dict = PyDict::new_bound(py);
    dict.set_item("lat2", r.lat2)?;
    dict.set_item("lon2", r.lon2)?;
    dict.set_item("azi2", r.azi2)?;
    dict.set_item("a12", r.a12)?;
    Ok(dict.into_py(py))
}

/// WGS84 geodetic (lon, lat in degrees, ELLIPSOIDAL height in metres) →
/// geocentric ECEF metres, full f64 (EPSG method 9602).
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (lon, lat, h = 0.0))]
pub(crate) fn wgs84_to_ecef(lon: f64, lat: f64, h: f64) -> PyResult<(f64, f64, f64)> {
    let v = crate::geo::projections::geocentric::wgs84_geodetic_to_ecef(lon, lat, h)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((v.x, v.y, v.z))
}

/// Geocentric ECEF metres → WGS84 geodetic (lon, lat degrees, ellipsoidal
/// height metres), full f64.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (x, y, z))]
pub(crate) fn ecef_to_wgs84(x: f64, y: f64, z: f64) -> PyResult<(f64, f64, f64)> {
    crate::geo::projections::geocentric::wgs84_ecef_to_geodetic(glam::DVec3::new(x, y, z))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert a DEM of orthometric (EGM96) heights to ellipsoidal heights by
/// adding N(lat, lon) per pixel. `bounds` is (left, bottom, right, top) in
/// EPSG:4326 degrees; pixel centres are sampled. Returns float64.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (dem, bounds))]
pub(crate) fn dem_orthometric_to_ellipsoidal<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    bounds: (f64, f64, f64, f64),
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (left, bottom, right, top) = bounds;
    if !left.is_finite()
        || !right.is_finite()
        || !(right > left && top > bottom)
        || !(-90.0..=90.0).contains(&bottom)
        || !(-90.0..=90.0).contains(&top)
    {
        return Err(PyValueError::new_err(format!(
            "invalid_bounds: expected (left, bottom, right, top) EPSG:4326 degrees, got {bounds:?}"
        )));
    }
    let arr = dem.as_array();
    let (rows, cols) = (arr.nrows(), arr.ncols());
    let mut out = ndarray::Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        let lat = top - (r as f64 + 0.5) * (top - bottom) / rows as f64;
        for c in 0..cols {
            let lon = left + (c as f64 + 0.5) * (right - left) / cols as f64;
            out[(r, c)] = arr[(r, c)] + crate::geo::geoid::undulation_deg(lat, lon);
        }
    }
    Ok(PyArray2::from_owned_array_bound(py, out))
}
