//! Python boundary for the AETHER spectral atmosphere.

use ndarray::Array3;
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::atmosphere::{
    reference_aerial_radiance, spectral_to_linear_rgb, AtmosphereConfig, AtmosphereError,
    AtmosphereLutHandle, AtmosphereLuts, NUM_WAVELENGTHS,
};
use crate::py_types::PyAtmosphereLutHandle;

fn py_error(error: AtmosphereError) -> PyErr {
    match error {
        AtmosphereError::InvalidConfig(_) => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

fn config(
    turbidity: f32,
    ozone_du: f32,
    mie_g: f32,
    ground_albedo: f32,
    scattering_orders: u32,
) -> Result<AtmosphereConfig, PyErr> {
    let config = AtmosphereConfig {
        turbidity,
        ozone_du,
        mie_g,
        ground_albedo,
        scattering_orders,
        ..Default::default()
    };
    config.validate().map_err(py_error)?;
    Ok(config)
}

fn resolve_luts(config: AtmosphereConfig) -> PyResult<AtmosphereLuts> {
    match crate::core::atmosphere::load_precomputed_atmosphere_luts(config.clone()) {
        Ok(luts) => Ok(luts),
        Err(AtmosphereError::UnsupportedPrecomputedConfig(_reason)) => {
            #[cfg(feature = "atmosphere-bake")]
            {
                crate::core::atmosphere::bake_atmosphere_luts(config).map_err(py_error)
            }
            #[cfg(not(feature = "atmosphere-bake"))]
            {
                let error = AtmosphereError::UnsupportedPrecomputedConfig(_reason);
                Err(PyRuntimeError::new_err(format!(
                    "{error}. Custom AETHER LUT inputs require a forge3d build with the \
                 Cargo feature 'atmosphere-bake'; no nearest shipped table or legacy \
                 RGB atmosphere was substituted."
                )))
            }
        }
        Err(other) => Err(py_error(other)),
    }
}

/// Resolve a shipped AETHER LUT or run the feature-gated offline bake.
#[pyfunction]
#[pyo3(signature = (turbidity = 2.0, ozone_du = 300.0, mie_g = 0.8, ground_albedo = 0.3, scattering_orders = 4))]
pub(crate) fn atmosphere_bake_luts(
    py: Python<'_>,
    turbidity: f32,
    ozone_du: f32,
    mie_g: f32,
    ground_albedo: f32,
    scattering_orders: u32,
) -> PyResult<Py<PyAtmosphereLutHandle>> {
    let luts = resolve_luts(config(
        turbidity,
        ozone_du,
        mie_g,
        ground_albedo,
        scattering_orders,
    )?)?;
    let handle = AtmosphereLutHandle::from_luts(luts).map_err(py_error)?;
    Py::new(py, PyAtmosphereLutHandle::new(handle))
}

/// Convert the complete AETHER wavelength basis to linear sRGB.
#[pyfunction]
pub(crate) fn atmosphere_spectral_to_linear_rgb(samples: Vec<f32>) -> PyResult<(f32, f32, f32)> {
    if samples.len() != NUM_WAVELENGTHS {
        return Err(PyValueError::new_err(format!(
            "AETHER requires exactly {NUM_WAVELENGTHS} wavelength samples, got {}",
            samples.len()
        )));
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "AETHER wavelength samples must all be finite",
        ));
    }
    let rgb = spectral_to_linear_rgb(&samples);
    Ok((rgb[0], rgb[1], rgb[2]))
}

fn sun_direction(elevation_deg: f32) -> PyResult<[f32; 3]> {
    if !elevation_deg.is_finite() || !(-90.0..=90.0).contains(&elevation_deg) {
        return Err(PyValueError::new_err(
            "sun_elevation_deg must be finite and in [-90, 90]",
        ));
    }
    let elevation = elevation_deg.to_radians();
    Ok([0.0, elevation.sin(), elevation.cos()])
}

/// Generate a small equirectangular AETHER validation environment.
///
/// Outside CENSOR's render-certificate scope: this is a CPU diagnostic and
/// golden-fixture generator, not an input to the hard PROMETHEUS closure and
/// not a product render entry point. It neither submits GPU work nor claims
/// render provenance.
#[pyfunction]
#[pyo3(signature = (width, height, sun_elevation_deg, turbidity = 2.0, ozone_du = 300.0, mie_g = 0.8, ground_albedo = 0.3, mode = "lut"))]
pub(crate) fn atmosphere_generate_environment(
    py: Python<'_>,
    width: u32,
    height: u32,
    sun_elevation_deg: f32,
    turbidity: f32,
    ozone_du: f32,
    mie_g: f32,
    ground_albedo: f32,
    mode: &str,
) -> PyResult<PyObject> {
    if width < 2 || height < 2 || width > 2048 || height > 1024 {
        return Err(PyValueError::new_err(
            "AETHER environment dimensions must be 2..2048 by 2..1024",
        ));
    }
    let config = config(turbidity, ozone_du, mie_g, ground_albedo, 4)?;
    let sun = sun_direction(sun_elevation_deg)?;
    let rgb_linear = match mode {
        "reference" => {
            crate::core::atmosphere::generate_reference_equirectangular(
                &config, width, height, 0.0, sun,
            )
            .map_err(py_error)?
            .rgb_linear
        }
        "lut" => {
            let luts = crate::core::atmosphere::load_precomputed_atmosphere_luts(config)
                .map_err(py_error)?;
            let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
            for y in 0..height {
                let latitude = std::f32::consts::FRAC_PI_2
                    - std::f32::consts::PI * (y as f32 + 0.5) / height as f32;
                let cos_latitude = latitude.cos();
                for x in 0..width {
                    let longitude = 2.0 * std::f32::consts::PI * (x as f32 + 0.5) / width as f32
                        - std::f32::consts::PI;
                    let view = [
                        cos_latitude * longitude.sin(),
                        latitude.sin(),
                        cos_latitude * longitude.cos(),
                    ];
                    rgb.extend_from_slice(&luts.sky_radiance(0.0, view, sun).map_err(py_error)?);
                }
            }
            rgb
        }
        _ => return Err(PyValueError::new_err("mode must be 'lut' or 'reference'")),
    };
    let array = Array3::from_shape_vec((height as usize, width as usize, 3), rgb_linear)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let report = PyDict::new_bound(py);
    report.set_item("width", width)?;
    report.set_item("height", height)?;
    report.set_item("mode", mode)?;
    report.set_item("linear_hdr", true)?;
    report.set_item("rgb_linear", array.into_pyarray_bound(py))?;
    Ok(report.into())
}

/// Evaluate the CPU spectral aerial-transport diagnostic at one ray state.
#[pyfunction]
#[pyo3(signature = (surface_rgb, observer_altitude_m, distance_m, view_dir, sun_dir, turbidity = 2.0, ozone_du = 300.0, mie_g = 0.8, ground_albedo = 0.3))]
pub(crate) fn atmosphere_reference_aerial(
    surface_rgb: (f32, f32, f32),
    observer_altitude_m: f32,
    distance_m: f32,
    view_dir: (f32, f32, f32),
    sun_dir: (f32, f32, f32),
    turbidity: f32,
    ozone_du: f32,
    mie_g: f32,
    ground_albedo: f32,
) -> PyResult<(f32, f32, f32)> {
    let config = config(turbidity, ozone_du, mie_g, ground_albedo, 4)?;
    let rgb = reference_aerial_radiance(
        &config,
        [surface_rgb.0, surface_rgb.1, surface_rgb.2],
        observer_altitude_m,
        distance_m,
        [view_dir.0, view_dir.1, view_dir.2],
        [sun_dir.0, sun_dir.1, sun_dir.2],
    )
    .map_err(py_error)?;
    Ok((rgb[0], rgb[1], rgb[2]))
}
