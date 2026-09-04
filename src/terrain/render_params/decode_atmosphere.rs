use super::*;

pub(super) fn parse_volumetrics_settings(params: &Bound<'_, PyAny>) -> VolumetricsSettingsNative {
    if let Ok(vol) = params.getattr("volumetrics") {
        if !vol.is_none() {
            let mode_str: String = vol
                .getattr("mode")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "uniform".to_string());
            VolumetricsSettingsNative {
                enabled: vol
                    .getattr("enabled")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                mode: match mode_str.as_str() {
                    "height" => VolumetricsModeNative::Height,
                    "exponential" => VolumetricsModeNative::Exponential,
                    _ => VolumetricsModeNative::Uniform,
                },
                density: vol
                    .getattr("density")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.01),
                height_falloff: vol
                    .getattr("height_falloff")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
                base_height: vol
                    .getattr("base_height")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                scattering: vol
                    .getattr("scattering")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.5),
                absorption: vol
                    .getattr("absorption")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.1),
                phase_g: vol
                    .getattr("phase_g")
                    .and_then(|v| v.extract())
                    .unwrap_or(0.0),
                light_shafts: vol
                    .getattr("light_shafts")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
                shaft_intensity: vol
                    .getattr("shaft_intensity")
                    .and_then(|v| v.extract())
                    .unwrap_or(1.0),
                shaft_samples: vol
                    .getattr("shaft_samples")
                    .and_then(|v| v.extract())
                    .unwrap_or(32),
                use_shadows: vol
                    .getattr("use_shadows")
                    .and_then(|v| v.extract())
                    .unwrap_or(true),
                half_res: vol
                    .getattr("half_res")
                    .and_then(|v| v.extract())
                    .unwrap_or(false),
            }
        } else {
            VolumetricsSettingsNative::default()
        }
    } else {
        VolumetricsSettingsNative::default()
    }
}

fn optional_attr<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match obj.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(obj.py()) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

pub(super) fn parse_sky_settings(params: &Bound<'_, PyAny>) -> PyResult<SkySettingsNative> {
    let Some(sky) = optional_attr(params, "sky")? else {
        return Ok(SkySettingsNative::default());
    };
    if sky.is_none() {
        return Ok(SkySettingsNative::default());
    }

    let model_name = optional_attr(&sky, "model")?
        .map(|value| value.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "hosek-wilkie".to_string());
    let model = match model_name.as_str() {
        "preetham" => 0,
        "hosek-wilkie" | "hosek_wilkie" | "hosekwilkie" => 1,
        "approximate" | "legacy" => 2,
        "aether" => 3,
        _ => {
            return Err(PyValueError::new_err(format!(
            "sky.model must be preetham, hosek-wilkie, approximate, or aether; got {model_name:?}"
        )))
        }
    };
    let provided_lut_handle = match optional_attr(&sky, "lut_handle")? {
        Some(value) if !value.is_none() => {
            let handle = value
                .extract::<PyRef<'_, crate::py_types::PyAtmosphereLutHandle>>()
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "sky.lut_handle must be an AtmosphereLutHandle returned by atmosphere_bake_luts()",
                    )
                })?;
            Some(handle.core_handle().clone())
        }
        _ => None,
    };
    if provided_lut_handle.is_some() && model != 3 {
        return Err(PyValueError::new_err(
            "sky.lut_handle requires sky.model='aether'",
        ));
    }

    let turbidity_supplied = optional_attr(&sky, "turbidity")?
        .as_ref()
        .map(|value| value.extract::<f32>())
        .transpose()?;
    let ground_albedo_supplied = optional_attr(&sky, "ground_albedo")?
        .as_ref()
        .map(|value| value.extract::<f32>())
        .transpose()?;
    let ozone_du_supplied = optional_attr(&sky, "ozone_du")?
        .as_ref()
        .map(|value| value.extract::<f32>())
        .transpose()?;
    let mie_g_supplied = optional_attr(&sky, "mie_g")?
        .as_ref()
        .map(|value| value.extract::<f32>())
        .transpose()?;
    if let Some(handle) = provided_lut_handle.as_ref() {
        let config = handle.config();
        let mismatched = [
            ("turbidity", turbidity_supplied, config.turbidity),
            (
                "ground_albedo",
                ground_albedo_supplied,
                config.ground_albedo,
            ),
            ("ozone_du", ozone_du_supplied, config.ozone_du),
            ("mie_g", mie_g_supplied, config.mie_g),
        ]
        .into_iter()
        .find_map(|(name, supplied, expected)| {
            supplied
                .filter(|actual| actual.to_bits() != expected.to_bits())
                .map(|actual| (name, actual, expected))
        });
        if let Some((name, actual, expected)) = mismatched {
            return Err(PyValueError::new_err(format!(
                "sky.{name}={actual} does not match the exact LUT handle value {expected}; refusing to substitute or relabel transport"
            )));
        }
    }
    let handle_config = provided_lut_handle.as_ref().map(|handle| handle.config());
    let turbidity =
        turbidity_supplied.unwrap_or_else(|| handle_config.map_or(2.0, |config| config.turbidity));
    let ground_albedo = ground_albedo_supplied
        .unwrap_or_else(|| handle_config.map_or(0.3, |config| config.ground_albedo));
    let ozone_du =
        ozone_du_supplied.unwrap_or_else(|| handle_config.map_or(300.0, |config| config.ozone_du));
    let mie_g = mie_g_supplied.unwrap_or_else(|| handle_config.map_or(0.8, |config| config.mie_g));

    let sun_intensity = optional_attr(&sky, "sun_intensity")?
        .map(|value| value.extract::<f32>())
        .transpose()?
        .unwrap_or(1.0);
    let sun_size = optional_attr(&sky, "sun_size")?
        .map(|value| value.extract::<f32>())
        .transpose()?
        .unwrap_or(1.0);
    let aerial_density = optional_attr(&sky, "aerial_density")?
        .map(|value| value.extract::<f32>())
        .transpose()?
        .unwrap_or(1.0);
    let sky_exposure = optional_attr(&sky, "sky_exposure")?
        .map(|value| value.extract::<f32>())
        .transpose()?
        .unwrap_or(1.0);
    if !turbidity.is_finite() || !(1.0..=10.0).contains(&turbidity) {
        return Err(PyValueError::new_err(
            "sky.turbidity must be finite and in [1, 10]",
        ));
    }
    if !ground_albedo.is_finite() || !(0.0..=1.0).contains(&ground_albedo) {
        return Err(PyValueError::new_err(
            "sky.ground_albedo must be finite and in [0, 1]",
        ));
    }
    if !ozone_du.is_finite() || !(0.0..=600.0).contains(&ozone_du) {
        return Err(PyValueError::new_err(
            "sky.ozone_du must be finite and in [0, 600] DU",
        ));
    }
    if !mie_g.is_finite() || !(0.0..=0.99).contains(&mie_g) {
        return Err(PyValueError::new_err(
            "sky.mie_g must be finite and in [0.0, 0.99]",
        ));
    }
    if !sun_intensity.is_finite() || sun_intensity < 0.0 {
        return Err(PyValueError::new_err(
            "sky.sun_intensity must be finite and >= 0",
        ));
    }
    if !sun_size.is_finite() || sun_size < 0.0 {
        return Err(PyValueError::new_err(
            "sky.sun_size must be finite and >= 0",
        ));
    }
    if !aerial_density.is_finite() || !(0.0..=10.0).contains(&aerial_density) {
        return Err(PyValueError::new_err(
            "sky.aerial_density must be finite and in [0.0, 10.0]",
        ));
    }
    if !sky_exposure.is_finite() || sky_exposure < 0.0 {
        return Err(PyValueError::new_err(
            "sky.sky_exposure must be finite and >= 0",
        ));
    }
    let lut_handle = if model == 3 {
        match provided_lut_handle {
            Some(handle) => Some(handle),
            None => {
                let config = crate::core::atmosphere::AtmosphereConfig {
                    turbidity,
                    ozone_du,
                    mie_g,
                    ground_albedo,
                    ..Default::default()
                };
                Some(
                    crate::core::atmosphere::AtmosphereLutHandle::load_shipped(config).map_err(
                        |error| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "AETHER TerrainRenderer could not resolve the shipped LUT bank: {error}. Custom physical inputs require lut_handle=atmosphere_bake_luts(...) from an atmosphere-bake build; no nearby or legacy LUT was substituted."
                            ))
                        },
                    )?,
                )
            }
        }
    } else {
        None
    };
    Ok(SkySettingsNative {
        enabled: optional_attr(&sky, "enabled")?
            .map(|value| value.extract::<bool>())
            .transpose()?
            .unwrap_or(false),
        model,
        turbidity,
        ground_albedo,
        ozone_du,
        mie_g,
        sun_intensity,
        sun_size,
        aerial_perspective: optional_attr(&sky, "aerial_perspective")?
            .map(|value| value.extract::<bool>())
            .transpose()?
            .unwrap_or(true),
        aerial_density,
        sky_exposure,
        lut_handle,
    })
}
