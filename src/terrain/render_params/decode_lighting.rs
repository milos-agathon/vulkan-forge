use super::parse::*;
use super::*;

pub(super) fn parse_light_settings(light: &Bound<'_, PyAny>) -> PyResult<LightSettingsNative> {
    let light_type: String = light.getattr("light_type")?.extract()?;
    let azimuth = to_finite_f32(
        light.getattr("azimuth_deg")?.as_gil_ref(),
        "light.azimuth_deg",
    )?;
    let elevation = to_finite_f32(
        light.getattr("elevation_deg")?.as_gil_ref(),
        "light.elevation_deg",
    )?;
    let intensity =
        to_finite_f32(light.getattr("intensity")?.as_gil_ref(), "light.intensity")?.max(0.0);
    let color: Vec<f32> = light
        .getattr("color")?
        .extract()
        .map_err(|_| PyValueError::new_err("light.color must be a sequence of three floats"))?;
    if color.len() != 3 {
        return Err(PyValueError::new_err(
            "light.color must contain exactly three components",
        ));
    }

    let azimuth_rad = azimuth.to_radians();
    let elevation_rad = elevation.to_radians();
    let cos_el = elevation_rad.cos();
    let sin_el = elevation_rad.sin();
    let direction = match light_type.as_str() {
        "Directional" | "directional" => normalize_direction(
            cos_el * azimuth_rad.cos(),
            cos_el * azimuth_rad.sin(),
            sin_el,
        ),
        _ => normalize_direction(
            cos_el * azimuth_rad.cos(),
            cos_el * azimuth_rad.sin(),
            sin_el,
        ),
    };

    Ok(LightSettingsNative {
        direction,
        intensity,
        color: [color[0], color[1], color[2]],
    })
}

pub(super) fn parse_triplanar_settings(
    triplanar: &Bound<'_, PyAny>,
) -> PyResult<TriplanarSettingsNative> {
    Ok(TriplanarSettingsNative {
        scale: to_finite_f32(triplanar.getattr("scale")?.as_gil_ref(), "triplanar.scale")?,
        blend_sharpness: to_finite_f32(
            triplanar.getattr("blend_sharpness")?.as_gil_ref(),
            "triplanar.blend_sharpness",
        )?,
        normal_strength: to_finite_f32(
            triplanar.getattr("normal_strength")?.as_gil_ref(),
            "triplanar.normal_strength",
        )?,
    })
}

pub(super) fn parse_pom_settings(pom: &Bound<'_, PyAny>) -> PyResult<PomSettingsNative> {
    Ok(PomSettingsNative {
        enabled: pom.getattr("enabled")?.extract()?,
        scale: to_finite_f32(pom.getattr("scale")?.as_gil_ref(), "pom.scale")?,
        min_steps: pom.getattr("min_steps")?.extract::<i64>()? as u32,
        max_steps: pom.getattr("max_steps")?.extract::<i64>()? as u32,
        refine_steps: pom.getattr("refine_steps")?.extract::<i64>()? as u32,
        shadow: pom.getattr("shadow")?.extract()?,
        occlusion: pom.getattr("occlusion")?.extract()?,
    })
}

pub(super) fn parse_lod_settings(lod: &Bound<'_, PyAny>) -> PyResult<LodSettingsNative> {
    Ok(LodSettingsNative {
        level: lod.getattr("level")?.extract::<i64>()? as i32,
        bias: to_finite_f32(lod.getattr("bias")?.as_gil_ref(), "lod.bias")?,
        lod0_bias: to_finite_f32(lod.getattr("lod0_bias")?.as_gil_ref(), "lod.lod0_bias")?,
    })
}

pub(super) fn parse_clamp_settings(clamp: &Bound<'_, PyAny>) -> PyResult<ClampSettingsNative> {
    Ok(ClampSettingsNative {
        height_range: tuple_to_f32_pair(
            clamp.getattr("height_range")?.as_gil_ref(),
            "clamp.height_range",
        )?,
        slope_range: tuple_to_f32_pair(
            clamp.getattr("slope_range")?.as_gil_ref(),
            "clamp.slope_range",
        )?,
        ambient_range: tuple_to_f32_pair(
            clamp.getattr("ambient_range")?.as_gil_ref(),
            "clamp.ambient_range",
        )?,
        shadow_range: tuple_to_f32_pair(
            clamp.getattr("shadow_range")?.as_gil_ref(),
            "clamp.shadow_range",
        )?,
        occlusion_range: tuple_to_f32_pair(
            clamp.getattr("occlusion_range")?.as_gil_ref(),
            "clamp.occlusion_range",
        )?,
    })
}

pub(super) fn parse_sampling_settings(
    sampling: &Bound<'_, PyAny>,
) -> PyResult<SamplingSettingsNative> {
    Ok(SamplingSettingsNative {
        mag_filter: parse_filter_mode(
            &sampling.getattr("mag_filter")?.extract::<String>()?,
            "sampling.mag_filter",
        )?,
        min_filter: parse_filter_mode(
            &sampling.getattr("min_filter")?.extract::<String>()?,
            "sampling.min_filter",
        )?,
        mip_filter: parse_filter_mode(
            &sampling.getattr("mip_filter")?.extract::<String>()?,
            "sampling.mip_filter",
        )?,
        anisotropy: sampling
            .getattr("anisotropy")?
            .extract::<i64>()?
            .clamp(1, 16) as u32,
        address_u: parse_address_mode(
            &sampling.getattr("address_u")?.extract::<String>()?,
            "sampling.address_u",
        )?,
        address_v: parse_address_mode(
            &sampling.getattr("address_v")?.extract::<String>()?,
            "sampling.address_v",
        )?,
        address_w: parse_address_mode(
            &sampling.getattr("address_w")?.extract::<String>()?,
            "sampling.address_w",
        )?,
    })
}

fn validate_pcss_controls(
    pcss_light_radius: f32,
    pcss_blocker_radius: f32,
    pcss_filter_radius: f32,
    light_size: f32,
) -> PyResult<()> {
    for (name, value, allow_zero) in [
        ("pcss_light_radius", pcss_light_radius, true),
        ("pcss_blocker_radius", pcss_blocker_radius, true),
        ("pcss_filter_radius", pcss_filter_radius, true),
        ("light_size", light_size, false),
    ] {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!("{name} must be finite")));
        }
        if value < 0.0 || (!allow_zero && value == 0.0) {
            let operator = if allow_zero { ">=" } else { ">" };
            return Err(PyValueError::new_err(format!(
                "{name} must be {operator} 0"
            )));
        }
    }
    Ok(())
}

fn validate_shadow_dimensions(resolution: i64, cascades: i64) -> PyResult<(u32, u32)> {
    let resolution = u32::try_from(resolution)
        .map_err(|_| PyValueError::new_err("resolution must fit in u32"))?;
    let cascades =
        u32::try_from(cascades).map_err(|_| PyValueError::new_err("cascades must fit in u32"))?;
    crate::shadows::validate_shadow_dimensions(resolution, cascades)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((resolution, cascades))
}

pub(super) fn parse_shadow_settings(shadows: &Bound<'_, PyAny>) -> PyResult<ShadowSettingsNative> {
    let softness = shadows.getattr("softness")?.extract().unwrap_or(0.01);
    let pcss_light_radius = shadows
        .getattr("pcss_light_radius")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or(0.0);
    let pcss_blocker_radius = shadows
        .getattr("pcss_blocker_radius")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or(crate::shadows::DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS);
    let pcss_filter_radius = shadows
        .getattr("pcss_filter_radius")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or(crate::shadows::DEFAULT_PCSS_FILTER_RADIUS_TEXELS);
    let light_size = shadows
        .getattr("light_size")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or(crate::shadows::DEFAULT_PCSS_LIGHT_SIZE);
    validate_pcss_controls(
        pcss_light_radius,
        pcss_blocker_radius,
        pcss_filter_radius,
        light_size,
    )?;
    let (resolution, cascades) = validate_shadow_dimensions(
        shadows.getattr("resolution")?.extract::<i64>()?,
        shadows.getattr("cascades")?.extract::<i64>()?,
    )?;
    Ok(ShadowSettingsNative {
        enabled: shadows.getattr("enabled")?.extract().unwrap_or(true),
        technique: shadows
            .getattr("technique")?
            .extract::<String>()
            .unwrap_or_else(|_| "PCSS".to_string()),
        resolution,
        cascades,
        max_distance: shadows.getattr("max_distance")?.extract().unwrap_or(3000.0),
        softness,
        pcss_light_radius,
        pcss_blocker_radius,
        pcss_filter_radius,
        light_size,
        intensity: shadows.getattr("intensity")?.extract().unwrap_or(1.0),
        slope_scale_bias: shadows
            .getattr("slope_scale_bias")?
            .extract()
            .unwrap_or(0.001),
        depth_bias: shadows.getattr("depth_bias")?.extract().unwrap_or(0.0005),
        normal_bias: shadows.getattr("normal_bias")?.extract().unwrap_or(0.0002),
    })
}

#[cfg(test)]
mod tests {
    use super::{validate_pcss_controls, validate_shadow_dimensions};

    #[test]
    fn native_pcss_boundary_rejects_non_finite_controls() {
        for values in [
            [f32::NAN, 6.0, 4.0, 1.0],
            [0.0, f32::INFINITY, 4.0, 1.0],
            [0.0, 6.0, f32::NEG_INFINITY, 1.0],
            [0.0, 6.0, 4.0, f32::NAN],
        ] {
            assert!(validate_pcss_controls(values[0], values[1], values[2], values[3]).is_err());
        }
    }

    #[test]
    fn native_shadow_decode_rejects_signed_and_unbounded_dimensions() {
        for resolution in [-1, 0, 511, 513, 16_384, i64::MAX] {
            assert!(validate_shadow_dimensions(resolution, 1).is_err());
        }
        for cascades in [-1, 0, 5, i64::MAX] {
            assert!(validate_shadow_dimensions(512, cascades).is_err());
        }
        assert_eq!(
            validate_shadow_dimensions(512, 1).expect("valid dimensions"),
            (512, 1)
        );
    }
}
