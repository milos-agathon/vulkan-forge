use super::parse::*;
use super::*;

pub(super) struct CoreTerrainParams {
    pub size_px: (u32, u32),
    pub render_scale: f32,
    pub terrain_span: f32,
    pub msaa_samples: u32,
    pub z_scale: f32,
    pub cam_target: [f32; 3],
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_gamma_deg: f32,
    pub fov_y_deg: f32,
    pub clip: (f32, f32),
    pub exposure: f32,
    pub gamma: f32,
    pub albedo_mode: String,
    pub colormap_strength: f32,
    pub hue_variation_strength: f32,
    pub ao_weight: f32,
    pub height_curve_mode: String,
    pub height_curve_strength: f32,
    pub height_curve_power: f32,
    pub lambert_contrast: f32,
    pub colormap_srgb: bool,
    pub output_srgb_eotf: bool,
    pub camera_mode: String,
    pub culling: String,
    pub shading: String,
    pub vt_store_path: Option<String>,
    pub prefetch_horizon_ms: f32,
    pub vt_upload_budget_bytes: u64,
    pub debug_mode: u32,
    pub aa_samples: u32,
    pub aa_seed: Option<u64>,
    pub terrain_data_revision: Option<u64>,
    pub height_curve_lut: Option<Arc<Vec<f32>>>,
}

pub(super) fn parse_core_params(params: &Bound<'_, PyAny>) -> PyResult<CoreTerrainParams> {
    let size_px = tuple_to_u32_pair(params.getattr("size_px")?.as_gil_ref(), "size_px")?;
    let render_scale = to_finite_f32(params.getattr("render_scale")?.as_gil_ref(), "render_scale")?;
    let terrain_span = to_finite_f32(params.getattr("terrain_span")?.as_gil_ref(), "terrain_span")?;
    let msaa_samples: u32 = params
        .getattr("msaa_samples")?
        .extract::<u32>()
        .map_err(|_| PyValueError::new_err("msaa_samples must be an integer >= 1"))?;
    let z_scale = to_finite_f32(params.getattr("z_scale")?.as_gil_ref(), "z_scale")?;
    if !matches!(msaa_samples, 1 | 2 | 4 | 8) {
        return Err(PyValueError::new_err(
            "msaa_samples must be one of 1, 2, 4, or 8",
        ));
    }
    if render_scale <= 0.0 {
        return Err(PyValueError::new_err("render_scale must be positive"));
    }
    if terrain_span <= 0.0 {
        return Err(PyValueError::new_err("terrain_span must be positive"));
    }
    if z_scale <= 0.0 {
        return Err(PyValueError::new_err("z_scale must be positive"));
    }

    let cam_target = list_to_vec3(params.getattr("cam_target")?.as_gil_ref(), "cam_target")?;
    let cam_radius = to_finite_f32(params.getattr("cam_radius")?.as_gil_ref(), "cam_radius")?;
    let cam_phi_deg = to_finite_f32(params.getattr("cam_phi_deg")?.as_gil_ref(), "cam_phi_deg")?;
    let cam_theta_deg = to_finite_f32(
        params.getattr("cam_theta_deg")?.as_gil_ref(),
        "cam_theta_deg",
    )?;
    if cam_radius <= 0.0 {
        return Err(PyValueError::new_err("cam_radius must be positive"));
    }
    let cam_gamma_deg = params
        .getattr("cam_gamma_deg")?
        .extract::<f32>()
        .map_err(|_| PyValueError::new_err("cam_gamma_deg must be a float value"))?;
    let fov_y_deg = to_finite_f32(params.getattr("fov_y_deg")?.as_gil_ref(), "fov_y_deg")?;
    if !(0.0..=180.0).contains(&fov_y_deg) {
        return Err(PyValueError::new_err("fov_y_deg must be within [0, 180]"));
    }
    let clip = tuple_to_f32_pair(params.getattr("clip")?.as_gil_ref(), "clip")?;
    if clip.0 <= 0.0 || clip.0 >= clip.1 {
        return Err(PyValueError::new_err(
            "clip tuple must satisfy near > 0 and near < far",
        ));
    }

    let exposure = to_finite_f32(params.getattr("exposure")?.as_gil_ref(), "exposure")?;
    let gamma = to_finite_f32(params.getattr("gamma")?.as_gil_ref(), "gamma")?;
    if gamma <= 0.0 {
        return Err(PyValueError::new_err("gamma must be positive"));
    }
    let albedo_mode: String = params
        .getattr("albedo_mode")?
        .extract()
        .map_err(|_| PyValueError::new_err("albedo_mode must be a string"))?;
    let colormap_strength = to_finite_f32(
        params.getattr("colormap_strength")?.as_gil_ref(),
        "colormap_strength",
    )?;
    match albedo_mode.as_str() {
        "colormap" | "mix" | "material" => {}
        other => {
            return Err(PyValueError::new_err(format!(
                "albedo_mode '{}' is not supported",
                other
            )))
        }
    }
    if !(0.0..=1.0).contains(&colormap_strength) {
        return Err(PyValueError::new_err(
            "colormap_strength must be between 0 and 1",
        ));
    }
    let hue_variation_strength = match params.getattr("hue_variation_strength") {
        Ok(value) => to_finite_f32(value.as_gil_ref(), "hue_variation_strength")?,
        Err(_) => 0.08,
    }
    .clamp(0.0, 0.2);

    let ao_weight = params
        .getattr("ao_weight")
        .ok()
        .and_then(|v| v.extract::<f32>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);

    let height_curve_mode: String = params
        .getattr("height_curve_mode")?
        .extract()
        .map_err(|_| PyValueError::new_err("height_curve_mode must be a string"))?;
    let valid_modes = ["linear", "pow", "smoothstep", "lut"];
    if !valid_modes.contains(&height_curve_mode.as_str()) {
        return Err(PyValueError::new_err(format!(
            "height_curve_mode must be one of {:?}, got {}",
            valid_modes, height_curve_mode
        )));
    }
    let height_curve_strength = to_finite_f32(
        params.getattr("height_curve_strength")?.as_gil_ref(),
        "height_curve_strength",
    )?
    .clamp(0.0, 1.0);
    let height_curve_power = to_finite_f32(
        params.getattr("height_curve_power")?.as_gil_ref(),
        "height_curve_power",
    )?;
    if !matches!(
        height_curve_power.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return Err(PyValueError::new_err(
            "height_curve_power must be greater than zero",
        ));
    }

    let lambert_contrast = params
        .getattr("lambert_contrast")
        .ok()
        .and_then(|v| v.extract::<f32>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let colormap_srgb = params
        .getattr("colormap_srgb")
        .ok()
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false);
    let output_srgb_eotf = params
        .getattr("output_srgb_eotf")
        .ok()
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false);

    let camera_mode = params
        .getattr("camera_mode")
        .ok()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_else(|| "screen".to_string());
    let culling = params
        .getattr("culling")
        .ok()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_else(|| "frustum".to_string());
    if !matches!(culling.as_str(), "none" | "frustum" | "hzb_two_phase") {
        return Err(PyValueError::new_err(
            "culling must be one of 'none', 'frustum', or 'hzb_two_phase'",
        ));
    }
    let shading = params
        .getattr("shading")
        .ok()
        .and_then(|value| value.extract::<String>().ok())
        .unwrap_or_else(|| "forward".to_string());
    if !matches!(shading.as_str(), "forward" | "visibility") {
        return Err(PyValueError::new_err(
            "shading must be one of 'forward' or 'visibility'",
        ));
    }
    let vt_store_path = match params.getattr("vt_store").ok() {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => value
            .extract::<String>()
            .ok()
            .or_else(|| value.getattr("path").ok()?.extract::<String>().ok()),
    };
    let prefetch_horizon_ms = params
        .getattr("prefetch_horizon_ms")
        .ok()
        .and_then(|value| value.extract::<f32>().ok())
        .unwrap_or(100.0);
    if !prefetch_horizon_ms.is_finite() || prefetch_horizon_ms < 0.0 {
        return Err(PyValueError::new_err(
            "prefetch_horizon_ms must be finite and >= 0",
        ));
    }
    let vt_upload_budget_bytes = params
        .getattr("vt_upload_budget_bytes")
        .ok()
        .and_then(|value| value.extract::<u64>().ok())
        .unwrap_or(16 * 1024 * 1024);
    if vt_upload_budget_bytes == 0 {
        return Err(PyValueError::new_err(
            "vt_upload_budget_bytes must be greater than zero",
        ));
    }
    let debug_mode = params
        .getattr("debug_mode")
        .ok()
        .and_then(|v| v.extract::<u32>().ok())
        .unwrap_or(0);
    let aa_samples = params
        .getattr("aa_samples")
        .ok()
        .and_then(|v| v.extract::<u32>().ok())
        .unwrap_or(1)
        .max(1);
    let aa_seed = params.getattr("aa_seed").ok().and_then(|v| {
        if v.is_none() {
            None
        } else {
            v.extract::<u64>().ok()
        }
    });
    let terrain_data_revision = match params.getattr("terrain_data_revision").ok() {
        Some(value) if value.is_none() => None,
        Some(value) => Some(value.extract::<u64>().map_err(|_| {
            PyValueError::new_err("terrain_data_revision must be a non-negative integer")
        })?),
        None => None,
    };

    let height_curve_lut = if height_curve_mode == "lut" {
        let raw_lut = params.getattr("height_curve_lut")?;
        let lut_vec: Vec<f32> = raw_lut.extract().map_err(|_| {
            PyValueError::new_err("height_curve_lut must be convertible to a 1D float array")
        })?;
        if lut_vec.len() != 256 {
            return Err(PyValueError::new_err(
                "height_curve_lut must have length 256 when height_curve_mode='lut'",
            ));
        }
        if lut_vec
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
        {
            return Err(PyValueError::new_err(
                "height_curve_lut values must be finite floats within [0, 1]",
            ));
        }
        Some(Arc::new(lut_vec))
    } else {
        None
    };

    Ok(CoreTerrainParams {
        size_px,
        render_scale,
        terrain_span,
        msaa_samples,
        z_scale,
        cam_target,
        cam_radius,
        cam_phi_deg,
        cam_theta_deg,
        cam_gamma_deg,
        fov_y_deg,
        clip,
        exposure,
        gamma,
        albedo_mode,
        colormap_strength,
        hue_variation_strength,
        ao_weight,
        height_curve_mode,
        height_curve_strength,
        height_curve_power,
        lambert_contrast,
        colormap_srgb,
        output_srgb_eotf,
        camera_mode,
        culling,
        shading,
        vt_store_path,
        prefetch_horizon_ms,
        vt_upload_budget_bytes,
        debug_mode,
        aa_samples,
        aa_seed,
        terrain_data_revision,
        height_curve_lut,
    })
}
