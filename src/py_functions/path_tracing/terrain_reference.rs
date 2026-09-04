// src/py_functions/path_tracing/terrain_reference.rs
// PROMETHEUS Python seam: GPU-backed converged terrain path-traced reference
// rooted in HybridPathTracer::render_terrain_reference. This is the honest
// GPU entry point — the legacy `hybrid_render` stays a CPU/SDF compatibility
// wrapper and never claimed to reach the GPU path.
// RELEVANT FILES: src/path_tracing/hybrid_compute/render_terrain.rs,
//                 python/forge3d/path_tracing.py

use super::super::super::*;

#[cfg(feature = "extension-module")]
fn extract_sun_color(obj: &Bound<'_, PyAny>) -> PyResult<[f32; 3]> {
    let reject =
        || PyValueError::new_err("sun_color must be exactly three finite, non-negative numbers");
    if obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
        || obj.is_instance_of::<PyMemoryView>()
    {
        return Err(reject());
    }

    let mut out = [0.0f32; 3];
    let mut count = 0usize;
    for item in obj.iter().map_err(|_| reject())? {
        let item = item.map_err(|_| reject())?;
        if count == 3
            || item.is_instance_of::<PyString>()
            || item.is_instance_of::<PyBytes>()
            || item.is_instance_of::<PyByteArray>()
            || item.is_instance_of::<PyMemoryView>()
        {
            return Err(reject());
        }
        out[count] = item.extract::<f32>().map_err(|_| reject())?;
        count += 1;
    }
    if count != 3 || out.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err(reject());
    }
    Ok(out)
}

/// Render a converged path-traced reference of a real DEM under sun + IBL,
/// optionally mixed with mesh geometry (terrain stays a first-class primitive
/// of the shared hybrid traversal).
///
/// Returns a dict:
///   rgba (H,W,4) u8, albedo (H,W,3) f32, normal (H,W,3) f32, depth (H,W) f32
///   frames: int, variance: float (max per-pixel variance of the running-mean
///   luminance across the last convergence window), converged: bool (always
///   True — non-convergence raises), and peak_host_visible_bytes /
///   minmax_pyramid_bytes / gpu_resource_bytes memory diagnostics.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    heightmap,
    width,
    height,
    cam,
    spacing = (1.0, 1.0),
    exaggeration = 1.0,
    albedo = (0.6, 0.6, 0.6),
    sun_azimuth_deg = 315.0,
    sun_elevation_deg = 45.0,
    sun_intensity = 2.5,
    env_map = None,
    env_intensity = 0.35,
    mesh_vertices = None,
    mesh_indices = None,
    spp = 1u32,
    max_frames = 512,
    min_frames = 32,
    variance_threshold = 1e-3,
    seed = 7u32,
    certificate = None,
    sun_color = None,
    cache = None,
    camera_model = None,
    sensor_rect = None,
    full_width = None,
    full_height = None,
    pixel_offset = None,
    albedo_map = None,
    albedo_sampling = "nearest",
))]
pub(crate) fn hybrid_render_terrain_reference(
    py: Python<'_>,
    heightmap: numpy::PyReadonlyArray2<'_, f32>,
    width: u32,
    height: u32,
    cam: &Bound<'_, PyDict>,
    spacing: (f32, f32),
    exaggeration: f32,
    albedo: (f32, f32, f32),
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    env_map: Option<numpy::PyReadonlyArray3<'_, f32>>,
    env_intensity: f32,
    mesh_vertices: Option<numpy::PyReadonlyArray2<'_, f32>>,
    mesh_indices: Option<numpy::PyReadonlyArray2<'_, u32>>,
    spp: u32,
    max_frames: u32,
    min_frames: u32,
    variance_threshold: f32,
    seed: u32,
    certificate: Option<Bound<'_, PyAny>>,
    sun_color: Option<Bound<'_, PyAny>>,
    cache: Option<Bound<'_, PyAny>>,
    camera_model: Option<String>,
    sensor_rect: Option<(f32, f32, f32, f32)>,
    full_width: Option<u32>,
    full_height: Option<u32>,
    pixel_offset: Option<(u32, u32)>,
    albedo_map: Option<numpy::PyReadonlyArray3<'_, f32>>,
    albedo_sampling: &str,
) -> PyResult<Py<PyAny>> {
    let _ = cache;
    use crate::path_tracing::hybrid_compute::{
        AlbedoSampling, CameraModel, HybridPathTracer, TerrainReferenceDesc,
    };
    use numpy::PyArray1;

    let sun_color = match sun_color.as_ref() {
        None => [1.0, 0.97, 0.92],
        Some(obj) => extract_sun_color(obj)?,
    };

    let certificate_capture =
        crate::core::certificate::begin_render_capture("hybrid_render_terrain_reference");
    // Fallible first GPU touch: later ctx() calls cannot fail once this succeeds.
    crate::core::gpu::try_ctx()?;

    let dem = heightmap.as_array();
    let (dem_h, dem_w) = (dem.shape()[0] as u32, dem.shape()[1] as u32);
    let heights: Vec<f32> = dem.iter().copied().collect();

    let get_vec3 = |key: &str, default: [f32; 3]| -> PyResult<[f32; 3]> {
        match cam.get_item(key)? {
            Some(v) => {
                let t: (f32, f32, f32) = v.extract()?;
                Ok([t.0, t.1, t.2])
            }
            None => Ok(default),
        }
    };
    let cam_origin = get_vec3("origin", [0.0, 50.0, 120.0])?;
    let cam_look_at = get_vec3("look_at", [0.0, 0.0, 0.0])?;
    let cam_up = get_vec3("up", [0.0, 1.0, 0.0])?;
    let exposure: f32 = match cam.get_item("exposure")? {
        Some(v) => v.extract()?,
        None => 1.0,
    };
    let camera_model = match camera_model {
        Some(value) => Some(value),
        None => match cam.get_item("model")? {
            Some(value) if !value.is_none() => Some(value.extract::<String>()?),
            _ => None,
        },
    };
    let camera_model = match camera_model.as_deref().unwrap_or("pinhole") {
        "pinhole" => CameraModel::Pinhole,
        "orthographic" => CameraModel::Orthographic,
        "off_axis" => CameraModel::OffAxis,
        value => {
            return Err(PyValueError::new_err(format!(
                "camera_model must be 'pinhole', 'orthographic', or 'off_axis', got {value:?}"
            )))
        }
    };
    let ortho_half_height: f32 = if camera_model == CameraModel::Orthographic {
        match cam.get_item("half_height")? {
            Some(value) if !value.is_none() => value.extract()?,
            _ => 1.0,
        }
    } else {
        1.0
    };
    let fov_y_deg: f32 = if camera_model == CameraModel::Orthographic {
        45.0
    } else {
        match cam.get_item("fov_y")? {
            Some(value) if !value.is_none() => value.extract()?,
            _ => 45.0,
        }
    };
    let sensor_rect = match sensor_rect {
        Some(value) => Some(value),
        None => match cam.get_item("sensor_rect")? {
            Some(value) if !value.is_none() => Some(value.extract::<(f32, f32, f32, f32)>()?),
            _ => None,
        },
    };
    if full_width.is_some() != full_height.is_some() {
        return Err(PyValueError::new_err(
            "full_width and full_height must be provided together",
        ));
    }
    if pixel_offset.is_some_and(|offset| offset != (0, 0)) && full_width.is_none() {
        return Err(PyValueError::new_err(
            "offset renders require full_width and full_height",
        ));
    }
    if sensor_rect.is_some_and(|rect| rect != (0.0, 0.0, 1.0, 1.0)) && full_width.is_none() {
        return Err(PyValueError::new_err(
            "cropped sensor_rect requires full_width and full_height",
        ));
    }
    let global_camera_contract = full_width.is_some() || full_height.is_some();
    let full_width = full_width.unwrap_or(width);
    let full_height = full_height.unwrap_or(height);
    let mut resolved_pixel_offset = pixel_offset.unwrap_or((0, 0));
    if let Some((x0, y0, x1, y1)) = sensor_rect {
        let span_x = (x1 - x0) * full_width as f32;
        let span_y = (y1 - y0) * full_height as f32;
        if (span_x - width as f32).abs() > 1e-4 || (span_y - height as f32).abs() > 1e-4 {
            return Err(PyValueError::new_err(
                "sensor_rect span must match the output tile dimensions",
            ));
        }
        let origin_x = x0 * full_width as f32;
        let origin_y = y0 * full_height as f32;
        if (origin_x - origin_x.round()).abs() > 1e-4 || (origin_y - origin_y.round()).abs() > 1e-4
        {
            return Err(PyValueError::new_err(
                "sensor_rect origin must be pixel-aligned",
            ));
        }
        let rect_offset = (origin_x.round() as u32, origin_y.round() as u32);
        match pixel_offset {
            Some(explicit) if explicit != rect_offset => {
                return Err(PyValueError::new_err(format!(
                    "pixel_offset {explicit:?} does not match sensor_rect origin {rect_offset:?}"
                )));
            }
            Some(_) => {}
            None => resolved_pixel_offset = rect_offset,
        }
    }
    let pixel_offset = resolved_pixel_offset;
    let seamless_camera =
        camera_model != CameraModel::Pinhole || global_camera_contract || pixel_offset != (0, 0);
    let sensor_rect = sensor_rect.unwrap_or_else(|| {
        (
            pixel_offset.0 as f32 / full_width as f32,
            pixel_offset.1 as f32 / full_height as f32,
            (pixel_offset.0 + width) as f32 / full_width as f32,
            (pixel_offset.1 + height) as f32 / full_height as f32,
        )
    });

    let env = match &env_map {
        Some(arr) => {
            let a = arr.as_array();
            if a.shape()[2] != 3 {
                return Err(PyValueError::new_err("env_map must have shape (H, W, 3)"));
            }
            Some((
                a.iter().copied().collect::<Vec<f32>>(),
                a.shape()[1] as u32,
                a.shape()[0] as u32,
            ))
        }
        None => None,
    };

    let albedo_sampling = match albedo_sampling {
        "nearest" => AlbedoSampling::Nearest,
        "bilinear" => AlbedoSampling::Bilinear,
        value => {
            return Err(PyValueError::new_err(format!(
                "albedo_sampling must be 'nearest' or 'bilinear', got {value:?}"
            )))
        }
    };
    let albedo_map = match &albedo_map {
        Some(arr) => {
            let map = arr.as_array();
            let expected = [dem_h as usize, dem_w as usize, 4];
            if map.shape() != expected {
                return Err(PyValueError::new_err(format!(
                    "albedo_map must have shape ({dem_h}, {dem_w}, 4), got {:?}",
                    map.shape()
                )));
            }
            Some(map.iter().copied().collect::<Vec<f32>>())
        }
        None => None,
    };

    let mesh = match (&mesh_vertices, &mesh_indices) {
        (Some(v), Some(i)) => {
            let v = v.as_array();
            let i = i.as_array();
            if v.shape()[1] != 3 {
                return Err(PyValueError::new_err(
                    "mesh_vertices must have shape (N, 3)",
                ));
            }
            if i.shape()[1] != 3 {
                return Err(PyValueError::new_err("mesh_indices must have shape (M, 3)"));
            }
            Some((
                v.iter().copied().collect::<Vec<f32>>(),
                i.iter().copied().collect::<Vec<u32>>(),
            ))
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "mesh_vertices and mesh_indices must be provided together",
            ))
        }
    };

    let desc = TerrainReferenceDesc {
        heights,
        dem_width: dem_w,
        dem_height: dem_h,
        spacing,
        exaggeration,
        albedo: [albedo.0, albedo.1, albedo.2],
        albedo_map,
        albedo_sampling,
        cam_origin,
        cam_look_at,
        cam_up,
        camera_model,
        seamless_camera,
        fov_y_deg,
        ortho_half_height,
        sensor_rect: [sensor_rect.0, sensor_rect.1, sensor_rect.2, sensor_rect.3],
        full_width,
        full_height,
        pixel_offset_x: pixel_offset.0,
        pixel_offset_y: pixel_offset.1,
        exposure,
        sun_azimuth_deg,
        sun_elevation_deg,
        sun_intensity,
        sun_color,
        env_map: env,
        env_intensity,
        mesh,
        width,
        height,
        seed,
        spp,
        max_frames,
        min_frames,
        variance_threshold,
    };

    let model_name = match camera_model {
        CameraModel::Pinhole => "pinhole",
        CameraModel::Orthographic => "orthographic",
        CameraModel::OffAxis => "off_axis",
    };
    crate::core::certificate::record_input("camera_model", model_name);
    crate::core::certificate::record_input(
        "sensor_rect",
        format!(
            "{},{},{},{}",
            sensor_rect.0, sensor_rect.1, sensor_rect.2, sensor_rect.3
        ),
    );
    crate::core::certificate::record_input(
        "albedo_map_sha256",
        desc.albedo_map
            .as_deref()
            .map(|map| {
                crate::core::provenance::to_hex(&crate::core::provenance::sha256(
                    bytemuck::cast_slice(map),
                ))
            })
            .unwrap_or_else(|| "none".to_string()),
    );
    crate::core::certificate::record_input(
        "albedo_sampling",
        match albedo_sampling {
            AlbedoSampling::Nearest => "nearest",
            AlbedoSampling::Bilinear => "bilinear",
        },
    );
    crate::core::certificate::record_input(
        "albedo",
        format!("{},{},{}", desc.albedo[0], desc.albedo[1], desc.albedo[2]),
    );

    let owner = crate::core::resource_tracker::AllocationOwner::new();
    let _owner_guard = owner.activate();
    let owner_capture = crate::core::resource_tracker::begin_owner_capture(owner.id());
    let tracer = HybridPathTracer::new()?;
    let out = tracer.render_terrain_reference(&desc)?;
    let owner_report = owner_capture.finish();
    if owner_report.peak_host_visible_bytes > 512 * 1024 * 1024 {
        return Err(PyRuntimeError::new_err(format!(
            "terrain tile exceeded 512 MiB host-visible budget: {}",
            owner_report.peak_host_visible_bytes
        )));
    }

    let d = PyDict::new_bound(py);
    let rgba = PyArray1::<u8>::from_vec_bound(py, out.rgba).reshape([
        height as usize,
        width as usize,
        4,
    ])?;
    let albedo_arr = PyArray1::<f32>::from_vec_bound(py, out.albedo).reshape([
        height as usize,
        width as usize,
        3,
    ])?;
    let normal_arr = PyArray1::<f32>::from_vec_bound(py, out.normal).reshape([
        height as usize,
        width as usize,
        3,
    ])?;
    let depth_arr = PyArray1::<f32>::from_vec_bound(py, out.depth)
        .reshape([height as usize, width as usize])?;
    d.set_item("rgba", rgba)?;
    d.set_item("albedo", albedo_arr)?;
    d.set_item("normal", normal_arr)?;
    d.set_item("depth", depth_arr)?;
    d.set_item("frames", out.frames)?;
    d.set_item("variance", out.variance)?;
    d.set_item("converged", out.converged)?;
    d.set_item(
        "peak_host_visible_bytes",
        owner_report.peak_host_visible_bytes,
    )?;
    d.set_item(
        "peak_device_local_bytes",
        owner_report.peak_device_local_bytes,
    )?;
    d.set_item("reservoir_valid_count", out.reservoir_valid_count)?;
    d.set_item("reservoir_m_min", out.reservoir_m_min)?;
    d.set_item("reservoir_m_max", out.reservoir_m_max)?;
    d.set_item("minmax_pyramid_bytes", out.minmax_pyramid_bytes)?;
    d.set_item("gpu_resource_bytes", out.gpu_resource_bytes)?;
    // The hybrid_pt.* passes (live gpu_ms when timestamps are granted) are
    // recorded inside HybridPathTracer::render_terrain_reference.
    certificate_capture.finish();
    crate::core::certificate::emit_certificate_for_kwarg(py, certificate.as_ref())?;
    Ok(d.into_py(py))
}
