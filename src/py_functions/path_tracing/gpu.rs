use super::*;

#[pyfunction]
pub(crate) fn _pt_render_gpu(
    py: Python<'_>,
    width: u32,
    height: u32,
    scene: &Bound<'_, PyAny>,
    cam: &Bound<'_, PyAny>,
    seed: u32,
    _frames: u32,
) -> PyResult<Py<PyAny>> {
    use crate::path_tracing::compute::{PathTracerGPU, Sphere as PtSphere, Uniforms as PtUniforms};

    let mut spheres: Vec<PtSphere> = Vec::new();
    if let Ok(seq) = scene.extract::<Vec<&PyAny>>() {
        for item in seq.iter() {
            let dict = item
                .downcast::<pyo3::types::PyDict>()
                .map_err(|_| PyValueError::new_err("scene items must be dicts"))?;
            let center: (f32, f32, f32) = dict
                .get_item("center")?
                .ok_or_else(|| PyValueError::new_err("sphere missing 'center'"))?
                .extract()?;
            let radius: f32 = dict
                .get_item("radius")?
                .ok_or_else(|| PyValueError::new_err("sphere missing 'radius'"))?
                .extract()?;
            let albedo: (f32, f32, f32) = if let Some(value) = dict.get_item("albedo")? {
                value.extract()?
            } else {
                (0.8, 0.8, 0.8)
            };
            let metallic: f32 = if let Some(value) = dict.get_item("metallic")? {
                value.extract()?
            } else {
                0.0
            };
            let roughness: f32 = if let Some(value) = dict.get_item("roughness")? {
                value.extract()?
            } else {
                0.5
            };
            let emissive: (f32, f32, f32) = if let Some(value) = dict.get_item("emissive")? {
                value.extract()?
            } else {
                (0.0, 0.0, 0.0)
            };
            let ior: f32 = if let Some(value) = dict.get_item("ior")? {
                value.extract()?
            } else {
                1.0
            };
            let ax: f32 = if let Some(value) = dict.get_item("ax")? {
                value.extract()?
            } else {
                0.2
            };
            let ay: f32 = if let Some(value) = dict.get_item("ay")? {
                value.extract()?
            } else {
                0.2
            };

            spheres.push(PtSphere {
                center: [center.0, center.1, center.2],
                radius,
                albedo: [albedo.0, albedo.1, albedo.2],
                metallic,
                emissive: [emissive.0, emissive.1, emissive.2],
                roughness,
                ior,
                ax,
                ay,
                _pad1: 0.0,
            });
        }
    }

    let origin: (f32, f32, f32) = cam.get_item("origin")?.extract()?;
    let look_at: (f32, f32, f32) = cam.get_item("look_at")?.extract()?;
    let up: (f32, f32, f32) = cam
        .get_item("up")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or((0.0, 1.0, 0.0));
    let fov_y: f32 = cam
        .get_item("fov_y")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or(45.0);
    let aspect: f32 = cam
        .get_item("aspect")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or((width as f32) / (height as f32));
    let exposure: f32 = cam
        .get_item("exposure")
        .ok()
        .and_then(|value| value.extract().ok())
        .unwrap_or(1.0);

    let origin_vec = Vec3::new(origin.0, origin.1, origin.2);
    let look_at_vec = Vec3::new(look_at.0, look_at.1, look_at.2);
    let up_vec = Vec3::new(up.0, up.1, up.2);
    let forward = (look_at_vec - origin_vec).normalize_or_zero();
    let right = forward.cross(up_vec).normalize_or_zero();
    let camera_up = right.cross(forward).normalize_or_zero();

    let uniforms = PtUniforms {
        width,
        height,
        frame_index: 0,
        aov_flags: 0,
        cam_origin: [origin.0, origin.1, origin.2],
        cam_fov_y: fov_y,
        cam_right: [right.x, right.y, right.z],
        cam_aspect: aspect,
        cam_up: [camera_up.x, camera_up.y, camera_up.z],
        cam_exposure: exposure,
        cam_forward: [forward.x, forward.y, forward.z],
        seed_hi: seed,
        seed_lo: 0,
        camera_model: 0,
        full_width: width,
        full_height: height,
        pixel_offset_x: 0,
        pixel_offset_y: 0,
        ortho_half_height: 1.0,
        camera_flags: 0,
        sensor_rect: [0.0, 0.0, 1.0, 1.0],
    };

    let rgba =
        std::panic::catch_unwind(|| PathTracerGPU::render(width, height, &spheres, uniforms))
            .map_err(|_| {
                PyRuntimeError::new_err("GPU path tracer panicked; no fallback image was emitted")
            })?
            .map_err(|error| PyRuntimeError::new_err(format!("GPU path tracer failed: {error}")))?;
    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}
