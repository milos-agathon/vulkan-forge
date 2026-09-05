use super::*;

#[pyfunction]
pub(crate) fn _pt_render_gpu_mesh(
    py: Python<'_>,
    width: u32,
    height: u32,
    vertices: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    cam: &Bound<'_, PyAny>,
    seed: u32,
    frames: u32,
    lighting_type: &str,
    lighting_intensity: f32,
    lighting_azimuth: f32,
    lighting_elevation: f32,
    shadows: bool,
    shadow_intensity: f32,
) -> PyResult<Py<PyAny>> {
    use numpy::{PyArray1, PyReadonlyArray2};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};

    // Fallible first GPU touch: later ctx() calls cannot fail once this succeeds.
    crate::core::gpu::try_ctx()?;

    let verts_arr: PyReadonlyArray2<f32> = vertices.extract().map_err(|_| {
        PyValueError::new_err("vertices must be a NumPy array with shape (N,3) float32")
    })?;
    let idx_arr: PyReadonlyArray2<u32> = indices.extract().map_err(|_| {
        PyValueError::new_err("indices must be a NumPy array with shape (M,3) uint32")
    })?;

    let v = verts_arr.as_array();
    let i = idx_arr.as_array();
    if v.ndim() != 2 || v.shape()[1] != 3 {
        return Err(PyValueError::new_err("vertices must have shape (N,3)"));
    }
    if i.ndim() != 2 || i.shape()[1] != 3 {
        return Err(PyValueError::new_err("indices must have shape (M,3)"));
    }

    let mut verts: Vec<crate::sdf::hybrid::Vertex> = Vec::with_capacity(v.shape()[0]);
    for row in v.rows() {
        verts.push(crate::sdf::hybrid::Vertex {
            position: [row[0], row[1], row[2]],
            _pad: 0.0,
        });
    }

    let mut flat_idx: Vec<u32> = Vec::with_capacity(i.shape()[0] * 3);
    for row in i.rows() {
        flat_idx.push(row[0]);
        flat_idx.push(row[1]);
        flat_idx.push(row[2]);
    }

    let mut tris: Vec<crate::accel::types::Triangle> = Vec::with_capacity(i.shape()[0]);
    for row in i.rows() {
        let iv0 = row[0] as usize;
        let iv1 = row[1] as usize;
        let iv2 = row[2] as usize;
        if iv0 >= v.shape()[0] || iv1 >= v.shape()[0] || iv2 >= v.shape()[0] {
            return Err(PyValueError::new_err(
                "indices reference out-of-bounds vertex",
            ));
        }
        let v0 = [v[[iv0, 0]], v[[iv0, 1]], v[[iv0, 2]]];
        let v1 = [v[[iv1, 0]], v[[iv1, 1]], v[[iv1, 2]]];
        let v2 = [v[[iv2, 0]], v[[iv2, 1]], v[[iv2, 2]]];
        tris.push(crate::accel::types::Triangle::new(v0, v1, v2));
    }

    let options = crate::accel::types::BuildOptions::default();
    let bvh_handle =
        crate::accel::build_bvh(&tris, &options, crate::accel::GpuContext::NotAvailable)
            .map_err(|e| PyRuntimeError::new_err(format!("BVH build failed: {}", e)))?;

    let mut hybrid = crate::sdf::hybrid::HybridScene::mesh_only(verts, flat_idx, bvh_handle);

    let origin: (f32, f32, f32) = cam.get_item("origin")?.extract()?;
    let look_at: (f32, f32, f32) = cam.get_item("look_at")?.extract()?;
    let up: (f32, f32, f32) = cam
        .get_item("up")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((0.0, 1.0, 0.0));
    let fov_y: f32 = cam
        .get_item("fov_y")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(45.0);
    let aspect: f32 = cam
        .get_item("aspect")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or((width as f32) / (height as f32));
    let exposure: f32 = cam
        .get_item("exposure")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(1.0);

    let o = glam::Vec3::new(origin.0, origin.1, origin.2);
    let la = glam::Vec3::new(look_at.0, look_at.1, look_at.2);
    let upv = glam::Vec3::new(up.0, up.1, up.2);
    let forward = (la - o).normalize_or_zero();
    let right = forward.cross(upv).normalize_or_zero();
    let cup = right.cross(forward).normalize_or_zero();

    let uniforms = crate::path_tracing::compute::Uniforms {
        width,
        height,
        frame_index: 0,
        aov_flags: 0,
        cam_origin: [origin.0, origin.1, origin.2],
        cam_fov_y: fov_y,
        cam_right: [right.x, right.y, right.z],
        cam_aspect: aspect,
        cam_up: [cup.x, cup.y, cup.z],
        cam_exposure: exposure,
        cam_forward: [forward.x, forward.y, forward.z],
        seed_hi: seed,
        seed_lo: frames,
        camera_model: 0,
        full_width: width,
        full_height: height,
        pixel_offset_x: 0,
        pixel_offset_y: 0,
        ortho_half_height: 1.0,
        camera_flags: 0,
        sensor_rect: [0.0, 0.0, 1.0, 1.0],
    };

    let lighting_type_id = match lighting_type.to_lowercase().as_str() {
        "flat" => 0u32,
        "lambertian" | "lambert" => 1u32,
        "phong" => 2u32,
        "blinn-phong" | "blinn_phong" | "blinnphong" => 3u32,
        _ => 1u32,
    };

    let azimuth_rad = lighting_azimuth.to_radians();
    let elevation_rad = lighting_elevation.to_radians();
    let light_dir = [
        azimuth_rad.cos() * elevation_rad.cos(),
        elevation_rad.sin(),
        azimuth_rad.sin() * elevation_rad.cos(),
    ];
    let len =
        (light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2])
            .sqrt();
    let light_dir_normalized = if len > 1e-6 {
        [light_dir[0] / len, light_dir[1] / len, light_dir[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    };

    let lighting_uniforms = crate::path_tracing::hybrid_compute::LightingUniforms {
        light_dir: light_dir_normalized,
        lighting_type: lighting_type_id,
        light_color: [
            lighting_intensity,
            lighting_intensity * 0.95,
            lighting_intensity * 0.8,
        ],
        shadows_enabled: if shadows { 1 } else { 0 },
        ambient_color: [0.1, 0.12, 0.15],
        shadow_intensity,
        hdri_intensity: 0.0,
        hdri_rotation: 0.0,
        specular_power: 32.0,
        _pad: [0, 0, 0, 0, 0],
    };

    let params = crate::path_tracing::hybrid_compute::HybridTracerParams {
        base_uniforms: uniforms,
        lighting_uniforms,
        traversal_mode: crate::path_tracing::hybrid_compute::TraversalMode::MeshOnly,
        early_exit_distance: 0.01,
        shadow_softness: 4.0,
    };

    let rgba: Vec<u8> = {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let params = params.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            hybrid
                .prepare_gpu_resources()
                .map_err(|error| format!("hybrid scene preparation failed: {error}"))?;
            let tracer = crate::path_tracing::hybrid_compute::HybridPathTracer::new()
                .map_err(|error| format!("hybrid path tracer creation failed: {error}"))?;
            tracer
                .render(width, height, &[], &hybrid, None, params)
                .map_err(|error| format!("hybrid path trace failed: {error}"))
        }));
        match result {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(error)) => return Err(PyRuntimeError::new_err(error)),
            Err(_) => {
                return Err(PyRuntimeError::new_err(
                    "hybrid GPU path tracer panicked; no fallback image was emitted",
                ))
            }
        }
    };

    let arr1 = PyArray1::<u8>::from_vec_bound(py, rgba);
    let arr3 = arr1.reshape([height as usize, width as usize, 4])?;
    Ok(arr3.into_py(py))
}
