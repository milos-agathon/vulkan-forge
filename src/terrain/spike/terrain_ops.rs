use super::*;

#[pymethods]
impl TerrainSpike {
    // E2: Set morph factor (0..1) and optional coarse_factor (>=1)
    #[pyo3(text_signature = "($self, morph, coarse_factor=None)")]
    pub fn set_lod_morph(&mut self, morph: f32, coarse_factor: Option<f32>) -> PyResult<()> {
        let m = morph.clamp(0.0, 1.0);
        self.globals.lod_morph = m;
        if let Some(cf) = coarse_factor {
            if !cf.is_finite() || cf < 1.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "coarse_factor must be finite and >= 1.0",
                ));
            }
            self.globals.coarse_factor = cf;
        }

        // Rebuild uniforms with same view/proj but updated tail
        let view = glam::Mat4::from_cols_array_2d(&self.last_uniforms.view);
        let proj = glam::Mat4::from_cols_array_2d(&self.last_uniforms.proj);
        let uniforms = self.globals.to_uniforms(view, proj);
        self.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));
        self.last_uniforms = uniforms;
        Ok(())
    }

    // E2: Set skirt depth (>=0)
    #[pyo3(text_signature = "($self, depth)")]
    pub fn set_skirt_depth(&mut self, depth: f32) -> PyResult<()> {
        if !depth.is_finite() || depth < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "skirt depth must be finite and >= 0",
            ));
        }
        self.globals.skirt_depth = depth;
        // Rebuild uniforms with same view/proj but updated tail
        let view = glam::Mat4::from_cols_array_2d(&self.last_uniforms.view);
        let proj = glam::Mat4::from_cols_array_2d(&self.last_uniforms.proj);
        let uniforms = self.globals.to_uniforms(view, proj);
        self.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));
        self.last_uniforms = uniforms;
        Ok(())
    }

    // E2: Set per-edge skirt mask (bitmask: 1=left, 2=right, 4=bottom, 8=top)
    #[pyo3(text_signature = "($self, mask)")]
    pub fn set_skirt_mask(&mut self, mask: u32) -> PyResult<()> {
        let m = mask & 0xF;
        self.globals.skirt_mask = m;
        // Rebuild uniforms with same view/proj but updated tail
        let view = glam::Mat4::from_cols_array_2d(&self.last_uniforms.view);
        let proj = glam::Mat4::from_cols_array_2d(&self.last_uniforms.proj);
        let uniforms = self.globals.to_uniforms(view, proj);
        self.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));
        self.last_uniforms = uniforms;
        Ok(())
    }

    // E2: Convenience method to set skirt edges using booleans
    #[pyo3(text_signature = "($self, left, right, bottom, top)")]
    pub fn set_skirt_edges(
        &mut self,
        left: bool,
        right: bool,
        bottom: bool,
        top: bool,
    ) -> PyResult<()> {
        let mut mask = 0u32;
        if left {
            mask |= 0x1;
        }
        if right {
            mask |= 0x2;
        }
        if bottom {
            mask |= 0x4;
        }
        if top {
            mask |= 0x8;
        }
        self.set_skirt_mask(mask)
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_lut_format(&self) -> &'static str {
        self.lut_format
    }

    #[pyo3(text_signature = "($self, eye, target, up, fovy_deg, znear, zfar)")]
    pub fn set_camera_look_at(
        &mut self,
        eye: (f32, f32, f32),
        target: (f32, f32, f32),
        up: (f32, f32, f32),
        fovy_deg: f32,
        znear: f32,
        zfar: f32,
    ) -> PyResult<()> {
        use crate::camera;

        // Compute aspect ratio from current framebuffer dimensions
        let aspect = self.width as f32 / self.height as f32;

        // Validate parameters using camera module validators
        let eye_vec = glam::Vec3::new(eye.0, eye.1, eye.2);
        let target_vec = glam::Vec3::new(target.0, target.1, target.2);
        let up_vec = glam::Vec3::new(up.0, up.1, up.2);

        camera::validate_camera_params(eye_vec, target_vec, up_vec, fovy_deg, znear, zfar)?;

        // Compute view and projection matrices
        let view = glam::Mat4::look_at_rh(eye_vec, target_vec, up_vec);
        let fovy_rad = fovy_deg.to_radians();
        let proj = camera::perspective_wgpu(fovy_rad, aspect, znear, zfar);

        // Build new uniforms using existing globals
        let uniforms = self.globals.to_uniforms(view, proj);

        // Write to UBO
        self.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&uniforms));

        // Store for debugging
        self.last_uniforms = uniforms;

        Ok(())
    }

    #[pyo3(text_signature = "($self)")]
    pub fn debug_uniforms_f32<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
        // Return only the first 44 floats (176 bytes) for T31 compatibility:
        // [0..15]=view, [16..31]=proj, [32..35]=sun_exposure, [36..39]=spacing/h_range/exag/0, [40..43]=pad
        let uniform_lanes = self.last_uniforms.to_debug_lanes_44();
        Ok(numpy::PyArray1::from_slice_bound(py, &uniform_lanes))
    }

    // B15: Expose memory metrics to Python
    #[pyo3(text_signature = "($self)")]
    pub fn get_memory_metrics<'py>(
        &self,
        py: pyo3::Python<'py>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        let tracker = global_tracker();
        let metrics = tracker.get_metrics();

        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("buffer_count", metrics.buffer_count)?;
        dict.set_item("texture_count", metrics.texture_count)?;
        dict.set_item("buffer_bytes", metrics.buffer_bytes)?;
        dict.set_item("texture_bytes", metrics.texture_bytes)?;
        dict.set_item("host_visible_bytes", metrics.host_visible_bytes)?;
        dict.set_item("total_bytes", metrics.total_bytes)?;
        dict.set_item("limit_bytes", metrics.limit_bytes)?;
        dict.set_item("within_budget", metrics.within_budget)?;
        dict.set_item("utilization_ratio", metrics.utilization_ratio)?;
        dict.set_item("resident_tiles", metrics.resident_tiles)?;
        dict.set_item("resident_tile_bytes", metrics.resident_tile_bytes)?;
        for (slot, family) in ["albedo", "normal", "mask", "height"].iter().enumerate() {
            dict.set_item(
                format!("resident_tiles_{family}"),
                metrics.resident_tiles_by_family[slot],
            )?;
            dict.set_item(
                format!("resident_bytes_{family}"),
                metrics.resident_tile_bytes_by_family[slot],
            )?;
        }
        dict.set_item("staging_bytes_in_flight", metrics.staging_bytes_in_flight)?;
        dict.set_item("staging_ring_count", metrics.staging_ring_count)?;
        dict.set_item("staging_buffer_size", metrics.staging_buffer_size)?;
        dict.set_item("staging_buffer_stalls", metrics.staging_buffer_stalls)?;

        Ok(dict)
    }
}
