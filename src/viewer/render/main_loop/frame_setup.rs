use glam::Mat4;

use crate::viewer::{SkyUniforms, Viewer, VIEWER_SNAPSHOT_MAX_MEGAPIXELS};

/// Solar altitude at which direct sunlight is fully extinguished, degrees.
///
/// Matches the lower edge of the declared SIDERA civil-to-astronomical ramp in
/// `src/shaders/sky.wgsl`, so the disc, the scattering lobe and the sky
/// background all reach night together.
const SUN_FADE_START_DEG: f32 = -18.0;
/// Solar altitude above which direct sunlight is at full strength, degrees.
const SUN_FADE_END_DEG: f32 = -4.0;

/// Hermite smoothstep, matching the WGSL builtin used by the sky shader.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Viewer {
    pub(super) fn prepare_render_frame(
        &mut self,
    ) -> Result<
        (
            wgpu::SurfaceTexture,
            wgpu::TextureView,
            Option<(u32, u32)>,
            wgpu::CommandEncoder,
        ),
        wgpu::SurfaceError,
    > {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if self.frame_count == 0 {
            eprintln!("[viewer-debug] entering render loop (first frame)");
        }
        self.sync_astro_observation();
        self.update_lit_uniform();

        // Ensure auto-snapshot request is registered before encoding so we render to an offscreen texture
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
                eprintln!("[viewer-debug] auto snapshot requested: {}", p);
            }
        }

        // Compute snapshot dimensions if requested (but don't resize yet - we'll render to offscreen)
        let _snapshot_dimensions = if self.snapshot_request.is_some() {
            let (req_w, req_h) = if let (Some(w), Some(h)) = (
                self.view_config.snapshot_width,
                self.view_config.snapshot_height,
            ) {
                (w, h)
            } else {
                (self.config.width, self.config.height)
            };

            // Apply soft megapixel clamp
            let mut snap_w = req_w;
            let mut snap_h = req_h;
            if self.view_config.snapshot_width.is_some()
                && self.view_config.snapshot_height.is_some()
            {
                let pixels = snap_w as u64 * snap_h as u64;
                let max_pixels = (VIEWER_SNAPSHOT_MAX_MEGAPIXELS * 1_000_000.0) as u64;
                if pixels > max_pixels {
                    let scale = (max_pixels as f32 / pixels as f32).sqrt();
                    snap_w = ((snap_w as f32) * scale).floor().max(1.0) as u32;
                    snap_h = ((snap_h as f32) * scale).floor().max(1.0) as u32;
                }
            }

            if snap_w != self.config.width || snap_h != self.config.height {
                eprintln!(
                    "[viewer] Snapshot requested at {}x{} (window: {}x{})",
                    snap_w, snap_h, self.config.width, self.config.height
                );
            }

            Some((snap_w, snap_h))
        } else {
            None
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Update labels with current camera (before any rendering)
        // Use terrain camera if terrain is loaded, otherwise use viewer camera
        let frame = self.current_frame_camera();
        self.update_labels_for_frame(frame);

        // Render sky background (compute) before opaques
        if self.sky_enabled {
            // Build camera matrices (view, proj, inv_view, inv_proj) and eye
            let frame = self.current_frame_camera();
            let proj = frame.projection(self.config.width, self.config.height);
            let view_mat = frame.view();
            let inv_view = view_mat.inverse();
            let inv_proj = proj.inverse();
            fn to_arr4(m: Mat4) -> [[f32; 4]; 4] {
                let c = m.to_cols_array();
                [
                    [c[0], c[1], c[2], c[3]],
                    [c[4], c[5], c[6], c[7]],
                    [c[8], c[9], c[10], c[11]],
                    [c[12], c[13], c[14], c[15]],
                ]
            }
            let eye = frame.render_eye();
            let cam_buf: [[[f32; 4]; 4]; 4] = [
                to_arr4(view_mat),
                to_arr4(proj),
                to_arr4(inv_view),
                to_arr4(inv_proj),
            ];
            // Write matrices
            self.queue
                .write_buffer(&self.sky_camera, 0, bytemuck::cast_slice(&cam_buf));
            // Write eye position (vec4 packed)
            let eye4: [f32; 4] = [eye.x, eye.y, eye.z, 0.0];
            let base = (std::mem::size_of::<[[f32; 4]; 4]>() * 4) as u64;
            self.queue
                .write_buffer(&self.sky_camera, base, bytemuck::cast_slice(&eye4));
            let viewport: [f32; 4] = [
                self.config.width as f32,
                self.config.height as f32,
                1.0 / self.config.width as f32,
                1.0 / self.config.height as f32,
            ];
            self.queue.write_buffer(
                &self.sky_camera,
                base + std::mem::size_of::<[f32; 4]>() as u64,
                bytemuck::cast_slice(&viewport),
            );

            // Update sky params each frame based on viewer-set fields
            let sun_dir_ws = glam::Vec3::from_array(self.sky_sun_direction_ws).normalize();
            let model_id: u32 = self.sky_model_id;
            let turb: f32 = self.sky_turbidity.clamp(1.0, 10.0);
            let ground: f32 = self.sky_ground_albedo.clamp(0.0, 1.0);
            let expose: f32 = self.sky_exposure.max(0.0);
            // Fade the sun disc and its scattering lobe on the *same* declared
            // ramp `sky.wgsl` uses to fade the daylight fit, rather than
            // stepping to zero at altitude 0: a hard step pops the disc, the
            // scattering and the fog term a full frame before the sky itself
            // starts to darken, which is visible in any sunset sweep.
            let solar_altitude_deg = sun_dir_ws.y.clamp(-1.0, 1.0).asin().to_degrees();
            let sun_i: f32 = self.sky_sun_intensity.max(0.0)
                * smoothstep(SUN_FADE_START_DEG, SUN_FADE_END_DEG, solar_altitude_deg);

            let sky_params_frame = SkyUniforms::new(
                [sun_dir_ws.x, sun_dir_ws.y, sun_dir_ws.z],
                turb,
                ground,
                1.0,
                sun_i,
                expose,
                model_id,
            );
            self.queue
                .write_buffer(&self.sky_params, 0, bytemuck::bytes_of(&sky_params_frame));

            self.ensure_sky_bind_groups();
            let sky_bg0 = self.sky_bg0_cache.borrow();
            let sky_bg1 = self.sky_bg1_cache.borrow();
            let gx = (self.config.width + 7) / 8;
            let gy = (self.config.height + 7) / 8;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("viewer.sky.compute"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.sky_pipeline);
                cpass.set_bind_group(0, sky_bg0.as_ref().unwrap(), &[]);
                cpass.set_bind_group(1, sky_bg1.as_ref().unwrap(), &[]);
                cpass.dispatch_workgroups(gx, gy, 1);
            }
            if self.night_instance_count > 0 {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer.night.overlay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.sky_output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.night_pipeline);
                pass.set_bind_group(0, &self.night_bind_group, &[]);
                pass.set_bind_group(1, sky_bg1.as_ref().unwrap(), &[]);
                pass.draw(0..6, 0..self.night_instance_count);
            }
        }

        // Composite debug: after GI/geometry, show GBuffer material on swapchain

        Ok((output, view, _snapshot_dimensions, encoder))
    }
}
