use super::*;

impl ViewerTerrainScene {
    pub fn render_shadow_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_view: glam::Mat4,
        camera_proj: glam::Mat4,
        sun_direction: glam::Vec3,
        render_origin_span: [f32; 4],
    ) {
        // Early exit if shadow infrastructure not ready
        let csm = match self.csm_renderer.as_mut() {
            Some(c) => c,
            None => return,
        };

        let terrain = match self.terrain.as_ref() {
            Some(t) => t,
            None => return,
        };

        if self.shadow_pipeline.is_none() || self.shadow_bind_groups.is_empty() {
            return;
        }

        let cascade_count = csm.config.cascade_count;
        let shadow_map_size = csm.config.shadow_map_size;

        // Mark that shadow passes are running (will be copied to csm_uniforms later)
        csm.uniforms.technique_reserved[0] = 1.0; // Flag: shadow passes executed

        // Update CSM cascade matrices based on camera and light
        let near_plane = 1.0;
        let far_plane = csm.config.max_shadow_distance;
        csm.update_cascades(
            camera_view,
            camera_proj,
            sun_direction,
            near_plane,
            far_plane,
        );

        // Get terrain parameters for shadow pass uniforms
        let (min_h, max_h) = terrain.domain;
        let terrain_width = terrain.terrain_width();
        let z_scale = terrain.z_scale;
        let grid_res = 512u32; // Shadow pass grid resolution

        // Height curve params: use linear (mode=0) with no curve transformation
        // This matches the default terrain rendering behavior
        let height_curve_mode = 0.0_f32; // 0=linear
        let height_curve_strength = 0.0_f32; // No curve applied
        let height_curve_power = 1.0_f32; // Default power

        // Render each cascade
        for cascade_idx in 0..cascade_count as usize {
            if cascade_idx >= self.shadow_bind_groups.len()
                || cascade_idx >= self.shadow_uniform_buffers.len()
            {
                break;
            }

            // Render with the exact cascade matrix the shading pass will sample.
            // Overriding this with a guessed terrain-wide projection caused large
            // swaths of the map to compare as shadowed.
            let light_view_proj = csm.uniforms.cascades[cascade_idx].light_view_proj;

            // Build shadow pass uniforms
            // Match main shader terrain_params layout: [min_h, h_range, terrain_width, z_scale]
            let shadow_uniforms = ShadowPassUniforms {
                light_view_proj,
                terrain_params: [min_h, max_h - min_h, terrain_width, z_scale],
                grid_params: [grid_res as f32, 0.0, 0.0, 0.0],
                height_curve: [
                    height_curve_mode,
                    height_curve_strength,
                    height_curve_power,
                    0.0,
                ],
                render_origin_xz: [render_origin_span[0], render_origin_span[1]],
                render_span_xz: [render_origin_span[2], render_origin_span[3]],
            };

            // Upload uniforms
            self.queue.write_buffer(
                &self.shadow_uniform_buffers[cascade_idx],
                0,
                bytemuck::cast_slice(&[shadow_uniforms]),
            );

            // Get shadow map view for this cascade
            let shadow_map_view = &csm.shadow_map_views[cascade_idx];

            // Begin depth-only render pass for this cascade
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("shadow_depth_pass_cascade_{}", cascade_idx)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, &self.shadow_bind_groups[cascade_idx], &[]);

            // Draw terrain grid (6 vertices per quad, (grid_res-1)^2 quads)
            let vertex_count = 6 * (grid_res - 1) * (grid_res - 1);
            render_pass.draw(0..vertex_count, 0..1);
        }

        // Execute moment generation pass for VSM/EVSM/MSM techniques
        // This converts the depth maps into moment statistics
        let Some(technique) =
            crate::lighting::shadow::ShadowTechnique::from_name(&self.pbr_config.shadow_technique)
        else {
            return;
        };
        if !technique.requires_moments() {
            return;
        }

        // Prepare and execute moment pass if we have the resources
        let mut disable_failed_blur = false;
        if let (Some(ref mut moment_pass), Some(ref csm)) =
            (&mut self.moment_pass, &self.csm_renderer)
        {
            if let Some(ref moment_texture) = csm.evsm_maps {
                if let Err(error) = moment_pass.prepare_textures_checked(
                    &self.device,
                    csm.shadow_maps.as_ref(),
                    moment_texture,
                ) {
                    eprintln!("[terrain_scene] failed to prepare moment maps: {error}");
                    crate::core::degradation::record_degradation(
                        "validation_failure",
                        "viewer.csm_moment_generation",
                        &error.to_string(),
                    );
                    return;
                }
                if let Err(error) = moment_pass.execute_checked(
                    &self.queue,
                    encoder,
                    technique,
                    cascade_count,
                    shadow_map_size,
                    csm.config.evsm_positive_exp,
                    csm.config.evsm_negative_exp,
                ) {
                    eprintln!("[terrain_scene] failed to generate moment maps: {error}");
                    crate::core::degradation::record_degradation(
                        "validation_failure",
                        "viewer.csm_moment_generation",
                        &error.to_string(),
                    );
                    return;
                }

                if crate::shadows::requires_moment_blur(technique) {
                    let Some(ref mut blur_pass) = self.moment_blur_pass else {
                        return;
                    };
                    if let Err(e) = blur_pass.execute(
                        &self.device,
                        &self.queue,
                        encoder,
                        moment_texture,
                        cascade_count,
                        shadow_map_size,
                        crate::shadows::DEFAULT_MOMENT_BLUR_RADIUS,
                        technique,
                        csm.config.evsm_positive_exp,
                    ) {
                        disable_failed_blur = true;
                        eprintln!("[terrain_scene] failed to blur moment maps: {e}");
                        crate::core::degradation::record_degradation(
                            "allocation_fallback",
                            "viewer.csm_moment_blur",
                            "moment-map filtering unavailable",
                        );
                    }
                }
            }
        }
        if disable_failed_blur {
            self.moment_blur_pass = None;
        }
    }
}
