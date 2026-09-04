use super::*;

fn terrain_shadow_uniform_params(
    settings: &crate::terrain::render_params::ShadowSettingsNative,
) -> ([f32; 4], [f32; 4]) {
    (
        [
            settings.pcss_blocker_radius,
            settings.pcss_filter_radius,
            0.0005,
            settings.light_size,
        ],
        [settings.pcss_light_radius, 0.0, 0.0, 0.0],
    )
}

pub(in crate::terrain::renderer) struct ShadowSetup {
    pub(in crate::terrain::renderer) eye: glam::Vec3,
    pub(in crate::terrain::renderer) view_matrix: glam::Mat4,
    pub(in crate::terrain::renderer) proj_matrix: glam::Mat4,
    pub(in crate::terrain::renderer) height_exag: f32,
    pub(in crate::terrain::renderer) height_min: f32,
    pub(in crate::terrain::renderer) shadow_bind_group: Option<wgpu::BindGroup>,
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn generate_shadow_moments(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let Some(moment_pass) = self.moment_pass.as_mut() else {
            return Ok(());
        };
        let Some(moment_texture) = self.csm_renderer.evsm_maps.as_ref() else {
            return Ok(());
        };

        let technique = crate::lighting::types::ShadowTechnique::from_u32(self.shadow_technique);
        let cascade_count = self.csm_renderer.config.cascade_count;
        let shadow_map_size = self.csm_renderer.config.shadow_map_size;
        let positive_exponent = self.csm_renderer.uniforms.evsm_positive_exp;
        let negative_exponent = self.csm_renderer.uniforms.evsm_negative_exp;
        moment_pass.prepare_textures_checked(
            &self.device,
            self.csm_renderer.shadow_maps.as_ref(),
            moment_texture,
        )?;
        moment_pass.execute_checked(
            &self.queue,
            encoder,
            technique,
            cascade_count,
            shadow_map_size,
            positive_exponent,
            negative_exponent,
        )?;
        if let Some(blur_pass) = self.moment_blur_pass.as_mut() {
            blur_pass.execute(
                &self.device,
                &self.queue,
                encoder,
                moment_texture,
                cascade_count,
                shadow_map_size,
                crate::shadows::DEFAULT_MOMENT_BLUR_RADIUS,
                technique,
                positive_exponent,
            )?;
        }
        log::debug!(
            target: "terrain.shadow",
            "Executed moment generation pass for technique {} with {} cascades",
            self.shadow_technique,
            cascade_count
        );
        Ok(())
    }

    pub(in crate::terrain::renderer) fn ensure_shadow_atlas(
        &mut self,
        settings: &crate::terrain::render_params::ShadowSettingsNative,
    ) -> Result<()> {
        let requires_moments = matches!(
            settings.technique.to_uppercase().as_str(),
            "VSM" | "EVSM" | "MSM"
        );
        crate::shadows::validate_shadow_device_limits(
            &self.device,
            settings.resolution,
            settings.cascades,
        )?;
        let requested_bytes = crate::shadows::shadow_allocation_bytes(
            settings.resolution,
            settings.cascades,
            requires_moments,
        )?;
        if requested_bytes > crate::shadows::MAX_SHADOW_ALLOCATION_BYTES {
            return Err(crate::core::error::RenderError::budget(format!(
                "shadow resources require {:.1} MiB, exceeding the 512 MiB terrain shadow budget",
                requested_bytes as f64 / (1024.0 * 1024.0)
            ))
            .into());
        }
        let atlas_mismatch = self.csm_renderer.allocation_size != settings.resolution
            || self.csm_renderer.allocation_layers != settings.cascades
            || self.csm_renderer.evsm_maps.is_some() != requires_moments;
        if atlas_mismatch {
            let mut replacement_config = self.csm_renderer.config.clone();
            replacement_config.shadow_map_size = settings.resolution;
            replacement_config.cascade_count = settings.cascades;
            replacement_config.enable_evsm = requires_moments;
            let replacement = crate::shadows::CsmRenderer::new(&self.device, replacement_config)?;
            // Moment passes may retain bind groups/views into the current CSM
            // atlas. Drop those dependents before replacing the tracked owner.
            self.moment_pass = None;
            self.moment_blur_pass = None;
            self.csm_renderer = replacement;
        }
        Ok(())
    }

    pub(in crate::terrain::renderer) fn prepare_shadow_setup(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        heightmap_view: &wgpu::TextureView,
        height_curve_view: &wgpu::TextureView,
        heightmap_width: u32,
        heightmap_height: u32,
    ) -> Result<ShadowSetup> {
        let phi_rad = params.cam_phi_deg.to_radians();
        let theta_rad = params.cam_theta_deg.to_radians();
        let eye_x = params.cam_target[0] + params.cam_radius * theta_rad.sin() * phi_rad.cos();
        let eye_y = params.cam_target[1] + params.cam_radius * theta_rad.cos();
        let eye_z = params.cam_target[2] + params.cam_radius * theta_rad.sin() * phi_rad.sin();
        let eye = glam::Vec3::new(eye_x, eye_y, eye_z);
        let target = glam::Vec3::from_array(params.cam_target);
        let up = glam::Vec3::Y;
        let view_matrix = glam::Mat4::look_at_rh(eye, target, up);
        let aspect = params.size_px.0 as f32 / params.size_px.1 as f32;
        let proj_matrix = glam::Mat4::perspective_rh(
            params.fov_y_deg.to_radians(),
            aspect,
            params.clip.0,
            params.clip.1,
        );

        let sun_direction = glam::Vec3::new(
            -decoded.light.direction[0],
            -decoded.light.direction[1],
            -decoded.light.direction[2],
        );
        let terrain_spacing = params.terrain_span.max(1e-3);
        let height_exag = params.z_scale;
        let height_min = decoded.clamp.height_range.0;
        let height_max = decoded.clamp.height_range.1;

        let shadow_settings = &decoded.shadow;
        self.shadow_pcss_radius = shadow_settings.pcss_light_radius.max(0.0);
        use crate::lighting::types::ShadowTechnique;
        let technique_enum = match shadow_settings.technique.to_uppercase().as_str() {
            "HARD" => ShadowTechnique::Hard,
            "PCF" => ShadowTechnique::PCF,
            "PCSS" => ShadowTechnique::PCSS,
            "VSM" => ShadowTechnique::VSM,
            "EVSM" => ShadowTechnique::EVSM,
            "MSM" => ShadowTechnique::MSM,
            _ => {
                log::warn!(
                    target: "terrain.shadow",
                    "Unknown shadow technique '{}', defaulting to PCF",
                    shadow_settings.technique
                );
                ShadowTechnique::PCF
            }
        };
        let requires_moments = matches!(
            technique_enum,
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM
        );
        self.ensure_shadow_atlas(shadow_settings)?;
        let cascade_count = shadow_settings.cascades;
        let shadow_far = params
            .clip
            .1
            .min(shadow_settings.max_distance)
            .max(params.clip.0);

        let mut cascade_splits: Vec<f32> = Vec::with_capacity(cascade_count as usize + 1);
        cascade_splits.push(params.clip.0);
        for split in TERRAIN_DEFAULT_CASCADE_SPLITS
            .iter()
            .take(cascade_count.saturating_sub(1) as usize)
        {
            let clamped = (*split).min(shadow_far);
            if clamped > *cascade_splits.last().unwrap_or(&params.clip.0) {
                cascade_splits.push(clamped);
            }
        }
        while cascade_splits.len() < cascade_count as usize {
            let last = *cascade_splits.last().unwrap_or(&params.clip.0);
            let remaining = cascade_count as usize + 1 - cascade_splits.len();
            let step = (shadow_far - last) / remaining.max(1) as f32;
            cascade_splits.push((last + step).min(shadow_far));
        }
        if cascade_splits.len() == cascade_count as usize {
            cascade_splits.push(shadow_far);
        } else {
            *cascade_splits.last_mut().unwrap() = shadow_far;
        }

        self.csm_renderer.config.cascade_count = cascade_count;
        self.csm_renderer.config.cascade_splits = cascade_splits.clone();
        self.csm_renderer.config.shadow_map_size = shadow_settings.resolution;
        self.csm_renderer.config.max_shadow_distance = shadow_far;
        self.csm_renderer.config.depth_bias = shadow_settings.depth_bias;
        self.csm_renderer.config.slope_bias = shadow_settings.slope_scale_bias;
        self.csm_renderer.config.peter_panning_offset = shadow_settings.normal_bias;
        self.csm_renderer.config.pcf_kernel_size =
            if self.shadow_pcss_radius > 0.0 { 3 } else { 1 };

        self.csm_renderer.uniforms.technique = technique_enum.as_u32();
        self.shadow_technique = technique_enum.as_u32();

        let requires_moment_blur = crate::shadows::requires_moment_blur(technique_enum);
        if requires_moments && self.moment_pass.is_none() {
            self.moment_pass = Some(crate::shadows::MomentGenerationPass::new(&self.device)?);
            log::info!(
                target: "terrain.shadow",
                "Created moment generation pass for technique: {:?}",
                technique_enum
            );
        }
        if requires_moment_blur && self.moment_blur_pass.is_none() {
            self.moment_blur_pass = Some(crate::shadows::ShadowBlurPass::new(&self.device)?);
        }
        if !requires_moment_blur {
            self.moment_blur_pass = None;
        }
        if !requires_moments && self.moment_pass.is_some() {
            self.moment_pass = None;
            log::info!(target: "terrain.shadow", "Removed moment generation pass");
        }
        self.csm_renderer.config.pcf_kernel_size = match technique_enum {
            ShadowTechnique::Hard => 1,
            ShadowTechnique::PCSS => 5,
            _ => 3,
        };
        let (technique_params, technique_reserved) = terrain_shadow_uniform_params(shadow_settings);
        self.csm_renderer.uniforms.technique_params = technique_params;
        self.csm_renderer.uniforms.technique_reserved = technique_reserved;

        log::info!(
            target: "terrain.shadow",
            "Shadow CLI params: enabled={}, technique={} (id={}), cascades={}, resolution={}, max_dist={:.0}, pcss_radius={:.4}",
            shadow_settings.enabled, shadow_settings.technique, technique_enum.as_u32(), shadow_settings.cascades,
            shadow_settings.resolution, shadow_settings.max_distance, self.shadow_pcss_radius
        );
        log::info!(
            target: "terrain.shadow",
            "Shadow bias: depth={:.6}, slope={:.6}, normal={:.6}, softness={:.4}, splits={:?}",
            shadow_settings.depth_bias, shadow_settings.slope_scale_bias,
            shadow_settings.normal_bias, shadow_settings.softness, cascade_splits
        );

        let height_curve = [
            match params.height_curve_mode.as_str() {
                "linear" => 0.0,
                "pow" => 1.0,
                "smoothstep" => 2.0,
                "lut" => 3.0,
                _ => 0.0,
            },
            params.height_curve_strength.clamp(0.0, 1.0),
            params.height_curve_power.max(0.01),
            0.0,
        ];

        let shadow_bind_group = if shadow_settings.enabled {
            let bind_group = self.render_shadow_depth_passes(
                encoder,
                heightmap_view,
                height_curve_view,
                heightmap_width,
                heightmap_height,
                terrain_spacing,
                height_exag,
                height_min,
                height_max,
                view_matrix,
                proj_matrix,
                sun_direction,
                params.clip.0,
                shadow_far,
                height_curve,
            )?;

            self.generate_shadow_moments(encoder)?;

            Some(bind_group)
        } else {
            None
        };

        Ok(ShadowSetup {
            eye,
            view_matrix,
            proj_matrix,
            height_exag,
            height_min,
            shadow_bind_group,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_terrain_pcss_params_preserve_texel_controls() {
        let settings = crate::terrain::render_params::ShadowSettingsNative {
            pcss_blocker_radius: 7.25,
            pcss_filter_radius: 3.5,
            light_size: 2.75,
            ..Default::default()
        };

        assert_eq!(
            terrain_shadow_uniform_params(&settings).0,
            [7.25, 3.5, 0.0005, 2.75]
        );
    }

    #[test]
    fn legacy_world_radius_is_encoded_for_per_cascade_conversion() {
        let settings = crate::terrain::render_params::ShadowSettingsNative {
            light_size: 2.75,
            pcss_light_radius: 0.75,
            ..Default::default()
        };

        let (params, reserved) = terrain_shadow_uniform_params(&settings);
        assert_eq!(params[3], 2.75);
        assert_eq!(reserved, [0.75, 0.0, 0.0, 0.0]);
    }
}
