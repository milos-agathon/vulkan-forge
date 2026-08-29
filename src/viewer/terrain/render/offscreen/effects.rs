use super::SnapshotRenderState;
use crate::core::resource_tracker::TrackedTexture;
use crate::viewer::terrain::dof;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn apply_snapshot_effects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        _depth_texture: &TrackedTexture,
        depth_view: &wgpu::TextureView,
        color_tex: TrackedTexture,
        color_view: wgpu::TextureView,
        state: &SnapshotRenderState,
    ) -> anyhow::Result<TrackedTexture> {
        let mut out_tex = color_tex;
        let mut out_view = color_view;
        let canonical_media = self.canonical_media.is_some();
        let scene_format = self.scene_color_format_for(target_format);

        if let Some(medium) = self.canonical_media.clone() {
            if self.post_process.is_none() {
                self.init_post_process();
            }
            let mut pass = self.canonical_media_pass.take().map_or_else(
                || {
                    crate::terrain::realtime_media::ViewerMediaPass::new(
                        self.device.as_ref(),
                        self.queue.as_ref(),
                        (width, height),
                        medium,
                        self.canonical_media_version,
                    )
                },
                Ok,
            )?;
            let terrain = self
                .terrain
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("canonical media requires loaded terrain"))?;
            let csm = self.csm_renderer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("canonical media shadow resources are unavailable")
            })?;
            pass.prepare_viewer_terrain_trace(
                self.device.as_ref(),
                self.queue.as_ref(),
                self.adapter.as_ref(),
                (width, height),
                &terrain.heightmap,
                terrain.dimensions,
                [state.render_origin_span[0], state.render_origin_span[1]],
                [state.render_origin_span[2], state.render_origin_span[3]],
                terrain.domain.0,
                terrain.height_range(),
                state.shader_z_scale,
                terrain.revision,
            )?;
            let output = pass.encode(
                self.device.as_ref(),
                self.queue.as_ref(),
                self.adapter.as_ref(),
                encoder,
                (width, height),
                state.eye,
                state.view_mat,
                state.proj,
                1.0,
                terrain.cam_radius * 10.0,
                state.sun_dir,
                [terrain.sun_intensity; 3],
                terrain.revision,
                _depth_texture,
                depth_view,
                &out_view,
                csm,
            )?;
            self.canonical_media_diagnostics = Some(pass.diagnostics(self.adapter.as_ref()));
            self.canonical_media_pass = Some(pass);
            out_view = output;
        }
        let needs_volumetrics =
            self.canonical_media.is_none() && self.pbr_config.volumetrics.is_effectively_enabled();
        if needs_volumetrics {
            if self.volumetrics_pass.is_none() {
                self.init_volumetrics_pass();
            }

            let vol_target = match self.create_snapshot_color_target(
                "terrain_viewer.snapshot_vol_output",
                target_format,
                width,
                height,
            ) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("[terrain] failed to allocate volumetrics target: {e}");
                    crate::core::degradation::record_degradation(
                        "allocation_fallback",
                        "viewer.volumetrics",
                        "volumetric lighting skipped; snapshot rendered without volumetrics",
                    );
                    None
                }
            };
            if let (Some((vol_output_tex, vol_output_view)), Some(ref mut vol_pass)) =
                (vol_target, self.volumetrics_pass.as_mut())
            {
                let terrain = self.terrain.as_ref().unwrap();
                let cam_radius = terrain.cam_radius;
                let terrain_sun_intensity = terrain.sun_intensity;

                if let Err(e) = vol_pass.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    depth_view,
                    &terrain.heightmap_view,
                    &terrain.heightmap,
                    terrain.dimensions,
                    terrain.revision,
                    &vol_output_view,
                    width,
                    height,
                    state.view_proj.inverse().to_cols_array_2d(),
                    [state.eye.x, state.eye.y, state.eye.z],
                    1.0,
                    cam_radius * 10.0,
                    [state.sun_dir.x, state.sun_dir.y, state.sun_dir.z],
                    terrain_sun_intensity,
                    [
                        state.render_origin_span[2]
                            .abs()
                            .max(state.render_origin_span[3].abs()),
                        terrain.domain.0,
                        state.shader_z_scale,
                        state.h_range,
                    ],
                    state.render_origin_span,
                    &self.pbr_config.volumetrics,
                ) {
                    eprintln!("[terrain] volumetrics apply failed: {e}");
                }

                out_tex = vol_output_tex;
                out_view = vol_output_view;
            }
        }

        let needs_dof = self.pbr_config.dof.enabled;
        if needs_dof {
            if self.dof_pass.is_none() {
                self.init_dof_pass();
            }

            let dof_target = match self.create_snapshot_color_target(
                "terrain_viewer.snapshot_dof_output",
                scene_format,
                width,
                height,
            ) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("[terrain] failed to allocate DoF target: {e}");
                    crate::core::degradation::record_degradation(
                        "allocation_fallback",
                        "viewer.dof",
                        "depth-of-field skipped; snapshot rendered fully in focus",
                    );
                    None
                }
            };
            if let (Some((dof_output_tex, dof_output_view)), Some(ref mut dof)) =
                (dof_target, self.dof_pass.as_mut())
            {
                let _ = dof.get_input_view(width, height, scene_format);
                let cam_radius = self
                    .terrain
                    .as_ref()
                    .map(|t| t.cam_radius)
                    .unwrap_or(2000.0);
                let dof_cfg = dof::DofConfig {
                    focus_distance: self.pbr_config.dof.focus_distance,
                    f_stop: self.pbr_config.dof.f_stop,
                    focal_length: self.pbr_config.dof.focal_length,
                    quality: self.pbr_config.dof.quality,
                    max_blur_radius: self.pbr_config.dof.max_blur_radius,
                    blur_strength: self.pbr_config.dof.blur_strength,
                    tilt_pitch: self.pbr_config.dof.tilt_pitch,
                    tilt_yaw: self.pbr_config.dof.tilt_yaw,
                };

                if let Err(e) = dof.apply(
                    encoder,
                    &self.queue,
                    &out_view,
                    depth_view,
                    &dof_output_view,
                    width,
                    height,
                    scene_format,
                    &dof_cfg,
                    1.0,
                    cam_radius * 10.0,
                ) {
                    eprintln!("[terrain] DoF apply failed: {e}");
                }

                out_tex = dof_output_tex;
                out_view = dof_output_view;
            }
        }

        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);
        if canonical_media {
            let (output_tex, output_view) = self.create_snapshot_color_target(
                "terrain_viewer.snapshot_nephele_resolve",
                target_format,
                width,
                height,
            )?;
            let lens = &self.pbr_config.lens_effects;
            let (distortion, chromatic_aberration, vignette_strength) = if lens.enabled {
                (
                    lens.distortion,
                    lens.chromatic_aberration,
                    lens.vignette_strength,
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            self.post_process
                .as_mut()
                .expect("canonical media initializes post-process pass")
                .apply_from_linear_hdr(
                    encoder,
                    &self.queue,
                    &out_view,
                    &output_view,
                    width,
                    height,
                    distortion,
                    chromatic_aberration,
                    vignette_strength,
                    lens.vignette_radius,
                    lens.vignette_softness,
                );
            return Ok(output_tex);
        }
        if needs_post_process {
            if self.post_process.is_none() {
                self.init_post_process();
            }

            let lens_target = match self.create_snapshot_color_target(
                "terrain_viewer.snapshot_lens_output",
                target_format,
                width,
                height,
            ) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("[terrain] failed to allocate lens target: {e}");
                    crate::core::degradation::record_degradation(
                        "allocation_fallback",
                        "viewer.lens",
                        "lens post-process skipped; snapshot missing distortion/vignette",
                    );
                    None
                }
            };
            if let (Some((lens_output_tex, lens_output_view)), Some(ref mut pp)) =
                (lens_target, self.post_process.as_mut())
            {
                let lens = &self.pbr_config.lens_effects;

                pp.apply_from_input(
                    encoder,
                    &self.queue,
                    &out_view,
                    &lens_output_view,
                    width,
                    height,
                    lens.distortion,
                    lens.chromatic_aberration,
                    lens.vignette_strength,
                    lens.vignette_radius,
                    lens.vignette_softness,
                );
                return Ok(lens_output_tex);
            }
        }

        Ok(out_tex)
    }
}
