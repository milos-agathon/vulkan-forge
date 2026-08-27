use super::{ScreenRenderFlags, ScreenRenderState};
use crate::viewer::terrain::dof;
use crate::viewer::terrain::post_process::PostProcessPass;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    fn execute_canonical_screen_media(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        state: &ScreenRenderState,
    ) -> anyhow::Result<()> {
        let medium = self
            .canonical_media
            .clone()
            .ok_or_else(|| anyhow::anyhow!("canonical media attachment disappeared"))?;
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
        let input_view = self
            .post_process
            .as_mut()
            .and_then(|post| post.intermediate_view.take())
            .ok_or_else(|| anyhow::anyhow!("canonical media input target is unavailable"))?;
        let result = (|| {
            let terrain = self
                .terrain
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("canonical media requires loaded terrain"))?;
            let depth_texture = self
                .depth_texture
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("canonical media depth texture is unavailable"))?;
            let depth_view = self
                .depth_view
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("canonical media depth view is unavailable"))?;
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
                state.cam_radius * 10.0,
                state.sun_dir,
                [terrain.sun_intensity; 3],
                terrain.revision,
                depth_texture,
                depth_view,
                &input_view,
                csm,
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
                .expect("canonical media prepares post-process pass")
                .apply_from_linear_hdr(
                    encoder,
                    &self.queue,
                    &output,
                    view,
                    width,
                    height,
                    distortion,
                    chromatic_aberration,
                    vignette_strength,
                    lens.vignette_radius,
                    lens.vignette_softness,
                );
            self.canonical_media_diagnostics = Some(pass.diagnostics(self.adapter.as_ref()));
            Ok(())
        })();
        self.canonical_media_pass = Some(pass);
        self.post_process
            .as_mut()
            .expect("canonical media prepares post-process pass")
            .intermediate_view = Some(input_view);
        result
    }

    pub(super) fn apply_screen_effects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        flags: &ScreenRenderFlags,
        state: &ScreenRenderState,
    ) -> anyhow::Result<()> {
        if flags.needs_canonical_media {
            return self.execute_canonical_screen_media(encoder, view, width, height, state);
        }
        if flags.needs_denoise {
            let (iterations, sigma_color) = {
                let config = &self.pbr_config.denoise;
                (config.iterations, config.sigma_color)
            };

            let ViewerTerrainScene {
                denoise_pass,
                post_process,
                dof_pass,
                depth_view,
                queue,
                device,
                surface_format,
                ..
            } = self;

            if let Some(denoise) = denoise_pass.as_mut() {
                let depth_view = depth_view.as_ref().unwrap();
                if let Err(e) = denoise.apply(encoder, depth_view, iterations, sigma_color) {
                    eprintln!("[terrain] denoise apply failed: {e}");
                    return Ok(());
                }

                let denoise_result = denoise
                    .get_last_result_view(iterations)
                    .unwrap_or(denoise.view_a.as_ref().unwrap());

                if post_process.is_none() {
                    match PostProcessPass::new(device.clone(), *surface_format) {
                        Ok(pass) => *post_process = Some(pass),
                        Err(e) => {
                            eprintln!(
                                "[terrain] failed to initialize post-process pass for denoise: {e}"
                            );
                            return Ok(());
                        }
                    }
                }

                let post_process = post_process.as_mut().unwrap();
                let mut intermediate_view = None;
                let next_target = if flags.needs_volumetrics {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else if flags.needs_dof {
                    dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else if flags.needs_post_process {
                    intermediate_view = post_process.intermediate_view.take();
                    intermediate_view.as_ref().unwrap()
                } else {
                    view
                };

                post_process.apply_from_input(
                    encoder,
                    queue,
                    denoise_result,
                    next_target,
                    width,
                    height,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                );

                if let Some(view) = intermediate_view {
                    post_process.intermediate_view = Some(view);
                }
            }
        }

        if flags.needs_volumetrics {
            if let Some(ref mut vol_pass) = self.volumetrics_pass {
                let terrain = self.terrain.as_ref().unwrap();
                let depth_view = self.depth_view.as_ref().unwrap();
                let taa_input = self
                    .taa_renderer
                    .as_ref()
                    .filter(|_| flags.needs_taa)
                    .map(crate::core::taa::TaaRenderer::history_view);
                let post_input = self
                    .post_process
                    .as_ref()
                    .and_then(|pp| pp.intermediate_view.as_ref());
                let color_input = taa_input.or(post_input).unwrap();
                let vol_output = if flags.needs_dof || flags.needs_post_process {
                    self.dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
                } else {
                    view
                };

                if let Err(e) = vol_pass.apply(
                    encoder,
                    &self.queue,
                    color_input,
                    depth_view,
                    &terrain.heightmap_view,
                    &terrain.heightmap,
                    terrain.dimensions,
                    terrain.revision,
                    vol_output,
                    width,
                    height,
                    state.view_proj.inverse().to_cols_array_2d(),
                    [state.eye.x, state.eye.y, state.eye.z],
                    1.0,
                    state.cam_radius * 10.0,
                    [state.sun_dir.x, state.sun_dir.y, state.sun_dir.z],
                    terrain.sun_intensity,
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
            }
        }

        if flags.needs_dof {
            let dof_output = if flags.needs_post_process {
                self.post_process
                    .as_ref()
                    .unwrap()
                    .intermediate_view
                    .as_ref()
                    .unwrap()
            } else {
                view
            };

            if let Some(ref mut dof) = self.dof_pass {
                let depth_view = self.depth_view.as_ref().unwrap();
                let taa_input = self
                    .taa_renderer
                    .as_ref()
                    .filter(|_| flags.needs_taa && !flags.needs_volumetrics)
                    .map(crate::core::taa::TaaRenderer::history_view);
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
                let result = if let Some(input_view) = taa_input {
                    dof.apply(
                        encoder,
                        &self.queue,
                        input_view,
                        depth_view,
                        dof_output,
                        width,
                        height,
                        self.surface_format,
                        &dof_cfg,
                        1.0,
                        state.cam_radius * 10.0,
                    )
                } else {
                    dof.apply_from_input(
                        encoder,
                        &self.queue,
                        depth_view,
                        dof_output,
                        width,
                        height,
                        self.surface_format,
                        &dof_cfg,
                        1.0,
                        state.cam_radius * 10.0,
                    )
                };
                if let Err(e) = result {
                    eprintln!("[terrain] DoF apply failed: {e}");
                }
            }
        }

        if flags.needs_post_process
            || (flags.needs_taa && !flags.needs_volumetrics && !flags.needs_dof)
        {
            let external_input = if !flags.needs_dof && flags.needs_volumetrics {
                self.dof_pass
                    .as_ref()
                    .and_then(|dof| dof.input_view.as_ref())
            } else if flags.needs_taa && !flags.needs_dof {
                self.taa_renderer
                    .as_ref()
                    .map(crate::core::taa::TaaRenderer::history_view)
            } else {
                None
            };

            if let Some(ref mut pp) = self.post_process {
                let lens = &self.pbr_config.lens_effects;
                if let Some(input_view) = external_input {
                    pp.apply_from_input(
                        encoder,
                        &self.queue,
                        input_view,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                } else {
                    pp.apply(
                        encoder,
                        &self.queue,
                        view,
                        width,
                        height,
                        lens.distortion,
                        lens.chromatic_aberration,
                        lens.vignette_strength,
                        lens.vignette_radius,
                        lens.vignette_softness,
                    );
                }
            }
        }
        Ok(())
    }
}
