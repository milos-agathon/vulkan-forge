use super::{SnapshotRenderState, TerrainPbrUniforms, TerrainUniforms};
use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn prepare_snapshot_resources(&mut self, width: u32, height: u32) {
        if let Err(e) = self.ensure_fallback_texture() {
            eprintln!("[terrain] fallback texture allocation failed: {e}");
        }
        if self.depth_view.is_none() {
            if let Err(e) = self.ensure_depth(width, height) {
                eprintln!("[terrain] depth target allocation failed: {e}");
            }
        }
    }

    pub(super) fn create_snapshot_color_target(
        &self,
        label: &'static str,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> RenderResult<(TrackedTexture, wgpu::TextureView)> {
        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((texture, view))
    }

    pub(super) fn create_snapshot_depth_target(
        &self,
        width: u32,
        height: u32,
    ) -> RenderResult<(TrackedTexture, wgpu::TextureView)> {
        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain_viewer.snapshot_depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((texture, view))
    }

    pub(super) fn build_snapshot_render_state(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) -> SnapshotRenderState {
        if self.pbr_config.enabled && self.pbr_pipeline.is_none() {
            if let Err(e) = self.init_pbr_pipeline(target_format) {
                eprintln!("[snapshot] Failed to initialize PBR pipeline: {}", e);
            }
        }

        if self.pbr_config.enabled
            && (self.pbr_config.height_ao.enabled || self.pbr_config.sun_visibility.enabled)
        {
            if let Err(e) = self.init_heightfield_compute_pipelines() {
                eprintln!(
                    "[snapshot] Failed to initialize heightfield compute pipelines: {}",
                    e
                );
            }
        }

        let use_pbr = self.pbr_config.enabled && self.pbr_pipeline.is_some();
        let (terrain_z_scale, h_range, domain, sun_azimuth_deg, sun_elevation_deg) = {
            let terrain = self.terrain.as_ref().unwrap();
            (
                terrain.z_scale,
                terrain.height_range(),
                terrain.domain,
                terrain.sun_azimuth_deg,
                terrain.sun_elevation_deg,
            )
        };
        let shader_z_scale = terrain_z_scale;
        let (origin_world, span_world) = {
            let terrain = self.terrain.as_ref().unwrap();
            (terrain.world_origin_xz, terrain.world_span_xz)
        };
        let origin =
            frame
                .anchor
                .to_render_vec3(glam::DVec3::new(origin_world.x, 0.0, origin_world.y));
        let span =
            frame
                .anchor
                .to_render_direction(glam::DVec3::new(span_world.x, 0.0, span_world.y));
        let render_origin_span = [origin.x, origin.z, span.x, span.z];
        let terrain_width = span.x;
        let proj_base = frame.projection(width, height);
        let proj = if self.taa_jitter.enabled {
            crate::core::jitter::apply_jitter(
                proj_base,
                self.taa_jitter.offset.0,
                self.taa_jitter.offset.1,
                width,
                height,
            )
        } else {
            proj_base
        };
        let view_mat = frame.view();
        let view_proj = proj * view_mat;
        let eye = frame.render_eye();

        let sun_az = sun_azimuth_deg.to_radians();
        let sun_el = sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el.cos() * sun_az.sin(),
            sun_el.sin(),
            sun_el.cos() * sun_az.cos(),
        )
        .normalize();

        if use_pbr && self.shadow_pipeline.is_none() {
            match self.init_shadow_depth_pipeline() {
                Ok(()) => self.update_shadow_bind_groups(),
                Err(error) => eprintln!("[snapshot] Failed to initialize shadow pipeline: {error}"),
            }
        }
        if use_pbr && self.shadow_pipeline.is_some() {
            self.render_shadow_passes(encoder, view_mat, proj, -sun_dir, render_origin_span);
        } else if use_pbr {
            eprintln!(
                "[snapshot] Skipping shadow passes: pipeline={}",
                self.shadow_pipeline.is_some()
            );
        }

        let terrain = self.terrain.as_ref().unwrap();
        let uniforms = TerrainUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            terrain_params: [domain.0, h_range, terrain_width, shader_z_scale],
            lighting: [
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
            ],
            background: [
                terrain.background_color[0],
                terrain.background_color[1],
                terrain.background_color[2],
                0.0,
            ],
            water_color: [
                terrain.water_color[0],
                terrain.water_color[1],
                terrain.water_color[2],
                0.0,
            ],
            render_origin_xz: [render_origin_span[0], render_origin_span[1]],
            render_span_xz: [render_origin_span[2], render_origin_span[3]],
        };
        self.queue.write_buffer(
            &terrain.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
        let vo_lighting = [
            terrain.sun_intensity,
            terrain.ambient,
            terrain.shadow_intensity,
            terrain_width,
        ];

        let pbr_uniforms_data = if use_pbr {
            Some((
                domain,
                terrain_z_scale,
                terrain.sun_intensity,
                terrain.ambient,
                terrain.shadow_intensity,
                terrain.water_level,
                terrain.background_color,
                terrain.water_color,
            ))
        } else {
            None
        };
        let _ = terrain;

        if let Some((
            domain,
            z_scale,
            sun_intensity,
            ambient,
            shadow_intensity,
            water_level,
            background_color,
            water_color,
        )) = pbr_uniforms_data
        {
            if let Err(e) = self.ensure_terrain_ibl_resources() {
                eprintln!("[terrain] IBL resource preparation failed: {e}");
            }
            let pbr_uniforms = TerrainPbrUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                terrain_params: [domain.0, domain.1 - domain.0, terrain_width, z_scale],
                lighting: [sun_intensity, ambient, shadow_intensity, water_level],
                background: [
                    background_color[0],
                    background_color[1],
                    background_color[2],
                    0.0,
                ],
                water_color: [water_color[0], water_color[1], water_color[2], 0.0],
                pbr_params: [
                    self.pbr_config.exposure,
                    self.pbr_config.normal_strength,
                    self.pbr_config.ibl_intensity,
                    if self.pbr_config.overlay.preserve_colors {
                        1.0
                    } else {
                        0.0
                    },
                ],
                ibl_params: self.terrain_ibl_uniform_params(),
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                lens_params: [
                    self.pbr_config.lens_effects.vignette_strength,
                    self.pbr_config.lens_effects.vignette_radius,
                    self.pbr_config.lens_effects.vignette_softness,
                    0.0,
                ],
                screen_dims: [width as f32, height as f32, 0.0, 0.0],
                overlay_params: [
                    if self.pbr_config.overlay.enabled {
                        1.0
                    } else {
                        0.0
                    },
                    self.pbr_config.overlay.global_opacity,
                    0.0,
                    if self.pbr_config.overlay.solid {
                        1.0
                    } else {
                        0.0
                    },
                ],
                render_origin_xz: [render_origin_span[0], render_origin_span[1]],
                render_span_xz: [render_origin_span[2], render_origin_span[3]],
            };
            eprintln!(
                "[DEBUG render_to_texture] overlay_params: enabled={}, opacity={}, blend={}, solid={}",
                pbr_uniforms.overlay_params[0],
                pbr_uniforms.overlay_params[1],
                pbr_uniforms.overlay_params[2],
                pbr_uniforms.overlay_params[3]
            );
            if let Err(e) = self.prepare_pbr_bind_group_internal(&pbr_uniforms) {
                eprintln!("[terrain] PBR bind group preparation failed: {e}");
            }
        }

        self.dispatch_heightfield_compute(encoder, [span.x, span.z], sun_dir);

        SnapshotRenderState {
            use_pbr,
            view_mat,
            proj,
            view_proj,
            sun_dir,
            eye,
            render_origin_span,
            h_range,
            shader_z_scale,
            vo_view_proj: view_proj.to_cols_array_2d(),
            vo_sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z],
            vo_lighting,
        }
    }
}
