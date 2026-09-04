use super::{ScreenRenderFlags, ScreenRenderState, TerrainPbrUniforms, TerrainUniforms};
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn build_screen_render_state(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        flags: &ScreenRenderFlags,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) -> ScreenRenderState {
        let (terrain_z_scale, h_range, domain, sun_azimuth_deg, sun_elevation_deg, target) = {
            let terrain = self.terrain.as_ref().unwrap();
            (
                terrain.z_scale,
                terrain.height_range(),
                terrain.domain,
                terrain.sun_azimuth_deg,
                terrain.sun_elevation_deg,
                terrain.camera_target(),
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

        if flags.use_pbr && self.shadow_pipeline.is_none() {
            match self.init_shadow_depth_pipeline() {
                Ok(()) => self.update_shadow_bind_groups(),
                Err(error) => eprintln!("[render] Failed to initialize shadow pipeline: {error}"),
            }
        }
        if flags.use_pbr && self.shadow_pipeline.is_some() {
            self.render_shadow_passes(encoder, view_mat, proj, -sun_dir, render_origin_span);
        } else if flags.use_pbr {
            eprintln!(
                "[render] Skipping shadow passes: pipeline={}",
                self.shadow_pipeline.is_some()
            );
        }

        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            println!(
                "[render] terrain_params: min_h={:.1}, h_range={:.1}, width={:.1}, z_scale={:.2}",
                domain.0, h_range, terrain_width, shader_z_scale
            );
            let max_y = h_range * terrain_z_scale;
            println!("[render] Expected Y range: 0 to {:.1}", max_y);
            println!(
                "[render] Camera target: ({:.1}, {:.1}, {:.1})",
                target.x, target.y, target.z
            );
        });

        let terrain = self.terrain.as_ref().unwrap();
        let uniforms = TerrainUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            terrain_params: [
                terrain.domain.0,
                terrain.domain.1 - terrain.domain.0,
                terrain_width,
                shader_z_scale,
            ],
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

        let cam_radius = terrain.cam_radius;
        let vo_lighting = [
            terrain.sun_intensity,
            terrain.ambient,
            terrain.shadow_intensity,
            terrain_width,
        ];
        let pbr_uniforms_data = if flags.use_pbr {
            Some((
                terrain.domain,
                terrain.z_scale,
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
            if let Err(e) = self.prepare_pbr_bind_group_internal(&pbr_uniforms) {
                eprintln!("[terrain] PBR bind group preparation failed: {e}");
            }
        }

        self.dispatch_heightfield_compute(encoder, [span.x, span.z], sun_dir);

        ScreenRenderState {
            view_mat,
            proj,
            view_proj,
            view_proj_array: view_proj.to_cols_array_2d(),
            sun_dir,
            eye,
            render_origin_span,
            h_range,
            shader_z_scale,
            cam_radius,
            vo_sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z],
            vo_lighting,
        }
    }
}
