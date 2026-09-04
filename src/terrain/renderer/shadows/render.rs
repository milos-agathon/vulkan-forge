use super::*;
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_buffer_init};

impl TerrainScene {
    pub(super) fn render_shadow_depth_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        heightmap_view: &wgpu::TextureView,
        height_curve_view: &wgpu::TextureView,
        heightmap_width: u32,
        heightmap_height: u32,
        terrain_spacing: f32,
        height_exag: f32,
        height_min: f32,
        height_max: f32,
        _view_matrix: glam::Mat4,
        _proj_matrix: glam::Mat4,
        sun_direction: glam::Vec3,
        near_plane: f32,
        far_plane: f32,
        height_curve: [f32; 4],
    ) -> Result<wgpu::BindGroup> {
        let light_dir = sun_direction.normalize();
        let light_up = if light_dir.z.abs() > 0.99 {
            glam::Vec3::Y
        } else {
            glam::Vec3::Z
        };

        let half_spacing = terrain_spacing * 0.5;
        let shadow_z_min = 0.0;
        let shadow_z_max = height_exag;

        let terrain_min = glam::Vec3::new(-half_spacing, -half_spacing, shadow_z_min);
        let terrain_max = glam::Vec3::new(half_spacing, half_spacing, shadow_z_max);
        let terrain_center = (terrain_min + terrain_max) * 0.5;

        let terrain_diagonal = (terrain_max - terrain_min).length();
        let light_camera_distance = terrain_diagonal * 2.0;
        let light_camera_pos = terrain_center - light_dir * light_camera_distance;
        let light_view = glam::Mat4::look_to_rh(light_camera_pos, light_dir, light_up);

        let corners = [
            glam::Vec3::new(terrain_min.x, terrain_min.y, terrain_min.z),
            glam::Vec3::new(terrain_max.x, terrain_min.y, terrain_min.z),
            glam::Vec3::new(terrain_min.x, terrain_max.y, terrain_min.z),
            glam::Vec3::new(terrain_max.x, terrain_max.y, terrain_min.z),
            glam::Vec3::new(terrain_min.x, terrain_min.y, terrain_max.z),
            glam::Vec3::new(terrain_max.x, terrain_min.y, terrain_max.z),
            glam::Vec3::new(terrain_min.x, terrain_max.y, terrain_max.z),
            glam::Vec3::new(terrain_max.x, terrain_max.y, terrain_max.z),
        ];

        let mut light_min = glam::Vec3::splat(f32::MAX);
        let mut light_max = glam::Vec3::splat(f32::MIN);
        for corner in &corners {
            let light_pos = (light_view * corner.extend(1.0)).truncate();
            light_min = light_min.min(light_pos);
            light_max = light_max.max(light_pos);
        }

        let padding = terrain_spacing * 0.3;
        light_min -= glam::Vec3::splat(padding);
        light_max += glam::Vec3::splat(padding);

        let z_padding = terrain_spacing * 0.1;
        let proj_near = -light_max.z - z_padding;
        let proj_far = -light_min.z + z_padding;
        let light_proj = glam::Mat4::orthographic_rh(
            light_min.x,
            light_max.x,
            light_min.y,
            light_max.y,
            proj_near,
            proj_far,
        );

        let mut cascade_count = self
            .csm_renderer
            .config
            .cascade_count
            .max(1)
            .min(self.csm_renderer.shadow_map_views.len() as u32);
        let splits = if self.csm_renderer.config.cascade_splits.len() >= cascade_count as usize + 1
        {
            self.csm_renderer.config.cascade_splits.clone()
        } else {
            let mut fallback = Vec::with_capacity(cascade_count as usize + 1);
            fallback.push(near_plane);
            let step = (far_plane - near_plane) / cascade_count as f32;
            for i in 1..=cascade_count {
                fallback.push((near_plane + step * i as f32).min(far_plane));
            }
            fallback
        };
        if splits.len() < 2 {
            cascade_count = 1;
        }

        let light_view_proj = light_proj * light_view;
        let texel_size =
            (light_max.x - light_min.x) / self.csm_renderer.config.shadow_map_size as f32;

        self.csm_renderer.uniforms.light_direction = [light_dir.x, light_dir.y, light_dir.z, 0.0];
        self.csm_renderer.uniforms.light_view = light_view.to_cols_array();
        self.csm_renderer.uniforms.cascade_count = cascade_count;
        self.csm_renderer.uniforms.pcf_kernel_size = self.csm_renderer.config.pcf_kernel_size;
        self.csm_renderer.uniforms.depth_bias = self.csm_renderer.config.depth_bias;
        self.csm_renderer.uniforms.slope_bias = self.csm_renderer.config.slope_bias;
        self.csm_renderer.uniforms.shadow_map_size =
            self.csm_renderer.config.shadow_map_size as f32;
        self.csm_renderer.uniforms.peter_panning_offset =
            self.csm_renderer.config.peter_panning_offset;
        self.csm_renderer.uniforms.debug_mode = self.csm_renderer.config.debug_mode;

        for cascade_idx in 0..cascade_count as usize {
            let near_d = splits.get(cascade_idx).copied().unwrap_or(near_plane);
            let far_d = splits.get(cascade_idx + 1).copied().unwrap_or(far_plane);
            self.csm_renderer.uniforms.cascades[cascade_idx].light_projection =
                light_proj.to_cols_array();
            self.csm_renderer.uniforms.cascades[cascade_idx].light_view_proj =
                light_view_proj.to_cols_array_2d();
            self.csm_renderer.uniforms.cascades[cascade_idx].near_distance = near_d;
            self.csm_renderer.uniforms.cascades[cascade_idx].far_distance = far_d;
            self.csm_renderer.uniforms.cascades[cascade_idx].texel_size = texel_size;
            self.csm_renderer.uniforms.cascades[cascade_idx]._padding = 0.0;
        }
        self.csm_renderer.upload_uniforms(&self.queue);

        const SHADOW_GRID_RES: u32 = 1024;
        let vertices_per_cascade = (SHADOW_GRID_RES - 1) * (SHADOW_GRID_RES - 1) * 6;

        #[cfg(feature = "enable-gpu-instancing")]
        let scatter_shadow = if self.scatter_batches.is_empty() {
            None
        } else {
            let terrain_width = heightmap_width.max(heightmap_height).max(1) as f32;
            let scale_xy = terrain_spacing.max(1e-3) / terrain_width;
            let centered_z_offset = -0.5 * (height_max - height_min).max(0.0) * height_exag;
            let render_from_contract = glam::Mat4::from_cols_array(&[
                scale_xy,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                scale_xy,
                0.0,
                0.0,
                -terrain_spacing * 0.5,
                -terrain_spacing * 0.5,
                centered_z_offset,
                1.0,
            ]);
            let eye_contract = render_from_contract
                .inverse()
                .transform_point3(light_camera_pos);
            let identity_packed =
                crate::terrain::scatter::pack_hlod_identity_instance(render_from_contract);
            let hlod_inst_bytes = (std::mem::size_of::<f32>() * 16) as u64;
            let hlod_instbuf = tracked_create_buffer(
                self.device.as_ref(),
                &wgpu::BufferDescriptor {
                    label: Some("terrain.shadow.scatter.hlod.instance_buffer"),
                    size: hlod_inst_bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?;
            self.queue
                .write_buffer(&hlod_instbuf, 0, bytemuck::cast_slice(&identity_packed));
            self.scatter_renderer.reset_shadow_draw_batch_uniforms();
            Some((render_from_contract, eye_contract, scale_xy, hlod_instbuf))
        };

        for cascade_idx in 0..cascade_count {
            let cascade = &self.csm_renderer.uniforms.cascades[cascade_idx as usize];
            let stored_light_view_proj = cascade.light_view_proj;
            let shadow_uniforms = ShadowPassUniforms {
                light_view_proj: stored_light_view_proj,
                terrain_params: [terrain_spacing, height_exag, height_min, height_max],
                grid_params: [SHADOW_GRID_RES as f32, 0.0, 0.0, 0.0],
                height_curve,
            };

            let shadow_uniform_buffer = tracked_create_buffer_init(
                self.device.as_ref(),
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("terrain.shadow.cascade_{}.uniforms", cascade_idx)),
                    contents: bytemuck::bytes_of(&shadow_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            )?;

            let shadow_depth_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!(
                        "terrain.shadow.cascade_{}.bind_group",
                        cascade_idx
                    )),
                    layout: &self.shadow_depth_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: shadow_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(heightmap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(height_curve_view),
                        },
                    ],
                });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("terrain.shadow.cascade_{}.pass", cascade_idx)),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.csm_renderer.shadow_map_views[cascade_idx as usize],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            crate::core::shader_registry::record_shader_use("terrain.shadow_depth.shader");
            pass.set_pipeline(&self.shadow_depth_pipeline);
            pass.set_bind_group(0, &shadow_depth_bind_group, &[]);
            pass.draw(0..vertices_per_cascade, 0..1);

            #[cfg(feature = "enable-gpu-instancing")]
            if let Some((render_from_contract, eye_contract, instance_scale, hlod_instbuf)) =
                scatter_shadow.as_ref()
            {
                let device = self.device.as_ref();
                let queue = self.queue.as_ref();
                let renderer = &self.scatter_renderer;
                for batch in &mut self.scatter_batches {
                    let (_batch_stats, draws) = batch.prepare_draws(
                        device,
                        queue,
                        *eye_contract,
                        *render_from_contract,
                        *instance_scale,
                    )?;
                    for draw in draws {
                        let Some(instbuf) = batch.level_instbuf(draw.level_index) else {
                            continue;
                        };
                        renderer.draw_shadow_batch_params(
                            &mut pass,
                            queue,
                            light_view,
                            light_proj,
                            batch.level_vbuf(draw.level_index),
                            batch.level_ibuf(draw.level_index),
                            instbuf,
                            batch.level_index_count(draw.level_index),
                            draw.instance_count,
                        );
                    }

                    let active_clusters = batch.hlod_active_clusters(*eye_contract);
                    for cluster_idx in active_clusters {
                        if let (Some(vbuf), Some(ibuf)) = (
                            batch.hlod_cluster_vbuf(cluster_idx),
                            batch.hlod_cluster_ibuf(cluster_idx),
                        ) {
                            renderer.draw_shadow_batch_params(
                                &mut pass,
                                queue,
                                light_view,
                                light_proj,
                                vbuf,
                                ibuf,
                                hlod_instbuf,
                                batch.hlod_cluster_index_count(cluster_idx),
                                1,
                            );
                        }
                    }
                }
            }
        }

        self.create_shadow_bind_group()
    }
}
