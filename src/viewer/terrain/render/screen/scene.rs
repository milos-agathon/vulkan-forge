use super::{ScreenRenderFlags, ScreenRenderState};
#[cfg(feature = "enable-gpu-instancing")]
use crate::viewer::terrain::scene::scatter::render_scatter_batches;
use crate::viewer::terrain::vector_overlay;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn render_screen_scene_path(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        selected_feature_id: u32,
        flags: &ScreenRenderFlags,
        state: &ScreenRenderState,
        has_vector_overlays: bool,
    ) {
        let render_target: &wgpu::TextureView = if flags.needs_denoise {
            self.denoise_pass.as_ref().unwrap().view_a.as_ref().unwrap()
        } else if flags.needs_taa || flags.needs_volumetrics || flags.needs_canonical_media {
            self.post_process
                .as_ref()
                .unwrap()
                .intermediate_view
                .as_ref()
                .unwrap()
        } else if flags.needs_dof {
            self.dof_pass.as_ref().unwrap().input_view.as_ref().unwrap()
        } else if flags.needs_post_process {
            self.post_process
                .as_ref()
                .unwrap()
                .intermediate_view
                .as_ref()
                .unwrap()
        } else {
            view
        };
        let depth_view = self.depth_view.as_ref().unwrap();
        #[cfg(feature = "enable-gpu-instancing")]
        let mut scatter_batches = std::mem::take(&mut self.scatter_batches);

        {
            let terrain = self.terrain.as_ref().unwrap();
            let bg = terrain.background_color;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg[0] as f64,
                            g: bg[1] as f64,
                            b: bg[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            if flags.use_pbr {
                if let Some(ref pbr_bind_group) = self.pbr_bind_group {
                    pass.set_pipeline(self.pbr_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, pbr_bind_group, &[]);
                } else {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &terrain.bind_group, &[]);
                }
            } else {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &terrain.bind_group, &[]);
            }

            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);
        }

        #[cfg(feature = "enable-gpu-instancing")]
        let scatter_result = {
            let terrain = self.terrain.as_ref().unwrap();
            render_scatter_batches(
                encoder,
                render_target,
                depth_view,
                &mut scatter_batches,
                state.view_mat,
                state.proj,
                state.eye,
                &terrain.heightmap_view,
                state.render_origin_span,
                terrain.dimensions,
                terrain.domain.0,
                terrain.z_scale,
                [-state.sun_dir.x, -state.sun_dir.y, -state.sun_dir.z],
                state.vo_lighting[0],
                self.scatter_elapsed_time,
                self.device.as_ref(),
                self.queue.as_ref(),
                &mut self.scatter_renderer,
                &self.scatter_hlod_instance_buffer,
            )
        };

        if has_vector_overlays && !self.oit_enabled {
            if let Some(ref stack) = self.vector_overlay_stack {
                if stack.pipelines_ready() && stack.bind_group.is_some() {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("terrain_viewer.overlay_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: render_target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    let layer_count = stack.visible_layer_count();
                    let highlight_color = [1.0, 0.8, 0.0, 0.5];
                    for i in 0..layer_count {
                        stack.render_layer_with_highlight(
                            &mut pass,
                            vector_overlay::RenderLayerParams {
                                layer_index: i,
                                view_proj: state.view_proj_array,
                                sun_dir: state.vo_sun_dir,
                                lighting: state.vo_lighting,
                                render_origin_span: state.render_origin_span,
                                selected_feature_id,
                                highlight_color,
                            },
                        );
                    }
                }
            }
        }

        if has_vector_overlays && self.oit_enabled {
            if let (Some(color_view), Some(reveal_view)) = (
                self.wboit_color_view.as_ref(),
                self.wboit_reveal_view.as_ref(),
            ) {
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.wboit.accumulation_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if let Some(ref stack) = self.vector_overlay_stack {
                    if stack.oit_pipelines_ready() && stack.bind_group.is_some() {
                        let layer_count = stack.visible_layer_count();
                        let highlight_color = [1.0, 0.8, 0.0, 0.5];
                        for i in 0..layer_count {
                            stack.render_layer_oit(
                                &mut oit_pass,
                                vector_overlay::RenderLayerParams {
                                    layer_index: i,
                                    view_proj: state.view_proj_array,
                                    sun_dir: state.vo_sun_dir,
                                    lighting: state.vo_lighting,
                                    render_origin_span: state.render_origin_span,
                                    selected_feature_id,
                                    highlight_color,
                                },
                            );
                        }
                    }
                }
            }

            if let (Some(pipeline), Some(bind_group)) = (
                self.wboit_compose_pipeline.as_ref(),
                self.wboit_compose_bind_group.as_ref(),
            ) {
                let mut compose_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain_viewer.wboit.compose_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                compose_pass.set_pipeline(pipeline);
                compose_pass.set_bind_group(0, bind_group, &[]);
                compose_pass.draw(0..3, 0..1);
            }

            static OIT_LOG_ONCE: std::sync::Once = std::sync::Once::new();
            OIT_LOG_ONCE.call_once(|| {
                println!("[render] WBOIT active: mode={}", self.oit_mode);
            });
        }

        if flags.needs_taa {
            let color_view = self
                .post_process
                .as_ref()
                .and_then(|pp| pp.intermediate_view.as_ref());
            let velocity_view = self.taa_velocity_view.as_ref();
            if let (Some(color_view), Some(velocity_view), Some(taa)) =
                (color_view, velocity_view, self.taa_renderer.as_mut())
            {
                taa.update_settings(
                    &self.queue,
                    self.taa_jitter.offset_array(),
                    self.taa_jitter.index,
                );
                taa.execute(
                    &self.device,
                    encoder,
                    &self.queue,
                    color_view,
                    velocity_view,
                    velocity_view,
                );
                self.taa_jitter.advance();
            }
        }

        #[cfg(feature = "enable-gpu-instancing")]
        {
            self.scatter_batches = scatter_batches;
            match scatter_result {
                Ok(stats) => {
                    self.scatter_last_frame_stats = stats;
                }
                Err(err) => {
                    self.scatter_last_frame_stats =
                        crate::terrain::scatter::TerrainScatterFrameStats::default();
                    eprintln!("[terrain_scatter] screen render failed: {err:#}");
                }
            }
        }
    }

    pub(super) fn prepare_screen_overlays(&mut self) -> bool {
        let scene_format = self.scene_color_format();
        let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
            stack.is_enabled() && stack.visible_layer_count() > 0
        } else {
            false
        };

        if has_vector_overlays {
            if self.fallback_texture.is_none() {
                let texture = match crate::core::resource_tracker::tracked_create_texture(
                    &self.device,
                    &wgpu::TextureDescriptor {
                        label: Some("vector_overlay_fallback_texture"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[terrain] failed to allocate overlay fallback texture: {e}");
                        return false;
                    }
                };
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&[1.0f32]),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                self.fallback_texture_view =
                    Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
                self.fallback_texture = Some(texture);
            }

            if let Some(ref mut stack) = self.vector_overlay_stack {
                if !stack.pipelines_ready() || (self.oit_enabled && !stack.oit_pipelines_ready()) {
                    if let Err(e) = stack.init_pipelines(scene_format) {
                        eprintln!("[terrain] overlay pipeline init failed: {e}");
                    }
                }
                let texture_view = self
                    .sun_vis_view
                    .as_ref()
                    .or(self.fallback_texture_view.as_ref())
                    .unwrap();
                stack.prepare_bind_group(texture_view);
            }
        }

        has_vector_overlays
    }
}
