use super::SnapshotRenderState;
#[cfg(feature = "enable-gpu-instancing")]
use crate::viewer::terrain::scene::scatter::render_scatter_batches;
use crate::viewer::terrain::vector_overlay;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn prepare_snapshot_overlays(&mut self) -> bool {
        let scene_format = self.scene_color_format();
        let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
            let enabled = stack.is_enabled();
            let count = stack.visible_layer_count();
            enabled && count > 0
        } else {
            false
        };

        if has_vector_overlays {
            if self.fallback_texture.is_none() {
                let texture = match crate::core::resource_tracker::tracked_create_texture(
                    &self.device,
                    &wgpu::TextureDescriptor {
                        label: Some("vector_overlay_fallback_texture_snapshot"),
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
                let texture_view = self.fallback_texture_view.as_ref().unwrap();
                stack.prepare_bind_group(texture_view);
            }
        }

        has_vector_overlays
    }

    pub(super) fn render_snapshot_scene_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        selected_feature_id: u32,
        state: &SnapshotRenderState,
        has_vector_overlays: bool,
    ) {
        #[cfg(feature = "enable-gpu-instancing")]
        let mut scatter_batches = std::mem::take(&mut self.scatter_batches);

        {
            let terrain = self.terrain.as_ref().unwrap();
            let bg = terrain.background_color;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.snapshot_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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

            if state.use_pbr {
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
                color_view,
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
                        label: Some("terrain_viewer.snapshot_overlay_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
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
                                view_proj: state.vo_view_proj,
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
                    eprintln!("[terrain_scatter] snapshot render failed: {err:#}");
                }
            }
        }
    }

    pub(super) fn render_snapshot_oit_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        selected_feature_id: u32,
        state: &SnapshotRenderState,
        has_vector_overlays: bool,
    ) {
        if !(has_vector_overlays && self.oit_enabled) {
            return;
        }

        self.init_wboit_pipeline();

        let oit_color_tex = match crate::core::resource_tracker::tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain_viewer.snapshot_oit_color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[terrain] failed to allocate snapshot OIT color texture: {e}");
                crate::core::degradation::record_degradation(
                    "allocation_fallback",
                    "viewer.snapshot_wboit",
                    "snapshot OIT targets unavailable; transparent overlays may composite incorrectly",
                );
                return;
            }
        };
        let oit_color_view = oit_color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let oit_reveal_tex = match crate::core::resource_tracker::tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain_viewer.snapshot_oit_reveal"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[terrain] failed to allocate snapshot OIT reveal texture: {e}");
                crate::core::degradation::record_degradation(
                    "allocation_fallback",
                    "viewer.snapshot_wboit",
                    "snapshot OIT targets unavailable; transparent overlays may composite incorrectly",
                );
                return;
            }
        };
        let oit_reveal_view = oit_reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.snapshot_oit_accumulation"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &oit_color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &oit_reveal_view,
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
                                view_proj: state.vo_view_proj,
                                sun_dir: state.vo_sun_dir,
                                lighting: state.vo_lighting,
                                render_origin_span: state.render_origin_span,
                                selected_feature_id,
                                highlight_color,
                            },
                        );
                    }
                } else {
                    println!(
                        "[snapshot] OIT skip: pipelines_ready={} bind_group={}",
                        stack.oit_pipelines_ready(),
                        stack.bind_group.is_some()
                    );
                }
            }
        }

        if let (Some(ref pipeline), Some(ref sampler)) = (
            self.wboit_compose_pipeline.as_ref(),
            self.wboit_sampler.as_ref(),
        ) {
            println!("[snapshot] Compositing OIT result");
            let layout = &pipeline.get_bind_group_layout(0);
            let compose_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain_viewer.snapshot_oit_compose_bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&oit_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&oit_reveal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });

            let mut compose_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain_viewer.snapshot_oit_compose"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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
            compose_pass.set_bind_group(0, &compose_bind_group, &[]);
            compose_pass.draw(0..3, 0..1);
        }
    }
}
