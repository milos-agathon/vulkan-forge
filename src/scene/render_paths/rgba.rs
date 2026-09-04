impl Scene {
    pub(super) fn render_rgba_impl<'py>(
        &mut self,
        py: pyo3::Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
        #[cfg(feature = "extension-module")]
        self.ensure_legacy_scene_atmosphere_unconfigured()?;
        let (certificate_capture, _allocation_scope) =
            self.begin_certificate_capture("scene.render_rgba");
        let mut timing = self.take_render_timing();

        let g = crate::core::gpu::try_ctx()?;
        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene-encoder-rgba"),
            });
        self.encode_rgba_frame(&mut encoder, &mut timing)?;
        g.queue.submit(Some(encoder.finish()));

        let mut pixels =
            self.readback_color_pixels("scene-readback-rgba", "copy-encoder-rgba", &mut timing)?;

        // Read back live GPU-pass timings, record each into the certificate,
        // and freeze this render's capture (one render = one capture).
        self.record_render_timings(&mut timing);
        self.store_render_timing(timing);
        self.finish_certificate_capture(certificate_capture);

        self.apply_runtime_postfx_cpu(&mut pixels);

        let arr = Array3::from_shape_vec((self.height as usize, self.width as usize, 4), pixels)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    fn encode_rgba_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    ) -> PyResult<()> {
        let g = crate::core::gpu::try_ctx()?;
        if self.fast_softlight_only() {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let (normal_target, normal_resolve) = if self.sample_count > 1 {
                (
                    self.msaa_normal_view
                        .as_ref()
                        .expect("MSAA normal view missing when sample_count > 1"),
                    Some(&self.normal_view),
                )
            } else {
                (&self.normal_view, None)
            };
            let depth_attachment =
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    });
            let fast_scope = scene_ts_begin(timing, encoder, "scene.main");
            {
                let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scene-rp-fast-clear"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.03,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: normal_target,
                            resolve_target: normal_resolve,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Discard,
                            },
                        }),
                    ],
                    depth_stencil_attachment: depth_attachment,
                    ..Default::default()
                });
            }
            scene_ts_end(timing, encoder, fast_scope, 1);
            return Ok(());
        }

        let refl_scope = if self.reflections_enabled {
            scene_ts_begin(timing, encoder, "scene.reflections")
        } else {
            None
        };
        self.render_reflections(encoder).map_err(reflection_err)?;
        scene_ts_end(timing, encoder, refl_scope, 0);

        let cloud_shadow_scope = if self.cloud_shadows_enabled {
            scene_ts_begin(timing, encoder, "scene.cloud_shadows")
        } else {
            None
        };
        self.render_cloud_shadows(encoder)
            .map_err(cloud_shadow_err)?;
        scene_ts_end(timing, encoder, cloud_shadow_scope, 0);

        if let Some(ref mut renderer) = self.reflection_renderer {
            if renderer.bind_group().is_none() {
                renderer.create_bind_group(&g.device, &self.tp.bgl_reflection);
            }
        }

        let main_scope = scene_ts_begin(timing, encoder, "scene.main");
        {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let (normal_target, normal_resolve) = if self.sample_count > 1 {
                (
                    self.msaa_normal_view
                        .as_ref()
                        .expect("MSAA normal view missing when sample_count > 1"),
                    Some(&self.normal_view),
                )
            } else {
                (&self.normal_view, None)
            };
            let depth_attachment =
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    });

            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp-rgba"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.03,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: normal_target,
                        resolve_target: normal_resolve,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Discard,
                        },
                    }),
                ],
                depth_stencil_attachment: depth_attachment,
                ..Default::default()
            });

            if self.ground_plane_enabled {
                if let Some(ref mut ground_renderer) = self.ground_plane_renderer {
                    crate::core::shader_registry::record_shader_use("ground_plane_shader");
                    let view_proj = self.scene.proj * self.scene.view;
                    ground_renderer.set_camera(view_proj);
                    ground_renderer.upload_uniforms(&g.queue);
                    ground_renderer.render(&mut rp);
                }
            }

            if self.water_surface_enabled {
                if let Some(ref mut water_renderer) = self.water_surface_renderer {
                    crate::core::shader_registry::record_shader_use("water_surface_shader");
                    let view_proj = self.scene.proj * self.scene.view;
                    water_renderer.set_camera(view_proj);
                    water_renderer.upload_uniforms(&g.queue);
                    water_renderer.render(&mut rp);
                }
            }

            if self.soft_light_radius_enabled {
                if let Some(ref soft_light_renderer) = self.soft_light_radius_renderer {
                    crate::core::shader_registry::record_shader_use("soft_light_radius_shader");
                    soft_light_renderer.update_uniforms(&g.queue);
                    soft_light_renderer.render(&mut rp, false);
                }
            }

            if self.point_spot_lights_enabled {
                if let Some(ref mut lights_renderer) = self.point_spot_lights_renderer {
                    crate::core::shader_registry::record_shader_use("point_spot_lights_shader");
                    lights_renderer.set_camera(self.scene.view, self.scene.proj);
                    lights_renderer.update_buffers(&g.queue);
                    lights_renderer.render_deferred(&mut rp);
                }
            }

            if self.terrain_enabled {
                Self::record_terrain_shader_use();
                rp.set_pipeline(&self.tp.pipeline);
                rp.set_bind_group(0, &self.bg0_globals, &[]);
                rp.set_bind_group(1, &self.bg1_height, &[]);
                rp.set_bind_group(2, &self.bg2_lut, &[]);
                rp.set_bind_group(3, &self.bg3_tile, &[]);
                let max_groups = crate::core::gpu::try_ctx()?.device.limits().max_bind_groups;
                if max_groups >= 6 {
                    let cloud_bg = self
                        .bg3_cloud_shadows
                        .as_ref()
                        .unwrap_or(&self.bg4_dummy_cloud_shadows);
                    rp.set_bind_group(4, cloud_bg, &[]);
                }
                if max_groups >= 6 {
                    if let Some(ref renderer) = self.reflection_renderer {
                        if let Some(reflection_bg) = renderer.bind_group() {
                            rp.set_bind_group(5, reflection_bg, &[]);
                        }
                    }
                }

                rp.set_vertex_buffer(0, self.vbuf.slice(..));
                rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.nidx, 0, 0..1);
            }
        }
        scene_ts_end(timing, encoder, main_scope, 1);

        #[cfg(feature = "enable-gpu-instancing")]
        let has_instanced_batches =
            self.mesh_instanced_renderer.is_some() && !self.instanced_batches.is_empty();
        #[cfg(not(feature = "enable-gpu-instancing"))]
        let has_instanced_batches = false;

        if has_instanced_batches || self.text3d_enabled {
            let (target_view, resolve_target) = if self.sample_count > 1 {
                (
                    self.msaa_view
                        .as_ref()
                        .expect("MSAA view missing when sample_count > 1"),
                    Some(&self.color_view),
                )
            } else {
                (&self.color_view, None)
            };
            let depth_attachment = if self.sample_count > 1 {
                self.depth_view
                    .as_ref()
                    .map(|view| wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    })
            } else {
                None
            };
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp-rgba-forward"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                ..Default::default()
            });

            #[cfg(feature = "enable-gpu-instancing")]
            {
                if has_instanced_batches {
                    crate::core::shader_registry::record_shader_use("mesh_instanced_shader");
                    let view = self.scene.view;
                    let proj = self.scene.proj;
                    if let Some(renderer) = self.mesh_instanced_renderer.as_mut() {
                        renderer.reset_draw_batch_uniforms();
                        for batch in &self.instanced_batches {
                            renderer.draw_batch_params(
                                &g.device,
                                &mut rp,
                                &g.queue,
                                view,
                                proj,
                                batch.color,
                                batch.light_dir,
                                batch.light_intensity,
                                [0.0; 4],
                                [0.0; 4],
                                [0.0; 4],
                                [0.0, 0.75, 2.5, 0.0],
                                [0.0, 3.0, 0.35, 0.65],
                                None,
                                &batch.vbuf,
                                &batch.ibuf,
                                &batch.instbuf,
                                batch.index_count,
                                batch.instance_count,
                            );
                        }
                    }
                }
            }

            if self.text3d_enabled {
                if let Some(ref mut tm) = self.text3d_renderer {
                    crate::core::shader_registry::record_shader_use("mesh_basic_shader");
                    let g = crate::core::gpu::try_ctx()?;
                    tm.set_view_proj(self.scene.view, self.scene.proj);
                    tm.upload_uniforms(&g.queue);
                    for inst in &self.text3d_instances {
                        tm.draw_instance_with_light(
                            &mut rp,
                            &g.queue,
                            inst.model,
                            inst.color,
                            inst.light_dir,
                            inst.light_intensity,
                            inst.metallic,
                            inst.roughness,
                            &inst.vbuf,
                            &inst.ibuf,
                            inst.index_count,
                        );
                    }
                }
            }
        }

        if self.overlay_renderer.is_some() || self.text_overlay_enabled {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-rp-rgba-overlays"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            if let Some(ref ov) = self.overlay_renderer {
                crate::core::shader_registry::record_shader_use("overlays_shader");
                ov.render(&mut rp);
            }

            if self.text_overlay_enabled {
                if let Some(ref mut tr) = self.text_overlay_renderer {
                    crate::core::shader_registry::record_shader_use("text_overlay_shader");
                    let g = crate::core::gpu::try_ctx()?;
                    tr.set_resolution(self.width, self.height);
                    tr.set_alpha(self.text_overlay_alpha);
                    tr.set_enabled(true);
                    tr.upload_uniforms(&g.queue);
                    if !self.text_instances.is_empty() {
                        let inst = self.text_instances.clone();
                        tr.upload_instances(&g.device, &g.queue, &inst)?;
                    }
                    tr.render(&mut rp);
                }
            }
        }

        if self.ssao_enabled {
            crate::core::shader_registry::record_shader_use("ssao-compute");
            let ssao_scope = scene_ts_begin(timing, encoder, "scene.ssao");
            self.ssao
                .dispatch(
                    &g.device,
                    &g.queue,
                    encoder,
                    &self.normal_view,
                    &self.color,
                    &self.scene.proj,
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            scene_ts_end(timing, encoder, ssao_scope, 0);
        }

        let clouds_scope = if self.clouds_enabled {
            scene_ts_begin(timing, encoder, "scene.clouds")
        } else {
            None
        };
        self.render_clouds(encoder).map_err(cloud_render_err)?;
        scene_ts_end(timing, encoder, clouds_scope, 0);

        let dof_scope = if self.dof_enabled {
            scene_ts_begin(timing, encoder, "scene.dof")
        } else {
            None
        };
        self.render_dof(encoder).map_err(dof_err)?;
        scene_ts_end(timing, encoder, dof_scope, 0);
        Ok(())
    }
}
