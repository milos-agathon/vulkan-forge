use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::viewer::Viewer;

impl Viewer {
    pub(super) fn present_sky_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        depth: Option<&wgpu::TextureView>,
    ) {
        if self.sky_present_bg_cache.borrow().is_none() {
            self.sky_present_bg_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.sky.present.bg"),
                        layout: &self.sky_present_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.sky_present_sampler),
                            },
                        ],
                    },
                )));
        }
        crate::core::shader_registry::record_shader_use("viewer.sky.present.shader");
        let depth_attachment = depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        });
        let bind_group = self.sky_present_bg_cache.borrow();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewer.sky.present.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(if depth.is_some() {
            &self.sky_present_depth_pipeline
        } else {
            &self.sky_present_flat_pipeline
        });
        pass.set_bind_group(
            0,
            bind_group
                .as_ref()
                .expect("sky presentation bind group initialized"),
            &[],
        );
        pass.draw(0..3, 0..1);
    }

    pub(super) fn ensure_sky_bind_groups(&self) {
        if self.sky_bg0_cache.borrow().is_none() {
            self.sky_bg0_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.sky.bg0"),
                        layout: &self.sky_bind_group_layout0,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.sky_params.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                            },
                        ],
                    },
                )));
        }
        if self.sky_bg1_cache.borrow().is_none() {
            self.sky_bg1_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.sky.bg1"),
                        layout: &self.sky_bind_group_layout1,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.sky_camera.as_entire_binding(),
                        }],
                    },
                )));
        }
    }

    pub(super) fn ensure_fog_bind_groups(
        &self,
        gi: &crate::core::screen_space_effects::ScreenSpaceEffectsManager,
    ) {
        if self.fog_bg0_cache.borrow().is_none() {
            self.fog_bg0_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.bg0"),
                        layout: &self.fog_bgl0,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.fog_params.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.fog_camera.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().depth_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&self.fog_depth_sampler),
                            },
                        ],
                    },
                )));
        }
        if self.fog_bg1_cache.borrow().is_none() {
            let (view, uniforms) = if let Some(csm) = self.csm.as_ref() {
                (csm.shadow_array_view(), csm.uniform_buffer())
            } else {
                (&self.fog_shadow_view, self.fog_shadow_matrix.inner())
            };
            self.fog_bg1_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.bg1"),
                        layout: &self.fog_bgl1,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.fog_shadow_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: uniforms.as_entire_binding(),
                            },
                        ],
                    },
                )));
        }
        if self.fog_bg2_cache.borrow().is_none() {
            self.fog_bg2_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.bg2"),
                        layout: &self.fog_bgl2,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.fog_output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.fog_history_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.fog_history_sampler),
                            },
                        ],
                    },
                )));
        }
    }

    pub(super) fn ensure_half_res_fog_bind_groups(
        &self,
        gi: &crate::core::screen_space_effects::ScreenSpaceEffectsManager,
    ) {
        if self.fog_bg2_half_cache.borrow().is_none() {
            self.fog_bg2_half_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.bg2.half"),
                        layout: &self.fog_bgl2,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.fog_output_half_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.fog_history_half_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.fog_history_sampler),
                            },
                        ],
                    },
                )));
        }
        if self.fog_upsample_bg_cache.borrow().is_none() {
            self.fog_upsample_bg_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.upsample.bg"),
                        layout: &self.fog_upsample_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.fog_output_half_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.fog_history_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&self.fog_output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(
                                    &gi.gbuffer().depth_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::Sampler(&self.fog_depth_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.fog_upsample_params.as_entire_binding(),
                            },
                        ],
                    },
                )));
        }
    }

    pub(super) fn ensure_froxel_bind_group(&self) {
        if self.fog_bg3_cache.borrow().is_none() {
            self.fog_bg3_cache
                .replace(Some(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("viewer.fog.bg3"),
                        layout: &self.fog_bgl3,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.froxel_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Sampler(&self.froxel_sampler),
                            },
                        ],
                    },
                )));
        }
    }

    pub(super) fn ensure_lit_bind_group(&self, gi: &ScreenSpaceEffectsManager) {
        if self.lit_bind_group_cache.borrow().is_some() {
            return;
        }
        let env_view = self
            .ibl_env_view
            .as_ref()
            .expect("viewer IBL environment view must be initialized");
        let env_sampler = self
            .ibl_sampler
            .as_ref()
            .expect("viewer IBL sampler must be initialized");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.lit.bg.cached"),
            layout: &self.lit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gi.gbuffer().normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gi.gbuffer().material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gi.gbuffer().depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.lit_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.lit_uniform.as_entire_binding(),
                },
            ],
        });
        *self.lit_bind_group_cache.borrow_mut() = Some(bind_group);
    }

    pub(super) fn ensure_composite_bind_group(
        &self,
        depth_view: &wgpu::TextureView,
        fog_view: &wgpu::TextureView,
        params: &wgpu::Buffer,
        color_view: &wgpu::TextureView,
    ) -> Option<[usize; 5]> {
        let layout = self.comp_bind_group_layout.as_ref()?;
        let key = [
            &self.sky_output_view as *const wgpu::TextureView as usize,
            depth_view as *const wgpu::TextureView as usize,
            fog_view as *const wgpu::TextureView as usize,
            params as *const wgpu::Buffer as usize,
            color_view as *const wgpu::TextureView as usize,
        ];
        if self.comp_bind_group_cache.borrow().contains_key(&key) {
            return Some(key);
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.comp.bg.cached"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.sky_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(fog_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
            ],
        });
        self.comp_bind_group_cache
            .borrow_mut()
            .insert(key, bind_group);
        Some(key)
    }
}
