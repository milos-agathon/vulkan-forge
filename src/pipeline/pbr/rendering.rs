use super::*;

impl PbrPipelineWithShadows {
    pub fn ensure_pipeline(
        &mut self,
        device: &Device,
        surface_format: TextureFormat,
    ) -> &wgpu::RenderPipeline {
        if self.shadow_manager.is_none() {
            self.rebuild_shadow_resources(device);
        }

        if self.shadow_bind_group_layout.is_none() {
            if let Some(manager) = self.shadow_manager.as_ref() {
                let layout = manager.create_bind_group_layout(device);
                self.shadow_bind_group_layout = Some(layout);
                self.render_pipeline = None;
            }
        }

        if self.pipeline_format != Some(surface_format) {
            self.render_pipeline = None;
            self.pipeline_format = Some(surface_format);
        }

        if self.render_pipeline.is_none() {
            let pipeline = self.build_render_pipeline(device, surface_format);
            self.render_pipeline = Some(pipeline);
        }

        self.render_pipeline
            .as_ref()
            .expect("PBR render pipeline should be initialized")
    }

    pub fn bind_shadow_resources<'a>(
        &'a mut self,
        device: &Device,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        if let Some(bind_group) = self.get_or_create_shadow_bind_group(device) {
            pass.set_bind_group(3, bind_group, &[]);
        }
    }

    pub fn bind_ibl_resources<'a>(&'a mut self, device: &Device, pass: &mut wgpu::RenderPass<'a>) {
        let bind_group = self.ensure_ibl_bind_group(device);
        pass.set_bind_group(2, bind_group, &[]); // group(2) as per P4 spec
    }

    pub fn begin_render<'a>(
        &'a mut self,
        device: &Device,
        surface_format: TextureFormat,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        self.ensure_pipeline(device, surface_format);
        let render_pipeline_ptr = self
            .render_pipeline
            .as_ref()
            .expect("render pipeline should be initialized")
            as *const wgpu::RenderPipeline;
        let globals_bind_group_ptr = self.ensure_globals_bind_group(device) as *const BindGroup;
        let material_bind_group_ptr = self
            .material
            .bind_group
            .as_ref()
            .map(|bg| bg as *const BindGroup);
        let ibl_bind_group_ptr = self.ensure_ibl_bind_group(device) as *const BindGroup;
        let shadow_bind_group_ptr = self
            .get_or_create_shadow_bind_group(device)
            .map(|bg| bg as *const BindGroup);

        // Safety: render pipeline lives as long as `self`; pass only reads it.
        pass.set_pipeline(unsafe { &*render_pipeline_ptr });
        pass.set_bind_group(0, unsafe { &*globals_bind_group_ptr }, &[]);
        if let Some(ptr) = material_bind_group_ptr {
            // Safety: bind group pointer derived from stored Option; lifetime tied to `self`.
            pass.set_bind_group(1, unsafe { &*ptr }, &[]);
        }
        // P4 spec: IBL at group(2), Shadows at group(3) to match shader layout
        pass.set_bind_group(2, unsafe { &*ibl_bind_group_ptr }, &[]);
        if let Some(ptr) = shadow_bind_group_ptr {
            pass.set_bind_group(3, unsafe { &*ptr }, &[]);
        }
    }

    pub fn bind_global_uniforms<'a>(
        &'a mut self,
        device: &Device,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        let bind_group = self.ensure_globals_bind_group(device);
        pass.set_bind_group(0, bind_group, &[]);
    }

    pub fn update_shadows(
        &mut self,
        queue: &Queue,
        camera_view: glam::Mat4,
        camera_projection: glam::Mat4,
        light_direction: glam::Vec3,
        near_plane: f32,
        far_plane: f32,
    ) {
        if let Some(ref mut manager) = self.shadow_manager {
            manager.update_cascades(
                camera_view,
                camera_projection,
                light_direction,
                near_plane,
                far_plane,
            );
            manager.upload_uniforms(queue);
        }
    }

    pub fn get_or_create_shadow_bind_group(&mut self, device: &Device) -> Option<&BindGroup> {
        let manager = match self.shadow_manager.as_ref() {
            Some(manager) => manager,
            None => return None,
        };

        if self.shadow_bind_group_layout.is_none() {
            let layout = manager.create_bind_group_layout(device);
            self.shadow_bind_group_layout = Some(layout);
            self.render_pipeline = None;
        }

        let layout = self.shadow_bind_group_layout.as_ref()?;

        if self.shadow_bind_group.is_none() {
            let shadow_view = manager.shadow_view();
            let shadow_sampler = manager.shadow_sampler();
            let moment_view = manager.moment_view();
            let moment_sampler = manager.moment_sampler();

            let entries = [
                BindGroupEntry {
                    binding: 0,
                    resource: manager.renderer().uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&shadow_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(shadow_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&moment_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(moment_sampler),
                },
            ];

            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("pbr_shadow_bind_group"),
                layout,
                entries: &entries,
            });

            self.shadow_bind_group = Some(bind_group);
        }

        self.shadow_bind_group.as_ref()
    }

    fn build_render_pipeline(
        &mut self,
        device: &Device,
        surface_format: TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shadow_layout = self
            .shadow_bind_group_layout
            .as_ref()
            .expect("shadow layout must exist before building pipeline");

        // P4 spec: Pipeline layout must match shader bind groups
        // group(0) = globals, group(1) = material, group(2) = IBL (spec requirement), group(3) = shadows
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr_pipeline_layout"),
            bind_group_layouts: &[
                &self.globals_bind_group_layout,  // group(0)
                &self.material_bind_group_layout, // group(1)
                &self.ibl_bind_group_layout,      // group(2) - IBL (P4 spec requirement)
                shadow_layout,                    // group(3) - shadows (moved to avoid conflict)
            ],
            push_constant_ranges: &[],
        });

        // Remap shadows from group(2) to group(3) to allow IBL at group(2) per P4 spec.
        let shader_source = crate::shader_sources::pbr();

        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "pbr_shader_module",
            &shader_source,
        );

        crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("pbr_render_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[TbnVertex::buffer_layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(windows))]
    use super::*;

    #[test]
    fn live_evsm_pbr_pipeline_and_bind_group_create_without_validation_errors() {
        let shader_source = crate::shader_sources::pbr();
        crate::shader_sources::assert_valid_wgsl(&shader_source);

        // GitHub-hosted Windows exposes only a virtual graphics path here, and
        // requesting a wgpu device can terminate the test process in the driver
        // before Rust receives an error. The exact assembled shader contract is
        // still checked above; live creation remains covered on macOS and
        // other non-Windows adapters.
        #[cfg(windows)]
        {
            eprintln!(
                "live EVSM PBR pipeline: hosted Windows device creation is unsafe; \
                 validated the exact assembled WGSL contract statically"
            );
            return;
        }

        #[cfg(not(windows))]
        {
            let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
                crate::shader_sources::assert_valid_wgsl_without_gpu(
                    "live EVSM PBR pipeline",
                    &shader_source,
                );
                return;
            };
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mut pbr = PbrPipelineWithShadows::new(
                &device,
                &queue,
                crate::core::material::PbrMaterial::default(),
                true,
            )
            .expect("PBR pipeline resources");
            pbr.configure_shadows(&device, 3, 256, 0);
            pbr.set_shadow_technique(&device, ShadowTechnique::EVSM);

            pbr.get_or_create_shadow_bind_group(&device)
                .expect("EVSM shadow bind group");

            crate::core::degradation::begin_degradation_capture();
            pbr.ensure_pipeline(&device, TextureFormat::Rgba8Unorm);
            let captured = crate::core::degradation::finish_degradation_capture();
            assert!(!captured
                .iter()
                .any(|item| item.kind == "validation_error" && item.name == "pbr_render_pipeline"));

            device.poll(wgpu::Maintain::Wait);
            if let Some(error) = pollster::block_on(device.pop_error_scope()) {
                panic!("live EVSM PBR resources are invalid: {error}");
            }
        }
    }
}
