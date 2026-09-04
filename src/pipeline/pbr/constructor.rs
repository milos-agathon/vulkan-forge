use super::*;

impl PbrPipelineWithShadows {
    pub fn new(
        device: &Device,
        queue: &Queue,
        material: PbrMaterial,
        enable_shadows: bool,
    ) -> RenderResult<Self> {
        let material_gpu = PbrMaterialGpu::new(device, material)?;
        let scene_uniforms = PbrSceneUniforms::default();
        let scene_uniform_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("pbr_scene_uniforms_buffer"),
                contents: bytemuck::bytes_of(&scene_uniforms),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        )?;
        let lighting_uniforms = PbrLighting::default();
        let lighting_uniform_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("pbr_lighting_uniforms_buffer"),
                contents: bytemuck::bytes_of(&lighting_uniforms),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        )?;

        // P2-06: Shading (BRDF selection) uniforms using MaterialShading
        let shading_uniforms = MaterialShading::default();
        let shading_uniform_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("pbr_shading_uniforms_buffer"),
                contents: bytemuck::bytes_of(&shading_uniforms),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            },
        )?;

        let mut shadow_config = ShadowManagerConfig::default();
        shadow_config.technique = ShadowTechnique::PCF;
        shadow_config.csm.cascade_count = 3;
        shadow_config.csm.shadow_map_size = 2048;
        shadow_config.csm.max_shadow_distance = 200.0;
        shadow_config.csm.pcf_kernel_size = 3;
        shadow_config.csm.depth_bias = 0.005;
        shadow_config.csm.slope_bias = 0.01;
        shadow_config.csm.peter_panning_offset = 0.001;
        shadow_config.csm.debug_mode = 0;

        let globals_bind_group_layout = Self::create_globals_bind_group_layout(device);
        let material_bind_group_layout = Self::create_material_bind_group_layout(device);
        let ibl_bind_group_layout = Self::create_ibl_bind_group_layout(device);
        let ibl_resources = create_fallback_ibl_resources(device, queue)?;

        // P1-06: Initialize light buffer for multi-light support
        let light_buffer = LightBuffer::new(device)?;
        // P4 spec: group(2) bindings - binding(0)=specular, binding(1)=irradiance, binding(2)=sampler, binding(3)=brdfLUT
        // Use fallback resources until the IBL renderer owns these textures.
        let ibl_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pbr_ibl_bind_group"),
            layout: &ibl_bind_group_layout,
            entries: &[
                // @group(2) @binding(0) envSpecular : texture_cube<f32>
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&ibl_resources.prefilter_view),
                },
                // @group(2) @binding(1) envIrradiance : texture_cube<f32>
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&ibl_resources.irradiance_view),
                },
                // @group(2) @binding(2) envSampler : sampler
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&ibl_resources.irradiance_sampler), // Shared sampler
                },
                // @group(2) @binding(3) brdfLUT : texture_2d<f32>
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&ibl_resources.brdf_lut_view),
                },
            ],
        });
        let globals_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pbr_globals_bind_group"),
            layout: &globals_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: scene_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lighting_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: shading_uniform_buffer.as_entire_binding(),
                },
                // P1-06: Light buffer bindings (3, 4, 5)
                BindGroupEntry {
                    binding: 3,
                    resource: light_buffer.current_light_buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: light_buffer.current_count_buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: light_buffer.environment_buffer().as_entire_binding(),
                },
            ],
        });

        let mut shadow_manager = None;
        let mut shadow_bind_group_layout = None;

        if enable_shadows {
            let manager = ShadowManager::new(device, shadow_config.clone())?;
            shadow_config = manager.config().clone();
            shadow_bind_group_layout = Some(manager.create_bind_group_layout(device));
            shadow_manager = Some(manager);
        }

        Ok(Self {
            material: material_gpu,
            scene_uniforms,
            scene_uniform_buffer,
            lighting_uniforms,
            lighting_uniform_buffer,
            shading_uniforms,
            shading_uniform_buffer,
            globals_bind_group: Some(globals_bind_group),
            ibl_resources,
            ibl_bind_group: Some(ibl_bind_group),
            shadow_config,
            shadow_manager,
            shadow_bind_group: None,
            shadow_bind_group_layout,
            globals_bind_group_layout,
            material_bind_group_layout,
            ibl_bind_group_layout,
            render_pipeline: None,
            pipeline_format: None,
            tone_mapping: ToneMappingConfig::new(ToneMappingMode::Reinhard, 1.0),
            light_buffer,
        })
    }

    pub fn set_shadow_enabled(&mut self, device: &Device, enabled: bool) {
        if enabled {
            if self.shadow_manager.is_none() {
                self.rebuild_shadow_resources(device);
            }
        } else if self.shadow_manager.is_some() {
            self.drop_shadow_resources();
        }
    }

    pub fn configure_shadows(
        &mut self,
        device: &Device,
        pcf_kernel_size: u32,
        shadow_map_size: u32,
        debug_mode: u32,
    ) {
        self.shadow_config.csm.pcf_kernel_size = pcf_kernel_size;
        self.shadow_config.csm.shadow_map_size = shadow_map_size;
        self.shadow_config.csm.debug_mode = debug_mode;

        if self.shadow_manager.is_some() {
            self.rebuild_shadow_resources(device);
        }
        self.shadow_bind_group = None;
    }

    pub fn set_shadow_technique(&mut self, device: &Device, technique: ShadowTechnique) {
        if self.shadow_config.technique == technique {
            return;
        }

        self.shadow_config.technique = technique;
        if self.shadow_manager.is_some() {
            self.rebuild_shadow_resources(device);
        } else {
            self.shadow_bind_group_layout = None;
        }

        self.shadow_bind_group = None;
    }
}
