use super::*;

impl PbrPipelineWithShadows {
    pub(super) fn create_globals_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_globals_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // M2: ShadingParamsGPU (BRDF selection)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // P1-06: Binding 3 - Light array SSBO (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // P1-06: Binding 4 - Light metadata uniform (count, frame, seeds)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // P1-06: Binding 5 - Environment params uniform (zeroed until P4 IBL)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub(super) fn ensure_globals_bind_group(&mut self, device: &Device) -> &BindGroup {
        if self.globals_bind_group.is_none() {
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("pbr_globals_bind_group"),
                layout: &self.globals_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: self.scene_uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: self.lighting_uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.shading_uniform_buffer.as_entire_binding(),
                    },
                    // P1-06: Light buffer bindings (3, 4, 5)
                    BindGroupEntry {
                        binding: 3,
                        resource: self.light_buffer.current_light_buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: self.light_buffer.current_count_buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: self.light_buffer.environment_buffer().as_entire_binding(),
                    },
                ],
            });
            self.globals_bind_group = Some(bind_group);
        }
        self.globals_bind_group
            .as_ref()
            .expect("global bind group should exist")
    }

    pub(super) fn ensure_ibl_bind_group(&mut self, device: &Device) -> &BindGroup {
        if self.ibl_bind_group.is_none() {
            // P4 spec: group(2) bindings - binding(0)=specular, binding(1)=irradiance, binding(2)=sampler, binding(3)=brdfLUT
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("pbr_ibl_bind_group"),
                layout: &self.ibl_bind_group_layout,
                entries: &[
                    // @group(2) @binding(0) envSpecular : texture_cube<f32>
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&self.ibl_resources.prefilter_view),
                    },
                    // @group(2) @binding(1) envIrradiance : texture_cube<f32>
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&self.ibl_resources.irradiance_view),
                    },
                    // @group(2) @binding(2) envSampler : sampler
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(&self.ibl_resources.irradiance_sampler), // Shared sampler
                    },
                    // @group(2) @binding(3) brdfLUT : texture_2d<f32>
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(&self.ibl_resources.brdf_lut_view),
                    },
                ],
            });
            self.ibl_bind_group = Some(bind_group);
        }
        self.ibl_bind_group
            .as_ref()
            .expect("ibl bind group should exist")
    }

    pub(super) fn create_material_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_material_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    pub(super) fn create_ibl_bind_group_layout(device: &Device) -> BindGroupLayout {
        // P4 spec: group(2) bindings - cubemaps for env/spec/irr, 2D for BRDF LUT
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_ibl_bind_group_layout"),
            entries: &[
                // @group(2) @binding(0) envSpecular : texture_cube<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // @group(2) @binding(1) envIrradiance : texture_cube<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // @group(2) @binding(2) envSampler : sampler (filtering + clamp-to-edge)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // @group(2) @binding(3) brdfLUT : texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }
}
