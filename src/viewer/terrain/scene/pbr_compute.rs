use super::*;
use crate::core::resource_tracker::{tracked_create_buffer, tracked_create_texture};

impl ViewerTerrainScene {
    pub fn init_pbr_pipeline(&mut self, target_format: wgpu::TextureFormat) -> Result<()> {
        if self.pbr_pipeline.is_some() {
            return Ok(()); // Already initialized
        }

        // P6.2: Initialize shadows
        self.init_shadows()?;

        let pbr_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("terrain_viewer_pbr.bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(256),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        // Height AO texture (R32Float, non-filterable)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Sun visibility texture (R32Float, non-filterable)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Overlay texture (RGBA8, filterable)
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
                        // Overlay sampler (filterable)
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // P6.2: Shadow Map Array (Depth)
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // P6.2: Shadow Sampler (Comparison)
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                            count: None,
                        },
                        // P6.2: Moment Map Array (Float, for VSM/EVSM/MSM)
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // P6.2: Moment Sampler (Filtering)
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // P6.2: CSM Uniforms (Storage Buffer)
                        wgpu::BindGroupLayoutEntry {
                            binding: 11,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Terrain HDRI specular cube map
                        wgpu::BindGroupLayoutEntry {
                            binding: 12,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Terrain HDRI irradiance cube map
                        wgpu::BindGroupLayoutEntry {
                            binding: 13,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Terrain HDRI sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 14,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // Terrain HDRI BRDF LUT
                        wgpu::BindGroupLayoutEntry {
                            binding: 15,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &self.device,
            "terrain_viewer_pbr.shader",
            crate::viewer::terrain::shader_pbr::TERRAIN_PBR_SHADER,
        );

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain_viewer_pbr.pipeline_layout"),
                bind_group_layouts: &[&pbr_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pbr_pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            &self.device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("terrain_viewer_pbr.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        );

        self.pbr_pipeline = Some(pbr_pipeline);
        self.pbr_bind_group_layout = Some(pbr_bind_group_layout);
        println!("[terrain] PBR pipeline initialized");
        Ok(())
    }

    /// Initialize compute pipelines for heightfield AO and sun visibility
    pub fn init_heightfield_compute_pipelines(&mut self) -> Result<()> {
        let terrain = self
            .terrain
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No terrain loaded"))?;
        let (width, height) = terrain.dimensions;

        // Create non-filtering sampler for R32Float textures (R32Float doesn't support filtering on Metal)
        if self.sampler_nearest.is_none() {
            self.sampler_nearest = Some(self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("terrain_viewer.sampler_nearest"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        }

        // Initialize height AO compute pipeline if enabled and not already initialized
        if self.pbr_config.height_ao.enabled && self.height_ao_pipeline.is_none() {
            let ao_width = (width as f32 * self.pbr_config.height_ao.resolution_scale) as u32;
            let ao_height = (height as f32 * self.pbr_config.height_ao.resolution_scale) as u32;

            // Create AO texture
            let ao_texture = tracked_create_texture(
                &self.device,
                &wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.height_ao_texture"),
                    size: wgpu::Extent3d {
                        width: ao_width,
                        height: ao_height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )?;
            self.height_ao_view =
                Some(ao_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.height_ao_texture = Some(ao_texture);

            // Create uniform buffer
            self.height_ao_uniform_buffer = Some(tracked_create_buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("terrain_viewer.height_ao_uniforms"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?);

            // Create bind group layout
            let ao_bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("terrain_viewer.height_ao_bind_group_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: false,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Sampler(
                                    wgpu::SamplerBindingType::NonFiltering,
                                ),
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::StorageTexture {
                                    access: wgpu::StorageTextureAccess::WriteOnly,
                                    format: wgpu::TextureFormat::R32Float,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                },
                                count: None,
                            },
                        ],
                    });

            let ao_shader = crate::core::shader_registry::create_labeled_shader_module(
                &self.device,
                "terrain_viewer.height_ao_shader",
                include_str!("../../../shaders/heightfield_ao.wgsl"),
            );

            let ao_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("terrain_viewer.height_ao_pipeline_layout"),
                        bind_group_layouts: &[&ao_bind_group_layout],
                        push_constant_ranges: &[],
                    });

            self.height_ao_pipeline = Some(
                crate::core::shader_registry::create_compute_pipeline_scoped(
                    &self.device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("terrain_viewer.height_ao_pipeline"),
                        layout: Some(&ao_pipeline_layout),
                        module: &ao_shader,
                        entry_point: "main",
                    },
                ),
            );
            self.height_ao_bind_group_layout = Some(ao_bind_group_layout);
            println!(
                "[terrain] Height AO compute pipeline initialized ({}x{})",
                ao_width, ao_height
            );
        }

        // Initialize sun visibility compute pipeline if enabled and not already initialized
        if self.pbr_config.sun_visibility.enabled && self.sun_vis_pipeline.is_none() {
            let sv_width = (width as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;
            let sv_height =
                (height as f32 * self.pbr_config.sun_visibility.resolution_scale) as u32;

            // Create sun vis texture
            let sv_texture = tracked_create_texture(
                &self.device,
                &wgpu::TextureDescriptor {
                    label: Some("terrain_viewer.sun_vis_texture"),
                    size: wgpu::Extent3d {
                        width: sv_width,
                        height: sv_height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )?;
            self.sun_vis_view =
                Some(sv_texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.sun_vis_texture = Some(sv_texture);

            // Create uniform buffer
            self.sun_vis_uniform_buffer = Some(tracked_create_buffer(
                &self.device,
                &wgpu::BufferDescriptor {
                    label: Some("terrain_viewer.sun_vis_uniforms"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?);

            // Create bind group layout
            let sv_bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("terrain_viewer.sun_vis_bind_group_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: false,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Sampler(
                                    wgpu::SamplerBindingType::NonFiltering,
                                ),
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::StorageTexture {
                                    access: wgpu::StorageTextureAccess::WriteOnly,
                                    format: wgpu::TextureFormat::R32Float,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                },
                                count: None,
                            },
                        ],
                    });

            let sv_shader = crate::core::shader_registry::create_labeled_shader_module(
                &self.device,
                "terrain_viewer.sun_vis_shader",
                include_str!("../../../shaders/heightfield_sun_vis.wgsl"),
            );

            let sv_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("terrain_viewer.sun_vis_pipeline_layout"),
                        bind_group_layouts: &[&sv_bind_group_layout],
                        push_constant_ranges: &[],
                    });

            self.sun_vis_pipeline = Some(
                crate::core::shader_registry::create_compute_pipeline_scoped(
                    &self.device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("terrain_viewer.sun_vis_pipeline"),
                        layout: Some(&sv_pipeline_layout),
                        module: &sv_shader,
                        entry_point: "main",
                    },
                ),
            );
            self.sun_vis_bind_group_layout = Some(sv_bind_group_layout);
            println!(
                "[terrain] Sun visibility compute pipeline initialized ({}x{})",
                sv_width, sv_height
            );
        }

        Ok(())
    }
}
