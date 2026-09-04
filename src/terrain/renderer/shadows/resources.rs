use super::*;
use crate::core::resource_tracker::{tracked_create_buffer_init, tracked_create_texture};

impl TerrainScene {
    pub(in crate::terrain::renderer) fn create_noop_shadow(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shadow_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<NoopShadow> {
        use crate::core::shadow_mapping::{CsmCascadeData, CsmUniforms};

        let identity_mat = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let default_cascade = CsmCascadeData {
            light_projection: identity_mat,
            light_view_proj: identity_mat,
            near_distance: 0.0,
            far_distance: 100000.0,
            texel_size: 1.0,
            _padding: 0.0,
        };
        let csm_uniforms = CsmUniforms {
            light_direction: [0.0, -1.0, 0.0, 0.0],
            light_view: identity_mat,
            cascades: [default_cascade; 4],
            cascade_count: 0,
            pcf_kernel_size: 1,
            depth_bias: 0.0,
            slope_bias: 0.0,
            shadow_map_size: 1.0,
            debug_mode: 0,
            evsm_positive_exp: 40.0,
            evsm_negative_exp: 5.0,
            peter_panning_offset: 0.0,
            enable_unclipped_depth: 0,
            depth_clip_factor: 1.0,
            technique: 1,
            technique_flags: 0,
            _padding1: [0.0; 3],
            technique_params: [0.0; 4],
            technique_reserved: [0.0; 4],
            cascade_blend_range: 0.0,
            _padding2: [0.0; 27],
        };
        let csm_uniform_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.noop_shadow.csm_uniforms"),
                contents: bytemuck::bytes_of(&csm_uniforms),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )?;

        let shadow_maps_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.noop_shadow.maps"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        )?;
        let shadow_clear_view = shadow_maps_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.noop_shadow.maps.clear_view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain.noop_shadow.clear_encoder"),
        });
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.noop_shadow.depth_clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shadow_clear_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        queue.submit(Some(encoder.finish()));

        let shadow_maps_view = shadow_maps_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.noop_shadow.maps.view"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.noop_shadow.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let moment_maps_format = wgpu::TextureFormat::Rgba16Float;
        let moment_maps_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.noop_shadow.moments"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: moment_maps_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        debug_assert_eq!(moment_maps_format, wgpu::TextureFormat::Rgba16Float);
        let moment_maps_view = moment_maps_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.noop_shadow.moments.view"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        let moment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.noop_shadow.moment_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: csm_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_maps_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&moment_maps_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&moment_sampler),
            },
        ];

        debug_assert_eq!(entries.len(), 5);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.noop_shadow.bind_group"),
            layout: shadow_bind_group_layout,
            entries: &entries,
        });

        Ok(NoopShadow {
            _csm_uniform_buffer: csm_uniform_buffer,
            _shadow_maps_texture: shadow_maps_texture,
            _shadow_maps_view: shadow_maps_view,
            _shadow_sampler: shadow_sampler,
            _moment_maps_texture: moment_maps_texture,
            moment_maps_view,
            moment_sampler,
            bind_group,
        })
    }

    pub(in crate::terrain::renderer) fn create_shadow_depth_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader_source = crate::shader_sources::terrain_shadow_depth();
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "terrain.shadow_depth.shader",
            &shader_source,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.shadow_depth.pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        crate::core::shader_registry::create_render_pipeline_scoped(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("terrain.shadow_depth.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_shadow",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_shadow",
                    targets: &[],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        )
    }
}
