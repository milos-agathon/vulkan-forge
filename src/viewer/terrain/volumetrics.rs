// src/viewer/terrain/volumetrics.rs
// P5: Volumetric fog pass for terrain viewer

use std::sync::Arc;
use wgpu;

use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use crate::viewer::event_loop::update_terrain_volumetrics_report;
use crate::viewer::ipc::TerrainVolumetricsReport;
use crate::viewer::terrain::volume_density::{
    self, build_density_volume_atlas_data_checked, DensityVolumeAtlasGpu, TerrainVolumeContext,
};

const MAX_DENSITY_VOLUMES: usize = volume_density::MAX_DENSITY_VOLUMES;

/// Uniforms for volumetrics shader (must match viewer_volumetrics.wgsl)
/// Note: vec3 in WGSL has 16-byte alignment, so we use [f32; 4] for vec3 fields
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricsUniforms {
    pub inv_view_proj: [[f32; 4]; 4], // 64 bytes, offset 0
    pub camera_pos: [f32; 4],         // 16 bytes (vec3 + pad), offset 64
    pub near_far: [f32; 4],           // near, far, density, height_falloff; offset 80
    pub scatter_absorb: [f32; 4],     // scattering, absorption, pad, pad; offset 96
    pub sun_direction: [f32; 4],      // 16 bytes (vec3 + intensity), offset 112
    pub shaft_params: [f32; 4], // shaft_intensity, light_shafts_enabled, steps, mode; offset 128
    pub screen_dims: [f32; 4],  // width, height, pad, pad; offset 144
    pub terrain_params: [f32; 4], // reference span, min_h, h_scale, h_range; offset 160
    pub render_origin_span: [f32; 4], // anchored origin x/z and physical span x/z
    pub density_volume_count: [f32; 4],
    pub density_volume_min: [[f32; 4]; MAX_DENSITY_VOLUMES],
    pub density_volume_inv_size: [[f32; 4]; MAX_DENSITY_VOLUMES],
    pub density_volume_atlas_offset: [[f32; 4]; MAX_DENSITY_VOLUMES],
    pub density_volume_atlas_scale: [[f32; 4]; MAX_DENSITY_VOLUMES],
}

/// Volumetric fog pass manager
pub struct VolumetricsPass {
    device: Arc<wgpu::Device>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: TrackedBuffer,
    color_sampler: wgpu::Sampler,
    depth_sampler: wgpu::Sampler,
    fallback_density_volume_texture: Option<TrackedTexture>,
    fallback_density_volume_view: Option<wgpu::TextureView>,
    density_volume_sampler: wgpu::Sampler,
    density_volume_atlas: Option<DensityVolumeAtlasGpu>,
    last_report: TerrainVolumetricsReport,
}

impl VolumetricsPass {
    pub fn new(
        device: Arc<wgpu::Device>,
        target_format: wgpu::TextureFormat,
    ) -> anyhow::Result<Self> {
        // Create shader module
        let shader_source = include_str!("../../shaders/viewer_volumetrics.wgsl");
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &device,
            "viewer_volumetrics.wgsl",
            shader_source,
        );

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetrics.bind_group_layout"),
            entries: &[
                // Uniforms
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
                // Color texture
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
                // Color sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Depth texture (Depth32Float, non-filterable on some backends)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth sampler (non-filtering for depth)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Heightmap texture (R32Float, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Optional 3D density volume atlas
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volumetrics.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("volumetrics.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
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
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        );

        // Create uniform buffer
        let uniform_buffer = tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("volumetrics.uniforms"),
                size: std::mem::size_of::<VolumetricsUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;

        // Create color sampler (filtering for color texture)
        let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetrics.color_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        // Create depth sampler (non-filtering for depth texture)
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetrics.depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        let density_volume_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volumetrics.density_volume_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            color_sampler,
            depth_sampler,
            fallback_density_volume_texture: None,
            fallback_density_volume_view: None,
            density_volume_sampler,
            density_volume_atlas: None,
            last_report: TerrainVolumetricsReport {
                memory_budget_bytes: volume_density::DENSITY_VOLUME_MEMORY_BUDGET_BYTES,
                ..TerrainVolumetricsReport::default()
            },
        })
    }

    fn ensure_fallback_density_volume(&mut self) -> anyhow::Result<()> {
        if self.fallback_density_volume_view.is_some() {
            return Ok(());
        }

        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain_viewer.density_volume_fallback"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        self.fallback_density_volume_texture = Some(texture);
        self.fallback_density_volume_view = Some(view);
        Ok(())
    }

    fn default_report(
        &self,
        config: &super::pbr_renderer::VolumetricsConfig,
    ) -> TerrainVolumetricsReport {
        TerrainVolumetricsReport {
            memory_budget_bytes: volume_density::DENSITY_VOLUME_MEMORY_BUDGET_BYTES,
            raymarch_steps: config.steps,
            half_res: config.half_res,
            ..TerrainVolumetricsReport::default()
        }
    }

    fn ensure_density_volume_atlas(
        &mut self,
        queue: &wgpu::Queue,
        heightmap: &[f32],
        height_dims: (u32, u32),
        terrain_revision: u64,
        terrain_params: [f32; 4],
        render_origin_span: [f32; 4],
        config: &super::pbr_renderer::VolumetricsConfig,
    ) -> anyhow::Result<()> {
        if config.density_volumes.is_empty() {
            self.density_volume_atlas = None;
            self.last_report = self.default_report(config);
            return Ok(());
        }

        let context = TerrainVolumeContext {
            heightmap,
            height_dims,
            contract_extent: [height_dims.0.max(1) as f32, height_dims.1.max(1) as f32],
            render_origin_span,
            domain: (terrain_params[1], terrain_params[1] + terrain_params[3]),
            z_scale: terrain_params[2],
            terrain_revision,
        };

        let Some(data) = build_density_volume_atlas_data_checked(context, &config.density_volumes)
            .map_err(|error| anyhow::anyhow!("canonical density atlas rejected: {error}"))?
        else {
            self.density_volume_atlas = None;
            self.last_report = self.default_report(config);
            return Ok(());
        };

        let needs_upload = self
            .density_volume_atlas
            .as_ref()
            .map(|atlas| atlas.fingerprint != data.fingerprint)
            .unwrap_or(true);

        if needs_upload {
            self.density_volume_atlas = Some(DensityVolumeAtlasGpu::upload(
                &self.device,
                queue,
                data,
                config.steps,
                config.half_res,
            )?);
        } else if let Some(atlas) = self.density_volume_atlas.as_mut() {
            let atlas_allocation_id = atlas.report.atlas_allocation_id;
            atlas.metadata = data.metadata;
            atlas.report = data.report;
            atlas.report.atlas_allocation_id = atlas_allocation_id;
            atlas.report.raymarch_steps = config.steps;
            atlas.report.half_res = config.half_res;
        }

        self.last_report = self
            .density_volume_atlas
            .as_ref()
            .map(|atlas| atlas.report.clone())
            .unwrap_or_else(|| self.default_report(config));
        Ok(())
    }

    pub fn current_report(&self) -> TerrainVolumetricsReport {
        self.last_report.clone()
    }

    /// Apply volumetric fog to the scene
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        heightmap_view: &wgpu::TextureView,
        heightmap: &[f32],
        height_dims: (u32, u32),
        terrain_revision: u64,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        inv_view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        near: f32,
        far: f32,
        sun_direction: [f32; 3],
        sun_intensity: f32,
        terrain_params: [f32; 4], // reference span, min_h, h_scale, h_range
        render_origin_span: [f32; 4],
        config: &super::pbr_renderer::VolumetricsConfig,
    ) -> anyhow::Result<()> {
        // Update uniforms with proper alignment
        let mode = match config.mode.as_str() {
            "uniform" => 0u32,
            "height" => 1u32,
            _ => 1u32,
        };

        self.ensure_fallback_density_volume()?;
        self.ensure_density_volume_atlas(
            queue,
            heightmap,
            height_dims,
            terrain_revision,
            terrain_params,
            render_origin_span,
            config,
        )?;

        let mut density_volume_min = [[0.0; 4]; MAX_DENSITY_VOLUMES];
        let mut density_volume_inv_size = [[0.0; 4]; MAX_DENSITY_VOLUMES];
        let mut density_volume_atlas_offset = [[0.0; 4]; MAX_DENSITY_VOLUMES];
        let mut density_volume_atlas_scale = [[0.0; 4]; MAX_DENSITY_VOLUMES];

        if let Some(atlas) = self.density_volume_atlas.as_ref() {
            for (index, metadata) in atlas.metadata.iter().enumerate() {
                density_volume_min[index] = [
                    metadata.min_corner[0],
                    metadata.min_corner[1],
                    metadata.min_corner[2],
                    0.0,
                ];
                density_volume_inv_size[index] = [
                    metadata.inv_size[0],
                    metadata.inv_size[1],
                    metadata.inv_size[2],
                    0.0,
                ];
                density_volume_atlas_offset[index] = [
                    metadata.atlas_offset[0],
                    metadata.atlas_offset[1],
                    metadata.atlas_offset[2],
                    0.0,
                ];
                density_volume_atlas_scale[index] = [
                    metadata.atlas_scale[0],
                    metadata.atlas_scale[1],
                    metadata.atlas_scale[2],
                    0.0,
                ];
            }
        }

        let uniforms = VolumetricsUniforms {
            inv_view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
            near_far: [near, far, config.density, config.height_falloff],
            scatter_absorb: [config.scattering, config.absorption, 0.0, 0.0],
            sun_direction: [
                sun_direction[0],
                sun_direction[1],
                sun_direction[2],
                sun_intensity,
            ],
            shaft_params: [
                config.shaft_intensity,
                if config.light_shafts { 1.0 } else { 0.0 },
                config.steps as f32,
                mode as f32,
            ],
            screen_dims: [width as f32, height as f32, 0.0, 0.0],
            terrain_params,
            render_origin_span,
            density_volume_count: [
                self.last_report.active_volume_count as f32,
                if self.last_report.active_volume_count > 0 {
                    1.0
                } else {
                    0.0
                },
                0.0,
                0.0,
            ],
            density_volume_min,
            density_volume_inv_size,
            density_volume_atlas_offset,
            density_volume_atlas_scale,
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let density_volume_view = self
            .density_volume_atlas
            .as_ref()
            .map(|atlas| &atlas.view)
            .unwrap_or_else(|| self.fallback_density_volume_view.as_ref().unwrap());
        let density_volume_sampler = self
            .density_volume_atlas
            .as_ref()
            .map(|atlas| &atlas.sampler)
            .unwrap_or(&self.density_volume_sampler);

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetrics.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.color_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(density_volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(density_volume_sampler),
                },
            ],
        });

        // Render pass - clear to black, shader samples from color_input and writes composited result
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("volumetrics.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);

        update_terrain_volumetrics_report(self.current_report());
        Ok(())
    }
}
