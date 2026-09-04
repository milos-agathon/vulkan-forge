// src/shadows/moment_pass.rs
// Moment generation pass for VSM/EVSM/MSM shadow techniques
// Converts depth maps into moment statistics via compute shader

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor,
    Queue, ShaderStages, StorageTextureAccess, Texture, TextureFormat, TextureSampleType,
    TextureView, TextureViewDimension,
};

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use crate::lighting::types::ShadowTechnique;

/// Parameters for moment generation
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MomentGenParams {
    technique: u32,
    cascade_count: u32,
    evsm_positive_exp: f32,
    evsm_negative_exp: f32,
    shadow_map_size: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    // vec3<u32> in WGSL requires 16-byte alignment
    _padding3: [u32; 3],
    _padding4: u32,
}

#[derive(Clone, Copy)]
struct BoundTextureInfo {
    width: u32,
    height: u32,
    layers: u32,
    sample_count: u32,
    dimension: wgpu::TextureDimension,
    format: TextureFormat,
    usage: wgpu::TextureUsages,
}

impl BoundTextureInfo {
    fn from_texture(texture: &Texture) -> Self {
        Self {
            width: texture.width(),
            height: texture.height(),
            layers: texture.depth_or_array_layers(),
            sample_count: texture.sample_count(),
            dimension: texture.dimension(),
            format: texture.format(),
            usage: texture.usage(),
        }
    }
}

/// Moment generation compute pass
pub struct MomentGenerationPass {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    params_buffer: TrackedBuffer,
    bind_group: Option<BindGroup>,
    bound_textures: Option<(BoundTextureInfo, BoundTextureInfo)>,
}

impl MomentGenerationPass {
    /// Create a new moment generation pass
    pub fn new(device: &Device) -> RenderResult<Self> {
        // Load shader
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "moment_generation_shader",
            include_str!("../shaders/moment_generation.wgsl"),
        );

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("moment_gen_bind_group_layout"),
            entries: &[
                // Depth texture input (binding 0)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Moment texture output (binding 1)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                // Parameters uniform (binding 2)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("moment_gen_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("moment_gen_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        );

        // Create params buffer
        let params_buffer = tracked_create_buffer(
            device,
            &BufferDescriptor {
                label: Some("moment_gen_params"),
                size: std::mem::size_of::<MomentGenParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;

        Ok(Self {
            pipeline,
            bind_group_layout,
            params_buffer,
            bind_group: None,
            bound_textures: None,
        })
    }

    /// Prepare bind group for rendering
    pub fn prepare_bind_group(
        &mut self,
        device: &Device,
        depth_view: &TextureView,
        moment_view: &TextureView,
    ) {
        self.bound_textures = None;
        self.bind_group = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("moment_gen_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(depth_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(moment_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Validate textures and prepare a bind group for checked execution.
    ///
    /// Unlike [`Self::prepare_bind_group`], this rejects incompatible resources
    /// before creating any texture views or bind groups.
    pub fn prepare_textures_checked(
        &mut self,
        device: &Device,
        depth_texture: &Texture,
        moment_texture: &Texture,
    ) -> RenderResult<()> {
        let depth = BoundTextureInfo::from_texture(depth_texture);
        let moments = BoundTextureInfo::from_texture(moment_texture);
        validate_bound_textures(depth, moments)?;
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("moment_generation_depth_view"),
            format: Some(TextureFormat::Depth32Float),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(depth_texture.depth_or_array_layers()),
        });
        let moment_view =
            create_moment_storage_view(moment_texture, moment_texture.depth_or_array_layers());
        self.prepare_bind_group(device, &depth_view, &moment_view);
        self.bound_textures = Some((depth, moments));
        Ok(())
    }

    /// Update parameters and execute the compute pass.
    ///
    /// This is the legacy view-based API. Callers that own the textures should
    /// prefer [`Self::prepare_textures_checked`] and [`Self::execute_checked`].
    pub fn execute(
        &self,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        technique: ShadowTechnique,
        cascade_count: u32,
        shadow_map_size: u32,
        evsm_positive_exp: f32,
        evsm_negative_exp: f32,
    ) {
        self.encode(
            queue,
            encoder,
            technique,
            cascade_count,
            shadow_map_size,
            evsm_positive_exp,
            evsm_negative_exp,
        );
    }

    /// Validate the bound texture allocation and execute the compute pass.
    pub fn execute_checked(
        &self,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        technique: ShadowTechnique,
        cascade_count: u32,
        shadow_map_size: u32,
        evsm_positive_exp: f32,
        evsm_negative_exp: f32,
    ) -> RenderResult<()> {
        super::validate_shadow_dimensions(shadow_map_size, cascade_count)?;
        if !technique.requires_moments() {
            return Err(RenderError::render(format!(
                "{} does not produce moment shadows",
                technique.name()
            )));
        }
        let (depth, moments) = self
            .bound_textures
            .ok_or_else(|| RenderError::render("moment textures must be bound before execution"))?;
        let requested = (shadow_map_size, shadow_map_size, cascade_count);
        if (depth.width, depth.height, depth.layers) != requested
            || (moments.width, moments.height, moments.layers) != requested
        {
            return Err(RenderError::render(format!(
                "moment generation texture mismatch: depth={}x{}x{}, moments={}x{}x{}, requested={}x{}x{}",
                depth.width,
                depth.height,
                depth.layers,
                moments.width,
                moments.height,
                moments.layers,
                shadow_map_size,
                shadow_map_size,
                cascade_count
            )));
        }
        validate_bound_textures(depth, moments)?;
        self.encode(
            queue,
            encoder,
            technique,
            cascade_count,
            shadow_map_size,
            evsm_positive_exp,
            evsm_negative_exp,
        );
        Ok(())
    }

    fn encode(
        &self,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        technique: ShadowTechnique,
        cascade_count: u32,
        shadow_map_size: u32,
        evsm_positive_exp: f32,
        evsm_negative_exp: f32,
    ) {
        // Update parameters. Exponents are clamped to what the Rgba16Float moment
        // atlas can hold; the sampling side clamps identically.
        let params = MomentGenParams {
            technique: technique.as_u32(),
            cascade_count,
            evsm_positive_exp: super::clamp_evsm_exponent(evsm_positive_exp),
            evsm_negative_exp: super::clamp_evsm_exponent(evsm_negative_exp),
            shadow_map_size,
            _padding0: 0,
            _padding1: 0,
            _padding2: 0,
            _padding3: [0; 3],
            _padding4: 0,
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Execute compute pass
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("Bind group must be prepared before execution");

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("moment_generation_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);

        // Dispatch compute shader (8x8 workgroup size)
        let workgroup_size = 8;
        let dispatch_x = (shadow_map_size + workgroup_size - 1) / workgroup_size;
        let dispatch_y = (shadow_map_size + workgroup_size - 1) / workgroup_size;
        let dispatch_z = cascade_count;

        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }
}

fn validate_bound_textures(depth: BoundTextureInfo, moments: BoundTextureInfo) -> RenderResult<()> {
    super::validate_shadow_dimensions(depth.width, depth.layers)?;
    if depth.sample_count != 1 || moments.sample_count != 1 {
        return Err(RenderError::render(
            "moment generation textures must be single-sampled",
        ));
    }
    if depth.dimension != wgpu::TextureDimension::D2
        || moments.dimension != wgpu::TextureDimension::D2
    {
        return Err(RenderError::render(
            "moment generation textures must use the D2 texture dimension",
        ));
    }
    if depth.width != depth.height
        || moments.width != depth.width
        || moments.height != depth.height
        || moments.layers != depth.layers
    {
        return Err(RenderError::render(format!(
            "moment generation texture mismatch: depth={}x{}x{}, moments={}x{}x{}",
            depth.width, depth.height, depth.layers, moments.width, moments.height, moments.layers
        )));
    }
    if depth.format != TextureFormat::Depth32Float
        || !depth.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING)
    {
        return Err(RenderError::render(
            "moment generation depth texture must be sampleable Depth32Float",
        ));
    }
    let required_moment_usage =
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
    if moments.format != TextureFormat::Rgba16Float
        || !moments.usage.contains(required_moment_usage)
    {
        return Err(RenderError::render(
            "moment generation output must be sampleable and storage-bindable Rgba16Float",
        ));
    }
    Ok(())
}

/// Helper to create a storage texture view for moment generation output
pub fn create_moment_storage_view(moment_texture: &Texture, cascade_count: u32) -> TextureView {
    moment_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("moment_storage_view"),
        format: Some(TextureFormat::Rgba16Float),
        dimension: Some(TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(cascade_count),
    })
}

#[cfg(test)]
mod dimension_tests {
    use super::*;

    #[test]
    fn generation_and_blur_cover_a_1024_moment_atlas() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "1024 moment generation",
                include_str!("../shaders/moment_generation.wgsl"),
            );
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "1024 moment blur",
                include_str!("../shaders/shadow_blur.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;
        let mut config = crate::shadows::CsmConfig::default();
        config.shadow_map_size = 1024;
        config.cascade_count = 1;
        config.enable_evsm = true;
        let renderer = crate::shadows::CsmRenderer::new(device, config).expect("CSM renderer");
        let moments = renderer.evsm_maps.as_ref().expect("moment atlas");
        let depth_view = &renderer.shadow_map_views[0];
        let mut generation = MomentGenerationPass::new(device).expect("moment pass");
        generation
            .prepare_textures_checked(device, renderer.shadow_maps.as_ref(), moments)
            .expect("valid moment textures");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moment_1024_contract"),
        });
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("moment_1024_depth_clear"),
                color_attachments: &[],
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
        }
        generation
            .execute_checked(queue, &mut encoder, ShadowTechnique::VSM, 1, 1024, 9.0, 9.0)
            .expect("moment generation");
        queue.submit(Some(encoder.finish()));
        let generated = crate::core::hdr::read_hdr_texture(
            device,
            queue,
            moments,
            1024,
            1024,
            TextureFormat::Rgba16Float,
        )
        .expect("generated moment readback");
        assert!(generated[0] > 0.99, "1024 moment atlas generation failed");

        let mut blur = crate::shadows::ShadowBlurPass::new(device).expect("blur pass");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moment_1024_blur"),
        });
        blur.execute(
            device,
            queue,
            &mut encoder,
            moments,
            1,
            1024,
            crate::shadows::DEFAULT_MOMENT_BLUR_RADIUS,
            ShadowTechnique::VSM,
            9.0,
        )
        .expect("blur");
        queue.submit(Some(encoder.finish()));
        let output = crate::core::hdr::read_hdr_texture(
            device,
            queue,
            moments,
            1024,
            1024,
            TextureFormat::Rgba16Float,
        )
        .expect("moment readback");
        assert!(output[0] > 0.99, "1024 moment atlas was not populated");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(
        device: &Device,
        label: &'static str,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> crate::core::resource_tracker::TrackedTexture {
        texture_with_descriptor(
            device,
            label,
            width,
            height,
            1,
            wgpu::TextureDimension::D2,
            format,
            usage,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn texture_with_descriptor(
        device: &Device,
        label: &'static str,
        width: u32,
        height: u32,
        sample_count: u32,
        dimension: wgpu::TextureDimension,
        format: TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> crate::core::resource_tracker::TrackedTexture {
        crate::core::resource_tracker::tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension,
                format,
                usage,
                view_formats: &[],
            },
        )
        .expect("test texture")
    }

    #[test]
    fn test_moment_pass_creation() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "moment pass creation",
                include_str!("../shaders/moment_generation.wgsl"),
            );
            return;
        };
        let _pass = MomentGenerationPass::new(&device).expect("alloc");
        // Just verify it constructs without panicking
    }

    #[test]
    fn test_moment_params_size() {
        // Verify struct is properly aligned for GPU
        // WGSL vec3<u32> requires 16-byte alignment, making total size 48 bytes
        assert_eq!(
            std::mem::size_of::<MomentGenParams>(),
            48,
            "MomentGenParams must be 48 bytes (aligned for WGSL vec3)"
        );
    }

    #[test]
    fn execute_rejects_unbound_and_mismatched_allocations() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "moment allocation validation",
                include_str!("../shaders/moment_generation.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;
        let mut generation = MomentGenerationPass::new(device).expect("moment pass");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moment_generation_invalid"),
        });
        assert!(generation
            .execute_checked(queue, &mut encoder, ShadowTechnique::VSM, 1, 512, 9.0, 9.0,)
            .is_err());

        let mut config = crate::shadows::CsmConfig::default();
        config.shadow_map_size = 1024;
        config.cascade_count = 1;
        config.enable_evsm = true;
        let renderer = crate::shadows::CsmRenderer::new(device, config).expect("CSM renderer");
        generation
            .prepare_textures_checked(
                device,
                renderer.shadow_maps.as_ref(),
                renderer.evsm_maps.as_ref().expect("moment atlas"),
            )
            .expect("valid moment textures");
        assert!(generation
            .execute_checked(queue, &mut encoder, ShadowTechnique::VSM, 1, 512, 9.0, 9.0,)
            .is_err());
        assert!(generation
            .execute_checked(queue, &mut encoder, ShadowTechnique::PCF, 1, 1024, 9.0, 9.0,)
            .is_err());
    }

    #[test]
    fn checked_prepare_rejects_invalid_resources_before_view_creation() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "moment texture contract validation",
                include_str!("../shaders/moment_generation.wgsl"),
            );
            return;
        };
        let device = &device;
        let valid_depth = texture(
            device,
            "valid_depth",
            512,
            512,
            TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let valid_moments = texture(
            device,
            "valid_moments",
            512,
            512,
            TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let wrong_depth_format = texture(
            device,
            "wrong_depth_format",
            512,
            512,
            TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let incomplete_moment_usage = texture(
            device,
            "incomplete_moment_usage",
            512,
            512,
            TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        let wrong_extent = texture(
            device,
            "wrong_extent",
            1024,
            512,
            TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let multisampled_depth = texture_with_descriptor(
            device,
            "multisampled_depth",
            512,
            512,
            4,
            wgpu::TextureDimension::D2,
            TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let three_dimensional_moments = texture_with_descriptor(
            device,
            "three_dimensional_moments",
            512,
            512,
            1,
            wgpu::TextureDimension::D3,
            TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        let mut generation = MomentGenerationPass::new(device).expect("moment pass");

        assert!(generation
            .prepare_textures_checked(device, &wrong_depth_format, &valid_moments)
            .is_err());
        assert!(generation
            .prepare_textures_checked(device, &valid_depth, &incomplete_moment_usage)
            .is_err());
        assert!(generation
            .prepare_textures_checked(device, &valid_depth, &wrong_extent)
            .is_err());
        assert!(generation
            .prepare_textures_checked(device, &multisampled_depth, &valid_moments)
            .is_err());
        assert!(generation
            .prepare_textures_checked(device, &valid_depth, &three_dimensional_moments)
            .is_err());
    }

    #[test]
    fn legacy_view_prepare_and_execute_sequence_remains_compatible() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "legacy moment pass sequence",
                include_str!("../shaders/moment_generation.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;
        let mut config = crate::shadows::CsmConfig::default();
        config.shadow_map_size = 512;
        config.cascade_count = 1;
        config.enable_evsm = true;
        let renderer = crate::shadows::CsmRenderer::new(device, config).expect("CSM renderer");
        let moment_texture = renderer.evsm_maps.as_ref().expect("moment atlas");
        let moment_view = create_moment_storage_view(moment_texture, 1);
        let mut generation = MomentGenerationPass::new(device).expect("moment pass");
        generation.prepare_bind_group(device, &renderer.shadow_texture_view(), &moment_view);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("legacy_moment_api"),
        });

        generation.execute(queue, &mut encoder, ShadowTechnique::VSM, 1, 512, 9.0, 9.0);
        queue.submit(Some(encoder.finish()));
    }
}
