// src/shadows/blur_pass.rs
// P0.2/M3: Separable Gaussian blur pass for VSM/EVSM/MSM moment maps
// Applies two-pass blur (horizontal then vertical) to smooth moment statistics

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, Extent3d,
    PipelineLayoutDescriptor, Queue, ShaderStages, StorageTextureAccess, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension,
};

pub const DEFAULT_MOMENT_BLUR_RADIUS: u32 = 3;
pub const MAX_MOMENT_BLUR_RADIUS: u32 = 4;

/// Parameters for shadow blur pass
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BlurParams {
    direction: [f32; 2], // (1,0) for horizontal, (0,1) for vertical
    kernel_radius: u32,
    cascade_count: u32,
    texture_size: u32,
    technique: u32,
    evsm_positive_exp: f32,
    evsm_depth_sigma: f32,
    _padding: [u32; 4],
}

/// Shadow blur pass for VSM/EVSM/MSM moment maps
pub struct ShadowBlurPass {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    params_buffers: [TrackedBuffer; 2],
    // Intermediate texture for two-pass blur
    intermediate_texture: Option<TrackedTexture>,
    intermediate_view: Option<TextureView>,
    moment_view: Option<TextureView>,
    bind_groups: Option<[BindGroup; 2]>,
    current_atlas_id: Option<u64>,
    current_size: u32,
    current_cascades: u32,
}

impl ShadowBlurPass {
    /// Create a new shadow blur pass
    pub fn new(device: &Device) -> RenderResult<Self> {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "shadow_blur_shader",
            include_str!("../shaders/shadow_blur.wgsl"),
        );

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_blur_bind_group_layout"),
            entries: &[
                // Input texture (binding 0)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output texture (binding 1)
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

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_blur_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = crate::core::shader_registry::create_compute_pipeline_scoped(
            device,
            &ComputePipelineDescriptor {
                label: Some("shadow_blur_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "cs_blur",
            },
        );

        let create_params_buffer = |label| {
            tracked_create_buffer(
                device,
                &BufferDescriptor {
                    label: Some(label),
                    size: std::mem::size_of::<BlurParams>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        };
        let params_buffers = [
            create_params_buffer("shadow_blur_horizontal_params")?,
            create_params_buffer("shadow_blur_vertical_params")?,
        ];

        Ok(Self {
            pipeline,
            bind_group_layout,
            params_buffers,
            intermediate_texture: None,
            intermediate_view: None,
            moment_view: None,
            bind_groups: None,
            current_atlas_id: None,
            current_size: 0,
            current_cascades: 0,
        })
    }

    /// Ensure intermediate texture is allocated with correct size
    fn ensure_intermediate_texture(
        &mut self,
        device: &Device,
        size: u32,
        cascades: u32,
    ) -> RenderResult<()> {
        if self.current_size == size && self.current_cascades == cascades {
            return Ok(());
        }

        let texture = tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("shadow_blur_intermediate"),
                size: Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: cascades,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
        )?;

        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("shadow_blur_intermediate_view"),
            format: Some(TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(cascades),
        });

        // Bind groups retain both the old intermediate and moment views.
        // Release every dependent handle before dropping the tracked owner.
        self.bind_groups = None;
        self.intermediate_view = None;
        self.moment_view = None;
        self.current_atlas_id = None;
        self.intermediate_texture = Some(texture);
        self.intermediate_view = Some(view);
        self.current_size = size;
        self.current_cascades = cascades;
        Ok(())
    }

    /// Execute two-pass separable Gaussian blur on moment maps
    pub fn execute(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        moment_texture: &TrackedTexture,
        cascade_count: u32,
        shadow_map_size: u32,
        kernel_radius: u32,
        technique: crate::lighting::types::ShadowTechnique,
        evsm_positive_exp: f32,
    ) -> RenderResult<()> {
        if moment_texture.format() != TextureFormat::Rgba16Float {
            return Err(RenderError::render(
                "moment blur requires an Rgba16Float atlas",
            ));
        }
        if moment_texture.dimension() != TextureDimension::D2 {
            return Err(RenderError::render(
                "moment blur requires a D2 texture or D2 array atlas",
            ));
        }
        if moment_texture.sample_count() != 1 {
            return Err(RenderError::render(
                "moment blur requires a single-sampled atlas",
            ));
        }
        let required_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
        if !moment_texture.usage().contains(required_usage) {
            return Err(RenderError::render(
                "moment blur atlas requires TEXTURE_BINDING and STORAGE_BINDING usage",
            ));
        }
        if shadow_map_size == 0 || cascade_count == 0 {
            return Err(RenderError::render(
                "moment blur dimensions and cascade count must be nonzero",
            ));
        }
        if cascade_count > 4 {
            return Err(RenderError::render(
                "moment blur supports at most four cascades",
            ));
        }
        if kernel_radius > MAX_MOMENT_BLUR_RADIUS {
            return Err(RenderError::render(format!(
                "moment blur radius {kernel_radius} exceeds maximum {MAX_MOMENT_BLUR_RADIUS}"
            )));
        }
        if technique == crate::lighting::types::ShadowTechnique::EVSM
            && (!evsm_positive_exp.is_finite() || evsm_positive_exp <= 0.0)
        {
            return Err(RenderError::render(
                "EVSM moment blur exponent must be finite and greater than zero",
            ));
        }
        if moment_texture.width() != shadow_map_size
            || moment_texture.height() != shadow_map_size
            || moment_texture.depth_or_array_layers() != cascade_count
        {
            return Err(RenderError::render(format!(
                "moment blur atlas mismatch: texture={}x{}x{}, requested={}x{}x{}",
                moment_texture.width(),
                moment_texture.height(),
                moment_texture.depth_or_array_layers(),
                shadow_map_size,
                shadow_map_size,
                cascade_count
            )));
        }
        let evsm_positive_exp = super::clamp_evsm_exponent(evsm_positive_exp);

        self.ensure_intermediate_texture(device, shadow_map_size, cascade_count)?;
        self.ensure_bind_groups(device, moment_texture)?;

        // Pass 1: Horizontal blur (moment -> intermediate)
        self.execute_pass(
            queue,
            encoder,
            [1.0, 0.0], // Horizontal
            &self.params_buffers[0],
            0,
            kernel_radius,
            technique,
            evsm_positive_exp,
            cascade_count,
            shadow_map_size,
            "shadow_blur_horizontal",
        );

        // Pass 2: Vertical blur (intermediate -> moment)
        self.execute_pass(
            queue,
            encoder,
            [0.0, 1.0], // Vertical
            &self.params_buffers[1],
            1,
            kernel_radius,
            technique,
            evsm_positive_exp,
            cascade_count,
            shadow_map_size,
            "shadow_blur_vertical",
        );

        Ok(())
    }

    fn ensure_bind_groups(
        &mut self,
        device: &Device,
        moment_texture: &TrackedTexture,
    ) -> RenderResult<()> {
        let atlas_id = moment_texture.ledger_id();
        if self.current_atlas_id == Some(atlas_id) && self.bind_groups.is_some() {
            return Ok(());
        }

        let moment_view = moment_texture.create_view(&TextureViewDescriptor {
            label: Some("shadow_blur_moment_view"),
            format: Some(TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(self.current_cascades),
        });
        let intermediate_view = self
            .intermediate_view
            .as_ref()
            .ok_or_else(|| RenderError::render("moment blur intermediate is unavailable"))?;
        let make_bind_group = |label, input: &TextureView, output: &TextureView, index: usize| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(input),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(output),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffers[index].as_entire_binding(),
                    },
                ],
            })
        };
        let horizontal =
            make_bind_group("shadow_blur_horizontal", &moment_view, intermediate_view, 0);
        let vertical = make_bind_group("shadow_blur_vertical", intermediate_view, &moment_view, 1);
        self.moment_view = Some(moment_view);
        self.bind_groups = Some([horizontal, vertical]);
        self.current_atlas_id = Some(atlas_id);
        Ok(())
    }

    fn execute_pass(
        &self,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        direction: [f32; 2],
        params_buffer: &TrackedBuffer,
        bind_group_index: usize,
        kernel_radius: u32,
        technique: crate::lighting::types::ShadowTechnique,
        evsm_positive_exp: f32,
        cascade_count: u32,
        texture_size: u32,
        label: &str,
    ) {
        // Update parameters
        let params = BlurParams {
            direction,
            kernel_radius,
            cascade_count,
            texture_size,
            technique: technique.as_u32(),
            evsm_positive_exp,
            evsm_depth_sigma: 0.01,
            _padding: [0; 4],
        };
        queue.write_buffer(params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Dispatch compute shader
        let workgroup_size = 8;
        let dispatch_x = (texture_size + workgroup_size - 1) / workgroup_size;
        let dispatch_y = (texture_size + workgroup_size - 1) / workgroup_size;

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(
            0,
            &self.bind_groups.as_ref().expect("bind groups prepared")[bind_group_index],
            &[],
        );
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, cascade_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_moment_texture(device: &Device, size: u32, cascades: u32) -> TrackedTexture {
        tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("shadow_blur_test_moments"),
                size: Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: cascades,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
        .expect("moment texture")
    }

    fn test_texture(
        device: &Device,
        format: TextureFormat,
        usage: TextureUsages,
        dimension: TextureDimension,
        sample_count: u32,
    ) -> TrackedTexture {
        tracked_create_texture(
            device,
            &TextureDescriptor {
                label: Some("shadow_blur_invalid_moments"),
                size: Extent3d {
                    width: 4,
                    height: if dimension == TextureDimension::D1 {
                        1
                    } else {
                        4
                    },
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
        .expect("invalid-contract texture")
    }

    #[test]
    fn test_blur_params_size() {
        assert_eq!(
            std::mem::size_of::<BlurParams>(),
            48,
            "BlurParams must match WGSL's 48-byte uniform layout"
        );
    }

    #[test]
    fn intermediate_texture_is_cached_and_resized_with_the_atlas() {
        let Some(device) = crate::core::gpu::create_device_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "shadow blur intermediate lifecycle",
                include_str!("../shaders/shadow_blur.wgsl"),
            );
            return;
        };
        let device = &device;
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");

        blur.ensure_intermediate_texture(device, 9, 1)
            .expect("initial intermediate");
        let initial_id = blur
            .intermediate_texture
            .as_ref()
            .expect("intermediate texture")
            .ledger_id();

        blur.ensure_intermediate_texture(device, 9, 1)
            .expect("cached intermediate");
        assert_eq!(
            blur.intermediate_texture
                .as_ref()
                .expect("intermediate texture")
                .ledger_id(),
            initial_id,
            "unchanged atlas dimensions must reuse the intermediate texture"
        );

        blur.ensure_intermediate_texture(device, 13, 2)
            .expect("resized intermediate");
        assert_ne!(
            blur.intermediate_texture
                .as_ref()
                .expect("intermediate texture")
                .ledger_id(),
            initial_id,
            "atlas resize must recreate the intermediate texture"
        );
        assert_eq!((blur.current_size, blur.current_cascades), (13, 2));
    }

    #[test]
    fn separable_blur_filters_both_axes() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "separable shadow blur",
                include_str!("../shaders/shadow_blur.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;

        let size = 9u32;
        let texture = test_moment_texture(device, size, 1);
        let mut values = vec![0.0f32; (size * size * 4) as usize];
        let center = ((size / 2 * size + size / 2) * 4) as usize;
        values[center..center + 4].fill(1.0);
        let bytes = values
            .into_iter()
            .flat_map(|value| half::f16::from_f32(value).to_le_bytes())
            .collect::<Vec<_>>();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(size * 8),
                rows_per_image: Some(size),
            },
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");
        device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("blur pipeline raised a GPU validation error: {error}");
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shadow_blur_test"),
        });
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        blur.execute(
            device,
            queue,
            &mut encoder,
            &texture,
            1,
            size,
            2,
            crate::lighting::types::ShadowTechnique::VSM,
            crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
        )
        .expect("blur execute");
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            panic!("blur dispatch raised a GPU validation error: {error}");
        }

        let output = crate::core::hdr::read_hdr_texture(
            device,
            queue,
            &texture,
            size,
            size,
            TextureFormat::Rgba16Float,
        )
        .expect("read blurred moments");
        let sample = |x: u32, y: u32| output[((y * size + x) * 4) as usize];
        let center_value = sample(size / 2, size / 2);
        let horizontal = sample(size / 2 + 1, size / 2);
        let vertical = sample(size / 2, size / 2 + 1);
        println!(
            "blur impulse center={center_value}, horizontal={horizontal}, vertical={vertical}"
        );
        assert!(
            horizontal > 0.08,
            "horizontal pass was not applied: {horizontal}"
        );
        assert!(vertical > 0.08, "vertical pass was not applied: {vertical}");
        assert!(
            (horizontal - vertical).abs() < 0.01,
            "separable blur is anisotropic: horizontal={horizontal}, vertical={vertical}"
        );
    }

    #[test]
    fn execute_rejects_invalid_or_mismatched_atlas_dimensions() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "shadow blur dimension validation",
                include_str!("../shaders/shadow_blur.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;
        let texture = test_moment_texture(device, 9, 1);
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");

        for (cascades, size, radius) in [(0, 9, 2), (1, 0, 2), (5, 9, 2), (1, 9, 5)] {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_blur_invalid"),
            });
            assert!(
                blur.execute(
                    device,
                    queue,
                    &mut encoder,
                    &texture,
                    cascades,
                    size,
                    radius,
                    crate::lighting::types::ShadowTechnique::VSM,
                    crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
                )
                .is_err(),
                "invalid ({cascades}, {size}, {radius}) was accepted"
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shadow_blur_mismatch"),
        });
        assert!(blur
            .execute(
                device,
                queue,
                &mut encoder,
                &texture,
                1,
                8,
                2,
                crate::lighting::types::ShadowTechnique::VSM,
                crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            )
            .is_err());
    }

    #[test]
    fn execute_rejects_invalid_texture_contracts_before_view_creation() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "shadow blur texture contract validation",
                include_str!("../shaders/shadow_blur.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");
        let cases = [
            test_texture(
                device,
                TextureFormat::Rgba8Unorm,
                TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                TextureDimension::D2,
                1,
            ),
            test_texture(
                device,
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING,
                TextureDimension::D2,
                1,
            ),
            test_texture(
                device,
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING,
                TextureDimension::D1,
                1,
            ),
            test_texture(
                device,
                TextureFormat::Rgba16Float,
                TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                TextureDimension::D2,
                4,
            ),
        ];

        for texture in &cases {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_blur_invalid_contract"),
            });
            assert!(blur
                .execute(
                    device,
                    queue,
                    &mut encoder,
                    texture,
                    1,
                    4,
                    2,
                    crate::lighting::types::ShadowTechnique::VSM,
                    crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
                )
                .is_err());
        }
    }

    #[test]
    fn execute_rejects_invalid_evsm_exponents() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "shadow blur EVSM exponent validation",
                include_str!("../shaders/shadow_blur.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;
        let texture = test_moment_texture(device, 9, 1);
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");

        for exponent in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_blur_invalid_evsm_exponent"),
            });
            assert!(
                blur.execute(
                    device,
                    queue,
                    &mut encoder,
                    &texture,
                    1,
                    9,
                    2,
                    crate::lighting::types::ShadowTechnique::EVSM,
                    exponent,
                )
                .is_err(),
                "invalid EVSM exponent {exponent:?} was accepted"
            );
        }
    }

    #[test]
    fn bind_groups_follow_atlas_resource_identity() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            crate::shader_sources::assert_valid_wgsl_without_gpu(
                "shadow blur atlas identity",
                include_str!("../shaders/shadow_blur.wgsl"),
            );
            return;
        };
        let device = &device;
        let queue = &queue;
        let first = test_moment_texture(device, 9, 1);
        let second = test_moment_texture(device, 9, 1);
        let mut blur = ShadowBlurPass::new(device).expect("blur pass");

        let execute = |blur: &mut ShadowBlurPass, texture: &TrackedTexture| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow_blur_identity"),
            });
            blur.execute(
                device,
                queue,
                &mut encoder,
                texture,
                1,
                9,
                2,
                crate::lighting::types::ShadowTechnique::VSM,
                crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            )
            .expect("blur execute");
        };
        execute(&mut blur, &first);
        assert_eq!(blur.current_atlas_id, Some(first.ledger_id()));
        execute(&mut blur, &second);
        assert_eq!(blur.current_atlas_id, Some(second.ledger_id()));
    }
}
