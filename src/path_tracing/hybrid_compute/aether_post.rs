use super::*;
use crate::core::atmosphere::{
    tracked_lut_upload_bytes, AtmosphereLutHandle, AtmosphereLuts, LutData,
    ACCUMULATED_SCATTERING_LUT_SEMANTICS, AETHER_RADIOMETRIC_SCALE_MAX,
};
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrometheusAetherUniforms {
    dimensions_frames: [u32; 4],
    camera_origin_exposure: [f32; 4],
    camera_right_tan_half_fov: [f32; 4],
    camera_up_aspect: [f32; 4],
    camera_forward_ground: [f32; 4],
    sun_direction_intensity: [f32; 4],
    planet_radii_path: [f32; 4],
    mie_turbidity_scales: [f32; 4],
    lut_dimensions0: [u32; 4],
    lut_dimensions1: [u32; 4],
}

pub(super) struct AetherPostPass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: TrackedBuffer,
    uniform_template: PrometheusAetherUniforms,
    depth_copy_texture: TrackedTexture,
    visibility_copy_texture: TrackedTexture,
    _transmittance_texture: TrackedTexture,
    _scattering_texture: TrackedTexture,
    _aerial_texture: TrackedTexture,
    width: u32,
    height: u32,
    gpu_bytes: u64,
}

impl AetherPostPass {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lut_handle: &AtmosphereLutHandle,
        width: u32,
        height: u32,
        cam_origin: [f32; 3],
        cam_right: [f32; 3],
        cam_up: [f32; 3],
        cam_forward: [f32; 3],
        fov_y_radians: f32,
        exposure: f32,
        light_dir: [f32; 3],
        sun_intensity: f32,
        accum_buffer: &wgpu::Buffer,
        output_view: &wgpu::TextureView,
    ) -> Result<Self, RenderError> {
        let config = lut_handle.config();
        config
            .validate()
            .map_err(|error| RenderError::Render(format!("invalid AETHER PT settings: {error}")))?;
        let exposure = exposure.clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX);
        let sun_intensity = sun_intensity.clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX);
        let luts = lut_handle.luts();
        validate_luts(luts)?;

        let (transmittance_texture, transmittance_view) = upload_lut(
            device,
            queue,
            "hybrid-pt-aether-transmittance",
            &luts.transmittance,
            wgpu::TextureDimension::D2,
            wgpu::TextureViewDimension::D2,
        )?;
        let (scattering_texture, scattering_view) = upload_lut(
            device,
            queue,
            "hybrid-pt-aether-accumulated-scattering",
            &luts.multiple_scattering,
            wgpu::TextureDimension::D3,
            wgpu::TextureViewDimension::D3,
        )?;
        let (aerial_texture, aerial_view) = upload_lut(
            device,
            queue,
            "hybrid-pt-aether-aerial",
            &luts.aerial_perspective,
            wgpu::TextureDimension::D3,
            wgpu::TextureViewDimension::D3,
        )?;
        let depth_copy_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-aether-authoritative-depth-copy"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let depth_copy_view =
            depth_copy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let visibility_copy_texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("hybrid-pt-aether-authoritative-visibility-copy"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let visibility_copy_view =
            visibility_copy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let dims = luts.metadata.dimensions;
        let uniform_template = PrometheusAetherUniforms {
            dimensions_frames: [width, height, 0, 11],
            camera_origin_exposure: [cam_origin[0], cam_origin[1], cam_origin[2], exposure],
            camera_right_tan_half_fov: [
                cam_right[0],
                cam_right[1],
                cam_right[2],
                (0.5 * fov_y_radians).tan(),
            ],
            camera_up_aspect: [
                cam_up[0],
                cam_up[1],
                cam_up[2],
                width as f32 / height as f32,
            ],
            camera_forward_ground: [
                cam_forward[0],
                cam_forward[1],
                cam_forward[2],
                config.ground_albedo,
            ],
            sun_direction_intensity: [light_dir[0], light_dir[1], light_dir[2], sun_intensity],
            planet_radii_path: [
                config.bottom_radius_m,
                config.top_radius_m,
                config.max_aerial_distance_m,
                config.ozone_du,
            ],
            mie_turbidity_scales: [
                config.mie_g,
                config.turbidity,
                config.rayleigh_scale_height_m,
                config.mie_scale_height_m,
            ],
            lut_dimensions0: [
                dims.transmittance_mu,
                dims.transmittance_height,
                dims.scattering_height,
                dims.scattering_nu,
            ],
            lut_dimensions1: [
                dims.aerial_distance,
                dims.aerial_mu_view,
                dims.aerial_height,
                0,
            ],
        };
        let uniform_buffer = tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("hybrid-pt-aether-uniforms"),
                contents: bytemuck::bytes_of(&uniform_template),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hybrid-pt-aether-post-bgl"),
            entries: &[
                storage_buffer_entry(0),
                sampled_texture_entry(1, wgpu::TextureViewDimension::D2),
                sampled_texture_entry(2, wgpu::TextureViewDimension::D2),
                sampled_texture_entry(3, wgpu::TextureViewDimension::D3),
                sampled_texture_entry(4, wgpu::TextureViewDimension::D3),
                uniform_entry(5),
                storage_texture_entry(6),
                sampled_texture_entry(7, wgpu::TextureViewDimension::D2),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hybrid-pt-aether-post-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_copy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&scattering_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&aerial_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&visibility_copy_view),
                },
            ],
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "hybrid-pt-aether-post",
            &crate::shader_sources::prometheus_aerial(),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hybrid-pt-aether-post-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("hybrid-pt-aether-post-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        )
        .map_err(|error| {
            RenderError::Render(format!(
                "PROMETHEUS AETHER post pipeline validation failed: {error}"
            ))
        })?;

        let gpu_bytes = luts.transmittance.byte_size()
            + luts.multiple_scattering.byte_size()
            + luts.aerial_perspective.byte_size()
            + u64::from(width) * u64::from(height) * 8
            + uniform_buffer.size();
        Ok(Self {
            pipeline,
            bind_group,
            uniform_buffer,
            uniform_template,
            depth_copy_texture,
            visibility_copy_texture,
            _transmittance_texture: transmittance_texture,
            _scattering_texture: scattering_texture,
            _aerial_texture: aerial_texture,
            width,
            height,
            gpu_bytes,
        })
    }

    pub(super) fn gpu_bytes(&self) -> u64 {
        self.gpu_bytes
    }

    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        authoritative_depth: &wgpu::Texture,
        authoritative_visibility: &wgpu::Texture,
        frames: u32,
    ) {
        let mut uniforms = self.uniform_template;
        uniforms.dimensions_frames[2] = frames;
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: authoritative_depth,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.depth_copy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: authoritative_visibility,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.visibility_copy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hybrid-pt-aether-post-pass"),
            ..Default::default()
        });
        crate::core::shader_registry::record_shader_use("hybrid-pt-aether-post");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }
}

fn validate_luts(luts: &AtmosphereLuts) -> Result<(), RenderError> {
    if luts.metadata.storage_format != "rgba16float"
        || luts.metadata.scattering_lut_semantics != ACCUMULATED_SCATTERING_LUT_SEMANTICS
    {
        return Err(RenderError::Render(
            "PROMETHEUS AETHER requires rgba16float accumulated-scattering LUTs".into(),
        ));
    }
    let dims = luts.metadata.dimensions;
    let packed = dims
        .scattering_height
        .checked_mul(dims.scattering_nu)
        .ok_or_else(|| RenderError::Render("AETHER scattering depth overflow".into()))?;
    if luts.transmittance.dimensions != [dims.transmittance_mu, dims.transmittance_height, 1]
        || luts.multiple_scattering.dimensions
            != [dims.scattering_mu_view, dims.scattering_mu_sun, packed]
        || luts.aerial_perspective.dimensions
            != [
                dims.aerial_distance,
                dims.aerial_mu_view,
                dims.aerial_height,
            ]
    {
        return Err(RenderError::Render(
            "PROMETHEUS AETHER LUT dimensions do not match metadata".into(),
        ));
    }
    Ok(())
}

fn upload_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    data: &LutData,
    dimension: wgpu::TextureDimension,
    view_dimension: wgpu::TextureViewDimension,
) -> Result<(TrackedTexture, wgpu::TextureView), RenderError> {
    let [width, height, depth] = data.dimensions;
    let expected = u64::from(width) * u64::from(height) * u64::from(depth) * 8;
    let bytes = tracked_lut_upload_bytes(data, "hybrid-pt-aether.lut-upload-staging")?;
    if width == 0 || height == 0 || depth == 0 || bytes.as_slice().len() as u64 != expected {
        return Err(RenderError::Render(format!(
            "{label} has an invalid RGBA16F payload"
        )));
    }
    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes.as_slice(),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(view_dimension),
        ..Default::default()
    });
    Ok((texture, view))
}

fn storage_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampled_texture_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
