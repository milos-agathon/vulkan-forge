use super::night::{NightInstance, MAX_NIGHT_INSTANCES};
use crate::core::{
    error::{RenderError, RenderResult},
    resource_tracker::{
        tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture, TrackedBuffer,
        TrackedTexture,
    },
};
use std::{num::NonZeroU64, sync::Arc};

const MOON_DATA: &[u8] = include_bytes!("../../assets/astro/moon_albedo.bin");

pub struct NightGpuResources {
    pub pipeline: wgpu::RenderPipeline,
    pub instances: TrackedBuffer,
    pub bind_group: wgpu::BindGroup,
    pub moon_texture: TrackedTexture,
}

pub fn create_resources(
    device: &Arc<wgpu::Device>,
    queue: &wgpu::Queue,
    camera_layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> RenderResult<NightGpuResources> {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("astro.night.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<NightInstance>() as u64),
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "astro.night.shader",
        include_str!("../shaders/stars.wgsl"),
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("astro.night.pl"),
        bind_group_layouts: &[&layout, camera_layout],
        push_constant_ranges: &[],
    });
    let pipeline =
        crate::core::shader_registry::with_error_scope(device, "astro.night.pipeline", || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some("astro.night.pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_night",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_night",
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            )
        });
    let instances = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("astro.night.instances"),
            size: (MAX_NIGHT_INSTANCES * std::mem::size_of::<NightInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;
    let (moon_pixels, moon_width, moon_height) = moon_texture_data()?;
    let moon_texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some("astro.night.moon_albedo"),
            size: wgpu::Extent3d {
                width: moon_width,
                height: moon_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &moon_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        moon_pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(moon_width),
            rows_per_image: Some(moon_height),
        },
        wgpu::Extent3d {
            width: moon_width,
            height: moon_height,
            depth_or_array_layers: 1,
        },
    );
    let moon_view = moon_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("astro.night.moon_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("astro.night.bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: instances.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&moon_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    Ok(NightGpuResources {
        pipeline,
        instances,
        bind_group,
        moon_texture,
    })
}

pub fn render_test_frame(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    backend: wgpu::Backend,
    observation: &super::observation::ObservationSnapshot,
    width: u32,
    height: u32,
) -> RenderResult<TrackedTexture> {
    if width == 0 || height == 0 || width > 4_096 || height > 4_096 {
        return Err(RenderError::render(
            "night-sky dimensions must be within 1..=4096",
        ));
    }
    let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("astro.night.test.camera_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(288),
            },
            count: None,
        }],
    });
    let camera = fixed_camera(width, height);
    let camera_buffer = tracked_create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("astro.night.test.camera"),
            contents: bytemuck::bytes_of(&camera),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    )?;
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("astro.night.test.camera_bg"),
        layout: &camera_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });
    let night = create_resources(
        &device,
        &queue,
        &camera_layout,
        wgpu::TextureFormat::Rgba16Float,
    )?;
    let instances = super::night::instances(observation);
    queue.write_buffer(&night.instances, 0, bytemuck::cast_slice(&instances));
    let hdr_target = tracked_create_texture(
        &device,
        &wgpu::TextureDescriptor {
            label: Some("astro.night.test.hdr_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )?;
    let target_view = hdr_target.create_view(&wgpu::TextureViewDescriptor::default());
    let target = tracked_create_texture(
        &device,
        &wgpu::TextureDescriptor {
            label: Some("astro.night.test.sdr_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        },
    )?;
    let resolved_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut timing = night_golden_uses_timestamp_queries(backend)
        .then(|| crate::core::gpu_timing::OneShotTiming::for_device(device.clone(), queue.clone()));
    crate::core::shader_registry::record_shader_use("astro.night.shader");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("astro.night.test.encoder"),
    });
    let overlay_scope = timing
        .as_mut()
        .map(|timing| timing.begin(&mut encoder, "astro.night.overlay"));
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("astro.night.test.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.004,
                        g: 0.006,
                        b: 0.018,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&night.pipeline);
        pass.set_bind_group(0, &night.bind_group, &[]);
        pass.set_bind_group(1, &camera_bind_group, &[]);
        pass.draw(0..6, 0..instances.len() as u32);
    }
    if let Some(timing) = timing.as_mut() {
        timing.end(&mut encoder, overlay_scope.flatten(), 1);
    }
    let resolve_scope = timing
        .as_mut()
        .map(|timing| timing.begin(&mut encoder, "astro.night.resolve"));
    encode_sdr_resolve(&device, &mut encoder, &target_view, &resolved_view);
    if let Some(timing) = timing.as_mut() {
        timing.end(&mut encoder, resolve_scope.flatten(), 1);
        timing.resolve(&mut encoder);
    }
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    crate::core::certificate::record_model(
        "astro.twilight",
        "SIDERA civil-to-astronomical smoothstep; solar altitude -4 to -18 degrees; rendering visibility model",
    );
    crate::core::certificate::record_model(
        "astro.moonlight",
        "Krisciunas and Schaefer 1991 phase and extinction; 0.172 mag per airmass; 0.25 lux pre-extinction reference; live terrain term normalized by the modeled zenith full Moon after extinction (not applied to this sky-only golden)",
    );
    crate::core::certificate::record_model(
        "astro.star_photometry",
        "catalogue magnitude-to-irradiance ratio; resolution-invariant pixel-integrated tent splat in linear HDR; exposure 0.20; one Reinhard plus sRGB display resolve; no visibility floor",
    );
    crate::core::certificate::record_model(
        "astro.planet_display",
        "artistic fixed display intensities; no physical planetary photometry claim",
    );
    let timing_recorded = timing
        .take()
        .is_some_and(crate::core::gpu_timing::OneShotTiming::record_into_certificate);
    if !timing_recorded {
        if backend == wgpu::Backend::Metal {
            for label in ["astro.night.overlay", "astro.night.resolve"] {
                crate::core::degradation::record_degradation(
                    "timing_unavailable",
                    label,
                    "one-shot timestamp queries are disabled for the SIDERA golden path on Metal because synchronous query readback is unreliable; gpu_ms is reported as 0",
                );
            }
        }
        crate::core::certificate::record_pass("astro.night.overlay", 0.0, 1);
        crate::core::certificate::record_pass("astro.night.resolve", 0.0, 1);
    }
    Ok(target)
}

fn night_golden_uses_timestamp_queries(backend: wgpu::Backend) -> bool {
    backend != wgpu::Backend::Metal
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    inverse_view: [[f32; 4]; 4],
    inverse_projection: [[f32; 4]; 4],
    eye: [f32; 4],
    viewport: [f32; 4],
}

fn fixed_camera(width: u32, height: u32) -> CameraUniform {
    let direction = super::night::horizontal_direction(180.0, 80.0).as_vec3();
    let view = glam::Mat4::look_at_rh(glam::Vec3::ZERO, direction, glam::Vec3::Y);
    let projection = glam::Mat4::perspective_rh(
        30_f32.to_radians(),
        width as f32 / height as f32,
        0.01,
        10.0,
    );
    CameraUniform {
        view: view.to_cols_array_2d(),
        projection: projection.to_cols_array_2d(),
        inverse_view: view.inverse().to_cols_array_2d(),
        inverse_projection: projection.inverse().to_cols_array_2d(),
        eye: [0.0; 4],
        viewport: [
            width as f32,
            height as f32,
            1.0 / width as f32,
            1.0 / height as f32,
        ],
    }
}

fn encode_sdr_resolve(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    hdr_view: &wgpu::TextureView,
    output_view: &wgpu::TextureView,
) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("astro.night.resolve.bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "astro.night.resolve.shader",
        NIGHT_RESOLVE_SHADER,
    );
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("astro.night.resolve.pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some("astro.night.resolve.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_resolve",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_resolve",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        },
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("astro.night.resolve.bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(hdr_view),
        }],
    });
    crate::core::shader_registry::record_shader_use("astro.night.resolve.shader");
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("astro.night.resolve.pass"),
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
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}

const NIGHT_RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var hdr: texture_2d<f32>;

@vertex
fn vs_resolve(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, linear <= vec3<f32>(0.0031308));
}

@fragment
fn fs_resolve(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(hdr);
    let coordinate = clamp(vec2<i32>(position.xy), vec2<i32>(0), vec2<i32>(dimensions) - 1);
    let linear = max(textureLoad(hdr, coordinate, 0).rgb, vec3<f32>(0.0));
    let mapped = linear / (vec3<f32>(1.0) + linear);
    return vec4<f32>(linear_to_srgb(mapped), 1.0);
}
"#;

fn moon_texture_data() -> RenderResult<(&'static [u8], u32, u32)> {
    if MOON_DATA.len() < 16 || &MOON_DATA[..8] != b"F3DMAP01" {
        return Err(RenderError::render("invalid SIDERA Moon texture header"));
    }
    let width = u32::from_le_bytes(
        MOON_DATA[8..12]
            .try_into()
            .map_err(|_| RenderError::render("invalid SIDERA Moon texture width"))?,
    );
    let height = u32::from_le_bytes(
        MOON_DATA[12..16]
            .try_into()
            .map_err(|_| RenderError::render("invalid SIDERA Moon texture height"))?,
    );
    let pixels = &MOON_DATA[16..];
    if pixels.len() != width as usize * height as usize {
        return Err(RenderError::render("invalid SIDERA Moon texture length"));
    }
    Ok((pixels, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moon_texture_asset_is_exact() {
        let (pixels, width, height) = moon_texture_data().unwrap();
        assert_eq!((width, height, pixels.len()), (128, 64, 8_192));
        assert!(pixels.iter().copied().min().unwrap() < 130);
        assert!(pixels.iter().copied().max().unwrap() > 200);
    }

    #[test]
    fn metal_only_disables_one_shot_timing_for_the_night_golden() {
        assert!(!night_golden_uses_timestamp_queries(wgpu::Backend::Metal));
        assert!(night_golden_uses_timestamp_queries(wgpu::Backend::Vulkan));
        assert!(night_golden_uses_timestamp_queries(wgpu::Backend::Dx12));
    }
}
