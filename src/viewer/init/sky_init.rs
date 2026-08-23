// src/viewer/init/sky_init.rs
// Sky pipeline initialization for the Viewer

use std::sync::Arc;
use wgpu::{BindGroupLayout, ComputePipeline, Device, TextureView};

use crate::core::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

use super::super::viewer_types::SkyUniforms;

/// Storage, sampling, and night-overlay format shared by every viewer sky path.
pub const SKY_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Resources created during sky initialization
pub struct SkyResources {
    pub sky_bind_group_layout0: BindGroupLayout,
    pub sky_bind_group_layout1: BindGroupLayout,
    pub sky_pipeline: ComputePipeline,
    pub sky_params: TrackedBuffer,
    pub sky_camera: TrackedBuffer,
    pub sky_output: TrackedTexture,
    pub sky_output_view: TextureView,
    pub sky_present_bind_group_layout: BindGroupLayout,
    pub sky_present_depth_pipeline: wgpu::RenderPipeline,
    pub sky_present_flat_pipeline: wgpu::RenderPipeline,
    pub sky_present_sampler: wgpu::Sampler,
    pub night_pipeline: wgpu::RenderPipeline,
    pub night_instances: TrackedBuffer,
    pub night_bind_group: wgpu::BindGroup,
    pub night_moon_texture: TrackedTexture,
}

/// The one definition of the sky output texture.
///
/// `Viewer::resize_render_targets` recreates this texture, so init and resize
/// must build it from the same descriptor: the SIDERA night overlay draws into
/// `sky_output_view` as a colour attachment, and a resize that silently dropped
/// `RENDER_ATTACHMENT` would fail wgpu validation on the next frame.
pub fn sky_output_descriptor(width: u32, height: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("viewer.sky.output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SKY_OUTPUT_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    }
}

/// Create sky compute pipeline and resources
pub fn create_sky_resources(
    device: &Arc<Device>,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    target_format: wgpu::TextureFormat,
) -> RenderResult<SkyResources> {
    // Sky BGL0: params (binding 0) + output texture (binding 1)
    let sky_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.sky.bgl0"),
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
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: SKY_OUTPUT_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    // Sky BGL1: camera uniform (binding 0)
    let sky_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.sky.bgl1"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let sky_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.sky.shader",
        include_str!("../../shaders/sky.wgsl"),
    );

    let sky_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.sky.pl"),
        bind_group_layouts: &[&sky_bgl0, &sky_bgl1],
        push_constant_ranges: &[],
    });

    let sky_pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some("viewer.sky.pipeline"),
            layout: Some(&sky_pl),
            module: &sky_shader,
            entry_point: "cs_render_sky",
        },
    )
    .map_err(|message| RenderError::render(format!("viewer.sky.pipeline: {message}")))?;

    let sky_params_data = SkyUniforms::new([0.3, 0.8, -0.5], 2.0, 0.3, 1.0, 5.0, 1.0, 0);
    let sky_params = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sky.params"),
            contents: bytemuck::bytes_of(&sky_params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    // The first 272 bytes match CameraUniforms in sky.wgsl; stars.wgsl reads
    // the trailing viewport vec4 to build a resolution-invariant point splat.
    let sky_camera_data: [f32; 72] = [0.0; 72]; // 288 bytes
    let sky_camera = tracked_create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("viewer.sky.camera"),
            contents: bytemuck::cast_slice(&sky_camera_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    )?;

    // Sky output texture
    let sky_output = tracked_create_texture(device, &sky_output_descriptor(width, height))?;
    let sky_output_view = sky_output.create_view(&wgpu::TextureViewDescriptor::default());
    let sky_present_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer.sky.present.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
    let sky_present_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("viewer.sky.present.sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let sky_present_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.sky.present.shader",
        SKY_PRESENT_SHADER,
    );
    let sky_present_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.sky.present.pl"),
        bind_group_layouts: &[&sky_present_bind_group_layout],
        push_constant_ranges: &[],
    });
    let make_present_pipeline = |label: &'static str, depth_aware: bool| {
        crate::core::shader_registry::with_error_scope(device, label, || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&sky_present_layout),
                    vertex: wgpu::VertexState {
                        module: &sky_present_shader,
                        entry_point: "vs_present_sky",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &sky_present_shader,
                        entry_point: "fs_present_sky",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: target_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: sky_present_depth_state(depth_aware),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                },
            )
        })
    };
    let sky_present_depth_pipeline =
        make_present_pipeline("viewer.sky.present.depth_pipeline", true);
    let sky_present_flat_pipeline =
        make_present_pipeline("viewer.sky.present.flat_pipeline", false);
    let night =
        crate::astro::night_gpu::create_resources(device, queue, &sky_bgl1, SKY_OUTPUT_FORMAT)?;

    Ok(SkyResources {
        sky_bind_group_layout0: sky_bgl0,
        sky_bind_group_layout1: sky_bgl1,
        sky_pipeline,
        sky_params,
        sky_camera,
        sky_output,
        sky_output_view,
        sky_present_bind_group_layout,
        sky_present_depth_pipeline,
        sky_present_flat_pipeline,
        sky_present_sampler,
        night_pipeline: night.pipeline,
        night_instances: night.instances,
        night_bind_group: night.bind_group,
        night_moon_texture: night.moon_texture,
    })
}

const SKY_PRESENT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var sky_texture: texture_2d<f32>;
@group(0) @binding(1) var sky_sampler: sampler;

@vertex
fn vs_present_sky(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    var output: VertexOutput;
    // Far-plane depth plus LessEqual means this pass fills only pixels whose
    // terrain depth stayed at the clear value of 1.0.
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 1.0, 1.0);
    output.uv = vec2<f32>(x, 1.0 - y);
    return output;
}

@fragment
fn fs_present_sky(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(sky_texture, sky_sampler, input.uv);
}
"#;

fn sky_present_depth_state(depth_aware: bool) -> Option<wgpu::DepthStencilState> {
    depth_aware.then_some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        create_sky_resources, sky_output_descriptor, sky_present_depth_state, SKY_OUTPUT_FORMAT,
    };
    use std::sync::Arc;

    #[test]
    fn viewer_and_terrain_share_hdr_storage_format() {
        let shared = include_str!("../../shaders/sky.wgsl");
        assert_eq!(sky_output_descriptor(1, 1).format, SKY_OUTPUT_FORMAT);
        assert_eq!(
            shared
                .matches("texture_storage_2d<rgba16float, write>")
                .count(),
            1
        );
    }

    #[test]
    fn creates_sky_pipeline_when_adapter_available() {
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("No GPU adapter available, skipping viewer sky pipeline test");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        else {
            eprintln!("Could not request GPU device, skipping viewer sky pipeline test");
            return;
        };

        let device = Arc::new(device);
        let resources =
            create_sky_resources(&device, &queue, 16, 16, wgpu::TextureFormat::Rgba8Unorm)
                .expect("sky resources");

        let source_view = &resources.sky_output_view;
        let bind_group = resources
            .sky_present_depth_pipeline
            .get_bind_group_layout(0);
        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer.sky.present.test.bg"),
            layout: &bind_group,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&resources.sky_present_sampler),
                },
            ],
        });
        let target = crate::core::resource_tracker::tracked_create_texture(
            &device,
            &wgpu::TextureDescriptor {
                label: Some("viewer.sky.present.test.target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
        .expect("tracked test target");
        let depth = crate::core::resource_tracker::tracked_create_texture(
            &device,
            &wgpu::TextureDescriptor {
                label: Some("viewer.sky.present.test.depth"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        )
        .expect("tracked test depth");
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewer.sky.present.test.encoder"),
        });
        {
            let _clear_sky = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.sky.present.test.clear_sky"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: source_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        {
            let _clear_target = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.sky.present.test.clear_target"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
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
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.sky.present.test.present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&resources.sky_present_depth_pipeline);
            pass.set_bind_group(0, &sky_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        let pixels = crate::renderer::readback::read_texture_tight(
            &device,
            &queue,
            &target,
            (1, 1),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("read presented sky pixel");
        assert_eq!(&pixels[..4], &[255, 0, 0, 255]);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewer.sky.present.test.occluded.encoder"),
        });
        {
            let _clear_target = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.sky.present.test.occluded.clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.5),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewer.sky.present.test.occluded.present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&resources.sky_present_depth_pipeline);
            pass.set_bind_group(0, &sky_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        let pixels = crate::renderer::readback::read_texture_tight(
            &device,
            &queue,
            &target,
            (1, 1),
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("read terrain-covered pixel");
        assert_eq!(&pixels[..4], &[0, 0, 255, 255]);
    }

    #[test]
    fn terrain_sky_presenter_uses_far_depth_without_writing_it() {
        let state = sky_present_depth_state(true).expect("depth-aware presenter");
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
        assert!(!state.depth_write_enabled);
        assert!(sky_present_depth_state(false).is_none());
    }
}
