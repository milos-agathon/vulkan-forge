use super::MotionBlurDepthAccumulator;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use std::sync::Arc;

const SAMPLE_SHADER: &str = r#"
@vertex
fn vs_sample(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_left(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    return select(1.0, 0.25, position.x < 1.0);
}

@fragment
fn fs_right(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    return select(1.0, 0.50, position.x >= 1.0);
}
"#;

fn depth_texture(device: &wgpu::Device, label: &'static str) -> TrackedTexture {
    tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
    .expect("tracked depth texture")
}

fn sample_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry_point: &'static str,
) -> wgpu::RenderPipeline {
    crate::core::shader_registry::create_render_pipeline_scoped(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some(entry_point),
            layout: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_sample",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point,
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        },
    )
}

fn encode_depth_sample(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("motion_blur.depth_union.test.sample"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: target,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(pipeline);
    pass.draw(0..3, 0..1);
}

fn encode_present(
    encoder: &mut wgpu::CommandEncoder,
    output: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("motion_blur.depth_union.test.present"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

#[test]
fn union_blocks_sky_where_either_motion_sample_has_geometry() {
    let instance = wgpu::Instance::default();
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("No GPU adapter available, skipping motion depth union test");
        return;
    };
    let Ok((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
    else {
        eprintln!("Could not request GPU device, skipping motion depth union test");
        return;
    };
    let device = Arc::new(device);
    let sample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("motion_blur.depth_union.test.shader"),
        source: wgpu::ShaderSource::Wgsl(SAMPLE_SHADER.into()),
    });
    let left_pipeline = sample_pipeline(&device, &sample_shader, "fs_left");
    let right_pipeline = sample_pipeline(&device, &sample_shader, "fs_right");
    let left = depth_texture(&device, "motion_blur.depth_union.test.left");
    let right = depth_texture(&device, "motion_blur.depth_union.test.right");
    let union = depth_texture(&device, "motion_blur.depth_union.test.union");
    let left_view = left.create_view(&wgpu::TextureViewDescriptor::default());
    let right_view = right.create_view(&wgpu::TextureViewDescriptor::default());
    let union_view = union.create_view(&wgpu::TextureViewDescriptor::default());

    let sky = crate::viewer::init::create_sky_resources(
        &device,
        &queue,
        2,
        1,
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .expect("sky presenter resources");
    let present_layout = sky.sky_present_depth_pipeline.get_bind_group_layout(0);
    let present_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("motion_blur.depth_union.test.present_bg"),
        layout: &present_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&sky.sky_output_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sky.sky_present_sampler),
            },
        ],
    });
    let output = tracked_create_texture(
        &device,
        &wgpu::TextureDescriptor {
            label: Some("motion_blur.depth_union.test.output"),
            size: wgpu::Extent3d {
                width: 2,
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
    .expect("tracked output");
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

    let accumulator = MotionBlurDepthAccumulator::new(device.clone());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("motion_blur.depth_union.test.encoder"),
    });
    encode_depth_sample(&mut encoder, &left_view, &left_pipeline);
    encode_depth_sample(&mut encoder, &right_view, &right_pipeline);
    {
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("motion_blur.depth_union.test.clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &union_view,
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
        let _red = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("motion_blur.depth_union.test.red_sky"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &sky.sky_output_view,
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
    accumulator.encode(&mut encoder, &left_view, &union_view);
    accumulator.encode(&mut encoder, &right_view, &union_view);
    encode_present(
        &mut encoder,
        &output_view,
        &union_view,
        &sky.sky_present_depth_pipeline,
        &present_bg,
    );
    queue.submit(Some(encoder.finish()));
    let pixels = crate::renderer::readback::read_texture_tight(
        &device,
        &queue,
        &output,
        (2, 1),
        wgpu::TextureFormat::Rgba8Unorm,
    )
    .expect("read union composite");
    assert_eq!(&pixels[..8], &[0, 0, 255, 255, 0, 0, 255, 255]);
}
