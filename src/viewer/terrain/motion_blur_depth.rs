use std::sync::Arc;

/// Conservative regular-Z coverage union for motion-blurred snapshots.
///
/// Each jittered sample contributes its depth with a `Less` test into one
/// texture cleared to 1.0. A pixel is therefore clear only when every sample
/// was clear, which is the mask a sharp post-resolve celestial sky requires.
pub(super) struct MotionBlurDepthAccumulator {
    device: Arc<wgpu::Device>,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl MotionBlurDepthAccumulator {
    pub(super) fn new(device: Arc<wgpu::Device>) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion_blur.depth_union.bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &device,
            "motion_blur.depth_union.shader",
            DEPTH_UNION_SHADER,
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion_blur.depth_union.pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("motion_blur.depth_union.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_union",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_union",
                    targets: &[],
                }),
                primitive: wgpu::PrimitiveState::default(),
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
        Self {
            device,
            layout,
            pipeline,
        }
    }

    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sample_depth: &wgpu::TextureView,
        union_depth: &wgpu::TextureView,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion_blur.depth_union.bg"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(sample_depth),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("motion_blur.depth_union.pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: union_depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

const DEPTH_UNION_SHADER: &str = r#"
@group(0) @binding(0) var sample_depth: texture_depth_2d;

@vertex
fn vs_union(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_union(@builtin(position) position: vec4<f32>) -> @builtin(frag_depth) f32 {
    let dimensions = textureDimensions(sample_depth);
    let coordinate = clamp(vec2<i32>(position.xy), vec2<i32>(0), vec2<i32>(dimensions) - 1);
    return textureLoad(sample_depth, coordinate, 0);
}
"#;

#[cfg(test)]
mod tests;
