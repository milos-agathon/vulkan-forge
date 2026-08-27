// src/viewer/terrain/post_process.rs
// Full-screen post-process pass for lens effects (distortion, CA, vignette)

use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};
use std::sync::Arc;

/// Post-process uniforms for lens effects
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessUniforms {
    /// Screen dimensions: width, height, 1/width, 1/height
    pub screen_dims: [f32; 4],
    /// Lens params: distortion, chromatic_aberration, vignette_strength, vignette_radius
    pub lens_params: [f32; 4],
    /// Lens params 2: vignette_softness, _, _, _
    pub lens_params2: [f32; 4],
}

/// Full-screen post-process pass manager
pub struct PostProcessPass {
    device: Arc<wgpu::Device>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: TrackedBuffer,
    // Intermediate texture for ping-pong rendering
    intermediate_texture: Option<TrackedTexture>,
    pub intermediate_view: Option<wgpu::TextureView>,
    current_size: (u32, u32),
    current_format: Option<wgpu::TextureFormat>,
}

impl PostProcessPass {
    pub fn new(
        device: Arc<wgpu::Device>,
        surface_format: wgpu::TextureFormat,
    ) -> anyhow::Result<Self> {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post_process.bind_group_layout"),
            entries: &[
                // Input texture
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
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post_process.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shader module
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &device,
            "post_process.shader",
            POST_PROCESS_SHADER,
        );

        // Create render pipeline
        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("post_process.pipeline"),
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
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        );

        // Create sampler (linear filtering with clamp)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post_process.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create uniform buffer
        let uniform_buffer = tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("post_process.uniforms"),
                contents: bytemuck::cast_slice(&[PostProcessUniforms {
                    screen_dims: [1.0, 1.0, 1.0, 1.0],
                    lens_params: [0.0, 0.0, 0.0, 0.7],
                    lens_params2: [0.3, 0.0, 0.0, 0.0],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            intermediate_texture: None,
            intermediate_view: None,
            current_size: (0, 0),
            current_format: None,
        })
    }

    /// Ensure intermediate texture is allocated at the correct size
    fn ensure_intermediate_texture(
        &mut self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> anyhow::Result<()> {
        if self.current_size != (width, height)
            || self.current_format != Some(format)
            || self.intermediate_texture.is_none()
        {
            let texture = tracked_create_texture(
                &self.device,
                &wgpu::TextureDescriptor {
                    label: Some("post_process.intermediate"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )?;
            self.intermediate_view =
                Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            self.intermediate_texture = Some(texture);
            self.current_size = (width, height);
            self.current_format = Some(format);
        }
        Ok(())
    }

    /// Get the intermediate texture view for rendering the scene to
    pub fn get_intermediate_view(
        &mut self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> anyhow::Result<&wgpu::TextureView> {
        self.ensure_intermediate_texture(width, height, format)?;
        Ok(self.intermediate_view.as_ref().unwrap())
    }

    pub fn apply_from_input(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        distortion: f32,
        chromatic_aberration: f32,
        vignette_strength: f32,
        vignette_radius: f32,
        vignette_softness: f32,
    ) {
        self.apply_from_input_internal(
            encoder,
            queue,
            input_view,
            output_view,
            width,
            height,
            distortion,
            chromatic_aberration,
            vignette_strength,
            vignette_radius,
            vignette_softness,
            false,
        );
    }

    /// Resolve a linear-HDR input into the display/output format. This is the
    /// single authoritative tonemap on the canonical-media viewer path.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_from_linear_hdr(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        distortion: f32,
        chromatic_aberration: f32,
        vignette_strength: f32,
        vignette_radius: f32,
        vignette_softness: f32,
    ) {
        self.apply_from_input_internal(
            encoder,
            queue,
            input_view,
            output_view,
            width,
            height,
            distortion,
            chromatic_aberration,
            vignette_strength,
            vignette_radius,
            vignette_softness,
            true,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_from_input_internal(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        distortion: f32,
        chromatic_aberration: f32,
        vignette_strength: f32,
        vignette_radius: f32,
        vignette_softness: f32,
        tonemap_linear_hdr: bool,
    ) {
        let uniforms = PostProcessUniforms {
            screen_dims: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            lens_params: [
                distortion,
                chromatic_aberration,
                vignette_strength,
                vignette_radius,
            ],
            lens_params2: [
                vignette_softness,
                if tonemap_linear_hdr { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post_process.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_process.render_pass"),
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
        }
    }

    /// Apply post-process effects and render to final target
    pub fn apply(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        distortion: f32,
        chromatic_aberration: f32,
        vignette_strength: f32,
        vignette_radius: f32,
        vignette_softness: f32,
    ) {
        // Update uniforms
        let uniforms = PostProcessUniforms {
            screen_dims: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            lens_params: [
                distortion,
                chromatic_aberration,
                vignette_strength,
                vignette_radius,
            ],
            lens_params2: [vignette_softness, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Create bind group with intermediate texture as input
        let input_view = self
            .intermediate_view
            .as_ref()
            .expect("Intermediate texture not allocated");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post_process.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Render post-process pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_process.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1); // Full-screen triangle
        }
    }
}

/// Post-process shader with full-screen triangle, distortion, CA, and vignette
const POST_PROCESS_SHADER: &str = r#"
// Post-process shader for lens effects

struct Uniforms {
    screen_dims: vec4<f32>,    // width, height, 1/width, 1/height
    lens_params: vec4<f32>,    // distortion, chromatic_aberration, vignette_strength, vignette_radius
    lens_params2: vec4<f32>,   // vignette_softness, tonemap_linear_hdr, _, _
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate full-screen triangle covering [-1,1] clip space
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV: (0,1) at top-left, (1,0) at bottom-right
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Barrel distortion (Brown-Conrady model, simplified)
fn apply_distortion(uv: vec2<f32>, k: f32) -> vec2<f32> {
    let centered = uv - 0.5;
    let r2 = dot(centered, centered);
    let factor = 1.0 + k * r2;
    return centered * factor + 0.5;
}

// Clamp UV to valid texture range
fn clamp_uv(uv: vec2<f32>) -> vec2<f32> {
    return clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let distortion = u.lens_params.x;
    let ca_strength = u.lens_params.y;
    let vignette_strength = u.lens_params.z;
    let vignette_radius = u.lens_params.w;
    let vignette_softness = u.lens_params2.x;
    
    var uv = in.uv;
    
    // Apply barrel distortion
    if abs(distortion) > 0.001 {
        uv = apply_distortion(uv, distortion);
    }
    
    var color: vec3<f32>;
    
    // Apply chromatic aberration (radial RGB split)
    if ca_strength > 0.001 {
        let centered = uv - 0.5;
        let r_uv = clamp_uv(centered * (1.0 + ca_strength) + 0.5);
        let g_uv = clamp_uv(uv);
        let b_uv = clamp_uv(centered * (1.0 - ca_strength) + 0.5);
        
        color = vec3<f32>(
            textureSample(input_tex, samp, r_uv).r,
            textureSample(input_tex, samp, g_uv).g,
            textureSample(input_tex, samp, b_uv).b
        );
    } else {
        color = textureSample(input_tex, samp, clamp_uv(uv)).rgb;
    }
    
    // Apply vignette
    if vignette_strength > 0.001 {
        let center_dist = length(in.uv - vec2<f32>(0.5));
        let vignette = 1.0 - smoothstep(
            vignette_radius - vignette_softness,
            vignette_radius + vignette_softness,
            center_dist * 2.0
        );
        color = color * mix(1.0, vignette, vignette_strength);
    }


    if u.lens_params2.y > 0.5 {
        color = pow(aces_tonemap(max(color, vec3<f32>(0.0))), vec3<f32>(1.0 / 2.2));
    }
    
    return vec4<f32>(color, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::POST_PROCESS_SHADER;

    #[test]
    fn canonical_media_resolve_contains_one_authoritative_tonemap_call() {
        assert!(POST_PROCESS_SHADER.contains("u.lens_params2.y > 0.5"));
        assert_eq!(
            POST_PROCESS_SHADER
                .matches("color = pow(aces_tonemap")
                .count(),
            1
        );
    }
}
