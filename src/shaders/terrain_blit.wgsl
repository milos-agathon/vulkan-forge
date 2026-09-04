// src/shaders/terrain_blit.wgsl
// Fullscreen blit shader to resolve render_scale outputs
// Exists to resample offscreen terrain renders back to the requested viewport
// RELEVANT FILES: src/terrain_renderer.rs, src/shaders/terrain_pbr_pom.wgsl, src/terrain_render_params.rs, src/ibl_wrapper.rs

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_id : u32) -> VertexOutput {
    var out : VertexOutput;

    let uv_x = f32((vertex_id << 1u) & 2u);
    let uv_y = f32(vertex_id & 2u);
    let uv = vec2<f32>(uv_x, uv_y);

    out.clip_position = vec4<f32>(uv * vec2<f32>(2.0, 2.0) - vec2<f32>(1.0, 1.0), 0.0, 1.0);
    out.uv = uv * 0.5;
    return out;
}

@group(0) @binding(0)
var source_tex : texture_2d<f32>;

@group(0) @binding(1)
var source_sampler : sampler;

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source_tex, source_sampler, input.uv);
}
