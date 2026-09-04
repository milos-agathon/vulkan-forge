// Linear-HDR AETHER sky presentation. The atmosphere shader never tonemaps;
// this display-only pass feeds the existing operator and then encodes sRGB.

struct AetherBlitVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32) -> AetherBlitVertexOutput {
    let uv = vec2<f32>(
        f32((vertex_id << 1u) & 2u),
        f32(vertex_id & 2u),
    );
    var out: AetherBlitVertexOutput;
    out.clip_position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // The oversize triangle's `uv` interpolates over [0,1] inside the
    // viewport. Halving it samples only one source quadrant and stretches the
    // below-horizon LUT over the frame (magenta on Metal).
    out.uv = uv;
    return out;
}

@group(0) @binding(0)
var aether_hdr_source: texture_2d<f32>;

@group(0) @binding(1)
var aether_hdr_sampler: sampler;

@fragment
fn fs_main(input: AetherBlitVertexOutput) -> @location(0) vec4<f32> {
    // Full-screen NDC is bottom-up while texture coordinates are top-down.
    // Preserve positive atmospheric elevation above the displayed horizon.
    let source_uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
    let linear_hdr = max(textureSample(aether_hdr_source, aether_hdr_sampler, source_uv).rgb, vec3<f32>(0.0));
    let display_linear = tonemap_apply_operator(
        linear_hdr,
        TONEMAP_OPERATOR_FILMIC_TERRAIN,
        11.2,
    );
    return vec4<f32>(linear_to_srgb(clamp(display_linear, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
