//! B16: Dual-source OIT composition and fallback
//! Composition shader for dual-source OIT and WBOIT fallback integration

// Composition uniforms
struct DualSourceComposeUniforms {
    use_dual_source: u32,        // 1 if dual-source is active, 0 for WBOIT fallback
    tone_mapping_mode: u32,      // 0=none, 1=reinhard, 2=aces_approx
    exposure: f32,               // Exposure adjustment
    gamma: f32,                  // Gamma correction factor
}

@group(0) @binding(0)
var<uniform> compose_uniforms: DualSourceComposeUniforms;

// Dual-source OIT result (when available)
@group(0) @binding(1)
var dual_source_color: texture_2d<f32>;

// WBOIT fallback buffers
@group(0) @binding(2)
var wboit_color_accum: texture_2d<f32>;

@group(0) @binding(3)
var wboit_reveal_accum: texture_2d<f32>;

// Background/opaque scene color
@group(0) @binding(4)
var background_color: texture_2d<f32>;

@group(0) @binding(5)
var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle for composition
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;

    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, -y) * 0.5 + 0.5;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;

    // Sample background color
    let bg_color = textureSample(background_color, tex_sampler, uv);

    var final_color: vec4<f32>;

    if (compose_uniforms.use_dual_source != 0u) {
        // Use dual-source OIT result
        final_color = compose_dual_source(uv, bg_color);
    } else {
        // Use WBOIT fallback
        final_color = compose_wboit_fallback(uv, bg_color);
    }

    // Apply tone mapping and color correction
    final_color = apply_tone_mapping(final_color);

    return final_color;
}

// Compose dual-source OIT result with background
fn compose_dual_source(uv: vec2<f32>, bg_color: vec4<f32>) -> vec4<f32> {
    // Sample dual-source result
    let ds_color = textureSample(dual_source_color, tex_sampler, uv);

    // Dual-source OIT should already be properly blended
    // Just composite with background using standard alpha blending
    let src_alpha = ds_color.a;
    let dst_alpha = bg_color.a;

    // Standard over operator
    let final_rgb = ds_color.rgb + bg_color.rgb * (1.0 - src_alpha);
    let final_alpha = src_alpha + dst_alpha * (1.0 - src_alpha);

    return vec4<f32>(final_rgb, final_alpha);
}

// Compose WBOIT fallback result with background
fn compose_wboit_fallback(uv: vec2<f32>, bg_color: vec4<f32>) -> vec4<f32> {
    // Sample WBOIT accumulation buffers
    let accum_color = textureSample(wboit_color_accum, tex_sampler, uv);
    let reveal = textureSample(wboit_reveal_accum, tex_sampler, uv).r;

    // WBOIT resolve (similar to existing oit_compose.wgsl)
    let epsilon = 1e-5;

    var transparent_color: vec3<f32>;
    if (accum_color.a > epsilon) {
        transparent_color = accum_color.rgb / accum_color.a;
    } else {
        transparent_color = vec3<f32>(0.0);
    }

    // Calculate transparency alpha
    let transparent_alpha = 1.0 - reveal;

    // Composite with background
    let final_rgb = transparent_color * transparent_alpha + bg_color.rgb * (1.0 - transparent_alpha);
    let final_alpha = transparent_alpha + bg_color.a * (1.0 - transparent_alpha);

    return vec4<f32>(final_rgb, final_alpha);
}

// Apply tone mapping and color correction
fn apply_tone_mapping(color: vec4<f32>) -> vec4<f32> {
    var result = color.rgb;

    // Apply exposure
    result *= pow(2.0, compose_uniforms.exposure);

    // Apply tone mapping based on mode
    switch (compose_uniforms.tone_mapping_mode) {
        case 1u: {
            // Reinhard tone mapping
            result = reinhard_tonemap(result);
        }
        case 2u: {
            // ACES approximation
            result = aces_approx_tonemap(result);
        }
        default: {
            // No tone mapping, just clamp
            result = clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }

    // Apply gamma correction
    result = pow(result, vec3<f32>(1.0 / compose_uniforms.gamma));

    return vec4<f32>(result, color.a);
}

// Reinhard tone mapping
fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (1.0 + color);
}

// ACES approximation tone mapping
fn aces_approx_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Debug visualization fragment shader
@fragment
fn fs_debug(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;

    if (compose_uniforms.use_dual_source != 0u) {
        // Visualize dual-source OIT
        let ds_color = textureSample(dual_source_color, tex_sampler, uv);

        // Show dual-source blend quality
        let quality = ds_color.a;  // Use alpha as quality indicator
        let debug_color = mix(
            vec3<f32>(1.0, 0.0, 0.0),  // Red for low quality
            vec3<f32>(0.0, 1.0, 0.0),  // Green for high quality
            quality
        );

        return vec4<f32>(debug_color, 1.0);
    } else {
        // Visualize WBOIT fallback
        let accum_color = textureSample(wboit_color_accum, tex_sampler, uv);
        let reveal = textureSample(wboit_reveal_accum, tex_sampler, uv).r;

        // Show fragment count estimation
        let fragment_count = accum_color.a / max(accum_color.r + accum_color.g + accum_color.b, 0.001);
        let normalized_count = clamp(fragment_count / 10.0, 0.0, 1.0);  // Assume max 10 fragments

        let debug_color = mix(
            vec3<f32>(0.0, 0.0, 1.0),  // Blue for low fragment count
            vec3<f32>(1.0, 1.0, 0.0),  // Yellow for high fragment count
            normalized_count
        );

        return vec4<f32>(debug_color, 1.0);
    }
}

// Quality assessment for automatic fallback
fn assess_dual_source_quality(uv: vec2<f32>) -> f32 {
    if (compose_uniforms.use_dual_source != 0u) {
        let ds_color = textureSample(dual_source_color, tex_sampler, uv);

        // Simple quality metric based on alpha consistency
        let alpha_variance = abs(ds_color.a - 0.5) * 2.0;  // 0 = good, 1 = poor
        return 1.0 - alpha_variance;
    } else {
        // WBOIT quality based on reveal factor
        let reveal = textureSample(wboit_reveal_accum, tex_sampler, uv).r;
        return clamp(1.0 - reveal, 0.0, 1.0);  // Higher reveal = lower quality
    }
}

// Runtime switching between dual-source and WBOIT
@fragment
fn fs_adaptive(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let bg_color = textureSample(background_color, tex_sampler, uv);

    // Assess quality of both methods (if available)
    let dual_source_quality = assess_dual_source_quality(uv);

    var final_color: vec4<f32>;

    // Use dual-source if available and quality is good, otherwise fallback to WBOIT
    if (compose_uniforms.use_dual_source != 0u && dual_source_quality > 0.7) {
        final_color = compose_dual_source(uv, bg_color);
    } else {
        final_color = compose_wboit_fallback(uv, bg_color);
    }

    final_color = apply_tone_mapping(final_color);

    return final_color;
}
