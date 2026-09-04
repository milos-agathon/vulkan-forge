// terrain_shadow_depth.wgsl
// Depth-only terrain rendering for CSM shadow passes
// Renders terrain heightmap as a tessellated grid from light's perspective

// Shadow pass uniforms (per-cascade)
// Size: 112 bytes - must match Rust struct exactly
struct ShadowPassUniforms {
    // Light view-projection matrix for this cascade (64 bytes)
    light_view_proj: mat4x4<f32>,
    // Terrain params: (terrain_span, z_scale, min_h, max_h) (16 bytes)
    terrain_params: vec4<f32>,
    // Grid params: (grid_resolution, _pad, _pad, _pad) (16 bytes)
    grid_params: vec4<f32>,
    // Height curve params: (mode, strength, power, _pad) (16 bytes)
    // mode: 0=linear, 1=pow, 2=smoothstep, 3=lut
    height_curve: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u_shadow: ShadowPassUniforms;

@group(0) @binding(1)
var height_tex: texture_2d<f32>;

@group(0) @binding(2)
var height_samp: sampler;

@group(0) @binding(3)
var height_curve_lut_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

fn height_curve_lut_sample(t: f32) -> f32 {
    let dims = textureDimensions(height_curve_lut_tex, 0);
    let max_x = max(i32(dims.x) - 1, 0);
    let u = clamp(t, 0.0, 1.0);
    let x = i32(round(u * f32(max_x)));
    return textureLoad(height_curve_lut_tex, vec2<i32>(x, 0), 0).r;
}

// Keep the shadow caster on the same portable R32Float sampling path as the
// visible terrain. R32Float is not filterable on every supported adapter, so
// reconstruct bilinear filtering explicitly instead of relying on a sampler.
fn sample_height_bilinear(uv: vec2<f32>) -> f32 {
    let dimensions = textureDimensions(height_tex, 0);
    let max_x = max(i32(dimensions.x) - 1, 0);
    let max_y = max(i32(dimensions.y) - 1, 0);
    let texel_x = clamp(uv.x, 0.0, 1.0) * f32(max_x);
    let texel_y = clamp(uv.y, 0.0, 1.0) * f32(max_y);
    let x0 = i32(floor(texel_x));
    let y0 = i32(floor(texel_y));
    let x1 = clamp(x0 + 1, 0, max_x);
    let y1 = clamp(y0 + 1, 0, max_y);
    let blend = clamp(
        vec2<f32>(texel_x - f32(x0), texel_y - f32(y0)),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );

    let h00 = textureLoad(height_tex, vec2<i32>(x0, y0), 0).r;
    let h10 = textureLoad(height_tex, vec2<i32>(x1, y0), 0).r;
    let h01 = textureLoad(height_tex, vec2<i32>(x0, y1), 0).r;
    let h11 = textureLoad(height_tex, vec2<i32>(x1, y1), 0).r;
    return det_mix(det_mix(h00, h10, blend.x), det_mix(h01, h11, blend.x), blend.y);
}

/// Apply height curve to normalized height value (matching main shader exactly)
/// t: input normalized height [0, 1]
/// Returns: curved normalized height [0, 1]
fn apply_height_curve(t: f32) -> f32 {
    let mode = u32(u_shadow.height_curve.x + 0.5);
    let strength = clamp(u_shadow.height_curve.y, 0.0, 1.0);

    if (strength <= 0.0) {
        return t;
    }

    var curved = t;
    if (mode == 1u) {
        let power = max(u_shadow.height_curve.z, 0.01);
        curved = det_pow(t, power);
    } else if (mode == 2u) {
        curved = t * t * (3.0 - 2.0 * t);
    } else if (mode == 3u) {
        curved = height_curve_lut_sample(t);
    }

    return det_mix(t, curved, strength);
}

/// Vertex shader for shadow depth pass
/// Uses vertex_index to generate a grid of vertices covering the terrain
@vertex
fn vs_shadow(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Extract parameters in the order uploaded by the native terrain renderer.
    let terrain_span = u_shadow.terrain_params.x;
    let z_scale = u_shadow.terrain_params.y;
    let height_min = u_shadow.terrain_params.z;
    let height_max = u_shadow.terrain_params.w;
    let grid_res = u32(u_shadow.grid_params.x);
    
    // Decode vertex position from index
    // We're rendering as triangles, 6 vertices per quad, (grid_res-1)^2 quads
    let quads_per_row = grid_res - 1u;
    
    // Which quad and which vertex within the quad
    let triangle_idx = vertex_id / 3u;
    let vertex_in_tri = vertex_id % 3u;
    let quad_idx = triangle_idx / 2u;
    let tri_in_quad = triangle_idx % 2u;
    
    let quad_x = quad_idx % quads_per_row;
    let quad_y = quad_idx / quads_per_row;
    
    // Vertex offsets for the two triangles in a quad
    // Triangle 0: (0,0), (1,0), (0,1)
    // Triangle 1: (1,0), (1,1), (0,1)
    var dx: u32;
    var dy: u32;
    if (tri_in_quad == 0u) {
        // First triangle
        if (vertex_in_tri == 0u) { dx = 0u; dy = 0u; }
        else if (vertex_in_tri == 1u) { dx = 1u; dy = 0u; }
        else { dx = 0u; dy = 1u; }
    } else {
        // Second triangle
        if (vertex_in_tri == 0u) { dx = 1u; dy = 0u; }
        else if (vertex_in_tri == 1u) { dx = 1u; dy = 1u; }
        else { dx = 0u; dy = 1u; }
    }
    
    let grid_x = quad_x + dx;
    let grid_y = quad_y + dy;
    
    // Convert grid position to UV [0,1]
    let uv = vec2<f32>(
        f32(grid_x) / f32(grid_res - 1u),
        f32(grid_y) / f32(grid_res - 1u)
    );
    
    let h_raw = sample_height_bilinear(uv);
    
    // Match terrain_pbr_pom.wgsl::normalize_for_shadow exactly.
    let world_xy = (uv - vec2<f32>(0.5)) * terrain_span;
    let height_range = max(height_max - height_min, 1e-6);
    let height_normalized = clamp((h_raw - height_min) / height_range, 0.0, 1.0);
    let world_z = apply_height_curve(height_normalized) * z_scale;
    let world_pos = vec3<f32>(world_xy, world_z);
    
    // Transform to light clip space
    out.clip_position = det_mat4_mul_vec4(
        u_shadow.light_view_proj,
        vec4<f32>(world_pos, 1.0),
    );
    
    return out;
}

/// Fragment shader for shadow depth pass
/// No color output - depth is written automatically
@fragment
fn fs_shadow() {
    // No-op: depth is written automatically by the rasterizer
}
