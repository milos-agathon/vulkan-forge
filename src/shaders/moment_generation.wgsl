// shaders/moment_generation.wgsl
// Generate moment maps (VSM/EVSM/MSM) from shadow depth maps
// This compute shader populates the moment atlas used by variance-based shadow techniques

@group(0) @binding(0) var depth_texture: texture_depth_2d_array;
@group(0) @binding(1) var moment_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: MomentGenParams;

struct MomentGenParams {
    technique: u32,           // 3=VSM, 4=EVSM, 5=MSM
    cascade_count: u32,
    evsm_positive_exp: f32,
    evsm_negative_exp: f32,
    shadow_map_size: u32,
    _padding: vec3<u32>,
}

const TECHNIQUE_VSM: u32 = 3u;
const TECHNIQUE_EVSM: u32 = 4u;
const TECHNIQUE_MSM: u32 = 5u;

// VSM: Compute mean and variance (2 moments)
fn generate_vsm_moments(depth: f32) -> vec4<f32> {
    let m1 = depth;           // E[x]
    let m2 = depth * depth;   // E[x²]
    return vec4<f32>(m1, m2, 0.0, 0.0);
}

// EVSM: Exponential warp for positive and negative
// Normalize both lobes into [-1, 1] so their squared moments cannot overflow
// Rgba16Float. The caller still clamps the exponent to preserve fp16 precision.
fn generate_evsm_moments(depth: f32, pos_exp: f32, neg_exp: f32) -> vec4<f32> {
    // Shift the positive warp so its largest value is one.
    let pos_warped = exp(pos_exp * (depth - 1.0));
    let pos_m1 = pos_warped;
    let pos_m2 = pos_warped * pos_warped;

    // Negative exponential warp. Stored NEGATED so the warped depth stays a
    // monotonically increasing function of depth - Chebyshev's upper bound is
    // only valid in that direction, and every consumer compares against
    // -exp(-c*d). Squaring uses the unnegated magnitude: E[x^2] = exp(-2c*d).
    let neg_warped = exp(-neg_exp * depth);
    let neg_m1 = -neg_warped;
    let neg_m2 = neg_warped * neg_warped;

    return vec4<f32>(pos_m1, pos_m2, neg_m1, neg_m2);
}

// MSM: Compute 4 moments for polynomial reconstruction
fn generate_msm_moments(depth: f32) -> vec4<f32> {
    let z = depth;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z3 * z;
    
    return vec4<f32>(z, z2, z3, z4);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = global_id.xy;
    let cascade_idx = global_id.z;
    
    // Bounds check
    if (coords.x >= params.shadow_map_size || 
        coords.y >= params.shadow_map_size ||
        cascade_idx >= params.cascade_count) {
        return;
    }
    
    // Sample depth from shadow map
    let depth = textureLoad(depth_texture, coords, cascade_idx, 0);
    
    // Generate moments based on technique
    var moments: vec4<f32>;
    
    switch (params.technique) {
        case TECHNIQUE_VSM: {
            moments = generate_vsm_moments(depth);
        }
        case TECHNIQUE_EVSM: {
            moments = generate_evsm_moments(
                depth,
                params.evsm_positive_exp,
                params.evsm_negative_exp
            );
        }
        case TECHNIQUE_MSM: {
            moments = generate_msm_moments(depth);
        }
        default: {
            // Fallback to VSM
            moments = generate_vsm_moments(depth);
        }
    }
    
    // Write moments to output texture
    textureStore(moment_texture, coords, cascade_idx, moments);
}
