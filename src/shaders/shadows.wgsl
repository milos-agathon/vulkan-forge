// Cascaded Shadow Maps with PCF/PCSS/VSM/EVSM/MSM filtering
// Bind Groups and Layouts:
// - @group(2) Shadow resources
//   - @binding(0): uniform buffer `CsmUniforms`
//   - @binding(1): texture_depth_2d_array `shadow_maps` (Depth32Float)
//   - @binding(2): sampler_comparison `shadow_sampler`
//   - @binding(3): texture_2d_array<f32> `moment_maps` (Rgba16Float, for VSM/EVSM/MSM)
//   - @binding(4): sampler `moment_sampler` (filtering sampler for moment maps)
// Formats:
// - Depth maps: Depth32Float
// - Moment maps: Rgba16Float (VSM uses 2 channels, EVSM/MSM use 4)
// Address Space: `uniform`, `fragment`
// Provides high-quality shadows for directional lights with pluggable techniques

// Shadow cascade data
struct ShadowCascade {
    light_projection: mat4x4<f32>,  // Light-space projection matrix
    light_view_proj: mat4x4<f32>,   // Combined light_view_proj matrix
    near_distance: f32,             // Near plane distance
    far_distance: f32,              // Far plane distance
    texel_size: f32,                // Texel size in world space
    _padding: f32,
}

// CSM uniform data (must match Rust src/shadows/csm.rs::CsmUniforms)
struct CsmUniforms {
    light_direction: vec4<f32>,     // Light direction in world space
    light_view: mat4x4<f32>,        // Light view matrix
    cascades: array<ShadowCascade, 4>, // Shadow cascades (4 total)
    cascade_count: u32,             // Number of active cascades
    pcf_kernel_size: u32,           // PCF kernel size (1, 3, 5, or 7)
    depth_bias: f32,                // Depth bias to prevent acne
    slope_bias: f32,                // Slope-scaled bias
    shadow_map_size: f32,           // Shadow map resolution
    debug_mode: u32,                // Debug visualization mode
    evsm_positive_exp: f32,         // EVSM positive exponent
    evsm_negative_exp: f32,         // EVSM negative exponent
    peter_panning_offset: f32,      // Peter-panning prevention offset
    enable_unclipped_depth: u32,    // Enable unclipped depth (B17)
    depth_clip_factor: f32,         // Depth clipping distance factor
    technique: u32,                 // Active shadow technique (Hard=0, PCF=1, PCSS=2, VSM=3, EVSM=4, MSM=5)
    technique_flags: u32,           // Technique feature flags
    _pad1a: f32,
    _pad1b: f32,
    _pad1c: f32,
    // [blocker radius (texels), max filter radius (texels), moment bias, light radius (texels)]
    technique_params: vec4<f32>,
    technique_reserved: vec4<f32>,  // Reserved for future use
    cascade_blend_range: f32,       // Cascade blend range (0.0 = no blend, 0.1 = 10% blend)
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
    _pad2d: array<vec4<f32>, 6>,
}

// Bind group for shadow resources
@group(2) @binding(0) var<uniform> csm_uniforms: CsmUniforms;
@group(2) @binding(1) var shadow_maps: texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(3) var moment_maps: texture_2d_array<f32>;
@group(2) @binding(4) var moment_sampler: sampler;

// Convert world position to light space for cascade
fn world_to_light_space(world_pos: vec3<f32>, cascade_idx: u32) -> vec4<f32> {
    let light_space_pos = csm_uniforms.cascades[cascade_idx].light_view_proj * vec4<f32>(world_pos, 1.0);
    return light_space_pos;
}

// Select appropriate shadow cascade based on view depth
fn select_cascade(view_depth: f32) -> u32 {
    var cascade_idx = csm_uniforms.cascade_count - 1u;
    
    for (var i = 0u; i < csm_uniforms.cascade_count; i++) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            cascade_idx = i;
            break;
        }
    }
    
    return cascade_idx;
}

// Basic shadow sampling (single sample) - Hard shadows
fn sample_shadow_basic(light_space_pos: vec4<f32>, cascade_idx: u32) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check if position is within shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0; // Outside shadow map bounds - not in shadow
    }
    
    // Apply depth bias with clamping to prevent excessive peter-panning
    // Use per-cascade texel size for better bias scaling
    let cascade_texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    let max_bias = cascade_texel_size * 2.0; // Clamp bias to 2 texels
    let bias = min(csm_uniforms.depth_bias, max_bias);
    let biased_depth = clamp(shadow_coords.z - bias, 0.0, 1.0);
    
    // Sample shadow map with comparison
    return textureSampleCompare(shadow_maps, shadow_sampler, 
                               shadow_coords.xy, cascade_idx, biased_depth);
}

// PCF (Percentage-Closer Filtering) implementation
fn sample_shadow_pcf(light_space_pos: vec4<f32>, cascade_idx: u32, world_normal: vec3<f32>) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check if position is within shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0; // Outside shadow map bounds
    }
    
    // Calculate slope-scaled bias with proper clamping
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, -light_dir), 0.001); // Avoid division by zero
    
    // Clamp slope scale to prevent excessive bias at grazing angles
    let slope_scale = min(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / n_dot_l, 10.0);
    
    // Use per-cascade texel size for better bias scaling
    let cascade_texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    let max_bias = cascade_texel_size * 3.0; // Clamp total bias to 3 texels
    let bias = min(csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale, max_bias);
    let biased_depth = clamp(shadow_coords.z - bias, 0.0, 1.0);
    
    // PCF kernel size
    let kernel_size = i32(csm_uniforms.pcf_kernel_size);
    let half_kernel = kernel_size / 2;
    
    // Use per-cascade texel size in texture space
    let texel_size = 1.0 / csm_uniforms.shadow_map_size;
    
    // Accumulate shadow samples
    var shadow_factor = 0.0;
    var sample_count = 0.0;
    
    for (var x = -half_kernel; x <= half_kernel; x++) {
        for (var y = -half_kernel; y <= half_kernel; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coords = shadow_coords.xy + offset;
            
            // Bounds check for each sample
            if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 && 
                sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
                
                shadow_factor += textureSampleCompare(shadow_maps, shadow_sampler, 
                                                    sample_coords, cascade_idx, biased_depth);
                sample_count += 1.0;
            }
        }
    }
    
    return shadow_factor / sample_count;
}

// Advanced PCF with Poisson disk sampling for better quality
fn sample_shadow_poisson_pcf(light_space_pos: vec4<f32>, cascade_idx: u32, world_normal: vec3<f32>) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }
    
    // Calculate bias with proper clamping
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, -light_dir), 0.001); // Avoid division by zero
    
    // Clamp slope scale to prevent excessive bias at grazing angles
    let slope_scale = min(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / n_dot_l, 10.0);
    
    // Use per-cascade texel size for better bias scaling
    let cascade_texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    let max_bias = cascade_texel_size * 3.0; // Clamp total bias to 3 texels
    let bias = min(csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale, max_bias);
    let biased_depth = clamp(shadow_coords.z - bias, 0.0, 1.0);
    
    // Poisson disk samples for better distribution
    var poisson_disk = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216),
        vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870),
        vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432),
        vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845),
        vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554),
        vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023),
        vec2<f32>(0.79197514, 0.19090188),
        vec2<f32>(-0.24188840, 0.99706507),
        vec2<f32>(-0.81409955, 0.91437590),
        vec2<f32>(0.19984126, 0.78641367),
        vec2<f32>(0.14383161, -0.14100790)
    );
    
    // Sample using Poisson disk
    let texel_size = 1.0 / csm_uniforms.shadow_map_size;
    let filter_radius = f32(csm_uniforms.pcf_kernel_size) * texel_size * 0.5;
    
    var shadow_factor = 0.0;
    let sample_count = 16.0; // Using 16 Poisson samples
    
    for (var i = 0; i < 16; i++) {
        let offset = poisson_disk[i] * filter_radius;
        let sample_coords = shadow_coords.xy + offset;
        
        if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 && 
            sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
            
            shadow_factor += textureSampleCompare(shadow_maps, shadow_sampler, 
                                                sample_coords, cascade_idx, biased_depth);
        } else {
            shadow_factor += 1.0; // Outside bounds - not shadowed
        }
    }
    
    return shadow_factor / sample_count;
}

// PCSS: Search for blockers in the blocker search region
// Returns average blocker depth, or -1.0 if no blockers found
fn pcss_blocker_search(
    shadow_coords: vec2<f32>,
    receiver_depth: f32,
    cascade_idx: u32,
    search_radius: f32
) -> f32 {
    let layer_count = textureNumLayers(shadow_maps);
    if (cascade_idx >= layer_count) {
        return -1.0;
    }
    // Poisson disk for blocker search (using first 12 samples for performance)
    var poisson_disk = array<vec2<f32>, 12>(
        vec2<f32>(-0.94201624, -0.39906216),
        vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870),
        vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432),
        vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845),
        vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554),
        vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023),
        vec2<f32>(0.79197514, 0.19090188)
    );
    
    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    
    let texel_size = 1.0 / csm_uniforms.shadow_map_size;
    let scaled_search_radius = search_radius * texel_size;
    
    // Search for blockers
    for (var i = 0; i < 12; i++) {
        let offset = poisson_disk[i] * scaled_search_radius;
        let sample_coords = shadow_coords + offset;
        
        // Check bounds
        if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
            sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
            
            // Blocker search needs the stored depth, not a comparison result.
            let dimensions = textureDimensions(shadow_maps);
            let texel_coords = vec2<i32>(clamp(
                sample_coords * vec2<f32>(dimensions),
                vec2<f32>(0.0),
                vec2<f32>(dimensions) - vec2<f32>(1.0)
            ));
            let shadow_depth =
                textureLoad(shadow_maps, texel_coords, i32(cascade_idx), 0);
            
            // If this sample is closer than receiver (blocking)
            if (shadow_depth < receiver_depth) {
                blocker_sum += shadow_depth;
                blocker_count += 1.0;
            }
        }
    }
    
    // Return average blocker depth, or -1 if no blockers
    if (blocker_count > 0.0) {
        return blocker_sum / blocker_count;
    } else {
        return -1.0; // No blockers found
    }
}

// PCSS: Estimate penumbra size based on blocker distance
fn pcss_penumbra_size(
    receiver_depth: f32,
    blocker_depth: f32,
    light_size: f32
) -> f32 {
    // Penumbra estimation: (receiver - blocker) * light_size / blocker
    // Clamped to avoid extreme values
    let depth_diff = max(receiver_depth - blocker_depth, 0.0);
    let penumbra = (depth_diff * light_size) / max(blocker_depth, 0.001);
    
    // Clamp to reasonable range
    return clamp(penumbra, 0.0, 100.0);
}

// PCSS (Percentage-Closer Soft Shadows) implementation
fn sample_shadow_pcss(
    light_space_pos: vec4<f32>,
    cascade_idx: u32,
    world_normal: vec3<f32>
) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }
    
    // Calculate bias
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, -light_dir), 0.001);
    let slope_scale = min(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / n_dot_l, 10.0);
    
    let cascade_texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    let max_bias = cascade_texel_size * 3.0;
    let bias = min(csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale, max_bias);
    let receiver_depth = shadow_coords.z - bias;
    
    // Extract PCSS parameters
    let blocker_search_radius = csm_uniforms.technique_params.x;
    let base_filter_radius = csm_uniforms.technique_params.y;
    let light_size = csm_uniforms.technique_params.w;
    
    // PCSS radii are measured in shadow-map texels; cascade_texel_size is world-space.
    let clamped_blocker_radius = min(blocker_search_radius, 50.0);
    
    // Step 1: Blocker search
    let avg_blocker_depth = pcss_blocker_search(
        shadow_coords.xy,
        receiver_depth,
        cascade_idx,
        clamped_blocker_radius
    );
    
    // If no blockers found, fully lit
    if (avg_blocker_depth < 0.0) {
        return 1.0;
    }
    
    // Step 2: Penumbra estimation
    let penumbra = pcss_penumbra_size(receiver_depth, avg_blocker_depth, light_size);
    
    // Step 3: Adaptive PCF filter with penumbra-based radius
    let max_filter_radius = min(base_filter_radius, 100.0);
    let clamped_filter_radius = min(
        max(penumbra, min(max_filter_radius, 1.0)),
        max_filter_radius
    );
    
    // Use Poisson disk for final filtering
    var poisson_disk = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216),
        vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870),
        vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432),
        vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845),
        vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554),
        vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023),
        vec2<f32>(0.79197514, 0.19090188),
        vec2<f32>(-0.24188840, 0.99706507),
        vec2<f32>(-0.81409955, 0.91437590),
        vec2<f32>(0.19984126, 0.78641367),
        vec2<f32>(0.14383161, -0.14100790)
    );
    
    let texel_size = 1.0 / csm_uniforms.shadow_map_size;
    let scaled_filter_radius = clamped_filter_radius * texel_size;
    
    var shadow_factor = 0.0;
    let sample_count = 16.0;
    
    for (var i = 0; i < 16; i++) {
        let offset = poisson_disk[i] * scaled_filter_radius;
        let sample_coords = shadow_coords.xy + offset;
        
        if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
            sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
            
            shadow_factor += textureSampleCompare(
                shadow_maps,
                shadow_sampler,
                sample_coords,
                cascade_idx,
                clamp(receiver_depth, 0.0, 1.0)
            );
        } else {
            shadow_factor += 1.0;
        }
    }
    
    return shadow_factor / sample_count;
}

// Light leak reduction: reduce shadow factor near light sources
fn reduce_light_leak(shadow_factor: f32, amount: f32) -> f32 {
    // amount is typically moment_bias (technique_params[2])
    // Darken the shadow to reduce light leaks
    return clamp(shadow_factor - amount, 0.0, 1.0);
}

// VSM (Variance Shadow Maps) - 2 moments
fn sample_shadow_vsm(
    light_space_pos: vec4<f32>,
    cascade_idx: u32,
    world_normal: vec3<f32>
) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }
    
    // Calculate bias
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, -light_dir), 0.001);
    let slope_scale = min(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / n_dot_l, 10.0);
    
    let cascade_texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    let max_bias = cascade_texel_size * 3.0;
    let bias = min(csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale, max_bias);
    let receiver_depth = clamp(shadow_coords.z - bias, 0.0, 1.0);
    
    // Sample moment map (RG channels contain E[x] and E[x^2])
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords.xy, cascade_idx);
    let mean = moments.r;      // E[x]
    let mean_sq = moments.g;   // E[x^2]
    
    // If receiver is closer than mean, it is definitely lit.
    if (receiver_depth <= mean) {
        return 1.0;
    }
    
    // Calculate variance: Var(x) = E[x^2] - E[x]^2
    let variance = max(mean_sq - mean * mean, 0.0001); // Clamp to avoid division by zero
    
    // Apply Chebyshev inequality
    var shadow_factor = chebyshev_upper_bound_visibility(mean, variance, receiver_depth);
    
    // Apply light leak reduction
    let moment_bias = csm_uniforms.technique_params.z;
    if (moment_bias > 0.0) {
        shadow_factor = reduce_light_leak(shadow_factor, moment_bias);
    }
    
    return shadow_factor;
}

// EVSM (Exponential Variance Shadow Maps) - 4 moments with exponential warp
fn sample_shadow_evsm(
    light_space_pos: vec4<f32>,
    cascade_idx: u32,
    world_normal: vec3<f32>
) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }
    
    // Calculate bias
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, -light_dir), 0.001);
    let slope_scale = min(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / n_dot_l, 10.0);
    
    let cascade_texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    let max_bias = cascade_texel_size * 3.0;
    let bias = min(csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale, max_bias);
    let receiver_depth = clamp(shadow_coords.z - bias, 0.0, 1.0);
    
    // Sample moment map (RGBA channels)
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords.xy, cascade_idx);
    
    // EVSM uses exponential warp to reduce light leaking
    let c_pos = csm_uniforms.evsm_positive_exp;
    let c_neg = csm_uniforms.evsm_negative_exp;
    
    // Warp receiver depth
    let warp_depth_pos = exp(c_pos * (receiver_depth - 1.0));
    let warp_depth_neg = -exp(-c_neg * receiver_depth);
    let variance_floor = evsm_minimum_variance(
        vec2<f32>(warp_depth_pos, warp_depth_neg),
        vec2<f32>(c_pos, c_neg)
    );
    
    // Apply each lobe's front-of-mean shortcut independently, then combine.
    var shadow_factor =
        evsm_visibility_from_moments(
            moments,
            warp_depth_pos,
            warp_depth_neg,
            variance_floor
        );
    shadow_factor = min(
        shadow_factor,
        evsm_moment_leak_control(
            moments.rg,
            warp_depth_pos,
            c_pos,
            variance_floor.x
        )
    );
    
    // Apply light leak reduction
    let moment_bias = csm_uniforms.technique_params.z;
    if (moment_bias > 0.0) {
        shadow_factor = reduce_light_leak(shadow_factor, moment_bias);
    }
    
    return shadow_factor;
}

// MSM (Moment Shadow Maps) - 4 moments for better quality
fn sample_shadow_msm(
    light_space_pos: vec4<f32>,
    cascade_idx: u32,
    world_normal: vec3<f32>
) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }
    
    // Calculate bias
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(world_normal, -light_dir), 0.001);
    let slope_scale = min(sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / n_dot_l, 10.0);
    
    let cascade_texel_size = csm_uniforms.cascades[cascade_idx].texel_size;
    let max_bias = cascade_texel_size * 3.0;
    let bias = min(csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale, max_bias);
    let receiver_depth = clamp(shadow_coords.z - bias, 0.0, 1.0);
    
    // Sample moment map (4 moments in RGBA)
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords.xy, cascade_idx);
    
    let moment_bias = csm_uniforms.technique_params.z;
    return msm_visibility_from_moments(moments, receiver_depth, moment_bias);
}

// Main shadow calculation function with optional cascade blending
fn calculate_shadow(world_pos: vec3<f32>, view_depth: f32, world_normal: vec3<f32>) -> f32 {
    // Skip shadow calculation if light doesn't cast shadows
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }
    
    // Select appropriate cascade
    let cascade_idx = select_cascade(view_depth);
    
    // Transform to light space
    let light_space_pos = world_to_light_space(world_pos, cascade_idx);
    
    // Choose filtering method based on technique and kernel size
    var shadow_factor: f32;
    
    // Dispatch based on shadow technique
    if (csm_uniforms.technique == 3u) {
        // VSM (Variance Shadow Maps)
        shadow_factor = sample_shadow_vsm(light_space_pos, cascade_idx, world_normal);
    } else if (csm_uniforms.technique == 4u) {
        // EVSM (Exponential Variance Shadow Maps)
        shadow_factor = sample_shadow_evsm(light_space_pos, cascade_idx, world_normal);
    } else if (csm_uniforms.technique == 5u) {
        // MSM (Moment Shadow Maps)
        shadow_factor = sample_shadow_msm(light_space_pos, cascade_idx, world_normal);
    } else if (csm_uniforms.technique == 2u) {
        // PCSS (Percentage-Closer Soft Shadows)
        shadow_factor = sample_shadow_pcss(light_space_pos, cascade_idx, world_normal);
    } else if (csm_uniforms.pcf_kernel_size <= 1u) {
        // Hard shadows (no filtering)
        shadow_factor = sample_shadow_basic(light_space_pos, cascade_idx);
    } else if (csm_uniforms.pcf_kernel_size <= 5u) {
        // Standard PCF
        shadow_factor = sample_shadow_pcf(light_space_pos, cascade_idx, world_normal);
    } else {
        // High-quality Poisson PCF
        shadow_factor = sample_shadow_poisson_pcf(light_space_pos, cascade_idx, world_normal);
    }
    
    // Optional cascade blending at boundaries to reduce visible transitions
    if (csm_uniforms.cascade_blend_range > 0.0 && cascade_idx < csm_uniforms.cascade_count - 1u) {
        let current_far = csm_uniforms.cascades[cascade_idx].far_distance;
        let blend_start = current_far * (1.0 - csm_uniforms.cascade_blend_range);
        
        // Check if we're in the blend region
        if (view_depth > blend_start) {
            // Sample next cascade
            let next_light_space_pos = world_to_light_space(world_pos, cascade_idx + 1u);
            
            var next_shadow_factor: f32;
            if (csm_uniforms.technique == 3u) {
                next_shadow_factor = sample_shadow_vsm(next_light_space_pos, cascade_idx + 1u, world_normal);
            } else if (csm_uniforms.technique == 4u) {
                next_shadow_factor = sample_shadow_evsm(next_light_space_pos, cascade_idx + 1u, world_normal);
            } else if (csm_uniforms.technique == 5u) {
                next_shadow_factor = sample_shadow_msm(next_light_space_pos, cascade_idx + 1u, world_normal);
            } else if (csm_uniforms.technique == 2u) {
                next_shadow_factor = sample_shadow_pcss(next_light_space_pos, cascade_idx + 1u, world_normal);
            } else if (csm_uniforms.pcf_kernel_size <= 1u) {
                next_shadow_factor = sample_shadow_basic(next_light_space_pos, cascade_idx + 1u);
            } else if (csm_uniforms.pcf_kernel_size <= 5u) {
                next_shadow_factor = sample_shadow_pcf(next_light_space_pos, cascade_idx + 1u, world_normal);
            } else {
                next_shadow_factor = sample_shadow_poisson_pcf(next_light_space_pos, cascade_idx + 1u, world_normal);
            }
            
            // Blend between cascades based on depth
            let blend_factor = (view_depth - blend_start) / (current_far - blend_start);
            shadow_factor = mix(shadow_factor, next_shadow_factor, blend_factor);
        }
    }
    
    return shadow_factor;
}

// Debug visualization colors for cascades
fn get_cascade_debug_color(cascade_idx: u32) -> vec3<f32> {
    switch (cascade_idx) {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); } // Red
        case 1u: { return vec3<f32>(0.0, 1.0, 0.0); } // Green  
        case 2u: { return vec3<f32>(0.0, 0.0, 1.0); } // Blue
        case 3u: { return vec3<f32>(1.0, 1.0, 0.0); } // Yellow
        default: { return vec3<f32>(1.0, 0.0, 1.0); } // Magenta
    }
}

// Apply debug cascade visualization
fn apply_debug_visualization(base_color: vec3<f32>, world_pos: vec3<f32>, view_depth: f32) -> vec3<f32> {
    if (csm_uniforms.debug_mode == 0u) {
        return base_color;
    }
    
    let cascade_idx = select_cascade(view_depth);
    let debug_color = get_cascade_debug_color(cascade_idx);
    
    // Blend base color with cascade debug color
    return mix(base_color, debug_color, 0.3);
}

// Vertex shader for shadow map rendering
struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
}

struct ShadowVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn shadow_vs_main(input: ShadowVertexInput, @builtin(instance_index) cascade_idx: u32) -> ShadowVertexOutput {
    var out: ShadowVertexOutput;
    
    // Transform vertex to light space for current cascade
    out.clip_position = csm_uniforms.cascades[cascade_idx].light_view_proj * vec4<f32>(input.position, 1.0);
    
    return out;
}

// Fragment shader for shadow map rendering
@fragment
fn shadow_fs_main() -> @location(0) vec4<f32> {
    // Depth is written automatically, just return dummy color
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// Standard vertex shader that includes shadow calculation
struct StandardVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct StandardVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_depth: f32,
}

// Requires camera uniforms to be bound at group(0) binding(0)
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@vertex
fn standard_vs_main(input: StandardVertexInput) -> StandardVertexOutput {
    var out: StandardVertexOutput;
    
    // Transform to world space
    out.world_position = input.position; // Assuming input is already in world space
    out.world_normal = normalize(input.normal);
    out.uv = input.uv;
    
    // Transform to clip space
    out.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    
    // Calculate view depth for cascade selection
    let view_pos = camera.view * vec4<f32>(input.position, 1.0);
    out.view_depth = -view_pos.z; // Negative Z in view space
    
    return out;
}

// Standard fragment shader with shadow calculation
struct StandardFragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn standard_fs_main(input: StandardVertexOutput) -> StandardFragmentOutput {
    var out: StandardFragmentOutput;
    
    // Base material color (white for demonstration)
    var base_color = vec3<f32>(0.8, 0.8, 0.8);
    
    // Calculate lighting
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(input.world_normal, -light_dir), 0.0);
    
    // Calculate shadow
    let shadow_factor = calculate_shadow(input.world_position, input.view_depth, input.world_normal);
    
    // Apply lighting and shadows
    let ambient = vec3<f32>(0.1, 0.1, 0.1);
    let diffuse = base_color * n_dot_l * shadow_factor;
    let final_color = ambient + diffuse;
    
    // Apply debug visualization if enabled
    let debug_color = apply_debug_visualization(final_color, input.world_position, input.view_depth);
    
    out.color = vec4<f32>(debug_color, 1.0);
    return out;
}
