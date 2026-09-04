
// Terrain PBR Viewer Shader
// Enhanced lighting with Blinn-Phong, soft shadows, and improved materials

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    terrain_params: vec4<f32>,  // min_h, h_range, terrain_width, z_scale
    lighting: vec4<f32>,        // sun_intensity, ambient, shadow_intensity, water_level
    background: vec4<f32>,      // r, g, b, _
    water_color: vec4<f32>,     // r, g, b, _
    // PBR params
    pbr_params: vec4<f32>,      // exposure, normal_strength, ibl_intensity, overlay_preserve_colors
    ibl_params: vec4<f32>,      // use_hdri (>0.5), specular_max_mip, sin_theta, cos_theta
    camera_pos: vec4<f32>,      // camera world position
    // Lens effects: vignette_strength, vignette_radius, vignette_softness, _
    lens_params: vec4<f32>,
    // Screen dimensions for UV calculation
    screen_dims: vec4<f32>,     // width, height, _, _
    // Overlay params: enabled (>0.5), opacity, blend_mode (0=normal, 1=multiply, 2=overlay), solid (>0.5)
    overlay_params: vec4<f32>,
    render_origin_xz: vec2<f32>, // append-only ABI offset 240
    render_span_xz: vec2<f32>,   // append-only ABI offset 248
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var heightmap: texture_2d<f32>;
@group(0) @binding(2) var height_sampler: sampler;
@group(0) @binding(3) var height_ao_tex: texture_2d<f32>;
@group(0) @binding(4) var sun_vis_tex: texture_2d<f32>;
// Overlay texture and sampler for lit draped overlays
@group(0) @binding(5) var overlay_tex: texture_2d<f32>;
@group(0) @binding(6) var overlay_sampler: sampler;

// P6.2: CSM shadow bindings
@group(0) @binding(7) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(8) var shadow_sampler: sampler_comparison;
@group(0) @binding(9) var moment_maps: texture_2d_array<f32>;
@group(0) @binding(10) var moment_sampler: sampler;
@group(0) @binding(11) var<storage, read> csm_uniforms: CsmUniforms;
@group(0) @binding(12) var envSpecular: texture_cube<f32>;
@group(0) @binding(13) var envIrradiance: texture_cube<f32>;
@group(0) @binding(14) var envSampler: sampler;
@group(0) @binding(15) var brdfLUT: texture_2d<f32>;

// Shadow cascade data (matches Rust CsmCascadeData: 144 bytes)
struct ShadowCascade {
    light_projection: mat4x4<f32>,  // 64 bytes
    light_view_proj: mat4x4<f32>,   // 64 bytes
    near_distance: f32,             // 4 bytes
    far_distance: f32,              // 4 bytes
    texel_size: f32,                // 4 bytes
    _padding: f32,                  // 4 bytes
}

// CSM uniforms (matches Rust CsmUniforms layout)
struct CsmUniforms {
    light_direction: vec4<f32>,
    light_view: mat4x4<f32>,
    cascades: array<ShadowCascade, 4>,
    cascade_count: u32,
    pcf_kernel_size: u32,
    depth_bias: f32,
    slope_bias: f32,
    shadow_map_size: f32,
    debug_mode: u32,
    evsm_positive_exp: f32,
    evsm_negative_exp: f32,
    peter_panning_offset: f32,
    enable_unclipped_depth: u32,
    depth_clip_factor: f32,
    technique: u32,
    technique_flags: u32,
    _pad1a: f32,
    _pad1b: f32,
    _pad1c: f32,
    // [blocker radius (texels), max filter radius (texels), moment bias, light radius (texels)]
    technique_params: vec4<f32>,
    technique_reserved: vec4<f32>,
    cascade_blend_range: f32,
    _pad2a: f32,
    _pad2b: f32,
    _pad2c: f32,
    _pad2d: array<vec4<f32>, 6>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) raw_height: f32,
};

fn height_to_world_y(h: f32) -> f32 {
    return (h - u.terrain_params.x) * u.terrain_params.w;
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VertexOutput {
    let dims = textureDimensions(heightmap);
    let max_texel = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
    let texel = clamp(
        vec2<i32>(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y))),
        vec2<i32>(0, 0),
        max_texel
    );
    let h = textureLoad(heightmap, texel, 0).r;
    
    let world_y = height_to_world_y(h);
    
    let world_xz = u.render_origin_xz + uv * u.render_span_xz;
    
    var out: VertexOutput;
    out.world_pos = vec3<f32>(world_xz.x, world_y, world_xz.y);
    out.position = u.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.uv = uv;
    out.raw_height = h;
    return out;
}

// Improved normal calculation from height gradient
fn compute_normal(uv: vec2<f32>) -> vec3<f32> {
    let dims_u = textureDimensions(heightmap);
    let dims = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
    let max_texel = vec2<i32>(i32(dims_u.x) - 1, i32(dims_u.y) - 1);
    let texel = clamp(
        vec2<i32>(
            i32(uv.x * max(f32(dims_u.x) - 1.0, 0.0)),
            i32(uv.y * max(f32(dims_u.y) - 1.0, 0.0)),
        ),
        vec2<i32>(0, 0),
        max_texel
    );
    let left = vec2<i32>(max(texel.x - 1, 0), texel.y);
    let right = vec2<i32>(min(texel.x + 1, max_texel.x), texel.y);
    let down = vec2<i32>(texel.x, max(texel.y - 1, 0));
    let up = vec2<i32>(texel.x, min(texel.y + 1, max_texel.y));
    let step_x = u.render_span_xz.x / max(f32(dims_u.x) - 1.0, 1.0);
    let step_z = u.render_span_xz.y / max(f32(dims_u.y) - 1.0, 1.0);
    let h_l = height_to_world_y(textureLoad(heightmap, left, 0).r);
    let h_r = height_to_world_y(textureLoad(heightmap, right, 0).r);
    let h_d = height_to_world_y(textureLoad(heightmap, down, 0).r);
    let h_u = height_to_world_y(textureLoad(heightmap, up, 0).r);
    let tangent_x = vec3<f32>(2.0 * step_x, h_r - h_l, 0.0);
    let tangent_z = vec3<f32>(0.0, h_u - h_d, 2.0 * step_z);
    var n = normalize(cross(tangent_z, tangent_x));
    // Amplify normal detail based on pbr_params.y (normal_strength)
    let strength = u.pbr_params.y;
    n.x *= strength;
    n.z *= strength;
    return normalize(n);
}

// Blinn-Phong specular
fn blinn_phong_specular(normal: vec3<f32>, light_dir: vec3<f32>, view_dir: vec3<f32>, shininess: f32) -> f32 {
    let half_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal, half_dir), 0.0), shininess);
    return spec;
}

// Soft shadow approximation based on normal and light direction
fn soft_shadow_factor(normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let ndotl = dot(normal, light_dir);
    // Smooth transition from lit to shadow
    let shadow = smoothstep(-0.1, 0.3, ndotl);
    return shadow;
}

// Height-based material (PBR-like with Imhof palette and roughness variation)
fn get_material(h_norm: f32, slope: f32) -> vec4<f32> {
    // albedo.rgb, roughness
    var albedo: vec3<f32>;
    var roughness: f32;
    
    // Imhof palette: green valleys -> brown slopes -> white peaks
    if h_norm < 0.3 {
        // Darker foothill greens with less pastel bias
        albedo = mix(vec3<f32>(0.16, 0.42, 0.14), vec3<f32>(0.32, 0.56, 0.21), h_norm / 0.3);
        roughness = 0.8;
    } else if h_norm < 0.7 {
        // Warmer, earthier midslopes
        albedo = mix(vec3<f32>(0.32, 0.56, 0.21), vec3<f32>(0.50, 0.39, 0.24), (h_norm - 0.3) / 0.4);
        roughness = 0.7;
    } else {
        // Snow/high rock without clipping to pure white
        albedo = mix(vec3<f32>(0.50, 0.39, 0.24), vec3<f32>(0.88, 0.88, 0.84), (h_norm - 0.7) / 0.3);
        roughness = 0.4;
    }
    
    // Add slope-based rock exposure (darkens steep areas)
    if slope > 0.5 && h_norm < 0.8 {
        let rock_blend = smoothstep(0.5, 0.75, slope);
        albedo = mix(albedo, vec3<f32>(0.35, 0.32, 0.28), rock_blend * 0.4);
        roughness = mix(roughness, 0.65, rock_blend);
    }
    
    return vec4<f32>(albedo, roughness);
}

// ACES tonemapping
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Preserve categorical overlay hue while carrying stronger terrain structure
// through a scalar derived from the lit terrain field plus signed ridge relief.
fn preserve_overlay_scalar(
    lit_linear: vec3<f32>,
    normal: vec3<f32>,
    sun_dir: vec3<f32>,
    wrapped_ndotl: f32,
    shadow_term: f32,
    height_ao: f32,
    view_dir: vec3<f32>,
    exposure: f32,
    normal_strength: f32,
) -> f32 {
    let base_intensity = max(dot(lit_linear, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0);

    // Macro terrain form: focus the added contrast on mountain slopes, not flats.
    let slope_steepness = clamp(1.0 - abs(normal.y), 0.0, 1.0);
    let mountain_weight = smoothstep(0.05, 0.42, slope_steepness);
    let normal_gain = 1.0 + smoothstep(1.0, 3.8, normal_strength) * 0.18;
    let light_facing = clamp(dot(normal, sun_dir), 0.0, 1.0);
    let macro_shade = mix(wrapped_ndotl, light_facing, 0.68);
    let macro_scale = mix(1.0, 0.55 + macro_shade * 0.94, mountain_weight)
        * (1.0 + mountain_weight * 0.16 * normal_gain);

    // Local ridge structure from screen-space normal variation.
    let dndx = dpdx(normal);
    let dndy = dpdy(normal);
    let normal_gradient = length(dndx) + length(dndy);
    let edge_signal = clamp(
        mountain_weight * 0.24 + normal_gradient * (8.5 + 1.2 * normal_gain),
        0.0,
        1.0,
    );

    let ridge_bright = clamp(
        edge_signal * (light_facing + 0.22) * 0.20,
        0.0,
        0.16,
    );
    let ridge_dark = clamp(
        edge_signal * (1.0 - light_facing) * 0.20,
        0.0,
        0.10,
    );
    let local_scale = 1.0 + ridge_bright - ridge_dark;

    // Keep shadows coherent with AO/shadow terms while adding a subtle view rim.
    let occlusion_scale = mix(0.965, 1.0, height_ao * shadow_term);
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0) * mountain_weight * 0.055;

    var preserve_intensity = base_intensity * macro_scale * local_scale * occlusion_scale;
    preserve_intensity += base_intensity * rim;
    preserve_intensity = max(preserve_intensity, 0.0);
    return pow(aces_tonemap(vec3<f32>(preserve_intensity * exposure)), vec3<f32>(1.0 / 2.2)).x;
}

// Hemisphere sky ambient with warm ground bounce
fn sky_ambient(normal: vec3<f32>) -> vec3<f32> {
    let sky_zenith = vec3<f32>(0.36, 0.55, 0.88);   // Deep blue zenith
    let sky_horizon = vec3<f32>(0.55, 0.68, 0.85);   // Lighter horizon
    let ground_bounce = vec3<f32>(0.22, 0.18, 0.12);  // Warm earth bounce
    let sky_factor = normal.y * 0.5 + 0.5;
    // Blend sky zenith/horizon based on normal elevation, then mix with ground
    let sky = mix(sky_horizon, sky_zenith, clamp(normal.y, 0.0, 1.0));
    return mix(ground_bounce, sky, sky_factor);
}

// GGX-approximate specular distribution for terrain
fn ggx_specular(normal: vec3<f32>, light_dir: vec3<f32>, view_dir: vec3<f32>, roughness_val: f32) -> f32 {
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let a = roughness_val * roughness_val;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159 * denom * denom + 0.0001);
    // Schlick geometry approximation
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let k = (roughness_val + 1.0) * (roughness_val + 1.0) / 8.0;
    let G_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
    let G_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 0.0001);
    return D * G_v * G_l;
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn rotate_y(v: vec3<f32>, sin_theta: f32, cos_theta: f32) -> vec3<f32> {
    return vec3<f32>(
        v.x * cos_theta + v.z * sin_theta,
        v.y,
        -v.x * sin_theta + v.z * cos_theta,
    );
}

fn eval_hdri_ibl(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    base_color: vec3<f32>,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let roughness_clamped = clamp(roughness, 0.0, 1.0);
    let rotated_normal = rotate_y(normal, u.ibl_params.z, u.ibl_params.w);
    let rotated_view = rotate_y(view_dir, u.ibl_params.z, u.ibl_params.w);
    let reflection_dir = reflect(-rotated_view, rotated_normal);
    let fresnel = fresnel_schlick_roughness(n_dot_v, f0, roughness_clamped);
    let diffuse_weight = vec3<f32>(1.0) - fresnel;
    let irradiance = textureSampleLevel(envIrradiance, envSampler, rotated_normal, 0.0).rgb;
    let diffuse = diffuse_weight * base_color * irradiance;
    let mip_level = roughness_clamped * roughness_clamped * max(u.ibl_params.y, 0.0);
    let prefiltered = textureSampleLevel(envSpecular, envSampler, reflection_dir, mip_level).rgb;
    let brdf = textureSampleLevel(brdfLUT, envSampler, vec2<f32>(n_dot_v, roughness_clamped), 0.0).rg;
    let specular = prefiltered * (fresnel * brdf.x + brdf.y);
    return diffuse + specular;
}

// ─────────────────────────────────────────────────────────────────────────────
// P6.2: CSM Shadow Sampling Functions
// ─────────────────────────────────────────────────────────────────────────────

// Select cascade based on view depth
fn select_cascade(view_depth: f32) -> u32 {
    let count = csm_uniforms.cascade_count;
    for (var i = 0u; i < count; i = i + 1u) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            return i;
        }
    }
    return count - 1u;
}

// VSM: Chebyshev upper bound
fn sample_shadow_vsm(shadow_coords: vec2<f32>, receiver_depth: f32, cascade_idx: u32, moment_bias: f32) -> f32 {
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    let mean = moments.r;
    let mean_sq = moments.g;
    
    // Receiver is in front of mean - fully lit
    if (receiver_depth <= mean) {
        return 1.0;
    }
    
    // Variance with bias to prevent light bleeding
    let variance = max(mean_sq - mean * mean, moment_bias);
    let d = receiver_depth - mean;
    let p_max = variance / (variance + d * d);
    
    return clamp(p_max, 0.0, 1.0);
}

// EVSM: Exponential variance shadow maps
fn sample_shadow_evsm(shadow_coords: vec2<f32>, receiver_depth: f32, cascade_idx: u32, moment_bias: f32) -> f32 {
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    
    let c_pos = csm_uniforms.evsm_positive_exp;
    let c_neg = csm_uniforms.evsm_negative_exp;
    
    // Warp receiver depth. The negative lobe is negated to match the moment
    // atlas (moment_generation.wgsl) so both warps increase with depth.
    let warp_depth_pos = exp(c_pos * (receiver_depth - 1.0));
    let warp_depth_neg = -exp(-c_neg * receiver_depth);
    let variance_floor = evsm_minimum_variance(
        vec2<f32>(warp_depth_pos, warp_depth_neg),
        vec2<f32>(c_pos, c_neg)
    );

    let variance_visibility = clamp(
        evsm_visibility_from_moments(
            moments,
            warp_depth_pos,
            warp_depth_neg,
            variance_floor
        ),
        0.0,
        1.0
    );
    let moment_visibility = min(
        variance_visibility,
        evsm_moment_leak_control(
            moments.rg,
            warp_depth_pos,
            c_pos,
            variance_floor.x
        )
    );
    return moment_visibility;
}

// MSM: Moment shadow maps (4 moments)
fn sample_shadow_msm(shadow_coords: vec2<f32>, receiver_depth: f32, cascade_idx: u32, moment_bias: f32) -> f32 {
    let moments = textureSample(moment_maps, moment_sampler, shadow_coords, i32(cascade_idx));
    return msm_visibility_from_moments(moments, receiver_depth, moment_bias);
}

fn pcss_blocker_search(
    shadow_coords: vec2<f32>,
    receiver_depth: f32,
    cascade_idx: u32,
    search_radius: f32,
) -> f32 {
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
    let texel_size = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
    let scaled_search_radius = search_radius * texel_size;
    let dimensions = textureDimensions(shadow_maps);
    var blocker_sum = 0.0;
    var blocker_count = 0.0;

    for (var i = 0; i < 12; i++) {
        let sample_coords = shadow_coords + poisson_disk[i] * scaled_search_radius;
        if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
            sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
            let texel_coords = vec2<i32>(clamp(
                sample_coords * vec2<f32>(dimensions),
                vec2<f32>(0.0),
                vec2<f32>(dimensions) - vec2<f32>(1.0)
            ));
            let shadow_depth =
                textureLoad(shadow_maps, texel_coords, i32(cascade_idx), 0);
            if (shadow_depth < receiver_depth) {
                blocker_sum += shadow_depth;
                blocker_count += 1.0;
            }
        }
    }

    if (blocker_count > 0.0) {
        return blocker_sum / blocker_count;
    }
    return -1.0;
}

fn pcss_penumbra_size(
    receiver_depth: f32,
    blocker_depth: f32,
    light_size: f32,
) -> f32 {
    let depth_diff = max(receiver_depth - blocker_depth, 0.0);
    let penumbra = (depth_diff * light_size) / max(blocker_depth, 0.001);
    return clamp(penumbra, 0.0, 100.0);
}

// Main CSM shadow sampling with technique dispatch
fn sample_csm_shadow(world_pos: vec3<f32>, normal: vec3<f32>, cascade_idx: u32) -> f32 {
    // Early out: no cascades means fully lit
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }
    
    let cascade = csm_uniforms.cascades[cascade_idx];
    
    // Transform to light space
    let light_space_pos = cascade.light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = light_space_pos.xyz / light_space_pos.w;
    
    // Convert from NDC to shadow texture UVs. Y must be flipped to match the
    // depth target's texture-space orientation.
    let shadow_coords = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    let depth_01 = ndc.z;  // orthographic_rh maps Z to [0,1] directly
    
    // Out of bounds check
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        depth_01 < 0.0 || depth_01 > 1.0) {
        return 1.0;
    }
    
    // Compute bias
    let light_dir_norm = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(normal, light_dir_norm), 0.0);
    let slope_factor = clamp(1.0 - n_dot_l, 0.0, 1.0);
    let bias = csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_factor + csm_uniforms.peter_panning_offset;
    let compare_depth = depth_01 - bias;
    
    let technique = csm_uniforms.technique;
    let moment_bias = csm_uniforms.technique_params.z;
    
    // HARD shadows (technique=0)
    if (technique == 0u) {
        return textureSampleCompare(shadow_maps, shadow_sampler, shadow_coords, i32(cascade_idx), compare_depth);
    }
    
    // PCF (technique=1): 3x3 kernel
    if (technique == 1u) {
        let texel_size = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
        var shadow_sum = 0.0;
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
                shadow_sum += textureSampleCompare(shadow_maps, shadow_sampler, shadow_coords + offset, i32(cascade_idx), compare_depth);
            }
        }
        return shadow_sum / 9.0;
    }
    
    // PCSS (technique=2): blocker-dependent penumbra and adaptive PCF
    if (technique == 2u) {
        let avg_blocker_depth = pcss_blocker_search(
            shadow_coords,
            compare_depth,
            cascade_idx,
            min(csm_uniforms.technique_params.x, 50.0)
        );
        if (avg_blocker_depth < 0.0) {
            return 1.0;
        }

        let penumbra = pcss_penumbra_size(
            compare_depth,
            avg_blocker_depth,
            csm_uniforms.technique_params.w
        );
        let max_filter_radius = min(csm_uniforms.technique_params.y, 100.0);
        let filter_radius = min(
            max(penumbra, min(max_filter_radius, 1.0)),
            max_filter_radius
        );
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
        let scaled_filter_radius =
            filter_radius / max(csm_uniforms.shadow_map_size, 1.0);
        var shadow_sum = 0.0;
        for (var i = 0; i < 16; i++) {
            let sample_coords = shadow_coords + poisson_disk[i] * scaled_filter_radius;
            if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 &&
                sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
                shadow_sum += textureSampleCompare(
                    shadow_maps,
                    shadow_sampler,
                    sample_coords,
                    i32(cascade_idx),
                    clamp(compare_depth, 0.0, 1.0)
                );
            } else {
                shadow_sum += 1.0;
            }
        }
        return shadow_sum / 16.0;
    }
    
    // VSM (technique=3)
    if (technique == 3u) {
        return sample_shadow_vsm(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // EVSM (technique=4)
    if (technique == 4u) {
        return sample_shadow_evsm(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // MSM (technique=5)
    if (technique == 5u) {
        return sample_shadow_msm(shadow_coords, compare_depth, cascade_idx, moment_bias);
    }
    
    // Fallback: PCF
    let texel_size = 1.0 / max(csm_uniforms.shadow_map_size, 1.0);
    var shadow_sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow_sum += textureSampleCompare(shadow_maps, shadow_sampler, shadow_coords + offset, i32(cascade_idx), compare_depth);
        }
    }
    return shadow_sum / 9.0;
}

// Calculate CSM shadow visibility
fn calculate_csm_shadow(world_pos: vec3<f32>, normal: vec3<f32>, view_depth: f32) -> f32 {
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }
    
    let cascade_idx = select_cascade(view_depth);
    return sample_csm_shadow(world_pos, normal, cascade_idx);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_intensity = u.lighting.x;
    let ambient_strength = u.lighting.y;
    let shadow_strength = u.lighting.z;
    let water_level = u.lighting.w;
    let exposure = u.pbr_params.x;
    let ibl_intensity = u.pbr_params.z;
    
    // P6.2: Debug mode 33 - Shadow technique visualization
    if csm_uniforms.debug_mode == 33u {
        let tech = csm_uniforms.technique;
        var tech_color = vec3<f32>(1.0, 0.0, 1.0); // Magenta = unknown
        if tech == 0u {
            tech_color = vec3<f32>(1.0, 0.0, 0.0); // Red = HARD
        } else if tech == 1u {
            tech_color = vec3<f32>(0.0, 1.0, 0.0); // Green = PCF
        } else if tech == 2u {
            tech_color = vec3<f32>(0.0, 0.0, 1.0); // Blue = PCSS
        } else if tech == 3u {
            tech_color = vec3<f32>(0.0, 1.0, 1.0); // Cyan = VSM
        } else if tech == 4u {
            tech_color = vec3<f32>(1.0, 1.0, 0.0); // Yellow = EVSM
        } else if tech == 5u {
            tech_color = vec3<f32>(1.0, 0.0, 1.0); // Magenta = MSM
        }
        return vec4<f32>(tech_color, 1.0);
    }
    
    // Debug mode 34 - Show cascade_count status (Green=has cascades, Red=no cascades)
    if csm_uniforms.debug_mode == 34u {
        if csm_uniforms.cascade_count > 0u {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0); // Green = cascades available
        } else {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red = no cascades
        }
    }
    
    // Debug mode 35 - Visualize raw shadow value (grayscale: black=shadow, white=lit)
    if csm_uniforms.debug_mode == 35u {
        let normal = compute_normal(in.uv);
        let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);
        let shadow_val = calculate_csm_shadow(in.world_pos, normal, view_depth);
        return vec4<f32>(shadow_val, shadow_val, shadow_val, 1.0);
    }
    
    // Debug mode 36 - Check if shadow passes ran (technique_reserved[0] == 1.0)
    if csm_uniforms.debug_mode == 36u {
        if csm_uniforms.technique_reserved.x > 0.5 {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0); // Green = shadow passes ran
        } else {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red = shadow passes did NOT run
        }
    }
    
    // Debug mode 37 - Visualize light-space coordinates
    // R=shadow_uv.x, G=shadow_uv.y, B=receiver_depth
    if csm_uniforms.debug_mode == 37u {
        let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);
        let cascade_idx = select_cascade(view_depth);
        let cascade = csm_uniforms.cascades[cascade_idx];
        
        // Transform to light space
        let light_space_pos = cascade.light_view_proj * vec4<f32>(in.world_pos, 1.0);
        let ndc = light_space_pos.xyz / light_space_pos.w;
        let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
        let receiver_depth = ndc.z;
        
        // Visualize light-space coords: R=u, G=v, B=depth
        return vec4<f32>(shadow_uv.x, shadow_uv.y, receiver_depth, 1.0);
    }
    
    // Debug mode 38 - Sample shadow map and show result with depth info
    // R=shadow_result (0=shadow, 1=lit), G=receiver_depth, B=cascade_idx/4
    if csm_uniforms.debug_mode == 38u {
        let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);
        let cascade_idx = select_cascade(view_depth);
        let cascade = csm_uniforms.cascades[cascade_idx];
        
        let light_space_pos = cascade.light_view_proj * vec4<f32>(in.world_pos, 1.0);
        let ndc = light_space_pos.xyz / light_space_pos.w;
        let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
        let receiver_depth = ndc.z;
        
        // Sample shadow map using comparison sampler (returns 0 or 1)
        let shadow_result = textureSampleCompare(shadow_maps, shadow_sampler, shadow_uv, i32(cascade_idx), receiver_depth);
        
        return vec4<f32>(shadow_result, receiver_depth, f32(cascade_idx) / 4.0, 1.0);
    }
    
    // Check if below water level
    let is_water = in.raw_height < water_level;
    
    // Normalized height for material selection
    let h_norm = clamp((in.raw_height - u.terrain_params.x) / max(u.terrain_params.y, 1.0), 0.0, 1.0);
    
    // Compute surface normal
    let normal = compute_normal(in.uv);
    
    // Slope for material blending (0 = flat, 1 = vertical)
    let slope = 1.0 - abs(normal.y);
    
    // Get material properties
    let material = get_material(h_norm, slope);
    var albedo = material.rgb;
    let roughness = material.a;
    
    // Water handling
    if is_water {
        albedo = u.water_color.rgb;
    }
    
    // === OVERLAY BLENDING (before lighting, so overlays are lit) ===
    let overlay_enabled = u.overlay_params.x > 0.5;
    let overlay_opacity = u.overlay_params.y;
    let overlay_blend_mode = u.overlay_params.z;
    let solid_surface = u.overlay_params.w > 0.5;
    let overlay_preserve_colors = u.pbr_params.w > 0.5;

    // Sample overlay texture for solid check and blending
    let overlay = textureSample(overlay_tex, overlay_sampler, in.uv);
    let overlay_alpha = overlay.a * overlay_opacity;
    let solid_background = overlay_enabled && solid_surface && overlay_alpha <= 0.01;
    
    // When solid=false and overlay is enabled, discard fragments where overlay alpha is low
    // This hides the base surface outside the region of interest (like rayshader solid=FALSE)
    // Use threshold of 0.5 to account for bilinear filtering edge bleed - edge pixels between
    // alpha=1.0 and alpha=0.0 can have intermediate values due to texture filtering
    if overlay_enabled && !solid_surface && overlay.a < 0.5 {
        discard;
    }
    
    let preserve_overlay_active = overlay_enabled && overlay_preserve_colors && overlay_alpha > 0.01;

    if overlay_enabled && overlay_opacity > 0.001 {
        if overlay_alpha > 0.01 {
            // Apply blend mode (overlay already sampled above for solid check)
            if overlay_blend_mode < 0.5 {
                var blend_alpha = overlay_alpha;
                if overlay_preserve_colors {
                    blend_alpha = clamp(overlay_alpha * 1.10, 0.0, 1.0);
                }
                // Normal blend: fully replace albedo with overlay when alpha is high
                // For land cover overlays, we want the categorical colors to show through
                albedo = mix(albedo, overlay.rgb, blend_alpha);
            } else if overlay_blend_mode < 1.5 {
                // Multiply blend
                let multiplied = albedo * overlay.rgb;
                albedo = mix(albedo, multiplied, overlay_alpha);
            } else {
                // Overlay blend (Photoshop-style)
                let base = albedo;
                var blended: vec3<f32>;
                // Overlay formula: 2*base*blend if base < 0.5, else 1 - 2*(1-base)*(1-blend)
                blended.r = select(1.0 - 2.0 * (1.0 - base.r) * (1.0 - overlay.r), 2.0 * base.r * overlay.r, base.r < 0.5);
                blended.g = select(1.0 - 2.0 * (1.0 - base.g) * (1.0 - overlay.g), 2.0 * base.g * overlay.g, base.g < 0.5);
                blended.b = select(1.0 - 2.0 * (1.0 - base.b) * (1.0 - overlay.b), 2.0 * base.b * overlay.b, base.b < 0.5);
                albedo = mix(albedo, blended, overlay_alpha);
            }
        }
    }
    // === END OVERLAY BLENDING ===
    
    // Sun lighting
    let sun_dir = normalize(u.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);

    // Wrapped diffuse for softer light/shadow transition (half-Lambert)
    let wrapped_ndotl = ndotl * 0.7 + 0.3 * max(dot(normal, sun_dir) * 0.5 + 0.5, 0.0);

    // Sample heightfield AO (use textureLoad for R32Float)
    let ao_dims = textureDimensions(height_ao_tex, 0);
    let ao_pixel = vec2<i32>(in.uv * vec2<f32>(ao_dims));
    let ao_clamped = clamp(ao_pixel, vec2<i32>(0), vec2<i32>(ao_dims) - vec2<i32>(1));
    let height_ao = textureLoad(height_ao_tex, ao_clamped, 0).r;

    // Sample sun visibility (use textureLoad for R32Float)
    let sv_dims = textureDimensions(sun_vis_tex, 0);
    let sv_pixel = vec2<i32>(in.uv * vec2<f32>(sv_dims));
    let sv_clamped = clamp(sv_pixel, vec2<i32>(0), vec2<i32>(sv_dims) - vec2<i32>(1));
    let sun_vis = textureLoad(sun_vis_tex, sv_clamped, 0).r;

    // P6.2: CSM shadow sampling with technique dispatch
    // Calculate view depth for cascade selection (approximate from world position)
    let view_pos = u.view_proj * vec4<f32>(in.world_pos, 1.0);
    let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);

    // Sample CSM shadow using technique-specific sampling (HARD, PCF, PCSS, VSM, EVSM, MSM)
    let csm_shadow = calculate_csm_shadow(in.world_pos, normal, view_depth);

    // Fallback soft shadow for when CSM has no cascades
    let soft_shadow = soft_shadow_factor(normal, sun_dir);

    // Use CSM shadow if cascades available, otherwise fallback to soft shadow
    let shadow = select(soft_shadow, csm_shadow, csm_uniforms.cascade_count > 0u);

    // Combine shadow with sun visibility (multiplicative) - softer blending
    let combined_shadow = shadow * sun_vis;
    let shadow_term = mix(1.0, combined_shadow, shadow_strength);
    // Soften shadow color: shadowed areas get a cool blue tint instead of pure black
    let shadow_tint = mix(vec3<f32>(0.42, 0.47, 0.56), vec3<f32>(1.0), shadow_term);
    let effective_shadow_tint = select(shadow_tint, vec3<f32>(1.0), preserve_overlay_active);

    // View direction
    let view_dir = normalize(u.camera_pos.xyz - in.world_pos);
    let terrain_span = max(abs(u.render_span_xz.x), abs(u.render_span_xz.y));
    let hdri_enabled = u.ibl_params.x > 0.5;

    // === DIRECT SUN LIGHTING ===
    let sun_color = vec3<f32>(1.0, 0.93, 0.78); // Warm golden sunlight
    let diffuse = albedo * sun_color * wrapped_ndotl * sun_intensity * shadow_term;

    // GGX specular (energy-conserving, roughness-based)
    var spec_val = ggx_specular(normal, sun_dir, view_dir, roughness);
    spec_val *= sun_intensity * shadow_term * 0.15;
    let specular_color = sun_color * spec_val;

    // Water specular boost
    var water_spec = vec3<f32>(0.0);
    if is_water {
        let water_s = ggx_specular(vec3<f32>(0.0, 1.0, 0.0), sun_dir, view_dir, 0.05);
        water_spec = sun_color * water_s * sun_intensity * 0.6;
    }

    // === FILL LIGHT (simulates bounce/GI from opposite side of sun) ===
    let fill_dir = normalize(vec3<f32>(-sun_dir.x, 0.35, -sun_dir.z));
    let fill_ndotl = max(dot(normal, fill_dir), 0.0);
    let fill_color = vec3<f32>(0.22, 0.26, 0.34); // Subtle cool bounce
    let fill_light = albedo * fill_color * fill_ndotl * 0.08 * sun_intensity;

    // === GROUND BOUNCE (warm light from below) ===
    let bounce_ndotl = max(dot(normal, vec3<f32>(0.0, -1.0, 0.0)), 0.0);
    let bounce_color = vec3<f32>(0.18, 0.14, 0.10); // Warm earth tone
    let ground_bounce = albedo * bounce_color * bounce_ndotl * 0.05;

    // === INDIRECT LIGHTING (HDRI when loaded, analytic hemisphere fallback otherwise) ===
    let ambient_hemi = sky_ambient(normal) * albedo * ambient_strength * ibl_intensity * height_ao * 0.75;
    let ibl_roughness = select(roughness, 0.05, is_water);
    let ibl_f0 = select(vec3<f32>(0.04), vec3<f32>(0.02), is_water);
    let ambient_ibl = eval_hdri_ibl(normal, view_dir, albedo, ibl_roughness, ibl_f0)
        * ambient_strength
        * ibl_intensity
        * height_ao;
    let indirect_light = select(ambient_hemi, ambient_ibl, hdri_enabled);

    // === FRESNEL RIM LIGHT (depth cue at grazing angles) ===
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 4.0);
    let rim_light = fresnel * sun_color * 0.06 * shadow_term * sun_intensity;

    // === COMBINE ALL LIGHTING ===
    var color = diffuse * effective_shadow_tint
              + specular_color
              + water_spec
              + fill_light * height_ao
              + ground_bounce
              + indirect_light
              + rim_light;

    if preserve_overlay_active {
        let preserve_display = preserve_overlay_scalar(
            color,
            normal,
            sun_dir,
            wrapped_ndotl,
            shadow_term,
            height_ao,
            view_dir,
            exposure,
            u.pbr_params.y,
        );
        let overlay_display_rgb = pow(
            max(overlay.rgb, vec3<f32>(0.0)),
            vec3<f32>(1.0 / 2.2),
        );
        let preserve_display_rgb = overlay_display_rgb * preserve_display;
        color = pow(
            clamp(preserve_display_rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
            vec3<f32>(2.2),
        );
    } else {
        // Apply exposure and tonemapping
        color = color * exposure;
        color = aces_tonemap(color);

        // Gamma correction (linear to sRGB)
        color = pow(color, vec3<f32>(1.0 / 2.2));
    }

    // === ATMOSPHERIC PERSPECTIVE (depth-based haze) ===
    let atmo_scale = 0.045 / max(terrain_span, 1.0);
    let view_dist = length(u.camera_pos.xyz - in.world_pos);
    let atmo_factor = pow(clamp(1.0 - exp(-view_dist * atmo_scale), 0.0, 1.0), 1.35);
    let atmo_color = mix(
        vec3<f32>(0.55, 0.62, 0.72),  // Haze near color
        vec3<f32>(0.64, 0.70, 0.79),  // Haze far color
        clamp(atmo_factor, 0.0, 1.0)
    );

    // Apply atmospheric haze AFTER tonemapping (in display space)
    var haze_mix = clamp(atmo_factor * 0.055, 0.0, 0.08);
    if overlay_enabled && overlay_preserve_colors && overlay_alpha > 0.01 {
        haze_mix = 0.0;
    }
    color = mix(color, atmo_color, haze_mix);

    // Apply vignette (lens effect)
    let vignette_strength = u.lens_params.x;
    let vignette_radius = u.lens_params.y;
    let vignette_softness = u.lens_params.z;
    if vignette_strength > 0.001 {
        let screen_uv = in.position.xy / u.screen_dims.xy;
        let center_dist = length(screen_uv - vec2<f32>(0.5));
        let vignette = 1.0 - smoothstep(vignette_radius - vignette_softness, vignette_radius + vignette_softness, center_dist * 2.0);
        color = color * mix(1.0, vignette, vignette_strength);
    }

    if solid_background {
        color = u.background.rgb;
    }

    return vec4<f32>(color, 1.0);
}
