// shaders/cloud_shadows.wgsl
// Cloud shadow overlay system for B7 - 2D shadow texture modulation over terrain
// RELEVANT FILES: src/core/cloud_shadows.rs, src/scene/mod.rs, examples/cloud_shadows_demo.py

// Cloud shadow parameters matching Rust CloudShadowUniforms
struct CloudShadowUniforms {
    // Cloud movement parameters
    cloud_speed: vec2<f32>,        // Cloud movement speed (x, y)
    time: f32,                     // Current time for animation
    cloud_scale: f32,              // Scale of cloud patterns

    // Cloud appearance parameters
    cloud_density: f32,            // Base cloud density [0, 1]
    cloud_coverage: f32,           // Cloud coverage amount [0, 1]
    shadow_intensity: f32,         // Shadow strength [0, 1]
    shadow_softness: f32,          // Shadow edge softness

    // Texture parameters
    texture_size: vec2<f32>,       // Size of cloud shadow texture
    inv_texture_size: vec2<f32>,   // 1.0 / texture_size

    // Noise parameters
    noise_octaves: u32,            // Number of noise octaves
    noise_frequency: f32,          // Base noise frequency
    noise_amplitude: f32,          // Noise amplitude
    wind_direction: f32,           // Wind direction in radians

    // Debug and visualization
    debug_mode: u32,               // Debug visualization mode
    show_clouds_only: u32,         // Show only cloud patterns
    _padding: vec2<f32>,           // Padding for alignment
};

@group(0) @binding(0) var<uniform> cloud_params: CloudShadowUniforms;
@group(0) @binding(1) var cloud_shadow_texture: texture_storage_2d<rgba8unorm, write>;

// Simple pseudo-random function
fn random(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

// Smooth noise function
fn noise(st: vec2<f32>) -> f32 {
    let i = floor(st);
    let f = fract(st);

    let a = random(i);
    let b = random(i + vec2<f32>(1.0, 0.0));
    let c = random(i + vec2<f32>(0.0, 1.0));
    let d = random(i + vec2<f32>(1.0, 1.0));

    let u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Fractal noise with multiple octaves
fn fractal_noise(st: vec2<f32>, octaves: u32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = cloud_params.noise_frequency;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * noise(st * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    return value;
}

// Turbulence noise for more organic cloud shapes
fn turbulence(st: vec2<f32>, octaves: u32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = cloud_params.noise_frequency;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * abs(noise(st * frequency) * 2.0 - 1.0);
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    return value;
}

// Worley noise for cellular cloud patterns
fn worley_noise(st: vec2<f32>, scale: f32) -> f32 {
    let scaled_st = st * scale;
    let i_st = floor(scaled_st);
    let f_st = fract(scaled_st);

    var min_dist = 1.0;

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            var point = random(i_st + neighbor) * 0.5 + 0.25;
            point = 0.5 + 0.5 * sin(cloud_params.time * 0.5 + 6.2831 * point);
            let diff = neighbor + point - f_st;
            let dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }

    return min_dist;
}

// Generate cloud density at given position
fn generate_cloud_density(uv: vec2<f32>) -> f32 {
    // Apply wind movement
    let wind_offset = vec2<f32>(
        cos(cloud_params.wind_direction),
        sin(cloud_params.wind_direction)
    ) * cloud_params.time * 0.1;

    // Add cloud speed movement
    let moved_uv = uv + cloud_params.cloud_speed * cloud_params.time + wind_offset;

    // Scale UV coordinates
    let scaled_uv = moved_uv * cloud_params.cloud_scale;

    // Generate base cloud pattern using fractal noise
    let base_clouds = fractal_noise(scaled_uv, cloud_params.noise_octaves);

    // Add turbulence for more organic shapes
    let turb = turbulence(scaled_uv * 0.5, 3u) * 0.3;

    // Add cellular structure with Worley noise
    let cellular = worley_noise(scaled_uv, 2.0) * 0.4;

    // Combine noise patterns
    var cloud_density = base_clouds + turb - cellular;

    // Apply coverage and density parameters
    cloud_density = (cloud_density - (1.0 - cloud_params.cloud_coverage)) / cloud_params.cloud_coverage;
    cloud_density = clamp(cloud_density * cloud_params.cloud_density, 0.0, 1.0);

    // Smooth the edges
    cloud_density = smoothstep(0.1, 0.9, cloud_density);

    return cloud_density;
}

// Apply cloud shadows to lighting
fn apply_cloud_shadow(base_lighting: f32, cloud_density: f32) -> f32 {
    // Calculate shadow amount based on cloud density
    let shadow_amount = cloud_density * cloud_params.shadow_intensity;

    // Apply shadow with softness
    let shadowed_lighting = base_lighting * (1.0 - shadow_amount);

    // Blend between full lighting and shadowed lighting based on softness
    return mix(shadowed_lighting, base_lighting, 1.0 - cloud_params.shadow_softness);
}

// Main compute shader for generating cloud shadow texture
@compute @workgroup_size(8, 8, 1)
fn cs_generate_cloud_shadows(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texel_coord = vec2<i32>(global_id.xy);
    let texture_size = vec2<i32>(cloud_params.texture_size);

    // Check bounds
    if (texel_coord.x >= texture_size.x || texel_coord.y >= texture_size.y) {
        return;
    }

    // Convert to UV coordinates [0, 1]
    let uv = vec2<f32>(texel_coord) / vec2<f32>(texture_size);

    // Generate cloud density
    let cloud_density = generate_cloud_density(uv);

    // Calculate shadow intensity
    let shadow_intensity = cloud_density * cloud_params.shadow_intensity;

    var output_color: vec4<f32>;

    // Debug modes
    if (cloud_params.debug_mode == 1u) {
        // Show cloud density
        output_color = vec4<f32>(cloud_density, cloud_density, cloud_density, 1.0);
    } else if (cloud_params.debug_mode == 2u) {
        // Show shadow intensity
        output_color = vec4<f32>(1.0 - shadow_intensity, 1.0 - shadow_intensity, 1.0 - shadow_intensity, 1.0);
    } else if (cloud_params.show_clouds_only == 1u) {
        // Show only cloud patterns (white clouds on black sky)
        output_color = vec4<f32>(cloud_density, cloud_density, cloud_density, 1.0);
    } else {
        // Normal mode: store shadow multiplier in all channels
        let shadow_multiplier = 1.0 - shadow_intensity;
        output_color = vec4<f32>(shadow_multiplier, shadow_multiplier, shadow_multiplier, 1.0);
    }

    textureStore(cloud_shadow_texture, texel_coord, output_color);
}

// Alternative: Direct cloud shadow sampling for use in terrain shaders
fn sample_cloud_shadow(world_pos: vec2<f32>, terrain_scale: f32) -> f32 {
    // Convert world position to UV coordinates
    let uv = (world_pos / terrain_scale + 1.0) * 0.5;

    // Generate cloud density at this position
    let cloud_density = generate_cloud_density(uv);

    // Return shadow multiplier
    return 1.0 - (cloud_density * cloud_params.shadow_intensity);
}

// Utility functions for external use
fn get_cloud_density_at_uv(uv: vec2<f32>) -> f32 {
    return generate_cloud_density(uv);
}

fn get_shadow_multiplier_at_uv(uv: vec2<f32>) -> f32 {
    let cloud_density = generate_cloud_density(uv);
    return 1.0 - (cloud_density * cloud_params.shadow_intensity);
}

// Animation helper - update cloud movement based on time
fn animate_clouds(delta_time: f32) {
    // This would be called from the host to update time-based parameters
    // Implementation handled on the Rust side
}
