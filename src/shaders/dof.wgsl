// shaders/dof.wgsl
// Realtime Depth of Field implementation with circle-of-confusion and gather blur (B6)
// RELEVANT FILES: src/core/dof.rs, python/forge3d/camera.py, tests/test_b6_dof.py

// DOF Parameters uniform buffer
struct DofUniforms {
    // Camera parameters
    aperture: f32,              // Aperture size (f-stop reciprocal)
    focus_distance: f32,        // Focus distance in world units
    focal_length: f32,          // Camera focal length
    sensor_size: f32,           // Sensor size for CoC calculations

    // Quality and performance settings
    blur_radius_scale: f32,     // Scale factor for blur radius
    max_blur_radius: f32,       // Maximum blur radius in pixels
    sample_count: u32,          // Number of samples for gather
    quality_level: u32,         // Quality level (0=low, 1=medium, 2=high, 3=ultra)

    // Near and far field settings
    near_transition_range: f32, // Transition range for near field
    far_transition_range: f32,  // Transition range for far field
    coc_bias: f32,             // CoC bias for fine-tuning
    bokeh_rotation: f32,       // Bokeh shape rotation

    // Screen space parameters
    screen_size: vec2<f32>,    // Screen resolution
    inv_screen_size: vec2<f32>, // 1.0 / screen_size

    // Debug and visualization
    debug_mode: u32,           // Debug visualization mode
    show_coc: u32,            // Show circle-of-confusion

    // M3: Tilt-shift parameters for Scheimpflug effect
    tilt_pitch: f32,          // Tilt around horizontal axis (radians)
    tilt_yaw: f32,            // Tilt around vertical axis (radians)
};

@group(0) @binding(0) var<uniform> dof_params: DofUniforms;
@group(0) @binding(1) var color_texture: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;
@group(0) @binding(3) var color_sampler: sampler;
@group(0) @binding(4) var dof_output: texture_storage_2d<rgba16float, write>;

// Poisson disk samples for gather blur (precomputed for different quality levels)
var<private> POISSON_SAMPLES_16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216), vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.094184101, -0.92938870), vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432), vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845), vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554), vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023), vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507), vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367), vec2<f32>(0.14383161, -0.14100790)
);

var<private> POISSON_SAMPLES_32: array<vec2<f32>, 32> = array<vec2<f32>, 32>(
    vec2<f32>(-0.975402, -0.0711386), vec2<f32>(-0.920505, -0.41142), vec2<f32>(-0.883908, 0.217872),
    vec2<f32>(-0.884518, 0.568041), vec2<f32>(-0.811945, 0.90521), vec2<f32>(-0.792474, -0.779962),
    vec2<f32>(-0.614856, 0.386578), vec2<f32>(-0.580859, -0.208777), vec2<f32>(-0.53795, 0.716666),
    vec2<f32>(-0.515427, 0.0899991), vec2<f32>(-0.454634, -0.707938), vec2<f32>(-0.420942, 0.991272),
    vec2<f32>(-0.261147, 0.588488), vec2<f32>(-0.211219, 0.114841), vec2<f32>(-0.146336, -0.259194),
    vec2<f32>(-0.139439, -0.888668), vec2<f32>(0.0116886, 0.326395), vec2<f32>(0.0380566, 0.625477),
    vec2<f32>(0.0625935, -0.50853), vec2<f32>(0.125584, 0.0469069), vec2<f32>(0.169469, -0.997253),
    vec2<f32>(0.320597, 0.291055), vec2<f32>(0.359172, -0.173468), vec2<f32>(0.435581, -0.250811),
    vec2<f32>(0.507934, 0.76124), vec2<f32>(0.566009, 0.208748), vec2<f32>(0.639979, 0.481617),
    vec2<f32>(0.652408, -0.634408), vec2<f32>(0.773463, -0.309951), vec2<f32>(0.859312, 0.271189),
    vec2<f32>(0.901218, 0.751167), vec2<f32>(0.937749, -0.832366)
);

// Hexagonal sample pattern for bokeh simulation
var<private> HEX_SAMPLES: array<vec2<f32>, 7> = array<vec2<f32>, 7>(
    vec2<f32>(0.0, 0.0),                    // Center
    vec2<f32>(1.0, 0.0),                    // Right
    vec2<f32>(0.5, 0.866025),               // Top-right
    vec2<f32>(-0.5, 0.866025),              // Top-left
    vec2<f32>(-1.0, 0.0),                   // Left
    vec2<f32>(-0.5, -0.866025),             // Bottom-left
    vec2<f32>(0.5, -0.866025)               // Bottom-right
);

// M3: Calculate effective focus distance with tilt-shift (Scheimpflug principle)
// The tilted focus plane varies focus distance across the image
fn calculate_tilted_focus_distance(uv: vec2<f32>) -> f32 {
    // Convert UV to normalized screen coordinates centered at origin [-1, 1]
    let centered_uv = (uv - 0.5) * 2.0;
    
    // Calculate tilt offset based on screen position
    // Tilt pitch affects vertical (Y) variation
    // Tilt yaw affects horizontal (X) variation
    let tilt_offset = centered_uv.y * tan(dof_params.tilt_pitch) + 
                      centered_uv.x * tan(dof_params.tilt_yaw);
    
    // Scale the tilt effect by focus distance to create realistic plane tilt
    // A larger focus distance means more dramatic variation across the frame
    let focus_variation = dof_params.focus_distance * tilt_offset * 0.5;
    
    // Return modified focus distance (clamped to positive values)
    return max(dof_params.focus_distance + focus_variation, 0.1);
}

// Calculate circle of confusion (CoC) from depth with tilt-shift support
fn calculate_coc_tilt(depth: f32, uv: vec2<f32>) -> f32 {
    // Get effective focus distance for this pixel (accounts for tilt)
    let effective_focus = calculate_tilted_focus_distance(uv);
    
    let object_distance = depth;
    let distance_diff = abs(object_distance - effective_focus);
    let denominator = object_distance * (effective_focus + dof_params.focal_length);
    
    if (denominator < 0.001) {
        return 0.0;
    }
    
    let coc = (dof_params.aperture * dof_params.focal_length * distance_diff) / denominator;
    let coc_pixels = coc * dof_params.sensor_size * dof_params.blur_radius_scale;
    
    return clamp(coc_pixels + dof_params.coc_bias, 0.0, dof_params.max_blur_radius);
}

// Calculate circle of confusion (CoC) from depth
fn calculate_coc(depth: f32) -> f32 {
    // Convert depth to world space distance
    let object_distance = depth;

    // Calculate CoC using thin lens equation
    // CoC = (aperture * focal_length * |object_distance - focus_distance|) /
    //       (object_distance * (focus_distance + focal_length))

    let distance_diff = abs(object_distance - dof_params.focus_distance);
    let denominator = object_distance * (dof_params.focus_distance + dof_params.focal_length);

    if (denominator < 0.001) {
        return 0.0;
    }

    let coc = (dof_params.aperture * dof_params.focal_length * distance_diff) / denominator;

    // Scale by sensor size and convert to screen space
    let coc_pixels = coc * dof_params.sensor_size * dof_params.blur_radius_scale;

    // Apply bias and clamp to maximum radius
    return clamp(coc_pixels + dof_params.coc_bias, 0.0, dof_params.max_blur_radius);
}

// Determine if a point is in near field, far field, or in focus
fn get_field_type(depth: f32) -> u32 {
    if (depth < dof_params.focus_distance - dof_params.near_transition_range) {
        return 0u; // Near field
    } else if (depth > dof_params.focus_distance + dof_params.far_transition_range) {
        return 2u; // Far field
    } else {
        return 1u; // In focus
    }
}

// Sample color with bilateral filtering (depth-aware)
fn sample_bilateral(uv: vec2<f32>, center_depth: f32, blur_radius: f32) -> vec4<f32> {
    let sample_depth = textureSampleLevel(depth_texture, color_sampler, uv, 0.0).r;
    let depth_diff = abs(sample_depth - center_depth);

    // Reduce weight for samples with significantly different depths
    let depth_weight = exp(-depth_diff * 10.0);
    let spatial_weight = 1.0; // Could add spatial weighting here

    let color = textureSampleLevel(color_texture, color_sampler, uv, 0.0);
    let weight = depth_weight * spatial_weight;

    return vec4<f32>(color.rgb * weight, weight);
}

// Rotate a 2D vector by the bokeh rotation angle
fn rotate_sample(sample: vec2<f32>) -> vec2<f32> {
    let cos_angle = cos(dof_params.bokeh_rotation);
    let sin_angle = sin(dof_params.bokeh_rotation);

    return vec2<f32>(
        sample.x * cos_angle - sample.y * sin_angle,
        sample.x * sin_angle + sample.y * cos_angle
    );
}

// Gather blur using Poisson disk sampling
fn gather_blur_poisson(uv: vec2<f32>, blur_radius: f32, center_depth: f32) -> vec4<f32> {
    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    let sample_count = min(dof_params.sample_count, 32u);

    for (var i = 0u; i < sample_count; i++) {
        var sample_offset: vec2<f32>;

        if (sample_count <= 16u) {
            sample_offset = POISSON_SAMPLES_16[i];
        } else {
            sample_offset = POISSON_SAMPLES_32[i];
        }

        // Rotate and scale the sample
        sample_offset = rotate_sample(sample_offset) * blur_radius * dof_params.inv_screen_size;
        let sample_uv = uv + sample_offset;

        // Check bounds
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        let sample_result = sample_bilateral(sample_uv, center_depth, blur_radius);
        total_color += sample_result.rgb;
        total_weight += sample_result.a;
    }

    if (total_weight > 0.0) {
        total_color /= total_weight;
    }

    return vec4<f32>(total_color, 1.0);
}

// Gather blur using hexagonal pattern for better bokeh
fn gather_blur_hexagonal(uv: vec2<f32>, blur_radius: f32, center_depth: f32) -> vec4<f32> {
    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    // Multi-ring hexagonal sampling
    let ring_count = max(1u, dof_params.quality_level + 1u);

    for (var ring = 0u; ring < ring_count; ring++) {
        let ring_radius = f32(ring + 1u) / f32(ring_count);

        for (var i = 0u; i < 7u; i++) {
            var sample_offset = HEX_SAMPLES[i] * ring_radius;

            // Skip center sample for outer rings
            if (ring > 0u && i == 0u) {
                continue;
            }

            // Rotate and scale the sample
            sample_offset = rotate_sample(sample_offset) * blur_radius * dof_params.inv_screen_size;
            let sample_uv = uv + sample_offset;

            // Check bounds
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }

            let sample_result = sample_bilateral(sample_uv, center_depth, blur_radius);
            total_color += sample_result.rgb;
            total_weight += sample_result.a;
        }
    }

    if (total_weight > 0.0) {
        total_color /= total_weight;
    }

    return vec4<f32>(total_color, 1.0);
}

// Main DOF computation
@compute @workgroup_size(8, 8)
fn cs_dof(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coord = vec2<i32>(global_id.xy);
    let screen_size = vec2<i32>(dof_params.screen_size);

    // Check bounds
    if (pixel_coord.x >= screen_size.x || pixel_coord.y >= screen_size.y) {
        return;
    }

    let uv = (vec2<f32>(pixel_coord) + 0.5) / dof_params.screen_size;

    // Sample depth and calculate CoC
    let depth = textureSampleLevel(depth_texture, color_sampler, uv, 0.0).r;
    
    // M3: Use tilt-shift CoC calculation when tilt is enabled
    let has_tilt = abs(dof_params.tilt_pitch) > 0.001 || abs(dof_params.tilt_yaw) > 0.001;
    var coc: f32;
    if (has_tilt) {
        coc = calculate_coc_tilt(depth, uv);
    } else {
        coc = calculate_coc(depth);
    }
    let field_type = get_field_type(depth);

    var final_color: vec4<f32>;

    // Debug modes
    if (dof_params.debug_mode == 1u) {
        // Show CoC as grayscale
        let coc_normalized = coc / dof_params.max_blur_radius;
        final_color = vec4<f32>(coc_normalized, coc_normalized, coc_normalized, 1.0);
    } else if (dof_params.debug_mode == 2u) {
        // Show field types as colors
        if (field_type == 0u) {
            final_color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Near field: red
        } else if (field_type == 2u) {
            final_color = vec4<f32>(0.0, 0.0, 1.0, 1.0); // Far field: blue
        } else {
            final_color = vec4<f32>(0.0, 1.0, 0.0, 1.0); // In focus: green
        }
    } else {
        // Standard DOF rendering
        let base_color = textureSampleLevel(color_texture, color_sampler, uv, 0.0);

        if (coc < 0.5) {
            // Sharp/in-focus region
            final_color = base_color;
        } else {
            // Apply blur based on quality setting
            if (dof_params.quality_level <= 1u) {
                // Low/Medium quality: Poisson disk sampling
                final_color = gather_blur_poisson(uv, coc, depth);
            } else {
                // High/Ultra quality: Hexagonal pattern for better bokeh
                final_color = gather_blur_hexagonal(uv, coc, depth);
            }

            // Blend with sharp image based on transition
            let blur_amount = clamp((coc - 0.5) / 2.0, 0.0, 1.0);
            final_color = mix(base_color, final_color, blur_amount);
        }
    }

    // Apply CoC visualization if enabled
    if (dof_params.show_coc == 1u && dof_params.debug_mode == 0u) {
        let coc_overlay = coc / dof_params.max_blur_radius;
        final_color = vec4<f32>(
            mix(final_color.rgb, vec3<f32>(1.0, 1.0, 0.0), coc_overlay * 0.3),
            final_color.a,
        );
    }

    textureStore(dof_output, pixel_coord, final_color);
}

// Separable blur pass for performance optimization (optional)
@compute @workgroup_size(8, 8)
fn cs_dof_separable_h(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Horizontal pass implementation for separable blur
    let pixel_coord = vec2<i32>(global_id.xy);
    let screen_size = vec2<i32>(dof_params.screen_size);

    if (pixel_coord.x >= screen_size.x || pixel_coord.y >= screen_size.y) {
        return;
    }

    let uv = (vec2<f32>(pixel_coord) + 0.5) / dof_params.screen_size;
    let depth = textureSampleLevel(depth_texture, color_sampler, uv, 0.0).r;
    let coc = calculate_coc(depth);

    if (coc < 0.5) {
        // No blur needed
        let color = textureSampleLevel(color_texture, color_sampler, uv, 0.0);
        textureStore(dof_output, pixel_coord, color);
        return;
    }

    // Horizontal Gaussian blur
    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    let blur_radius = min(coc, dof_params.max_blur_radius);
    let sample_step = blur_radius / 4.0;

    for (var i = -4; i <= 4; i++) {
        let offset = f32(i) * sample_step * dof_params.inv_screen_size.x;
        let sample_uv = vec2<f32>(uv.x + offset, uv.y);

        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0) {
            let weight = exp(-f32(i * i) * 0.25); // Gaussian weight
            let color = textureSampleLevel(color_texture, color_sampler, sample_uv, 0.0);

            total_color += color.rgb * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        total_color /= total_weight;
    }

    textureStore(dof_output, pixel_coord, vec4<f32>(total_color, 1.0));
}

@compute @workgroup_size(8, 8)
fn cs_dof_separable_v(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Vertical pass implementation for separable blur
    let pixel_coord = vec2<i32>(global_id.xy);
    let screen_size = vec2<i32>(dof_params.screen_size);

    if (pixel_coord.x >= screen_size.x || pixel_coord.y >= screen_size.y) {
        return;
    }

    let uv = (vec2<f32>(pixel_coord) + 0.5) / dof_params.screen_size;
    let depth = textureSampleLevel(depth_texture, color_sampler, uv, 0.0).r;
    let coc = calculate_coc(depth);

    if (coc < 0.5) {
        // No blur needed
        let color = textureSampleLevel(color_texture, color_sampler, uv, 0.0);
        textureStore(dof_output, pixel_coord, color);
        return;
    }

    // Vertical Gaussian blur
    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    let blur_radius = min(coc, dof_params.max_blur_radius);
    let sample_step = blur_radius / 4.0;

    for (var i = -4; i <= 4; i++) {
        let offset = f32(i) * sample_step * dof_params.inv_screen_size.y;
        let sample_uv = vec2<f32>(uv.x, uv.y + offset);

        if (sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            let weight = exp(-f32(i * i) * 0.25); // Gaussian weight
            let color = textureSampleLevel(color_texture, color_sampler, sample_uv, 0.0);

            total_color += color.rgb * weight;
            total_weight += weight;
        }
    }

    if (total_weight > 0.0) {
        total_color /= total_weight;
    }

    textureStore(dof_output, pixel_coord, vec4<f32>(total_color, 1.0));
}
