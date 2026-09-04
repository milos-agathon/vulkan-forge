// SIDERA star/planet/Moon overlay. NightInstance is 64 bytes in Rust and WGSL.

struct NightInstance {
    direction_radius: vec4<f32>,
    right_kind: vec4<f32>,
    up_phase: vec4<f32>,
    color_intensity: vec4<f32>,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
    _pad0: f32,
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<NightInstance>;
@group(0) @binding(1) var moon_albedo: texture_2d<f32>;
@group(0) @binding(2) var moon_sampler: sampler;
@group(1) @binding(0) var<uniform> camera: CameraUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) kind: u32,
    @location(2) @interpolate(flat) phase: f32,
    @location(3) @interpolate(flat) color_intensity: vec4<f32>,
    @location(4) @interpolate(flat) center_px: vec2<f32>,
}

fn quad_corner(index: u32) -> vec2<f32> {
    switch index {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u, 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>(1.0, 1.0); }
    }
}

@vertex
fn vs_night(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let item = instances[instance_index];
    let corner = quad_corner(vertex_index);
    let direction = item.direction_radius.xyz;
    let center_view_direction = (camera.view * vec4<f32>(direction, 0.0)).xyz;

    var output: VertexOutput;
    // Celestial objects are at infinity, so only the clip-space XY/W of the
    // ray matters. Pin Z to the near plane instead of letting it fall out of
    // the unit-length direction: the terrain camera clamps `near` as high as
    // 1.0 world unit (frame_anchor.rs), which would clip a unit-distance quad
    // away everywhere except exactly on the optical axis. This pass has no
    // depth attachment, so the pinned Z affects clipping only.
    let center_clip = camera.proj * vec4<f32>(center_view_direction, 1.0);
    let center_ndc = center_clip.xy / center_clip.w;
    output.center_px = vec2<f32>(
        (center_ndc.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - center_ndc.y * 0.5) * camera.viewport.y,
    );
    if (u32(item.right_kind.w) == 0u) {
        let pixel_offset = vec2<f32>(
            2.0 * corner.x * item.direction_radius.w / camera.viewport.x,
            2.0 * corner.y * item.direction_radius.w / camera.viewport.y,
        );
        output.position = vec4<f32>(
            center_clip.xy + pixel_offset * center_clip.w,
            0.0,
            center_clip.w,
        );
    } else {
        let world_direction = normalize(
            direction
            + item.direction_radius.w
                * (item.right_kind.xyz * corner.x + item.up_phase.xyz * corner.y)
        );
        let view_direction = (camera.view * vec4<f32>(world_direction, 0.0)).xyz;
        let clip = camera.proj * vec4<f32>(view_direction, 1.0);
        output.position = vec4<f32>(clip.xy, 0.0, clip.w);
    }
    if (direction.y <= 0.0) {
        output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    output.uv = corner * 0.5 + 0.5;
    output.kind = u32(item.right_kind.w);
    output.phase = item.up_phase.w;
    output.color_intensity = item.color_intensity;
    return output;
}

@fragment
fn fs_night(input: VertexOutput) -> @location(0) vec4<f32> {
    if (input.kind == 0u) {
        // Pixel-integrated separable tent. For any fractional point position,
        // the weights of its at-most-four recipient pixels sum to exactly one.
        let delta = abs(input.position.xy - input.center_px);
        let weight = max(1.0 - delta.x, 0.0) * max(1.0 - delta.y, 0.0);
        return vec4<f32>(
            input.color_intensity.rgb,
            weight * clamp(input.color_intensity.w, 0.0, 1.0) * input.phase,
        );
    }
    let local = input.uv * 2.0 - 1.0;
    let radius_squared = dot(local, local);
    if (radius_squared > 1.0) {
        discard;
    }

    if (input.kind == 2u) {
        let normal = vec3<f32>(local, sqrt(max(0.0, 1.0 - radius_squared)));
        let texture_uv = vec2<f32>(
            0.5 + atan2(normal.x, normal.z) / (2.0 * 3.14159265359),
            0.5 - asin(normal.y) / 3.14159265359,
        );
        let albedo = textureSample(moon_albedo, moon_sampler, texture_uv).r;
        // Exact illuminated-sphere geometry: k = (1 + cos(phase)) / 2.
        // The projected Sun lies along +local.x, so the bright limb faces the Sun.
        let to_sun = vec3<f32>(sin(input.phase), 0.0, cos(input.phase));
        let illumination = max(dot(normal, to_sun), 0.0);
        let edge = smoothstep(1.0, 0.94, radius_squared);
        // `up_phase.w` carries the phase angle; the disc alpha is independent
        // of the star-only twilight visibility ramp.
        return vec4<f32>(
            input.color_intensity.rgb * albedo * (0.015 + 1.35 * illumination),
            edge * input.color_intensity.w,
        );
    }

    let radial = sqrt(radius_squared);
    let profile = 1.0 - smoothstep(0.15, 1.0, radial);
    if (input.kind == 1u) {
        return vec4<f32>(
            input.color_intensity.rgb * (0.45 + input.color_intensity.w),
            profile * input.phase,
        );
    }
    return vec4<f32>(0.0);
}
