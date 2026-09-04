// Physical sky models: Hosek-Wilkie and Preetham (P6)
// Implements analytic atmospheric scattering for realistic sky rendering

struct SkyParams {
    sun_direction_turbidity: vec4<f32>,
    ground_albedo_sun_size_sun_intensity_exposure: vec4<f32>,
    model_pad: vec4<u32>,
    hosek_coeffs_a_d: array<vec4<f32>, 3>,
    hosek_coeffs_e_h: array<vec4<f32>, 3>,
    hosek_coeff_i: vec4<f32>,
    hosek_radiance: vec4<f32>,
}

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;

fn sky_sun_direction(params: SkyParams) -> vec3<f32> {
    return params.sun_direction_turbidity.xyz;
}

fn sky_turbidity(params: SkyParams) -> f32 {
    return params.sun_direction_turbidity.w;
}

fn sky_ground_albedo(params: SkyParams) -> f32 {
    return params.ground_albedo_sun_size_sun_intensity_exposure.x;
}

fn sky_sun_size(params: SkyParams) -> f32 {
    return params.ground_albedo_sun_size_sun_intensity_exposure.y;
}

fn sky_sun_intensity(params: SkyParams) -> f32 {
    return params.ground_albedo_sun_size_sun_intensity_exposure.z;
}

fn sky_exposure(params: SkyParams) -> f32 {
    return params.ground_albedo_sun_size_sun_intensity_exposure.w;
}

fn sky_model(params: SkyParams) -> u32 {
    return params.model_pad.x;
}

// ============================================================================
// Hosek-Wilkie sky model (2012) - More accurate than Preetham
// ============================================================================

// Hosek-Wilkie sky model coefficients (precomputed for RGB channels).
// Values are cooked on the CPU from the published native RGB coefficient tables.
struct HosekCoeffs {
    A: vec3<f32>,
    B: vec3<f32>,
    C: vec3<f32>,
    D: vec3<f32>,
    E: vec3<f32>,
    F: vec3<f32>,
    G: vec3<f32>,
    H: vec3<f32>,
    I: vec3<f32>,
}

fn hosek_wilkie_eval_channel(
    cos_theta: f32,     // cos(angle between view and zenith)
    cos_gamma: f32,     // cos(angle between view and sun)
    abcd: vec4<f32>,
    efgh: vec4<f32>,
    I: f32,
    radiance: f32
) -> f32 {
    let gamma = acos(clamp(cos_gamma, -1.0, 1.0));
    let ray_m = cos_gamma * cos_gamma;
    let mie_denom = max(1.0e-4, 1.0 + I * I - 2.0 * I * cos_gamma);
    let mie_m = (1.0 + ray_m) / pow(mie_denom, 1.5);
    let zenith = sqrt(max(0.0, cos_theta));

    return radiance
        * (1.0 + abcd.x * exp(abcd.y / (cos_theta + 0.01)))
        * (abcd.z + abcd.w * exp(efgh.x * gamma) + efgh.y * ray_m + efgh.z * mie_m + efgh.w * zenith);
}

fn approximate_hosek_eval_channel(
    cos_theta: f32,
    cos_gamma: f32,
    A: f32, B: f32, C: f32, D: f32, E: f32, F: f32, G: f32, H: f32, I: f32
) -> f32 {
    let gamma = acos(clamp(cos_gamma, -1.0, 1.0));
    let chi = (1.0 + cos_gamma * cos_gamma) / pow(1.0 + H * H - 2.0 * H * cos_gamma, 1.5);
    let exp_term = exp(E * gamma);
    return (1.0 + A * exp(B / (cos_theta + 0.01))) *
           (C + D * exp_term + F * chi + G * cos_gamma + I * sqrt(max(0.0, cos_theta)));
}

fn approximate_hosek_compute_coeffs(turbidity: f32, albedo: f32, sun_elevation: f32) -> HosekCoeffs {
    let t = clamp(turbidity, 1.0, 10.0);

    var coeffs: HosekCoeffs;

    // RGB channel coefficients (approximate)
    // Red channel
    coeffs.A.x = -1.0 + 0.1 * t;
    coeffs.B.x = -0.3 + 0.05 * t;
    coeffs.C.x = 0.1 + 0.8 * t;
    coeffs.D.x = -1.2 + 0.15 * t;
    coeffs.E.x = 0.06;
    coeffs.F.x = -0.9 + 0.1 * t;
    coeffs.G.x = 0.2;
    coeffs.H.x = 4.0 - 0.3 * t;
    coeffs.I.x = 0.35;

    // Green channel
    coeffs.A.y = -1.1 + 0.12 * t;
    coeffs.B.y = -0.32 + 0.06 * t;
    coeffs.C.y = 0.2 + 0.7 * t;
    coeffs.D.y = -1.3 + 0.18 * t;
    coeffs.E.y = 0.065;
    coeffs.F.y = -1.0 + 0.12 * t;
    coeffs.G.y = 0.18;
    coeffs.H.y = 4.2 - 0.35 * t;
    coeffs.I.y = 0.4;

    // Blue channel
    coeffs.A.z = -1.2 + 0.15 * t;
    coeffs.B.z = -0.35 + 0.07 * t;
    coeffs.C.z = 0.3 + 0.6 * t;
    coeffs.D.z = -1.4 + 0.2 * t;
    coeffs.E.z = 0.07;
    coeffs.F.z = -1.1 + 0.15 * t;
    coeffs.G.z = 0.15;
    coeffs.H.z = 4.5 - 0.4 * t;
    coeffs.I.z = 0.45;

    // Ground albedo influence
    let albedo_factor = 1.0 + albedo * 0.3;
    coeffs.C = coeffs.C * albedo_factor;

    return coeffs;
}

fn eval_hosek_wilkie(view_dir: vec3<f32>, params: SkyParams) -> vec3<f32> {
    let cos_theta = max(0.0, view_dir.y);  // angle to zenith
    let sun_direction = sky_sun_direction(params);
    let cos_gamma = dot(view_dir, sun_direction);  // angle to sun

    var sky_color: vec3<f32>;
    sky_color.x = hosek_wilkie_eval_channel(
        cos_theta,
        cos_gamma,
        params.hosek_coeffs_a_d[0],
        params.hosek_coeffs_e_h[0],
        params.hosek_coeff_i.x,
        params.hosek_radiance.x,
    );
    sky_color.y = hosek_wilkie_eval_channel(
        cos_theta,
        cos_gamma,
        params.hosek_coeffs_a_d[1],
        params.hosek_coeffs_e_h[1],
        params.hosek_coeff_i.y,
        params.hosek_radiance.y,
    );
    sky_color.z = hosek_wilkie_eval_channel(
        cos_theta,
        cos_gamma,
        params.hosek_coeffs_a_d[2],
        params.hosek_coeffs_e_h[2],
        params.hosek_coeff_i.z,
        params.hosek_radiance.z,
    );

    return max(sky_color, vec3<f32>(0.0));
}

fn eval_approximate_hosek(view_dir: vec3<f32>, params: SkyParams) -> vec3<f32> {
    let cos_theta = max(0.0, view_dir.y);
    let sun_direction = sky_sun_direction(params);
    let cos_gamma = dot(view_dir, sun_direction);
    let cos_theta_sun = max(0.0, sun_direction.y);
    let sun_elevation = asin(cos_theta_sun);
    let coeffs = approximate_hosek_compute_coeffs(
        sky_turbidity(params),
        sky_ground_albedo(params),
        sun_elevation,
    );

    var sky_color: vec3<f32>;
    sky_color.x = approximate_hosek_eval_channel(cos_theta, cos_gamma,
        coeffs.A.x, coeffs.B.x, coeffs.C.x, coeffs.D.x, coeffs.E.x,
        coeffs.F.x, coeffs.G.x, coeffs.H.x, coeffs.I.x);
    sky_color.y = approximate_hosek_eval_channel(cos_theta, cos_gamma,
        coeffs.A.y, coeffs.B.y, coeffs.C.y, coeffs.D.y, coeffs.E.y,
        coeffs.F.y, coeffs.G.y, coeffs.H.y, coeffs.I.y);
    sky_color.z = approximate_hosek_eval_channel(cos_theta, cos_gamma,
        coeffs.A.z, coeffs.B.z, coeffs.C.z, coeffs.D.z, coeffs.E.z,
        coeffs.F.z, coeffs.G.z, coeffs.H.z, coeffs.I.z);

    let zenith_y = approximate_hosek_eval_channel(1.0, cos_theta_sun,
        coeffs.A.y, coeffs.B.y, coeffs.C.y, coeffs.D.y, coeffs.E.y,
        coeffs.F.y, coeffs.G.y, coeffs.H.y, coeffs.I.y);

    sky_color = sky_color / max(zenith_y, 0.01);
    return max(sky_color, vec3<f32>(0.0));
}

// ============================================================================
// Preetham sky model (1999) - Classic analytic sky
// ============================================================================

fn preetham_perez_function(
    cos_theta: f32,
    cos_gamma: f32,
    A: f32, B: f32, C: f32, D: f32, E: f32
) -> f32 {
    let gamma = acos(clamp(cos_gamma, -1.0, 1.0));
    let cos_gamma_sq = cos_gamma * cos_gamma;

    let num = (1.0 + A * exp(B / (cos_theta + 0.01))) *
              (1.0 + C * exp(D * gamma) + E * cos_gamma_sq);

    return num;
}

fn preetham_compute_coeffs(turbidity: f32) -> vec3<f32> {
    let t = clamp(turbidity, 1.0, 10.0);

    // Preetham model coefficients for Y (luminance) channel
    let A = 0.1787 * t - 1.4630;
    let B = -0.3554 * t + 0.4275;
    let C = -0.0227 * t + 5.3251;
    let D = 0.1206 * t - 2.5771;
    let E = -0.0670 * t + 0.3703;

    return vec3<f32>(A, B, C);  // Simplified, full model has more coefficients
}

fn eval_preetham(view_dir: vec3<f32>, params: SkyParams) -> vec3<f32> {
    let cos_theta = max(0.0, view_dir.y);
    let sun_direction = sky_sun_direction(params);
    let cos_gamma = dot(view_dir, sun_direction);
    let cos_theta_sun = max(0.0, sun_direction.y);

    let t = sky_turbidity(params);

    // Preetham luminance coefficients
    let A = 0.1787 * t - 1.4630;
    let B = -0.3554 * t + 0.4275;
    let C = -0.0227 * t + 5.3251;
    let D = 0.1206 * t - 2.5771;
    let E = -0.0670 * t + 0.3703;

    // Compute luminance
    let F = preetham_perez_function(cos_theta, cos_gamma, A, B, C, D, E);
    let F_zenith = preetham_perez_function(1.0, cos_theta_sun, A, B, C, D, E);

    let Y = F / max(F_zenith, 0.01);

    // Simple RGB approximation based on sun angle and turbidity
    let sun_angle = acos(cos_theta_sun);
    let sunset_factor = smoothstep(1.4, 1.8, sun_angle);  // Reddish near horizon

    var sky_color: vec3<f32>;

    // Sky color varies with sun elevation
    if (cos_theta_sun > 0.1) {
        // Daytime sky - blue
        sky_color = vec3<f32>(0.3, 0.5, 1.0) * Y;
    } else {
        // Sunrise/sunset - orange to red gradient
        let horizon_color = vec3<f32>(1.0, 0.6, 0.3);
        let zenith_color = vec3<f32>(0.4, 0.5, 0.8);
        sky_color = mix(zenith_color, horizon_color, sunset_factor) * Y;
    }

    // Add turbidity tint (hazier sky is more white/gray)
    let haze_tint = vec3<f32>(1.0) * (t - 2.0) / 8.0;
    sky_color = mix(sky_color, haze_tint, min(t / 10.0, 0.5));

    // Ground albedo contribution
    sky_color = sky_color * (1.0 + sky_ground_albedo(params) * 0.2);

    return max(sky_color, vec3<f32>(0.0));
}

// ============================================================================
// Sun disk rendering
// ============================================================================

fn render_sun_disk(
    view_dir: vec3<f32>,
    sun_dir: vec3<f32>,
    intensity: f32,
    sun_size: f32
) -> vec3<f32> {
    let cos_angle = dot(view_dir, sun_dir);

    // Sun angular diameter is ~0.53 degrees = ~0.0093 radians
    let sun_radius = 0.0093 * max(sun_size, 0.01);
    let sun_cos_radius = cos(sun_radius);

    if (cos_angle >= sun_cos_radius) {
        // Inside sun disk
        let sun_color = vec3<f32>(1.0, 0.95, 0.9);
        let limb_darkening = smoothstep(sun_cos_radius, 1.0, cos_angle);
        return sun_color * intensity * limb_darkening * 50.0;
    }

    // Sun corona/glow
    let glow_angle = max(0.05 * max(sun_size, 0.25), sun_radius * 2.0);
    let glow_cos = cos(glow_angle);
    if (cos_angle >= glow_cos) {
        let glow_factor = smoothstep(glow_cos, sun_cos_radius, cos_angle);
        return vec3<f32>(1.0, 0.8, 0.6) * glow_factor * intensity * 2.0;
    }

    return vec3<f32>(0.0);
}

fn render_solar_scattering(view_dir: vec3<f32>, params: SkyParams) -> vec3<f32> {
    let sun_dir = sky_sun_direction(params);
    let sun_alignment = max(dot(view_dir, sun_dir), 0.0);
    let sun_elevation = max(sun_dir.y, 0.0);
    let low_sun = 1.0 - smoothstep(0.18, 0.72, sun_elevation);
    let haze = clamp((sky_turbidity(params) - 1.0) / 9.0, 0.0, 1.0);
    let intensity = sky_sun_intensity(params);
    let size_norm = clamp(sky_sun_size(params) / 4.0, 0.0, 1.0);
    let horizon = 1.0 - clamp(view_dir.y, 0.0, 1.0);

    let forward_focus = mix(22.0, 4.0, size_norm);
    let forward_scatter = pow(sun_alignment, forward_focus);
    let broad_scatter = pow(sun_alignment, mix(10.0, 2.5, size_norm));
    let horizon_glow = pow(horizon, 2.0) * low_sun * (0.35 + haze * 0.35 + size_norm * 0.2);
    let ambient_scatter = intensity * (0.02 + haze * 0.03);

    let sunset_color = mix(
        vec3<f32>(1.0, 0.95, 0.9),
        vec3<f32>(1.0, 0.72, 0.42),
        low_sun * (0.75 + haze * 0.2),
    );
    let daylight_color = mix(
        vec3<f32>(1.0, 0.97, 0.92),
        vec3<f32>(1.0, 0.9, 0.78),
        haze * 0.6,
    );
    let scatter_color = mix(daylight_color, sunset_color, low_sun);

    return scatter_color * (
        forward_scatter * intensity * 0.35
        + broad_scatter * intensity * (0.06 + size_norm * 0.08)
        + horizon_glow * intensity * 0.22
        + ambient_scatter
    );
}

// ============================================================================
// Main sky evaluation function
// ============================================================================

fn eval_sky(view_dir: vec3<f32>, params: SkyParams) -> vec3<f32> {
    let normalized_view = normalize(view_dir);

    var sky_color: vec3<f32>;

    if (sky_model(params) == 1u) {
        sky_color = eval_hosek_wilkie(normalized_view, params);
    } else if (sky_model(params) == 2u) {
        sky_color = eval_approximate_hosek(normalized_view, params);
    } else {
        sky_color = eval_preetham(normalized_view, params);
    }

    // Declared SIDERA civil-to-astronomical smoothstep rendering model.
    // It fades the daylight fit from solar altitude -4 degrees to -18 degrees.
    let solar_altitude = asin(clamp(sky_sun_direction(params).y, -1.0, 1.0)) * 180.0 / PI;
    let daylight = smoothstep(-18.0, -4.0, solar_altitude);
    let horizon = 1.0 - clamp(normalized_view.y, 0.0, 1.0);
    let night_color = mix(
        vec3<f32>(0.002, 0.003, 0.009),
        vec3<f32>(0.008, 0.012, 0.024),
        horizon * horizon,
    );
    sky_color = mix(night_color, sky_color, daylight);

    // Add sun disk
    let sun_contribution = render_sun_disk(
        normalized_view,
        sky_sun_direction(params),
        sky_sun_intensity(params),
        sky_sun_size(params),
    );
    sky_color = sky_color + sun_contribution + render_solar_scattering(normalized_view, params);

    // Apply exposure
    sky_color = sky_color * sky_exposure(params);

    // Simple tonemapping
    sky_color = sky_color / (sky_color + vec3<f32>(1.0));

    return sky_color;
}

// ============================================================================
// Compute shader for full-screen sky rendering
// ============================================================================

@group(0) @binding(0) var<uniform> sky_params: SkyParams;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
    _pad0: f32,
}

@group(1) @binding(0) var<uniform> camera: CameraUniforms;

@compute @workgroup_size(8, 8, 1)
fn cs_render_sky(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = global_id.xy;
    let dims = textureDimensions(output_texture);

    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }

    // Compute view ray direction
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Reconstruct view direction
    let clip_pos = vec4<f32>(ndc, 1.0, 1.0);
    let view_pos = camera.inv_proj * clip_pos;
    let view_dir_vs = normalize(view_pos.xyz / view_pos.w);

    // Transform to world space
    let view_dir_ws = normalize((camera.inv_view * vec4<f32>(view_dir_vs, 0.0)).xyz);

    // Evaluate sky
    let sky_color = eval_sky(view_dir_ws, sky_params);

    textureStore(output_texture, pixel, vec4<f32>(sky_color, 1.0));
}

// ============================================================================
// Fragment shader variant for rasterized sky dome
// ============================================================================

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) view_dir: vec3<f32>,
}

@vertex
fn vs_sky_dome(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );

    var output: VertexOutput;
    output.position = vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0);

    // Reconstruct view direction (will be interpolated)
    let clip_pos = vec4<f32>(output.position.xy, 1.0, 1.0);
    let view_pos = camera.inv_proj * clip_pos;
    output.view_dir = (camera.inv_view * vec4<f32>(normalize(view_pos.xyz), 0.0)).xyz;

    return output;
}

@fragment
fn fs_sky_dome(in: VertexOutput) -> @location(0) vec4<f32> {
    let sky_color = eval_sky(normalize(in.view_dir), sky_params);
    return vec4<f32>(sky_color, 1.0);
}
