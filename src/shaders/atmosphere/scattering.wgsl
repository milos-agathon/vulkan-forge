// AETHER runtime atmosphere evaluation. This source is assembled after the
// terrain sky/camera prelude; all AETHER resources occupy group 2.

struct AtmosphereScatteringUniforms {
    planet_radii_path: vec4<f32>,
    mie_ground_scales: vec4<f32>,
    sun_direction_intensity: vec4<f32>,
    camera_exposure_density_model: vec4<f32>,
    lut_dimensions0: vec4<u32>,
    lut_dimensions1: vec4<u32>,
}

@group(2) @binding(0) var<uniform> atmosphere: AtmosphereScatteringUniforms;
@group(2) @binding(1) var atmosphere_transmittance_lut: texture_2d<f32>;
// Accumulated single + higher orders. It is sampled exactly once.
@group(2) @binding(2) var atmosphere_scattering_lut: texture_3d<f32>;
@group(2) @binding(3) var atmosphere_aerial_lut: texture_3d<f32>;

const AETHER_PI: f32 = 3.141592653589793;

fn phase_rayleigh(cos_theta: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    return 3.0 * (1.0 + c * c) / (16.0 * AETHER_PI);
}

fn phase_mie(cos_theta: f32, g_raw: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    let g = clamp(g_raw, -0.999, 0.999);
    let gg = g * g;
    return 3.0 * (1.0 - gg) * (1.0 + c * c)
        / (8.0 * AETHER_PI * (2.0 + gg)
            * pow(max(1.0 + gg - 2.0 * g * c, 1.0e-6), 1.5));
}

fn atmosphere_load_2d_linear(tex: texture_2d<f32>, uv_raw: vec2<f32>) -> vec4<f32> {
    let dims_u = textureDimensions(tex, 0);
    let p = clamp(uv_raw, vec2<f32>(0.0), vec2<f32>(1.0))
        * (vec2<f32>(dims_u) - vec2<f32>(1.0));
    let lo = vec2<i32>(floor(p));
    let hi = min(lo + vec2<i32>(1), vec2<i32>(dims_u) - vec2<i32>(1));
    let f = fract(p);
    let a = mix(textureLoad(tex, lo, 0), textureLoad(tex, vec2<i32>(hi.x, lo.y), 0), f.x);
    let b = mix(textureLoad(tex, vec2<i32>(lo.x, hi.y), 0), textureLoad(tex, hi, 0), f.x);
    return mix(a, b, f.y);
}

fn atmosphere_distance_to_boundary(altitude_m: f32, mu: f32, params: AtmosphereScatteringUniforms) -> f32 {
    let bottom = params.planet_radii_path.x;
    let top = params.planet_radii_path.y;
    let radius = bottom + clamp(altitude_m, 0.0, top - bottom);
    let top_disc = radius * radius * mu * mu + (top - radius) * (top + radius);
    let top_distance = max(-radius * mu + sqrt(max(top_disc, 0.0)), 0.0);
    let ground_disc = radius * radius * mu * mu
        - (radius - bottom) * (radius + bottom);
    if (mu < 0.0 && ground_disc >= 0.0) {
        return max(-radius * mu - sqrt(ground_disc), 0.0);
    }
    return top_distance;
}

fn atmosphere_ray_hits_ground(altitude_m: f32, mu: f32,
    params: AtmosphereScatteringUniforms) -> bool {
    if (mu >= 0.0) { return false; }
    let bottom = params.planet_radii_path.x;
    let top = params.planet_radii_path.y;
    let radius = bottom + clamp(altitude_m, 0.0, top - bottom);
    let ground_disc = radius * radius * mu * mu
        - (radius - bottom) * (radius + bottom);
    return ground_disc >= 0.0 && -radius * mu - sqrt(max(ground_disc, 0.0)) >= 0.0;
}

fn sample_transmittance(
    transmittance_lut: texture_2d<f32>,
    altitude_m: f32,
    zenith_cosine: f32,
    distance_m: f32,
    params: AtmosphereScatteringUniforms,
) -> vec3<f32> {
    let atmosphere_height = max(params.planet_radii_path.y - params.planet_radii_path.x, 1.0);
    // Transmittance retains its linear cosine axis; only scattering axes use
    // the nonlinear horizon/forward-lobe parameterization.
    let uv = vec2<f32>(0.5 * (clamp(zenith_cosine, -1.0, 1.0) + 1.0),
        clamp(altitude_m / atmosphere_height, 0.0, 1.0));
    let boundary_t = clamp(atmosphere_load_2d_linear(transmittance_lut, uv).rgb,
        vec3<f32>(1.0e-6), vec3<f32>(1.0));
    let boundary_distance = max(atmosphere_distance_to_boundary(altitude_m, zenith_cosine, params), 1.0);
    return pow(boundary_t, vec3<f32>(clamp(distance_m / boundary_distance, 0.0, 1.0)));
}

fn sample_inscatter(
    accumulated_scatter_lut: texture_3d<f32>,
    altitude_m: f32,
    zenith_cosine: f32,
    sun_zenith_cosine: f32,
    view_sun_cosine: f32,
    params: AtmosphereScatteringUniforms,
) -> vec3<f32> {
    let atmosphere_height = max(params.planet_radii_path.y - params.planet_radii_path.x, 1.0);
    return aether_eval_sample_accumulated_scattering(
        accumulated_scatter_lut,
        altitude_m / atmosphere_height,
        sun_zenith_cosine,
        zenith_cosine,
        view_sun_cosine,
        params.lut_dimensions1.x,
        params.lut_dimensions1.y,
    );
}

fn sky_radiance(view_dir_y_up: vec3<f32>, sun_dir_y_up: vec3<f32>,
    camera_altitude_m: f32, multiple_scatter_lut: texture_3d<f32>,
    params: AtmosphereScatteringUniforms) -> vec3<f32> {
    let view = normalize(view_dir_y_up); let sun = normalize(sun_dir_y_up);
    return sample_inscatter(multiple_scatter_lut, camera_altitude_m, view.y,
        sun.y, dot(view,sun), params);
}

fn aether_sky_radiance(view_dir_raw: vec3<f32>) -> vec3<f32> {
    let view=normalize(view_dir_raw); let sun=normalize(atmosphere.sun_direction_intensity.xyz);
    let sun_intensity = aether_eval_clamp_radiometric_scale(
        atmosphere.sun_direction_intensity.w);
    let atmosphere_exposure = aether_eval_clamp_radiometric_scale(
        atmosphere.camera_exposure_density_model.y);
    var radiance=sky_radiance(view,sun,atmosphere.camera_exposure_density_model.x,
        atmosphere_scattering_lut,atmosphere)*sun_intensity;
    let sun_radius=0.0093*max(sky_sun_size(sky_params),0.01);
    let alignment=dot(view,sun);
    let sun_visible = !atmosphere_ray_hits_ground(
        atmosphere.camera_exposure_density_model.x, sun.y, atmosphere);
    if (sun_visible && alignment>=cos(sun_radius)) {
        let boundary_t=sample_transmittance(atmosphere_transmittance_lut,
            atmosphere.camera_exposure_density_model.x,sun.y,1.0e30,atmosphere);
        radiance += boundary_t*sun_intensity
            *smoothstep(cos(sun_radius),1.0,alignment)*40.0;
    }
    return aether_eval_clamp_hdr_radiance(radiance*atmosphere_exposure);
}

@compute @workgroup_size(8,8,1)
fn cs_render_aether_sky(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel=gid.xy; let dims=textureDimensions(output_texture);
    if (pixel.x>=dims.x || pixel.y>=dims.y) { return; }
    let uv=(vec2<f32>(pixel)+0.5)/vec2<f32>(dims);
    let clip=vec4<f32>(uv.x*2.0-1.0,1.0-uv.y*2.0,1.0,1.0);
    let view_pos=camera.inv_proj*clip;
    let view_vs=normalize(view_pos.xyz/view_pos.w);
    let view_ws=normalize((camera.inv_view*vec4<f32>(view_vs,0.0)).xyz);
    // Terrain/world cameras are Z-up; the atmosphere LUT is parameterized in
    // Y-up coordinates, matching the already-swizzled sun uniform.
    let view_y_up = view_ws.xzy;
    textureStore(output_texture,pixel,vec4<f32>(aether_sky_radiance(view_y_up),1.0));
}
