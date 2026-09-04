// AETHER aerial-perspective post for the converged PROMETHEUS reference.
//
// The traversal/accumulation shader is intentionally unchanged. This kernel
// consumes its authoritative linear accumulation plus the exact frame-0 depth
// AOV, applies L_out = L_surface*T + L_inscatter, and writes the existing
// RGBA16F output with PROMETHEUS' unchanged Reinhard resolve.

struct PrometheusAetherUniforms {
    // width, height, frame count, wavelength count
    dimensions_frames: vec4<u32>,
    camera_origin_exposure: vec4<f32>,
    camera_right_tan_half_fov: vec4<f32>,
    camera_up_aspect: vec4<f32>,
    // xyz camera forward, w exact AETHER ground albedo
    camera_forward_ground: vec4<f32>,
    sun_direction_intensity: vec4<f32>,
    // bottom radius, top radius, max aerial distance, ozone DU
    planet_radii_path: vec4<f32>,
    // Mie g, turbidity, Rayleigh scale height, Mie scale height
    mie_turbidity_scales: vec4<f32>,
    // trans width/height, scattering height/nu
    lut_dimensions0: vec4<u32>,
    // aerial distance/mu/height, reserved
    lut_dimensions1: vec4<u32>,
}

@group(0) @binding(0)
var<storage, read> prometheus_accum_hdr: array<vec4<f32>>;

@group(0) @binding(1)
var prometheus_depth_aov: texture_2d<f32>;

@group(0) @binding(2)
var prometheus_transmittance_lut: texture_2d<f32>;

@group(0) @binding(3)
var prometheus_scattering_lut: texture_3d<f32>;

@group(0) @binding(4)
var prometheus_aerial_lut: texture_3d<f32>;

@group(0) @binding(5)
var<uniform> prometheus_atmosphere: PrometheusAetherUniforms;

@group(0) @binding(6)
var prometheus_output: texture_storage_2d<rgba16float, write>;

@group(0) @binding(7)
var prometheus_visibility_aov: texture_2d<f32>;

fn prometheus_load_boundary_transmittance(height_unit: f32, mu: f32) -> vec3<f32> {
    let dims = textureDimensions(prometheus_transmittance_lut, 0);
    let coord = vec2<i32>(
        // The transmittance table retains its linear cosine axis. Only the
        // accumulated-scattering table uses the nonlinear horizon mapping.
        i32(round((0.5 * (clamp(mu, -1.0, 1.0) + 1.0)) * f32(max(dims.x, 1u) - 1u))),
        i32(round(clamp(height_unit, 0.0, 1.0) * f32(max(dims.y, 1u) - 1u))),
    );
    return clamp(textureLoad(prometheus_transmittance_lut, coord, 0).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn prometheus_load_endpoint_scattering(
    height_unit: f32,
    mu_sun: f32,
    mu_view: f32,
    nu: f32,
) -> vec3<f32> {
    return aether_eval_sample_accumulated_scattering(
        prometheus_scattering_lut,
        height_unit,
        mu_sun,
        mu_view,
        nu,
        prometheus_atmosphere.lut_dimensions0.z,
        prometheus_atmosphere.lut_dimensions0.w,
    );
}

// The optional aerial froxel is deliberately a scalar transmittance
// accelerator: its RGB channels are specified as zero and are never exposed
// to radiance code. Finite in-scatter comes from the accumulated-scattering
// LUT through the endpoint identity in `main`.
fn prometheus_load_aerial_transmittance(distance_unit: f32, height_unit: f32, mu_view: f32) -> f32 {
    let dims = textureDimensions(prometheus_aerial_lut, 0);
    let coord = vec3<i32>(
        i32(round(clamp(distance_unit, 0.0, 1.0) * f32(max(dims.x, 1u) - 1u))),
        // The aerial LUT is baked on a linear mu_view axis. The nonlinear
        // horizon mapping is exclusive to the accumulated-scattering table.
        i32(round(0.5 * (clamp(mu_view, -1.0, 1.0) + 1.0)
            * f32(max(dims.y, 1u) - 1u))),
        i32(round(clamp(height_unit, 0.0, 1.0) * f32(max(dims.z, 1u) - 1u))),
    );
    return clamp(textureLoad(prometheus_aerial_lut, coord, 0).a, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = prometheus_atmosphere.dimensions_frames.x;
    let height = prometheus_atmosphere.dimensions_frames.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let pixel_index = gid.y * width + gid.x;
    let accumulated = prometheus_accum_hdr[pixel_index];
    let surface_or_environment = aether_eval_clamp_hdr_radiance(
        accumulated.rgb / max(accumulated.a, 1.0),
    );
    let depth = textureLoad(prometheus_depth_aov, vec2<i32>(gid.xy), 0).x;
    let visibility = textureLoad(prometheus_visibility_aov, vec2<i32>(gid.xy), 0).x;

    let ndc_x = ((f32(gid.x) + 0.5) / f32(width)) * 2.0 - 1.0;
    let ndc_y = (1.0 - (f32(gid.y) + 0.5) / f32(height)) * 2.0 - 1.0;
    let tan_half_fov = prometheus_atmosphere.camera_right_tan_half_fov.w;
    let aspect = prometheus_atmosphere.camera_up_aspect.w;
    let ray = normalize(
        prometheus_atmosphere.camera_right_tan_half_fov.xyz * (ndc_x * tan_half_fov * aspect)
        + prometheus_atmosphere.camera_up_aspect.xyz * (ndc_y * tan_half_fov)
        + prometheus_atmosphere.camera_forward_ground.xyz
    );
    let sun_dir = normalize(prometheus_atmosphere.sun_direction_intensity.xyz);
    let sun_intensity = aether_eval_clamp_radiometric_scale(
        prometheus_atmosphere.sun_direction_intensity.w,
    );
    let atmosphere_exposure = aether_eval_clamp_radiometric_scale(
        prometheus_atmosphere.camera_origin_exposure.w,
    );
    let atmosphere_height = max(
        prometheus_atmosphere.planet_radii_path.y - prometheus_atmosphere.planet_radii_path.x,
        1.0,
    );
    let camera_height = max(prometheus_atmosphere.camera_origin_exposure.y, 0.0);
    let camera_height_unit = clamp(camera_height / atmosphere_height, 0.0, 1.0);

    // Use PROMETHEUS' explicit frame-0 visibility AOV for classification.
    // A self-inequality NaN probe is not portable under Metal fast-math.
    if (visibility < 0.5) {
        let miss_scattering = prometheus_load_endpoint_scattering(
            camera_height_unit,
            sun_dir.y,
            ray.y,
            dot(ray, sun_dir),
        ) * sun_intensity;
        let ldr = tonemap_apply_operator(
            aether_eval_clamp_hdr_radiance(miss_scattering) * atmosphere_exposure,
            TONEMAP_OPERATOR_REINHARD,
            1.0,
        );
        textureStore(
            prometheus_output,
            vec2<i32>(gid.xy),
            vec4<f32>(ldr, 1.0),
        );
        return;
    }

    let endpoint_height = aether_eval_spherical_altitude(
        camera_height,
        ray.y,
        depth,
        prometheus_atmosphere.planet_radii_path.x,
    );
    let view_sun_nu = dot(ray, sun_dir);
    let endpoint_mus = aether_eval_spherical_endpoint_mus(
        camera_height,
        ray.y,
        sun_dir.y,
        view_sun_nu,
        depth,
        prometheus_atmosphere.planet_radii_path.x,
    );
    let analytic_segment_t = aether_eval_segment_transmittance(
        depth,
        camera_height,
        ray.y,
        prometheus_atmosphere.planet_radii_path.x,
        1.0,
        prometheus_atmosphere.mie_turbidity_scales.y,
        prometheus_atmosphere.planet_radii_path.w,
    );
    let boundary_t = prometheus_load_boundary_transmittance(camera_height_unit, ray.y);
    let camera_scattering = prometheus_load_endpoint_scattering(
        camera_height_unit,
        sun_dir.y,
        ray.y,
        view_sun_nu,
    ) * sun_intensity;
    let endpoint_height_unit = clamp(endpoint_height / atmosphere_height, 0.0, 1.0);
    let endpoint_scattering = prometheus_load_endpoint_scattering(
        endpoint_height_unit,
        endpoint_mus.y,
        endpoint_mus.x,
        view_sun_nu,
    ) * sun_intensity;
    let distance_unit = depth / max(prometheus_atmosphere.planet_radii_path.z, 1.0);
    let aerial_mean_transmittance = prometheus_load_aerial_transmittance(
        distance_unit,
        camera_height_unit,
        ray.y,
    );
    // Preserve the genuine per-wavelength color of the segment integral while
    // anchoring its mean extinction to the shipped aerial froxel.
    let analytic_mean_t = dot(analytic_segment_t, vec3<f32>(0.2126, 0.7152, 0.0722));
    let transmittance = max(
        clamp(
            analytic_segment_t * (aerial_mean_transmittance / max(analytic_mean_t, 1.0e-6)),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ),
        boundary_t,
    );
    // Finite-segment identity: subtract the boundary radiance transported from
    // the endpoint. The aerial froxel contributes its transmittance anchor;
    // inscatter comes from the accumulated single+higher-order LUT itself.
    let finite_inscatter = max(
        camera_scattering - transmittance * endpoint_scattering,
        vec3<f32>(0.0),
    );
    let linear_hdr = aether_eval_clamp_hdr_radiance(
        surface_or_environment * transmittance + finite_inscatter,
    );
    let ldr = tonemap_apply_operator(
        linear_hdr * atmosphere_exposure,
        TONEMAP_OPERATOR_REINHARD,
        1.0,
    );
    textureStore(
        prometheus_output,
        vec2<i32>(gid.xy),
        vec4<f32>(ldr, 1.0),
    );
}
