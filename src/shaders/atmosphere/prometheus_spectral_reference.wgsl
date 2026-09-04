// AETHER acceptance-only stochastic spectral reference owned by PROMETHEUS.
//
// This source is concatenated into the hybrid path-tracing module after
// `hybrid_terrain_traversal.wgsl`.  It deliberately reads no AETHER LUT and no
// environment texture: escaped paths see black.  Camera rays, terrain hits,
// and terrain/sun visibility use the exact PROMETHEUS functions and bindings.

const AETHER_REF_WAVELENGTH_COUNT: u32 = 11u;
const AETHER_REF_MAX_DEPTH: u32 = 6u;
const AETHER_REF_RR_START_DEPTH: u32 = 3u;
const AETHER_REF_MAX_NULL_COLLISIONS: u32 = 2048u;
const AETHER_REF_PI: f32 = 3.14159265358979323846;
const AETHER_REF_BOTTOM_RADIUS_M: f32 = 6360000.0;
const AETHER_REF_TOP_RADIUS_M: f32 = 6460000.0;
const AETHER_REF_RAYLEIGH_SCALE_M: f32 = 8000.0;
const AETHER_REF_MIE_SCALE_M: f32 = 1200.0;
const AETHER_REF_MIE_ALBEDO: f32 = 0.9;
// f32 spacing at the 6,360 km planet radius is 0.5 m. Planet-ground rays need
// a radial offset larger than that ULP; the local PROMETHEUS terrain remains
// on its established centimetre-scale offset.
const AETHER_REF_PLANET_RAY_OFFSET_M: f32 = 2.0;
const AETHER_REF_TERRAIN_RAY_OFFSET_M: f32 = 1e-2;

struct AetherRefBoundary {
    t: f32,
    kind: u32, // 0 = black top-of-atmosphere, 1 = exact terrain, 2 = planet ground
    terrain_hit: HybridHitResult,
}

struct AetherRefPhaseSample {
    direction: vec3<f32>,
    weight: f32,
}

fn aether_ref_wavelength_nm(index: u32) -> f32 {
    switch index {
        case 0u: { return 380.0; }
        case 1u: { return 420.0; }
        case 2u: { return 460.0; }
        case 3u: { return 500.0; }
        case 4u: { return 540.0; }
        case 5u: { return 580.0; }
        case 6u: { return 620.0; }
        case 7u: { return 660.0; }
        case 8u: { return 700.0; }
        case 9u: { return 740.0; }
        default: { return 780.0; }
    }
}

fn aether_ref_cie_xyz(index: u32) -> vec3<f32> {
    switch index {
        case 0u: { return vec3<f32>(0.001368, 0.000039, 0.006450); }
        case 1u: { return vec3<f32>(0.134380, 0.004000, 0.645600); }
        case 2u: { return vec3<f32>(0.290800, 0.060000, 1.669200); }
        case 3u: { return vec3<f32>(0.004900, 0.323000, 0.272000); }
        case 4u: { return vec3<f32>(0.290400, 0.954000, 0.020300); }
        case 5u: { return vec3<f32>(0.916300, 0.870000, 0.001650); }
        case 6u: { return vec3<f32>(0.854450, 0.381000, 0.000190); }
        case 7u: { return vec3<f32>(0.164900, 0.061000, 0.000000); }
        case 8u: { return vec3<f32>(0.011359, 0.004102, 0.000000); }
        case 9u: { return vec3<f32>(0.000690, 0.000249, 0.000000); }
        default: { return vec3<f32>(0.000042, 0.000015, 0.000000); }
    }
}

fn aether_ref_turbidity() -> f32 {
    return bitcast<f32>(terrain.extra.z);
}

fn aether_ref_mie_g() -> f32 {
    return bitcast<f32>(terrain.extra.w);
}

fn aether_ref_ozone_scale() -> f32 {
    return terrain.h_params.w;
}

fn aether_ref_ground_albedo() -> f32 {
    return terrain.albedo_pad.a;
}

fn aether_ref_rayleigh_beta(wavelength_nm: f32) -> f32 {
    return 5.10e-31 * 2.546899e25 * pow(550.0 / wavelength_nm, 4.0);
}

fn aether_ref_mie_extinction(wavelength_nm: f32) -> f32 {
    return 1.0e-5 * aether_ref_turbidity() * (550.0 / wavelength_nm);
}

fn aether_ref_ozone_absorption(wavelength_nm: f32) -> f32 {
    let delta = (wavelength_nm - 600.0) / 85.0;
    return 1.2e-6 * exp(-0.5 * delta * delta);
}

fn aether_ref_altitude(position: vec3<f32>) -> f32 {
    let planet_center = vec3<f32>(0.0, -AETHER_REF_BOTTOM_RADIUS_M, 0.0);
    return max(length(position - planet_center) - AETHER_REF_BOTTOM_RADIUS_M, 0.0);
}

fn aether_ref_density(position: vec3<f32>) -> vec3<f32> {
    let altitude = aether_ref_altitude(position);
    let rayleigh = exp(-altitude / AETHER_REF_RAYLEIGH_SCALE_M);
    let mie = exp(-altitude / AETHER_REF_MIE_SCALE_M);
    let ozone = max(1.0 - abs((altitude - 25000.0) / 15000.0), 0.0)
        * aether_ref_ozone_scale();
    return vec3<f32>(rayleigh, mie, ozone);
}

fn aether_ref_sphere_roots(ray: Ray, radius: f32) -> vec2<f32> {
    let center = vec3<f32>(0.0, -AETHER_REF_BOTTOM_RADIUS_M, 0.0);
    let oc = ray.origin - center;
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - c;
    if (discriminant < 0.0) {
        return vec2<f32>(1e30);
    }
    let root = sqrt(discriminant);
    return vec2<f32>(-b - root, -b + root);
}

fn aether_ref_positive_root(roots: vec2<f32>) -> f32 {
    if (roots.x > 1e-3) { return roots.x; }
    if (roots.y > 1e-3) { return roots.y; }
    return 1e30;
}

fn aether_ref_boundary(ray: Ray) -> AetherRefBoundary {
    var out: AetherRefBoundary;
    let top_roots = aether_ref_sphere_roots(ray, AETHER_REF_TOP_RADIUS_M);
    // All supported cameras start inside the top sphere, so the far root is
    // the black-environment escape distance.
    out.t = select(top_roots.x, top_roots.y, top_roots.y > 1e-3);
    out.kind = 0u;
    out.terrain_hit = intersect_hybrid(ray);

    let ground_roots = aether_ref_sphere_roots(ray, AETHER_REF_BOTTOM_RADIUS_M);
    let ground_t = aether_ref_positive_root(ground_roots);
    if (ground_t < out.t) {
        out.t = ground_t;
        out.kind = 2u;
    }
    if (out.terrain_hit.hit != 0u && out.terrain_hit.t < out.t) {
        out.t = out.terrain_hit.t;
        out.kind = 1u;
    }
    return out;
}

fn aether_ref_extinction(wavelength_nm: f32, density: vec3<f32>) -> f32 {
    return aether_ref_rayleigh_beta(wavelength_nm) * density.x
        + aether_ref_mie_extinction(wavelength_nm) * density.y
        + aether_ref_ozone_absorption(wavelength_nm) * density.z;
}

fn aether_ref_transmittance_to_sun(
    position: vec3<f32>,
    sun_direction: vec3<f32>,
    wavelength_nm: f32,
) -> f32 {
    let shadow_ray = Ray(position + sun_direction * 1e-2, 1e-3, sun_direction, 1e30);
    let top_roots = aether_ref_sphere_roots(shadow_ray, AETHER_REF_TOP_RADIUS_M);
    let top_t = select(top_roots.x, top_roots.y, top_roots.y > 1e-3);
    let ground_t = aether_ref_positive_root(
        aether_ref_sphere_roots(shadow_ray, AETHER_REF_BOTTOM_RADIUS_M)
    );
    if (ground_t < top_t || intersect_shadow_ray(shadow_ray, top_t)) {
        return 0.0;
    }
    // Sixty-four midpoint cells keep the thin 1.2 km Mie layer converged
    // against the 512-cell validation oracle without sharing LUT transport.
    let step_count = 64u;
    let step_length = top_t / f32(step_count);
    var optical_depth = 0.0;
    for (var step = 0u; step < step_count; step = step + 1u) {
        let t = (f32(step) + 0.5) * step_length;
        optical_depth = optical_depth
            + aether_ref_extinction(wavelength_nm, aether_ref_density(shadow_ray.origin + sun_direction * t))
                * step_length;
    }
    return exp(-max(optical_depth, 0.0));
}

fn aether_ref_rayleigh_phase(cos_theta: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    return 3.0 * (1.0 + c * c) / (16.0 * AETHER_REF_PI);
}

fn aether_ref_mie_phase(cos_theta: f32, g: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    let gg = clamp(g, -0.999, 0.999);
    let denominator = pow(max(1.0 + gg * gg - 2.0 * gg * c, 1e-6), 1.5);
    return 3.0 * (1.0 - gg * gg) * (1.0 + c * c)
        / (8.0 * AETHER_REF_PI * (2.0 + gg * gg) * denominator);
}

fn aether_ref_basis_direction(axis: vec3<f32>, cosine: f32, phi: f32) -> vec3<f32> {
    let n = normalize(axis);
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    let sine = sqrt(max(1.0 - cosine * cosine, 0.0));
    return normalize(
        cosine * n + sine * cos(phi) * tangent + sine * sin(phi) * bitangent
    );
}

fn aether_ref_sample_rayleigh(
    incoming: vec3<f32>,
    state: ptr<function, u32>,
) -> AetherRefPhaseSample {
    var cosine = 0.0;
    var accepted = false;
    for (var attempt = 0u; attempt < 16u; attempt = attempt + 1u) {
        cosine = 2.0 * xorshift32(state) - 1.0;
        if (xorshift32(state) <= 0.5 * (1.0 + cosine * cosine)) {
            accepted = true;
            break;
        }
    }
    // Rejection exhaustion is made unbiased by falling back to an analytic
    // isotropic proposal with the Rayleigh/pdf importance ratio.
    var weight = 1.0;
    if (!accepted) {
        cosine = 2.0 * xorshift32(state) - 1.0;
        weight = aether_ref_rayleigh_phase(cosine) / (0.25 / AETHER_REF_PI);
    }
    var out: AetherRefPhaseSample;
    out.direction = aether_ref_basis_direction(
        incoming, cosine, 2.0 * AETHER_REF_PI * xorshift32(state)
    );
    out.weight = weight;
    return out;
}

fn aether_ref_sample_mie(
    incoming: vec3<f32>,
    state: ptr<function, u32>,
) -> AetherRefPhaseSample {
    let g = clamp(aether_ref_mie_g(), -0.999, 0.999);
    let u = xorshift32(state);
    var cosine = 2.0 * u - 1.0;
    if (abs(g) > 1e-3) {
        let ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        cosine = clamp((1.0 + g * g - ratio * ratio) / (2.0 * g), -1.0, 1.0);
    }
    let hg_pdf = (1.0 - g * g)
        / (4.0 * AETHER_REF_PI * pow(max(1.0 + g * g - 2.0 * g * cosine, 1e-6), 1.5));
    var out: AetherRefPhaseSample;
    out.direction = aether_ref_basis_direction(
        incoming, cosine, 2.0 * AETHER_REF_PI * xorshift32(state)
    );
    out.weight = aether_ref_mie_phase(cosine, g) / max(hg_pdf, 1e-12);
    return out;
}

fn aether_ref_sample_cosine(normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let u1 = xorshift32(state);
    let u2 = xorshift32(state);
    let cosine = sqrt(max(1.0 - u1, 0.0));
    return aether_ref_basis_direction(normal, cosine, 2.0 * AETHER_REF_PI * u2);
}

fn aether_ref_surface_ray_origin(
    surface_position: vec3<f32>,
    normal: vec3<f32>,
    boundary_kind: u32,
) -> vec3<f32> {
    let terrain_origin = surface_position + normal * AETHER_REF_TERRAIN_RAY_OFFSET_M;
    let planet_center = vec3<f32>(0.0, -AETHER_REF_BOTTOM_RADIUS_M, 0.0);
    let planet_origin = planet_center
        + normal * (AETHER_REF_BOTTOM_RADIUS_M + AETHER_REF_PLANET_RAY_OFFSET_M);
    return select(terrain_origin, planet_origin, boundary_kind == 2u);
}

fn aether_ref_rr(
    throughput: ptr<function, f32>,
    depth: u32,
    state: ptr<function, u32>,
) -> bool {
    if (depth < AETHER_REF_RR_START_DEPTH) { return true; }
    let survival = clamp(*throughput, 0.1, 0.95);
    if (xorshift32(state) > survival) { return false; }
    *throughput = *throughput / survival;
    return true;
}

fn aether_ref_trace_wavelength(
    camera_ray: Ray,
    wavelength_nm: f32,
    state: ptr<function, u32>,
) -> f32 {
    var ray = camera_ray;
    var throughput = 1.0;
    var radiance = 0.0;
    let sun_direction = normalize(lighting.light_dir);
    let sun_radiance = dot(lighting.light_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let beta_rayleigh = aether_ref_rayleigh_beta(wavelength_nm);
    let beta_mie_ext = aether_ref_mie_extinction(wavelength_nm);
    let beta_mie_sca = beta_mie_ext * AETHER_REF_MIE_ALBEDO;
    let beta_ozone = aether_ref_ozone_absorption(wavelength_nm);
    let majorant = beta_rayleigh + beta_mie_ext + beta_ozone * aether_ref_ozone_scale();

    for (var depth = 0u; depth < AETHER_REF_MAX_DEPTH; depth = depth + 1u) {
        let boundary = aether_ref_boundary(ray);
        if (!(boundary.t > ray.tmin) || !(boundary.t < 1e29)) {
            return radiance; // explicit black environment
        }

        var travelled = 0.0;
        var scatter_kind = 0u; // 0 none, 1 Rayleigh, 2 Mie, 3 absorption
        var scatter_position = vec3<f32>(0.0);
        var null_count = 0u;
        loop {
            if (null_count >= AETHER_REF_MAX_NULL_COLLISIONS) {
                // Loud numerical failure: the host rejects non-finite output.
                return bitcast<f32>(0x7fc00000u);
            }
            null_count = null_count + 1u;
            let free_flight = -log(max(1.0 - xorshift32(state), 1e-7)) / max(majorant, 1e-12);
            if (travelled + free_flight >= boundary.t) { break; }
            travelled = travelled + free_flight;
            scatter_position = ray.origin + ray.direction * travelled;
            let density = aether_ref_density(scatter_position);
            let sigma_rayleigh = beta_rayleigh * density.x;
            let sigma_mie_sca = beta_mie_sca * density.y;
            let sigma_mie_abs = (beta_mie_ext - beta_mie_sca) * density.y;
            let sigma_ozone = beta_ozone * density.z;
            let sigma_total = sigma_rayleigh + sigma_mie_sca + sigma_mie_abs + sigma_ozone;
            if (xorshift32(state) * majorant >= sigma_total) { continue; }
            let event = xorshift32(state) * sigma_total;
            if (event < sigma_rayleigh) {
                scatter_kind = 1u;
            } else if (event < sigma_rayleigh + sigma_mie_sca) {
                scatter_kind = 2u;
            } else {
                scatter_kind = 3u;
            }
            break;
        }

        if (scatter_kind == 3u) { return radiance; }
        if (scatter_kind != 0u) {
            let cosine_to_sun = dot(ray.direction, sun_direction);
            let phase = select(
                aether_ref_mie_phase(cosine_to_sun, aether_ref_mie_g()),
                aether_ref_rayleigh_phase(cosine_to_sun),
                scatter_kind == 1u,
            );
            let sun_t = aether_ref_transmittance_to_sun(
                scatter_position, sun_direction, wavelength_nm
            );
            radiance = radiance + throughput * sun_radiance * phase * sun_t;

            var phase_sample: AetherRefPhaseSample;
            if (scatter_kind == 1u) {
                phase_sample = aether_ref_sample_rayleigh(ray.direction, state);
            } else {
                phase_sample = aether_ref_sample_mie(ray.direction, state);
            }
            throughput = throughput * phase_sample.weight;
            ray = Ray(scatter_position + phase_sample.direction * 1e-2, 1e-3, phase_sample.direction, 1e30);
            if (!aether_ref_rr(&throughput, depth + 1u, state)) { return radiance; }
            continue;
        }

        // The sampled free flight reached a real boundary.  Top-of-atmosphere
        // is black; terrain and planetary ground receive sun NEE then bounce.
        if (boundary.kind == 0u) { return radiance; }
        let surface_position = ray.origin + ray.direction * boundary.t;
        var normal = normalize(
            surface_position - vec3<f32>(0.0, -AETHER_REF_BOTTOM_RADIUS_M, 0.0)
        );
        if (boundary.kind == 1u) { normal = boundary.terrain_hit.normal; }
        let surface_ray_origin = aether_ref_surface_ray_origin(
            surface_position, normal, boundary.kind
        );
        let ndotl = max(dot(normal, sun_direction), 0.0);
        if (ndotl > 0.0) {
            let sun_t = aether_ref_transmittance_to_sun(
                surface_ray_origin, sun_direction, wavelength_nm
            );
            radiance = radiance + throughput * aether_ref_ground_albedo()
                * sun_radiance * sun_t * ndotl / AETHER_REF_PI;
        }
        throughput = throughput * aether_ref_ground_albedo();
        let bounce_direction = aether_ref_sample_cosine(normal, state);
        ray = Ray(surface_ray_origin, 1e-3, bounce_direction, 1e30);
        if (!aether_ref_rr(&throughput, depth + 1u, state)) { return radiance; }
    }
    return radiance;
}

// One invocation owns one pixel and all of its samples.  Keeping the sample
// loop local avoids floating-point atomics and returns an unbiased XYZ sum,
// CIE-Y Welford M2, seed, and SPP contract to the host. RGB conversion and
// non-negative display-domain clipping happen only after the host averages the
// complete XYZ estimator.
@compute @workgroup_size(8, 8, 1)
fn main_aether_spectral_reference(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) { return; }
    let pixel = gid.y * uniforms.width + gid.x;
    let enabled = uniforms.aov_flags != 0u;
    if (!enabled) {
        accum_hdr[pixel] = vec4<f32>(0.0);
        terrain_welford[pixel] = vec2<f32>(0.0);
        return;
    }

    var state = uniforms.seed_hi ^ (gid.x * 1664525u) ^ (gid.y * 1013904223u)
        ^ (uniforms.frame_index * 92837111u) ^ uniforms.seed_lo;
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let spp = max(terrain.extra.x, 1u);
    var sum_xyz = vec3<f32>(0.0);
    var mean_y = 0.0;
    var m2_y = 0.0;
    var terrain_primary_hits = 0u;

    for (var sample = 0u; sample < spp; sample = sample + 1u) {
        // Exact PROMETHEUS pixel/frame/sample RNG and primary-ray transform.
        let jitter_x = terrain_tent_offset(xorshift32(&state)) * 0.5;
        let jitter_y = terrain_tent_offset(xorshift32(&state)) * 0.5;
        let ndc_x = ((f32(gid.x) + 0.5 + jitter_x) / f32(uniforms.width)) * 2.0 - 1.0;
        let ndc_y = (1.0 - (f32(gid.y) + 0.5 + jitter_y) / f32(uniforms.height)) * 2.0 - 1.0;
        var direction = normalize(vec3<f32>(ndc_x * half_w, ndc_y * half_h, -1.0));
        direction = normalize(
            direction.x * uniforms.cam_right
            + direction.y * uniforms.cam_up
            + direction.z * (-uniforms.cam_forward)
        );
        let camera_ray = Ray(uniforms.cam_origin, 1e-3, direction, 1e30);
        let primary_hit = intersect_hybrid(camera_ray);
        if (primary_hit.hit != 0u && primary_hit.hit_type == 3u) {
            terrain_primary_hits = terrain_primary_hits + 1u;
        }

        var xyz = vec3<f32>(0.0);
        for (var wavelength_index = 0u;
             wavelength_index < AETHER_REF_WAVELENGTH_COUNT;
             wavelength_index = wavelength_index + 1u) {
            // Decorrelate wavelength paths without changing the canonical
            // camera sample sequence owned by `state`.
            var wavelength_state = state ^ ((wavelength_index + 1u) * 0x9e3779b9u);
            let value = aether_ref_trace_wavelength(
                camera_ray, aether_ref_wavelength_nm(wavelength_index), &wavelength_state
            );
            let trapezoid_weight = select(
                1.0, 0.5,
                wavelength_index == 0u || wavelength_index + 1u == AETHER_REF_WAVELENGTH_COUNT,
            );
            xyz = xyz + value * aether_ref_cie_xyz(wavelength_index) * trapezoid_weight;
            // Advance the canonical stream once per spectral path so a new
            // sample cannot replay a previous wavelength state.
            let advance = xorshift32(&state);
        }
        sum_xyz = sum_xyz + xyz;
        let sample_y = xyz.y;
        let count = f32(sample + 1u);
        let delta = sample_y - mean_y;
        mean_y = mean_y + delta / count;
        m2_y = m2_y + delta * (sample_y - mean_y);
    }

    accum_hdr[pixel] = vec4<f32>(sum_xyz, f32(terrain_primary_hits));
    terrain_welford[pixel] = vec2<f32>(mean_y, m2_y);
}
