// AETHER's single production LUT-evaluation core.
//
// This source owns the spectral basis, CIE conversion, nonlinear scattering
// coordinates, quadrilinear accumulated-scattering lookup, and the complete
// 16-sample camera-to-hit transmittance integral. It declares no bindings:
// sky, terrain, and PROMETHEUS pass their active textures, dimensions, and
// physical uniforms explicitly.

const AETHER_EVAL_WAVELENGTH_COUNT: u32 = 11u;
var<private> AETHER_EVAL_WAVELENGTHS_NM: array<f32, 11> = array<f32, 11>(
    380.0, 420.0, 460.0, 500.0, 540.0, 580.0,
    620.0, 660.0, 700.0, 740.0, 780.0,
);
var<private> AETHER_EVAL_CIE_XYZ: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.001368, 0.000039, 0.006450),
    vec3<f32>(0.134380, 0.004000, 0.645600),
    vec3<f32>(0.290800, 0.060000, 1.669200),
    vec3<f32>(0.004900, 0.323000, 0.272000),
    vec3<f32>(0.290400, 0.954000, 0.020300),
    vec3<f32>(0.916300, 0.870000, 0.001650),
    vec3<f32>(0.854450, 0.381000, 0.000190),
    vec3<f32>(0.164900, 0.061000, 0.000000),
    vec3<f32>(0.011359, 0.004102, 0.000000),
    vec3<f32>(0.000690, 0.000249, 0.000000),
    vec3<f32>(0.000042, 0.000015, 0.000000),
);

fn aether_eval_clamp_radiometric_scale(value: f32) -> f32 {
    // Bound accepted finite API scales before products so intermediates stay
    // finite and the eventual linear-HDR f16 storage contract is reachable.
    return min(max(value, 0.0), 65504.0);
}

fn aether_eval_clamp_hdr_radiance(radiance: vec3<f32>) -> vec3<f32> {
    return min(max(radiance, vec3<f32>(0.0)), vec3<f32>(65504.0));
}

fn aether_eval_xyz_to_rgb(xyz: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(vec3<f32>(3.2404542, -1.5371385, -0.4985314), xyz) / 3.2613921,
        dot(vec3<f32>(-0.9692660, 1.8760108, 0.0415560), xyz) / 2.5069624,
        dot(vec3<f32>(0.0556434, -0.2040259, 1.0572252), xyz) / 2.3679786,
    );
}

fn aether_eval_spectral_xyz(
    wavelength_index: u32,
    rayleigh_column: f32,
    mie_column: f32,
    ozone_column: f32,
    turbidity: f32,
) -> vec3<f32> {
    let lambda_nm = AETHER_EVAL_WAVELENGTHS_NM[wavelength_index];
    let wavelength_ratio = 550.0 / lambda_nm;
    let wavelength_ratio_squared = wavelength_ratio * wavelength_ratio;
    let rayleigh_beta = 1.2989e-5 * wavelength_ratio_squared * wavelength_ratio_squared;
    let mie_beta = 1.0e-5 * turbidity * wavelength_ratio;
    let ozone_wavelength_delta = (lambda_nm - 600.0) / 85.0;
    let ozone_beta = 1.2e-6
        * det_exp(-0.5 * ozone_wavelength_delta * ozone_wavelength_delta);
    let spectral_t = det_exp(-max(
        rayleigh_beta * rayleigh_column
            + mie_beta * mie_column
            + ozone_beta * ozone_column,
        0.0,
    ));
    let endpoint_weight = select(
        1.0,
        0.5,
        wavelength_index == 0u || wavelength_index + 1u == AETHER_EVAL_WAVELENGTH_COUNT,
    );
    return AETHER_EVAL_CIE_XYZ[wavelength_index] * spectral_t * endpoint_weight;
}

fn aether_eval_mu_to_unit(mu: f32) -> f32 {
    let bounded = clamp(mu, -1.0, 1.0);
    let magnitude = sqrt(abs(bounded));
    // Avoid sign(0.0): some Metal toolchains have produced a NaN for the
    // otherwise well-defined horizon coordinate during pipeline execution.
    let signed_root = select(-magnitude, magnitude, bounded >= 0.0);
    return 0.5 * (signed_root + 1.0);
}

fn aether_eval_nu_to_unit(nu: f32) -> f32 {
    return 1.0 - sqrt(max(0.5 * (1.0 - clamp(nu, -1.0, 1.0)), 0.0));
}

fn aether_eval_scattering_height_to_unit(height_unit: f32) -> f32 {
    // The bake stores h = H*u^2 so eight slices resolve the dense lower
    // atmosphere instead of landing 14.3 km apart on a linear altitude axis.
    return sqrt(clamp(height_unit, 0.0, 1.0));
}

fn aether_eval_load_scattering_texel(
    accumulated_scattering: texture_3d<f32>,
    view_index: i32,
    sun_index: i32,
    height_index: i32,
    nu_index: i32,
    nu_count: i32,
) -> vec4<f32> {
    let dimensions = textureDimensions(accumulated_scattering, 0);
    let coordinate = clamp(
        vec3<i32>(view_index, sun_index, height_index * nu_count + nu_index),
        vec3<i32>(0),
        vec3<i32>(dimensions) - vec3<i32>(1),
    );
    return textureLoad(accumulated_scattering, coordinate, 0);
}

fn aether_eval_sample_accumulated_scattering(
    accumulated_scattering: texture_3d<f32>,
    height_unit: f32,
    mu_sun: f32,
    mu_view: f32,
    nu: f32,
    height_count_raw: u32,
    nu_count_raw: u32,
) -> vec3<f32> {
    let dimensions = textureDimensions(accumulated_scattering, 0);
    let height_count = max(i32(height_count_raw), 2);
    let nu_count = max(i32(nu_count_raw), 2);
    let coordinates = vec4<f32>(
        aether_eval_mu_to_unit(mu_view) * f32(dimensions.x - 1u),
        aether_eval_mu_to_unit(mu_sun) * f32(dimensions.y - 1u),
        aether_eval_scattering_height_to_unit(height_unit) * f32(height_count - 1),
        aether_eval_nu_to_unit(nu) * f32(nu_count - 1),
    );
    let lower = vec4<i32>(floor(coordinates));
    let upper = min(
        lower + vec4<i32>(1),
        vec4<i32>(
            i32(dimensions.x) - 1,
            i32(dimensions.y) - 1,
            height_count - 1,
            nu_count - 1,
        ),
    );
    let fraction = fract(coordinates);
    var accumulated = vec4<f32>(0.0);
    for (var height_side = 0; height_side < 2; height_side = height_side + 1) {
        for (var nu_side = 0; nu_side < 2; nu_side = nu_side + 1) {
            for (var sun_side = 0; sun_side < 2; sun_side = sun_side + 1) {
                for (var view_side = 0; view_side < 2; view_side = view_side + 1) {
                    let view_index = select(lower.x, upper.x, view_side == 1);
                    let sun_index = select(lower.y, upper.y, sun_side == 1);
                    let height_index = select(lower.z, upper.z, height_side == 1);
                    let nu_index = select(lower.w, upper.w, nu_side == 1);
                    let weight = select(1.0 - fraction.x, fraction.x, view_side == 1)
                        * select(1.0 - fraction.y, fraction.y, sun_side == 1)
                        * select(1.0 - fraction.z, fraction.z, height_side == 1)
                        * select(1.0 - fraction.w, fraction.w, nu_side == 1);
                    accumulated = accumulated + weight * aether_eval_load_scattering_texel(
                        accumulated_scattering,
                        view_index,
                        sun_index,
                        height_index,
                        nu_index,
                        nu_count,
                    );
                }
            }
        }
    }
    return max(accumulated.rgb, vec3<f32>(0.0));
}

fn aether_eval_spherical_radius_m(
    camera_height_m: f32,
    view_mu: f32,
    distance_m: f32,
    bottom_radius_m: f32,
) -> f32 {
    let radius_m = max(bottom_radius_m, 1.0) + clamp(camera_height_m, 0.0, 100000.0);
    let bounded_distance_m = clamp(distance_m, 0.0, 20000000.0);
    let radial_squared = max(
        radius_m * radius_m + bounded_distance_m * bounded_distance_m
            + 2.0 * radius_m * bounded_distance_m * clamp(view_mu, -1.0, 1.0),
        0.0,
    );
    return sqrt(radial_squared);
}

fn aether_eval_spherical_altitude(
    camera_height_m: f32,
    view_mu: f32,
    distance_m: f32,
    bottom_radius_m: f32,
) -> f32 {
    let endpoint_radius_m = aether_eval_spherical_radius_m(
        camera_height_m, view_mu, distance_m, bottom_radius_m,
    );
    return clamp(endpoint_radius_m - max(bottom_radius_m, 1.0), 0.0, 100000.0);
}

fn aether_eval_spherical_endpoint_mus(
    camera_height_m: f32,
    view_mu: f32,
    sun_mu: f32,
    view_sun_nu: f32,
    distance_m: f32,
    bottom_radius_m: f32,
) -> vec2<f32> {
    let radius_m = max(bottom_radius_m, 1.0) + clamp(camera_height_m, 0.0, 100000.0);
    let bounded_distance_m = clamp(distance_m, 0.0, 20000000.0);
    let endpoint_radius_m = max(
        aether_eval_spherical_radius_m(
            camera_height_m, view_mu, bounded_distance_m, bottom_radius_m,
        ),
        1.0,
    );
    let endpoint_view_mu =
        (radius_m * clamp(view_mu, -1.0, 1.0) + bounded_distance_m) / endpoint_radius_m;
    let endpoint_sun_mu = (
        radius_m * clamp(sun_mu, -1.0, 1.0)
            + bounded_distance_m * clamp(view_sun_nu, -1.0, 1.0)
    ) / endpoint_radius_m;
    return clamp(vec2<f32>(endpoint_view_mu, endpoint_sun_mu), vec2<f32>(-1.0), vec2<f32>(1.0));
}

fn aether_eval_segment_transmittance(
    distance_m: f32,
    camera_height_m: f32,
    view_mu: f32,
    bottom_radius_m: f32,
    density_scale: f32,
    turbidity: f32,
    ozone_du: f32,
) -> vec3<f32> {
    let bounded_distance_m = clamp(distance_m, 0.0, 20000000.0);
    let bounded_camera_height_m = clamp(camera_height_m, 0.0, 100000.0);
    // Sixteen explicit midpoint samples cover the complete camera-to-hit
    // segment along the same spherical geometry as the Bruneton bake. Keeping
    // the samples explicit preserves the standalone shader-proof contract.
    let h00 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.03125, bottom_radius_m,
    );
    let h01 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.09375, bottom_radius_m,
    );
    let h02 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.15625, bottom_radius_m,
    );
    let h03 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.21875, bottom_radius_m,
    );
    let h04 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.28125, bottom_radius_m,
    );
    let h05 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.34375, bottom_radius_m,
    );
    let h06 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.40625, bottom_radius_m,
    );
    let h07 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.46875, bottom_radius_m,
    );
    let h08 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.53125, bottom_radius_m,
    );
    let h09 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.59375, bottom_radius_m,
    );
    let h10 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.65625, bottom_radius_m,
    );
    let h11 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.71875, bottom_radius_m,
    );
    let h12 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.78125, bottom_radius_m,
    );
    let h13 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.84375, bottom_radius_m,
    );
    let h14 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.90625, bottom_radius_m,
    );
    let h15 = aether_eval_spherical_altitude(
        bounded_camera_height_m, view_mu, bounded_distance_m * 0.96875, bottom_radius_m,
    );
    let rayleigh_density_sum = det_exp(-h00 / 8000.0) + det_exp(-h01 / 8000.0)
        + det_exp(-h02 / 8000.0) + det_exp(-h03 / 8000.0)
        + det_exp(-h04 / 8000.0) + det_exp(-h05 / 8000.0)
        + det_exp(-h06 / 8000.0) + det_exp(-h07 / 8000.0)
        + det_exp(-h08 / 8000.0) + det_exp(-h09 / 8000.0)
        + det_exp(-h10 / 8000.0) + det_exp(-h11 / 8000.0)
        + det_exp(-h12 / 8000.0) + det_exp(-h13 / 8000.0)
        + det_exp(-h14 / 8000.0) + det_exp(-h15 / 8000.0);
    let mie_density_sum = det_exp(-h00 / 1200.0) + det_exp(-h01 / 1200.0)
        + det_exp(-h02 / 1200.0) + det_exp(-h03 / 1200.0)
        + det_exp(-h04 / 1200.0) + det_exp(-h05 / 1200.0)
        + det_exp(-h06 / 1200.0) + det_exp(-h07 / 1200.0)
        + det_exp(-h08 / 1200.0) + det_exp(-h09 / 1200.0)
        + det_exp(-h10 / 1200.0) + det_exp(-h11 / 1200.0)
        + det_exp(-h12 / 1200.0) + det_exp(-h13 / 1200.0)
        + det_exp(-h14 / 1200.0) + det_exp(-h15 / 1200.0);
    let ozone_density_sum = max(1.0 - abs((h00 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h01 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h02 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h03 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h04 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h05 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h06 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h07 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h08 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h09 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h10 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h11 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h12 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h13 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h14 - 25000.0) / 15000.0), 0.0)
        + max(1.0 - abs((h15 - 25000.0) / 15000.0), 0.0);

    let path_per_sample = bounded_distance_m * density_scale * 0.0625;
    let rayleigh_column = path_per_sample * rayleigh_density_sum;
    let mie_column = path_per_sample * mie_density_sum;
    let ozone_column = path_per_sample * ozone_density_sum * ozone_du / 300.0;
    let xyz = aether_eval_spectral_xyz(
        0u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        1u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        2u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        3u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        4u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        5u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        6u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        7u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        8u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        9u, rayleigh_column, mie_column, ozone_column, turbidity,
    ) + aether_eval_spectral_xyz(
        10u, rayleigh_column, mie_column, ozone_column, turbidity,
    );
    return clamp(aether_eval_xyz_to_rgb(xyz), vec3<f32>(0.0), vec3<f32>(1.0));
}
