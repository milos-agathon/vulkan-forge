struct ViewshedUniforms {
    dimensions: vec4<u32>,
    observer: vec4<f32>,
    metric: vec4<f32>,
    physics: vec4<f32>,
    geodetic: vec4<f32>,
}

struct ViewshedCell {
    visible: u32,
    curvature_drop_m: f32,
    refraction_gain_m: f32,
    horizon_distance_m: f32,
}

@group(0) @binding(0) var<uniform> uniforms: ViewshedUniforms;
@group(0) @binding(2) var<storage, read> geodesic_positions_m: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> result: array<ViewshedCell>;
@group(0) @binding(4) var<storage, read> shadow_inputs: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> shadow_result: array<atomic<u32>>;
@group(0) @binding(6) var height_texture: texture_2d<f32>;
@group(0) @binding(7) var minmax_texture: texture_2d<f32>;

fn height_at(pixel: vec2<f32>) -> f32 {
    let limit = vec2<f32>(
        f32(uniforms.dimensions.x - 1u),
        f32(uniforms.dimensions.y - 1u),
    );
    let p = clamp(pixel, vec2<f32>(0.0), limit);
    let lo = vec2<u32>(floor(p));
    let hi = min(lo + vec2<u32>(1u), uniforms.dimensions.xy - vec2<u32>(1u));
    let fraction = p - vec2<f32>(lo);
    let h00 = textureLoad(height_texture, vec2<i32>(lo), 0).x;
    let h10 = textureLoad(height_texture, vec2<i32>(i32(hi.x), i32(lo.y)), 0).x;
    let h01 = textureLoad(height_texture, vec2<i32>(i32(lo.x), i32(hi.y)), 0).x;
    let h11 = textureLoad(height_texture, vec2<i32>(hi), 0).x;
    return det_mix(
        det_mix(h00, h10, fraction.x),
        det_mix(h01, h11, fraction.x),
        fraction.y,
    );
}

// HELIOS uses the same packed-node layout, slab descent, and exact bilinear
// leaf polynomial as PROMETHEUS' production `terrain_trace`. The
// input here is a geodesic chord in DEM-pixel coordinates rather than a 3-D
// render ray; both viewshed and solar-shadow entry points call this one trace.
const TERRAIN_INVALID_NODE: u32 = 0xFFFFFFFFu;

struct TerrainChild {
    id: u32,
    entry_t: f32,
}

fn terrain_safe_inv(direction: f32) -> f32 {
    let magnitude = max(abs(direction), 1e-12);
    return select(1.0 / magnitude, -1.0 / magnitude, direction < 0.0);
}

fn terrain_slab_xz(
    origin: vec2<f32>,
    direction: vec2<f32>,
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
) -> vec2<f32> {
    let inverse_x = terrain_safe_inv(direction.x);
    let inverse_z = terrain_safe_inv(direction.y);
    var tx0 = (x0 - origin.x) * inverse_x;
    var tx1 = (x1 - origin.x) * inverse_x;
    if tx0 > tx1 { let temporary = tx0; tx0 = tx1; tx1 = temporary; }
    var tz0 = (z0 - origin.y) * inverse_z;
    var tz1 = (z1 - origin.y) * inverse_z;
    if tz0 > tz1 { let temporary = tz0; tz0 = tz1; tz1 = temporary; }
    return vec2<f32>(max(tx0, tz0), min(tx1, tz1));
}

fn terrain_pack_node(level: u32, x: u32, y: u32) -> u32 {
    return (level << 26u) | (y << 13u) | x;
}

fn terrain_height_limit(distance_m: f32, coefficients: vec3<f32>) -> f32 {
    return det_fma(
        coefficients.z,
        distance_m * distance_m,
        det_fma(coefficients.y, distance_m, coefficients.x),
    );
}

// The raw-height limit is a convex quadratic in camera-relative distance.
// Endpoints give its maximum and the in-span vertex gives its exact minimum.
fn terrain_height_limit_range(
    distance0_m: f32,
    distance1_m: f32,
    coefficients: vec3<f32>,
) -> vec2<f32> {
    let height0 = terrain_height_limit(distance0_m, coefficients);
    let height1 = terrain_height_limit(distance1_m, coefficients);
    var minimum = min(height0, height1);
    if coefficients.z > 0.0 {
        let vertex = -coefficients.y / (2.0 * coefficients.z);
        let distance_min = min(distance0_m, distance1_m);
        let distance_max = max(distance0_m, distance1_m);
        if vertex >= distance_min && vertex <= distance_max {
            minimum = min(minimum, terrain_height_limit(vertex, coefficients));
        }
    }
    return vec2<f32>(minimum, max(height0, height1));
}

fn terrain_cell_heights(cell_x: u32, cell_y: u32) -> vec4<f32> {
    let h00 = textureLoad(height_texture, vec2<i32>(i32(cell_x), i32(cell_y)), 0).x;
    let h10 = textureLoad(height_texture, vec2<i32>(i32(cell_x + 1u), i32(cell_y)), 0).x;
    let h01 = textureLoad(height_texture, vec2<i32>(i32(cell_x), i32(cell_y + 1u)), 0).x;
    let h11 = textureLoad(height_texture, vec2<i32>(i32(cell_x + 1u), i32(cell_y + 1u)), 0).x;
    return vec4<f32>(h00, h10, h01, h11);
}

fn terrain_leaf_deviation(
    origin: vec2<f32>,
    direction: vec2<f32>,
    heights: vec4<f32>,
    cell_x: u32,
    cell_y: u32,
    segment_t: f32,
    distance0_m: f32,
    distance1_m: f32,
    height_coefficients: vec3<f32>,
) -> f32 {
    let pixel = origin + segment_t * direction;
    let u = clamp(pixel.x - f32(cell_x), 0.0, 1.0);
    let v = clamp(pixel.y - f32(cell_y), 0.0, 1.0);
    let terrain_height = det_mix(
        det_mix(heights.x, heights.y, u),
        det_mix(heights.z, heights.w, u),
        v,
    );
    let distance_m = det_mix(distance0_m, distance1_m, segment_t);
    return terrain_height - terrain_height_limit(distance_m, height_coefficients);
}

fn terrain_leaf_occluded(
    origin: vec2<f32>,
    direction: vec2<f32>,
    cell_x: u32,
    cell_y: u32,
    segment_t0: f32,
    segment_t1: f32,
    distance0_m: f32,
    distance1_m: f32,
    height_coefficients: vec3<f32>,
    tolerance_m: f32,
) -> bool {
    let heights = terrain_cell_heights(cell_x, cell_y);
    let segment_mid = 0.5 * (segment_t0 + segment_t1);
    let deviation = vec3<f32>(
        terrain_leaf_deviation(origin, direction, heights, cell_x, cell_y,
            segment_t0, distance0_m, distance1_m, height_coefficients),
        terrain_leaf_deviation(origin, direction, heights, cell_x, cell_y,
            segment_mid, distance0_m, distance1_m, height_coefficients),
        terrain_leaf_deviation(origin, direction, heights, cell_x, cell_y,
            segment_t1, distance0_m, distance1_m, height_coefficients),
    );

    // Exact quadratic fit on q in [0,1], identical to the production leaf
    // intersection. Here its maximum directly answers the any-hit query.
    let quadratic = 2.0 * deviation.z + 2.0 * deviation.x - 4.0 * deviation.y;
    let linear = deviation.z - deviation.x - quadratic;
    var maximum = max(deviation.x, deviation.z);
    if abs(quadratic) > 1e-12 {
        let vertex = -linear / (2.0 * quadratic);
        if vertex > 0.0 && vertex < 1.0 {
            maximum = max(
                maximum,
                det_fma(quadratic, vertex * vertex, det_fma(linear, vertex, deviation.x)),
            );
        }
    }
    return maximum > tolerance_m;
}

// FXC cannot address dynamically indexed function arrays without forcing the
// containing loop to unroll. Select the next intersecting child in scalar
// state so the same depth-first traversal remains representable on DX12.
fn terrain_select_child(
    origin: vec2<f32>,
    direction: vec2<f32>,
    parent_level: u32,
    parent_x: u32,
    parent_y: u32,
    after_entry_t: f32,
    after_id: u32,
) -> TerrainChild {
    let cell_width = uniforms.dimensions.x - 1u;
    let cell_height = uniforms.dimensions.y - 1u;
    let child_level = parent_level - 1u;
    var best = TerrainChild(TERRAIN_INVALID_NODE, 2.0);
    for (var child_index = 0u; child_index < 4u; child_index += 1u) {
        let next_x = parent_x * 2u + (child_index & 1u);
        let next_y = parent_y * 2u + (child_index >> 1u);
        let cell_x0 = next_x << child_level;
        let cell_y0 = next_y << child_level;
        if cell_x0 >= cell_width || cell_y0 >= cell_height { continue; }
        let cell_x1 = min((next_x + 1u) << child_level, cell_width);
        let cell_y1 = min((next_y + 1u) << child_level, cell_height);
        let span = terrain_slab_xz(
            origin,
            direction,
            f32(cell_x0),
            f32(cell_x1),
            f32(cell_y0),
            f32(cell_y1),
        );
        let entry_t = max(span.x, 0.0);
        let exit_t = min(span.y, 1.0);
        if entry_t > exit_t { continue; }
        let id = terrain_pack_node(child_level, next_x, next_y);
        let follows = after_id == TERRAIN_INVALID_NODE
            || entry_t > after_entry_t
            || (entry_t == after_entry_t && id > after_id);
        if follows && (best.id == TERRAIN_INVALID_NODE
            || entry_t < best.entry_t
            || (entry_t == best.entry_t && id < best.id)) {
            best = TerrainChild(id, entry_t);
        }
    }
    return best;
}

// Continuous any-hit descent through every node overlapped by one geodesic
// chord. Node pruning uses the exact height-limit range over the node slab;
// leaf tests use the exact bilinear polynomial, so blockers between chord
// endpoints cannot be missed by point sampling.
fn terrain_trace_segment(
    origin: vec2<f32>,
    endpoint: vec2<f32>,
    distance0_m: f32,
    distance1_m: f32,
    height_coefficients: vec3<f32>,
    tolerance_m: f32,
) -> bool {
    let direction = endpoint - origin;
    let cell_width = uniforms.dimensions.x - 1u;
    let cell_height = uniforms.dimensions.y - 1u;
    let root_level = textureNumLevels(minmax_texture) - 1u;
    var node = terrain_pack_node(root_level, 0u, 0u);

    loop {
        let level = node >> 26u;
        let node_y = (node >> 13u) & 0x1FFFu;
        let node_x = node & 0x1FFFu;
        let cell_x0 = node_x << level;
        let cell_y0 = node_y << level;
        var descend = false;
        if cell_x0 < cell_width && cell_y0 < cell_height {
            let cell_x1 = min((node_x + 1u) << level, cell_width);
            let cell_y1 = min((node_y + 1u) << level, cell_height);
            let span = terrain_slab_xz(
                origin,
                direction,
                f32(cell_x0),
                f32(cell_x1),
                f32(cell_y0),
                f32(cell_y1),
            );
            let segment_t0 = max(span.x, 0.0);
            let segment_t1 = min(span.y, 1.0);
            if segment_t0 <= segment_t1 {
                let node_distance0_m = det_mix(distance0_m, distance1_m, segment_t0);
                let node_distance1_m = det_mix(distance0_m, distance1_m, segment_t1);
                let height_range = terrain_height_limit_range(
                    node_distance0_m,
                    node_distance1_m,
                    height_coefficients,
                );
                let minmax = textureLoad(
                    minmax_texture,
                    vec2<i32>(i32(node_x), i32(node_y)),
                    i32(level),
                ).xy;
                if height_range.x + tolerance_m < minmax.y {
                    if level == 0u {
                        if terrain_leaf_occluded(
                            origin,
                            direction,
                            cell_x0,
                            cell_y0,
                            segment_t0,
                            segment_t1,
                            distance0_m,
                            distance1_m,
                            height_coefficients,
                            tolerance_m,
                        ) {
                            return true;
                        }
                    } else {
                        let child = terrain_select_child(
                            origin,
                            direction,
                            level,
                            node_x,
                            node_y,
                            0.0,
                            TERRAIN_INVALID_NODE,
                        );
                        if child.id != TERRAIN_INVALID_NODE {
                            node = child.id;
                            descend = true;
                        }
                    }
                }
            }
        }
        if descend { continue; }

        // The current subtree is complete. Ascend until an intersecting sibling
        // follows it in deterministic entry order, or until the root is done.
        var completed_level = level;
        var completed_x = node_x;
        var completed_y = node_y;
        var advanced = false;
        loop {
            if completed_level >= root_level { break; }
            let parent_level = completed_level + 1u;
            let parent_x = completed_x >> 1u;
            let parent_y = completed_y >> 1u;
            let completed_x0 = completed_x << completed_level;
            let completed_y0 = completed_y << completed_level;
            let completed_x1 = min((completed_x + 1u) << completed_level, cell_width);
            let completed_y1 = min((completed_y + 1u) << completed_level, cell_height);
            let completed_span = terrain_slab_xz(
                origin,
                direction,
                f32(completed_x0),
                f32(completed_x1),
                f32(completed_y0),
                f32(completed_y1),
            );
            let sibling = terrain_select_child(
                origin,
                direction,
                parent_level,
                parent_x,
                parent_y,
                max(completed_span.x, 0.0),
                terrain_pack_node(completed_level, completed_x, completed_y),
            );
            if sibling.id != TERRAIN_INVALID_NODE {
                node = sibling.id;
                advanced = true;
                break;
            }
            completed_level = parent_level;
            completed_x = parent_x;
            completed_y = parent_y;
        }
        if !advanced { break; }
    }
    return false;
}

fn inverse_radius(direction_m: vec2<f32>) -> f32 {
    let distance_squared = dot(direction_m, direction_m);
    if distance_squared == 0.0 {
        return 0.0;
    }
    let east_fraction_squared = direction_m.x * direction_m.x / distance_squared;
    let north_fraction_squared = direction_m.y * direction_m.y / distance_squared;
    return north_fraction_squared * uniforms.physics.x
        + east_fraction_squared * uniforms.physics.y;
}

fn latlon_to_pixel(latitude: f32, longitude: f32) -> vec2<f32> {
    var longitude_deg = degrees(longitude);
    if longitude_deg < uniforms.geodetic.z {
        longitude_deg += 360.0;
    }
    if longitude_deg > uniforms.geodetic.z + 180.0 {
        longitude_deg -= 360.0;
    }
    return vec2<f32>(
        det_div(longitude_deg - uniforms.geodetic.z, uniforms.metric.y) - 0.5,
        det_div(uniforms.geodetic.w - degrees(latitude), uniforms.metric.z) - 0.5,
    );
}

fn geodesic_sample_pixel(
    origin_latitude: f32,
    origin_longitude: f32,
    azimuth: f32,
    distance_m: f32,
) -> vec2<f32> {
    // Flat disables the vertical drop only. EPSG:4326 raster navigation stays
    // geodesic so the ablation does not also change horizontal distances.
    if uniforms.metric.w > 0.0 {
        let angular_distance = det_div(distance_m, uniforms.metric.w);
        let sin_latitude = det_fma(
            det_sin(origin_latitude),
            det_cos(angular_distance),
            det_sin(angular_distance) * det_cos(origin_latitude) * det_cos(azimuth),
        );
        let latitude = 1.5707963267948966
            - det_acos(clamp(sin_latitude, -1.0, 1.0));
        let longitude = origin_longitude + det_atan2(
            det_sin(azimuth) * det_sin(angular_distance) * det_cos(origin_latitude),
            det_cos(angular_distance) - det_sin(origin_latitude) * det_sin(latitude),
        );
        return latlon_to_pixel(latitude, longitude);
    }
    let flattening = 1.0 / 298.257223563;
    let semi_major = 6378137.0;
    let semi_minor = semi_major * (1.0 - flattening);
    let reduced_latitude = det_atan2(
        (1.0 - flattening) * det_sin(origin_latitude),
        det_cos(origin_latitude),
    );
    let sin_u1 = det_sin(reduced_latitude);
    let cos_u1 = det_cos(reduced_latitude);
    let sin_azimuth = det_sin(azimuth);
    let cos_azimuth = det_cos(azimuth);
    let sigma1 = det_atan2(sin_u1, cos_u1 * cos_azimuth);
    let sin_alpha = cos_u1 * sin_azimuth;
    let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
    let u_sq = cos_sq_alpha
        * (semi_major * semi_major - semi_minor * semi_minor)
        / (semi_minor * semi_minor);
    let coefficient_a = 1.0 + u_sq / 16384.0
        * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let coefficient_b = u_sq / 1024.0
        * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
    var sigma = det_div(distance_m, semi_minor * coefficient_a);
    for (var iteration = 0u; iteration < 4u; iteration += 1u) {
        let two_sigma_m = 2.0 * sigma1 + sigma;
        let sin_sigma = det_sin(sigma);
        let cos_sigma = det_cos(sigma);
        let cos_two_sigma_m = det_cos(two_sigma_m);
        let delta_sigma = coefficient_b * sin_sigma
            * (cos_two_sigma_m + coefficient_b / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)
                    - coefficient_b / 6.0 * cos_two_sigma_m
                    * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                    * (-3.0 + 4.0 * cos_two_sigma_m * cos_two_sigma_m)));
        sigma = det_div(distance_m, semi_minor * coefficient_a) + delta_sigma;
    }
    let sin_sigma = det_sin(sigma);
    let cos_sigma = det_cos(sigma);
    let two_sigma_m = 2.0 * sigma1 + sigma;
    let temporary = sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_azimuth;
    let latitude = det_atan2(
        sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_azimuth,
        (1.0 - flattening)
            * det_sqrt(sin_alpha * sin_alpha + temporary * temporary),
    );
    let lambda = det_atan2(
        sin_sigma * sin_azimuth,
        cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_azimuth,
    );
    let coefficient_c = flattening / 16.0 * cos_sq_alpha
        * (4.0 + flattening * (4.0 - 3.0 * cos_sq_alpha));
    let cos_two_sigma_m = det_cos(two_sigma_m);
    let longitude_delta = lambda - (1.0 - coefficient_c) * flattening * sin_alpha
        * (sigma + coefficient_c * sin_sigma
            * (cos_two_sigma_m + coefficient_c * cos_sigma
                * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)));
    return latlon_to_pixel(latitude, origin_longitude + longitude_delta);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= uniforms.dimensions.x || id.y >= uniforms.dimensions.y {
        return;
    }
    let index = id.y * uniforms.dimensions.x + id.x;
    let target_pixel = vec2<f32>(id.xy);
    let delta_pixel = target_pixel - uniforms.observer.xy;
    let direction_m = geodesic_positions_m[index];
    let distance_m = length(direction_m);
    let azimuth = atan2(direction_m.x, direction_m.y);
    let inv_radius = inverse_radius(direction_m);
    let vacuum_drop = 0.5 * inv_radius * distance_m * distance_m;
    let effective_drop = vacuum_drop * uniforms.physics.z;
    let refraction_gain = vacuum_drop - effective_drop;
    let observer_elevation = height_at(uniforms.observer.xy) + uniforms.observer.z;
    let target_absolute_elevation = textureLoad(height_texture, vec2<i32>(id.xy), 0).x
        + uniforms.observer.w;
    var horizon_distance = uniforms.metric.x;
    if inv_radius > 0.0 {
        let effective_inv_radius = inv_radius * uniforms.physics.z;
        horizon_distance = sqrt(2.0 * max(observer_elevation, 0.0) / effective_inv_radius)
            + sqrt(2.0 * max(target_absolute_elevation, 0.0) / effective_inv_radius);
    }

    if distance_m == 0.0 {
        result[index] = ViewshedCell(1u, 0.0, 0.0, horizon_distance);
        return;
    }
    if distance_m > uniforms.metric.x {
        result[index] = ViewshedCell(0u, vacuum_drop, refraction_gain, horizon_distance);
        return;
    }

    let target_elevation = target_absolute_elevation - effective_drop;
    let height_coefficients = vec3<f32>(
        observer_elevation,
        det_div(target_elevation - observer_elevation, distance_m),
        0.5 * inv_radius * uniforms.physics.z,
    );
    // Geodesics are curved in the geographic raster. Consecutive chords stay
    // within half a cell, while terrain_trace_segment continuously traverses
    // all bilinear cells crossed by each chord.
    var visible = 1u;
    var segment_start_distance_m = 0.0;
    var segment_start_pixel = uniforms.observer.xy;
    loop {
        let segment_latitude = radians(det_fma(
            -(segment_start_pixel.y + 0.5),
            uniforms.metric.z,
            uniforms.geodetic.w,
        ));
        let segment_length_m = shadow_step_m(segment_latitude, azimuth);
        let segment_end_distance_m = min(
            segment_start_distance_m + segment_length_m,
            distance_m,
        );
        let segment_end_pixel = geodesic_sample_pixel(
            uniforms.geodetic.x,
            uniforms.geodetic.y,
            azimuth,
            segment_end_distance_m,
        );
        let outer_min = vec2<f32>(-0.5);
        let outer_max = vec2<f32>(uniforms.dimensions.xy) - vec2<f32>(0.5);
        if any(segment_end_pixel < outer_min) || any(segment_end_pixel > outer_max) {
            visible = 2u;
            break;
        }
        if terrain_trace_segment(
            segment_start_pixel,
            segment_end_pixel,
            segment_start_distance_m,
            segment_end_distance_m,
            height_coefficients,
            0.001,
        ) {
            visible = 0u;
            break;
        }
        if segment_end_distance_m >= distance_m { break; }
        segment_start_distance_m = segment_end_distance_m;
        segment_start_pixel = segment_end_pixel;
    }
    result[index] = ViewshedCell(
        visible,
        vacuum_drop,
        refraction_gain,
        horizon_distance,
    );
}

fn local_inverse_radius(latitude: f32, azimuth: f32) -> f32 {
    if uniforms.physics.w == 0.0 {
        return 0.0;
    }
    if uniforms.metric.w > 0.0 {
        return 1.0 / uniforms.metric.w;
    }
    let semi_major = 6378137.0;
    let eccentricity_squared = 0.0066943799901413165;
    let sin_latitude = det_sin(latitude);
    let w = det_sqrt(1.0 - eccentricity_squared * sin_latitude * sin_latitude);
    let meridional = det_div(
        semi_major * (1.0 - eccentricity_squared),
        w * w * w,
    );
    let prime_vertical = det_div(semi_major, w);
    let sin_azimuth = det_sin(azimuth);
    let cos_azimuth = det_cos(azimuth);
    return det_div(cos_azimuth * cos_azimuth, meridional)
        + det_div(sin_azimuth * sin_azimuth, prime_vertical);
}

fn shadow_step_m(latitude: f32, azimuth: f32) -> f32 {
    let sin_latitude = det_sin(latitude);
    let flattening_term = 1.0 - 0.0066943799901413165
        * sin_latitude * sin_latitude;
    let root = det_sqrt(flattening_term);
    let meridional = det_div(
        6378137.0 * (1.0 - 0.0066943799901413165),
        flattening_term * root,
    );
    let prime_vertical = det_div(6378137.0, root);
    let horizontal_meridional = select(meridional, uniforms.metric.w, uniforms.metric.w > 0.0);
    let horizontal_prime_vertical = select(
        prime_vertical,
        uniforms.metric.w,
        uniforms.metric.w > 0.0,
    );
    let north_cell_m = horizontal_meridional * radians(uniforms.metric.z);
    let east_cell_m = horizontal_prime_vertical * det_cos(latitude)
        * radians(uniforms.metric.y);
    let east_crossing_m = det_div(east_cell_m, max(abs(det_sin(azimuth)), 1e-6));
    let north_crossing_m = det_div(north_cell_m, max(abs(det_cos(azimuth)), 1e-6));
    return max(0.1, 0.5 * min(north_crossing_m, east_crossing_m));
}

@compute @workgroup_size(8, 8, 1)
fn shadow_mask_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= uniforms.dimensions.x || id.y >= uniforms.dimensions.y {
        return;
    }
    let index = id.y * uniforms.dimensions.x + id.x;
    let geodetic_and_sun = shadow_inputs[index];
    if geodetic_and_sun.w <= 0.0 {
        return;
    }
    let origin_geodetic = geodetic_and_sun.xy;
    let origin_height = textureLoad(height_texture, vec2<i32>(id.xy), 0).x;
    let azimuth = geodetic_and_sun.z;
    let ray_slope = det_div(
        det_sin(geodetic_and_sun.w),
        det_cos(geodetic_and_sun.w),
    );
    let inv_radius = local_inverse_radius(origin_geodetic.x, azimuth);
    let effective_inv_radius = inv_radius * uniforms.physics.z;

    var visible = 1u;
    let height_coefficients = vec3<f32>(
        origin_height,
        ray_slope,
        0.5 * effective_inv_radius,
    );
    var segment_start_distance_m = 0.0;
    var segment_start_pixel = vec2<f32>(id.xy);
    var segment_latitude = origin_geodetic.x;
    loop {
        let segment_end_distance_m = min(
            segment_start_distance_m + shadow_step_m(segment_latitude, azimuth),
            uniforms.metric.x,
        );
        let segment_end_pixel = geodesic_sample_pixel(
            origin_geodetic.x,
            origin_geodetic.y,
            azimuth,
            segment_end_distance_m,
        );
        let outer_min = vec2<f32>(-0.5);
        let outer_max = vec2<f32>(uniforms.dimensions.xy) - vec2<f32>(0.5);
        // A one-centimetre tie band is still well inside HELIOS' sub-metre
        // contract and absorbs backend-ULP noise exactly on a bilinear ridge.
        if terrain_trace_segment(
            segment_start_pixel,
            segment_end_pixel,
            segment_start_distance_m,
            segment_end_distance_m,
            height_coefficients,
            0.01,
        ) {
            visible = 0u;
            break;
        }
        if any(segment_end_pixel < outer_min) || any(segment_end_pixel > outer_max) {
            // The segment tracer clips its slab to the DEM, so every in-bounds
            // cell before this footprint exit was tested continuously.
            break;
        }
        if segment_end_distance_m >= uniforms.metric.x { break; }
        segment_latitude = radians(det_fma(
            -(segment_end_pixel.y + 0.5),
            uniforms.metric.z,
            uniforms.geodetic.w,
        ));
        segment_start_distance_m = det_barrier(segment_end_distance_m);
        segment_start_pixel = segment_end_pixel;
    }
    if visible != 0u {
        atomicOr(&shadow_result[index / 32u], 1u << (index % 32u));
    }
}
