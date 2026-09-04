// Moment-shadow visibility: 1.0 = lit, 0.0 = shadowed.
const MSM_FINITE_LIMIT: f32 = 1e20;
const MSM_MATRIX_EPSILON: f32 = 1e-7;
const MSM_DETERMINANT_EPSILON: f32 = 1e-12;
const EVSM_MINIMUM_VARIANCE: f32 = 0.000375;
const EVSM_MAX_EXPONENT_RGBA16F: f32 = 5.5;
const EVSM_FP16_UNIT_ROUNDOFF: f32 = 0.00048828125;
const EVSM_FP16_MIN_SUBNORMAL_HALF: f32 = 0.0000000298023223876953125;
const EVSM_FINITE_LIMIT: f32 = 65504.0;
const EVSM_VISIBILITY_CONTRAST_POWER: f32 = 52.0;

fn chebyshev_upper_bound_visibility(mean: f32, variance: f32, receiver: f32) -> f32 {
    if (receiver <= mean) {
        return 1.0;
    }
    let delta = receiver - mean;
    return variance / (variance + delta * delta);
}

fn evsm_minimum_variance(
    warped_receiver: vec2<f32>,
    exponents: vec2<f32>
) -> vec2<f32> {
    let depth_scale =
        EVSM_MINIMUM_VARIANCE * exponents * abs(warped_receiver);
    return depth_scale * depth_scale;
}

// Returns conservative mean-upper and variance-upper bounds for independently
// rounded Rgba16Float first and second moments.
fn evsm_fp16_moment_bounds(quantized: vec2<f32>) -> vec2<f32> {
    let error = (
        EVSM_FP16_UNIT_ROUNDOFF * abs(quantized)
        + vec2<f32>(EVSM_FP16_MIN_SUBNORMAL_HALF)
    ) / (1.0 - EVSM_FP16_UNIT_ROUNDOFF);
    let mean_lower = quantized.x - error.x;
    let mean_upper = quantized.x + error.x;
    let square_lower = select(
        min(mean_lower * mean_lower, mean_upper * mean_upper),
        0.0,
        mean_lower <= 0.0 && mean_upper >= 0.0
    );
    let variance_upper =
        max(quantized.y + error.y - square_lower, 0.0);
    return vec2<f32>(mean_upper, variance_upper);
}

fn evsm_reduce_light_bleed(visibility: f32) -> f32 {
    if (!(visibility >= 0.0) || !(visibility <= 1.0)) {
        return 1.0;
    }
    // A smooth contrast curve suppresses high-probability light leaks without
    // introducing the discontinuity of a hard light-bleed cutoff.
    return pow(visibility, EVSM_VISIBILITY_CONTRAST_POWER);
}

fn evsm_visibility_from_moments(
    moments: vec4<f32>,
    positive_receiver: f32,
    negative_receiver: f32,
    variance_floor: vec2<f32>
) -> f32 {
    if (
        !all(abs(moments) <= vec4<f32>(EVSM_FINITE_LIMIT))
        || !(abs(positive_receiver) <= EVSM_FINITE_LIMIT)
        || !(abs(negative_receiver) <= EVSM_FINITE_LIMIT)
        || !all(abs(variance_floor) <= vec2<f32>(EVSM_FINITE_LIMIT))
        || !all(variance_floor >= vec2<f32>(0.0))
    ) {
        return 1.0;
    }
    let positive_bounds = evsm_fp16_moment_bounds(moments.rg);
    let negative_bounds = evsm_fp16_moment_bounds(moments.ba);
    let positive_variance = max(positive_bounds.y, variance_floor.x);
    let negative_variance = max(negative_bounds.y, variance_floor.y);
    let positive_visibility =
        chebyshev_upper_bound_visibility(
            positive_bounds.x,
            positive_variance,
            positive_receiver
        );
    let negative_visibility =
        chebyshev_upper_bound_visibility(
            negative_bounds.x,
            negative_variance,
            negative_receiver
        );
    return evsm_reduce_light_bleed(
        min(positive_visibility, negative_visibility)
    );
}

fn evsm_moment_leak_control(
    positive_moments: vec2<f32>,
    positive_receiver: f32,
    positive_exponent: f32,
    minimum_variance: f32
) -> f32 {
    if (
        !(positive_exponent > 0.0)
        || !(positive_exponent <= EVSM_MAX_EXPONENT_RGBA16F)
        || !all(abs(positive_moments) <= vec2<f32>(EVSM_FINITE_LIMIT))
        || !(abs(positive_receiver) <= EVSM_FINITE_LIMIT)
        || !(minimum_variance >= 0.0)
        || !(minimum_variance <= EVSM_FINITE_LIMIT)
    ) {
        return 1.0;
    }

    let moment_bounds = evsm_fp16_moment_bounds(positive_moments);
    let variance = max(moment_bounds.y, minimum_variance);
    let visibility = chebyshev_upper_bound_visibility(
        moment_bounds.x,
        variance,
        positive_receiver
    );
    return evsm_reduce_light_bleed(visibility);
}

fn msm_visibility_from_moments(
    moments: vec4<f32>,
    receiver_depth: f32,
    moment_bias: f32
) -> f32 {
    if (!all(abs(moments) <= vec4<f32>(MSM_FINITE_LIMIT)) ||
        !(abs(receiver_depth) <= MSM_FINITE_LIMIT) ||
        !(abs(moment_bias) <= MSM_FINITE_LIMIT)) {
        return 1.0;
    }
    let receiver = clamp(receiver_depth, 0.0, 1.0);
    let raw_moments = clamp(moments, vec4<f32>(0.0), vec4<f32>(1.0));
    if (receiver <= raw_moments.x) {
        return 1.0;
    }

    let variance_floor = max(moment_bias, 0.000001);
    let variance = max(
        raw_moments.y - raw_moments.x * raw_moments.x,
        variance_floor
    );
    let fallback = clamp(
        chebyshev_upper_bound_visibility(raw_moments.x, variance, receiver),
        0.0,
        1.0
    );

    // Hamburger 4MSM. A small bias towards a valid moment vector compensates
    // Rgba16Float quantization before the Hankel-system reconstruction.
    let quantization_bias = clamp(max(moment_bias, 0.00003), 0.0, 0.01);
    let b = mix(raw_moments, vec4<f32>(0.5), quantization_bias);
    let d22 = b.y - b.x * b.x;
    let l32_d22 = b.z - b.x * b.y;
    let d33_d22 =
        (b.w - b.y * b.y) * d22 - l32_d22 * l32_d22;
    if (!(d22 > MSM_MATRIX_EPSILON) ||
        !(d33_d22 > MSM_DETERMINANT_EPSILON)) {
        return fallback;
    }

    let l32 = l32_d22 / d22;
    var coefficients = vec3<f32>(1.0, receiver, receiver * receiver);
    coefficients.y -= b.x;
    coefficients.z -= b.y + l32 * coefficients.y;
    coefficients.y /= d22;
    coefficients.z *= d22 / d33_d22;
    coefficients.y -= l32 * coefficients.z;
    coefficients.x -= dot(coefficients.yz, b.xy);
    if (!(abs(coefficients.z) > MSM_MATRIX_EPSILON) ||
        !all(abs(coefficients) <= vec3<f32>(MSM_FINITE_LIMIT))) {
        return fallback;
    }

    let p = coefficients.y / coefficients.z;
    let q = coefficients.x / coefficients.z;
    let discriminant = p * p * 0.25 - q;
    if (!(discriminant >= 0.0) || !(discriminant <= MSM_FINITE_LIMIT)) {
        return fallback;
    }

    let root_radius = sqrt(discriminant);
    let root1 = -p * 0.5 - root_radius;
    let root2 = -p * 0.5 + root_radius;
    var branch = vec4<f32>(0.0);
    if (root2 < receiver) {
        branch = vec4<f32>(root1, receiver, 1.0, 1.0);
    } else if (root1 < receiver) {
        branch = vec4<f32>(receiver, root1, 0.0, 1.0);
    }
    if (branch.w == 0.0) {
        return 1.0;
    }

    let denominator = (root2 - branch.y) * (receiver - root1);
    if (!(abs(denominator) > MSM_MATRIX_EPSILON) ||
        !(abs(denominator) <= MSM_FINITE_LIMIT)) {
        return fallback;
    }
    let quotient =
        (branch.x * root2 - b.x * (branch.x + root2) + b.y) /
        denominator;
    let visibility = 1.0 - clamp(branch.z + branch.w * quotient, 0.0, 1.0);
    if (!(visibility >= 0.0) || !(visibility <= 1.0)) {
        return fallback;
    }
    return visibility;
}
