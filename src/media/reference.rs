//! Unbiased heterogeneous-media reference kernel shared by terrain reference callers.

use super::model::normalize;
use super::{
    delta_track_counted, power_heuristic, ratio_track, russian_roulette, DirectionalSun,
    EnvironmentDistribution, MediaError, Ray, Rgb, SampleIdentity, TrackingContext,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceSurfaceHit {
    pub distance: f32,
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub albedo: Rgb,
}

/// The portion of a geometry ray occupied by the canonical medium. Geometry
/// reach is deliberately separate: a bounded cloud can end before terrain or
/// the environment without turning that boundary into a surface or miss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceMediumInterval {
    pub start: f32,
    pub end: f32,
}

/// Geometry seam implemented by the hybrid terrain reference with its
/// authoritative `terrain_trace` traversal. Media never owns a second terrain
/// marcher; every camera, sun, environment, and phase ray returns through this
/// interface.
pub trait ReferenceScene {
    fn intersect(
        &self,
        ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceSurfaceHit>, MediaError>;
    fn occluded(&self, ray: Ray, maximum_distance: f32) -> Result<bool, MediaError>;
    fn geometry_reach(&self, ray: Ray) -> Result<f32, MediaError>;
    fn medium_interval(
        &self,
        ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceMediumInterval>, MediaError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceTransportConfig {
    pub roulette_start_bounce: u32,
    pub roulette_minimum_probability: f32,
}

impl ReferenceTransportConfig {
    fn validate(self) -> Result<(), MediaError> {
        if !self.roulette_minimum_probability.is_finite()
            || !(0.0..=1.0).contains(&self.roulette_minimum_probability)
        {
            return Err(MediaError::InvalidTransport(
                "roulette minimum must be a finite probability".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceTransportSample {
    pub radiance: Rgb,
    pub spectral_channel: u32,
    pub collision_count: u32,
    pub surface_count: u32,
    pub tracking_step_count: u64,
}

/// Trace one deterministic hero-wavelength path. RGB channels are stratified
/// by sample index (each consecutive triple covers all three) and weighted
/// by three. A deterministic per-triple rotation keeps incomplete terminal
/// triples unbiased without sacrificing exact triple coverage.
pub fn trace_reference_sample<S: ReferenceScene>(
    context: &TrackingContext,
    scene: &S,
    environment: &EnvironmentDistribution,
    sun: DirectionalSun,
    camera_ray: Ray,
    identity: SampleIdentity,
    config: ReferenceTransportConfig,
) -> Result<ReferenceTransportSample, MediaError> {
    config.validate()?;
    const SPECTRAL_CHANNEL_DIMENSION: u64 = 0x4e45_5048_454c_4500;
    let triple_identity = SampleIdentity {
        sample: identity.sample / 3,
        ..identity
    };
    let mut attempt = 0u64;
    let rotation = exact_ternary(|| {
        let bits = triple_identity.random_bits(
            SPECTRAL_CHANNEL_DIMENSION.wrapping_add(attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        );
        attempt = attempt.wrapping_add(1);
        bits
    });
    let channel = ((identity.sample % 3) as usize + rotation) % 3;
    let mut ray = Ray {
        origin: camera_ray.origin,
        direction: normalize(camera_ray.direction).ok_or_else(|| {
            MediaError::InvalidTransport("camera direction must be finite and nonzero".into())
        })?,
    };
    let mut throughput = 1.0f32;
    let mut radiance = 0.0f32;
    let mut collision_count = 0;
    let mut surface_count = 0;
    let mut tracking_step_count = 0;
    let mut previous_phase_pdf = None;

    let mut bounce = 0u32;
    loop {
        let bounce_id = SampleIdentity { bounce, ..identity };
        let geometry_reach = scene.geometry_reach(ray)?;
        validate_distance(geometry_reach)?;
        let surface = scene.intersect(ray, geometry_reach)?;
        if let Some(hit) = surface {
            validate_surface_hit(hit, geometry_reach)?;
        }
        let geometry_distance = surface.map_or(geometry_reach, |hit| hit.distance);
        let interval = validated_medium_interval(scene, ray, geometry_distance)?;
        let (collision, steps) = if let Some(interval) = interval {
            delta_track_counted(
                context,
                shifted_ray(ray, interval.start),
                interval.end - interval.start,
                channel,
                bounce_id,
            )?
        } else {
            (None, 0)
        };
        tracking_step_count += steps;

        if let Some(collision) = collision {
            collision_count += 1;
            let sigma_t = collision.extinction.components()[channel];
            let density = context
                .medium()
                .density()
                .physical_density(collision.position);
            let sigma_s = context.medium().sigma_s().components()[channel] * density;
            if sigma_t == 0.0 {
                return Err(MediaError::InvalidTransport(
                    "delta tracking accepted a zero-extinction collision".into(),
                ));
            }
            throughput *= sigma_s / sigma_t;
            if throughput == 0.0 {
                break;
            }

            let to_sun = sun.direction_to_sun;
            let sun_ray = Ray {
                origin: collision.position,
                direction: to_sun,
            };
            let sun_distance = scene.geometry_reach(sun_ray)?;
            validate_distance(sun_distance)?;
            if !scene.occluded(sun_ray, sun_distance)? {
                let (transmittance, steps) = reference_transmittance(
                    context,
                    scene,
                    sun_ray,
                    sun_distance,
                    SampleIdentity {
                        sample: identity.sample.wrapping_add(0x1000),
                        ..bounce_id
                    },
                )?;
                tracking_step_count += steps;
                radiance += throughput
                    * context
                        .medium()
                        .phase()
                        .evaluate(dot(ray.direction, to_sun))?
                    * sun.radiance.components()[channel]
                    * transmittance.components()[channel];
            }

            let env = environment.sample([
                bounce_id.uniform(10),
                bounce_id.uniform(11),
                bounce_id.uniform(12),
                bounce_id.uniform(13),
            ])?;
            let env_ray = Ray {
                origin: collision.position,
                direction: env.direction,
            };
            let env_distance = scene.geometry_reach(env_ray)?;
            validate_distance(env_distance)?;
            if !scene.occluded(env_ray, env_distance)? {
                let phase_pdf = context
                    .medium()
                    .phase()
                    .evaluate(dot(ray.direction, env.direction))?;
                let (light_weight, _) = power_heuristic(env.pdf_solid_angle, phase_pdf)?;
                let (transmittance, steps) = reference_transmittance(
                    context,
                    scene,
                    env_ray,
                    env_distance,
                    SampleIdentity {
                        sample: identity.sample.wrapping_add(0x2000),
                        ..bounce_id
                    },
                )?;
                tracking_step_count += steps;
                radiance += throughput
                    * phase_pdf
                    * env.radiance.components()[channel]
                    * transmittance.components()[channel]
                    * light_weight
                    / env.pdf_solid_angle;
            }

            let phase = context.medium().phase().sample(
                ray.direction,
                [bounce_id.uniform(20), bounce_id.uniform(21)],
            )?;
            throughput *= phase.value / phase.pdf;
            previous_phase_pdf = Some(phase.pdf);
            ray = Ray {
                origin: collision.position,
                direction: phase.direction,
            };
        } else if let Some(hit) = surface {
            surface_count += 1;
            let albedo = hit.albedo.components()[channel];
            let normal = normalize(hit.normal).expect("validated surface normal");
            let to_sun = sun.direction_to_sun;
            let cosine = dot(normal, to_sun).max(0.0);
            if cosine > 0.0 {
                let sun_ray = Ray {
                    origin: hit.position,
                    direction: to_sun,
                };
                let sun_distance = scene.geometry_reach(sun_ray)?;
                validate_distance(sun_distance)?;
                if !scene.occluded(sun_ray, sun_distance)? {
                    let (transmittance, steps) = reference_transmittance(
                        context,
                        scene,
                        sun_ray,
                        sun_distance,
                        SampleIdentity {
                            sample: identity.sample.wrapping_add(0x3000),
                            ..bounce_id
                        },
                    )?;
                    tracking_step_count += steps;
                    radiance += throughput
                        * albedo
                        * cosine
                        * sun.radiance.components()[channel]
                        * transmittance.components()[channel]
                        / std::f32::consts::PI;
                }
            }

            let direction = cosine_hemisphere(normal, bounce_id.uniform(30), bounce_id.uniform(31));
            throughput *= albedo;
            previous_phase_pdf = None;
            ray = Ray {
                origin: hit.position,
                direction,
            };
        } else {
            let env = environment.radiance(ray.direction).components()[channel];
            let weight = if let Some(phase_pdf) = previous_phase_pdf {
                power_heuristic(phase_pdf, environment.pdf(ray.direction))?.0
            } else {
                1.0
            };
            radiance += throughput * env * weight;
            break;
        }

        if bounce >= config.roulette_start_bounce {
            let rgb = channel_rgb(channel, throughput)?;
            let Some(scale) = russian_roulette(
                rgb,
                bounce_id.uniform(40),
                config.roulette_minimum_probability,
            )?
            else {
                break;
            };
            throughput *= scale;
        }
        if !throughput.is_finite() || throughput < 0.0 || !radiance.is_finite() {
            return Err(MediaError::InvalidTransport(
                "reference path produced non-finite or negative transport".into(),
            ));
        }
        bounce = bounce.checked_add(1).ok_or_else(|| {
            MediaError::InvalidTransport("reference bounce identity overflowed".into())
        })?;
    }

    Ok(ReferenceTransportSample {
        radiance: channel_rgb(channel, radiance * 3.0)?,
        spectral_channel: channel as u32,
        collision_count,
        surface_count,
        tracking_step_count,
    })
}

fn exact_ternary(mut random_bits: impl FnMut() -> u64) -> usize {
    loop {
        let bits = random_bits();
        // 0..u64::MAX contains u64::MAX values, exactly divisible by three.
        // Rejecting the sole remaining value makes modulo reduction unbiased.
        if bits != u64::MAX {
            return (bits % 3) as usize;
        }
    }
}

fn validate_distance(distance: f32) -> Result<(), MediaError> {
    if distance.is_finite() && distance >= 0.0 {
        Ok(())
    } else {
        Err(MediaError::InvalidTransport(
            "scene segment distances must be finite and non-negative".into(),
        ))
    }
}

fn validated_medium_interval<S: ReferenceScene>(
    scene: &S,
    ray: Ray,
    maximum_distance: f32,
) -> Result<Option<ReferenceMediumInterval>, MediaError> {
    let Some(interval) = scene.medium_interval(ray, maximum_distance)? else {
        return Ok(None);
    };
    if !interval.start.is_finite()
        || !interval.end.is_finite()
        || interval.start < 0.0
        || interval.end < interval.start
        || interval.end > maximum_distance
    {
        return Err(MediaError::InvalidTransport(
            "scene returned an invalid medium interval".into(),
        ));
    }
    Ok((interval.end > interval.start).then_some(interval))
}

fn shifted_ray(ray: Ray, distance: f32) -> Ray {
    Ray {
        origin: [
            ray.origin[0] + ray.direction[0] * distance,
            ray.origin[1] + ray.direction[1] * distance,
            ray.origin[2] + ray.direction[2] * distance,
        ],
        direction: ray.direction,
    }
}

fn reference_transmittance<S: ReferenceScene>(
    context: &TrackingContext,
    scene: &S,
    ray: Ray,
    maximum_distance: f32,
    identity: SampleIdentity,
) -> Result<(Rgb, u64), MediaError> {
    let Some(interval) = validated_medium_interval(scene, ray, maximum_distance)? else {
        return Ok((Rgb::ONE, 0));
    };
    ratio_track(
        context,
        shifted_ray(ray, interval.start),
        interval.end - interval.start,
        identity,
    )
}

fn validate_surface_hit(hit: ReferenceSurfaceHit, maximum_distance: f32) -> Result<(), MediaError> {
    if !hit.distance.is_finite()
        || hit.distance < 0.0
        || hit.distance > maximum_distance
        || hit
            .position
            .iter()
            .chain(&hit.normal)
            .any(|value| !value.is_finite())
        || normalize(hit.normal).is_none()
    {
        return Err(MediaError::InvalidTransport(
            "reference scene returned an invalid surface hit".into(),
        ));
    }
    Ok(())
}

fn channel_rgb(channel: usize, value: f32) -> Result<Rgb, MediaError> {
    let mut rgb = [0.0; 3];
    rgb[channel] = value;
    Rgb::new(rgb, "reference radiance")
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cosine_hemisphere(normal: [f32; 3], u0: f32, u1: f32) -> [f32; 3] {
    let sign = if normal[2] < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + normal[2]);
    let b = normal[0] * normal[1] * a;
    let tangent = [
        1.0 + sign * normal[0] * normal[0] * a,
        sign * b,
        -sign * normal[0],
    ];
    let bitangent = [b, sign + normal[1] * normal[1] * a, -normal[1]];
    let radius = u0.sqrt();
    let phi = std::f32::consts::TAU * u1;
    let local = [radius * phi.cos(), radius * phi.sin(), (1.0 - u0).sqrt()];
    [
        local[0] * tangent[0] + local[1] * bitangent[0] + local[2] * normal[0],
        local[0] * tangent[1] + local[1] * bitangent[1] + local[2] * normal[1],
        local[0] * tangent[2] + local[1] * bitangent[2] + local[2] * normal[2],
    ]
}

#[cfg(test)]
mod exact_ternary_tests {
    use super::exact_ternary;

    #[test]
    fn accepted_range_has_equal_ternary_buckets() {
        assert_eq!(u128::from(u64::MAX) % 3, 0);
        for (bits, expected) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (u64::MAX - 3, 0),
            (u64::MAX - 2, 1),
            (u64::MAX - 1, 2),
        ] {
            assert_eq!(exact_ternary(|| bits), expected);
        }
    }

    #[test]
    fn rejected_tail_value_consumes_the_next_draw() {
        let mut draws = [u64::MAX, 5].into_iter();
        assert_eq!(exact_ternary(|| draws.next().unwrap()), 2);
        assert_eq!(draws.next(), None);
    }
}
