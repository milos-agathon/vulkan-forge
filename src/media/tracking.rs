use serde::{Deserialize, Serialize};

use super::model::normalize;
use super::{MajorantGrid, MediaError, Medium, Rgb};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleIdentity {
    pub frame: u64,
    pub pixel: u64,
    pub sample: u64,
    pub bounce: u32,
}

impl SampleIdentity {
    pub(super) fn random_bits(self, dimension: u64) -> u64 {
        let mut value = self.frame ^ self.pixel.rotate_left(13) ^ self.sample.rotate_left(29);
        value ^= (self.bounce as u64).rotate_left(47) ^ dimension.wrapping_mul(0x9e3779b97f4a7c15);
        value ^= value >> 30;
        value = value.wrapping_mul(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94d049bb133111eb);
        value ^= value >> 31;
        value
    }

    pub fn uniform(self, dimension: u64) -> f32 {
        let value = self.random_bits(dimension);
        // A 23-bit midpoint grid is exactly representable in f32. Using 24
        // bits lets the upper midpoint round to 1.0, which violates the
        // sampler contract and can turn floor(3*u) into channel index 3.
        (((value >> 41) as f32) + 0.5) * (1.0 / (1u32 << 23) as f32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
}

impl Ray {
    fn normalized(self) -> Result<Self, MediaError> {
        if self.origin.iter().any(|value| !value.is_finite()) {
            return Err(MediaError::InvalidTransport(
                "ray origin must be finite".into(),
            ));
        }
        Ok(Self {
            origin: self.origin,
            direction: normalize(self.direction).ok_or_else(|| {
                MediaError::InvalidTransport("ray direction must be finite and nonzero".into())
            })?,
        })
    }

    fn at(self, distance: f32) -> [f32; 3] {
        std::array::from_fn(|axis| self.origin[axis] + self.direction[axis] * distance)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Collision {
    pub distance: f32,
    pub position: [f32; 3],
    pub extinction: Rgb,
    pub sample_identity: SampleIdentity,
    pub step_count: u64,
}

/// Immutable pairing of a validated medium and its canonical majorant.
/// Construction performs the structural proof/hash work once; individual
/// tracking samples only validate their ray and segment inputs.
#[derive(Debug, Clone)]
pub struct TrackingContext {
    medium: Medium,
    majorant: MajorantGrid,
    global_rate: f32,
}

impl TrackingContext {
    pub fn new(medium: Medium, majorant: MajorantGrid) -> Result<Self, MediaError> {
        majorant.validate_for(&medium)?;
        let global_rate = majorant.global();
        validate_segment(0.0, global_rate)?;
        Ok(Self {
            medium,
            majorant,
            global_rate,
        })
    }

    pub fn medium(&self) -> &Medium {
        &self.medium
    }

    pub fn majorant(&self) -> &MajorantGrid {
        &self.majorant
    }
}

fn validate_segment(distance: f32, majorant: f32) -> Result<(), MediaError> {
    if !distance.is_finite() || distance < 0.0 {
        return Err(MediaError::InvalidTransport(
            "segment distance must be finite and non-negative".into(),
        ));
    }
    if !majorant.is_finite() || majorant < 0.0 {
        return Err(MediaError::InvalidMajorant(
            "tracking majorant must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

/// Unbiased ratio-tracking estimator using a scalar majorant shared by RGB.
pub fn ratio_track(
    context: &TrackingContext,
    ray: Ray,
    distance: f32,
    identity: SampleIdentity,
) -> Result<(Rgb, u64), MediaError> {
    let ray = ray.normalized()?;
    let rate = context.global_rate;
    validate_segment(distance, rate)?;
    if rate == 0.0 || distance == 0.0 {
        return Ok((Rgb::ONE, 0));
    }
    let mut transmittance = [1.0; 3];
    let mut traveled = 0.0;
    let mut step = 0u64;
    loop {
        traveled += -identity.uniform(step).ln() / rate;
        if traveled >= distance {
            return Ok((Rgb::new(transmittance, "ratio tracking weight")?, step));
        }
        let extinction = context.medium.extinction_at(ray.at(traveled));
        for (weight, sigma) in transmittance.iter_mut().zip(extinction.components()) {
            if sigma > rate {
                return Err(MediaError::InvalidMajorant(format!(
                    "sampled extinction {sigma} exceeds tracking rate {rate}"
                )));
            }
            *weight *= 1.0 - sigma / rate;
        }
        step += 1;
    }
}

/// Woodcock collision sampling for one spectral channel.
pub fn delta_track(
    context: &TrackingContext,
    ray: Ray,
    distance: f32,
    channel: usize,
    identity: SampleIdentity,
) -> Result<Option<Collision>, MediaError> {
    delta_track_counted(context, ray, distance, channel, identity).map(|value| value.0)
}

/// Delta tracking with the number of candidate points evaluated, for
/// reference diagnostics. The estimator and sample identity are identical to
/// [`delta_track`].
pub fn delta_track_counted(
    context: &TrackingContext,
    ray: Ray,
    distance: f32,
    channel: usize,
    identity: SampleIdentity,
) -> Result<(Option<Collision>, u64), MediaError> {
    if channel >= 3 {
        return Err(MediaError::InvalidTransport(
            "spectral channel must be 0, 1, or 2".into(),
        ));
    }
    let ray = ray.normalized()?;
    let rate = context.global_rate;
    validate_segment(distance, rate)?;
    if rate == 0.0 || distance == 0.0 {
        return Ok((None, 0));
    }
    let mut traveled = 0.0;
    let mut step = 0u64;
    loop {
        traveled += -identity.uniform(step.wrapping_mul(2)).ln() / rate;
        if traveled >= distance {
            return Ok((None, step));
        }
        let position = ray.at(traveled);
        let extinction = context.medium.extinction_at(position);
        let sigma = extinction.components()[channel];
        if sigma > rate {
            return Err(MediaError::InvalidMajorant(format!(
                "sampled extinction {} exceeds tracking rate {rate}",
                sigma
            )));
        }
        if identity.uniform(step.wrapping_mul(2).wrapping_add(1)) < sigma / rate {
            return Ok((
                Some(Collision {
                    distance: traveled,
                    position,
                    extinction,
                    sample_identity: identity,
                    step_count: step + 1,
                }),
                step + 1,
            ));
        }
        step += 1;
    }
}

pub fn russian_roulette(
    throughput: Rgb,
    uniform: f32,
    minimum_probability: f32,
) -> Result<Option<f32>, MediaError> {
    if !uniform.is_finite()
        || !(0.0..1.0).contains(&uniform)
        || !minimum_probability.is_finite()
        || !(0.0..=1.0).contains(&minimum_probability)
    {
        return Err(MediaError::InvalidTransport(
            "roulette inputs must be finite probabilities".into(),
        ));
    }
    let probability = throughput.max_component().max(minimum_probability).min(1.0);
    Ok((uniform < probability).then(|| probability.recip()))
}
