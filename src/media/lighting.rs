use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

use super::model::normalize;
use super::{MediaError, Rgb};

pub fn power_heuristic(pdf_a: f32, pdf_b: f32) -> Result<(f32, f32), MediaError> {
    if !pdf_a.is_finite() || !pdf_b.is_finite() || pdf_a < 0.0 || pdf_b < 0.0 {
        return Err(MediaError::InvalidTransport(
            "MIS PDFs must be finite and non-negative".into(),
        ));
    }
    let scale = pdf_a.max(pdf_b);
    if scale == 0.0 {
        Ok((0.5, 0.5))
    } else {
        let a = pdf_a / scale;
        let b = pdf_b / scale;
        let sum = a * a + b * b;
        Ok((a * a / sum, b * b / sum))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectionalSun {
    pub direction_to_sun: [f32; 3],
    pub radiance: Rgb,
}

impl DirectionalSun {
    pub fn new(direction_to_sun: [f32; 3], radiance: [f32; 3]) -> Result<Self, MediaError> {
        Ok(Self {
            direction_to_sun: normalize(direction_to_sun).ok_or_else(|| {
                MediaError::InvalidTransport("sun direction must be finite and nonzero".into())
            })?,
            radiance: Rgb::new(radiance, "sun radiance")?,
        })
    }

    pub fn continuous_pdf(&self, _direction: [f32; 3]) -> Option<f32> {
        None
    }

    pub fn mis_weight(&self) -> f32 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentSample {
    pub direction: [f32; 3],
    pub radiance: Rgb,
    pub pdf_solid_angle: f32,
    pub texel: [u32; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentDistribution {
    width: u32,
    height: u32,
    radiance: Vec<Rgb>,
    probabilities: Vec<f64>,
    solid_angles: Vec<f64>,
    conditional_cdf: Vec<f64>,
    marginal_cdf: Vec<f64>,
}

impl EnvironmentDistribution {
    pub fn new(width: u32, height: u32, radiance: Vec<Rgb>) -> Result<Self, MediaError> {
        let count = (width as usize).checked_mul(height as usize);
        if width == 0 || height == 0 || count != Some(radiance.len()) {
            return Err(MediaError::InvalidEnvironment(
                "dimensions must be nonzero and match texel count".into(),
            ));
        }
        if radiance
            .iter()
            .flat_map(|rgb| rgb.components())
            .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err(MediaError::InvalidEnvironment(
                "radiance must be finite and non-negative".into(),
            ));
        }
        let mut weights = vec![0.0f64; radiance.len()];
        let mut solid_angles = vec![0.0f64; radiance.len()];
        let mut row_weights = vec![0.0f64; height as usize];
        for y in 0..height {
            let theta0 = PI * y as f64 / height as f64;
            let theta1 = PI * (y + 1) as f64 / height as f64;
            let solid_angle = (TAU / width as f64) * (theta0.cos() - theta1.cos());
            for x in 0..width {
                let index = y as usize * width as usize + x as usize;
                solid_angles[index] = solid_angle;
                weights[index] = radiance[index].luminance() as f64 * solid_angle;
                row_weights[y as usize] += weights[index];
            }
        }
        let total: f64 = row_weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(MediaError::InvalidEnvironment(
                "importance weights have zero or non-finite total".into(),
            ));
        }
        let probabilities = weights
            .iter()
            .map(|weight| weight / total)
            .collect::<Vec<_>>();
        if probabilities.iter().zip(&solid_angles).any(|(p, omega)| {
            let pdf = p / omega;
            let represented = pdf as f32;
            !omega.is_finite()
                || *omega <= 0.0
                || !pdf.is_finite()
                || pdf < 0.0
                || (pdf > 0.0 && (!represented.is_finite() || represented == 0.0))
        }) {
            return Err(MediaError::InvalidEnvironment(
                "solid-angle PDFs must be finite and non-negative".into(),
            ));
        }
        let mut conditional_cdf = vec![0.0; probabilities.len()];
        for y in 0..height as usize {
            let mut cumulative = 0.0;
            for x in 0..width as usize {
                let index = y * width as usize + x;
                cumulative += if row_weights[y] > 0.0 {
                    weights[index] / row_weights[y]
                } else {
                    1.0 / width as f64
                };
                conditional_cdf[index] = cumulative.min(1.0);
            }
            conditional_cdf[(y + 1) * width as usize - 1] = 1.0;
        }
        let mut marginal_cdf = Vec::with_capacity(height as usize);
        let mut cumulative = 0.0;
        for weight in row_weights {
            cumulative += weight / total;
            marginal_cdf.push(cumulative.min(1.0));
        }
        *marginal_cdf.last_mut().expect("height is nonzero") = 1.0;
        Ok(Self {
            width,
            height,
            radiance,
            probabilities,
            solid_angles,
            conditional_cdf,
            marginal_cdf,
        })
    }

    pub fn sample(&self, u: [f32; 4]) -> Result<EnvironmentSample, MediaError> {
        if !u.iter().all(|v| v.is_finite() && (0.0..1.0).contains(v)) {
            return Err(MediaError::InvalidEnvironment(
                "samples must lie in [0, 1)".into(),
            ));
        }
        let y = select_cdf(&self.marginal_cdf, u[0] as f64);
        let row = &self.conditional_cdf[y * self.width as usize..(y + 1) * self.width as usize];
        let x = select_cdf(row, u[1] as f64);
        let mut direction = self.texel_direction(x, y, u[2] as f64, u[3] as f64);
        let selected = [x as u32, y as u32];
        if self.direction_texel(direction) != selected {
            // Exact boundaries can move to an adjacent half-open texel after
            // f64 -> f32 direction quantization. They have zero continuous
            // measure; map them to the selected texel center so sample(),
            // pdf(), and radiance() remain on one represented measure.
            direction = self.texel_direction(x, y, 0.5, 0.5);
        }
        let direction_texel = self.direction_texel(direction);
        let index = direction_texel[1] as usize * self.width as usize + direction_texel[0] as usize;
        Ok(EnvironmentSample {
            direction,
            radiance: self.radiance[index],
            pdf_solid_angle: self.texel_pdf(direction_texel[0], direction_texel[1]),
            texel: direction_texel,
        })
    }

    pub fn pdf(&self, direction: [f32; 3]) -> f32 {
        let Some(direction) = normalize(direction) else {
            return 0.0;
        };
        let [x, y] = self.direction_texel(direction);
        self.texel_pdf(x, y)
    }

    pub fn radiance(&self, direction: [f32; 3]) -> Rgb {
        let Some(direction) = normalize(direction) else {
            return Rgb::ZERO;
        };
        let [x, y] = self.direction_texel(direction);
        self.radiance[y as usize * self.width as usize + x as usize]
    }

    pub fn probability_sum(&self) -> f64 {
        self.probabilities.iter().sum()
    }

    pub fn solid_angle_sum(&self) -> f64 {
        self.solid_angles.iter().sum()
    }

    fn texel_pdf(&self, x: u32, y: u32) -> f32 {
        let index = y as usize * self.width as usize + x as usize;
        (self.probabilities[index] / self.solid_angles[index]) as f32
    }

    fn direction_texel(&self, direction: [f32; 3]) -> [u32; 2] {
        let theta = (direction[0] as f64)
            .hypot(direction[2] as f64)
            .atan2(direction[1] as f64);
        let mut phi = (direction[2] as f64).atan2(direction[0] as f64);
        if phi < 0.0 {
            phi += TAU;
        }
        [
            ((phi / TAU) * self.width as f64)
                .floor()
                .min((self.width - 1) as f64) as u32,
            ((theta / PI) * self.height as f64)
                .floor()
                .min((self.height - 1) as f64) as u32,
        ]
    }

    fn texel_direction(&self, x: usize, y: usize, u_theta: f64, u_phi: f64) -> [f32; 3] {
        let theta0 = PI * y as f64 / self.height as f64;
        let theta1 = PI * (y + 1) as f64 / self.height as f64;
        let cos_theta = theta0.cos() + u_theta * (theta1.cos() - theta0.cos());
        let phi = TAU * (x as f64 + u_phi) / self.width as f64;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        [
            (sin_theta * phi.cos()) as f32,
            cos_theta as f32,
            (sin_theta * phi.sin()) as f32,
        ]
    }
}

fn select_cdf(cdf: &[f64], sample: f64) -> usize {
    cdf.partition_point(|value| *value <= sample)
        .min(cdf.len() - 1)
}
