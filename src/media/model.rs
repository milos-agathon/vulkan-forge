use serde::{Deserialize, Deserializer, Serialize};
use std::f32::consts::TAU;

use super::{density::medium_identity, DensityField, MediumIdentity};

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum MediaError {
    #[error("{field} must contain finite, non-negative values")]
    InvalidSpectrum { field: &'static str },
    #[error("phase anisotropy g must be finite and strictly between -1 and 1")]
    InvalidAnisotropy,
    #[error("{0}")]
    InvalidDensity(String),
    #[error("cannot construct a finite conservative majorant: {0}")]
    InvalidMajorant(String),
    #[error("invalid environment distribution: {0}")]
    InvalidEnvironment(String),
    #[error("invalid transport input: {0}")]
    InvalidTransport(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Rgb([f32; 3]);

impl Rgb {
    pub const ZERO: Self = Self([0.0; 3]);
    pub const ONE: Self = Self([1.0; 3]);

    pub fn new(value: [f32; 3], field: &'static str) -> Result<Self, MediaError> {
        if value.iter().all(|v| v.is_finite() && *v >= 0.0) {
            Ok(Self(value))
        } else {
            Err(MediaError::InvalidSpectrum { field })
        }
    }

    pub fn max_component(self) -> f32 {
        self.0.into_iter().fold(0.0, f32::max)
    }

    pub fn luminance(self) -> f32 {
        self.0[0] * 0.2126 + self.0[1] * 0.7152 + self.0[2] * 0.0722
    }

    pub fn components(self) -> [f32; 3] {
        self.0
    }

    pub(crate) fn scale(self, factor: f32) -> Self {
        Self(self.0.map(|v| v * factor))
    }
}

impl<'de> Deserialize<'de> for Rgb {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = <[f32; 3]>::deserialize(deserializer)?;
        Self::new(value, "RGB").map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Phase {
    Isotropic,
    HenyeyGreenstein { g: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSample {
    pub direction: [f32; 3],
    pub value: f32,
    pub pdf: f32,
}

impl Phase {
    pub fn henyey_greenstein(g: f32) -> Result<Self, MediaError> {
        if g.is_finite() && g > -1.0 && g < 1.0 {
            Ok(Self::HenyeyGreenstein { g })
        } else {
            Err(MediaError::InvalidAnisotropy)
        }
    }

    pub fn validate(self) -> Result<(), MediaError> {
        match self {
            Self::Isotropic => Ok(()),
            Self::HenyeyGreenstein { g } => Self::henyey_greenstein(g).map(|_| ()),
        }
    }

    pub fn evaluate(self, cos_theta: f32) -> Result<f32, MediaError> {
        self.validate()?;
        if !cos_theta.is_finite() {
            return Err(MediaError::InvalidTransport(
                "phase cosine must be finite".into(),
            ));
        }
        let cosine = f64::from(cos_theta.clamp(-1.0, 1.0));
        let value = match self {
            Self::Isotropic => 1.0 / (4.0 * std::f64::consts::PI),
            Self::HenyeyGreenstein { g } => {
                let g = f64::from(g);
                let denominator = if g >= 0.0 {
                    (1.0 - g).powi(2) + 2.0 * g * (1.0 - cosine)
                } else {
                    (1.0 + g).powi(2) - 2.0 * g * (1.0 + cosine)
                };
                ((1.0 - g) * (1.0 + g))
                    / (4.0 * std::f64::consts::PI * denominator * denominator.sqrt())
            }
        };
        Rgb::new([value as f32, 0.0, 0.0], "phase value").map(|rgb| rgb.0[0])
    }

    pub fn sample(self, incident: [f32; 3], u: [f32; 2]) -> Result<PhaseSample, MediaError> {
        self.validate()?;
        if !u.iter().all(|v| v.is_finite() && (0.0..1.0).contains(v)) {
            return Err(MediaError::InvalidTransport(
                "phase samples must lie in [0, 1)".into(),
            ));
        }
        let w = normalize(incident).ok_or_else(|| {
            MediaError::InvalidTransport("incident direction must be finite and nonzero".into())
        })?;
        let cos_theta = match self {
            Self::Isotropic => 2.0 * u[0] - 1.0,
            Self::HenyeyGreenstein { g: 0.0 } => 2.0 * u[0] - 1.0,
            Self::HenyeyGreenstein { g } => {
                let g = g as f64;
                let q = 2.0 * u[0] as f64 - 1.0;
                let denominator = 1.0 + g * q;
                // Algebraically factored inverse CDF. Unlike the common
                // difference-of-squares form, this retains the g -> 0 limit.
                let numerator =
                    2.0 * q + g * (q * q + 3.0) + 2.0 * g * g * q + g * g * g * (q * q - 1.0);
                (numerator / (2.0 * denominator * denominator)).clamp(-1.0, 1.0) as f32
            }
        };
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = TAU * u[1];
        let helper = if w[2].abs() < 0.999 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let tangent = normalize(cross(helper, w)).expect("helper is not parallel");
        let bitangent = cross(w, tangent);
        let direction = add(
            add(
                scale(tangent, sin_theta * phi.cos()),
                scale(bitangent, sin_theta * phi.sin()),
            ),
            scale(w, cos_theta),
        );
        let value = self.evaluate(cos_theta)?;
        Ok(PhaseSample {
            direction,
            value,
            pdf: value,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Medium {
    sigma_a: Rgb,
    sigma_s: Rgb,
    phase: Phase,
    density: DensityField,
}

#[derive(Deserialize)]
struct MediumWire {
    sigma_a: Rgb,
    sigma_s: Rgb,
    phase: Phase,
    density: DensityField,
}

impl<'de> Deserialize<'de> for Medium {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MediumWire::deserialize(deserializer)?;
        let medium = Self {
            sigma_a: wire.sigma_a,
            sigma_s: wire.sigma_s,
            phase: wire.phase,
            density: wire.density,
        };
        medium.validate().map_err(serde::de::Error::custom)?;
        Ok(medium)
    }
}

impl Medium {
    pub fn new(
        sigma_a: [f32; 3],
        sigma_s: [f32; 3],
        phase: Phase,
        density: DensityField,
    ) -> Result<Self, MediaError> {
        let medium = Self {
            sigma_a: Rgb::new(sigma_a, "sigma_a")?,
            sigma_s: Rgb::new(sigma_s, "sigma_s")?,
            phase,
            density,
        };
        medium.validate()?;
        Ok(medium)
    }

    pub fn validate(&self) -> Result<(), MediaError> {
        Rgb::new(self.sigma_a.0, "sigma_a")?;
        Rgb::new(self.sigma_s.0, "sigma_s")?;
        Rgb::new(self.sigma_t().0, "sigma_t")?;
        self.phase.validate()?;
        self.density.validate()?;
        let maximum_density = self.density.maximum_physical_density()?;
        if self
            .sigma_t()
            .0
            .into_iter()
            .any(|sigma| !(sigma * maximum_density).is_finite())
        {
            return Err(MediaError::InvalidSpectrum {
                field: "density * sigma_t",
            });
        }
        Ok(())
    }

    pub fn sigma_t(&self) -> Rgb {
        Rgb([
            self.sigma_a.0[0] + self.sigma_s.0[0],
            self.sigma_a.0[1] + self.sigma_s.0[1],
            self.sigma_a.0[2] + self.sigma_s.0[2],
        ])
    }

    pub fn sigma_a(&self) -> Rgb {
        self.sigma_a
    }

    pub fn sigma_s(&self) -> Rgb {
        self.sigma_s
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn density(&self) -> &DensityField {
        &self.density
    }

    /// Complete deterministic identity used by caches and temporal history.
    ///
    /// The digest covers the validated serialized density payload (including
    /// R16 storage bits, transforms, bounds, mapping and density parameters),
    /// both coefficient spectra, and the phase. `version` is the caller-owned
    /// resource generation and is deliberately part of the returned identity.
    pub fn identity(&self, version: u64) -> MediumIdentity {
        medium_identity(self, version)
    }

    pub fn extinction_at(&self, point: [f32; 3]) -> Rgb {
        self.sigma_t().scale(self.density.physical_density(point))
    }

    pub fn homogeneous_transmittance(&self, distance: f32) -> Result<Rgb, MediaError> {
        if !distance.is_finite() || distance < 0.0 {
            return Err(MediaError::InvalidTransport(
                "distance must be finite and non-negative".into(),
            ));
        }
        let density = match &self.density {
            DensityField::Homogeneous(field) => field.mapping.to_physical(field.authored_density),
            _ => {
                return Err(MediaError::InvalidTransport(
                    "closed-form transmittance requires a homogeneous field".into(),
                ))
            }
        };
        Ok(Rgb(self
            .sigma_t()
            .0
            .map(|v| (-v * density * distance).exp())))
    }

    /// Independent closed-form single-scatter oracle for a homogeneous slab.
    ///
    /// Camera and collimated source lie on opposite slab boundaries, making
    /// their combined extinction distance constant at every scattering point.
    pub fn homogeneous_single_scatter_slab(
        &self,
        distance: f32,
        cos_theta: f32,
        incident_radiance: [f32; 3],
    ) -> Result<Rgb, MediaError> {
        if !distance.is_finite() || distance < 0.0 || !cos_theta.is_finite() {
            return Err(MediaError::InvalidTransport(
                "slab distance and scattering cosine must be finite".into(),
            ));
        }
        let incident = Rgb::new(incident_radiance, "incident radiance")?;
        let density = match &self.density {
            DensityField::Homogeneous(field) => field.mapping.to_physical(field.authored_density),
            _ => {
                return Err(MediaError::InvalidTransport(
                    "single-scatter slab oracle requires a homogeneous field".into(),
                ))
            }
        };
        let phase = f64::from(self.phase.evaluate(cos_theta)?);
        Rgb::new(
            std::array::from_fn(|channel| {
                let optical_depth =
                    f64::from(self.sigma_t().0[channel]) * f64::from(density) * f64::from(distance);
                (f64::from(incident.0[channel])
                    * f64::from(self.sigma_s.0[channel])
                    * f64::from(density)
                    * phase
                    * f64::from(distance)
                    * (-optical_depth).exp()) as f32
            }),
            "single-scatter radiance",
        )
    }
}

pub(crate) fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let length2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if !length2.is_finite() || length2 <= 0.0 {
        return None;
    }
    let inv = length2.sqrt().recip();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}
