//! Canonical participating-media model and device-independent transport kernel.
//!
//! Densities are authored as non-negative unitless values and become physical
//! densities through [`DensityMapping`]. Extinction is
//! `density * (sigma_a + sigma_s)`.

mod density;
mod diagnostics;
mod lighting;
mod model;
mod reference;
mod tracking;

pub use density::{
    Bounds3, DensityField, DensityIdentity, DensityMapping, Grid3D, Homogeneous, MajorantGrid,
    MajorantProof, MediumIdentity, PerlinWorley, SpatialTransform,
};
pub use diagnostics::AllocationBreakdown;
pub use lighting::{power_heuristic, DirectionalSun, EnvironmentDistribution, EnvironmentSample};
pub use model::{MediaError, Medium, Phase, PhaseSample, Rgb};
pub use reference::{
    trace_reference_sample, ReferenceMediumInterval, ReferenceScene, ReferenceSurfaceHit,
    ReferenceTransportConfig, ReferenceTransportSample,
};
pub use tracking::{
    delta_track, delta_track_counted, ratio_track, russian_roulette, Collision, Ray,
    SampleIdentity, TrackingContext,
};

#[cfg(test)]
mod tests;
