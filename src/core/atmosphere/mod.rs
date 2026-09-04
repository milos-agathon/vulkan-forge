//! Spectral precomputed atmosphere transport.

mod bake;
mod precomputed;
mod runtime;
mod spectral;

/// Largest finite value representable by every AETHER rgba16float runtime
/// target. Host and shader seams clamp radiometric scale factors to this
/// shared bound before multiplication.
pub(crate) const AETHER_RADIOMETRIC_SCALE_MAX: f32 = 65_504.0;

#[cfg(feature = "atmosphere-bake")]
pub use bake::bake_atmosphere_luts;
pub use bake::{
    default_atmosphere_luts, generate_reference_equirectangular, load_precomputed_atmosphere_luts,
    reference_aerial_radiance, reference_sky_radiance, AtmosphereConfig, AtmosphereError,
    AtmosphereLutMetadata, AtmosphereLuts, LutData, LutDimensions, ReferenceEnvironment,
    ACCUMULATED_SCATTERING_LUT_SEMANTICS, AERIAL_TRANSMITTANCE_LUT_SEMANTICS,
};
pub(crate) use runtime::tracked_lut_upload_bytes;
pub use runtime::AtmosphereLutHandle;
pub use spectral::{
    mie_phase_cornette_shanks, rayleigh_phase, rayleigh_scattering_coefficient,
    rayleigh_scattering_cross_section, spectral_to_linear_rgb, MieParameters, NUM_WAVELENGTHS,
    WAVELENGTHS_NM,
};
