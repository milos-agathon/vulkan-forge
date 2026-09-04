mod defaults;
mod types;
mod updates;

#[allow(unused_imports)]
pub use types::{
    DenoiseConfig, DensityVolumeConfig, DofConfig, HeightAoConfig, LensEffectsConfig,
    MaterialLayerConfig, MotionBlurConfig, SunVisConfig, TonemapConfig, VectorOverlayConfig,
    ViewerTerrainPbrConfig, VolumetricsConfig,
};

pub(crate) fn shadow_technique_requires_moments(name: &str) -> bool {
    if name.eq_ignore_ascii_case("none") {
        return false;
    }
    crate::lighting::types::ShadowTechnique::from_name(name)
        .is_some_and(|technique| technique.requires_moments())
}
