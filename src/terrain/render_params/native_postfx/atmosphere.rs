#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct VolumetricsSettingsNative {
    pub enabled: bool,
    pub mode: VolumetricsModeNative,
    pub density: f32,
    pub height_falloff: f32,
    pub base_height: f32,
    pub scattering: f32,
    pub absorption: f32,
    pub phase_g: f32,
    pub light_shafts: bool,
    pub shaft_intensity: f32,
    pub shaft_samples: u32,
    pub use_shadows: bool,
    pub half_res: bool,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VolumetricsModeNative {
    Uniform,
    Height,
    Exponential,
}

#[cfg(feature = "extension-module")]
impl Default for VolumetricsSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: VolumetricsModeNative::Uniform,
            density: 0.01,
            height_falloff: 0.1,
            base_height: 0.0,
            scattering: 0.5,
            absorption: 0.1,
            phase_g: 0.0,
            light_shafts: false,
            shaft_intensity: 1.0,
            shaft_samples: 32,
            use_shadows: true,
            half_res: false,
        }
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct SkySettingsNative {
    pub enabled: bool,
    pub model: u32,
    pub turbidity: f32,
    pub ground_albedo: f32,
    pub ozone_du: f32,
    pub mie_g: f32,
    pub sun_intensity: f32,
    pub sun_size: f32,
    pub aerial_perspective: bool,
    pub aerial_density: f32,
    pub sky_exposure: f32,
    /// Exact tracked LUT payload. AETHER always resolves either the shipped
    /// bank or a caller-provided offline bake into this typed handoff.
    pub lut_handle: Option<crate::core::atmosphere::AtmosphereLutHandle>,
}

#[cfg(feature = "extension-module")]
impl Default for SkySettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            model: 1,
            turbidity: 2.0,
            ground_albedo: 0.3,
            ozone_du: 300.0,
            mie_g: 0.8,
            sun_intensity: 1.0,
            sun_size: 1.0,
            aerial_perspective: true,
            aerial_density: 1.0,
            sky_exposure: 1.0,
            lut_handle: None,
        }
    }
}
