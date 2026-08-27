#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct AovSettingsNative {
    pub enabled: bool,
    pub albedo: bool,
    pub normal: bool,
    pub depth: bool,
    pub transmittance: bool,
    pub in_scatter: bool,
    pub cloud_shadow: bool,
    pub optical_depth: bool,
    /// VERITAS: capture the per-pixel VT source-id map (R32Uint attachment).
    pub source_id: bool,
    pub output_dir: Option<String>,
    pub format: String,
}

#[cfg(feature = "extension-module")]
impl Default for AovSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            albedo: true,
            normal: true,
            depth: true,
            transmittance: false,
            in_scatter: false,
            cloud_shadow: false,
            optical_depth: false,
            source_id: false,
            output_dir: None,
            format: "png".to_string(),
        }
    }
}

#[cfg(feature = "extension-module")]
impl AovSettingsNative {
    pub fn any_enabled(&self) -> bool {
        self.enabled
            && (self.albedo
                || self.normal
                || self.depth
                || self.transmittance
                || self.in_scatter
                || self.cloud_shadow
                || self.optical_depth
                || self.source_id)
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct ScreenSpaceSettingsNative {
    pub enabled: bool,
    pub ssao_enabled: bool,
    pub ssao_radius: f32,
    pub ssao_intensity: f32,
    pub ssgi_enabled: bool,
    pub ssgi_intensity: f32,
    pub ssr_enabled: bool,
    pub ssr_intensity: f32,
    pub taa_enabled: bool,
    pub temporal_alpha: f32,
}

#[cfg(feature = "extension-module")]
impl Default for ScreenSpaceSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            ssao_enabled: false,
            ssao_radius: 1.5,
            ssao_intensity: 1.0,
            ssgi_enabled: false,
            ssgi_intensity: 1.0,
            ssr_enabled: false,
            ssr_intensity: 1.0,
            taa_enabled: false,
            temporal_alpha: 0.1,
        }
    }
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct DenoiseSettingsNative {
    pub enabled: bool,
    pub method: DenoiseMethodNative,
    pub iterations: u32,
    pub sigma_color: f32,
    pub sigma_normal: f32,
    pub sigma_depth: f32,
    pub edge_stopping: f32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMethodNative {
    Atrous,
    Oidn,
    None,
}

#[cfg(feature = "extension-module")]
impl Default for DenoiseSettingsNative {
    fn default() -> Self {
        Self {
            enabled: false,
            method: DenoiseMethodNative::Atrous,
            iterations: 3,
            sigma_color: 0.1,
            sigma_normal: 0.1,
            sigma_depth: 0.1,
            edge_stopping: 1.0,
        }
    }
}

#[cfg(feature = "extension-module")]
impl DenoiseSettingsNative {
    pub fn uses_guidance(&self) -> bool {
        (self.sigma_normal > 0.001 || self.sigma_depth > 0.001)
            && self.method == DenoiseMethodNative::Atrous
    }
}
