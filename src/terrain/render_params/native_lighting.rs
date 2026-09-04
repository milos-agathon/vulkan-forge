#[cfg(feature = "extension-module")]
#[derive(Clone, Copy)]
pub enum FilterModeNative {
    Linear,
    Nearest,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy)]
pub enum AddressModeNative {
    Repeat,
    ClampToEdge,
    MirrorRepeat,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct LightSettingsNative {
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct TriplanarSettingsNative {
    pub scale: f32,
    pub blend_sharpness: f32,
    pub normal_strength: f32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct PomSettingsNative {
    pub enabled: bool,
    pub scale: f32,
    pub min_steps: u32,
    pub max_steps: u32,
    pub refine_steps: u32,
    pub shadow: bool,
    pub occlusion: bool,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct LodSettingsNative {
    pub level: i32,
    pub bias: f32,
    pub lod0_bias: f32,
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct ClampSettingsNative {
    pub height_range: (f32, f32),
    pub slope_range: (f32, f32),
    pub ambient_range: (f32, f32),
    pub shadow_range: (f32, f32),
    pub occlusion_range: (f32, f32),
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct SamplingSettingsNative {
    pub mag_filter: FilterModeNative,
    pub min_filter: FilterModeNative,
    pub mip_filter: FilterModeNative,
    pub anisotropy: u32,
    pub address_u: AddressModeNative,
    pub address_v: AddressModeNative,
    pub address_w: AddressModeNative,
}

/// Shadow settings extracted from Python ShadowSettings dataclass
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct ShadowSettingsNative {
    pub enabled: bool,
    pub technique: String,
    pub resolution: u32,
    pub cascades: u32,
    pub max_distance: f32,
    pub softness: f32,
    /// Legacy world-space radius, converted to texels per cascade by WGSL.
    pub pcss_light_radius: f32,
    pub pcss_blocker_radius: f32,
    pub pcss_filter_radius: f32,
    /// PCSS area-light radius in shadow-map texels.
    pub light_size: f32,
    pub intensity: f32,
    pub slope_scale_bias: f32,
    pub depth_bias: f32,
    pub normal_bias: f32,
}

#[cfg(feature = "extension-module")]
impl Default for ShadowSettingsNative {
    fn default() -> Self {
        Self {
            enabled: true,
            technique: "PCSS".to_string(),
            resolution: 2048,
            cascades: 4,
            max_distance: 3000.0,
            softness: 0.01,
            pcss_light_radius: 0.0,
            pcss_blocker_radius: crate::shadows::DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS,
            pcss_filter_radius: crate::shadows::DEFAULT_PCSS_FILTER_RADIUS_TEXELS,
            light_size: crate::shadows::DEFAULT_PCSS_LIGHT_SIZE,
            intensity: 1.0,
            slope_scale_bias: 0.001,
            depth_bias: 0.0005,
            normal_bias: 0.0002,
        }
    }
}
