use super::common::normalize_key;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShadowTechnique {
    Hard,
    Pcf,
    Pcss,
    Vsm,
    Evsm,
    Msm,
    Csm,
}

impl ShadowTechnique {
    pub fn canonical(self) -> &'static str {
        match self {
            Self::Hard => "hard",
            Self::Pcf => "pcf",
            Self::Pcss => "pcss",
            Self::Vsm => "vsm",
            Self::Evsm => "evsm",
            Self::Msm => "msm",
            Self::Csm => "csm",
        }
    }
}

impl FromStr for ShadowTechnique {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let key = normalize_key(value);
        Ok(match key.as_str() {
            "hard" => Self::Hard,
            "pcf" => Self::Pcf,
            "pcss" => Self::Pcss,
            "vsm" => Self::Vsm,
            "evsm" => Self::Evsm,
            "msm" => Self::Msm,
            "csm" => Self::Csm,
            _ => return Err("unknown shadow technique"),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShadowParams {
    #[serde(default = "ShadowParams::default_enabled")]
    pub enabled: bool,
    #[serde(default = "ShadowParams::default_technique")]
    pub technique: ShadowTechnique,
    #[serde(default = "ShadowParams::default_map_size")]
    pub map_size: u32,
    #[serde(default = "ShadowParams::default_cascades")]
    pub cascades: u32,
    #[serde(default = "ShadowParams::default_contact_hardening")]
    pub contact_hardening: bool,
    /// Blocker-search radius in shadow-map texels.
    #[serde(default = "ShadowParams::default_pcss_blocker_radius")]
    pub pcss_blocker_radius: f32,
    /// Maximum adaptive PCSS filter radius in shadow-map texels.
    #[serde(default = "ShadowParams::default_pcss_filter_radius")]
    pub pcss_filter_radius: f32,
    /// Area-light radius in shadow-map texels used by the PCSS penumbra estimate.
    #[serde(default = "ShadowParams::default_light_size")]
    pub light_size: f32,
    #[serde(default = "ShadowParams::default_moment_bias")]
    pub moment_bias: f32,
}

impl ShadowParams {
    const fn default_enabled() -> bool {
        true
    }

    fn default_technique() -> ShadowTechnique {
        ShadowTechnique::Pcf
    }

    const fn default_map_size() -> u32 {
        2048
    }

    const fn default_cascades() -> u32 {
        4
    }

    const fn default_contact_hardening() -> bool {
        true
    }

    const fn default_pcss_blocker_radius() -> f32 {
        crate::shadows::DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS
    }

    const fn default_pcss_filter_radius() -> f32 {
        crate::shadows::DEFAULT_PCSS_FILTER_RADIUS_TEXELS
    }

    const fn default_light_size() -> f32 {
        crate::shadows::DEFAULT_PCSS_LIGHT_SIZE
    }

    const fn default_moment_bias() -> f32 {
        0.0005
    }

    pub fn requires_moments(&self) -> bool {
        matches!(
            self.technique,
            ShadowTechnique::Vsm | ShadowTechnique::Evsm | ShadowTechnique::Msm
        )
    }

    pub fn atlas_memory_bytes(&self) -> u64 {
        let cascades = self.cascades.max(1) as u64;
        let resolution = self.map_size.max(1) as u64;
        let depth_bytes = cascades * resolution * resolution * 4;
        // All moment techniques use one Rgba16Float atlas plus one persistent
        // Rgba16Float intermediate for the separable blur.
        let moment_bytes = if self.requires_moments() {
            cascades * resolution * resolution * 16
        } else {
            0
        };
        depth_bytes + moment_bytes
    }

    pub fn is_power_of_two_map(&self) -> bool {
        self.map_size.is_power_of_two()
    }

    /// Convert ShadowParams from RendererConfig to ShadowManagerConfig (P3-11)
    pub fn to_shadow_manager_config(&self) -> crate::shadows::ShadowManagerConfig {
        use crate::shadows::{CsmConfig, ShadowManagerConfig};

        let csm = CsmConfig {
            cascade_count: self.cascades,
            shadow_map_size: self.map_size,
            max_shadow_distance: 1000.0, // Will be overridden by camera far plane
            cascade_splits: vec![],      // Empty = auto-calculate splits
            pcf_kernel_size: 3,          // Default 3x3 PCF
            depth_bias: 0.0005,
            slope_bias: 0.001,
            peter_panning_offset: 0.002,
            enable_evsm: matches!(self.technique, ShadowTechnique::Evsm),
            evsm_positive_exp: crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            evsm_negative_exp: crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            debug_mode: 0,
            enable_unclipped_depth: false,
            depth_clip_factor: 1.0,
            stabilize_cascades: true,
            cascade_blend_range: 0.1,
        };

        // Convert ShadowTechnique from params to lighting::types::ShadowTechnique
        let technique = match self.technique {
            ShadowTechnique::Hard => crate::lighting::types::ShadowTechnique::Hard,
            ShadowTechnique::Pcf => crate::lighting::types::ShadowTechnique::PCF,
            ShadowTechnique::Pcss => crate::lighting::types::ShadowTechnique::PCSS,
            ShadowTechnique::Vsm => crate::lighting::types::ShadowTechnique::VSM,
            ShadowTechnique::Evsm => crate::lighting::types::ShadowTechnique::EVSM,
            ShadowTechnique::Msm => crate::lighting::types::ShadowTechnique::MSM,
            ShadowTechnique::Csm => crate::lighting::types::ShadowTechnique::PCF, // CSM is a layout, use PCF
        };

        ShadowManagerConfig {
            csm,
            technique,
            pcss_blocker_radius: self.pcss_blocker_radius,
            pcss_filter_radius: self.pcss_filter_radius,
            light_size: self.light_size,
            moment_bias: self.moment_bias,
            blur_kernel_radius: crate::shadows::DEFAULT_MOMENT_BLUR_RADIUS,
            max_memory_bytes: 256 * 1024 * 1024, // 256 MiB budget
        }
    }
}

impl Default for ShadowParams {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            technique: Self::default_technique(),
            map_size: Self::default_map_size(),
            cascades: Self::default_cascades(),
            contact_hardening: Self::default_contact_hardening(),
            pcss_blocker_radius: Self::default_pcss_blocker_radius(),
            pcss_filter_radius: Self::default_pcss_filter_radius(),
            light_size: Self::default_light_size(),
            moment_bias: Self::default_moment_bias(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pcss_texel_radii_reach_shadow_manager_unchanged() {
        let mut params = ShadowParams::default();
        let manager = params.to_shadow_manager_config();

        assert_eq!(
            (
                params.pcss_blocker_radius,
                params.pcss_filter_radius,
                params.light_size,
            ),
            (6.0, 4.0, 1.0)
        );
        assert_eq!(
            (
                manager.pcss_blocker_radius,
                manager.pcss_filter_radius,
                manager.light_size,
            ),
            (6.0, 4.0, 1.0)
        );

        params.pcss_blocker_radius = 7.5;
        params.pcss_filter_radius = 3.25;
        params.light_size = 1.5;
        let manager = params.to_shadow_manager_config();
        assert_eq!(
            (
                manager.pcss_blocker_radius,
                manager.pcss_filter_radius,
                manager.light_size,
            ),
            (7.5, 3.25, 1.5)
        );
    }

    #[test]
    fn vsm_memory_includes_persistent_blur_intermediate() {
        let params = ShadowParams {
            technique: ShadowTechnique::Vsm,
            map_size: 4096,
            cascades: 2,
            ..Default::default()
        };

        assert_eq!(params.atlas_memory_bytes(), 640 * 1024 * 1024);
    }
}
