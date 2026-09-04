// src/shadows/csm_types.rs
// Type definitions for Cascaded Shadow Maps
// RELEVANT FILES: shaders/shadows.wgsl, python/forge3d/lighting.py

use crate::lighting::types::ShadowTechnique;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// Shadow cascade configuration for a single cascade level
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowCascade {
    /// Light-space projection matrix for this cascade
    pub light_projection: [f32; 16],
    /// Combined light_view_proj matrix (projection * view)
    /// Pre-computed for efficiency and to ensure consistency with shadow depth pass
    pub light_view_proj: [[f32; 4]; 4],
    /// Near plane distance in view space
    pub near_distance: f32,
    /// Far plane distance in view space
    pub far_distance: f32,
    /// Texel size in world space for this cascade
    pub texel_size: f32,
    /// Padding for alignment
    pub _padding: f32,
}

impl ShadowCascade {
    /// Create a new shadow cascade
    pub fn new(
        near: f32,
        far: f32,
        light_projection: Mat4,
        light_view_proj: Mat4,
        texel_size: f32,
    ) -> Self {
        Self {
            light_projection: light_projection.to_cols_array(),
            light_view_proj: light_view_proj.to_cols_array_2d(),
            near_distance: near,
            far_distance: far,
            texel_size,
            _padding: 0.0,
        }
    }

    /// Get the projection matrix as Mat4
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::from_cols_array(&self.light_projection)
    }
}

/// CSM configuration and uniform data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CsmUniforms {
    /// Directional light direction in world space
    pub light_direction: [f32; 4],
    /// Light view matrix
    pub light_view: [f32; 16],
    /// Shadow cascades (up to 4)
    pub cascades: [ShadowCascade; 4],
    /// Number of active cascades
    pub cascade_count: u32,
    /// PCF kernel size: 1=none, 3=3x3, 5=5x5, 7=poisson
    pub pcf_kernel_size: u32,
    /// Base depth bias to prevent shadow acne
    pub depth_bias: f32,
    /// Slope-scaled bias factor
    pub slope_bias: f32,
    /// Shadow map resolution
    pub shadow_map_size: f32,
    /// Debug visualization mode
    pub debug_mode: u32,
    /// EVSM positive exponent
    pub evsm_positive_exp: f32,
    /// EVSM negative exponent
    pub evsm_negative_exp: f32,
    /// Peter-panning prevention offset
    pub peter_panning_offset: f32,
    /// Enable unclipped depth where supported (B17)
    pub enable_unclipped_depth: u32,
    /// Depth clipping distance factor for cascade adjustment
    pub depth_clip_factor: f32,
    /// Active shadow technique identifier
    pub technique: u32,
    /// Technique feature flags (bitmask)
    pub technique_flags: u32,
    /// Padding to align technique_params to 16-byte boundary
    pub _padding1: [f32; 3],
    /// PCSS blocker/max-filter texel radii, moment bias, and texel-space light radius.
    pub technique_params: [f32; 4],
    /// Reserved for future expansions (e.g., MSM tuning)
    pub technique_reserved: [f32; 4],
    /// Cascade blend range (0.0 = no blend, 0.1 = 10% blend at boundaries)
    pub cascade_blend_range: f32,
    /// Padding for std430 alignment (storage buffer) - must reach 864 bytes total
    pub _padding2: [f32; 27], // Additional padding: 108 bytes to reach 864 total
}

impl Default for CsmUniforms {
    fn default() -> Self {
        Self {
            light_direction: [0.0, -1.0, 0.0, 0.0],
            light_view: Mat4::IDENTITY.to_cols_array(),
            cascades: [ShadowCascade {
                light_projection: Mat4::IDENTITY.to_cols_array(),
                light_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                near_distance: 0.0,
                far_distance: 100.0,
                texel_size: 0.1,
                _padding: 0.0,
            }; 4],
            cascade_count: 3,
            pcf_kernel_size: 3,
            depth_bias: 0.005,
            slope_bias: 0.01,
            shadow_map_size: 2048.0,
            debug_mode: 0,
            evsm_positive_exp: crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            evsm_negative_exp: crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            peter_panning_offset: 0.001,
            enable_unclipped_depth: 0,
            depth_clip_factor: 1.0,
            technique: ShadowTechnique::PCF.as_u32(),
            technique_flags: 0,
            _padding1: [0.0; 3],
            technique_params: [0.0, 0.0, 0.0005, 1.0],
            technique_reserved: [0.0; 4],
            cascade_blend_range: 0.1,
            _padding2: [0.0; 27],
        }
    }
}

/// CSM configuration parameters
#[derive(Debug, Clone)]
pub struct CsmConfig {
    /// Number of cascades (3 or 4)
    pub cascade_count: u32,
    /// Shadow map resolution per cascade
    pub shadow_map_size: u32,
    /// Maximum shadow distance
    pub max_shadow_distance: f32,
    /// Cascade split distances (if empty, calculated automatically)
    pub cascade_splits: Vec<f32>,
    /// PCF kernel size
    pub pcf_kernel_size: u32,
    /// Base depth bias
    pub depth_bias: f32,
    /// Slope-scaled bias
    pub slope_bias: f32,
    /// Peter-panning prevention offset
    pub peter_panning_offset: f32,
    /// Enable EVSM filtering
    pub enable_evsm: bool,
    /// EVSM positive exponent
    pub evsm_positive_exp: f32,
    /// EVSM negative exponent
    pub evsm_negative_exp: f32,
    /// Debug visualization mode
    pub debug_mode: u32,
    /// Enable unclipped depth (B17)
    pub enable_unclipped_depth: bool,
    /// Depth clipping distance factor
    pub depth_clip_factor: f32,
    /// Enable cascade stabilization (texel snapping)
    pub stabilize_cascades: bool,
    /// Cascade blend range (0.0 = no blend, 0.1 = 10% blend at boundaries)
    pub cascade_blend_range: f32,
}

impl Default for CsmConfig {
    fn default() -> Self {
        Self {
            cascade_count: 3,
            shadow_map_size: 2048,
            max_shadow_distance: 200.0,
            cascade_splits: vec![],
            pcf_kernel_size: 3,
            depth_bias: 0.005,
            slope_bias: 0.01,
            peter_panning_offset: 0.001,
            enable_evsm: false,
            evsm_positive_exp: crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            evsm_negative_exp: crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            debug_mode: 0,
            enable_unclipped_depth: false,
            depth_clip_factor: 1.0,
            stabilize_cascades: true,
            cascade_blend_range: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba16f_evsm_defaults_match_the_supported_exponent() {
        let config = CsmConfig::default();
        let uniforms = CsmUniforms::default();

        assert_eq!(
            (config.evsm_positive_exp, config.evsm_negative_exp),
            (
                crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
                crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            )
        );
        assert_eq!(
            (uniforms.evsm_positive_exp, uniforms.evsm_negative_exp),
            (
                crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
                crate::shadows::EVSM_MAX_EXPONENT_RGBA16F,
            )
        );
    }
}

/// Statistics for cascade performance monitoring (B17)
#[derive(Debug, Clone)]
pub struct CascadeStatistics {
    /// Total texel area covered by all cascades
    pub total_texel_area: f32,
    /// Total depth range covered by all cascades
    pub depth_range_coverage: f32,
    /// Number of overlapping cascade transitions
    pub cascade_overlaps: u32,
    /// Whether unclipped depth is enabled
    pub unclipped_depth_enabled: bool,
    /// Current depth clip factor
    pub depth_clip_factor: f32,
    /// Effective shadow distance with clipping factor
    pub effective_shadow_distance: f32,
}
