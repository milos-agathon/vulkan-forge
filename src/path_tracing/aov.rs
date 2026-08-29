//! AOV (Arbitrary Output Variables) support for path tracing
//!
//! This module defines the AOV types, formats, and utilities for rendering
//! multiple output channels from the GPU path tracer.
//! RELEVANT FILES: src/shaders/pt_kernel.wgsl, src/path_tracing/compute.rs, python/forge3d/path_tracing.py

use crate::core::error::RenderResult;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use std::collections::HashMap;
use wgpu::{
    Device, Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

/// AOV (Arbitrary Output Variable) types supported by the path tracer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AovKind {
    /// Material albedo/base color
    Albedo,
    /// Surface normal in world space
    Normal,
    /// Linear depth from camera
    Depth,
    /// Direct illumination
    Direct,
    /// Indirect/ambient illumination
    Indirect,
    /// Emissive contribution
    Emission,
    /// Visibility mask (1 = hit geometry, 0 = sky)
    Visibility,
    /// RGB medium transmittance along the camera segment.
    Transmittance,
    /// RGB medium in-scattered radiance.
    InScatter,
    /// RGB medium transmittance applied to terrain direct lighting.
    CloudShadow,
    /// RGB optical depth along the camera segment.
    OpticalDepth,
}

impl AovKind {
    /// Get all available AOV types
    pub fn all() -> &'static [AovKind] {
        &[
            AovKind::Albedo,
            AovKind::Normal,
            AovKind::Depth,
            AovKind::Direct,
            AovKind::Indirect,
            AovKind::Emission,
            AovKind::Visibility,
            AovKind::Transmittance,
            AovKind::InScatter,
            AovKind::CloudShadow,
            AovKind::OpticalDepth,
        ]
    }

    /// Get the canonical string name for this AOV
    pub fn name(self) -> &'static str {
        match self {
            AovKind::Albedo => "albedo",
            AovKind::Normal => "normal",
            AovKind::Depth => "depth",
            AovKind::Direct => "direct",
            AovKind::Indirect => "indirect",
            AovKind::Emission => "emission",
            AovKind::Visibility => "visibility",
            AovKind::Transmittance => "transmittance",
            AovKind::InScatter => "in_scatter",
            AovKind::CloudShadow => "cloud_shadow",
            AovKind::OpticalDepth => "optical_depth",
        }
    }

    /// Parse AOV kind from string name
    pub fn from_name(name: &str) -> Option<AovKind> {
        match name.to_lowercase().as_str() {
            "albedo" => Some(AovKind::Albedo),
            "normal" => Some(AovKind::Normal),
            "depth" => Some(AovKind::Depth),
            "direct" => Some(AovKind::Direct),
            "indirect" => Some(AovKind::Indirect),
            "emission" => Some(AovKind::Emission),
            "visibility" => Some(AovKind::Visibility),
            "transmittance" => Some(AovKind::Transmittance),
            "in_scatter" => Some(AovKind::InScatter),
            "cloud_shadow" => Some(AovKind::CloudShadow),
            "optical_depth" => Some(AovKind::OpticalDepth),
            _ => None,
        }
    }

    /// Get the GPU texture format for this AOV
    pub fn texture_format(self) -> TextureFormat {
        match self {
            AovKind::Albedo
            | AovKind::Normal
            | AovKind::Direct
            | AovKind::Indirect
            | AovKind::Emission => TextureFormat::Rgba16Float,
            AovKind::Transmittance
            | AovKind::InScatter
            | AovKind::CloudShadow
            | AovKind::OpticalDepth => TextureFormat::Rgba16Float,
            AovKind::Depth => TextureFormat::R32Float,
            AovKind::Visibility => TextureFormat::Rgba8Unorm,
        }
    }

    /// Get the bind group binding index for this AOV in the shader
    pub fn binding_index(self) -> u32 {
        match self {
            AovKind::Albedo => 0,
            AovKind::Normal => 1,
            AovKind::Depth => 2,
            AovKind::Direct => 3,
            AovKind::Indirect => 4,
            AovKind::Emission => 5,
            AovKind::Visibility => 6,
            AovKind::Transmittance => 7,
            AovKind::InScatter => 8,
            AovKind::CloudShadow => 9,
            AovKind::OpticalDepth => 10,
        }
    }

    /// Get the number of color channels for this AOV
    pub fn channel_count(self) -> u32 {
        match self {
            AovKind::Albedo
            | AovKind::Normal
            | AovKind::Direct
            | AovKind::Indirect
            | AovKind::Emission => 3, // RGB
            AovKind::Transmittance
            | AovKind::InScatter
            | AovKind::CloudShadow
            | AovKind::OpticalDepth => 3,
            AovKind::Depth | AovKind::Visibility => 1, // Single channel
        }
    }

    /// Calculate memory usage for this AOV at given resolution
    pub fn memory_usage_bytes(self, width: u32, height: u32) -> u64 {
        let pixel_count = width as u64 * height as u64;
        match self {
            AovKind::Albedo
            | AovKind::Normal
            | AovKind::Direct
            | AovKind::Indirect
            | AovKind::Emission => {
                pixel_count * 8 // rgba16float = 4 channels * 2 bytes
            }
            AovKind::Transmittance
            | AovKind::InScatter
            | AovKind::CloudShadow
            | AovKind::OpticalDepth => pixel_count * 8,
            AovKind::Depth => {
                pixel_count * 4 // r32float = 1 channel * 4 bytes
            }
            AovKind::Visibility => {
                pixel_count * 4 // rgba8unorm = 4 channels * 1 byte
            }
        }
    }
}

/// AOV descriptor with enabled state
#[derive(Debug, Clone)]
pub struct AovDesc {
    pub kind: AovKind,
    pub enabled: bool,
}

impl AovDesc {
    pub fn new(kind: AovKind, enabled: bool) -> Self {
        Self { kind, enabled }
    }
}

/// GPU textures and resources for AOV rendering
#[derive(Debug)]
pub struct AovFrames {
    /// Map from AOV kind to GPU texture
    pub textures: HashMap<AovKind, TrackedTexture>,
    /// Enabled AOV flags bitmask
    pub enabled_mask: u32,
    /// Dimensions
    pub width: u32,
    pub height: u32,
}

impl AovFrames {
    /// Create AOV textures for the specified AOVs
    pub fn new(device: &Device, width: u32, height: u32, aovs: &[AovKind]) -> RenderResult<Self> {
        let mut textures = HashMap::new();
        let mut enabled_mask = 0u32;

        let extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        for &aov_kind in aovs {
            let texture = tracked_create_texture(
                device,
                &TextureDescriptor {
                    label: Some(&format!("AOV_{}", aov_kind.name())),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: aov_kind.texture_format(),
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )?;

            textures.insert(aov_kind, texture);
            enabled_mask |= 1u32 << aov_kind.binding_index();
        }

        Ok(Self {
            textures,
            enabled_mask,
            width,
            height,
        })
    }

    /// Get texture for specific AOV kind
    pub fn get_texture(&self, kind: AovKind) -> Option<&Texture> {
        self.textures.get(&kind).map(|t| &**t)
    }

    /// Check if AOV is enabled
    pub fn is_enabled(&self, kind: AovKind) -> bool {
        (self.enabled_mask & (1u32 << kind.binding_index())) != 0
    }

    /// Calculate total memory usage for all enabled AOVs
    pub fn total_memory_usage(&self) -> u64 {
        self.textures
            .keys()
            .map(|&kind| kind.memory_usage_bytes(self.width, self.height))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aov_kind_name_parsing() {
        assert_eq!(AovKind::from_name("albedo"), Some(AovKind::Albedo));
        assert_eq!(AovKind::from_name("NORMAL"), Some(AovKind::Normal));
        assert_eq!(AovKind::from_name("Depth"), Some(AovKind::Depth));
        assert_eq!(AovKind::from_name("invalid"), None);
    }

    #[test]
    fn test_aov_properties() {
        assert_eq!(AovKind::Albedo.name(), "albedo");
        assert_eq!(AovKind::Albedo.texture_format(), TextureFormat::Rgba16Float);
        assert_eq!(AovKind::Depth.texture_format(), TextureFormat::R32Float);
        assert_eq!(
            AovKind::Visibility.texture_format(),
            TextureFormat::Rgba8Unorm
        );

        assert!(AovKind::all()
            .iter()
            .copied()
            .map(AovKind::binding_index)
            .eq(0..AovKind::all().len() as u32));

        assert_eq!(AovKind::Albedo.channel_count(), 3);
        assert_eq!(AovKind::Depth.channel_count(), 1);
    }

    #[test]
    fn test_memory_calculations() {
        // 100x100 texture
        assert_eq!(AovKind::Albedo.memory_usage_bytes(100, 100), 80000); // 10k pixels * 8 bytes (rgba16f)
        assert_eq!(AovKind::Depth.memory_usage_bytes(100, 100), 40000); // 10k pixels * 4 bytes (r32f)
        assert_eq!(AovKind::Visibility.memory_usage_bytes(100, 100), 40000); // 10k pixels * 4 bytes (rgba8unorm)
    }
}
