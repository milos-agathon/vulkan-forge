//! AOV file I/O operations (EXR and PNG writers)
//!
//! This module provides writers for AOV data to standard image formats.
//! EXR is used for HDR float data, PNG for 8-bit data.
//! Feature-gated behind "images" feature flag.
//! RELEVANT FILES: src/path_tracing/aov.rs, python/forge3d/path_tracing.py

use crate::path_tracing::aov::AovKind;
#[cfg(feature = "images")]
use crate::util::exr_write;
use anyhow::{Context, Result};
use std::path::Path;

/// Represents AOV data for writing to files
#[derive(Debug, Clone)]
pub struct AovData {
    pub kind: AovKind,
    pub width: u32,
    pub height: u32,
    pub data: AovDataType,
}

/// Type-erased AOV data
#[derive(Debug, Clone)]
pub enum AovDataType {
    /// RGB or RGBA float data (normalized)
    Float32(Vec<f32>),
    /// Single channel float data
    Float32Single(Vec<f32>),
    /// Single channel 8-bit data
    Uint8Single(Vec<u8>),
}

impl AovData {
    /// Create AOV data from RGBA float buffer
    pub fn from_rgba_f32(kind: AovKind, width: u32, height: u32, data: Vec<f32>) -> Result<Self> {
        let expected_len = (width * height * 4) as usize;
        anyhow::ensure!(
            data.len() == expected_len,
            "Expected {} floats for {}x{} RGBA data, got {}",
            expected_len,
            width,
            height,
            data.len()
        );

        Ok(Self {
            kind,
            width,
            height,
            data: AovDataType::Float32(data),
        })
    }

    /// Create AOV data from single channel float buffer
    pub fn from_r_f32(kind: AovKind, width: u32, height: u32, data: Vec<f32>) -> Result<Self> {
        let expected_len = (width * height) as usize;
        anyhow::ensure!(
            data.len() == expected_len,
            "Expected {} floats for {}x{} single channel data, got {}",
            expected_len,
            width,
            height,
            data.len()
        );

        Ok(Self {
            kind,
            width,
            height,
            data: AovDataType::Float32Single(data),
        })
    }

    /// Create AOV data from single channel uint8 buffer
    pub fn from_r_u8(kind: AovKind, width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected_len = (width * height) as usize;
        anyhow::ensure!(
            data.len() == expected_len,
            "Expected {} bytes for {}x{} single channel data, got {}",
            expected_len,
            width,
            height,
            data.len()
        );

        Ok(Self {
            kind,
            width,
            height,
            data: AovDataType::Uint8Single(data),
        })
    }
}

/// AOV file writer with format selection based on data type
pub struct AovWriter;

impl AovWriter {
    /// Write AOV data to appropriate file format
    ///
    /// - HDR data (albedo, normal, depth, direct, indirect, emission) -> EXR
    /// - LDR data (visibility) -> PNG
    pub fn write_aov<P: AsRef<Path>>(aov_data: &AovData, path: P) -> Result<()> {
        #[cfg(feature = "images")]
        let path = path.as_ref();
        #[cfg(not(feature = "images"))]
        let _ = path;

        match &aov_data.data {
            AovDataType::Float32(_) | AovDataType::Float32Single(_) => {
                // Write as EXR for HDR data
                #[cfg(feature = "images")]
                {
                    Self::write_exr(aov_data, path)
                }
                #[cfg(not(feature = "images"))]
                {
                    anyhow::bail!("EXR writing requires 'images' feature to be enabled")
                }
            }
            AovDataType::Uint8Single(_) => {
                // Write as PNG for LDR data
                #[cfg(feature = "images")]
                {
                    Self::write_png(aov_data, path)
                }
                #[cfg(not(feature = "images"))]
                {
                    anyhow::bail!("PNG writing requires 'images' feature to be enabled")
                }
            }
        }
    }

    /// Write multiple AOVs to a directory with consistent naming
    pub fn write_aovs<P: AsRef<Path>>(
        aov_data_list: &[AovData],
        output_dir: P,
        basename: &str,
    ) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir).with_context(|| {
            format!(
                "Failed to create output directory: {}",
                output_dir.display()
            )
        })?;

        for aov_data in aov_data_list {
            let extension = match &aov_data.data {
                AovDataType::Float32(_) | AovDataType::Float32Single(_) => "exr",
                AovDataType::Uint8Single(_) => "png",
            };

            let filename = format!("{}_aov-{}.{}", basename, aov_data.kind.name(), extension);
            let filepath = output_dir.join(filename);

            Self::write_aov(aov_data, &filepath).with_context(|| {
                format!(
                    "Failed to write AOV {} to {}",
                    aov_data.kind.name(),
                    filepath.display()
                )
            })?;
        }

        Ok(())
    }

    #[cfg(feature = "images")]
    fn write_exr(aov_data: &AovData, path: &Path) -> Result<()> {
        match &aov_data.data {
            AovDataType::Float32(data) => exr_write::write_exr_rgb_f32_from_rgba(
                path,
                aov_data.width,
                aov_data.height,
                data,
                aov_data.kind.name(),
            ),
            AovDataType::Float32Single(data) => exr_write::write_exr_scalar_f32(
                path,
                aov_data.width,
                aov_data.height,
                data,
                aov_data.kind.name(),
            ),
            AovDataType::Uint8Single(_) => {
                anyhow::bail!("EXR writer does not support uint8 AOV data")
            }
        }
    }

    #[cfg(feature = "images")]
    fn write_png(aov_data: &AovData, path: &Path) -> Result<()> {
        match &aov_data.data {
            AovDataType::Uint8Single(data) => {
                // Use the 'image' crate to write PNG
                use image::{ImageBuffer, Luma};

                let img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                    aov_data.width,
                    aov_data.height,
                    data.clone(),
                )
                .context("Failed to create image buffer from data")?;

                img.save(path)
                    .with_context(|| format!("Failed to save PNG to {}", path.display()))?;

                Ok(())
            }
            _ => {
                anyhow::bail!("PNG writer only supports uint8 single channel data")
            }
        }
    }
}

/// Utility functions for AOV file I/O
pub mod utils {
    use super::*;

    /// Generate standard filename for AOV
    pub fn aov_filename(basename: &str, aov_kind: AovKind, use_exr: bool) -> String {
        let extension = if use_exr { "exr" } else { "png" };
        format!("{}_aov-{}.{}", basename, aov_kind.name(), extension)
    }

    /// Check if AOV should be written as EXR (HDR) or PNG (LDR)
    pub fn is_hdr_aov(kind: AovKind) -> bool {
        kind != AovKind::Visibility
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aov_data_creation() {
        let width = 4;
        let height = 4;
        let pixel_count = (width * height) as usize;

        // Test RGBA float data
        let rgba_data = vec![0.5f32; pixel_count * 4];
        let aov = AovData::from_rgba_f32(AovKind::Albedo, width, height, rgba_data).unwrap();
        assert_eq!(aov.width, width);
        assert_eq!(aov.height, height);

        // Test single channel float data
        let depth_data = vec![1.0f32; pixel_count];
        let aov = AovData::from_r_f32(AovKind::Depth, width, height, depth_data).unwrap();
        assert_eq!(aov.width, width);

        // Test single channel uint8 data
        let vis_data = vec![255u8; pixel_count];
        let aov = AovData::from_r_u8(AovKind::Visibility, width, height, vis_data).unwrap();
        assert_eq!(aov.width, width);
    }

    #[test]
    fn test_filename_generation() {
        assert_eq!(
            utils::aov_filename("frame001", AovKind::Albedo, true),
            "frame001_aov-albedo.exr"
        );
        assert_eq!(
            utils::aov_filename("test", AovKind::Visibility, false),
            "test_aov-visibility.png"
        );
    }

    #[test]
    fn test_hdr_classification() {
        assert_eq!(
            AovKind::all()
                .iter()
                .copied()
                .filter(|kind| !utils::is_hdr_aov(*kind))
                .collect::<Vec<_>>(),
            vec![AovKind::Visibility]
        );
    }
}
