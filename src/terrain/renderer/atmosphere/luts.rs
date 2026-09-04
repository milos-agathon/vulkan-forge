use anyhow::{anyhow, Result};

use crate::core::atmosphere::{
    tracked_lut_upload_bytes, AtmosphereLutHandle, AtmosphereLuts, LutData,
    ACCUMULATED_SCATTERING_LUT_SEMANTICS,
};
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};

pub(in crate::terrain::renderer) const AETHER_LUT_CACHE_CAPACITY: usize = 16;

pub(in crate::terrain::renderer) struct AtmosphereGpuLuts {
    pub(super) deterministic_sha256: [u8; 32],
    pub(super) dimensions: crate::core::atmosphere::LutDimensions,
    pub(super) transmittance_view: wgpu::TextureView,
    pub(super) scattering_view: wgpu::TextureView,
    pub(super) aerial_view: wgpu::TextureView,
    _transmittance_texture: TrackedTexture,
    _scattering_texture: TrackedTexture,
    _aerial_texture: TrackedTexture,
    byte_size: u64,
}

impl AtmosphereGpuLuts {
    pub(super) fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: &AtmosphereLutHandle,
    ) -> Result<Self> {
        let luts = handle.luts();
        validate_luts(luts)?;

        let (transmittance_texture, transmittance_view) = upload_lut(
            device,
            queue,
            "terrain.aether.transmittance",
            &luts.transmittance,
            wgpu::TextureDimension::D2,
            wgpu::TextureViewDimension::D2,
        )?;
        let (scattering_texture, scattering_view) = upload_lut(
            device,
            queue,
            "terrain.aether.accumulated_scattering",
            &luts.multiple_scattering,
            wgpu::TextureDimension::D3,
            wgpu::TextureViewDimension::D3,
        )?;
        let (aerial_texture, aerial_view) = upload_lut(
            device,
            queue,
            "terrain.aether.aerial",
            &luts.aerial_perspective,
            wgpu::TextureDimension::D3,
            wgpu::TextureViewDimension::D3,
        )?;
        let byte_size = luts.transmittance.byte_size()
            + luts.multiple_scattering.byte_size()
            + luts.aerial_perspective.byte_size();

        Ok(Self {
            deterministic_sha256: handle.deterministic_sha256(),
            dimensions: luts.metadata.dimensions,
            transmittance_view,
            scattering_view,
            aerial_view,
            _transmittance_texture: transmittance_texture,
            _scattering_texture: scattering_texture,
            _aerial_texture: aerial_texture,
            byte_size,
        })
    }

    pub(super) fn byte_size(&self) -> u64 {
        self.byte_size
    }

    pub(super) fn scattering_view(&self) -> wgpu::TextureView {
        self._scattering_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain.aether.accumulated_scattering.material-view"),
                dimension: Some(wgpu::TextureViewDimension::D3),
                ..Default::default()
            })
    }
}

fn validate_luts(luts: &AtmosphereLuts) -> Result<()> {
    if luts.metadata.storage_format != "rgba16float" {
        return Err(anyhow!(
            "AETHER runtime requires rgba16float LUTs, got {}",
            luts.metadata.storage_format
        ));
    }
    if luts.metadata.scattering_lut_semantics != ACCUMULATED_SCATTERING_LUT_SEMANTICS {
        return Err(anyhow!(
            "AETHER runtime requires accumulated single+higher-order scattering, got {}",
            luts.metadata.scattering_lut_semantics
        ));
    }
    let dims = luts.metadata.dimensions;
    let expected_transmittance = [dims.transmittance_mu, dims.transmittance_height, 1];
    let expected_scattering = [
        dims.scattering_mu_view,
        dims.scattering_mu_sun,
        dims.scattering_height
            .checked_mul(dims.scattering_nu)
            .ok_or_else(|| anyhow!("AETHER scattering depth overflow"))?,
    ];
    let expected_aerial = [
        dims.aerial_distance,
        dims.aerial_mu_view,
        dims.aerial_height,
    ];
    for (name, actual, expected) in [
        (
            "transmittance",
            luts.transmittance.dimensions,
            expected_transmittance,
        ),
        (
            "accumulated_scattering",
            luts.multiple_scattering.dimensions,
            expected_scattering,
        ),
        (
            "aerial",
            luts.aerial_perspective.dimensions,
            expected_aerial,
        ),
    ] {
        if actual != expected {
            return Err(anyhow!(
                "AETHER {name} LUT dimensions {actual:?} do not match metadata {expected:?}"
            ));
        }
    }
    Ok(())
}

fn upload_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    data: &LutData,
    dimension: wgpu::TextureDimension,
    view_dimension: wgpu::TextureViewDimension,
) -> Result<(TrackedTexture, wgpu::TextureView)> {
    let [width, height, depth] = data.dimensions;
    if width == 0 || height == 0 || depth == 0 {
        return Err(anyhow!("AETHER {label} LUT has an empty axis"));
    }
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|value| value.checked_mul(u64::from(depth)))
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| anyhow!("AETHER {label} LUT byte size overflow"))?;
    let bytes = tracked_lut_upload_bytes(data, "terrain.aether.lut-upload-staging")?;
    if bytes.as_slice().len() as u64 != expected {
        return Err(anyhow!(
            "AETHER {label} LUT payload has {} bytes, expected {expected}",
            bytes.as_slice().len()
        ));
    }

    let texture = tracked_create_texture(
        device,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes.as_slice(),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 8),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(view_dimension),
        ..Default::default()
    });
    Ok((texture, view))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_lut_metadata_matches_gpu_packing() {
        let handle = AtmosphereLutHandle::load_shipped(Default::default()).unwrap();
        let luts = handle.luts();
        validate_luts(luts).unwrap();
        assert_eq!(
            luts.multiple_scattering.dimensions[2],
            luts.metadata.dimensions.scattering_height * luts.metadata.dimensions.scattering_nu
        );
    }
}
