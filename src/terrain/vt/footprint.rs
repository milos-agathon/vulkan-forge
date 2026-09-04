//! Device-independent virtual-texture atlas footprint accounting.

/// Capacity of the three material-family atlases, ordered albedo, normal,
/// mask. Values describe physical allocations, not resident upload traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MaterialAtlasFootprints {
    pub uncompressed: [u64; 3],
    pub device_local: [u64; 3],
}

/// Albedo and mask are RGBA8 before BC7 (4 B/texel -> 1 B/texel); normal is
/// RG8 before BC5 (2 B/texel -> 1 B/texel).
///
/// The raw compatibility path has one dynamically shared RGBA8 texture, so it
/// has no honest static per-family device-local attribution. Its aggregate is
/// reported separately by the renderer's `atlas_device_local_bytes` metric.
pub(crate) fn compressed_material_atlas_footprints(atlas_texels: u64) -> MaterialAtlasFootprints {
    MaterialAtlasFootprints {
        uncompressed: [atlas_texels * 4, atlas_texels * 2, atlas_texels * 4],
        device_local: [atlas_texels; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindless_bc_atlas_footprints_are_exact_per_family() {
        let atlas_texels = 4096_u64 * 4096;
        let compressed = compressed_material_atlas_footprints(atlas_texels);
        assert_eq!(
            compressed.uncompressed,
            [atlas_texels * 4, atlas_texels * 2, atlas_texels * 4]
        );
        assert_eq!(compressed.device_local, [atlas_texels; 3]);
        assert_eq!(compressed.uncompressed[0] / compressed.device_local[0], 4);
        assert_eq!(compressed.uncompressed[1] / compressed.device_local[1], 2);
        assert_eq!(compressed.uncompressed[2] / compressed.device_local[2], 4);
        assert_eq!(
            compressed.uncompressed.iter().sum::<u64>(),
            atlas_texels * 10
        );
        assert_eq!(
            compressed.device_local.iter().sum::<u64>(),
            atlas_texels * 3
        );
    }
}
