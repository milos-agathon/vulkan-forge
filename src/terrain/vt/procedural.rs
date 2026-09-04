//! Deterministic, per-key page synthesis for procedural TESSELLA stores.
//!
//! Every page's bytes are a pure function of `(seed, family, mip, x, y)`, so
//! no two page keys share a payload and a wrong-tile fetch shows up as a
//! digest mismatch instead of being invisible. The generator is paired with
//! [`MaterializationPlan`]: the store's explicit, recorded statement of which
//! keys exist. Pages outside the plan simply do not exist, and reading one is
//! an error -- never a silent substitution.

use super::store::{PageBytes, PageFormat, PageKey, StoreMetadata};
use crate::core::compressed_textures::{encode_bc5_rg8, encode_bc7_rgba8};
use serde::{Deserialize, Serialize};

/// SplitMix64 increment. Odd, so multiplying a key component by it is a
/// bijection over `u64` and spreads a tiny component across all 64 bits.
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finalizer. Bijective over `u64`, so absorbing a key component
/// through it cannot lose information.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Absorb the full page key one component at a time, so two keys cannot
/// collide through a bit-packing accident the way a single packed integer
/// would allow.
pub fn page_tag(seed: u64, key: PageKey) -> u64 {
    let mut state = mix64(seed ^ GOLDEN_GAMMA);
    for component in [
        u64::from(key.family),
        u64::from(key.mip),
        u64::from(key.x),
        u64::from(key.y),
    ] {
        state = mix64(state ^ component.wrapping_mul(GOLDEN_GAMMA));
    }
    state
}

/// Four flat quadrant colours for a page, in
/// `[top-left, top-right, bottom-left, bottom-right]` order.
///
/// Every channel is forced EVEN: BC7 mode 6 shares one p-bit across all four
/// channels (`core/compressed_textures/bc7.rs`), so an even-valued constant
/// block round-trips bit-exactly. That is what lets a test assert decoded
/// texels exactly rather than within a tolerance band.
pub fn procedural_page_quadrants(seed: u64, key: PageKey) -> [[u8; 4]; 4] {
    let tag = page_tag(seed, key);
    std::array::from_fn(|quadrant| {
        let h = mix64(tag ^ (quadrant as u64 + 1).wrapping_mul(GOLDEN_GAMMA));
        [
            (h as u8) & 0xfe,
            ((h >> 16) as u8) & 0xfe,
            ((h >> 32) as u8) & 0xfe,
            254,
        ]
    })
}

/// Page encoding for a material family: albedo -> BC7 sRGB, normal -> BC5,
/// mask -> BC7 UNORM. Mirrors the per-family expectation enforced by
/// `TerrainMaterialVTRuntime::build_tile_data`.
pub fn procedural_page_format(family: u8) -> Result<PageFormat, String> {
    match family {
        0 => Ok(PageFormat::Bc7Srgb),
        1 => Ok(PageFormat::Bc5Unorm),
        2 => Ok(PageFormat::Bc7Unorm),
        _ => Err(format!("procedural VT family {family} is out of range")),
    }
}

/// Synthesize the one page `key` names. Deterministic across platforms and
/// runs; the only inputs are the metadata's seed/slot size and the key.
pub fn procedural_page(metadata: &StoreMetadata, key: PageKey) -> Result<PageBytes, String> {
    if u32::from(key.family) >= metadata.family_count {
        return Err(format!(
            "procedural VT family {} is out of range",
            key.family
        ));
    }
    let format = procedural_page_format(key.family)?;
    let side = metadata.slot_size();
    let quadrants = procedural_page_quadrants(metadata.procedural_seed, key);
    let data = encode_quadrant_page(format, side, &quadrants)?;
    PageBytes::new(format, side, side, data)
}

fn quadrant_index(half: u32, x: u32, y: u32) -> usize {
    usize::from(y >= half) * 2 + usize::from(x >= half)
}

fn encode_quadrant_page(
    format: PageFormat,
    side: u32,
    quadrants: &[[u8; 4]; 4],
) -> Result<Vec<u8>, String> {
    let half = side / 2;
    if half.is_multiple_of(4) {
        replicate_quadrant_blocks(format, side, half, quadrants)
    } else {
        encode_full_image(format, side, half, quadrants)
    }
}

/// Fast path. Every 4x4 block inside a quadrant is constant and both encoders
/// are strictly per-block, so one encoded block per quadrant colour can be
/// replicated verbatim -- 4 encoder calls per page instead of `(side/4)^2`.
/// `block_replication_matches_full_image_encode` locks this byte-for-byte
/// against the general path.
fn replicate_quadrant_blocks(
    format: PageFormat,
    side: u32,
    half: u32,
    quadrants: &[[u8; 4]; 4],
) -> Result<Vec<u8>, String> {
    let mut units = Vec::with_capacity(quadrants.len());
    for colour in quadrants {
        units.push(match format {
            PageFormat::Bc5Unorm => encode_bc5_rg8(&[colour[0], colour[1]].repeat(16), 4, 4)?,
            PageFormat::Bc7Srgb | PageFormat::Bc7Unorm => {
                encode_bc7_rgba8(&colour.repeat(16), 4, 4)?
            }
            other => return Err(format!("procedural VT pages cannot use {other:?}")),
        });
    }
    let blocks = side / 4;
    let mut data = Vec::with_capacity((blocks * blocks) as usize * 16);
    for block_y in 0..blocks {
        for block_x in 0..blocks {
            data.extend_from_slice(&units[quadrant_index(half, block_x * 4, block_y * 4)]);
        }
    }
    Ok(data)
}

/// General path, used when the quadrant boundary is not BC-block aligned and
/// blocks therefore straddle two colours.
fn encode_full_image(
    format: PageFormat,
    side: u32,
    half: u32,
    quadrants: &[[u8; 4]; 4],
) -> Result<Vec<u8>, String> {
    let texels = side as usize * side as usize;
    match format {
        PageFormat::Bc5Unorm => {
            let mut rg = Vec::with_capacity(texels * 2);
            for y in 0..side {
                for x in 0..side {
                    rg.extend_from_slice(&quadrants[quadrant_index(half, x, y)][0..2]);
                }
            }
            encode_bc5_rg8(&rg, side, side)
        }
        PageFormat::Bc7Srgb | PageFormat::Bc7Unorm => {
            let mut rgba = Vec::with_capacity(texels * 4);
            for y in 0..side {
                for x in 0..side {
                    rgba.extend_from_slice(&quadrants[quadrant_index(half, x, y)]);
                }
            }
            encode_bc7_rgba8(&rgba, side, side)
        }
        other => Err(format!("procedural VT pages cannot use {other:?}")),
    }
}

/// One rectangular `(family, mip)` block of materialized pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationBand {
    pub family: u8,
    pub mip: u8,
    pub origin_x: u32,
    pub origin_y: u32,
    pub pages_x: u32,
    pub pages_y: u32,
}

impl MaterializationBand {
    pub fn page_count(&self) -> u64 {
        u64::from(self.pages_x) * u64::from(self.pages_y)
    }

    pub fn keys(&self) -> impl Iterator<Item = PageKey> + '_ {
        (0..self.pages_y).flat_map(move |dy| {
            (0..self.pages_x).map(move |dx| PageKey {
                family: self.family,
                mip: self.mip,
                x: self.origin_x + dx,
                y: self.origin_y + dy,
            })
        })
    }
}

/// The store's explicit statement of which page keys it materializes.
///
/// Recorded in the store header (bytes 80..92) and in the manifest, so a
/// reader can enumerate the keys that exist without scanning the directory
/// and "mip N is complete" is a checked claim rather than an assumption.
///
/// A store may hold pages beyond the plan; it may never hold fewer, and both
/// `write_packed_store` and `MmapPageStore::open` reject a file that does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializationPlan {
    /// Every page at `mip >= coarse_min_mip` exists, for every family. Equal
    /// to `StoreMetadata::mip_count()` when the store makes no completeness
    /// claim at all (see [`MaterializationPlan::none`]).
    pub coarse_min_mip: u32,
    /// Finer mips `0..=detail_max_mip` exist only inside the detail window.
    pub detail_max_mip: u32,
    /// Side length in pages of the centred detail window; 0 disables it.
    pub detail_window_pages: u32,
}

impl MaterializationPlan {
    /// Every page of the whole pyramid.
    pub fn full_pyramid() -> Self {
        Self {
            coarse_min_mip: 0,
            detail_max_mip: 0,
            detail_window_pages: 0,
        }
    }

    /// No completeness claim: the store holds exactly the pages handed to the
    /// packer and a reader must not assume any mip is complete.
    pub fn none(metadata: &StoreMetadata) -> Self {
        Self {
            coarse_min_mip: metadata.mip_count(),
            detail_max_mip: 0,
            detail_window_pages: 0,
        }
    }

    pub fn validate(&self, metadata: &StoreMetadata) -> Result<(), String> {
        let mip_count = metadata.mip_count();
        if self.coarse_min_mip > mip_count {
            return Err(format!(
                "materialization plan coarse_min_mip {} exceeds the store's {mip_count} mip levels",
                self.coarse_min_mip
            ));
        }
        if self.detail_window_pages > 0 {
            if self.coarse_min_mip == 0 {
                return Err(
                    "materialization plan declares a detail window but coarse_min_mip 0 already materializes every page"
                        .to_string(),
                );
            }
            if self.detail_max_mip >= self.coarse_min_mip {
                return Err(format!(
                    "materialization plan detail_max_mip {} must be below coarse_min_mip {}",
                    self.detail_max_mip, self.coarse_min_mip
                ));
            }
        }
        Ok(())
    }

    /// The declared bands, grouped `(family, mip)` and ordered finest-first
    /// within each family.
    pub fn bands(&self, metadata: &StoreMetadata) -> Vec<MaterializationBand> {
        let mip_count = metadata.mip_count();
        let coarse_min_mip = self.coarse_min_mip.min(mip_count);
        let mut bands = Vec::new();
        for family in 0..metadata.family_count.min(u32::from(u8::MAX)) {
            let family = family as u8;
            if self.detail_window_pages > 0 && coarse_min_mip > 0 {
                let detail_max_mip = self.detail_max_mip.min(coarse_min_mip - 1);
                for mip in 0..=detail_max_mip {
                    let (grid_x, grid_y) = metadata.pages_at_mip(mip);
                    let pages_x = self.detail_window_pages.min(grid_x);
                    let pages_y = self.detail_window_pages.min(grid_y);
                    bands.push(MaterializationBand {
                        family,
                        mip: mip as u8,
                        origin_x: Self::window_origin(grid_x, pages_x, self.detail_window_pages),
                        origin_y: Self::window_origin(grid_y, pages_y, self.detail_window_pages),
                        pages_x,
                        pages_y,
                    });
                }
            }
            for mip in coarse_min_mip..mip_count {
                let (pages_x, pages_y) = metadata.pages_at_mip(mip);
                bands.push(MaterializationBand {
                    family,
                    mip: mip as u8,
                    origin_x: 0,
                    origin_y: 0,
                    pages_x,
                    pages_y,
                });
            }
        }
        bands
    }

    /// Every declared page key, in band order.
    pub fn keys(&self, metadata: &StoreMetadata) -> Vec<PageKey> {
        self.bands(metadata)
            .into_iter()
            .flat_map(|band| band.keys().collect::<Vec<_>>())
            .collect()
    }

    pub fn page_count(&self, metadata: &StoreMetadata) -> u64 {
        self.bands(metadata)
            .iter()
            .map(MaterializationBand::page_count)
            .sum()
    }

    fn window_origin(grid: u32, window: u32, requested: u32) -> u32 {
        (grid / 2)
            .saturating_sub(requested / 2)
            .min(grid.saturating_sub(window))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(tile_size: u32, tile_border: u32) -> StoreMetadata {
        StoreMetadata {
            virtual_width: 1 << 18,
            virtual_height: 1 << 18,
            tile_size,
            tile_border,
            family_count: 3,
            procedural: true,
            procedural_seed: 19,
        }
    }

    #[test]
    fn block_replication_matches_full_image_encode() {
        // slot 128 and 136 take the replication fast path (half is BC-block
        // aligned); slot 12 and 20 take the general path. All four must agree
        // with a full-image encode byte for byte.
        for (tile_size, tile_border) in [(128, 0), (128, 4), (8, 2), (16, 2)] {
            let metadata = metadata(tile_size, tile_border);
            let side = metadata.slot_size();
            let half = side / 2;
            for family in 0..3u8 {
                let format = procedural_page_format(family).unwrap();
                let key = PageKey {
                    family,
                    mip: 3,
                    x: 11,
                    y: 7,
                };
                let quadrants = procedural_page_quadrants(metadata.procedural_seed, key);
                let reference = encode_full_image(format, side, half, &quadrants).unwrap();
                let actual = encode_quadrant_page(format, side, &quadrants).unwrap();
                assert_eq!(
                    actual, reference,
                    "slot {side} family {family} replication drifted from the full encode"
                );
                assert_eq!(actual.len() as u64, format.block_bytes(side, side));
            }
        }
    }

    #[test]
    fn page_content_depends_on_every_key_component() {
        let metadata = metadata(128, 0);
        let base = PageKey {
            family: 0,
            mip: 4,
            x: 9,
            y: 13,
        };
        let variants = [
            PageKey { family: 2, ..base },
            PageKey { mip: 5, ..base },
            PageKey { x: 10, ..base },
            PageKey { y: 14, ..base },
        ];
        let reference = procedural_page(&metadata, base).unwrap().sha256;
        for variant in variants {
            let digest = procedural_page(&metadata, variant).unwrap().sha256;
            assert_ne!(
                digest, reference,
                "{variant:?} produced the same payload as {base:?}"
            );
        }
    }

    #[test]
    fn plan_bands_cover_the_declared_pages_exactly() {
        use std::collections::HashSet;

        let metadata = metadata(128, 0);
        assert_eq!(metadata.mip_count(), 12);
        // 3 families x (mips 6..11 complete = 1365 pages, plus 6 detail
        // windows). The CLI default window is 8; the committed GPU gate packs
        // 2 so the whole coarse working set still fits the atlas slot count.
        for (detail_window_pages, expected) in [(8u32, 5247u64), (2, 4167)] {
            let plan = MaterializationPlan {
                coarse_min_mip: 6,
                detail_max_mip: 5,
                detail_window_pages,
            };
            plan.validate(&metadata).unwrap();
            assert_eq!(plan.page_count(&metadata), expected);
            let keys = plan.keys(&metadata);
            assert_eq!(keys.len() as u64, expected);
            let declared = keys.iter().copied().collect::<HashSet<_>>();
            assert_eq!(
                declared.len() as u64,
                expected,
                "the plan enumerated a duplicate key"
            );
            for key in &keys {
                let (pages_x, pages_y) = metadata.pages_at_mip(u32::from(key.mip));
                assert!(key.x < pages_x && key.y < pages_y, "{key:?} is out of grid");
            }
            // Every declared key's coarser ancestor must also be declared:
            // the renderer walks ancestors when it queues a request, and a
            // gap there would name a page that does not exist.
            for key in &keys {
                if u32::from(key.mip) + 1 >= metadata.mip_count() {
                    continue;
                }
                let ancestor = PageKey {
                    family: key.family,
                    mip: key.mip + 1,
                    x: key.x / 2,
                    y: key.y / 2,
                };
                assert!(
                    declared.contains(&ancestor),
                    "{key:?} has no materialized ancestor {ancestor:?}"
                );
            }
        }
    }

    #[test]
    fn plan_validation_rejects_an_unreachable_detail_band() {
        let metadata = metadata(128, 0);
        assert!(MaterializationPlan {
            coarse_min_mip: 4,
            detail_max_mip: 4,
            detail_window_pages: 8,
        }
        .validate(&metadata)
        .is_err());
        assert!(MaterializationPlan {
            coarse_min_mip: 99,
            detail_max_mip: 0,
            detail_window_pages: 0,
        }
        .validate(&metadata)
        .is_err());
        MaterializationPlan::full_pyramid()
            .validate(&metadata)
            .unwrap();
        MaterializationPlan::none(&metadata)
            .validate(&metadata)
            .unwrap();
    }
}
