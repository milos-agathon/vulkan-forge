//! BOP-P2-02: runtime height-tile streaming for clipmap terrain.
//!
//! Connects the previously orphaned streaming stack — `ClipmapStreamer`
//! (camera-driven tile demand), `AsyncTileLoader` (threaded tile reads with
//! dedup/coalescing/backpressure), and `HeightMosaic` (fixed-LOD height
//! atlas) — to `TerrainRenderer`'s height texture binding.
//!
//! The mosaic runs in `fixed_lod` mode: slot (x, y) of the atlas is tile
//! (x, y) of the region, so the atlas *is* the region heightmap at the
//! streaming LOD and binds directly as `height_tex` with plain [0,1] UVs.
//! At enable time every tile is prefilled from a coarse read (the "coarse
//! ancestor" fallback), so frames rendered while fine tiles are still in
//! flight show coarse terrain instead of holes. GPU-resident height memory
//! is bounded by the mosaic dimensions regardless of source-region size.
//!
//! Coarse CPU passes (height AO, sun visibility, shadows) intentionally keep
//! using the overview heightmap passed to each render call; streaming
//! refines the geometry/shading height source only.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::*;
use crate::terrain::clipmap::{ClipmapConfig, ClipmapStreamer};
use crate::terrain::lod::LodConfig;
use crate::terrain::page_table::{AsyncTileLoader, CoalescePolicy, HeightReader};
use crate::terrain::stream::{HeightMosaic, MosaicConfig};
use crate::terrain::tiling::{QuadTreeNode, TileBounds, TileId};
use crate::terrain::vt_family_residency::{FamilyResidencyTracker, TileKey as VtTileKey};
use glam::{Mat4, Vec2, Vec3};

/// Slices height tiles out of a caller-provided DEM by bilinear sampling.
/// Worker threads of `AsyncTileLoader` invoke `read` off the render thread.
pub(in crate::terrain::renderer) struct DemSliceHeightReader {
    dem: Vec<f32>,
    width: usize,
    height: usize,
}

impl DemSliceHeightReader {
    pub(in crate::terrain::renderer) fn new(dem: Vec<f32>, width: usize, height: usize) -> Self {
        Self { dem, width, height }
    }

    fn sample_bilinear(&self, nx: f32, ny: f32) -> f32 {
        let fx = (nx.clamp(0.0, 1.0)) * (self.width - 1) as f32;
        let fy = (ny.clamp(0.0, 1.0)) * (self.height - 1) as f32;
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let h00 = self.dem[y0 * self.width + x0];
        let h10 = self.dem[y0 * self.width + x1];
        let h01 = self.dem[y1 * self.width + x0];
        let h11 = self.dem[y1 * self.width + x1];
        let top = h00 * (1.0 - tx) + h10 * tx;
        let bottom = h01 * (1.0 - tx) + h11 * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

impl HeightReader for DemSliceHeightReader {
    fn read(
        &self,
        root_bounds: &TileBounds,
        tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let bounds = QuadTreeNode::calculate_bounds(root_bounds, tile_id, tile_size);
        let root_size = root_bounds.max - root_bounds.min;
        let mut out = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let u = x as f32 / (width - 1).max(1) as f32;
                let v = y as f32 / (height - 1).max(1) as f32;
                let world = bounds.min + Vec2::new(u, v) * (bounds.max - bounds.min);
                let nx = (world.x - root_bounds.min.x) / root_size.x.max(1e-6);
                let ny = (world.y - root_bounds.min.y) / root_size.y.max(1e-6);
                out.push(self.sample_bilinear(nx, ny));
            }
        }
        out
    }
}

/// HeightReader compatibility adapter over TESSELLA's store contract.
pub(in crate::terrain::renderer) struct StoreHeightReader {
    store: Arc<dyn crate::terrain::vt::VirtualTextureStore>,
}

impl StoreHeightReader {
    pub(in crate::terrain::renderer) fn new(
        store: Arc<dyn crate::terrain::vt::VirtualTextureStore>,
    ) -> Self {
        Self { store }
    }
}

impl HeightReader for StoreHeightReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let page = match self.store.page(crate::terrain::vt::PageKey {
            family: crate::terrain::vt::HEIGHT_FAMILY,
            mip: tile_id.lod as u8,
            x: tile_id.x,
            y: tile_id.y,
        }) {
            Ok(page) => page,
            Err(error) => {
                crate::core::degradation::record_degradation(
                    "streaming_failure",
                    "terrain_cog_page",
                    &error,
                );
                return vec![0.0; (width * height) as usize];
            }
        };
        if page.format != crate::terrain::vt::PageFormat::R32Float {
            crate::core::degradation::record_degradation(
                "streaming_failure",
                "terrain_cog_page_format",
                "COG height store returned a non-R32Float page",
            );
            return vec![0.0; (width * height) as usize];
        }
        let source = bytemuck::cast_slice::<u8, f32>(&page.data);
        if page.width == width && page.height == height {
            source.to_vec()
        } else if page.width == page.height && width == height {
            upsample_bilinear(source, page.width, width)
        } else {
            crate::core::degradation::record_degradation(
                "streaming_failure",
                "terrain_cog_page_shape",
                "COG height page could not be resampled to the square mosaic tile",
            );
            vec![0.0; (width * height) as usize]
        }
    }
}

/// Store-backed adapter for caller-provided DEM readers. Even in-memory DEM
/// sources therefore enter the renderer through `VirtualTextureStore::page`;
/// COG and packed-file sources use the same downstream path.
struct ReaderPageStore {
    reader: Arc<dyn HeightReader>,
    root_bounds: TileBounds,
    tile_world_size: Vec2,
    tile_resolution: u32,
    metadata: crate::terrain::vt::StoreMetadata,
    digest: [u8; 32],
}

impl ReaderPageStore {
    fn new(
        reader: Arc<dyn HeightReader>,
        root_bounds: TileBounds,
        tile_world_size: Vec2,
        tile_resolution: u32,
        lod: u32,
    ) -> Self {
        let tiles_axis = 1u64 << lod;
        let virtual_side = tiles_axis * u64::from(tile_resolution);
        let metadata = crate::terrain::vt::StoreMetadata {
            virtual_width: virtual_side,
            virtual_height: virtual_side,
            tile_size: tile_resolution,
            tile_border: 0,
            family_count: 4,
            procedural: false,
            procedural_seed: 0,
        };
        let mut identity = Vec::new();
        identity.extend_from_slice(&virtual_side.to_le_bytes());
        identity.extend_from_slice(&tile_resolution.to_le_bytes());
        identity.extend_from_slice(&lod.to_le_bytes());
        let digest = crate::core::provenance::sha256(&identity);
        Self {
            reader,
            root_bounds,
            tile_world_size,
            tile_resolution,
            metadata,
            digest,
        }
    }
}

impl crate::terrain::vt::VirtualTextureStore for ReaderPageStore {
    fn page(
        &self,
        key: crate::terrain::vt::PageKey,
    ) -> std::result::Result<crate::terrain::vt::PageBytes, String> {
        if key.family != crate::terrain::vt::HEIGHT_FAMILY {
            return Err(format!("height store cannot serve family {}", key.family));
        }
        let heights = self.reader.read(
            &self.root_bounds,
            self.tile_world_size,
            TileId::new(u32::from(key.mip), key.x, key.y),
            self.tile_resolution,
            self.tile_resolution,
        );
        crate::terrain::vt::PageBytes::new(
            crate::terrain::vt::PageFormat::R32Float,
            self.tile_resolution,
            self.tile_resolution,
            bytemuck::cast_slice(&heights).to_vec(),
        )
    }

    fn metadata(&self) -> &crate::terrain::vt::StoreMetadata {
        &self.metadata
    }

    fn content_hash(&self) -> [u8; 32] {
        self.digest
    }

    fn page_count(&self) -> u64 {
        0
    }
}

fn upsample_bilinear(src: &[f32], src_res: u32, dst_res: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((dst_res * dst_res) as usize);
    for y in 0..dst_res {
        for x in 0..dst_res {
            let fx = x as f32 / (dst_res - 1).max(1) as f32 * (src_res - 1) as f32;
            let fy = y as f32 / (dst_res - 1).max(1) as f32 * (src_res - 1) as f32;
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min((src_res - 1) as usize);
            let y1 = (y0 + 1).min((src_res - 1) as usize);
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            let s = src_res as usize;
            let top = src[y0 * s + x0] * (1.0 - tx) + src[y0 * s + x1] * tx;
            let bottom = src[y1 * s + x0] * (1.0 - tx) + src[y1 * s + x1] * tx;
            out.push(top * (1.0 - ty) + bottom * ty);
        }
    }
    out
}

/// Map a tile at an arbitrary LOD onto a fixed mosaic LOD: same LOD passes
/// through, finer tiles collapse to their ancestor, coarser tiles expand to
/// all covered descendants. Coordinates outside the mosaic axis are dropped.
fn map_tile_to_fixed_lod(tile: TileId, fixed_lod: u32, tiles_axis: u32, out: &mut HashSet<TileId>) {
    use std::cmp::Ordering;
    let clamp = |x: u32, y: u32| -> Option<(u32, u32)> {
        (x < tiles_axis && y < tiles_axis).then_some((x, y))
    };
    match tile.lod.cmp(&fixed_lod) {
        Ordering::Equal => {
            if let Some((x, y)) = clamp(tile.x, tile.y) {
                out.insert(TileId::new(fixed_lod, x, y));
            }
        }
        Ordering::Greater => {
            let shift = tile.lod - fixed_lod;
            if let Some((x, y)) = clamp(tile.x >> shift, tile.y >> shift) {
                out.insert(TileId::new(fixed_lod, x, y));
            }
        }
        Ordering::Less => {
            let shift = fixed_lod - tile.lod;
            let scale = 1u32 << shift;
            for dy in 0..scale {
                for dx in 0..scale {
                    if let Some((x, y)) = clamp(tile.x * scale + dx, tile.y * scale + dy) {
                        out.insert(TileId::new(fixed_lod, x, y));
                    }
                }
            }
        }
    }
}

pub(in crate::terrain::renderer) struct HeightStreamingStats {
    pub center: Vec2,
    pub pending_ring_tiles: usize,
    pub loaded_ring_tiles: usize,
    pub resident_fine_tiles: usize,
    pub total_tiles: usize,
    pub tiles_requested: usize,
    pub tiles_uploaded: usize,
    pub coarse_prefilled: usize,
    pub resident_height_bytes: u64,
    pub converged: bool,
    pub loader_pending: usize,
    pub loader_completed: usize,
}

/// Fourth-family VT runtime. Height uses an R32Float physical mosaic, but its
/// demand retention and residency accounting are the same feedback-driven
/// `TileKey`/family policy used by albedo, normal, and mask.
pub(in crate::terrain::renderer) struct HeightVtFamilyRuntime {
    residency_owner_id: u64,
    pub(in crate::terrain::renderer) streamer: ClipmapStreamer,
    pub(in crate::terrain::renderer) mosaic: HeightMosaic,
    loader: AsyncTileLoader,
    reader: Arc<dyn HeightReader>,
    root_bounds: TileBounds,
    tile_world_size: Vec2,
    lod: u32,
    tiles_axis: u32,
    tile_resolution: u32,
    resident_fine: HashSet<TileId>,
    feedback_requests: crate::terrain::vt::requests::RetainedRequestSet,
    family_residency: FamilyResidencyTracker,
    /// ring tile -> fine tiles still missing before it counts as loaded
    ring_waiting: HashMap<TileId, HashSet<TileId>>,
    tiles_requested: usize,
    tiles_uploaded: usize,
    coarse_prefilled: usize,
}

impl HeightVtFamilyRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain_extent: f32,
        ring_count: u32,
        ring_resolution: u32,
        lod: u32,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn HeightReader>,
        coarse_prefill: bool,
        max_resident_bytes: Option<u64>,
    ) -> Result<Self> {
        let tiles_axis = 1u32 << lod;
        let resident_bytes = u64::from(tiles_axis)
            * u64::from(tiles_axis)
            * u64::from(tile_resolution)
            * u64::from(tile_resolution)
            * 4;
        if let Some(budget) = max_resident_bytes {
            if resident_bytes > budget {
                return Err(anyhow!(
                    "height streaming mosaic would use {} bytes, exceeding max_resident_bytes={}",
                    resident_bytes,
                    budget
                ));
            }
        }

        let half = terrain_extent * 0.5;
        let root_bounds = TileBounds::new(Vec2::new(-half, -half), Vec2::new(half, half));
        let tile_world_size = Vec2::splat(terrain_extent);

        let mosaic = HeightMosaic::new(
            device,
            MosaicConfig {
                tile_size_px: tile_resolution,
                tiles_x: tiles_axis,
                tiles_y: tiles_axis,
                fixed_lod: Some(lod),
            },
            false,
        )?;
        let store: Arc<dyn crate::terrain::vt::VirtualTextureStore> =
            Arc::new(ReaderPageStore::new(
                reader,
                root_bounds.clone(),
                tile_world_size,
                tile_resolution,
                lod,
            ));
        let reader: Arc<dyn HeightReader> = Arc::new(StoreHeightReader::new(store));
        let loader = AsyncTileLoader::new_with_reader(
            root_bounds.clone(),
            tile_world_size,
            tile_resolution,
            max_in_flight.max(1),
            pool_size.max(1),
            reader.clone(),
            CoalescePolicy::PreferFine,
        );
        let streamer = ClipmapStreamer::new(
            ClipmapConfig::new(ring_count.clamp(1, 8), ring_resolution.clamp(4, 256)),
            Vec2::ZERO,
            terrain_extent,
        );

        let mut state = Self {
            residency_owner_id: crate::core::memory_tracker::global_tracker()
                .allocate_resident_owner(),
            streamer,
            mosaic,
            loader,
            reader,
            root_bounds,
            tile_world_size,
            lod,
            tiles_axis,
            tile_resolution,
            resident_fine: HashSet::new(),
            feedback_requests: Default::default(),
            family_residency: FamilyResidencyTracker::new(
                max_resident_bytes.unwrap_or(resident_bytes),
                1u32 << crate::terrain::vt::HEIGHT_FAMILY,
                u64::from(tile_resolution) * u64::from(tile_resolution) * 4,
            ),
            ring_waiting: HashMap::new(),
            tiles_requested: 0,
            tiles_uploaded: 0,
            coarse_prefilled: 0,
        };
        if coarse_prefill {
            state.coarse_prefill(queue)?;
        }
        Ok(state)
    }

    /// Synchronously fill every mosaic slot from a low-resolution read so the
    /// terrain never shows holes while fine tiles are in flight.
    fn coarse_prefill(&mut self, queue: &wgpu::Queue) -> Result<()> {
        let coarse_res = (self.tile_resolution / 8).max(2);
        for y in 0..self.tiles_axis {
            for x in 0..self.tiles_axis {
                let id = TileId::new(self.lod, x, y);
                let coarse = self.reader.read(
                    &self.root_bounds,
                    self.tile_world_size,
                    id,
                    coarse_res,
                    coarse_res,
                );
                let up = upsample_bilinear(&coarse, coarse_res, self.tile_resolution);
                self.mosaic
                    .upload_tile(queue, id, &up)
                    .map_err(|e| anyhow!("coarse prefill upload failed for {:?}: {}", id, e))?;
                self.coarse_prefilled += 1;
            }
        }
        Ok(())
    }

    fn tile_center_world(&self, tile: TileId, tile_world: f32) -> Vec2 {
        self.root_bounds.min
            + Vec2::new(
                (tile.x as f32 + 0.5) * tile_world,
                (tile.y as f32 + 0.5) * tile_world,
            )
    }

    /// Map a tile at an arbitrary LOD onto the mosaic's fixed LOD.
    fn tiles_at_fixed_lod(&self, tile: TileId, out: &mut HashSet<TileId>) {
        map_tile_to_fixed_lod(tile, self.lod, self.tiles_axis, out);
    }

    /// One streaming step. Visibility feedback is the primary demand signal;
    /// camera-derived ring tiles are lower-priority predictive prefetch.
    pub(in crate::terrain::renderer) fn stream_step(
        &mut self,
        queue: &wgpu::Queue,
        camera_pos: Vec3,
        feedback_uvs: &[[f32; 2]],
        max_uploads: usize,
    ) -> HeightStreamingStats {
        for uv in feedback_uvs {
            let x = (uv[0].clamp(0.0, 1.0 - f32::EPSILON) * self.tiles_axis as f32) as u32;
            let y = (uv[1].clamp(0.0, 1.0 - f32::EPSILON) * self.tiles_axis as f32) as u32;
            let tile = TileId::new(self.lod, x, y);
            if self.resident_fine.contains(&tile) {
                continue;
            }
            let key = VtTileKey {
                family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                material_index: 0,
                x,
                y,
                mip_level: self.lod,
            };
            self.feedback_requests[crate::terrain::vt::HEIGHT_FAMILY as usize].insert(key);
            if self.loader.request(tile) {
                self.tiles_requested += 1;
            }
        }

        let lod_config = LodConfig::new(2.0, 1024, 768, 45.0f32.to_radians());
        let ring_tiles =
            self.streamer
                .update(camera_pos, Mat4::IDENTITY, Mat4::IDENTITY, &lod_config);

        for ring_tile in &ring_tiles {
            let mut fine = HashSet::new();
            self.tiles_at_fixed_lod(*ring_tile, &mut fine);
            fine.retain(|t| !self.resident_fine.contains(t));
            if fine.is_empty() {
                self.streamer.mark_loaded(std::slice::from_ref(ring_tile));
                continue;
            }
            for t in &fine {
                self.feedback_requests[crate::terrain::vt::HEIGHT_FAMILY as usize].insert(
                    VtTileKey {
                        family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                        material_index: 0,
                        x: t.x,
                        y: t.y,
                        mip_level: t.lod,
                    },
                );
                if self.loader.request(*t) {
                    self.tiles_requested += 1;
                }
            }
            self.ring_waiting
                .entry(*ring_tile)
                .or_default()
                .extend(fine);
        }

        // Top-up: keep the loader saturated with the nearest non-resident
        // tiles so residency converges even when the clipmap center is
        // stationary; `request` dedups pending ids and enforces the
        // max-in-flight backpressure budget.
        let center = self.streamer.center();
        let tile_world = self.tile_world_size.x / self.tiles_axis as f32;
        let mut missing: Vec<TileId> = (0..self.tiles_axis)
            .flat_map(|y| (0..self.tiles_axis).map(move |x| (x, y)))
            .map(|(x, y)| TileId::new(self.lod, x, y))
            .filter(|t| !self.resident_fine.contains(t))
            .collect();
        missing.sort_by(|a, b| {
            let da = self.tile_center_world(*a, tile_world).distance(center);
            let db = self.tile_center_world(*b, tile_world).distance(center);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        for t in missing {
            self.feedback_requests[crate::terrain::vt::HEIGHT_FAMILY as usize].insert(VtTileKey {
                family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                material_index: 0,
                x: t.x,
                y: t.y,
                mip_level: t.lod,
            });
            if self.loader.request(t) {
                self.tiles_requested += 1;
            }
        }

        let completed = self.loader.drain_completed(max_uploads.max(1));
        for td in completed {
            if (td.width * td.height) as usize != td.height_data.len() {
                continue;
            }
            if self
                .mosaic
                .upload_tile(queue, td.tile_id, &td.height_data)
                .is_ok()
            {
                self.resident_fine.insert(td.tile_id);
                let key = VtTileKey {
                    family_slot: crate::terrain::vt::HEIGHT_FAMILY as u32,
                    material_index: 0,
                    x: td.tile_id.x,
                    y: td.tile_id.y,
                    mip_level: td.tile_id.lod,
                };
                self.feedback_requests[crate::terrain::vt::HEIGHT_FAMILY as usize].remove(&key);
                self.family_residency.on_insert(key);
                self.tiles_uploaded += 1;
            }
        }

        let mut satisfied: Vec<TileId> = Vec::new();
        for (ring_tile, missing) in self.ring_waiting.iter_mut() {
            missing.retain(|t| !self.resident_fine.contains(t));
            if missing.is_empty() {
                satisfied.push(*ring_tile);
            }
        }
        for ring_tile in &satisfied {
            self.ring_waiting.remove(ring_tile);
        }
        if !satisfied.is_empty() {
            self.streamer.mark_loaded(&satisfied);
        }

        let stats = self.stats();
        let height = self
            .family_residency
            .family(crate::terrain::vt::HEIGHT_FAMILY as u32);
        super::virtual_texture::publish_height_family_stats(
            self.residency_owner_id,
            height.resident_tiles,
            height.resident_bytes,
            height.budget_bytes,
            self.feedback_requests
                .iter()
                .map(|bucket| bucket.len() as u32)
                .sum(),
        );
        stats
    }

    pub(in crate::terrain::renderer) fn stats(&self) -> HeightStreamingStats {
        let total_tiles = (self.tiles_axis * self.tiles_axis) as usize;
        let (loader_pending, _, _) = self.loader.stats();
        let (_, _, _, _, _, loader_completed) = self.loader.counters();
        HeightStreamingStats {
            center: self.streamer.center(),
            pending_ring_tiles: self.streamer.pending_count(),
            loaded_ring_tiles: self.streamer.loaded_count(),
            resident_fine_tiles: self.resident_fine.len(),
            total_tiles,
            tiles_requested: self.tiles_requested,
            tiles_uploaded: self.tiles_uploaded,
            coarse_prefilled: self.coarse_prefilled,
            resident_height_bytes: u64::from(self.tiles_axis)
                * u64::from(self.tiles_axis)
                * u64::from(self.tile_resolution)
                * u64::from(self.tile_resolution)
                * 4,
            converged: self.resident_fine.len() == total_tiles,
            loader_pending,
            loader_completed,
        }
    }
}

impl Drop for HeightVtFamilyRuntime {
    fn drop(&mut self) {
        super::virtual_texture::clear_height_family_stats(self.residency_owner_id);
    }
}

impl TerrainScene {
    /// Clipmap mesh center: follows the streaming center when height
    /// streaming is active, otherwise stays at the region origin.
    pub(in crate::terrain::renderer) fn height_streaming_center(&self) -> Vec2 {
        self.height_streaming
            .as_ref()
            .map(|s| s.streamer.center())
            .unwrap_or(Vec2::ZERO)
    }

    /// Height texture view for the main/AOV/offline terrain passes: the
    /// streaming mosaic when active, otherwise the per-render uploaded
    /// overview heightmap.
    pub(in crate::terrain::renderer) fn main_pass_height_view<'a>(
        &'a self,
        uploaded: &'a wgpu::TextureView,
    ) -> &'a wgpu::TextureView {
        self.height_streaming
            .as_ref()
            .map(|s| &s.mosaic.view)
            .unwrap_or(uploaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dem_slice_reader_reproduces_linear_ramp() {
        // A DEM that is linear in x should bilinear-sample exactly.
        let (w, h) = (9usize, 5usize);
        let dem: Vec<f32> = (0..h)
            .flat_map(|_| (0..w).map(|x| x as f32 / (w - 1) as f32))
            .collect();
        let reader = DemSliceHeightReader::new(dem, w, h);
        let root = TileBounds::new(Vec2::new(-50.0, -50.0), Vec2::new(50.0, 50.0));
        let tile_size = Vec2::splat(100.0);

        // LOD 0 tile spans the whole region: corners must match DEM corners.
        let full = reader.read(&root, tile_size, TileId::new(0, 0, 0), 9, 5);
        assert!((full[0] - 0.0).abs() < 1e-6);
        assert!((full[8] - 1.0).abs() < 1e-6);

        // LOD 1 tile (1, 0) covers the right half in x: values in [0.5, 1.0].
        let right = reader.read(&root, tile_size, TileId::new(1, 1, 0), 5, 3);
        for v in &right {
            assert!(
                *v >= 0.5 - 1e-6 && *v <= 1.0 + 1e-6,
                "value {} out of right-half range",
                v
            );
        }
    }

    #[test]
    fn upsample_bilinear_preserves_corners_and_range() {
        let src = vec![0.0f32, 1.0, 2.0, 3.0];
        let up = upsample_bilinear(&src, 2, 5);
        assert_eq!(up.len(), 25);
        assert!((up[0] - 0.0).abs() < 1e-6);
        assert!((up[4] - 1.0).abs() < 1e-6);
        assert!((up[20] - 2.0).abs() < 1e-6);
        assert!((up[24] - 3.0).abs() < 1e-6);
        for v in &up {
            assert!(*v >= 0.0 && *v <= 3.0);
        }
    }

    #[test]
    fn map_tile_to_fixed_lod_covers_all_cases() {
        let mut out = HashSet::new();

        // Same LOD passes through.
        map_tile_to_fixed_lod(TileId::new(2, 1, 3), 2, 4, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out.contains(&TileId::new(2, 1, 3)));

        // Finer collapses to ancestor.
        out.clear();
        map_tile_to_fixed_lod(TileId::new(4, 13, 6), 2, 4, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out.contains(&TileId::new(2, 3, 1)));

        // Coarser expands to all covered descendants.
        out.clear();
        map_tile_to_fixed_lod(TileId::new(0, 0, 0), 2, 4, &mut out);
        assert_eq!(out.len(), 16);

        // Out-of-axis coordinates are dropped.
        out.clear();
        map_tile_to_fixed_lod(TileId::new(2, 9, 0), 2, 4, &mut out);
        assert!(out.is_empty());
    }
}
