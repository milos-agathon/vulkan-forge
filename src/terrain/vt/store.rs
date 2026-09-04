//! Tile-indexed virtual-texture backing store.
//!
//! `MmapPageStore` keeps the public name from the TESSELLA contract. The
//! implementation deliberately uses positional `read_at`/`seek_read` calls
//! plus a bounded page cache instead of a new mmap dependency. Pages remain
//! out of the process address space until requested, work on Unix and Windows,
//! and never require copying the complete virtual image into a `Vec<u8>`.

use super::procedural::MaterializationPlan;
use crate::core::provenance::sha256;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAGIC: &[u8; 8] = b"F3DVT1\0\0";
/// v2 records the materialization plan in header bytes 80..92 and drops the
/// v1 aliasing rule (v1 resolved any missing procedural key onto one
/// canonical page per family, which made every tile byte-identical).
/// `decode_header` rejects v1 explicitly rather than reading it wrong.
const VERSION: u32 = 2;
const HEADER_SIZE: usize = 96;
const DIRECTORY_ENTRY_SIZE: usize = 64;
const FLAG_PROCEDURAL: u32 = 1;
const DEFAULT_CACHE_BYTES: u64 = 64 * 1024 * 1024;
/// `morton2` interleaves the low 16 bits of each axis, so the directory's
/// `(family, mip, morton2)` order is only a total order below this bound.
const MAX_PAGE_AXIS: u32 = 1 << 16;
pub const HEIGHT_FAMILY: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PageKey {
    pub family: u8,
    pub mip: u8,
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageFormat {
    Bc7Srgb,
    Bc5Unorm,
    Bc7Unorm,
    Rgba8Srgb,
    R32Float,
}

impl PageFormat {
    pub fn block_bytes(self, width: u32, height: u32) -> u64 {
        match self {
            Self::Bc7Srgb | Self::Bc5Unorm | Self::Bc7Unorm => {
                u64::from(width.div_ceil(4)) * u64::from(height.div_ceil(4)) * 16
            }
            Self::Rgba8Srgb | Self::R32Float => u64::from(width) * u64::from(height) * 4,
        }
    }

    pub fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Bc7Srgb => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
            Self::Bc5Unorm => wgpu::TextureFormat::Bc5RgUnorm,
            Self::Bc7Unorm => wgpu::TextureFormat::Bc7RgbaUnorm,
            Self::Rgba8Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::R32Float => wgpu::TextureFormat::R32Float,
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::Bc7Srgb => 1,
            Self::Bc5Unorm => 2,
            Self::Bc7Unorm => 3,
            Self::Rgba8Srgb => 4,
            Self::R32Float => 5,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, String> {
        match tag {
            1 => Ok(Self::Bc7Srgb),
            2 => Ok(Self::Bc5Unorm),
            3 => Ok(Self::Bc7Unorm),
            4 => Ok(Self::Rgba8Srgb),
            5 => Ok(Self::R32Float),
            _ => Err(format!("unknown VT page format tag {tag}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PageBytes {
    pub format: PageFormat,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub sha256: [u8; 32],
}

impl PageBytes {
    pub fn new(format: PageFormat, width: u32, height: u32, data: Vec<u8>) -> Result<Self, String> {
        let expected = format.block_bytes(width, height);
        if data.len() as u64 != expected {
            return Err(format!(
                "{format:?} page payload mismatch: got {} bytes, expected {expected}",
                data.len()
            ));
        }
        Ok(Self {
            format,
            width,
            height,
            sha256: sha256(&data),
            data,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PackedPage {
    pub key: PageKey,
    pub bytes: PageBytes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreMetadata {
    pub virtual_width: u64,
    pub virtual_height: u64,
    pub tile_size: u32,
    pub tile_border: u32,
    pub family_count: u32,
    pub procedural: bool,
    pub procedural_seed: u64,
}

impl StoreMetadata {
    pub fn slot_size(&self) -> u32 {
        self.tile_size + self.tile_border.saturating_mul(2)
    }

    pub fn logical_texel_bytes(&self) -> u128 {
        u128::from(self.virtual_width)
            * u128::from(self.virtual_height)
            * u128::from(self.family_count)
            * 4
    }

    /// Page-grid dimensions at `mip`. Identical by construction to the
    /// renderer's `pages_for_mip_counts(ceil_div(w, tile), ..., mip)`
    /// (`terrain/renderer/virtual_texture.rs`); a store completeness claim and
    /// the renderer's request grid must agree or the zero-fallback gate is
    /// meaningless. `pages_at_mip_matches_the_renderer_page_grid` locks it.
    pub fn pages_at_mip(&self, mip: u32) -> (u32, u32) {
        let tile_size = u64::from(self.tile_size.max(1));
        let pages_x = self.virtual_width.div_ceil(tile_size).max(1);
        let pages_y = self.virtual_height.div_ceil(tile_size).max(1);
        let div = 1u64 << mip.min(63);
        (
            u32::try_from(pages_x.div_ceil(div).max(1)).unwrap_or(u32::MAX),
            u32::try_from(pages_y.div_ceil(div).max(1)).unwrap_or(u32::MAX),
        )
    }

    /// Number of mip levels in the store's full page pyramid -- mirrors
    /// `TerrainMaterialVTRuntime::page_table_mip_levels`.
    pub fn mip_count(&self) -> u32 {
        let (pages_x, pages_y) = self.pages_at_mip(0);
        u32::BITS - pages_x.max(pages_y).max(1).leading_zeros()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreManifest {
    pub format: String,
    pub version: u32,
    pub path: String,
    pub virtual_width: u64,
    pub virtual_height: u64,
    pub logical_texel_bytes: String,
    pub tile_size: u32,
    pub tile_border: u32,
    pub family_count: u32,
    pub page_count: u64,
    pub procedural: bool,
    pub page_order: String,
    /// Recorded materialization plan: exactly which keys the store holds.
    pub materialization_plan: MaterializationPlan,
    /// `materialization_plan.coarse_min_mip`, hoisted for readers that only
    /// need the "every page at or above this mip exists" floor.
    pub min_materialized_mip: u32,
    /// Number of distinct page payloads. Anti-degeneracy invariant: a store
    /// whose pages are keyed by page identity has one payload per page, so a
    /// value below `page_count` means pages are aliasing each other.
    pub distinct_page_digests: u64,
    pub encodings: Vec<PageFormat>,
    pub directory_sha256: String,
    /// Integrity digest for every physically packed page.
    pub pages: Vec<ManifestPageDigest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestPageDigest {
    pub key: PageKey,
    pub sha256: String,
}

pub trait VirtualTextureStore: Send + Sync {
    fn page(&self, key: PageKey) -> Result<PageBytes, String>;
    fn metadata(&self) -> &StoreMetadata;
    fn content_hash(&self) -> [u8; 32];
    fn page_count(&self) -> u64;

    /// True when the store can serve this page. Never synthesizes a
    /// substitute: a `false` here is what lets the renderer count an explicit
    /// miss instead of hard-failing or silently reading the wrong tile.
    fn contains(&self, key: PageKey) -> bool {
        self.page(key).is_ok()
    }

    /// Coarsest-complete floor: every page at `mip >= min_materialized_mip`
    /// exists for every family. Stores that can serve any key of the pyramid
    /// (readers, COG) report 0.
    fn min_materialized_mip(&self) -> u32 {
        0
    }

    /// Re-derive this store for the atlas tiling the render params ask for.
    ///
    /// `Ok(None)` means "already correct, keep using me". A packed file
    /// commits to its tiling in the header, so the default implementation
    /// accepts only its own geometry and reports a mismatch here -- the
    /// earliest and clearest place to catch render params that disagree with
    /// the store they were handed. An in-RAM ingest (`MemoryPageStore`) has no
    /// committed tiling and returns a handle over the same shared mip chain,
    /// sliced the new way.
    fn rebind_tile_geometry(
        &self,
        tile_size: u32,
        tile_border: u32,
    ) -> Result<Option<Arc<dyn VirtualTextureStore>>, String> {
        let metadata = self.metadata();
        if metadata.tile_size == tile_size && metadata.tile_border == tile_border {
            return Ok(None);
        }
        Err(format!(
            "VT store is packed with {}-texel tiles (+{} border); the render params ask for {tile_size} (+{tile_border})",
            metadata.tile_size, metadata.tile_border
        ))
    }
}

/// RGBA8 bytes per texel for an in-RAM ingest. The renderer's material VT is
/// RGBA8 at its Python boundary; BC encoding happens on the way to the atlas,
/// not in the store.
const INGEST_BYTES_PER_PIXEL: usize = 4;

#[derive(Debug)]
struct MemoryMip {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

/// In-RAM virtual-texture store over an image handed to the renderer through
/// `TerrainRenderer.register_material_vt_source`.
///
/// TESSELLA spec item 4 requires every VT source to reach the atlas through
/// `VirtualTextureStore`. A Python entry point necessarily receives an image
/// rather than a path, so the ingest boundary converts it into a store handle
/// on the spot: the mip chain is built once, and from that moment the renderer
/// has exactly one way to obtain a page (`VirtualTextureStore::page`) and no
/// in-RAM special case. `page` slices the requested tile out of the chosen
/// level with the border clamp the atlas slot needs -- the work the renderer's
/// deleted in-RAM tile branch used to do inline.
///
/// The tiling is deliberately NOT fixed at ingest: an in-RAM image has no
/// committed page grid, so `new` records `tile_size == 0` ("unbound") and
/// `page` refuses to serve until `rebind_tile_geometry` has produced a handle
/// for the renderer's actual atlas slot size. Rebinding shares the mip chain
/// through an `Arc`, so it never copies the image.
pub struct MemoryPageStore {
    mips: Arc<Vec<MemoryMip>>,
    metadata: StoreMetadata,
    content_hash: [u8; 32],
}

impl MemoryPageStore {
    /// Ingest an RGBA8 image. `family_count` is the number of families the
    /// renderer will address through this handle; each registered source owns
    /// its own store, so pages are keyed by (mip, x, y) and the family axis of
    /// the key is not part of this store's address space.
    pub fn new(rgba: &[u8], virtual_size: (u32, u32), family_count: u32) -> Result<Self, String> {
        if virtual_size.0 == 0 || virtual_size.1 == 0 {
            return Err("VT ingest virtual size must be > 0 in both dimensions".to_string());
        }
        let expected = virtual_size.0 as usize * virtual_size.1 as usize * INGEST_BYTES_PER_PIXEL;
        if rgba.len() != expected {
            return Err(format!(
                "VT ingest size mismatch: expected {expected} RGBA8 bytes for {}x{}, got {}",
                virtual_size.0,
                virtual_size.1,
                rgba.len()
            ));
        }
        Ok(Self {
            content_hash: sha256(rgba),
            mips: Arc::new(build_mip_chain(rgba, virtual_size)),
            metadata: StoreMetadata {
                virtual_width: u64::from(virtual_size.0),
                virtual_height: u64::from(virtual_size.1),
                // Unbound: `rebind_tile_geometry` supplies the renderer's slot
                // geometry before any page is served.
                tile_size: 0,
                tile_border: 0,
                family_count,
                procedural: false,
                procedural_seed: 0,
            },
        })
    }

    fn level(&self, mip: u8) -> &MemoryMip {
        // The renderer bounds requests by `max_mip_levels`, which can exceed
        // the number of distinct levels; the 1x1 tail repeats, matching the
        // chain the deleted `build_rgba_mip_chain` used to pad out.
        let index = (mip as usize).min(self.mips.len() - 1);
        &self.mips[index]
    }
}

impl VirtualTextureStore for MemoryPageStore {
    fn page(&self, key: PageKey) -> Result<PageBytes, String> {
        if self.metadata.tile_size == 0 {
            return Err(
                "in-RAM VT store has no committed tiling; rebind_tile_geometry must run first"
                    .to_string(),
            );
        }
        if !self.contains(key) {
            return Err(format!("in-RAM VT store has no page {key:?}"));
        }
        let mip = self.level(key.mip);
        let slot_size = self.metadata.slot_size() as usize;
        let tile_size = self.metadata.tile_size as i32;
        let tile_border = self.metadata.tile_border as i32;
        let mut data = vec![0u8; slot_size * slot_size * INGEST_BYTES_PER_PIXEL];
        for slot_y in 0..slot_size {
            let src_y = (key.y as i32 * tile_size + slot_y as i32 - tile_border)
                .clamp(0, mip.height as i32 - 1) as usize;
            for slot_x in 0..slot_size {
                let src_x = (key.x as i32 * tile_size + slot_x as i32 - tile_border)
                    .clamp(0, mip.width as i32 - 1) as usize;
                let src = (src_y * mip.width as usize + src_x) * INGEST_BYTES_PER_PIXEL;
                let dst = (slot_y * slot_size + slot_x) * INGEST_BYTES_PER_PIXEL;
                data[dst..dst + INGEST_BYTES_PER_PIXEL]
                    .copy_from_slice(&mip.data[src..src + INGEST_BYTES_PER_PIXEL]);
            }
        }
        PageBytes::new(
            PageFormat::Rgba8Srgb,
            self.metadata.slot_size(),
            self.metadata.slot_size(),
            data,
        )
    }

    fn metadata(&self) -> &StoreMetadata {
        &self.metadata
    }

    fn content_hash(&self) -> [u8; 32] {
        self.content_hash
    }

    /// Every page of the pyramid is derivable, so the store's page count is
    /// the size of that pyramid rather than a directory length.
    fn page_count(&self) -> u64 {
        (0..self.mips.len() as u32)
            .map(|mip| {
                let (x, y) = self.metadata.pages_at_mip(mip);
                u64::from(x) * u64::from(y)
            })
            .sum()
    }

    /// Bounds-only, no slicing and no digest: the residency planner asks this
    /// thousands of times per frame.
    ///
    /// A mip past the end of the chain is still served (from the repeated 1x1
    /// tail, see `level`) because the renderer's `max_mip_levels` can exceed
    /// the chain depth and the page grid is 1x1 there anyway. Rejecting it
    /// would turn a legal request into a counted store miss and silently drop
    /// it.
    fn contains(&self, key: PageKey) -> bool {
        if self.metadata.tile_size == 0 {
            return false;
        }
        let (pages_x, pages_y) = self.metadata.pages_at_mip(u32::from(key.mip));
        key.x < pages_x && key.y < pages_y
    }

    fn rebind_tile_geometry(
        &self,
        tile_size: u32,
        tile_border: u32,
    ) -> Result<Option<Arc<dyn VirtualTextureStore>>, String> {
        if tile_size == 0 {
            return Err("VT atlas tile size must be > 0".to_string());
        }
        if self.metadata.tile_size == tile_size && self.metadata.tile_border == tile_border {
            return Ok(None);
        }
        let mut metadata = self.metadata.clone();
        metadata.tile_size = tile_size;
        metadata.tile_border = tile_border;
        Ok(Some(Arc::new(Self {
            mips: Arc::clone(&self.mips),
            metadata,
            content_hash: self.content_hash,
        })))
    }
}

/// Box-filtered RGBA8 mip chain, down to 1x1.
fn build_mip_chain(data: &[u8], size: (u32, u32)) -> Vec<MemoryMip> {
    let mut chain = vec![MemoryMip {
        width: size.0,
        height: size.1,
        data: data.to_vec(),
    }];
    while {
        let last = chain.last().unwrap();
        last.width > 1 || last.height > 1
    } {
        let previous = chain.last().unwrap();
        let next_width = previous.width.max(1).div_ceil(2);
        let next_height = previous.height.max(1).div_ceil(2);
        let mut next_data =
            vec![0u8; next_width as usize * next_height as usize * INGEST_BYTES_PER_PIXEL];
        for y in 0..next_height {
            for x in 0..next_width {
                let mut accum = [0u32; INGEST_BYTES_PER_PIXEL];
                let mut sample_count = 0u32;
                for src_y in (y * 2)..((y * 2 + 2).min(previous.height)) {
                    for src_x in (x * 2)..((x * 2 + 2).min(previous.width)) {
                        let src = (src_y as usize * previous.width as usize + src_x as usize)
                            * INGEST_BYTES_PER_PIXEL;
                        for channel in 0..INGEST_BYTES_PER_PIXEL {
                            accum[channel] += u32::from(previous.data[src + channel]);
                        }
                        sample_count += 1;
                    }
                }
                let dst = (y as usize * next_width as usize + x as usize) * INGEST_BYTES_PER_PIXEL;
                for channel in 0..INGEST_BYTES_PER_PIXEL {
                    next_data[dst + channel] = (accum[channel] / sample_count.max(1)) as u8;
                }
            }
        }
        chain.push(MemoryMip {
            width: next_width,
            height: next_height,
            data: next_data,
        });
    }
    chain
}

#[derive(Clone, Debug)]
struct DirectoryEntry {
    key: PageKey,
    format: PageFormat,
    width: u32,
    height: u32,
    offset: u64,
    length: u32,
    raw_length: u32,
    digest: [u8; 32],
}

/// Bounded LRU page cache.
///
/// Recency is a monotonic tick stored beside each entry and indexed by an
/// ordered map, so `get` re-stamps in O(log n). The previous `VecDeque` scan
/// was O(n) per hit, which a complete coarse working set (thousands of live
/// pages, each touched every frame) turns into tens of millions of
/// comparisons per frame.
struct PageCache {
    entries: HashMap<PageKey, (PageBytes, u64)>,
    order: BTreeMap<u64, PageKey>,
    tick: u64,
    used_bytes: u64,
    budget_bytes: u64,
}

impl PageCache {
    fn new(budget_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            order: BTreeMap::new(),
            tick: 0,
            used_bytes: 0,
            budget_bytes: budget_bytes.max(1),
        }
    }

    fn get(&mut self, key: PageKey) -> Option<PageBytes> {
        let previous = self.entries.get(&key).map(|(_, stamp)| *stamp)?;
        self.tick += 1;
        let stamp = self.tick;
        self.order.remove(&previous);
        self.order.insert(stamp, key);
        let entry = self.entries.get_mut(&key)?;
        entry.1 = stamp;
        Some(entry.0.clone())
    }

    fn insert(&mut self, key: PageKey, page: PageBytes) {
        let bytes = page.data.len() as u64;
        if bytes > self.budget_bytes {
            return;
        }
        if let Some((previous, stamp)) = self.entries.remove(&key) {
            self.order.remove(&stamp);
            self.used_bytes = self.used_bytes.saturating_sub(previous.data.len() as u64);
        }
        while self.used_bytes + bytes > self.budget_bytes {
            let Some((_, victim)) = self.order.pop_first() else {
                break;
            };
            if let Some((removed, _)) = self.entries.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(removed.data.len() as u64);
            }
        }
        self.tick += 1;
        let stamp = self.tick;
        self.order.insert(stamp, key);
        self.entries.insert(key, (page, stamp));
        self.used_bytes += bytes;
    }
}

pub struct MmapPageStore {
    path: PathBuf,
    file: File,
    metadata: StoreMetadata,
    plan: MaterializationPlan,
    /// Sorted by `order_key` -- the manifest's committed
    /// `page_order: "family,mip,morton2"` is load-bearing, not decorative:
    /// lookup is a binary search over exactly that order.
    directory: Vec<DirectoryEntry>,
    directory_hash: [u8; 32],
    cache: Mutex<PageCache>,
}

impl MmapPageStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::open_with_cache(path, DEFAULT_CACHE_BYTES)
    }

    pub fn open_with_cache(
        path: impl AsRef<Path>,
        cache_budget_bytes: u64,
    ) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .map_err(|error| format!("failed to open VT store {}: {error}", path.display()))?;
        let mut header = [0u8; HEADER_SIZE];
        read_exact_at(&file, &mut header, 0)?;
        let decoded = decode_header(&header)?;
        let directory_bytes_len = decoded
            .page_count
            .checked_mul(DIRECTORY_ENTRY_SIZE as u64)
            .ok_or_else(|| "VT directory length overflow".to_string())?;
        let mut directory_bytes = vec![
            0u8;
            usize::try_from(directory_bytes_len).map_err(|_| {
                "VT directory does not fit address space".to_string()
            })?
        ];
        read_exact_at(&file, &mut directory_bytes, decoded.directory_offset)?;
        let mut directory: Vec<DirectoryEntry> = Vec::with_capacity(decoded.page_count as usize);
        for chunk in directory_bytes.chunks_exact(DIRECTORY_ENTRY_SIZE) {
            let entry = decode_directory_entry(chunk)?;
            if entry.offset < decoded.data_offset {
                return Err(format!(
                    "VT page {:?} points inside the header/directory",
                    entry.key
                ));
            }
            validate_page_axes(entry.key)?;
            if let Some(previous) = directory.last() {
                if Self::order_key(entry.key) <= Self::order_key(previous.key) {
                    return Err(format!(
                        "VT page directory is not strictly sorted by (family, mip, morton2) at {:?}",
                        entry.key
                    ));
                }
            }
            directory.push(entry);
        }
        let store = Self {
            path,
            file,
            metadata: decoded.metadata,
            plan: decoded.plan,
            directory,
            directory_hash: sha256(&directory_bytes),
            cache: Mutex::new(PageCache::new(cache_budget_bytes)),
        };
        store.plan.validate(&store.metadata)?;
        // The declared plan is the invariant every downstream residency
        // decision leans on, so it is checked against the physical directory
        // at open rather than trusted.
        validate_materialization(&store.plan, &store.metadata, |key| {
            store.entry(key).is_some()
        })?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn materialization_plan(&self) -> MaterializationPlan {
        self.plan
    }

    pub fn verify(&self) -> Result<(), String> {
        for index in 0..self.directory.len() {
            self.read_page_uncached(self.directory[index].key)?;
        }
        Ok(())
    }

    fn order_key(key: PageKey) -> (u8, u8, u64) {
        (key.family, key.mip, morton2(key.x, key.y))
    }

    fn entry(&self, key: PageKey) -> Option<&DirectoryEntry> {
        if key.x >= MAX_PAGE_AXIS || key.y >= MAX_PAGE_AXIS {
            return None;
        }
        self.directory
            .binary_search_by_key(&Self::order_key(key), |entry| Self::order_key(entry.key))
            .ok()
            .map(|index| &self.directory[index])
    }

    fn read_page_uncached(&self, key: PageKey) -> Result<PageBytes, String> {
        // No aliasing and no synthesis: a key the directory does not hold does
        // not exist. Callers turn this into a counted miss (see
        // `TerrainMaterialVTRuntime::store_resolve`), never a substitution.
        let Some(entry) = self.entry(key) else {
            return Err(format!("VT page not found: {key:?}"));
        };
        let mut data = vec![0u8; entry.length as usize];
        read_exact_at(&self.file, &mut data, entry.offset)?;
        if entry.length != entry.raw_length {
            return Err(format!(
                "VT page {:?} uses unsupported secondary compression (stored={}, raw={})",
                key, entry.length, entry.raw_length
            ));
        }
        let actual = sha256(&data);
        if actual != entry.digest {
            return Err(format!(
                "VT page SHA-256 mismatch for {:?}: expected {}, got {}",
                key,
                crate::core::provenance::to_hex(&entry.digest),
                crate::core::provenance::to_hex(&actual)
            ));
        }
        PageBytes::new(entry.format, entry.width, entry.height, data)
    }
}

impl VirtualTextureStore for MmapPageStore {
    fn page(&self, key: PageKey) -> Result<PageBytes, String> {
        if let Some(page) = self
            .cache
            .lock()
            .map_err(|_| "VT page cache mutex poisoned".to_string())?
            .get(key)
        {
            return Ok(page);
        }
        let page = self.read_page_uncached(key)?;
        self.cache
            .lock()
            .map_err(|_| "VT page cache mutex poisoned".to_string())?
            .insert(key, page.clone());
        Ok(page)
    }

    fn metadata(&self) -> &StoreMetadata {
        &self.metadata
    }

    fn content_hash(&self) -> [u8; 32] {
        self.directory_hash
    }

    fn page_count(&self) -> u64 {
        self.directory.len() as u64
    }

    /// Directory-only, no I/O and no digest verification: the residency
    /// planner asks this thousands of times per frame.
    fn contains(&self, key: PageKey) -> bool {
        self.entry(key).is_some()
    }

    fn min_materialized_mip(&self) -> u32 {
        self.plan.coarse_min_mip
    }
}

#[cfg(feature = "cog_streaming")]
pub struct CogPageStore {
    reader: Arc<crate::terrain::cog::CogHeightReader>,
    metadata: StoreMetadata,
    digest: [u8; 32],
}

#[cfg(feature = "cog_streaming")]
impl CogPageStore {
    pub fn from_reader(reader: Arc<crate::terrain::cog::CogHeightReader>, tile_size: u32) -> Self {
        let (width, height) = reader
            .header()
            .full_resolution()
            .map(|ifd| (u64::from(ifd.width), u64::from(ifd.height)))
            .unwrap_or((1, 1));
        let mut identity = Vec::new();
        identity.extend_from_slice(&width.to_le_bytes());
        identity.extend_from_slice(&height.to_le_bytes());
        identity.extend_from_slice(&tile_size.to_le_bytes());
        Self {
            reader,
            metadata: StoreMetadata {
                virtual_width: width,
                virtual_height: height,
                tile_size,
                tile_border: 0,
                family_count: 1,
                procedural: false,
                procedural_seed: 0,
            },
            digest: sha256(&identity),
        }
    }
}

#[cfg(feature = "cog_streaming")]
impl VirtualTextureStore for CogPageStore {
    fn page(&self, key: PageKey) -> Result<PageBytes, String> {
        if key.family != HEIGHT_FAMILY {
            return Err(format!(
                "CogPageStore serves height family {HEIGHT_FAMILY}, got {}",
                key.family
            ));
        }
        let heights = self
            .reader
            .read_tile(key.x, key.y, u32::from(key.mip))
            .map_err(|error| error.to_string())?;
        let side = (heights.len() as f64).sqrt() as u32;
        if u64::from(side) * u64::from(side) != heights.len() as u64 {
            return Err("COG height tile is not square".to_string());
        }
        PageBytes::new(
            PageFormat::R32Float,
            side,
            side,
            bytemuck::cast_slice(&heights).to_vec(),
        )
    }

    fn metadata(&self) -> &StoreMetadata {
        &self.metadata
    }

    fn content_hash(&self) -> [u8; 32] {
        self.digest
    }

    fn page_count(&self) -> u64 {
        0
    }

    /// The reader synthesizes any height tile of the pyramid on demand, so the
    /// only key it cannot serve is one from another family.
    fn contains(&self, key: PageKey) -> bool {
        key.family == HEIGHT_FAMILY
    }
}

pub fn write_packed_store(
    path: impl AsRef<Path>,
    metadata: &StoreMetadata,
    plan: &MaterializationPlan,
    pages: impl IntoIterator<Item = PackedPage>,
) -> Result<StoreManifest, String> {
    validate_metadata(metadata)?;
    plan.validate(metadata)?;
    let path = path.as_ref();
    let mut pages = pages.into_iter().collect::<Vec<_>>();
    pages.sort_by_key(|page| {
        (
            page.key.family,
            page.key.mip,
            morton2(page.key.x, page.key.y),
            page.key.y,
            page.key.x,
        )
    });
    let page_count = pages.len() as u64;
    let directory_offset = HEADER_SIZE as u64;
    let data_offset = directory_offset
        .checked_add(page_count * DIRECTORY_ENTRY_SIZE as u64)
        .ok_or_else(|| "VT store data offset overflow".to_string())?;
    let mut directory = Vec::with_capacity(pages.len() * DIRECTORY_ENTRY_SIZE);
    let mut payload = Vec::new();
    let mut encodings = Vec::new();
    let mut previous_order: Option<(u8, u8, u64)> = None;
    for page in &pages {
        if page.bytes.width != metadata.slot_size() || page.bytes.height != metadata.slot_size() {
            return Err(format!(
                "VT page {:?} is {}x{}, expected slot size {}x{}",
                page.key,
                page.bytes.width,
                page.bytes.height,
                metadata.slot_size(),
                metadata.slot_size()
            ));
        }
        validate_page_axes(page.key)?;
        let order = (
            page.key.family,
            page.key.mip,
            morton2(page.key.x, page.key.y),
        );
        if previous_order == Some(order) {
            return Err(format!("duplicate VT page-directory key {:?}", page.key));
        }
        previous_order = Some(order);
        if !encodings.contains(&page.bytes.format) {
            encodings.push(page.bytes.format);
        }
        let entry = DirectoryEntry {
            key: page.key,
            format: page.bytes.format,
            width: page.bytes.width,
            height: page.bytes.height,
            offset: data_offset + payload.len() as u64,
            length: page.bytes.data.len() as u32,
            raw_length: page.bytes.data.len() as u32,
            digest: page.bytes.sha256,
        };
        directory.extend_from_slice(&encode_directory_entry(&entry)?);
        payload.extend_from_slice(&page.bytes.data);
    }
    // The plan is a promise the reader will rely on, so refuse to write a
    // store that does not already keep it.
    let packed = pages.iter().map(|page| page.key).collect::<HashSet<_>>();
    validate_materialization(plan, metadata, |key| packed.contains(&key))?;
    let distinct_page_digests = pages
        .iter()
        .map(|page| page.bytes.sha256)
        .collect::<HashSet<_>>()
        .len() as u64;
    let header = encode_header(metadata, plan, page_count, directory_offset, data_offset)?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to create VT store {}: {error}", path.display()))?;
    file.write_all(&header)
        .and_then(|_| file.write_all(&directory))
        .and_then(|_| file.write_all(&payload))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("failed to write VT store {}: {error}", path.display()))?;
    Ok(StoreManifest {
        format: "forge3d-vtpack".to_string(),
        version: VERSION,
        path: path.display().to_string(),
        virtual_width: metadata.virtual_width,
        virtual_height: metadata.virtual_height,
        logical_texel_bytes: metadata.logical_texel_bytes().to_string(),
        tile_size: metadata.tile_size,
        tile_border: metadata.tile_border,
        family_count: metadata.family_count,
        page_count,
        procedural: metadata.procedural,
        page_order: "family,mip,morton2".to_string(),
        materialization_plan: *plan,
        min_materialized_mip: plan.coarse_min_mip,
        distinct_page_digests,
        encodings,
        directory_sha256: crate::core::provenance::to_hex(&sha256(&directory)),
        pages: pages
            .iter()
            .map(|page| ManifestPageDigest {
                key: page.key,
                sha256: crate::core::provenance::to_hex(&page.bytes.sha256),
            })
            .collect(),
    })
}

struct DecodedHeader {
    metadata: StoreMetadata,
    plan: MaterializationPlan,
    page_count: u64,
    directory_offset: u64,
    data_offset: u64,
}

fn encode_header(
    metadata: &StoreMetadata,
    plan: &MaterializationPlan,
    page_count: u64,
    directory_offset: u64,
    data_offset: u64,
) -> Result<[u8; HEADER_SIZE], String> {
    validate_metadata(metadata)?;
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[0..8].copy_from_slice(MAGIC);
    put_u32(&mut bytes, 8, VERSION);
    put_u32(&mut bytes, 12, HEADER_SIZE as u32);
    put_u32(&mut bytes, 16, metadata.tile_size);
    put_u32(&mut bytes, 20, metadata.tile_border);
    put_u32(&mut bytes, 24, metadata.family_count);
    put_u32(
        &mut bytes,
        28,
        if metadata.procedural {
            FLAG_PROCEDURAL
        } else {
            0
        },
    );
    put_u64(&mut bytes, 32, metadata.virtual_width);
    put_u64(&mut bytes, 40, metadata.virtual_height);
    put_u64(&mut bytes, 48, page_count);
    put_u64(&mut bytes, 56, directory_offset);
    put_u64(&mut bytes, 64, data_offset);
    put_u64(&mut bytes, 72, metadata.procedural_seed);
    put_u32(&mut bytes, 80, plan.coarse_min_mip);
    put_u32(&mut bytes, 84, plan.detail_max_mip);
    put_u32(&mut bytes, 88, plan.detail_window_pages);
    Ok(bytes)
}

fn decode_header(bytes: &[u8; HEADER_SIZE]) -> Result<DecodedHeader, String> {
    if &bytes[0..8] != MAGIC {
        return Err("invalid forge3d VT store magic".to_string());
    }
    if get_u32(bytes, 8) != VERSION {
        return Err(format!(
            "unsupported forge3d VT store version {}",
            get_u32(bytes, 8)
        ));
    }
    if get_u32(bytes, 12) as usize != HEADER_SIZE {
        return Err("invalid forge3d VT header size".to_string());
    }
    let metadata = StoreMetadata {
        tile_size: get_u32(bytes, 16),
        tile_border: get_u32(bytes, 20),
        family_count: get_u32(bytes, 24),
        procedural: get_u32(bytes, 28) & FLAG_PROCEDURAL != 0,
        virtual_width: get_u64(bytes, 32),
        virtual_height: get_u64(bytes, 40),
        procedural_seed: get_u64(bytes, 72),
    };
    validate_metadata(&metadata)?;
    let decoded = DecodedHeader {
        metadata,
        plan: MaterializationPlan {
            coarse_min_mip: get_u32(bytes, 80),
            detail_max_mip: get_u32(bytes, 84),
            detail_window_pages: get_u32(bytes, 88),
        },
        page_count: get_u64(bytes, 48),
        directory_offset: get_u64(bytes, 56),
        data_offset: get_u64(bytes, 64),
    };
    if decoded.directory_offset < HEADER_SIZE as u64
        || decoded.data_offset < decoded.directory_offset
    {
        return Err("invalid forge3d VT store offsets".to_string());
    }
    Ok(decoded)
}

fn encode_directory_entry(entry: &DirectoryEntry) -> Result<[u8; DIRECTORY_ENTRY_SIZE], String> {
    let mut bytes = [0u8; DIRECTORY_ENTRY_SIZE];
    bytes[0] = entry.key.family;
    bytes[1] = entry.key.mip;
    bytes[2] = entry.format.tag();
    put_u32(&mut bytes, 4, entry.key.x);
    put_u32(&mut bytes, 8, entry.key.y);
    let width = u16::try_from(entry.width).map_err(|_| "VT page width exceeds u16".to_string())?;
    let height =
        u16::try_from(entry.height).map_err(|_| "VT page height exceeds u16".to_string())?;
    bytes[12..14].copy_from_slice(&width.to_le_bytes());
    bytes[14..16].copy_from_slice(&height.to_le_bytes());
    put_u64(&mut bytes, 16, entry.offset);
    put_u32(&mut bytes, 24, entry.length);
    put_u32(&mut bytes, 28, entry.raw_length);
    bytes[32..64].copy_from_slice(&entry.digest);
    Ok(bytes)
}

fn decode_directory_entry(bytes: &[u8]) -> Result<DirectoryEntry, String> {
    if bytes.len() != DIRECTORY_ENTRY_SIZE {
        return Err("invalid VT directory entry length".to_string());
    }
    Ok(DirectoryEntry {
        key: PageKey {
            family: bytes[0],
            mip: bytes[1],
            x: get_u32(bytes, 4),
            y: get_u32(bytes, 8),
        },
        format: PageFormat::from_tag(bytes[2])?,
        width: u32::from(u16::from_le_bytes(bytes[12..14].try_into().unwrap())),
        height: u32::from(u16::from_le_bytes(bytes[14..16].try_into().unwrap())),
        offset: get_u64(bytes, 16),
        length: get_u32(bytes, 24),
        raw_length: get_u32(bytes, 28),
        digest: bytes[32..64].try_into().unwrap(),
    })
}

fn validate_metadata(metadata: &StoreMetadata) -> Result<(), String> {
    if metadata.virtual_width == 0 || metadata.virtual_height == 0 {
        return Err("VT virtual dimensions must be non-zero".to_string());
    }
    if metadata.tile_size == 0 || !metadata.tile_size.is_power_of_two() {
        return Err("VT tile_size must be a non-zero power of two".to_string());
    }
    if !metadata.slot_size().is_multiple_of(4) {
        return Err("VT tile_size + 2*tile_border must be BC block aligned".to_string());
    }
    if metadata.family_count == 0 || metadata.family_count > 4 {
        return Err("VT family_count must be in 1..=4".to_string());
    }
    Ok(())
}

fn validate_page_axes(key: PageKey) -> Result<(), String> {
    if key.x >= MAX_PAGE_AXIS || key.y >= MAX_PAGE_AXIS {
        return Err(format!(
            "VT page {key:?} exceeds the {MAX_PAGE_AXIS}-page Morton addressing limit"
        ));
    }
    Ok(())
}

/// Check a physical page set against the store's recorded materialization
/// plan. Shared by `write_packed_store` (refuse to write a store that breaks
/// its own promise) and `MmapPageStore::open` (refuse to trust one).
///
/// Pages beyond the plan are allowed; missing ones are not.
fn validate_materialization(
    plan: &MaterializationPlan,
    metadata: &StoreMetadata,
    contains: impl Fn(PageKey) -> bool,
) -> Result<(), String> {
    for band in plan.bands(metadata) {
        let want = band.page_count();
        let got = band.keys().filter(|key| contains(*key)).count() as u64;
        if got != want {
            return Err(format!(
                "VT store declares min_materialized_mip {} but family {} mip {} has {got}/{want} pages",
                plan.coarse_min_mip, band.family, band.mip
            ));
        }
    }
    Ok(())
}

fn morton2(x: u32, y: u32) -> u64 {
    fn spread(value: u32) -> u64 {
        let mut value = u64::from(value & 0x0000_ffff);
        value = (value | (value << 16)) & 0x0000_ffff_0000_ffff;
        value = (value | (value << 8)) & 0x00ff_00ff_00ff_00ff;
        value = (value | (value << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        value = (value | (value << 2)) & 0x3333_3333_3333_3333;
        (value | (value << 1)) & 0x5555_5555_5555_5555
    }
    spread(x) | (spread(y) << 1)
}

#[cfg(unix)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    read_exact_with(bytes, offset, |chunk, at| file.read_at(chunk, at))
}

#[cfg(windows)]
fn read_exact_at(file: &File, bytes: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::windows::fs::FileExt;
    read_exact_with(bytes, offset, |chunk, at| file.seek_read(chunk, at))
}

fn read_exact_with(
    mut bytes: &mut [u8],
    mut offset: u64,
    mut read: impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
) -> Result<(), String> {
    while !bytes.is_empty() {
        match read(bytes, offset) {
            Ok(0) => return Err("unexpected EOF in VT store".to_string()),
            Ok(count) => {
                offset += count as u64;
                bytes = &mut bytes[count..];
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(format!("VT store positional read failed: {error}")),
        }
    }
    Ok(())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::vt::procedural_page;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Small enough to pack in well under a second, while still declaring the
    /// full 768 GiB address space and exercising both plan bands.
    const UNIT_PLAN: MaterializationPlan = MaterializationPlan {
        coarse_min_mip: 9,
        detail_max_mip: 8,
        detail_window_pages: 2,
    };

    /// Confirmed against real packer output on 2026-07-29 -- all four digests
    /// matched the analytic derivation byte for byte:
    ///
    /// ```text
    /// cargo run --release --bin forge3d-vtpack -- --procedural \
    ///   --output golden.f3dvt --manifest golden.manifest.json \
    ///   --virtual-width 262144 --virtual-height 262144 \
    ///   --tile-size 128 --tile-border 0 --seed 19 \
    ///   --coarse-min-mip 6 --detail-max-mip 5 --detail-window-pages 8
    /// ```
    ///
    /// (the 8-page detail window is what puts mip-3 page (126,129) inside the
    /// materialization plan; the run packs 5247 pages, 5247 distinct digests).
    const PROCEDURAL_PAGE_GOLDEN_SHA256: [(u8, u8, u32, u32, &str); 4] = [
        (
            0,
            6,
            0,
            0,
            "1ccda709bf6c08d7297b6e5c57d77b11fc4b3bdc7d7c3f97fbc2848f7f402ee5",
        ),
        (
            1,
            7,
            5,
            11,
            "5c103e6af812f446d5dcc6c18f5a0bdeb8b38b911b48cab2c6ce61339b1c5351",
        ),
        (
            2,
            6,
            31,
            31,
            "ba05ea25109cf9c76e710514c5f586a2aa05e1e305d9118401f4b86da4761a15",
        ),
        (
            0,
            3,
            126,
            129,
            "fdfb7ec785154e81e8c1159ae4b46c2eea9be3ffbffe69c25e26d319c60bc976",
        ),
    ];

    fn scratch(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("forge3d-{name}-{nonce}.f3dvt"))
    }

    /// The shipped procedural layout: 2^18 square, 128-texel tiles, no border.
    fn procedural_metadata() -> StoreMetadata {
        StoreMetadata {
            virtual_width: 1 << 18,
            virtual_height: 1 << 18,
            tile_size: 128,
            tile_border: 0,
            family_count: 3,
            procedural: true,
            procedural_seed: 19,
        }
    }

    fn plan_pages(metadata: &StoreMetadata, plan: &MaterializationPlan) -> Vec<PackedPage> {
        plan.keys(metadata)
            .into_iter()
            .map(|key| PackedPage {
                key,
                bytes: procedural_page(metadata, key).unwrap(),
            })
            .collect()
    }

    fn pack_plan(
        path: &Path,
        metadata: &StoreMetadata,
        plan: &MaterializationPlan,
    ) -> StoreManifest {
        write_packed_store(path, metadata, plan, plan_pages(metadata, plan)).unwrap()
    }

    /// Re-derived here on purpose: this test must not restate
    /// `procedural::page_tag` by calling it.
    fn expected_quadrants(seed: u64, key: PageKey) -> [[u8; 4]; 4] {
        const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
        fn mix(mut z: u64) -> u64 {
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        let mut state = mix(seed ^ GAMMA);
        for component in [
            u64::from(key.family),
            u64::from(key.mip),
            u64::from(key.x),
            u64::from(key.y),
        ] {
            state = mix(state ^ component.wrapping_mul(GAMMA));
        }
        std::array::from_fn(|quadrant| {
            let h = mix(state ^ (quadrant as u64 + 1).wrapping_mul(GAMMA));
            [
                (h as u8) & 0xfe,
                ((h >> 16) as u8) & 0xfe,
                ((h >> 32) as u8) & 0xfe,
                254,
            ]
        })
    }

    #[test]
    fn page_directory_round_trip_and_sha_validation() {
        let path = scratch("vt-roundtrip");
        let metadata = StoreMetadata {
            virtual_width: 4096,
            virtual_height: 4096,
            tile_size: 8,
            tile_border: 2,
            family_count: 3,
            procedural: false,
            procedural_seed: 0,
        };
        let side = metadata.slot_size();
        let raw = vec![77u8; side as usize * side as usize * 4];
        let page = PageBytes::new(
            PageFormat::Bc7Srgb,
            side,
            side,
            crate::core::compressed_textures::encode_bc7_rgba8(&raw, side, side).unwrap(),
        )
        .unwrap();
        // One hand-packed page: the store makes no completeness claim at all.
        write_packed_store(
            &path,
            &metadata,
            &MaterializationPlan::none(&metadata),
            [PackedPage {
                key: PageKey {
                    family: 0,
                    mip: 0,
                    x: 2,
                    y: 3,
                },
                bytes: page.clone(),
            }],
        )
        .unwrap();
        let store = MmapPageStore::open_with_cache(&path, 1024).unwrap();
        assert_eq!(store.metadata().virtual_width, 4096);
        assert_eq!(
            store
                .page(PageKey {
                    family: 0,
                    mip: 0,
                    x: 2,
                    y: 3
                })
                .unwrap()
                .data,
            page.data
        );
        store.verify().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn sparse_procedural_store_declares_768_gib_and_materializes_only_the_plan() {
        let path = scratch("vt-procedural");
        let metadata = procedural_metadata();
        assert_eq!(metadata.logical_texel_bytes(), 824_633_720_832);
        assert!(metadata.logical_texel_bytes() >= 256u128 * 1024 * 1024 * 1024);

        let manifest = pack_plan(&path, &metadata, &UNIT_PLAN);
        // 3 families x (mips 9..11 complete = 16 + 4 + 1, plus 9 detail
        // windows of 2x2).
        assert_eq!(manifest.page_count, 3 * (21 + 36));
        assert_eq!(manifest.pages.len(), 171);
        assert_eq!(manifest.min_materialized_mip, 9);
        assert_eq!(manifest.materialization_plan, UNIT_PLAN);
        assert_eq!(manifest.page_order, "family,mip,morton2");

        let store = MmapPageStore::open(&path).unwrap();
        assert_eq!(store.min_materialized_mip(), 9);
        assert_eq!(store.materialization_plan(), UNIT_PLAN);
        assert_eq!(store.page_count(), 171);
        // The exact lookup the pre-TESSELLA store answered by aliasing every
        // virtual key onto one canonical page per family.
        let aliased = PageKey {
            family: 1,
            mip: 4,
            x: 17,
            y: 9,
        };
        assert!(!store.contains(aliased));
        assert!(
            store.page(aliased).is_err(),
            "a page outside the materialization plan must not exist"
        );
        let materialized = PageKey {
            family: 1,
            mip: 9,
            x: 1,
            y: 2,
        };
        assert!(store.contains(materialized));
        let page = store.page(materialized).unwrap();
        assert_eq!(page.format, PageFormat::Bc5Unorm);
        assert_eq!(page.data.len() as u64, page.format.block_bytes(128, 128));
        store.verify().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn every_materialized_page_has_a_distinct_payload() {
        let path = scratch("vt-distinct");
        let metadata = procedural_metadata();
        let manifest = pack_plan(&path, &metadata, &UNIT_PLAN);
        let digests = manifest
            .pages
            .iter()
            .map(|page| page.sha256.clone())
            .collect::<HashSet<_>>();
        assert_eq!(
            digests.len(),
            manifest.pages.len(),
            "physical pages are aliasing each other"
        );
        assert_eq!(manifest.distinct_page_digests, manifest.page_count);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn page_payload_decodes_to_the_quadrant_colours_its_key_implies() {
        let metadata = procedural_metadata();
        let side = metadata.slot_size();
        let half = side / 2;
        let mut sampled = 0usize;
        for family in 0..3u8 {
            for (mip, x, y) in [
                (0u8, 1020u32, 1023u32),
                (3, 126, 129),
                (6, 0, 0),
                (6, 31, 31),
                (7, 5, 11),
                (9, 2, 3),
                (11, 0, 0),
            ] {
                let key = PageKey { family, mip, x, y };
                let page = procedural_page(&metadata, key).unwrap();
                let expected = expected_quadrants(metadata.procedural_seed, key);
                // Constant even-valued blocks are lossless through both
                // codecs, so this is exact equality, not a tolerance band.
                match page.format {
                    PageFormat::Bc7Srgb | PageFormat::Bc7Unorm => {
                        let rgba = crate::core::compressed_textures::decode_bc7_rgba8(
                            &page.data, side, side,
                        )
                        .unwrap();
                        for (quadrant, colour) in expected.iter().enumerate() {
                            let cx = (quadrant as u32 % 2) * half + half / 2;
                            let cy = (quadrant as u32 / 2) * half + half / 2;
                            let offset = ((cy * side + cx) * 4) as usize;
                            assert_eq!(
                                &rgba[offset..offset + 4],
                                colour,
                                "{key:?} quadrant {quadrant}"
                            );
                        }
                    }
                    PageFormat::Bc5Unorm => {
                        let rg = crate::core::compressed_textures::decode_bc5_rg8(
                            &page.data, side, side,
                        )
                        .unwrap();
                        for (quadrant, colour) in expected.iter().enumerate() {
                            let cx = (quadrant as u32 % 2) * half + half / 2;
                            let cy = (quadrant as u32 / 2) * half + half / 2;
                            let offset = ((cy * side + cx) * 2) as usize;
                            assert_eq!(
                                &rg[offset..offset + 2],
                                &colour[0..2],
                                "{key:?} quadrant {quadrant}"
                            );
                        }
                    }
                    other => panic!("unexpected procedural page format {other:?}"),
                }
                sampled += 1;
            }
        }
        assert_eq!(sampled, 21);

        for (family, mip, x, y, golden) in PROCEDURAL_PAGE_GOLDEN_SHA256 {
            let key = PageKey { family, mip, x, y };
            let page = procedural_page(&metadata, key).unwrap();
            assert_eq!(
                crate::core::provenance::to_hex(&page.sha256),
                golden,
                "generator drifted for {key:?}"
            );
        }
    }

    #[test]
    fn morton_order_key_is_injective_and_directory_is_sorted() {
        // `binary_search_by_key` on `order_key` is only sound if morton2 is
        // injective over the addressable page range.
        let mut seen = HashSet::new();
        for y in 0..64u32 {
            for x in 0..64u32 {
                assert!(seen.insert(morton2(x, y)), "morton2 collided at ({x},{y})");
            }
        }
        assert!(validate_page_axes(PageKey {
            family: 0,
            mip: 0,
            x: MAX_PAGE_AXIS,
            y: 0,
        })
        .is_err());

        let path = scratch("vt-sorted");
        let metadata = procedural_metadata();
        pack_plan(&path, &metadata, &UNIT_PLAN);
        let store = MmapPageStore::open(&path).unwrap();
        assert!(
            store.directory.windows(2).all(|pair| {
                MmapPageStore::order_key(pair[0].key) < MmapPageStore::order_key(pair[1].key)
            }),
            "the packed directory is not in the committed page_order"
        );
        for entry in &store.directory {
            assert!(
                store.entry(entry.key).is_some(),
                "{:?} is unreachable by binary search",
                entry.key
            );
        }

        // An out-of-order file must be rejected, not silently mis-searched.
        let mut bytes = std::fs::read(&path).unwrap();
        for offset in 0..DIRECTORY_ENTRY_SIZE {
            bytes.swap(
                HEADER_SIZE + offset,
                HEADER_SIZE + DIRECTORY_ENTRY_SIZE + offset,
            );
        }
        let swapped = scratch("vt-unsorted");
        std::fs::write(&swapped, &bytes).unwrap();
        // `unwrap_err` would require Debug on the store, which owns a File.
        let error = match MmapPageStore::open(&swapped) {
            Ok(_) => panic!("an unsorted page directory must be rejected at open"),
            Err(error) => error,
        };
        assert!(error.contains("not strictly sorted"), "{error}");
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(swapped).unwrap();
    }

    #[test]
    fn pages_at_mip_matches_the_renderer_page_grid() {
        // Mirror of `ceil_div` + `pages_for_mip_counts` in
        // `terrain/renderer/virtual_texture.rs`; that copy sits behind the
        // `extension-module` feature and cannot be called from here.
        fn renderer_pages(pages0: u32, mip: u32) -> u32 {
            let div = 1u32.checked_shl(mip).unwrap_or(u32::MAX).max(1);
            pages0.max(1).div_ceil(div).max(1)
        }
        let metadata = procedural_metadata();
        assert_eq!(metadata.mip_count(), 12);
        assert_eq!(metadata.pages_at_mip(0), (2048, 2048));
        for mip in 0..12 {
            let expected = renderer_pages(2048, mip);
            assert_eq!(
                metadata.pages_at_mip(mip),
                (expected, expected),
                "mip {mip}"
            );
        }
        // Non-square, non-power-of-two: the identity
        // ceil(ceil(a/b)/c) == ceil(a/(b*c)) has to hold there too.
        let odd = StoreMetadata {
            virtual_width: 300,
            virtual_height: 1025,
            tile_size: 128,
            tile_border: 0,
            family_count: 1,
            procedural: false,
            procedural_seed: 0,
        };
        for mip in 0..odd.mip_count() {
            assert_eq!(
                odd.pages_at_mip(mip),
                (renderer_pages(3, mip), renderer_pages(9, mip)),
                "mip {mip}"
            );
        }
    }

    #[test]
    fn truncated_materialized_mip_is_rejected_at_open() {
        let metadata = procedural_metadata();

        // Write side: a plan the page set does not keep is refused outright.
        let mut pages = plan_pages(&metadata, &UNIT_PLAN);
        pages.pop();
        let short = scratch("vt-short");
        let error = write_packed_store(&short, &metadata, &UNIT_PLAN, pages).unwrap_err();
        assert!(error.contains("pages"), "{error}");

        // Open side: a header claiming a completeness the file does not have
        // must fail instead of serving a hole.
        let path = scratch("vt-truncated");
        pack_plan(&path, &metadata, &UNIT_PLAN);
        MmapPageStore::open(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        put_u32(&mut bytes, 80, 8);
        put_u32(&mut bytes, 84, 7);
        let patched = scratch("vt-overclaim");
        std::fs::write(&patched, &bytes).unwrap();
        let error = match MmapPageStore::open(&patched) {
            Ok(_) => panic!("a store that over-claims its materialization must be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("mip 8 has 4/64 pages"), "{error}");

        let _ = std::fs::remove_file(short);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(patched).unwrap();
    }

    #[test]
    fn page_cache_evicts_the_least_recently_used_page() {
        let page =
            |value: u8| PageBytes::new(PageFormat::Rgba8Srgb, 2, 2, vec![value; 16]).unwrap();
        let key = |x: u32| PageKey {
            family: 0,
            mip: 0,
            x,
            y: 0,
        };
        let mut cache = PageCache::new(32);
        cache.insert(key(0), page(0));
        cache.insert(key(1), page(1));
        // Touch 0 so 1 becomes the least recently used entry.
        assert!(cache.get(key(0)).is_some());
        cache.insert(key(2), page(2));
        assert!(cache.get(key(1)).is_none(), "LRU evicted the wrong page");
        assert!(cache.get(key(0)).is_some());
        assert!(cache.get(key(2)).is_some());
    }

    /// TESSELLA spec item 4: an ingested image is a store like any other.
    #[test]
    fn memory_page_store_is_unbound_until_the_atlas_geometry_is_known() {
        let side = 8u32;
        let image = (0..side * side)
            .flat_map(|index| {
                let x = (index % side) as u8;
                let y = (index / side) as u8;
                [x * 16, y * 16, 255 - x * 8, 255]
            })
            .collect::<Vec<u8>>();
        let ingest = MemoryPageStore::new(&image, (side, side), 3).unwrap();
        let key = PageKey {
            family: 0,
            mip: 0,
            x: 0,
            y: 0,
        };

        // Unbound: no committed tiling means no page, loudly.
        assert_eq!(ingest.metadata().tile_size, 0);
        assert!(!ingest.contains(key));
        assert!(ingest
            .page(key)
            .unwrap_err()
            .contains("no committed tiling"));

        let bound = ingest.rebind_tile_geometry(4, 0).unwrap().unwrap();
        assert_eq!(bound.metadata().tile_size, 4);
        assert_eq!(bound.metadata().slot_size(), 4);
        // 8/4 = 2x2 pages at mip 0, 1x1 from mip 1 on.
        assert_eq!(bound.metadata().pages_at_mip(0), (2, 2));
        assert!(bound.contains(PageKey { x: 1, y: 1, ..key }));
        assert!(!bound.contains(PageKey { x: 2, y: 0, ..key }));
        // Rebinding to the same geometry is a no-op, not a copy.
        assert!(bound.rebind_tile_geometry(4, 0).unwrap().is_none());

        // Content is keyed by page identity: the four mip-0 tiles are the four
        // distinct quadrants of the source image, in source order.
        let mut digests = HashSet::new();
        for y in 0..2u32 {
            for x in 0..2u32 {
                let page = bound.page(PageKey { x, y, ..key }).unwrap();
                assert_eq!(page.format, PageFormat::Rgba8Srgb);
                assert_eq!((page.width, page.height), (4, 4));
                for row in 0..4usize {
                    let src = ((y as usize * 4 + row) * side as usize + x as usize * 4) * 4;
                    let dst = row * 4 * 4;
                    assert_eq!(&page.data[dst..dst + 16], &image[src..src + 16]);
                }
                assert!(digests.insert(page.sha256), "page ({x},{y}) aliased");
            }
        }

        // A mip past the end of the chain resolves to the repeated 1x1 tail
        // rather than becoming a counted miss: `max_mip_levels` may exceed the
        // chain depth and the page grid is 1x1 up there anyway.
        let deep = PageKey { mip: 40, ..key };
        assert!(bound.contains(deep));
        assert_eq!(bound.page(deep).unwrap().data.len(), 4 * 4 * 4);

        // A border widens the slot and clamps at the image edge.
        let bordered = ingest.rebind_tile_geometry(4, 2).unwrap().unwrap();
        assert_eq!(bordered.metadata().slot_size(), 8);
        let page = bordered.page(key).unwrap();
        assert_eq!((page.width, page.height), (8, 8));
        // Slot texel (0,0) is source (-2,-2) clamped to (0,0).
        assert_eq!(&page.data[0..4], &image[0..4]);
    }

    #[test]
    fn memory_page_store_rejects_a_buffer_that_disagrees_with_its_virtual_size() {
        let Err(error) = MemoryPageStore::new(&[0u8; 12], (4, 4), 3) else {
            panic!("a short buffer was accepted");
        };
        assert!(error.contains("expected 64 RGBA8 bytes"), "{error}");
        assert!(MemoryPageStore::new(&[], (0, 4), 3).is_err());
    }

    /// A packed file commits to its tiling; asking it for another one is a
    /// mismatch report, never a silent re-slice.
    #[test]
    fn packed_store_refuses_a_tiling_it_was_not_packed_with() {
        let path = scratch("vt-rebind");
        let metadata = procedural_metadata();
        pack_plan(&path, &metadata, &UNIT_PLAN);
        let store = MmapPageStore::open(&path).unwrap();
        assert!(store.rebind_tile_geometry(128, 0).unwrap().is_none());
        let Err(error) = store.rebind_tile_geometry(64, 0) else {
            panic!("a packed store accepted a foreign tiling");
        };
        assert!(error.contains("packed with 128-texel tiles"), "{error}");
        drop(store);
        let _ = std::fs::remove_file(&path);
    }
}
