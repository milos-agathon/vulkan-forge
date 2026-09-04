use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[cfg(feature = "extension-module")]
use super::*;

#[cfg(feature = "extension-module")]
use crate::core::feedback_buffer::FeedbackBuffer;
#[cfg(feature = "extension-module")]
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
#[cfg(feature = "enable-staging-rings")]
use crate::core::staging_rings::StagingRing;
#[cfg(feature = "extension-module")]
use crate::core::tile_cache::{TileCache, TileData, TileId};
#[cfg(feature = "extension-module")]
use crate::terrain::vt::VirtualTextureStore;
#[cfg(feature = "extension-module")]
use crate::terrain::vt_family_residency::{
    decode_family_mip_feedback, fair_family_budget_admit, logical_resident_slot_bytes,
    validate_material_atlas_capacity, validate_material_family_names,
    validate_material_residency_budget, FamilyResidency, FamilyResidencyTracker, TileKey,
    VT_FAMILY_COUNT,
};

#[cfg(feature = "extension-module")]
const TERRAIN_VT_SUPPORTED_FAMILIES: &[&str] = &["albedo", "normal", "mask"];
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_COUNT: u32 = 3;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_ALBEDO: u32 = 0;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_NORMAL: u32 = 1;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FAMILY_MASK: u32 = 2;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_BYTES_PER_PIXEL: usize = 4;
#[cfg(feature = "extension-module")]
const TERRAIN_VT_FALLBACK_COUNT: usize =
    super::core::MATERIAL_LAYER_CAPACITY * TERRAIN_VT_FAMILY_COUNT as usize;

#[cfg(feature = "extension-module")]
fn feedback_key_in_bounds(key: TileKey, material_count: u32, max_mip_levels: u32) -> bool {
    key.family_slot < TERRAIN_VT_FAMILY_COUNT
        && key.material_index < material_count
        && key.mip_level < max_mip_levels
}

#[cfg(feature = "extension-module")]
pub(super) fn bindless_bc_supported(device: &wgpu::Device) -> bool {
    let features = device.features();
    features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        && features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
        && features
            .contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
}

/// A registered VT source.
///
/// TESSELLA spec item 4: there is exactly ONE way a source can produce a page
/// -- `VirtualTextureStore::page`. An image handed in through
/// `register_source` is converted to a `MemoryPageStore` at the ingest
/// boundary; a packed file arrives as an `MmapPageStore` through `bind_store`.
/// The renderer cannot tell them apart and holds no raw image bytes.
#[derive(Clone)]
pub(super) struct VTSource {
    pub virtual_size: (u32, u32),
    store: Arc<dyn crate::terrain::vt::VirtualTextureStore>,
    pub fallback_color: [f32; 4],
    /// VERITAS: stable, device-independent source id
    /// (`family_slot * 4 + material_index + 1`; 0 == SOURCE_ID_NONE).
    pub source_id: u32,
    /// VERITAS: SHA256 of `data`, computed once at ingest.
    pub content_hash: [u8; 32],
}

#[cfg(feature = "extension-module")]
pub(super) struct TerrainVTBindingResources<'a> {
    pub atlas_views: &'a [wgpu::TextureView],
    pub page_table_view: &'a wgpu::TextureView,
    pub feedback_buffer: Option<&'a wgpu::Buffer>,
}

#[cfg(feature = "extension-module")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainVtFamilyInfoGpu {
    /// x = enabled, y = page-table layer offset, z = atlas layer,
    /// w = registered source count.
    state: [u32; 4],
    /// x/y = virtual size in texels, z = data-family atlas is linear/BC,
    /// w reserved. Two vec4s give the array the required 32-byte std140 stride.
    virtual_size: [u32; 4],
}

#[cfg(feature = "extension-module")]
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainVTUniformsGpu {
    config0: [u32; 4],
    config1: [u32; 4],
    config2: [u32; 4],
    /// Per-family info (`TerrainVtFamilyInfo`): the single source of truth the
    /// shader reads per family. x = enabled (0/1), y = page-table array-layer
    /// offset, z = atlas descriptor layer (family slot for bindless BC, zero
    /// for the shared-atlas compatibility path), w = registered source count.
    /// Matches `family_info` in `terrain_pbr_pom.wgsl`; refreshed every
    /// `prepare_frame`.
    family_info: [TerrainVtFamilyInfoGpu; TERRAIN_VT_FAMILY_COUNT as usize],
    /// Bounded feedback append (TESSELLA win 1). x = slot capacity (power of
    /// two), y/z = physical page-table base width/height, w reserved. Matches
    /// `config3` in `terrain_pbr_pom.wgsl`.
    config3: [u32; 4],
}

/// Size of the `vt_uniforms` buffer allocated in `constructor.rs`.
///
/// A uniform buffer smaller than the shader's declared struct is a bind-group
/// validation failure at pipeline time, so this is derived rather than written
/// out by hand.
#[cfg(feature = "extension-module")]
pub(crate) const VT_UNIFORM_BUFFER_BYTES: u64 = std::mem::size_of::<TerrainVTUniformsGpu>() as u64;

#[cfg(feature = "extension-module")]
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct PageTableEntry {
    atlas_u: f32,
    atlas_v: f32,
    is_resident: u32,
    mip_bias: f32,
}

#[derive(Clone)]
struct PreparedVTSource {
    fallback_color: [f32; 4],
    /// The source's store, rebound to this runtime's atlas slot geometry.
    store: Arc<dyn crate::terrain::vt::VirtualTextureStore>,
    /// VERITAS: stable source id + SHA256 of the source payload (both
    /// copied from `VTSource`, assigned at ingest).
    source_id: u32,
    content_hash: [u8; 32],
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, Default)]
struct TerrainMaterialVTStats {
    resident_pages: u32,
    total_pages: u32,
    cache_budget_pages: u32,
    cache_budget_mb: f32,
    cache_hits: u32,
    cache_misses: u32,
    tiles_streamed: u32,
    evictions: u32,
    avg_upload_ms: f32,
    last_upload_ms: f32,
    resident_megabytes: f32,
    source_count: u32,
    feedback_requests: u32,
    retained_requests: u32,
    /// Per-frame uploads whose exact page key entered admission through the
    /// retained shader-feedback path, split by material family.
    feedback_tiles_streamed_by_family: [u32; VT_FAMILY_COUNT],
    prefetch_requests: u32,
    uploaded_bytes: u64,
    upload_budget_bytes: u64,
    atlas_device_local_bytes: u64,
    atlas_uncompressed_equivalent_bytes: u64,
    atlas_device_local_bytes_by_family: [u64; VT_FAMILY_COUNT],
    atlas_uncompressed_equivalent_bytes_by_family: [u64; VT_FAMILY_COUNT],
    /// Per-frame count of requests the bound store cannot serve at any mip in
    /// the chain. A store miss is dropped, never substituted, so this counter
    /// is the only place the loss is visible -- it must be 0 for a camera whose
    /// working set is inside the store's materialization plan.
    store_page_misses: u32,
    /// Session-cumulative number of DISTINCT store page keys actually read.
    /// With per-key page content this is the anti-degeneracy signal: an
    /// aliasing store would report a handful regardless of the camera.
    store_pages_fetched_distinct: u32,
    /// The bound store's declared completeness floor (`0` when no store).
    store_min_materialized_mip: u32,
    /// Slot capacity of the bounded feedback set (TESSELLA win 1).
    feedback_capacity: u32,
    /// Surface samples the bounded feedback set could not admit in the frame
    /// just read back. Non-zero means the frame's request set was incomplete;
    /// omitted samples must be regenerated by later frames before their pages
    /// can be admitted, so this counter is the only place truncation is visible.
    feedback_overflow: u32,
    bindless_bc: bool,
    families: [FamilyResidency; VT_FAMILY_COUNT],
}

#[cfg(feature = "extension-module")]
static LAST_VT_STATS: OnceLock<Mutex<HashMap<String, f32>>> = OnceLock::new();

#[cfg(feature = "extension-module")]
pub fn latest_stats() -> HashMap<String, f32> {
    LAST_VT_STATS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map(|stats| stats.clone())
        .unwrap_or_default()
}

#[cfg(feature = "extension-module")]
fn publish_stats(stats: &HashMap<String, f32>) {
    if let Ok(mut latest) = LAST_VT_STATS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        let height = latest
            .iter()
            .filter(|(key, _)| key.ends_with("_height") || key.starts_with("height_"))
            .map(|(key, value)| (key.clone(), *value))
            .collect::<Vec<_>>();
        latest.clone_from(stats);
        latest.extend(height);
    }
}

#[cfg(feature = "extension-module")]
pub(super) fn publish_height_family_stats(
    residency_owner_id: u64,
    resident_tiles: u32,
    resident_bytes: u64,
    budget_bytes: u64,
    pending_requests: u32,
) {
    crate::core::memory_tracker::global_tracker().set_resident_family_for_owner(
        residency_owner_id,
        crate::terrain::vt::HEIGHT_FAMILY as usize,
        resident_tiles,
        resident_bytes,
    );
    if let Ok(mut latest) = LAST_VT_STATS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        latest.insert("resident_tiles_height".to_string(), resident_tiles as f32);
        latest.insert("resident_bytes_height".to_string(), resident_bytes as f32);
        latest.insert("budget_bytes_height".to_string(), budget_bytes as f32);
        latest.insert(
            "height_pending_requests".to_string(),
            pending_requests as f32,
        );
    }
}

#[cfg(feature = "extension-module")]
pub(super) fn clear_height_family_stats(residency_owner_id: u64) {
    crate::core::memory_tracker::global_tracker().clear_resident_owner(residency_owner_id);
    if let Ok(mut latest) = LAST_VT_STATS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        latest.insert("resident_tiles_height".to_string(), 0.0);
        latest.insert("resident_bytes_height".to_string(), 0.0);
        latest.insert("budget_bytes_height".to_string(), 0.0);
        latest.insert("height_pending_requests".to_string(), 0.0);
    }
}

#[cfg(feature = "extension-module")]
struct TerrainMaterialVTRuntime {
    residency_owner_id: u64,
    virtual_size: (u32, u32),
    tile_size: u32,
    tile_border: u32,
    slot_size: u32,
    atlas_size: u32,
    material_count: u32,
    max_mip_levels: u32,
    pages_x0: u32,
    pages_y0: u32,
    page_table_width: u32,
    page_table_height: u32,
    atlas_textures: Vec<TrackedTexture>,
    atlas_views: Vec<wgpu::TextureView>,
    bindless_bc: bool,
    #[cfg(feature = "enable-staging-rings")]
    staging_ring: StagingRing,
    page_table_texture: TrackedTexture,
    page_table_view: wgpu::TextureView,
    page_tables: Vec<Vec<PageTableEntry>>,
    dirty_page_table_layers: HashSet<usize>,
    sources: HashMap<(u32, u32), PreparedVTSource>,
    /// SHA-256 of the exact page payload uploaded for each resident tile.
    /// VERITAS reports this instead of the per-source directory hash, so a
    /// contributing-tile record names the bytes of THAT tile.
    resident_page_digests: HashMap<TileKey, [u8; 32]>,
    /// Distinct store keys read this session; backs
    /// `store_pages_fetched_distinct`.
    store_fetched_keys: HashSet<crate::terrain::vt::PageKey>,
    tile_cache: TileCache,
    family_residency: FamilyResidencyTracker,
    feedback_buffer: Option<FeedbackBuffer>,
    pending_feedback: crate::terrain::vt::requests::RetainedRequestSet,
    latest_feedback_uvs: Vec<[f32; 2]>,
    /// Deduplicated records emitted by the shader in the most recently drained
    /// GPU feedback frame. This excludes predictive CPU requests and test
    /// retention seeds by construction.
    latest_shader_feedback: Vec<TileKey>,
    feedback_staged: bool,
    budget_pages: u32,
    residency_budget_mb: f32,
    source_generation: u64,
    use_feedback: bool,
    /// Bounded feedback-set capacity in slots, published to the shader through
    /// `TerrainVTUniforms.config3.x`.
    feedback_capacity: u32,
    family_mask: u32,
    layer_fallbacks: [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize],
    stats: TerrainMaterialVTStats,
    last_camera_target: Option<[f32; 2]>,
    /// Persistent starting family for fair admission across frames.
    admission_cursor: usize,
}

#[cfg(feature = "extension-module")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PriorityRequest {
    key: TileKey,
    score: u64,
}

#[cfg(feature = "extension-module")]
impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score).then_with(|| {
            (
                self.key.family_slot,
                self.key.material_index,
                self.key.mip_level,
                self.key.y,
                self.key.x,
            )
                .cmp(&(
                    other.key.family_slot,
                    other.key.material_index,
                    other.key.mip_level,
                    other.key.y,
                    other.key.x,
                ))
                .reverse()
        })
    }
}

#[cfg(feature = "extension-module")]
impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "extension-module")]
fn request_score(
    key: TileKey,
    desired_mip: u32,
    feedback: bool,
    prefetch: bool,
    screen_space_error: f32,
) -> u64 {
    let family_importance = match key.family_slot {
        TERRAIN_VT_FAMILY_ALBEDO => 3,
        TERRAIN_VT_FAMILY_NORMAL => 2,
        _ => 1,
    };
    let ancestor_priority = u64::from(key.mip_level.saturating_sub(desired_mip));
    u64::from(feedback) * 1_000_000_000
        + ancestor_priority * 10_000_000
        + family_importance * 1_000_000
        + (screen_space_error.clamp(0.0, 4096.0) * 1_000.0) as u64
        + if prefetch { 0 } else { 500_000 }
}

#[cfg(feature = "extension-module")]
pub(super) struct TerrainMaterialVT {
    pub sources: HashMap<(u32, String), VTSource>,
    runtime: Option<TerrainMaterialVTRuntime>,
    source_generation: u64,
    last_stats: TerrainMaterialVTStats,
    bound_store_path: Option<String>,
}

#[cfg(feature = "extension-module")]
impl TerrainMaterialVT {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            runtime: None,
            source_generation: 0,
            last_stats: TerrainMaterialVTStats::default(),
            bound_store_path: None,
        }
    }

    pub fn register_source(
        &mut self,
        material_index: u32,
        family: String,
        virtual_size_px: (u32, u32),
        data: Vec<u8>,
        fallback_color: [f32; 4],
    ) -> Result<(), String> {
        if virtual_size_px.0 == 0 || virtual_size_px.1 == 0 {
            return Err("virtual_size_px must be > 0 in both dimensions".to_string());
        }
        let family_supported = TERRAIN_VT_SUPPORTED_FAMILIES.contains(&family.as_str());
        if !family_supported {
            log::warn!(
                "terrain material VT received unsupported family '{family}'; storing it for diagnostics but the native runtime will ignore it",
                family = family,
            );
        }
        // Every family is size-checked, including the unsupported ones the
        // warning above keeps for diagnostics: a source is now an ingested
        // store handle, and a store cannot be built from a buffer whose length
        // disagrees with its declared virtual size.
        let store = crate::terrain::vt::MemoryPageStore::new(
            &data,
            virtual_size_px,
            TERRAIN_VT_FAMILY_COUNT,
        )
        .map_err(|error| format!("VT source '{family}': {error}"))?;

        if let Some(existing) = self.sources.get(&(material_index, family.clone())) {
            if existing.virtual_size != virtual_size_px {
                return Err(format!(
                    "Virtual size mismatch: existing {:?}, new {:?}",
                    existing.virtual_size, virtual_size_px
                ));
            }
        }

        // VERITAS provenance identity: derive the stable source id from the
        // (family, material) slot so it is reproducible across devices and
        // registration orders; hash the payload once at ingest.
        let source_id = Self::family_slot(&family)
            .map_or(crate::core::provenance::SOURCE_ID_NONE, |family_slot| {
                crate::core::provenance::source_id_for(family_slot, material_index)
            });
        let content_hash = store.content_hash();
        self.sources.insert(
            (material_index, family),
            VTSource {
                virtual_size: virtual_size_px,
                store: Arc::new(store),
                fallback_color,
                source_id,
                content_hash,
            },
        );
        self.source_generation = self.source_generation.wrapping_add(1);
        self.runtime = None;
        Ok(())
    }

    fn bind_store(&mut self, path: &str) -> Result<(), String> {
        if self.bound_store_path.as_deref() == Some(path) {
            return Ok(());
        }
        let store = Arc::new(crate::terrain::vt::MmapPageStore::open(path)?);
        let metadata = store.metadata();
        let width = u32::try_from(metadata.virtual_width).map_err(|_| {
            "VT store virtual_width exceeds the renderer's u32 contract".to_string()
        })?;
        let height = u32::try_from(metadata.virtual_height).map_err(|_| {
            "VT store virtual_height exceeds the renderer's u32 contract".to_string()
        })?;
        if metadata.family_count < TERRAIN_VT_FAMILY_COUNT {
            return Err(format!(
                "VT store exposes {} families; terrain material rendering requires {TERRAIN_VT_FAMILY_COUNT}",
                metadata.family_count
            ));
        }
        self.sources.clear();
        let fallbacks = Self::default_family_fallbacks();
        // The on-disk page key is (family, mip, x, y) with no material axis,
        // so registering the store under several material indices would alias
        // every layer onto the same pages. One material layer is what the
        // format honestly supports; `prepare_frame` clamps the count to match.
        const STORE_MATERIAL_INDEX: u32 = 0;
        for (family_slot, family) in TERRAIN_VT_SUPPORTED_FAMILIES.iter().enumerate() {
            self.sources.insert(
                (STORE_MATERIAL_INDEX, (*family).to_string()),
                VTSource {
                    virtual_size: (width, height),
                    store: store.clone(),
                    fallback_color: fallbacks[family_slot],
                    source_id: crate::core::provenance::source_id_for(
                        family_slot as u32,
                        STORE_MATERIAL_INDEX,
                    ),
                    content_hash: store.content_hash(),
                },
            );
        }
        self.source_generation = self.source_generation.wrapping_add(1);
        self.runtime = None;
        self.bound_store_path = Some(path.to_string());
        Ok(())
    }

    pub fn clear_sources(&mut self) {
        self.sources.clear();
        self.runtime = None;
        self.source_generation = self.source_generation.wrapping_add(1);
        self.last_stats = TerrainMaterialVTStats::default();
        self.bound_store_path = None;
        let _ = self.get_stats();
    }

    pub fn get_stats(&self) -> HashMap<String, f32> {
        let stats = if let Some(runtime) = self.runtime.as_ref() {
            runtime.stats
        } else {
            self.last_stats
        };
        let mut out = HashMap::new();
        out.insert("resident_pages".to_string(), stats.resident_pages as f32);
        out.insert("total_pages".to_string(), stats.total_pages as f32);
        out.insert(
            "cache_budget_pages".to_string(),
            stats.cache_budget_pages as f32,
        );
        out.insert("cache_budget_mb".to_string(), stats.cache_budget_mb);
        out.insert("cache_hits".to_string(), stats.cache_hits as f32);
        out.insert("cache_misses".to_string(), stats.cache_misses as f32);
        out.insert("miss_rate".to_string(), Self::miss_rate(stats));
        out.insert("tiles_streamed".to_string(), stats.tiles_streamed as f32);
        out.insert("evictions".to_string(), stats.evictions as f32);
        out.insert("avg_upload_ms".to_string(), stats.avg_upload_ms);
        out.insert("last_upload_ms".to_string(), stats.last_upload_ms);
        out.insert("resident_megabytes".to_string(), stats.resident_megabytes);
        out.insert("source_count".to_string(), stats.source_count as f32);
        out.insert(
            "feedback_requests".to_string(),
            stats.feedback_requests as f32,
        );
        out.insert(
            "retained_requests".to_string(),
            stats.retained_requests as f32,
        );
        out.insert(
            "prefetch_requests".to_string(),
            stats.prefetch_requests as f32,
        );
        out.insert("uploaded_bytes".to_string(), stats.uploaded_bytes as f32);
        out.insert(
            "upload_budget_bytes".to_string(),
            stats.upload_budget_bytes as f32,
        );
        out.insert(
            "atlas_device_local_bytes".to_string(),
            stats.atlas_device_local_bytes as f32,
        );
        out.insert(
            "atlas_uncompressed_equivalent_bytes".to_string(),
            stats.atlas_uncompressed_equivalent_bytes as f32,
        );
        out.insert(
            "atlas_compression_ratio".to_string(),
            if stats.atlas_device_local_bytes == 0 {
                1.0
            } else {
                stats.atlas_uncompressed_equivalent_bytes as f32
                    / stats.atlas_device_local_bytes as f32
            },
        );
        out.insert(
            "store_page_misses".to_string(),
            stats.store_page_misses as f32,
        );
        out.insert(
            "store_pages_fetched_distinct".to_string(),
            stats.store_pages_fetched_distinct as f32,
        );
        out.insert(
            "feedback_capacity".to_string(),
            stats.feedback_capacity as f32,
        );
        out.insert(
            "feedback_overflow".to_string(),
            stats.feedback_overflow as f32,
        );
        out.insert(
            "store_min_materialized_mip".to_string(),
            stats.store_min_materialized_mip as f32,
        );
        out.insert(
            "bindless_bc".to_string(),
            if stats.bindless_bc { 1.0 } else { 0.0 },
        );
        let mut resident_bytes_total = 0u64;
        for (slot, name) in TERRAIN_VT_SUPPORTED_FAMILIES.iter().enumerate() {
            let family = stats.families[slot];
            let atlas_device_local = stats.atlas_device_local_bytes_by_family[slot];
            let atlas_uncompressed = stats.atlas_uncompressed_equivalent_bytes_by_family[slot];
            out.insert(
                format!("atlas_device_local_bytes_{name}"),
                atlas_device_local as f32,
            );
            out.insert(
                format!("atlas_uncompressed_equivalent_bytes_{name}"),
                atlas_uncompressed as f32,
            );
            out.insert(
                format!("atlas_compression_ratio_{name}"),
                if atlas_device_local == 0 {
                    0.0
                } else {
                    atlas_uncompressed as f32 / atlas_device_local as f32
                },
            );
            out.insert(
                format!("resident_tiles_{name}"),
                family.resident_tiles as f32,
            );
            out.insert(
                format!("feedback_tiles_streamed_{name}"),
                stats.feedback_tiles_streamed_by_family[slot] as f32,
            );
            out.insert(
                format!("resident_bytes_{name}"),
                family.resident_bytes as f32,
            );
            out.insert(format!("budget_bytes_{name}"), family.budget_bytes as f32);
            resident_bytes_total += family.resident_bytes;
        }
        out.insert(
            "resident_bytes_total".to_string(),
            resident_bytes_total as f32,
        );
        publish_stats(&out);
        out
    }

    /// TESSELLA win 6: the retained (still-unsatisfied) VT request set, as
    /// `(family_slot, material_index, mip_level, tile_x, tile_y)` records.
    ///
    /// This is the set identity the retention gate asserts on, not just its
    /// cardinality: across a not-ready feedback window the set must be
    /// preserved element for element, and a key may only leave it by becoming
    /// resident. Sorted so a Python-side comparison is stable.
    pub fn retained_request_set(&self) -> Vec<(u32, u32, u32, u32, u32)> {
        let Some(runtime) = self.runtime.as_ref() else {
            return Vec::new();
        };
        let mut keys = runtime
            .pending_feedback
            .iter()
            .flatten()
            .map(|key| {
                (
                    key.family_slot,
                    key.material_index,
                    key.mip_level,
                    key.x,
                    key.y,
                )
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys
    }

    /// Arm the win-6 retention gate: seed an unsatisfied request set, then
    /// force the feedback map not-ready for `not_ready_frames` frames.
    ///
    /// The set is ONE DISTINCT finest-mip key per registered source, walked
    /// backwards from the far corner of the mip-0 page grid, so no two seeded
    /// keys share a (family, material) slot OR a tile coordinate. A
    /// single-key probe would make "the set is preserved" indistinguishable
    /// from "the count is preserved" -- which is the whole difference the gate
    /// is there to measure. Finest mip is deliberate: the camera-driven
    /// request grid sits at a coarser mip and only ever generates COARSER
    /// ancestors, so nothing here can be satisfied incidentally while the map
    /// is held not-ready.
    pub fn force_live_retention_probe(&mut self, not_ready_frames: u32) -> Result<(), String> {
        let runtime = self
            .runtime
            .as_mut()
            .ok_or_else(|| "material VT runtime is not initialized".to_string())?;
        let mut slots = runtime.sources.keys().copied().collect::<Vec<_>>();
        if slots.is_empty() {
            return Err("material VT runtime has no source".to_string());
        }
        slots.sort_unstable();
        let mip_level = 0;
        let (pages_x, pages_y) = runtime.pages_at_mip(mip_level);
        for (index, (family_slot, material_index)) in slots.into_iter().enumerate() {
            let index = index as u32;
            let key = TileKey {
                family_slot,
                material_index,
                x: pages_x.saturating_sub(1) - (index % pages_x.max(1)),
                y: pages_y.saturating_sub(1) - ((index / pages_x.max(1)) % pages_y.max(1)),
                mip_level,
            };
            runtime.pending_feedback[family_slot as usize].insert(key);
        }
        runtime
            .feedback_buffer
            .as_ref()
            .ok_or_else(|| "material VT feedback buffer is disabled".to_string())?
            .force_not_ready_polls_for_test(not_ready_frames);
        runtime.stats.feedback_requests = runtime
            .pending_feedback
            .iter()
            .map(|bucket| bucket.len() as u32)
            .sum();
        runtime.stats.retained_requests = runtime.stats.feedback_requests;
        self.last_stats = runtime.stats;
        Ok(())
    }

    fn miss_rate(stats: TerrainMaterialVTStats) -> f32 {
        let total_requests = stats.cache_hits + stats.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            stats.cache_misses as f32 / total_requests as f32
        }
    }

    pub fn latest_feedback_uvs(&self) -> Vec<[f32; 2]> {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.latest_feedback_uvs.clone())
            .unwrap_or_default()
    }

    pub fn binding_resources(&self) -> Option<TerrainVTBindingResources<'_>> {
        self.runtime
            .as_ref()
            .map(|runtime| TerrainVTBindingResources {
                atlas_views: &runtime.atlas_views,
                page_table_view: &runtime.page_table_view,
                feedback_buffer: runtime
                    .feedback_buffer
                    .as_ref()
                    .map(|buffer| buffer.buffer()),
            })
    }

    fn family_slot(family: &str) -> Option<u32> {
        match family {
            "albedo" => Some(TERRAIN_VT_FAMILY_ALBEDO),
            "normal" => Some(TERRAIN_VT_FAMILY_NORMAL),
            "mask" => Some(TERRAIN_VT_FAMILY_MASK),
            _ => None,
        }
    }

    fn active_layers(
        vt: &crate::terrain::render_params::TerrainVTSettingsNative,
    ) -> Result<Vec<&crate::terrain::render_params::VTLayerFamilyNative>, String> {
        if !vt.enabled {
            return Ok(Vec::new());
        }
        let family_names = vt
            .layers
            .iter()
            .map(|layer| layer.family.as_str())
            .collect::<Vec<_>>();
        validate_material_family_names(&family_names)?;
        Ok(vt.layers.iter().collect())
    }

    fn compatible_layout<'a>(
        layers: &'a [&crate::terrain::render_params::VTLayerFamilyNative],
    ) -> Result<&'a crate::terrain::render_params::VTLayerFamilyNative, String> {
        let Some(first) = layers.first().copied() else {
            return Err("terrain VT requires at least one supported family".to_string());
        };
        for layer in layers.iter().copied().skip(1) {
            if layer.virtual_size != first.virtual_size
                || layer.tile_size != first.tile_size
                || layer.tile_border != first.tile_border
            {
                return Err(format!(
                    "terrain VT families must share virtual_size_px/tile_size/tile_border; '{}' has {:?}/{}.{}, '{}' has {:?}/{}.{}",
                    first.family,
                    first.virtual_size,
                    first.tile_size,
                    first.tile_border,
                    layer.family,
                    layer.virtual_size,
                    layer.tile_size,
                    layer.tile_border,
                ));
            }
        }
        Ok(first)
    }

    fn family_mask(layers: &[&crate::terrain::render_params::VTLayerFamilyNative]) -> u32 {
        layers.iter().fold(0u32, |mask, layer| {
            mask | Self::family_slot(&layer.family).map_or(0u32, |slot| 1u32 << slot)
        })
    }

    fn default_family_fallbacks() -> [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize] {
        let mut fallbacks = [[0.5, 0.5, 0.5, 1.0]; TERRAIN_VT_FAMILY_COUNT as usize];
        fallbacks[TERRAIN_VT_FAMILY_NORMAL as usize] = [0.5, 0.5, 1.0, 1.0];
        fallbacks[TERRAIN_VT_FAMILY_MASK as usize] = [1.0, 1.0, 1.0, 1.0];
        fallbacks
    }

    fn layer_fallbacks(
        layers: &[&crate::terrain::render_params::VTLayerFamilyNative],
    ) -> [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize] {
        let mut fallbacks = Self::default_family_fallbacks();
        for layer in layers {
            if let Some(slot) = Self::family_slot(&layer.family) {
                fallbacks[slot as usize] = layer.fallback;
            }
        }
        fallbacks
    }

    /// Fail missing in-memory family/material identities before any render
    /// graph resources or command encoders are created. The same check remains
    /// in `prepare_frame` as the native boundary backstop.
    pub fn validate_requested_sources(
        &self,
        vt: &crate::terrain::render_params::TerrainVTSettingsNative,
        material_count: u32,
        store_bound_by_request: bool,
    ) -> Result<(), String> {
        // Validate family identity/uniqueness before any store is opened or
        // registered. A packed store supplies the source bytes, but it must
        // not turn malformed family settings into a state-mutating failure.
        let layers = Self::active_layers(vt)?;
        if store_bound_by_request {
            // The store is opened and validated inside prepare_frame; it owns
            // one source for every supported family at material zero.
            return Ok(());
        }
        let effective_material_count =
            material_count.clamp(1, super::core::MATERIAL_LAYER_CAPACITY as u32);
        for layer in layers {
            for material_index in 0..effective_material_count {
                if !self
                    .sources
                    .contains_key(&(material_index, layer.family.clone()))
                {
                    return Err(format!(
                        "terrain VT: family '{}' requested but no source registered for material {}; refusing to render with corrupted PBR",
                        layer.family, material_index
                    ));
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        material_count: u32,
        render_width: u32,
        render_height: u32,
        vt_uniform_buffer: &wgpu::Buffer,
        vt_fallback_uniform_buffer: &wgpu::Buffer,
    ) -> Result<bool, String> {
        if let Some(path) = params.vt_store_path.as_deref() {
            self.bind_store(path)?;
        }
        let layers = Self::active_layers(&decoded.vt)?;
        if layers.is_empty() {
            self.runtime = None;
            self.last_stats = TerrainMaterialVTStats::default();
            Self::write_disabled_uniforms(
                queue.as_ref(),
                vt_uniform_buffer,
                vt_fallback_uniform_buffer,
            );
            return Ok(false);
        }

        let mut effective_material_count =
            material_count.clamp(1, super::core::MATERIAL_LAYER_CAPACITY as u32);
        if params.vt_store_path.is_some() {
            // A packed store keys its pages by family only, so extra material
            // layers could never become resident and would report themselves
            // as fallback texels. Refuse to advertise capacity that does not
            // exist rather than degrade silently.
            effective_material_count = 1;
        }
        // Every requested family/material identity must be usable by this
        // runtime. Repeat the early Python-call preflight at the native render
        // boundary so non-Python/internal callers remain fail-closed.
        self.validate_requested_sources(
            &decoded.vt,
            effective_material_count,
            params.vt_store_path.is_some(),
        )?;
        self.ensure_runtime(
            device,
            queue,
            &layers,
            effective_material_count,
            &decoded.vt,
        )?;
        let runtime = self.runtime.as_mut().unwrap();
        runtime.reset_frame_stats(decoded.vt.residency_budget_mb);

        let fallback_colors = runtime.fallback_colors();
        Self::write_uniforms(queue.as_ref(), vt_uniform_buffer, runtime, true);
        queue.write_buffer(
            vt_fallback_uniform_buffer,
            0,
            bytemuck::cast_slice(&fallback_colors),
        );

        let (requests, feedback_admissions) =
            runtime.collect_requests(params, render_width, render_height, decoded.vt.use_feedback);
        // Keep VT uploads separate from the 4K terrain draw. `queue.write_*`
        // copies are deferred until submit, so a single encoder can both trip
        // the Windows Vulkan watchdog and reuse staging-ring offsets before
        // earlier copies execute. Bound each batch to one 8 MiB ring buffer.
        let max_staging_tile_bytes = if runtime.bindless_bc {
            let blocks_per_side = runtime.slot_size.div_ceil(4) as usize;
            blocks_per_side
                .saturating_mul(blocks_per_side)
                .saturating_mul(16)
        } else {
            runtime.slot_size as usize * runtime.slot_size as usize * TERRAIN_VT_BYTES_PER_PIXEL
        };
        let upload_batch_tiles = (8 * 1024 * 1024 / max_staging_tile_bytes.max(1))
            .max(1)
            .min(512);
        for request_batch in requests.chunks(upload_batch_tiles) {
            let mut vt_upload_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain.material_vt.uploads"),
                });
            for &key in request_batch {
                let streamed = runtime.ensure_tile_resident(
                    &mut vt_upload_encoder,
                    device.as_ref(),
                    queue.as_ref(),
                    key,
                )?;
                if streamed && feedback_admissions.contains(&key) {
                    let family = key.family_slot as usize;
                    runtime.stats.feedback_tiles_streamed_by_family[family] =
                        runtime.stats.feedback_tiles_streamed_by_family[family].saturating_add(1);
                }
            }
            queue.submit(Some(vt_upload_encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }
        runtime.upload_page_tables(device.as_ref(), queue.as_ref());
        runtime.refresh_stats();
        self.last_stats = runtime.stats;

        if let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() {
            feedback_buffer.clear(encoder);
        }
        let _ = self.get_stats();

        Ok(true)
    }
    pub fn stage_feedback_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), String> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(());
        };
        let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() else {
            return Ok(());
        };
        if feedback_buffer.has_pending_readback() {
            return Ok(());
        }
        feedback_buffer.prepare_readback(encoder);
        runtime.feedback_staged = true;
        Ok(())
    }

    pub fn finish_frame(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(());
        };
        if !runtime.feedback_staged {
            return Ok(());
        }

        if let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() {
            let Some(entries) = feedback_buffer.try_read_feedback_entries(device)? else {
                runtime.pending_feedback.on_not_ready();
                runtime.stats.retained_requests = runtime
                    .pending_feedback
                    .iter()
                    .map(|bucket| bucket.len() as u32)
                    .sum();
                self.last_stats = runtime.stats;
                let _ = self.get_stats();
                return Ok(());
            };
            runtime.ingest_shader_feedback(entries);
        }
        runtime.feedback_staged = false;
        self.last_stats = runtime.stats;
        let _ = self.get_stats();
        Ok(())
    }

    /// Blocking accessor for the exact, family-specific records emitted by
    /// the last shader feedback pass. Predictive CPU requests and acceptance
    /// seeds never enter this collection.
    pub fn drain_latest_shader_feedback(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Vec<(u32, u32, u32, u32, u32)>, String> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(Vec::new());
        };
        if runtime.feedback_staged {
            if let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() {
                let entries = feedback_buffer.read_feedback_entries_blocking(device)?;
                runtime.ingest_shader_feedback(entries);
            }
            runtime.feedback_staged = false;
        }
        Ok(runtime
            .latest_shader_feedback
            .iter()
            .map(|key| {
                (
                    key.family_slot,
                    key.material_index,
                    key.mip_level,
                    key.x,
                    key.y,
                )
            })
            .collect())
    }

    /// VERITAS: resolve each unresolved shader-demand record from the latest
    /// frame to the resident mip the shader actually landed on.
    ///
    /// The GPU walk starts at the desired mip and climbs coarser until a
    /// page-table entry is resident; this replays the identical walk against
    /// the CPU page-table mirror (which was uploaded before the pass and is
    /// unchanged since), so the leaf set describes exactly the tiles the
    /// composite sampled. Feedback chains with no resident ancestor sampled
    /// the fallback color and contribute no leaf (SOURCE_ID_NONE pixels).
    pub fn read_contributing_tiles(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Vec<crate::core::provenance::ContributingTile>, String> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Ok(Vec::new());
        };
        let Some(feedback_buffer) = runtime.feedback_buffer.as_ref() else {
            return Ok(Vec::new());
        };
        // `finish_frame` normally consumes the async readback first. Reuse
        // the exact decoded snapshot it retained; otherwise a second read of
        // the already-consumed staging buffer makes an active feedback frame
        // appear empty. If the async map is still pending, finish it here and
        // populate that same snapshot.
        if runtime.feedback_staged {
            let entries = feedback_buffer.read_feedback_entries_blocking(device)?;
            runtime.ingest_shader_feedback(entries);
            runtime.feedback_staged = false;
        }

        Ok(runtime.resolve_feedback_keys_to_tiles(&runtime.latest_shader_feedback))
    }

    /// Resolve a caller-retained shader-feedback snapshot after streaming has
    /// settled. Unlike `read_contributing_tiles`, this deliberately does not
    /// drain or replace the latest-frame feedback state.
    pub fn resolve_captured_feedback_tiles(
        &mut self,
        feedback: &[(u32, u32, u32, u32, u32)],
    ) -> Vec<crate::core::provenance::ContributingTile> {
        let Some(runtime) = self.runtime.as_mut() else {
            return Vec::new();
        };
        let keys: Vec<TileKey> = feedback
            .iter()
            .map(|&(family_slot, material_index, mip_level, x, y)| TileKey {
                family_slot,
                material_index,
                mip_level,
                x,
                y,
            })
            .collect();
        runtime.resolve_feedback_keys_to_tiles(&keys)
    }

    fn write_disabled_uniforms(
        queue: &wgpu::Queue,
        vt_uniform_buffer: &wgpu::Buffer,
        vt_fallback_uniform_buffer: &wgpu::Buffer,
    ) {
        let uniforms = TerrainVTUniformsGpu {
            config0: [0, 0, 0, 0],
            config1: [0, 0, 0, 0],
            config2: [0, 0, 0, 0],
            family_info: [TerrainVtFamilyInfoGpu::zeroed(); TERRAIN_VT_FAMILY_COUNT as usize],
            config3: [0, 0, 0, 0],
        };
        let fallback_colors = TerrainMaterialVTRuntime::default_fallback_colors();
        queue.write_buffer(vt_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            vt_fallback_uniform_buffer,
            0,
            bytemuck::cast_slice(&fallback_colors),
        );
    }

    fn write_uniforms(
        queue: &wgpu::Queue,
        vt_uniform_buffer: &wgpu::Buffer,
        runtime: &TerrainMaterialVTRuntime,
        enabled: bool,
    ) {
        let mut family_info = [TerrainVtFamilyInfoGpu::zeroed(); TERRAIN_VT_FAMILY_COUNT as usize];
        for (slot, info) in family_info.iter_mut().enumerate() {
            let slot_u32 = slot as u32;
            let family_enabled = enabled && (runtime.family_mask & (1u32 << slot_u32)) != 0;
            let source_count = runtime
                .sources
                .keys()
                .filter(|(family_slot, _)| *family_slot == slot_u32)
                .count() as u32;
            info.state = [
                if family_enabled && source_count > 0 {
                    1
                } else {
                    0
                },
                slot_u32 * runtime.material_count,
                if runtime.bindless_bc { slot_u32 } else { 0 },
                source_count,
            ];
            info.virtual_size = [
                runtime.virtual_size.0,
                runtime.virtual_size.1,
                if runtime.bindless_bc && slot_u32 != TERRAIN_VT_FAMILY_ALBEDO {
                    1
                } else {
                    0
                },
                0,
            ];
        }
        let uniforms = TerrainVTUniformsGpu {
            config0: [
                if enabled { runtime.family_mask } else { 0 },
                runtime.tile_size,
                runtime.tile_border,
                runtime.atlas_size,
            ],
            config1: [
                runtime.virtual_size.0,
                runtime.virtual_size.1,
                runtime.pages_x0,
                runtime.pages_y0,
            ],
            config2: [
                runtime.max_mip_levels,
                runtime.material_count,
                runtime.slot_size,
                if runtime.use_feedback { 1 } else { 0 },
            ],
            family_info,
            config3: [
                runtime.feedback_capacity,
                runtime.page_table_width,
                runtime.page_table_height,
                0,
            ],
        };
        queue.write_buffer(vt_uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn ensure_runtime(
        &mut self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        layers: &[&crate::terrain::render_params::VTLayerFamilyNative],
        material_count: u32,
        vt: &crate::terrain::render_params::TerrainVTSettingsNative,
    ) -> Result<(), String> {
        let layer = Self::compatible_layout(layers)?;
        let family_mask = Self::family_mask(layers);
        let layer_fallbacks = Self::layer_fallbacks(layers);
        let full_levels = TerrainMaterialVTRuntime::full_pyramid_levels(
            layer.virtual_size.0,
            layer.virtual_size.1,
            layer.tile_size,
        );
        let max_mip_levels = vt.max_mip_levels.min(full_levels).max(1);

        let runtime_matches = self.runtime.as_ref().is_some_and(|runtime| {
            runtime.virtual_size == layer.virtual_size
                && runtime.tile_size == layer.tile_size
                && runtime.tile_border == layer.tile_border
                && runtime.atlas_size == vt.atlas_size
                && runtime.material_count == material_count
                && runtime.max_mip_levels == max_mip_levels
                && runtime.source_generation == self.source_generation
                && runtime.use_feedback == vt.use_feedback
                && runtime.family_mask == family_mask
                && runtime.layer_fallbacks == layer_fallbacks
                // A budget change must rebuild so the shared tile-cache
                // capacity and the per-family budgets both pick it up.
                && runtime.residency_budget_mb == vt.residency_budget_mb
        });
        if runtime_matches {
            return Ok(());
        }

        let runtime = TerrainMaterialVTRuntime::new(
            device,
            queue,
            &self.sources,
            self.source_generation,
            layers,
            layer,
            family_mask,
            layer_fallbacks,
            material_count,
            vt.atlas_size,
            vt.residency_budget_mb,
            max_mip_levels,
            vt.use_feedback,
        )?;
        self.last_stats = runtime.stats;
        self.runtime = Some(runtime);
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
impl TerrainMaterialVTRuntime {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        sources: &HashMap<(u32, String), VTSource>,
        source_generation: u64,
        layers: &[&crate::terrain::render_params::VTLayerFamilyNative],
        layer: &crate::terrain::render_params::VTLayerFamilyNative,
        family_mask: u32,
        layer_fallbacks: [[f32; 4]; TERRAIN_VT_FAMILY_COUNT as usize],
        material_count: u32,
        atlas_size: u32,
        residency_budget_mb: f32,
        max_mip_levels: u32,
        use_feedback: bool,
    ) -> Result<Self, String> {
        let slot_size = layer
            .tile_border
            .checked_mul(2)
            .and_then(|border| layer.tile_size.checked_add(border))
            .ok_or_else(|| "terrain VT tile_size + 2*tile_border overflowed u32".to_string())?;
        let atlas_slots_total =
            validate_material_atlas_capacity(atlas_size, slot_size, family_mask)?;
        let budget_bytes = validate_material_residency_budget(
            residency_budget_mb,
            slot_size,
            family_mask,
            crate::core::memory_tracker::MEMORY_BUDGET_LIMIT,
        )? as usize;
        let pages_x0 = ceil_div(layer.virtual_size.0, layer.tile_size);
        let pages_y0 = ceil_div(layer.virtual_size.1, layer.tile_size);
        let max_mip_levels = max_mip_levels
            .min(Self::page_table_mip_levels(pages_x0, pages_y0))
            .max(1);

        let bindless_bc = bindless_bc_supported(device.as_ref());
        if !bindless_bc {
            let features = device.features();
            if !features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                crate::core::degradation::record_degradation(
                    "rendering_fallback",
                    "terrain_vt_bc_atlas",
                    "adapter lacks TEXTURE_COMPRESSION_BC; using raw RGBA8 atlas uploads",
                );
            }
            if !features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
                || !features.contains(
                    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                )
            {
                crate::core::degradation::record_degradation(
                    "rendering_fallback",
                    "terrain_vt_bindless_atlas",
                    "adapter lacks descriptor indexing; using the single-atlas compatibility path",
                );
            }
        }
        if bindless_bc && (!atlas_size.is_multiple_of(4) || !slot_size.is_multiple_of(4)) {
            return Err(
                "BC atlas_size and tile_size + 2*tile_border must be multiples of four".to_string(),
            );
        }
        let atlas_formats = if bindless_bc {
            vec![
                wgpu::TextureFormat::Bc7RgbaUnormSrgb,
                wgpu::TextureFormat::Bc5RgUnorm,
                wgpu::TextureFormat::Bc7RgbaUnorm,
            ]
        } else {
            vec![wgpu::TextureFormat::Rgba8UnormSrgb]
        };
        let mut atlas_textures = Vec::with_capacity(atlas_formats.len());
        let mut atlas_views = Vec::with_capacity(atlas_formats.len());
        for format in atlas_formats {
            let texture = tracked_create_texture(
                device,
                &wgpu::TextureDescriptor {
                    label: Some("terrain.material_vt.atlas"),
                    size: wgpu::Extent3d {
                        width: atlas_size,
                        height: atlas_size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )
            .map_err(|e| e.to_string())?;
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain.material_vt.atlas.view"),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            atlas_textures.push(texture);
            atlas_views.push(view);
        }
        #[cfg(feature = "enable-staging-rings")]
        let staging_ring = {
            let max_tile_bytes =
                slot_size as u64 * slot_size as u64 * TERRAIN_VT_BYTES_PER_PIXEL as u64;
            let buffer_size = max_tile_bytes.max(8 * 1024 * 1024);
            StagingRing::new(device.clone(), queue.clone(), 3, buffer_size)
                .map_err(|e| e.to_string())?
        };

        let page_table_descriptor =
            page_table_texture_descriptor(pages_x0, pages_y0, material_count, max_mip_levels);
        let (page_table_width, page_table_height) = page_table_physical_size(pages_x0, pages_y0);
        let page_table_texture =
            tracked_create_texture(device, &page_table_descriptor).map_err(|e| e.to_string())?;
        let page_table_view = page_table_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain.material_vt.page_table.view"),
            format: Some(wgpu::TextureFormat::Rgba32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(TERRAIN_VT_FAMILY_COUNT * material_count),
            ..Default::default()
        });

        let mut prepared_sources = HashMap::new();
        for ((material_index, family), source) in sources {
            let Some(family_slot) = TerrainMaterialVT::family_slot(family) else {
                continue;
            };
            if family_mask & (1u32 << family_slot) == 0 || *material_index >= material_count {
                continue;
            };
            let Some(layer_for_family) =
                layers.iter().find(|candidate| candidate.family == *family)
            else {
                continue;
            };
            if source.virtual_size != layer_for_family.virtual_size {
                return Err(format!(
                    "VT source {:?} virtual size {:?} does not match layer contract {:?}",
                    (material_index, family),
                    source.virtual_size,
                    layer_for_family.virtual_size
                ));
            }
            // One trait, one page path: bind the source's store to THIS
            // runtime's atlas slot geometry. A packed file rejects a tiling it
            // was not packed with; an in-RAM ingest re-slices the shared mip
            // chain without copying it.
            let store = source
                .store
                .rebind_tile_geometry(layer.tile_size, layer.tile_border)
                .map_err(|error| format!("VT source {:?}: {error}", (material_index, family)))?
                .unwrap_or_else(|| source.store.clone());
            prepared_sources.insert(
                (family_slot, *material_index),
                PreparedVTSource {
                    fallback_color: source.fallback_color,
                    store,
                    source_id: source.source_id,
                    content_hash: source.content_hash,
                },
            );
        }

        let total_pages =
            Self::total_pages_for(layer.virtual_size, layer.tile_size, max_mip_levels)
                .saturating_mul(prepared_sources.len() as u32);

        let slot_bytes = logical_resident_slot_bytes(slot_size) as usize;
        let budget_pages = (budget_bytes / slot_bytes) as u32;
        let budget_pages = budget_pages.min(atlas_slots_total);
        let effective_budget_bytes = u64::from(budget_pages) * slot_bytes as u64;

        // Per-family budgets: even split of the VT residency budget across the
        // enabled families; within-family LRU eviction keeps each family under
        // its own share before the shared tile cache evicts globally.
        let family_residency =
            FamilyResidencyTracker::new(effective_budget_bytes, family_mask, slot_bytes as u64);

        let mut tile_cache = TileCache::new(budget_pages as usize);
        tile_cache.configure_atlas(atlas_size, atlas_size, slot_size);

        // TESSELLA win 1. The pyramid slot count below is what a DIRECT-MAPPED
        // feedback table would have needed -- 100,663,296 slots (1.5 GiB of
        // MAP_READ memory) for the acceptance camera's 2^18 x 2^18 virtual
        // texture, three times the whole 512 MiB host-visible budget. It is now
        // only a hint: `FeedbackBuffer` rounds it to a power of two and clamps
        // it to `FEEDBACK_MAX_SLOTS`, so the host-visible footprint is bounded
        // no matter how large the virtual texture is. Samples that do not fit
        // are counted in the set's overflow header and must be regenerated by
        // later frames before their pages can be admitted.
        let _direct_mapped_pyramid_slots = material_count
            .saturating_mul(TERRAIN_VT_FAMILY_COUNT)
            .saturating_mul(max_mip_levels)
            .saturating_mul(pages_x0)
            .saturating_mul(pages_y0)
            .max(1);
        let feedback_layout = crate::core::feedback_buffer::FeedbackLayout {
            base_pages_x: pages_x0,
            base_pages_y: pages_y0,
            max_mip_levels,
            material_count,
        };
        // Fragment feedback is an append stream, so capacity must cover
        // contention (including duplicates), not only the number of distinct
        // logical pages. Keep it at the fixed 256 KiB maximum; it remains
        // independent of virtual texture dimensions and bounded by CENSOR.
        let feedback_capacity = crate::core::feedback_buffer::FEEDBACK_MAX_SLOTS;
        let feedback_buffer = if use_feedback {
            Some(FeedbackBuffer::new(
                device,
                feedback_capacity,
                feedback_layout,
            )?)
        } else {
            None
        };

        let mut page_tables = Vec::with_capacity(
            (TERRAIN_VT_FAMILY_COUNT * material_count * max_mip_levels) as usize,
        );
        for _family_slot in 0..TERRAIN_VT_FAMILY_COUNT {
            for _material_index in 0..material_count {
                for mip_level in 0..max_mip_levels {
                    let (pages_x, pages_y) = pages_for_mip_counts(pages_x0, pages_y0, mip_level);
                    page_tables.push(vec![
                        PageTableEntry::default();
                        (pages_x * pages_y) as usize
                    ]);
                }
            }
        }

        // Upload the complete zero-filled page-table atlas on first use.
        // Deferring these writes relies on the backend's lazy WebGPU
        // initialization clear for the entire texture view; on the protected
        // NVIDIA path that first-read clear can be a large, single-frame GPU
        // operation and trip the device watchdog. Explicitly publishing the
        // non-resident entries keeps the initialization work in the ordinary
        // upload path, while subsequent frames remain layer-scoped.
        let dirty_page_table_layers = (0..page_tables.len()).collect();

        let mut runtime = Self {
            residency_owner_id: crate::core::memory_tracker::global_tracker()
                .allocate_resident_owner(),
            virtual_size: layer.virtual_size,
            tile_size: layer.tile_size,
            tile_border: layer.tile_border,
            slot_size,
            atlas_size,
            material_count,
            max_mip_levels,
            pages_x0,
            pages_y0,
            page_table_width,
            page_table_height,
            atlas_textures,
            atlas_views,
            bindless_bc,
            #[cfg(feature = "enable-staging-rings")]
            staging_ring,
            page_table_texture,
            page_table_view,
            page_tables,
            dirty_page_table_layers,
            sources: prepared_sources,
            resident_page_digests: HashMap::new(),
            store_fetched_keys: HashSet::new(),
            tile_cache,
            family_residency,
            feedback_buffer,
            feedback_capacity,
            pending_feedback: Default::default(),
            latest_feedback_uvs: Vec::new(),
            latest_shader_feedback: Vec::new(),
            feedback_staged: false,
            budget_pages,
            residency_budget_mb,
            source_generation,
            use_feedback,
            family_mask,
            layer_fallbacks,
            stats: TerrainMaterialVTStats::default(),
            last_camera_target: None,
            admission_cursor: 0,
        };
        runtime.stats.total_pages = total_pages;
        runtime.stats.cache_budget_pages = budget_pages;
        runtime.stats.cache_budget_mb = residency_budget_mb;
        runtime.stats.source_count = runtime.sources.len() as u32;
        let atlas_texels = u64::from(atlas_size) * u64::from(atlas_size);
        if bindless_bc {
            let footprints =
                crate::terrain::vt::footprint::compressed_material_atlas_footprints(atlas_texels);
            let material_families = TERRAIN_VT_FAMILY_COUNT as usize;
            runtime.stats.atlas_uncompressed_equivalent_bytes_by_family[..material_families]
                .copy_from_slice(&footprints.uncompressed);
            runtime.stats.atlas_device_local_bytes_by_family[..material_families]
                .copy_from_slice(&footprints.device_local);
            runtime.stats.atlas_uncompressed_equivalent_bytes =
                footprints.uncompressed.iter().sum();
            runtime.stats.atlas_device_local_bytes = footprints.device_local.iter().sum();
        } else {
            // The compatibility path owns one shared RGBA8 atlas. Its slots are
            // dynamically shared by all families, so a per-family attribution
            // would be invented evidence; the family fields remain zero and
            // the exact aggregate ratio is 1:1.
            runtime.stats.atlas_uncompressed_equivalent_bytes = atlas_texels * 4;
            runtime.stats.atlas_device_local_bytes = atlas_texels * 4;
        }
        runtime.stats.bindless_bc = bindless_bc;
        runtime.stats.store_min_materialized_mip = runtime
            .sources
            .values()
            .map(|source| source.store.min_materialized_mip())
            .max()
            .unwrap_or(0);
        Ok(runtime)
    }

    fn default_fallback_colors() -> [[f32; 4]; TERRAIN_VT_FALLBACK_COUNT] {
        let mut colors = [[0.5, 0.5, 0.5, 1.0]; TERRAIN_VT_FALLBACK_COUNT];
        for material_index in 0..super::core::MATERIAL_LAYER_CAPACITY {
            colors[TERRAIN_VT_FAMILY_NORMAL as usize * super::core::MATERIAL_LAYER_CAPACITY
                + material_index] = [0.5, 0.5, 1.0, 1.0];
            colors[TERRAIN_VT_FAMILY_MASK as usize * super::core::MATERIAL_LAYER_CAPACITY
                + material_index] = [1.0, 1.0, 1.0, 1.0];
        }
        colors
    }

    fn fallback_colors(&self) -> [[f32; 4]; TERRAIN_VT_FALLBACK_COUNT] {
        let mut colors = Self::default_fallback_colors();
        for material_index in 0..super::core::MATERIAL_LAYER_CAPACITY {
            colors[TERRAIN_VT_FAMILY_ALBEDO as usize * super::core::MATERIAL_LAYER_CAPACITY
                + material_index] = self.layer_fallbacks[TERRAIN_VT_FAMILY_ALBEDO as usize];
        }
        for ((family_slot, material_index), source) in &self.sources {
            if *family_slot == TERRAIN_VT_FAMILY_ALBEDO
                && (*material_index as usize) < super::core::MATERIAL_LAYER_CAPACITY
            {
                colors[*family_slot as usize * super::core::MATERIAL_LAYER_CAPACITY
                    + *material_index as usize] = source.fallback_color;
            }
        }
        colors
    }

    fn reset_frame_stats(&mut self, residency_budget_mb: f32) {
        self.stats.cache_hits = 0;
        self.stats.cache_misses = 0;
        self.stats.tiles_streamed = 0;
        self.stats.evictions = 0;
        self.stats.last_upload_ms = 0.0;
        self.stats.avg_upload_ms = 0.0;
        self.stats.cache_budget_pages = self.budget_pages;
        self.stats.cache_budget_mb = residency_budget_mb;
        self.stats.source_count = self.sources.len() as u32;
        self.stats.retained_requests = self
            .pending_feedback
            .iter()
            .map(|bucket| bucket.len() as u32)
            .sum();
        self.stats.feedback_tiles_streamed_by_family = [0; VT_FAMILY_COUNT];
        self.stats.prefetch_requests = 0;
        self.stats.store_page_misses = 0;
        self.stats.uploaded_bytes = 0;
    }

    /// Decode and retain only records the GPU shader actually emitted. Each
    /// payload names one family/material identity; unlike the historical
    /// albedo fan-out, normal and mask requests therefore retain their own UV
    /// and mip sets independently.
    fn ingest_shader_feedback(
        &mut self,
        entries: Vec<crate::core::feedback_buffer::FeedbackEntry>,
    ) {
        let mut exact = HashSet::new();
        self.latest_feedback_uvs.clear();
        for entry in entries {
            let Some((family_slot, material_index, mip_level)) = decode_family_mip_feedback(
                entry.frame_number,
                entry.mip_level,
                self.material_count,
                self.max_mip_levels,
            ) else {
                continue;
            };
            if family_slot >= TERRAIN_VT_FAMILY_COUNT
                || !self.sources.contains_key(&(family_slot, material_index))
            {
                continue;
            }
            let (pages_x, pages_y) = self.pages_at_mip(mip_level);
            if entry.tile_x >= pages_x || entry.tile_y >= pages_y {
                continue;
            }
            let key = TileKey {
                family_slot,
                material_index,
                x: entry.tile_x,
                y: entry.tile_y,
                mip_level,
            };
            exact.insert(key);
            let feedback_uv = [
                (entry.tile_x as f32 + 0.5) / pages_x.max(1) as f32,
                (entry.tile_y as f32 + 0.5) / pages_y.max(1) as f32,
            ];
            if !self.latest_feedback_uvs.contains(&feedback_uv) {
                self.latest_feedback_uvs.push(feedback_uv);
            }
            if let Some(retained) = self.resolve_feedback_key(key) {
                let cache_tile = self.encode_cache_tile(retained);
                self.pending_feedback
                    .retain_if_nonresident(retained, self.tile_cache.is_resident(&cache_tile));
            }
        }
        self.latest_shader_feedback = exact.into_iter().collect();
        self.latest_shader_feedback.sort_by_key(|key| {
            (
                key.family_slot,
                key.material_index,
                key.mip_level,
                key.y,
                key.x,
            )
        });
        self.stats.feedback_requests = self
            .pending_feedback
            .iter()
            .map(|bucket| bucket.len() as u32)
            .sum();
        self.stats.retained_requests = self.stats.feedback_requests;
    }

    fn collect_requests(
        &mut self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_width: u32,
        render_height: u32,
        use_feedback: bool,
    ) -> (Vec<TileKey>, HashSet<TileKey>) {
        let desired_mip = self.target_mip_level(params, render_width, render_height);
        let (uv_min, uv_max) = self.visible_uv_rect(params);
        let current_target = [params.cam_target[0], params.cam_target[1]];
        let velocity = self
            .last_camera_target
            .map(|previous| {
                [
                    current_target[0] - previous[0],
                    current_target[1] - previous[1],
                ]
            })
            .unwrap_or([0.0, 0.0]);
        self.last_camera_target = Some(current_target);
        let prediction_scale = params.prefetch_horizon_ms / (1000.0 / 60.0);
        let mut predicted_params = params.clone();
        predicted_params.cam_target[0] += velocity[0] * prediction_scale;
        predicted_params.cam_target[1] += velocity[1] * prediction_scale;
        let predicted_rect = self.visible_uv_rect(&predicted_params);
        let (pages_x, pages_y) = self.pages_at_mip(desired_mip);
        let start_x = ((uv_min[0] * pages_x as f32).floor() as i32).clamp(0, pages_x as i32 - 1);
        let start_y = ((uv_min[1] * pages_y as f32).floor() as i32).clamp(0, pages_y as i32 - 1);
        let end_x = ((uv_max[0] * pages_x as f32).ceil() as i32 - 1).clamp(0, pages_x as i32 - 1);
        let end_y = ((uv_max[1] * pages_y as f32).ceil() as i32 - 1).clamp(0, pages_y as i32 - 1);

        let mut priorities = HashMap::<TileKey, u64>::new();
        let mut feedback_admissions = HashSet::<TileKey>::new();
        let source_slots = self.sources.keys().copied().collect::<Vec<_>>();
        for (family_slot, material_index) in source_slots.iter().copied() {
            for y in start_y..=end_y {
                for x in start_x..=end_x {
                    let key = TileKey {
                        family_slot,
                        material_index,
                        x: x as u32,
                        y: y as u32,
                        mip_level: desired_mip,
                    };
                    let score = request_score(
                        key,
                        desired_mip,
                        false,
                        false,
                        self.page_screen_space_error(key, params),
                    );
                    let _ = self.admit_request(&mut priorities, key, score);
                }
            }
        }

        if params.prefetch_horizon_ms > 0.0 && velocity != [0.0, 0.0] {
            let requests_before_prefetch = priorities.len();
            let predicted_pages =
                self.predicted_lod_pages(&predicted_params, desired_mip, predicted_rect);
            for (family_slot, material_index) in source_slots.iter().copied() {
                for &(x, y) in &predicted_pages {
                    let key = TileKey {
                        family_slot,
                        material_index,
                        x,
                        y,
                        mip_level: desired_mip,
                    };
                    let score = request_score(
                        key,
                        desired_mip,
                        false,
                        true,
                        self.page_screen_space_error(key, params),
                    );
                    let _ = self.admit_request(&mut priorities, key, score);
                }
            }
            self.stats.prefetch_requests =
                priorities.len().saturating_sub(requests_before_prefetch) as u32;
        }

        let feedback_map_delayed = self
            .feedback_buffer
            .as_ref()
            .is_some_and(FeedbackBuffer::is_forced_not_ready_for_test);
        if use_feedback && !feedback_map_delayed {
            let feedback_requests = self
                .pending_feedback
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            for feedback in feedback_requests {
                if self
                    .sources
                    .contains_key(&(feedback.family_slot, feedback.material_index))
                {
                    // The shader derives each family's desired mip from that
                    // family's own mapping derivatives, so it can name any
                    // mip 0..max: this is the uncontrolled source of
                    // sub-plan keys.
                    let score = request_score(
                        feedback,
                        desired_mip,
                        true,
                        false,
                        self.page_screen_space_error(feedback, params),
                    );
                    if let Some(resolved) = self.admit_request(&mut priorities, feedback, score) {
                        feedback_admissions.insert(resolved);
                    }
                }
            }
        }
        if feedback_map_delayed {
            self.pending_feedback.on_not_ready();
            self.stats.retained_requests = self
                .pending_feedback
                .iter()
                .map(|bucket| bucket.len() as u32)
                .sum();
        }

        let feedback_family_mask = if use_feedback && !feedback_map_delayed {
            self.latest_shader_feedback
                .iter()
                .fold(0u32, |mask, key| mask | (1u32 << key.family_slot))
        } else {
            0
        };
        let mut heaps: [BinaryHeap<PriorityRequest>; VT_FAMILY_COUNT] =
            std::array::from_fn(|_| BinaryHeap::new());
        for (key, score) in priorities {
            heaps[key.family_slot as usize].push(PriorityRequest { key, score });
        }
        let queues = std::array::from_fn(|slot| {
            let mut ordered = VecDeque::new();
            while let Some(request) = heaps[slot].pop() {
                let cache_tile = self.encode_cache_tile(request.key);
                let bytes = if self.tile_cache.is_resident(&cache_tile) {
                    0
                } else {
                    u64::from(self.slot_size)
                        * u64::from(self.slot_size)
                        * if self.bindless_bc { 1 } else { 4 }
                };
                ordered.push_back((request.key, bytes));
            }
            ordered
        });
        let ordered = fair_family_budget_admit(
            queues,
            feedback_family_mask,
            params.vt_upload_budget_bytes,
            self.admission_cursor,
        );
        self.admission_cursor = (self.admission_cursor + 1) % VT_FAMILY_COUNT;
        self.stats.upload_budget_bytes = params.vt_upload_budget_bytes;
        (ordered, feedback_admissions)
    }

    fn page_screen_space_error(
        &self,
        key: TileKey,
        params: &crate::terrain::render_params::TerrainRenderParams,
    ) -> f32 {
        let (pages_x, pages_y) = self.pages_at_mip(key.mip_level);
        let span = params.terrain_span.max(1.0);
        let page_world = (span / pages_x.max(1) as f32).max(span / pages_y.max(1) as f32);
        let center = glam::Vec3::new(
            ((key.x as f32 + 0.5) / pages_x.max(1) as f32 - 0.5) * span,
            ((key.y as f32 + 0.5) / pages_y.max(1) as f32 - 0.5) * span,
            0.0,
        );
        let (eye, _, _) = super::TerrainScene::build_camera_matrices(params);
        let distance = eye.distance(center).max(page_world * 0.5);
        let focal_pixels = params.size_px.1.max(1) as f32
            / (2.0 * (params.fov_y_deg.to_radians() * 0.5).tan().max(1e-4));
        page_world * focal_pixels / distance
    }

    /// Select predictive pages with the same CPU reference implementation
    /// used to verify the GPU clipmap LOD selector.  The UV rectangle is only
    /// a conservative candidate generator; the predicted camera frustum makes
    /// the final request decision.
    fn predicted_lod_pages(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        mip_level: u32,
        conservative_rect: ([f32; 2], [f32; 2]),
    ) -> Vec<(u32, u32)> {
        use crate::terrain::clipmap::gpu_lod::{cpu_lod_select, GpuLodConfig, TileInfo};

        let (pages_x, pages_y) = self.pages_at_mip(mip_level);
        let (rect_min, rect_max) = conservative_rect;
        let start_x = ((rect_min[0] * pages_x as f32).floor() as i32).clamp(0, pages_x as i32 - 1);
        let start_y = ((rect_min[1] * pages_y as f32).floor() as i32).clamp(0, pages_y as i32 - 1);
        let end_x = ((rect_max[0] * pages_x as f32).ceil() as i32 - 1).clamp(0, pages_x as i32 - 1);
        let end_y = ((rect_max[1] * pages_y as f32).ceil() as i32 - 1).clamp(0, pages_y as i32 - 1);
        let span = params.terrain_span.max(1.0);
        let tiles = (start_y..=end_y)
            .flat_map(|y| {
                (start_x..=end_x).map(move |x| {
                    let min = glam::Vec2::new(
                        (x as f32 / pages_x as f32 - 0.5) * span,
                        (y as f32 / pages_y as f32 - 0.5) * span,
                    );
                    let max = glam::Vec2::new(
                        ((x + 1) as f32 / pages_x as f32 - 0.5) * span,
                        ((y + 1) as f32 / pages_y as f32 - 0.5) * span,
                    );
                    TileInfo::new(mip_level, x as u32, y as u32, min, max)
                })
            })
            .collect::<Vec<_>>();
        let (eye, view, proj) = super::TerrainScene::build_camera_matrices(params);
        cpu_lod_select(
            &tiles,
            proj * view,
            eye,
            &GpuLodConfig {
                viewport_width: params.size_px.0,
                viewport_height: params.size_px.1,
                fov_y: params.fov_y_deg.to_radians(),
                max_lod: 0,
                terrain_width: span,
                tile_size: span / pages_x.max(pages_y).max(1) as f32,
                ..Default::default()
            },
            (-span, span),
        )
        .visible_tiles
        .into_iter()
        .map(|tile| {
            let (_, x, y) = TileInfo::unpack_id(tile.tile_id);
            (x, y)
        })
        .collect()
    }

    fn ensure_tile_resident(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: TileKey,
    ) -> Result<bool, String> {
        let cache_tile = self.encode_cache_tile(key);
        if self.tile_cache.is_resident(&cache_tile) {
            self.tile_cache.access_tile(&cache_tile);
            self.family_residency.note_access(key);
            self.stats.cache_hits += 1;
            self.pending_feedback[key.family_slot as usize].remove(&key);
            return Ok(false);
        }

        let Some(source) = self
            .sources
            .get(&(key.family_slot, key.material_index))
            .cloned()
        else {
            return Ok(false);
        };

        self.stats.cache_misses += 1;
        // Enforce the family's own residency budget first: evict within-family
        // LRU tiles before touching the shared pool, so one family's paging
        // pressure never drains another family's resident set.
        while let Some(victim) = self
            .family_residency
            .eviction_victim_for_insert(key.family_slot)
        {
            let victim_tile = self.encode_cache_tile(victim);
            self.tile_cache.evict_tile(&victim_tile);
            self.family_residency.on_evict(&victim);
            self.clear_page_entry(victim);
        }
        let Some((atlas_slot, evicted)) = self.tile_cache.allocate_tile_with_evicted(cache_tile)
        else {
            return Ok(false);
        };
        for evicted_tile in evicted {
            let victim = self.decode_cache_tile(evicted_tile);
            self.family_residency.on_evict(&victim);
            self.clear_page_entry(victim);
        }

        let tile_data = self.build_tile_data(&source, key)?;
        let upload_start = Instant::now();
        self.upload_tile_to_atlas(encoder, queue, key, &tile_data, atlas_slot);
        let upload_ms = upload_start.elapsed().as_secs_f32() * 1000.0;
        self.stats.tiles_streamed += 1;
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add(tile_data.data.len() as u64);
        self.stats.last_upload_ms = upload_ms;
        let stream_count = self.stats.tiles_streamed.max(1) as f32;
        self.stats.avg_upload_ms =
            ((self.stats.avg_upload_ms * (stream_count - 1.0)) + upload_ms) / stream_count;
        self.stats.evictions = self.tile_cache.stats().evictions as u32;
        self.set_page_entry(key, atlas_slot);
        self.family_residency.on_insert(key);
        self.pending_feedback[key.family_slot as usize].remove(&key);
        let _ = device;
        Ok(true)
    }

    fn refresh_stats(&mut self) {
        self.stats.feedback_capacity = self.feedback_capacity;
        self.stats.feedback_overflow = self
            .feedback_buffer
            .as_ref()
            .map(|buffer| buffer.last_overflow())
            .unwrap_or(0);
        self.stats.resident_pages = self.tile_cache.resident_count() as u32;
        let resident_bytes =
            self.stats.resident_pages as u64 * logical_resident_slot_bytes(self.slot_size);
        self.stats.resident_megabytes = resident_bytes as f32 / (1024.0 * 1024.0);
        for slot in 0..VT_FAMILY_COUNT {
            self.stats.families[slot] = self.family_residency.family(slot as u32);
        }
        let tracker = crate::core::memory_tracker::global_tracker();
        for slot in 0..TERRAIN_VT_FAMILY_COUNT as usize {
            let family = self.family_residency.family(slot as u32);
            tracker.set_resident_family_for_owner(
                self.residency_owner_id,
                slot,
                family.resident_tiles,
                family.resident_bytes,
            );
        }
    }

    fn upload_page_tables(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut dirty_layers = self.dirty_page_table_layers.drain().collect::<Vec<_>>();
        dirty_layers.sort_unstable();

        for layer_index in dirty_layers {
            let (array_layer, _) = page_table_subresource(layer_index, self.max_mip_levels);
            let entries = &self.page_tables[layer_index];
            let mip_level = layer_index as u32 % self.max_mip_levels;
            let (pages_x, pages_y) = self.pages_at_mip(mip_level);
            let (region_x, region_y) =
                page_table_region_origin(self.page_table_width, self.page_table_height, mip_level);
            // Keep each deferred queue write below one staging-ring-sized
            // chunk. The first page-table upload covers every family and mip;
            // submitting each row chunk prevents one 64 MiB mip-0 copy from
            // sharing a submit with the first terrain draw on Windows Vulkan.
            let row_bytes = pages_x as usize * 16;
            let rows_per_chunk = (8 * 1024 * 1024 / row_bytes.max(1))
                .max(1)
                .min(pages_y as usize);
            let packed_entries = entries
                .iter()
                .map(|entry| {
                    [
                        entry.atlas_u,
                        entry.atlas_v,
                        if entry.is_resident > 0 { 1.0 } else { 0.0 },
                        entry.mip_bias,
                    ]
                })
                .collect::<Vec<_>>();
            for row_start in (0..pages_y as usize).step_by(rows_per_chunk) {
                let rows = rows_per_chunk.min(pages_y as usize - row_start);
                let entry_start = row_start * pages_x as usize;
                let entry_end = (row_start + rows) * pages_x as usize;
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.page_table_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: region_x,
                            y: region_y + row_start as u32,
                            z: array_layer,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&packed_entries[entry_start..entry_end]),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(pages_x * 16),
                        rows_per_image: Some(rows as u32),
                    },
                    wgpu::Extent3d {
                        width: pages_x,
                        height: rows as u32,
                        depth_or_array_layers: 1,
                    },
                );
                let page_table_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("terrain.material_vt.page_table_upload"),
                    });
                queue.submit(Some(page_table_encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
            }
        }
    }

    /// Fetch `key` from the source's store and normalize it to the atlas
    /// format.
    ///
    /// TESSELLA spec item 4: this is the ONLY place a VT tile's bytes come
    /// from, and there is exactly one way to get them. Whether the source was
    /// registered as an in-RAM image (`MemoryPageStore`, which slices and
    /// returns `Rgba8Srgb`) or opened from a packed file (`MmapPageStore`,
    /// which returns pre-encoded BC blocks) is invisible here; only the
    /// page's declared format decides whether it is passed through, encoded,
    /// or decoded on the way to the atlas.
    fn build_tile_data(
        &mut self,
        source: &PreparedVTSource,
        key: TileKey,
    ) -> Result<TileData, String> {
        use crate::terrain::vt::PageFormat;

        let page_key = Self::page_key(key);
        // Fatal by design: every request reaching here was resolved through
        // `store.contains`, so a failure means the store promised a page it
        // cannot produce -- corruption, not a miss.
        let page = source.store.page(page_key)?;
        if page.width != self.slot_size || page.height != self.slot_size {
            return Err(format!(
                "VT store page {:?} is {}x{}, expected {}x{}",
                key, page.width, page.height, self.slot_size, self.slot_size
            ));
        }
        // Wrong-tile detector: record the digest of the bytes THIS tile
        // received, so a contributing-tile record can be checked against the
        // manifest entry for its own key.
        self.resident_page_digests.insert(key, page.sha256);
        self.store_fetched_keys.insert(page_key);
        self.stats.store_pages_fetched_distinct = self.store_fetched_keys.len() as u32;

        let atlas_format = if self.bindless_bc {
            match key.family_slot {
                TERRAIN_VT_FAMILY_ALBEDO => PageFormat::Bc7Srgb,
                TERRAIN_VT_FAMILY_NORMAL => PageFormat::Bc5Unorm,
                TERRAIN_VT_FAMILY_MASK => PageFormat::Bc7Unorm,
                _ => {
                    return Err(format!(
                        "unsupported material VT family {}",
                        key.family_slot
                    ))
                }
            }
        } else {
            PageFormat::Rgba8Srgb
        };

        let data = if page.format == atlas_format {
            page.data
        } else {
            match (page.format, atlas_format) {
                // Packed BC pages on a compatibility adapter: decode to RGBA8.
                (PageFormat::Bc7Srgb | PageFormat::Bc7Unorm, PageFormat::Rgba8Srgb) => {
                    crate::core::compressed_textures::decode_bc7_rgba8(
                        &page.data,
                        page.width,
                        page.height,
                    )?
                }
                (PageFormat::Bc5Unorm, PageFormat::Rgba8Srgb) => {
                    let rg = crate::core::compressed_textures::decode_bc5_rg8(
                        &page.data,
                        page.width,
                        page.height,
                    )?;
                    let mut rgba =
                        Vec::with_capacity(page.width as usize * page.height as usize * 4);
                    for sample in rg.chunks_exact(2) {
                        let nx = f32::from(sample[0]) / 127.5 - 1.0;
                        let ny = f32::from(sample[1]) / 127.5 - 1.0;
                        let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
                        rgba.extend_from_slice(&[
                            sample[0],
                            sample[1],
                            ((nz * 0.5 + 0.5) * 255.0).round() as u8,
                            255,
                        ]);
                    }
                    rgba
                }
                // In-RAM ingest on a BC adapter: encode on the way to the
                // atlas, which is where the compression belongs -- the store
                // holds the source image, not an atlas format.
                (PageFormat::Rgba8Srgb, PageFormat::Bc7Srgb | PageFormat::Bc7Unorm) => {
                    crate::core::compressed_textures::encode_bc7_rgba8(
                        &page.data,
                        self.slot_size,
                        self.slot_size,
                    )?
                }
                (PageFormat::Rgba8Srgb, PageFormat::Bc5Unorm) => {
                    let rg = page
                        .data
                        .chunks_exact(4)
                        .flat_map(|rgba| [rgba[0], rgba[1]])
                        .collect::<Vec<_>>();
                    crate::core::compressed_textures::encode_bc5_rg8(
                        &rg,
                        self.slot_size,
                        self.slot_size,
                    )?
                }
                (from, to) => {
                    return Err(format!(
                        "VT store page {key:?} is {from:?}; family {} needs {to:?} and no conversion exists",
                        key.family_slot
                    ))
                }
            }
        };

        Ok(TileData {
            id: self.encode_cache_tile(key),
            data,
            width: self.slot_size,
            height: self.slot_size,
            format: atlas_format.wgpu(),
        })
    }

    fn upload_tile_to_atlas(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        key: TileKey,
        tile_data: &TileData,
        atlas_slot: crate::core::tile_cache::AtlasSlot,
    ) {
        let origin = wgpu::Origin3d {
            x: atlas_slot.atlas_x,
            y: atlas_slot.atlas_y,
            z: 0,
        };
        let atlas_texture = &self.atlas_textures[if self.bindless_bc {
            key.family_slot as usize
        } else {
            0
        }];
        #[cfg(feature = "enable-staging-rings")]
        {
            if self.bindless_bc
                && self.staging_ring.upload_compressed_texture_region(
                    encoder,
                    queue,
                    atlas_texture,
                    origin,
                    &tile_data.data,
                    tile_data.width,
                    tile_data.height,
                    4,
                    4,
                    16,
                )
            {
                return;
            }
            if !self.bindless_bc
                && self.staging_ring.upload_texture_region(
                    encoder,
                    queue,
                    atlas_texture,
                    origin,
                    &tile_data.data,
                    tile_data.width,
                    tile_data.height,
                    TERRAIN_VT_BYTES_PER_PIXEL as u32,
                )
            {
                return;
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: atlas_texture,
                mip_level: 0,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            &tile_data.data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(if self.bindless_bc {
                    tile_data.width.div_ceil(4) * 16
                } else {
                    tile_data.width * TERRAIN_VT_BYTES_PER_PIXEL as u32
                }),
                rows_per_image: Some(if self.bindless_bc {
                    tile_data.height.div_ceil(4)
                } else {
                    tile_data.height
                }),
            },
            wgpu::Extent3d {
                width: tile_data.width,
                height: tile_data.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn set_page_entry(&mut self, key: TileKey, atlas_slot: crate::core::tile_cache::AtlasSlot) {
        let layer_index = self.layer_mip_index(key.family_slot, key.material_index, key.mip_level);
        let (pages_x, _pages_y) = self.pages_at_mip(key.mip_level);
        let page_index = (key.y * pages_x + key.x) as usize;
        if let Some(entry) = self.page_tables[layer_index].get_mut(page_index) {
            entry.atlas_u = atlas_slot.atlas_u;
            entry.atlas_v = atlas_slot.atlas_v;
            entry.is_resident = 1;
            entry.mip_bias = 0.0;
            self.dirty_page_table_layers.insert(layer_index);
        }
    }

    fn clear_page_entry(&mut self, key: TileKey) {
        self.resident_page_digests.remove(&key);
        if !feedback_key_in_bounds(key, self.material_count, self.max_mip_levels) {
            return;
        }
        let layer_index = self.layer_mip_index(key.family_slot, key.material_index, key.mip_level);
        let (pages_x, pages_y) = self.pages_at_mip(key.mip_level);
        if key.x >= pages_x || key.y >= pages_y {
            return;
        }
        let page_index = (key.y * pages_x + key.x) as usize;
        if let Some(entry) = self.page_tables[layer_index].get_mut(page_index) {
            *entry = PageTableEntry::default();
            self.dirty_page_table_layers.insert(layer_index);
        }
    }

    /// VERITAS: replay the shader's residency walk on the CPU page-table
    /// mirror -- climb from `key.mip_level` toward coarser mips and return the
    /// first resident tile, or `None` when the whole chain is non-resident.
    fn resolve_resident_mip(&self, key: TileKey) -> Option<TileKey> {
        if !feedback_key_in_bounds(key, self.material_count, self.max_mip_levels) {
            return None;
        }
        let mut mip_level = key.mip_level;
        loop {
            let (pages_x, pages_y) = self.pages_at_mip(mip_level);
            let shift = mip_level - key.mip_level;
            let x = (key.x >> shift).min(pages_x.saturating_sub(1));
            let y = (key.y >> shift).min(pages_y.saturating_sub(1));
            let layer_index = self.layer_mip_index(key.family_slot, key.material_index, mip_level);
            let page_index = (y * pages_x + x) as usize;
            if let Some(entry) = self
                .page_tables
                .get(layer_index)
                .and_then(|table| table.get(page_index))
            {
                if entry.is_resident > 0 {
                    return Some(TileKey {
                        family_slot: key.family_slot,
                        material_index: key.material_index,
                        x,
                        y,
                        mip_level,
                    });
                }
            }
            if mip_level + 1 >= self.max_mip_levels {
                return None;
            }
            mip_level += 1;
        }
    }

    fn resolve_feedback_keys_to_tiles(
        &self,
        feedback: &[TileKey],
    ) -> Vec<crate::core::provenance::ContributingTile> {
        use crate::core::provenance::ContributingTile;

        let mut resolved = HashSet::new();
        for &key in feedback {
            if let Some(resident) = self.resolve_resident_mip(key) {
                resolved.insert(resident);
            }
        }

        let mut tiles = Vec::with_capacity(resolved.len());
        for key in resolved {
            let Some(source) = self.sources.get(&(key.family_slot, key.material_index)) else {
                continue;
            };
            tiles.push(ContributingTile {
                family_slot: key.family_slot,
                source_id: source.source_id,
                tile_x: key.x,
                tile_y: key.y,
                mip_level: key.mip_level,
                content_hash: self
                    .resident_page_digests
                    .get(&key)
                    .copied()
                    .unwrap_or(source.content_hash),
            });
        }
        tiles.sort_by_key(|tile| {
            (
                tile.family_slot,
                tile.source_id,
                tile.mip_level,
                tile.tile_y,
                tile.tile_x,
            )
        });
        tiles
    }

    fn page_key(key: TileKey) -> crate::terrain::vt::PageKey {
        crate::terrain::vt::PageKey {
            family: key.family_slot as u8,
            mip: key.mip_level as u8,
            x: key.x,
            y: key.y,
        }
    }

    /// The store backing this tile, or `None` when no source is registered for
    /// its (family, material) slot.
    fn store_for(&self, key: TileKey) -> Option<Arc<dyn VirtualTextureStore>> {
        Some(
            self.sources
                .get(&(key.family_slot, key.material_index))?
                .store
                .clone(),
        )
    }

    /// Climb toward coarser mips until the store physically holds the page.
    ///
    /// Landing on a coarser materialized ancestor is legitimate -- the store
    /// genuinely has no finer data there. Returning `None` is an explicit
    /// miss: the caller drops the request and counts it, so the loss is never
    /// silent and never becomes a wrong-tile read.
    fn store_resolve(&self, store: &Arc<dyn VirtualTextureStore>, key: TileKey) -> Option<TileKey> {
        let mut candidate = key;
        loop {
            if store.contains(Self::page_key(candidate)) {
                return Some(candidate);
            }
            if candidate.mip_level + 1 >= self.max_mip_levels {
                return None;
            }
            candidate = TileKey {
                mip_level: candidate.mip_level + 1,
                x: candidate.x / 2,
                y: candidate.y / 2,
                ..candidate
            };
        }
    }

    /// Resolve `key` to the finest page the bound store actually holds, or
    /// `None` after counting an explicit miss. In-RAM sources pass through.
    fn resolve_store_key(&mut self, key: TileKey) -> Option<TileKey> {
        let Some(store) = self.store_for(key) else {
            return Some(key);
        };
        match self.store_resolve(&store, key) {
            Some(resolved) => Some(resolved),
            None => {
                self.stats.store_page_misses = self.stats.store_page_misses.saturating_add(1);
                None
            }
        }
    }

    /// Feedback keys come from the GPU and can name any mip, so they are
    /// resolved before entering the retained set: a key the store cannot
    /// materialize could never be cleared by residency and would pin
    /// `retained_requests` above zero forever.
    fn resolve_feedback_key(&mut self, key: TileKey) -> Option<TileKey> {
        self.resolve_store_key(key)
    }

    /// Admit one candidate request, resolving it against the store first so a
    /// key outside the materialization plan can never reach the fatal
    /// `store.page(...)?` in `build_tile_data`.
    fn admit_request(
        &mut self,
        requests: &mut HashMap<TileKey, u64>,
        key: TileKey,
        score: u64,
    ) -> Option<TileKey> {
        let Some(resolved) = self.resolve_store_key(key) else {
            return None;
        };
        self.insert_prioritized_with_ancestors(requests, resolved, score);
        Some(resolved)
    }

    fn insert_prioritized_with_ancestors(
        &self,
        requests: &mut HashMap<TileKey, u64>,
        mut key: TileKey,
        mut score: u64,
    ) {
        // Ancestors are generated, not observed, so they can leave the
        // materialized set; skip the ones the store does not hold rather than
        // queueing a request that would abort the frame.
        let store = self.store_for(key);
        loop {
            if store
                .as_ref()
                .is_none_or(|store| store.contains(Self::page_key(key)))
            {
                requests
                    .entry(key)
                    .and_modify(|current| *current = (*current).max(score))
                    .or_insert(score);
            }
            if key.mip_level + 1 >= self.max_mip_levels {
                break;
            }
            key = TileKey {
                family_slot: key.family_slot,
                material_index: key.material_index,
                x: key.x / 2,
                y: key.y / 2,
                mip_level: key.mip_level + 1,
            };
            score = score.saturating_add(10_000_000);
        }
    }

    fn visible_uv_rect(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
    ) -> ([f32; 2], [f32; 2]) {
        if super::core::is_mesh_camera_mode(&params.camera_mode) {
            let aspect = params.size_px.0 as f32 / params.size_px.1.max(1) as f32;
            let center = [
                (params.cam_target[0] / params.terrain_span.max(1e-3)) + 0.5,
                (params.cam_target[1] / params.terrain_span.max(1e-3)) + 0.5,
            ];
            let half_height =
                params.cam_radius.max(1.0) * (params.fov_y_deg.to_radians() * 0.5).tan();
            let half_width = half_height * aspect;
            let span_u = ((half_width * 2.5) / params.terrain_span.max(1e-3)).clamp(0.05, 1.0);
            let span_v = ((half_height * 2.5) / params.terrain_span.max(1e-3)).clamp(0.05, 1.0);
            let min = [
                (center[0] - span_u * 0.5).clamp(0.0, 1.0),
                (center[1] - span_v * 0.5).clamp(0.0, 1.0),
            ];
            let max = [
                (center[0] + span_u * 0.5).clamp(0.0, 1.0),
                (center[1] + span_v * 0.5).clamp(0.0, 1.0),
            ];
            (min, max)
        } else {
            ([0.0, 0.0], [1.0, 1.0])
        }
    }

    fn target_mip_level(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_width: u32,
        render_height: u32,
    ) -> u32 {
        let (uv_min, uv_max) = self.visible_uv_rect(params);
        let uv_span_x = (uv_max[0] - uv_min[0]).max(1.0 / render_width.max(1) as f32);
        let uv_span_y = (uv_max[1] - uv_min[1]).max(1.0 / render_height.max(1) as f32);
        let texels_per_pixel_x =
            self.virtual_size.0 as f32 * uv_span_x / render_width.max(1) as f32;
        let texels_per_pixel_y =
            self.virtual_size.1 as f32 * uv_span_y / render_height.max(1) as f32;
        let texels_per_pixel = texels_per_pixel_x.max(texels_per_pixel_y).max(1.0);
        let desired = texels_per_pixel.log2().floor().max(0.0) as u32;
        desired.min(self.max_mip_levels.saturating_sub(1))
    }

    fn pages_at_mip(&self, mip_level: u32) -> (u32, u32) {
        pages_for_mip_counts(self.pages_x0, self.pages_y0, mip_level)
    }

    fn layer_mip_index(&self, family_slot: u32, material_index: u32, mip_level: u32) -> usize {
        ((family_slot * self.material_count + material_index) * self.max_mip_levels + mip_level)
            as usize
    }

    fn encode_cache_tile(&self, key: TileKey) -> TileId {
        let logical_material = key.family_slot * self.material_count.max(1) + key.material_index;
        TileId {
            x: logical_material * self.pages_x0.max(1) + key.x,
            y: key.y,
            mip_level: key.mip_level,
        }
    }

    fn decode_cache_tile(&self, tile: TileId) -> TileKey {
        let logical_material = tile.x / self.pages_x0.max(1);
        TileKey {
            family_slot: logical_material / self.material_count.max(1),
            material_index: logical_material % self.material_count.max(1),
            x: tile.x % self.pages_x0.max(1),
            y: tile.y,
            mip_level: tile.mip_level,
        }
    }

    fn total_pages_for(virtual_size: (u32, u32), tile_size: u32, max_mip_levels: u32) -> u32 {
        let pages_x0 = ceil_div(virtual_size.0, tile_size);
        let pages_y0 = ceil_div(virtual_size.1, tile_size);
        let mut total = 0u32;
        for mip_level in 0..max_mip_levels {
            let (pages_x, pages_y) = pages_for_mip_counts(pages_x0, pages_y0, mip_level);
            total = total.saturating_add(pages_x.saturating_mul(pages_y));
        }
        total
    }

    fn full_pyramid_levels(width: u32, height: u32, tile_size: u32) -> u32 {
        let pages_x = ceil_div(width, tile_size).max(1);
        let pages_y = ceil_div(height, tile_size).max(1);
        Self::page_table_mip_levels(pages_x, pages_y)
    }

    fn page_table_mip_levels(pages_x0: u32, pages_y0: u32) -> u32 {
        let mut max_dim = pages_x0.max(pages_y0).max(1);
        let mut levels = 1;
        while max_dim > 1 {
            max_dim = max_dim.div_ceil(2);
            levels += 1;
        }
        levels
    }
}

#[cfg(feature = "extension-module")]
impl Drop for TerrainMaterialVTRuntime {
    fn drop(&mut self) {
        crate::core::memory_tracker::global_tracker().clear_resident_owner(self.residency_owner_id);
    }
}

#[cfg(feature = "extension-module")]
fn ceil_div(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor.max(1)
}

#[cfg(feature = "extension-module")]
fn pages_for_mip_counts(pages_x0: u32, pages_y0: u32, mip_level: u32) -> (u32, u32) {
    let div = 1u32.checked_shl(mip_level).unwrap_or(u32::MAX).max(1);
    (
        ceil_div(pages_x0.max(1), div).max(1),
        ceil_div(pages_y0.max(1), div).max(1),
    )
}

#[cfg(feature = "extension-module")]
fn page_table_texture_descriptor(
    pages_x0: u32,
    pages_y0: u32,
    material_count: u32,
    max_mip_levels: u32,
) -> wgpu::TextureDescriptor<'static> {
    // Logical page grids ceil-halve. Store mip rectangles in a single-level
    // tail atlas instead of a mipmapped image: the protected NVIDIA Vulkan
    // path loses the device when sampling the latter. The tail row is half
    // the physical height; its mip rectangles are packed left-to-right.
    let (physical_width, physical_height) = page_table_physical_size(pages_x0, pages_y0);
    let tail_width = page_table_tail_width(physical_width, max_mip_levels);
    wgpu::TextureDescriptor {
        label: Some("terrain.material_vt.page_table"),
        size: wgpu::Extent3d {
            width: physical_width.max(tail_width),
            height: physical_height
                + if max_mip_levels > 1 {
                    physical_height.div_ceil(2)
                } else {
                    0
                },
            depth_or_array_layers: TERRAIN_VT_FAMILY_COUNT * material_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

#[cfg(feature = "extension-module")]
fn page_table_physical_size(pages_x0: u32, pages_y0: u32) -> (u32, u32) {
    (pages_x0.next_power_of_two(), pages_y0.next_power_of_two())
}

#[cfg(feature = "extension-module")]
fn page_table_tail_width(base_width: u32, max_mip_levels: u32) -> u32 {
    (1..max_mip_levels)
        .map(|mip_level| (base_width >> mip_level.min(31)).max(1))
        .fold(0u32, u32::saturating_add)
        .max(1)
}

#[cfg(feature = "extension-module")]
fn page_table_region_origin(base_width: u32, base_height: u32, mip_level: u32) -> (u32, u32) {
    if mip_level == 0 {
        return (0, 0);
    }
    let tail_x = (1..mip_level)
        .map(|prior_mip| (base_width >> prior_mip.min(31)).max(1))
        .fold(0u32, u32::saturating_add);
    (tail_x, base_height)
}

#[cfg(feature = "extension-module")]
fn page_table_subresource(layer_index: usize, max_mip_levels: u32) -> (u32, u32) {
    let max_mip_levels = max_mip_levels.max(1);
    ((layer_index as u32) / max_mip_levels, 0)
}

#[cfg(feature = "extension-module")]
impl TerrainScene {
    pub(super) fn validate_material_vt_request(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        material_count: u32,
    ) -> Result<()> {
        let material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .validate_requested_sources(
                &params.decoded().vt,
                material_count,
                params.vt_store_path.is_some(),
            )
            .map_err(anyhow::Error::msg)
    }

    pub(super) fn prepare_material_vt_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        material_count: u32,
        render_width: u32,
        render_height: u32,
    ) -> Result<bool> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .prepare_frame(
                encoder,
                &self.device,
                &self.queue,
                params,
                decoded,
                material_count,
                render_width,
                render_height,
                &self.vt_uniform_buffer,
                &self.vt_fallback_uniform_buffer,
            )
            .map_err(anyhow::Error::msg)
    }
    pub(super) fn stage_material_vt_feedback_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .stage_feedback_readback(encoder)
            .map_err(anyhow::Error::msg)
    }

    pub(super) fn finish_material_vt_frame(&self) -> Result<()> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .finish_frame(self.device.as_ref(), self.queue.as_ref())
            .map_err(anyhow::Error::msg)
    }

    /// VERITAS: blocking drain of the VT feedback stream resolved to the
    /// resident tiles the last frame actually sampled.
    pub(super) fn read_material_vt_contributing_tiles(
        &self,
    ) -> Result<Vec<crate::core::provenance::ContributingTile>> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .read_contributing_tiles(self.device.as_ref())
            .map_err(anyhow::Error::msg)
    }

    /// Resolve a retained raw shader-feedback snapshot against the current
    /// resident page table without altering latest-frame feedback semantics.
    pub(super) fn resolve_captured_material_vt_feedback(
        &self,
        feedback: &[(u32, u32, u32, u32, u32)],
    ) -> Result<Vec<crate::core::provenance::ContributingTile>> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        Ok(material_vt.resolve_captured_feedback_tiles(feedback))
    }

    pub(super) fn drain_latest_material_vt_shader_feedback(
        &self,
    ) -> Result<Vec<(u32, u32, u32, u32, u32)>> {
        let mut material_vt = self
            .material_vt
            .lock()
            .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
        material_vt
            .drain_latest_shader_feedback(self.device.as_ref())
            .map_err(anyhow::Error::msg)
    }
}

#[cfg(not(feature = "extension-module"))]
pub(super) struct TerrainMaterialVT;

#[cfg(not(feature = "extension-module"))]
impl TerrainMaterialVT {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(all(test, feature = "extension-module"))]
mod bounded_feedback_tests {
    use super::*;

    /// The `vt_uniforms` allocation in `constructor.rs` must cover the struct
    /// the shader declares. It was a hand-written `96` until `config3` was
    /// added for the bounded feedback set.
    #[test]
    fn vt_uniform_buffer_covers_gpu_struct() {
        assert_eq!(std::mem::size_of::<TerrainVtFamilyInfoGpu>(), 32);
        assert_eq!(std::mem::align_of::<TerrainVtFamilyInfoGpu>(), 16);
        assert_eq!(VT_UNIFORM_BUFFER_BYTES, 160);
        assert_eq!(
            VT_UNIFORM_BUFFER_BYTES as usize,
            std::mem::size_of::<TerrainVTUniformsGpu>()
        );
        assert_eq!(VT_UNIFORM_BUFFER_BYTES % 16, 0, "std140 vec4 alignment");
    }

    #[test]
    fn captured_feedback_keys_fail_closed_outside_runtime_bounds() {
        let valid = TileKey {
            family_slot: 2,
            material_index: 3,
            mip_level: 7,
            x: u32::MAX,
            y: u32::MAX,
        };
        assert!(feedback_key_in_bounds(valid, 4, 8));
        assert!(!feedback_key_in_bounds(
            TileKey {
                family_slot: TERRAIN_VT_FAMILY_COUNT,
                ..valid
            },
            4,
            8
        ));
        assert!(!feedback_key_in_bounds(
            TileKey {
                material_index: 4,
                ..valid
            },
            4,
            8
        ));
        assert!(!feedback_key_in_bounds(
            TileKey {
                mip_level: u32::MAX,
                ..valid
            },
            4,
            8
        ));
    }

    /// Win 1's load-bearing claim: the host-visible feedback allocation does not
    /// grow with the virtual texture. The acceptance store is 2^18 x 2^18 with
    /// 128 px tiles and 8 mips over 3 families, which a direct-mapped table
    /// would have sized at 100,663,296 slots x 16 B = 1.5 GiB.
    #[test]
    fn acceptance_camera_feedback_footprint_is_bounded() {
        let pages_x0: u32 = (1 << 18) / 128;
        let pages_y0: u32 = (1 << 18) / 128;
        let pyramid_slots = 1u32
            .saturating_mul(TERRAIN_VT_FAMILY_COUNT)
            .saturating_mul(8)
            .saturating_mul(pages_x0)
            .saturating_mul(pages_y0);
        assert_eq!(pyramid_slots, 100_663_296);
        assert_eq!(pyramid_slots as u64 * 16, 1_610_612_736);

        let capacity = FeedbackBuffer::capacity_for(pyramid_slots);
        assert_eq!(capacity, crate::core::feedback_buffer::FEEDBACK_MAX_SLOTS);
        let host_visible_bytes = (capacity as u64 + 2) * 4;
        assert_eq!(host_visible_bytes, 262_152);
        assert!(host_visible_bytes < 512 * 1024 * 1024);
    }

    #[test]
    fn acceptance_page_table_packs_mips_without_a_mipmapped_image() {
        // The packed acceptance store uses one shared logical material.
        let descriptor = page_table_texture_descriptor(2048, 2048, 1, 8);
        assert_eq!(descriptor.size.depth_or_array_layers, 3);
        assert_eq!(descriptor.mip_level_count, 1);
        assert_eq!(descriptor.size.width, 2048);
        assert_eq!(descriptor.size.height, 3072);
        assert_eq!(page_table_subresource(0, 8), (0, 0));
        assert_eq!(page_table_subresource(7, 8), (0, 0));
        assert_eq!(page_table_subresource(8, 8), (1, 0));
        assert_eq!(page_table_region_origin(2048, 2048, 0), (0, 0));
        assert_eq!(page_table_region_origin(2048, 2048, 1), (0, 2048));
        assert_eq!(page_table_region_origin(2048, 2048, 2), (1024, 2048));
        assert_eq!(page_table_region_origin(2048, 2048, 7), (2016, 2048));

        let bytes = crate::core::resource_tracker::calculate_texture_descriptor_size(&descriptor);
        assert_eq!(bytes, 301_989_888);
        assert!(bytes < 512 * 1024 * 1024);
    }

    #[test]
    fn odd_logical_page_grids_fit_every_physical_mip() {
        let descriptor = page_table_texture_descriptor(13, 9, 2, 4);
        assert_eq!(descriptor.size.width, 16);
        assert_eq!(descriptor.size.height, 24);
        assert_eq!(descriptor.size.depth_or_array_layers, 6);

        for mip_level in 0..4 {
            let (logical_width, logical_height) = pages_for_mip_counts(13, 9, mip_level);
            let (origin_x, origin_y) = page_table_region_origin(16, 16, mip_level);
            assert!(origin_x + logical_width <= descriptor.size.width);
            assert!(origin_y + logical_height <= descriptor.size.height);
        }
        assert_eq!(page_table_texture_descriptor(1, 1, 1, 1).size.height, 1);

        let tall = page_table_texture_descriptor(1, 33, 1, 6);
        assert_eq!(tall.size.width, 5);
        assert_eq!(tall.size.height, 96);
        for mip_level in 0..6 {
            let (logical_width, logical_height) = pages_for_mip_counts(1, 33, mip_level);
            let (origin_x, origin_y) = page_table_region_origin(1, 64, mip_level);
            assert!(origin_x + logical_width <= tall.size.width);
            assert!(origin_y + logical_height <= tall.size.height);
        }
    }

    #[test]
    fn non_power_of_two_page_grids_keep_the_coarsest_mip() {
        assert_eq!(TerrainMaterialVTRuntime::page_table_mip_levels(13, 9), 5);
        assert_eq!(TerrainMaterialVTRuntime::page_table_mip_levels(16, 1), 5);
        assert_eq!(TerrainMaterialVTRuntime::page_table_mip_levels(1, 1), 1);
    }
}
