// A2-BEGIN:terrain-module
// T11-BEGIN:terrain-mesh-mod
pub mod mesh;
pub use mesh::{make_grid, GridMesh, GridVertex, Indices};
// T11-END:terrain-mesh-mod

// T33-BEGIN:terrain-mod
pub mod pipeline;
pub use pipeline::TerrainPipeline;
// T33-END:terrain-mod

// TerrainScene: reusable GPU terrain scene (M2)
pub mod scene;
#[cfg(feature = "extension-module")]
pub use scene::TerrainScene;
pub mod scatter;

// E1/E3: streaming mosaics
pub mod stream;
// E1 (scaffolding): GPU page table for tile->slot mapping
pub mod page_table;
pub use page_table::PageTable;

// P3: Cloud Optimized GeoTIFF streaming
#[cfg(feature = "cog_streaming")]
pub mod cog;
#[cfg(feature = "cog_streaming")]
pub use cog::{CogCacheStats, CogError, CogHeightReader, CogTileCache};

// B11-BEGIN:tiling-mod
pub mod tiling;
pub use tiling::{
    CacheStats, Frustum, QuadTreeNode, TileBounds, TileCache, TileData, TileId, TilingSystem,
};
// B11-END:tiling-mod

// B12-BEGIN:lod-mod
pub mod lod;
pub use lod::{
    calculate_triangle_reduction, screen_space_error, select_lod_for_tile, LodConfig,
    ScreenSpaceError,
};
// B12-END:lod-mod

// P2.1/M5: Clipmap terrain system for true scalability
pub mod clipmap;
pub mod culling;
pub mod vt;
pub use clipmap::{
    make_center_block, make_ring, make_ring_skirts, ClipmapConfig, ClipmapLevel, ClipmapMesh,
    ClipmapStreamer, ClipmapVertex,
};

// B13/B14-BEGIN:analysis-mod
pub mod analysis;
pub use analysis::{
    contour_extract, slope_aspect_compute, ContourPolyline, ContourResult, SlopeAspect,
};
// B13/B14-END:analysis-mod

// M1: Accumulation AA infrastructure for offline rendering
pub mod accumulation;
pub use accumulation::{AccumulationBuffer, AccumulationConfig, JitterSequence};

// M2: Bloom post-processing for terrain offline rendering
pub mod bloom_processor;
pub use bloom_processor::{TerrainBloomConfig, TerrainBloomProcessor};

// Terrain camera helpers (orbit camera, view-proj)
pub mod camera;
pub use camera::{build_view_proj, orbit_camera};

// DEM elevation statistics (min/max, percentile)
pub mod stats;
pub use stats::min_max;

// PyO3 terrain render parameter wrapper
#[cfg(feature = "extension-module")]
pub mod render_params;
#[cfg(feature = "extension-module")]
pub use render_params::{AddressModeNative, FilterModeNative, TerrainRenderParams};

// TerrainRenderer - GPU pipeline for PBR+POM terrain rendering
mod hosek_rgb_data;
pub(crate) mod hosek_sky;

#[cfg(feature = "extension-module")]
pub mod renderer;
#[cfg(feature = "extension-module")]
pub use renderer::{TerrainRenderer, TerrainScene as TerrainRendererScene};

// Per-family VT residency accounting; device-free so its unit tests run
// without the extension-module feature.
pub(crate) mod vt_family_residency;

pub mod probes;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::num::NonZeroU32;

// T33-BEGIN:colormap-imports
use crate::colormap::{
    decode_png_rgba8, map_name_to_type, to_linear_u8_rgba, ColormapType, SUPPORTED,
};
// T33-END:colormap-imports

// B15-BEGIN:memory-integration
use crate::core::memory_tracker::global_tracker;
// B15-END:memory-integration
mod colormap_lut;
mod globals;
mod helpers;
mod lights;
mod spike;
mod uniforms;

pub use self::colormap_lut::ColormapLUT;
pub use self::globals::Globals;
pub use self::lights::{PointLight, SpotLight};
pub use self::spike::TerrainSpike;
pub use self::uniforms::TerrainUniforms;
// E2: Per-tile uniforms (matches WGSL TileUniforms)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TileUniformsCPU {
    world_remap: [f32; 4], // (scale_x, scale_y, offset_x, offset_y)
}

// E1b: Tile slot UBO and mosaic params UBO (match WGSL TileSlot, MosaicParams)
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TileSlotCPU {
    lod: u32,
    x: u32,
    y: u32,
    slot: u32,
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MosaicParamsCPU {
    inv_tiles_x: f32,
    inv_tiles_y: f32,
    tiles_x: u32,
    tiles_y: u32,
}
// A2-END:terrain-module
