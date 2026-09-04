//! Core engine modules
//!
//! Contains foundational types and systems for the renderer.

// Foundational modules
pub mod context;
#[allow(dead_code)] // Removed once DUPLA Tasks 2-3 wire the arithmetic into GPU paths.
pub(crate) mod dd;
#[cfg(test)]
mod dd_tests;
pub mod device_caps;
pub mod error;
pub mod gpu;
#[cfg(feature = "extension-module")]
pub mod session;

// CENSOR: global degradation sink (fallback/placeholder/absent-capability records)
pub mod degradation;

// CENSOR: negotiated GPU capability set (replaces Features::empty())
pub mod capabilities;

// AETHER: spectral precomputed atmospheric transport.
pub mod atmosphere;

// CENSOR: WGSL shader-hash registry + validation error scopes
pub mod shader_registry;

// PROBATUM: feature-gated runtime shader-contract observations
pub(crate) mod shader_contract_runtime;

// CENSOR Task 9: RenderCertificate execution report (per-render pass timings)
pub mod certificate;

pub mod gpu_types;
pub mod memory_tracker;
pub mod resource_tracker;

// Q3: GPU profiling and timing
pub mod gpu_timing;

// Q1: Post-processing compute pipeline

// Q5: Bloom post-processing effect
pub mod bloom;

// Workstream O: Resource & Memory Management
// staging_rings and fence_tracker are declared below with feature flags
pub mod compressed_textures; // O3: Compressed texture pipeline
pub mod feedback_buffer; // O4: GPU feedback buffer for tile visibility
pub mod provenance; // VERITAS: per-pixel source provenance commitment
pub mod texture_format; // O3: Texture format registry and detection
pub mod texture_format_defs; // O3: Texture format definitions
pub mod tile_cache;

// New framegraph implementation
pub mod framegraph_impl;

// C8: Tonemap post-processing
pub mod tonemap;

// C9: Matrix stack utility
pub mod matrix_stack;

// P1.2: TAA jitter sequence
pub mod jitter;

// P1.3: Temporal Anti-Aliasing
pub mod taa;
pub(crate) mod temporal_history;

// C10: Hierarchical scene graph
pub mod scene_graph;

// C6: Multi-thread command recording
pub mod multi_thread;

// C7: Async compute prepasses
pub mod anamnesis;
pub mod async_compute;

// R9: Async and double-buffered readback system (opt-in)
#[cfg(feature = "async_readback")]
pub mod async_readback;

// I7: Big buffer pattern for per-object data
#[cfg(feature = "wsI_bigbuf")]
pub mod big_buffer;

// I8: Double-buffering for per-frame data
#[cfg(any(feature = "wsI_bigbuf", feature = "wsI_double_buf"))]
pub mod double_buffer;

// L4: Mipmap generation utilities
pub mod mipmap;

// L5: Sampler mode matrix and policy utilities
pub mod sampler_modes;

// L6: Texture upload helpers for HDR formats
pub mod texture_upload;

// N5: Environment mapping and IBL
pub mod envmap;

// N8: HDR rendering and tone mapping
pub mod hdr;
mod hdr_readback;
mod hdr_tonemapping;
mod hdr_types;

// N1: PBR materials
pub mod material;
pub mod pbr;

// N2: Shadow mapping
pub mod cascade_split;
pub mod shadow_mapping;
pub mod shadows;

// B5: Planar reflections
pub mod reflections;
mod reflections_math;
mod reflections_types;

// B6: Depth of Field
pub mod dof;

// B7: Cloud Shadows
pub mod cloud_shadows;

// B8: Realtime Clouds
pub mod clouds;

// B10: Ground Plane (Raster)
pub mod ground_plane;

// B11: Water Surface Color Toggle
pub mod water_surface;

// B12: Soft Light Radius (Raster)
pub mod soft_light_radius;

// B13: Point & Spot Lights (Realtime)
pub mod point_spot_lights;

// B14: Rect Area Lights (LTC)
pub mod ltc_area_lights;
mod ltc_lut;
mod ltc_types;

// B15: Image-Based Lighting (IBL) Polish
pub mod ibl;

// B16: Dual-source blending OIT
pub mod dual_source_oit;

// D: Overlays compositor (native)
pub mod overlays;

// D: Text overlay (native)
pub mod text_overlay;

// Overlay layer for terrain rendering
#[cfg(feature = "extension-module")]
pub mod overlay_layer;

// D11: 3D Text Mesh (native)
pub mod text_mesh;

// O1: Staging buffer rings with fence synchronization
#[cfg(feature = "enable-staging-rings")]
pub mod fence_tracker;
#[cfg(feature = "enable-staging-rings")]
pub mod staging_rings;

// O2: Memory pools are integrated into memory_tracker
// Available when enable-memory-pools feature is enabled

// O3: Compressed texture pipeline (already declared above)

// P5: Screen-space effects (SSAO/GTAO, SSGI, SSR)
pub mod gbuffer;
pub mod screen_space_effects;
