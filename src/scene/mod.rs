// T41-BEGIN:scene-module
#![allow(deprecated)]
mod core;
mod postfx_cpu;
mod private_impl;
#[cfg(feature = "extension-module")]
mod py_api;
mod render_paths;
mod ssao;
mod stats;
mod texture_helpers;
mod types;

use self::ssao::SsaoResources;
#[cfg(feature = "enable-gpu-instancing")]
use self::types::InstancedBatch;
pub use self::types::SceneGlobals;
use self::types::Text3DInstance;
#[cfg(feature = "extension-module")]
use crate::core::device_caps::DeviceCaps;
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "extension-module")]
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
#[cfg(feature = "extension-module")]
use pyo3::{prelude::*, types::PyBytes};
#[cfg(feature = "extension-module")]
use std::path::PathBuf;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
// Keep in sync with src/shaders/ssao.wgsl SsaoSettings.ao_min default comment.
const SSAO_AO_MIN_DEFAULT: f32 = 0.35;

#[cfg(feature = "extension-module")]
#[derive(Debug)]
struct SceneAtmosphereState {
    config: crate::core::atmosphere::AtmosphereConfig,
    precomputed_turbidity_bracket: Option<[f32; 2]>,
    byte_size: u64,
    deterministic_sha256: String,
}

#[cfg_attr(
    feature = "extension-module",
    pyclass(module = "forge3d._forge3d", name = "Scene")
)]
pub struct Scene {
    width: u32,
    height: u32,
    grid: u32,

    tp: crate::terrain::pipeline::TerrainPipeline,
    bg0_globals: wgpu::BindGroup,
    bg1_height: wgpu::BindGroup,
    bg2_lut: wgpu::BindGroup,

    // E2/E1: Per-tile uniforms bind group (group 3)
    bg3_tile: wgpu::BindGroup,
    _tile_ubo: TrackedBuffer,
    _tile_slot_ubo: TrackedBuffer,
    _mosaic_params_ubo: TrackedBuffer,

    vbuf: TrackedBuffer,
    ibuf: TrackedBuffer,
    nidx: u32,

    ubo: TrackedBuffer,
    colormap: crate::terrain::ColormapLUT,
    lut_format: &'static str,

    color: TrackedTexture,
    color_view: wgpu::TextureView,
    normal: TrackedTexture,
    normal_view: wgpu::TextureView,
    sample_count: u32,
    msaa_color: Option<TrackedTexture>,
    msaa_view: Option<wgpu::TextureView>,
    msaa_normal: Option<TrackedTexture>,
    msaa_normal_view: Option<wgpu::TextureView>,
    depth: Option<TrackedTexture>,
    depth_view: Option<wgpu::TextureView>,

    height_view: Option<wgpu::TextureView>,
    height_sampler: Option<wgpu::Sampler>,

    scene: SceneGlobals,
    // MENSURA: f64 camera anchor; world camera coordinates are narrowed to
    // f32 only relative to this origin (see src/camera/anchor.rs).
    camera_anchor: crate::camera::Anchor,
    last_uniforms: crate::terrain::TerrainUniforms,
    ssao: SsaoResources,
    ssao_enabled: bool,

    // SSGI/SSR state tracking
    ssgi_enabled: bool,
    ssgi_settings: crate::lighting::screen_space::SSGISettings,
    ssr_enabled: bool,
    ssr_settings: crate::lighting::screen_space::SSRSettings,

    // AETHER settings are provenance-resolved here, but this legacy Scene
    // renderer has no spectral-atmosphere consumer. Its pixel entry points
    // fail closed while configured; active rendering lives on TerrainRenderer.
    #[cfg(feature = "extension-module")]
    atmosphere_state: Option<SceneAtmosphereState>,

    // Bloom state tracking (P1.2)
    bloom_enabled: bool,
    bloom_config: crate::core::bloom::BloomConfig,

    // Toggle base terrain rendering
    terrain_enabled: bool,

    // B5: Planar reflections
    reflection_renderer: Option<crate::core::reflections::PlanarReflectionRenderer>,
    reflections_enabled: bool,

    // B6: Depth of Field
    dof_renderer: Option<crate::core::dof::DofRenderer>,
    dof_enabled: bool,
    dof_params: crate::core::dof::CameraDofParams,

    // B7: Cloud Shadows
    cloud_shadow_renderer: Option<crate::core::cloud_shadows::CloudShadowRenderer>,
    cloud_shadows_enabled: bool,
    bg3_cloud_shadows: Option<wgpu::BindGroup>,
    bg4_dummy_cloud_shadows: wgpu::BindGroup, // Dummy bind group for devices with >=6 bind groups

    // B8: Realtime Clouds
    cloud_renderer: Option<crate::core::clouds::CloudRenderer>,
    clouds_enabled: bool,

    // B10: Ground Plane (Raster)
    ground_plane_renderer: Option<crate::core::ground_plane::GroundPlaneRenderer>,
    ground_plane_enabled: bool,

    // B11: Water Surface Color Toggle
    water_surface_renderer: Option<crate::core::water_surface::WaterSurfaceRenderer>,
    water_surface_enabled: bool,

    // B12: Soft Light Radius (Raster)
    soft_light_radius_renderer: Option<crate::core::soft_light_radius::SoftLightRadiusRenderer>,
    soft_light_radius_enabled: bool,

    // B13: Point & Spot Lights (Realtime)
    point_spot_lights_renderer: Option<crate::core::point_spot_lights::PointSpotLightRenderer>,
    point_spot_lights_enabled: bool,

    // B14: Rect Area Lights (LTC)
    ltc_area_lights_renderer: Option<crate::core::ltc_area_lights::LTCRectAreaLightRenderer>,
    ltc_area_lights_enabled: bool,

    // B15: Image-Based Lighting (IBL) Polish
    ibl_renderer: Option<crate::core::ibl::IBLRenderer>,
    ibl_enabled: bool,

    // B16: Dual-source blending OIT
    dual_source_oit_renderer: Option<crate::core::dual_source_oit::DualSourceOITRenderer>,
    dual_source_oit_enabled: bool,

    // D: Native overlays compositor
    overlay_renderer: Option<crate::core::overlays::OverlayRenderer>,
    overlay_enabled: bool,

    // D: Native text overlay (rectangle quads until MSDF is wired)
    text_overlay_renderer: Option<crate::core::text_overlay::TextOverlayRenderer>,
    text_overlay_enabled: bool,
    text_overlay_alpha: f32,
    text_instances: Vec<crate::core::text_overlay::TextInstance>,

    // D11: 3D text meshes
    text3d_renderer: Option<crate::core::text_mesh::TextMeshRenderer>,
    text3d_enabled: bool,
    text3d_instances: Vec<Text3DInstance>,

    // CENSOR Task 9: per-render GPU-pass timing for the RenderCertificate.
    // Lazily constructed on first render when TIMESTAMP_QUERY is granted;
    // take/store between renders so each render owns one certificate capture.
    render_timing: std::sync::Mutex<Option<crate::core::gpu_timing::GpuTimingManager>>,
    allocation_owner: crate::core::resource_tracker::AllocationOwner,

    // F16: GPU Instancing (feature-gated)
    #[cfg(feature = "enable-gpu-instancing")]
    mesh_instanced_renderer: Option<crate::render::mesh_instanced::MeshInstancedRenderer>,
    #[cfg(feature = "enable-gpu-instancing")]
    instanced_batches: Vec<InstancedBatch>,
}
// T41-END:scene-module
