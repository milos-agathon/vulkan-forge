// src/viewer/viewer_struct.rs
// Viewer struct definition with pub(crate) fields to allow method extraction
// RELEVANT FILES: src/viewer/mod.rs

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
#[cfg(feature = "extension-module")]
use wgpu::Adapter;
use wgpu::{
    BindGroup, BindGroupLayout, ComputePipeline, Device, Queue, RenderPipeline, Sampler, Surface,
    SurfaceConfiguration, TextureView,
};
use winit::keyboard::KeyCode;
use winit::window::Window;

use crate::camera::Anchor;
use crate::cli::args::GiVizMode;
use crate::core::gpu_timing::GpuTimingManager;
use crate::core::ibl::IBLRenderer;
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};
use crate::core::screen_space_effects::ScreenSpaceEffectsManager;
use crate::core::shadows::{CsmConfig, CsmShadowMap};
use crate::core::temporal_history::TemporalHistoryState;
use crate::core::text_overlay::TextOverlayRenderer;
use crate::labels::LabelManager;
use crate::p5::ssr::SsrScenePreset;
use crate::passes::gi::GiPass;
use crate::picking::UnifiedPickingSystem;
use crate::render::params::SsrParams;

use super::camera_controller::CameraController;
use super::scene_review::ViewerSceneReviewRegistry;
use super::viewer_config::{FpsCounter, ViewerConfig};
use super::viewer_enums::{CaptureKind, FogMode, VizMode};
use super::viewer_types::FrameCamera;

pub(crate) struct SkySnapshotResources {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) camera: TrackedBuffer,
    pub(crate) _output: TrackedTexture,
    pub(crate) output_view: TextureView,
    pub(crate) compute_bind_group: BindGroup,
    pub(crate) camera_bind_group: BindGroup,
    pub(crate) present_bind_group: BindGroup,
}

pub struct Viewer {
    pub(crate) window: Arc<Window>,
    pub(crate) surface: Surface<'static>,
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    #[cfg(feature = "extension-module")]
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) config: SurfaceConfiguration,
    pub(crate) camera: CameraController,
    /// Sole persistent anchor owned by the interactive viewer.
    pub(crate) camera_anchor: Anchor,
    /// Immutable pose/anchor copied at the latest frame boundary.
    pub(crate) frame_camera: Option<FrameCamera>,
    pub(crate) camera_rebase_count: u64,
    pub(crate) history_invalidation_count: u64,
    pub(crate) last_vector_source_delta: [f64; 3],
    pub(crate) last_vector_packed_delta: [f32; 3],
    pub(crate) view_config: ViewerConfig,
    pub(crate) frame_count: u64,
    pub(crate) applied_command_revision: u64,
    pub(crate) rendered_frame_revision: u64,
    pub(crate) fps_counter: FpsCounter,
    #[cfg(feature = "extension-module")]
    pub(crate) terrain_scene: Option<crate::terrain::TerrainScene>,
    // Standalone terrain viewer (no PyO3 dependencies)
    pub(crate) terrain_viewer: Option<super::terrain::ViewerTerrainScene>,
    // Input state
    pub(crate) keys_pressed: HashSet<KeyCode>,
    pub(crate) shift_pressed: bool,
    // GI manager and toggles
    pub(crate) gi: Option<ScreenSpaceEffectsManager>,
    pub(crate) gi_pass: Option<GiPass>,
    pub(crate) ssr_params: SsrParams,
    pub(crate) gi_seed: Option<u32>,
    pub(crate) gi_timing: Option<GpuTimingManager>,
    pub(crate) gi_gpu_hzb_ms: f32,
    pub(crate) gi_gpu_ssao_ms: f32,
    pub(crate) gi_gpu_ssgi_ms: f32,
    pub(crate) gi_gpu_ssr_ms: f32,
    pub(crate) gi_gpu_composite_ms: f32,
    // Snapshot request path (processed on next frame before present)
    pub(crate) snapshot_request: Option<String>,
    // Offscreen color to read back when snapshotting this frame
    pub(crate) pending_snapshot_tex: Option<TrackedTexture>,
    // P5.1: deferred capture queue processed after rendering
    pub(crate) pending_captures: VecDeque<CaptureKind>,
    // GBuffer geometry pipeline and resources
    pub(crate) geom_bind_group_layout: Option<BindGroupLayout>,
    pub(crate) geom_pipeline: Option<RenderPipeline>,
    pub(crate) object_overlay_pipeline: Option<RenderPipeline>,
    pub(crate) geom_camera_buffer: Option<TrackedBuffer>,
    pub(crate) geom_bind_group: Option<BindGroup>,
    pub(crate) geom_vb: Option<TrackedBuffer>,
    pub(crate) geom_ib: Option<TrackedBuffer>,
    pub(crate) geom_index_count: u32,
    /// Retained object-local source used by the public absolute-f64 pick route.
    pub(crate) object_source_positions: Vec<[f32; 3]>,
    pub(crate) object_source_indices: Vec<u32>,
    pub(crate) z_texture: Option<TrackedTexture>,
    pub(crate) z_view: Option<TextureView>,
    // Albedo texture for geometry
    pub(crate) albedo_texture: Option<TrackedTexture>,
    pub(crate) albedo_view: Option<TextureView>,
    pub(crate) albedo_sampler: Option<Sampler>,
    pub(crate) ssr_env_texture: Option<TrackedTexture>,
    // Composite pipeline (debug show material GBuffer on screen)
    pub(crate) comp_bind_group_layout: Option<BindGroupLayout>,
    pub(crate) comp_pipeline: Option<RenderPipeline>,
    pub(crate) comp_uniform: Option<TrackedBuffer>,
    pub(crate) comp_bind_group_cache: RefCell<HashMap<[usize; 5], BindGroup>>,
    // Lit viz compute pipeline (albedo+normal shading)
    pub(crate) lit_bind_group_layout: BindGroupLayout,
    pub(crate) lit_pipeline: ComputePipeline,
    pub(crate) lit_uniform: TrackedBuffer,
    pub(crate) lit_output: TrackedTexture,
    pub(crate) lit_output_view: TextureView,
    pub(crate) lit_bind_group_cache: RefCell<Option<BindGroup>>,
    pub(crate) gi_baseline_hdr: TrackedTexture,
    pub(crate) gi_baseline_hdr_view: TextureView,
    pub(crate) gi_baseline_diffuse_hdr: TrackedTexture,
    pub(crate) gi_baseline_diffuse_hdr_view: TextureView,
    pub(crate) gi_baseline_spec_hdr: TrackedTexture,
    pub(crate) gi_baseline_spec_hdr_view: TextureView,
    pub(crate) gi_output_hdr: TrackedTexture,
    pub(crate) gi_output_hdr_view: TextureView,
    pub(crate) gi_debug: TrackedTexture,
    pub(crate) gi_debug_view: TextureView,
    pub(crate) gi_baseline_bgl: BindGroupLayout,
    pub(crate) gi_baseline_pipeline: ComputePipeline,
    pub(crate) gi_split_bgl: BindGroupLayout,
    pub(crate) gi_split_pipeline: ComputePipeline,
    pub(crate) gi_ao_weight: f32,
    pub(crate) gi_ssgi_weight: f32,
    pub(crate) gi_ssr_weight: f32,
    // Lit params (exposed via :lit-* commands)
    pub(crate) lit_sun_direction_ws: [f32; 3],
    pub(crate) sky_sun_direction_ws: [f32; 3],
    pub(crate) lit_sun_intensity: f32,
    pub(crate) lit_directional_scale: f32,
    pub(crate) astro_observation_active: bool,
    pub(crate) astro_observation_revision: u64,
    pub(crate) astro_terrain_revision: u64,
    pub(crate) lit_ibl_intensity: f32,
    pub(crate) lit_use_ibl: bool,
    pub(crate) lit_ibl_rotation_deg: f32,
    // Lit BRDF selection (0=Lambert,1=Phong,4=GGX,6=Disney)
    pub(crate) lit_brdf: u32,
    // Lit roughness (used by debug modes and future shading controls)
    pub(crate) lit_roughness: f32,
    // Lit debug mode: 0=off, 1=roughness smoke test, 2=NDF-only GGX
    pub(crate) lit_debug_mode: u32,
    // Fallback pipeline to draw a solid color when GI/geometry path is unavailable
    pub(crate) fallback_pipeline: RenderPipeline,
    pub(crate) viz_mode: VizMode,
    pub(crate) gi_viz_mode: GiVizMode,
    // SSAO composite control
    pub(crate) use_ssao_composite: bool,
    pub(crate) ssao_composite_mul: f32,
    // Cached SSAO blur toggle for query commands
    pub(crate) ssao_blur_enabled: bool,
    // IBL integration
    pub(crate) ibl_renderer: Option<IBLRenderer>,
    pub(crate) ibl_env_view: Option<TextureView>,
    pub(crate) ibl_sampler: Option<Sampler>,
    pub(crate) ibl_hdr_path: Option<String>,
    pub(crate) ibl_cache_dir: Option<std::path::PathBuf>,
    pub(crate) ibl_base_resolution: Option<u32>,
    // Viz depth override
    pub(crate) viz_depth_max_override: Option<f32>,
    // Auto-snapshot support (one-time)
    pub(crate) auto_snapshot_path: Option<String>,
    pub(crate) auto_snapshot_done: bool,
    // P5 dump request
    pub(crate) dump_p5_requested: bool,
    // Exact viewer adapter identity for runtime evidence.
    pub(crate) adapter_name: String,
    pub(crate) adapter_vendor: u32,
    pub(crate) adapter_device: u32,
    pub(crate) adapter_backend: String,
    pub(crate) adapter_device_type: String,
    pub(crate) adapter_driver: String,
    pub(crate) adapter_driver_info: String,
    // Debug: log render gate and snapshot once
    pub(crate) debug_logged_render_gate: bool,

    // Sky rendering (P6-01)
    pub(crate) sky_bind_group_layout0: BindGroupLayout,
    pub(crate) sky_bind_group_layout1: BindGroupLayout,
    pub(crate) sky_pipeline: ComputePipeline,
    pub(crate) sky_params: TrackedBuffer,
    pub(crate) sky_camera: TrackedBuffer,
    pub(crate) sky_output: TrackedTexture,
    pub(crate) sky_output_view: TextureView,
    pub(crate) sky_present_bind_group_layout: BindGroupLayout,
    pub(crate) sky_present_depth_pipeline: RenderPipeline,
    pub(crate) sky_present_flat_pipeline: RenderPipeline,
    pub(crate) sky_present_sampler: Sampler,
    pub(crate) night_pipeline: RenderPipeline,
    pub(crate) night_instances: TrackedBuffer,
    pub(crate) night_bind_group: BindGroup,
    pub(crate) _night_moon_texture: TrackedTexture,
    pub(crate) night_instance_count: u32,
    pub(crate) astro_night_revision: u64,
    pub(crate) sky_bg0_cache: RefCell<Option<BindGroup>>,
    pub(crate) sky_bg1_cache: RefCell<Option<BindGroup>>,
    pub(crate) sky_present_bg_cache: RefCell<Option<BindGroup>>,
    pub(crate) sky_snapshot_cache: Option<SkySnapshotResources>,
    pub(crate) sky_enabled: bool,

    // P6: Fog rendering resources and parameters
    pub(crate) fog_enabled: bool,
    pub(crate) fog_params: TrackedBuffer,
    pub(crate) fog_camera: TrackedBuffer,
    pub(crate) fog_output: TrackedTexture,
    pub(crate) fog_output_view: TextureView,
    pub(crate) fog_history: TrackedTexture,
    pub(crate) fog_history_view: TextureView,
    pub(crate) fog_bg0_cache: RefCell<Option<BindGroup>>,
    pub(crate) fog_bg1_cache: RefCell<Option<BindGroup>>,
    pub(crate) fog_bg2_cache: RefCell<Option<BindGroup>>,
    pub(crate) fog_bg2_half_cache: RefCell<Option<BindGroup>>,
    pub(crate) fog_upsample_bg_cache: RefCell<Option<BindGroup>>,
    pub(crate) fog_bg3_cache: RefCell<Option<BindGroup>>,
    pub(crate) fog_depth_sampler: Sampler,
    pub(crate) fog_history_sampler: Sampler,
    pub(crate) fog_pipeline: ComputePipeline,
    pub(crate) fog_frame_index: u32,
    pub(crate) fog_history_state: TemporalHistoryState,
    // Froxelized volumetrics (Milestone 4)
    pub(crate) fog_bgl3: BindGroupLayout,
    pub(crate) _froxel_tex: TrackedTexture,
    pub(crate) froxel_view: TextureView,
    pub(crate) froxel_sampler: Sampler,
    pub(crate) froxel_build_pipeline: ComputePipeline,
    pub(crate) froxel_apply_pipeline: ComputePipeline,
    // P6-10: Half-resolution fog + upsample
    pub(crate) fog_half_res_enabled: bool,
    pub(crate) fog_output_half: TrackedTexture,
    pub(crate) fog_output_half_view: TextureView,
    pub(crate) fog_history_half: TrackedTexture,
    pub(crate) fog_history_half_view: TextureView,
    pub(crate) fog_upsample_bgl: BindGroupLayout,
    pub(crate) fog_upsample_pipeline: ComputePipeline,
    pub(crate) fog_upsample_params: TrackedBuffer,
    // Bilateral upsample controls
    pub(crate) fog_bilateral: bool,
    pub(crate) fog_upsigma: f32,
    // Fog bind group layouts and shadow resources
    pub(crate) fog_bgl0: BindGroupLayout,
    pub(crate) fog_bgl1: BindGroupLayout,
    pub(crate) fog_bgl2: BindGroupLayout,
    pub(crate) _fog_shadow_map: TrackedTexture,
    pub(crate) fog_shadow_view: TextureView,
    pub(crate) fog_shadow_sampler: Sampler,
    pub(crate) fog_shadow_matrix: TrackedBuffer,
    // Fog zero fallback (1x1 RGBA16F zero) for disabled fog compositing
    pub(crate) _fog_zero_tex: TrackedTexture,
    pub(crate) fog_zero_view: TextureView,
    // Exposed toggles
    pub(crate) fog_density: f32,
    pub(crate) fog_g: f32,
    pub(crate) fog_steps: u32,
    pub(crate) fog_temporal_alpha: f32,
    pub(crate) fog_use_shadows: bool,
    pub(crate) fog_mode: FogMode,
    // Cascaded shadow maps for directional sun shadows (future fog + lighting)
    pub(crate) csm: Option<CsmShadowMap>,
    pub(crate) _csm_config: CsmConfig,
    pub(crate) csm_depth_pipeline: Option<RenderPipeline>,
    pub(crate) csm_depth_camera: Option<TrackedBuffer>,
    pub(crate) csm_depth_bind_group: Option<BindGroup>,
    // Sky exposed controls (runtime adjustable)
    pub(crate) sky_model_id: u32, // 0=Preetham,1=Hosek-Wilkie
    pub(crate) sky_turbidity: f32,
    pub(crate) sky_ground_albedo: f32,
    pub(crate) sky_exposure: f32,
    pub(crate) sky_sun_intensity: f32,

    // HUD overlay renderer
    pub(crate) hud_enabled: bool,
    pub(crate) hud: TextOverlayRenderer,
    // Label manager for screen-space text labels
    pub(crate) label_manager: LabelManager,
    // Unified picking system (Plan 3)
    pub(crate) unified_picking: UnifiedPickingSystem,
    // Currently selected feature for highlighting
    pub(crate) selected_feature_id: u32,
    pub(crate) selected_layer_name: String,
    pub(crate) ssr_scene_loaded: bool,
    pub(crate) ssr_scene_preset: Option<SsrScenePreset>,
    // Object transform (for IPC SetTransform command)
    pub(crate) object_translation: glam::DVec3,
    pub(crate) object_rotation: glam::Quat,
    pub(crate) object_scale: glam::Vec3,
    // Transform version counter for IPC ack (incremented on each set_transform)
    pub(crate) transform_version: u64,
    // P0.1/M1: OIT (Order-Independent Transparency)
    pub(crate) oit_enabled: bool,
    pub(crate) oit_mode: String,
    // P1.1: Previous frame view-projection matrix for motion vectors
    pub(crate) prev_view_proj: glam::Mat4,
    // P1.2: TAA jitter state
    pub(crate) taa_jitter: crate::core::jitter::JitterState,
    // P1.3: TAA renderer
    pub(crate) taa_renderer: Option<crate::core::taa::TaaRenderer>,
    // P5: Point cloud state
    pub(crate) point_cloud: Option<super::pointcloud::PointCloudState>,
    // Scene bundle save/load requests (handled by Python-side)
    pub(crate) pending_bundle_save: Option<(String, Option<String>)>,
    pub(crate) pending_bundle_load: Option<String>,
    // TV16 review-state registry
    pub(crate) scene_review_registry: ViewerSceneReviewRegistry,
    /// Per-command execution outcome consumed by the dispatcher and correlated
    /// IPC completion channel.
    pub(crate) command_error: Option<String>,
}
