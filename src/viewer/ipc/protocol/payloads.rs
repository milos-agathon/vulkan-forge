use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcHeightAoConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub directions: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub max_distance: Option<f32>,
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub resolution_scale: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcSunVisConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub samples: Option<u32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub max_distance: Option<f32>,
    #[serde(default)]
    pub softness: Option<f32>,
    #[serde(default)]
    pub bias: Option<f32>,
    #[serde(default)]
    pub resolution_scale: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcMaterialLayerConfig {
    #[serde(default)]
    pub snow_enabled: Option<bool>,
    #[serde(default)]
    pub snow_altitude_min: Option<f32>,
    #[serde(default)]
    pub snow_altitude_blend: Option<f32>,
    #[serde(default)]
    pub snow_slope_max: Option<f32>,
    #[serde(default)]
    pub rock_enabled: Option<bool>,
    #[serde(default)]
    pub rock_slope_min: Option<f32>,
    #[serde(default)]
    pub wetness_enabled: Option<bool>,
    #[serde(default)]
    pub wetness_strength: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcVectorOverlayConfig {
    #[serde(default)]
    pub depth_test: Option<bool>,
    #[serde(default)]
    pub depth_bias: Option<f32>,
    #[serde(default)]
    pub halo_enabled: Option<bool>,
    #[serde(default)]
    pub halo_width: Option<f32>,
    #[serde(default)]
    pub halo_color: Option<[f32; 4]>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTonemapConfig {
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default)]
    pub white_point: Option<f32>,
    #[serde(default)]
    pub white_balance_enabled: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub tint: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcDofConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub f_stop: Option<f32>,
    #[serde(default)]
    pub focus_distance: Option<f32>,
    #[serde(default)]
    pub focal_length: Option<f32>,
    #[serde(default)]
    pub tilt_pitch: Option<f32>,
    #[serde(default)]
    pub tilt_yaw: Option<f32>,
    #[serde(default)]
    pub quality: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcMotionBlurConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub samples: Option<u32>,
    #[serde(default)]
    pub shutter_open: Option<f32>,
    #[serde(default)]
    pub shutter_close: Option<f32>,
    #[serde(default)]
    pub cam_phi_delta: Option<f32>,
    #[serde(default)]
    pub cam_theta_delta: Option<f32>,
    #[serde(default)]
    pub cam_radius_delta: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcLensEffectsConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub distortion: Option<f32>,
    #[serde(default)]
    pub chromatic_aberration: Option<f32>,
    #[serde(default)]
    pub vignette_strength: Option<f32>,
    #[serde(default)]
    pub vignette_radius: Option<f32>,
    #[serde(default)]
    pub vignette_softness: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcDenoiseConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub iterations: Option<u32>,
    #[serde(default)]
    pub sigma_color: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcDensityVolumeConfig {
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub center: Option<[f32; 3]>,
    #[serde(default)]
    pub size: Option<[f32; 3]>,
    #[serde(default)]
    pub resolution: Option<[u32; 3]>,
    #[serde(default)]
    pub density_scale: Option<f32>,
    #[serde(default)]
    pub edge_softness: Option<f32>,
    #[serde(default)]
    pub noise_strength: Option<f32>,
    #[serde(default)]
    pub floor_offset: Option<f32>,
    #[serde(default)]
    pub ceiling: Option<f32>,
    #[serde(default)]
    pub plume_spread: Option<f32>,
    #[serde(default)]
    pub wind: Option<[f32; 3]>,
    #[serde(default)]
    pub seed: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcVolumetricsConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub density: Option<f32>,
    #[serde(default)]
    pub height_falloff: Option<f32>,
    #[serde(default)]
    pub scattering: Option<f32>,
    #[serde(default)]
    pub absorption: Option<f32>,
    #[serde(default)]
    pub light_shafts: Option<bool>,
    #[serde(default)]
    pub shaft_intensity: Option<f32>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub half_res: Option<bool>,
    #[serde(default)]
    pub density_volumes: Vec<IpcDensityVolumeConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcSkyConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub turbidity: Option<f32>,
    #[serde(default)]
    pub ground_albedo: Option<f32>,
    #[serde(default)]
    pub sun_intensity: Option<f32>,
    #[serde(default)]
    pub aerial_perspective: Option<bool>,
    #[serde(default)]
    pub sky_exposure: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTerrainScatterLevel {
    #[serde(default)]
    pub positions: Vec<[f32; 3]>,
    #[serde(default)]
    pub normals: Vec<[f32; 3]>,
    #[serde(default)]
    pub indices: Vec<u32>,
    #[serde(default)]
    pub max_distance: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IpcHlodConfig {
    pub hlod_distance: f32,
    pub cluster_radius: f32,
    pub simplify_ratio: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTerrainScatterBlend {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub bury_depth: Option<f32>,
    #[serde(default)]
    pub fade_distance: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTerrainScatterContact {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub distance: Option<f32>,
    #[serde(default)]
    pub strength: Option<f32>,
    #[serde(default)]
    pub vertical_weight: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcTerrainScatterBatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub color: Option<[f32; 4]>,
    #[serde(default)]
    pub max_draw_distance: Option<f32>,
    #[serde(default)]
    pub terrain_blend: Option<IpcTerrainScatterBlend>,
    #[serde(default)]
    pub terrain_contact: Option<IpcTerrainScatterContact>,
    #[serde(default)]
    pub transforms: Vec<[f32; 16]>,
    #[serde(default)]
    pub levels: Vec<IpcTerrainScatterLevel>,
    #[serde(default)]
    pub wind: Option<IpcScatterWind>,
    pub hlod: Option<IpcHlodConfig>,
}

fn default_speed() -> f32 {
    1.0
}
fn default_rigidity() -> f32 {
    0.5
}
fn default_bend_extent() -> f32 {
    1.0
}
fn default_gust_frequency() -> f32 {
    0.3
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct IpcScatterWind {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub direction_deg: f32,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default)]
    pub amplitude: f32,
    #[serde(default = "default_rigidity")]
    pub rigidity: f32,
    #[serde(default)]
    pub bend_start: f32,
    #[serde(default = "default_bend_extent")]
    pub bend_extent: f32,
    #[serde(default)]
    pub gust_strength: f32,
    #[serde(default = "default_gust_frequency")]
    pub gust_frequency: f32,
    #[serde(default)]
    pub fade_start: f32,
    #[serde(default)]
    pub fade_end: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcRasterOverlaySpec {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub extent: Option<[f32; 4]>,
    #[serde(default)]
    pub opacity: Option<f32>,
    #[serde(default)]
    pub z_order: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IpcSceneVectorOverlay {
    pub name: String,
    #[serde(default)]
    pub vertices: Vec<crate::viewer::viewer_enums::ViewerVectorVertex>,
    #[serde(default)]
    pub indices: Vec<u32>,
    #[serde(default = "super::defaults::default_primitive")]
    pub primitive: String,
    #[serde(default)]
    pub drape: bool,
    #[serde(default = "super::defaults::default_drape_offset")]
    pub drape_offset: f32,
    #[serde(default = "super::defaults::default_opacity")]
    pub opacity: f32,
    #[serde(default = "super::defaults::default_depth_bias")]
    pub depth_bias: f32,
    #[serde(default = "super::defaults::default_line_width")]
    pub line_width: f32,
    #[serde(default = "super::defaults::default_point_size")]
    pub point_size: f32,
    #[serde(default)]
    pub z_order: i32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcSceneBaseState {
    #[serde(default)]
    pub preset: Option<Map<String, Value>>,
    #[serde(default)]
    pub raster_overlays: Vec<IpcRasterOverlaySpec>,
    #[serde(default)]
    pub vector_overlays: Vec<IpcSceneVectorOverlay>,
    #[serde(default)]
    pub labels: Vec<Map<String, Value>>,
    #[serde(default)]
    pub scatter_batches: Vec<IpcTerrainScatterBatch>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcReviewLayer {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub raster_overlays: Vec<IpcRasterOverlaySpec>,
    #[serde(default)]
    pub vector_overlays: Vec<IpcSceneVectorOverlay>,
    #[serde(default)]
    pub labels: Vec<Map<String, Value>>,
    #[serde(default)]
    pub scatter_batches: Vec<IpcTerrainScatterBatch>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcSceneVariant {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub active_layer_ids: Vec<String>,
    #[serde(default)]
    pub preset: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct IpcSceneReviewState {
    #[serde(default)]
    pub base_state: IpcSceneBaseState,
    #[serde(default)]
    pub review_layers: Vec<IpcReviewLayer>,
    #[serde(default)]
    pub variants: Vec<IpcSceneVariant>,
    #[serde(default)]
    pub active_variant_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ViewerStats {
    pub adapter_name: String,
    pub adapter_vendor: u32,
    pub adapter_device: u32,
    pub adapter_backend: String,
    pub adapter_device_type: String,
    pub adapter_driver: String,
    pub adapter_driver_info: String,
    pub vb_ready: bool,
    pub vertex_count: u32,
    pub index_count: u32,
    pub scene_has_mesh: bool,
    pub transform_version: u64,
    pub transform_is_identity: bool,
    pub active_camera: String,
    pub camera_anchor_origin: [f64; 3],
    pub camera_rebase_count: u64,
    pub history_invalidation_count: u64,
    pub last_vector_source_delta: [f64; 3],
    pub last_vector_packed_delta: [f32; 3],
    pub vector_source_bytes: u64,
    pub vector_render_cache_bytes: u64,
    pub vector_gpu_bytes: u64,
    pub vector_gpu_allocation_ids: Vec<u64>,
    pub vector_bvh_cpu_bytes: u64,
    pub frame_count: u64,
    /// Latest command revision that completed validation and publication.
    pub applied_command_revision: u64,
    /// Applied revision observed by the latest submitted/presented frame.
    pub rendered_frame_revision: u64,
    pub taa_enabled: bool,
    pub taa_history_valid: bool,
    pub ssgi_enabled: bool,
    pub ssgi_temporal_enabled: bool,
    pub ssgi_history_valid: bool,
    pub ssr_enabled: bool,
    pub ssr_history_valid: bool,
    pub fog_enabled: bool,
    pub fog_history_valid: bool,
    /// TAA-A, TAA-B, SSGI, SSR, fog-full, fog-half ledger identities.
    pub temporal_history_allocation_ids: [u64; 6],
    pub terrain_revision: u64,
    pub terrain_heightmap_allocation_id: u64,
    pub terrain_heightmap_bytes: u64,
    pub terrain_shadow_binding_revision: u64,
    pub point_cloud_point_count: u64,
    pub point_cloud_visible_point_count: u64,
    pub point_cloud_source_bytes: u64,
    pub point_cloud_render_cache_bytes: u64,
    pub point_cloud_gpu_instance_bytes: u64,
    pub point_cloud_gpu_instance_id: u64,
    pub tracked_buffer_count: u32,
    pub tracked_texture_count: u32,
    pub tracked_total_bytes: u64,
    pub host_visible_bytes: u64,
    pub peak_host_visible_bytes: u64,
    pub host_visible_limit_bytes: u64,
    pub within_host_visible_budget: bool,
    pub media_diagnostics: Option<serde_json::Value>,
    pub media_render_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TerrainVolumetricsVolumeReport {
    pub preset: String,
    pub center: [f32; 3],
    pub size: [f32; 3],
    pub resolution: [u32; 3],
    pub atlas_offset: [u32; 3],
    pub voxel_count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TerrainVolumetricsReport {
    pub active_volume_count: u32,
    pub atlas_dimensions: [u32; 3],
    pub total_voxels: u64,
    pub texture_bytes: u64,
    /// Process-local ledger identity of the live atlas texture (zero if absent).
    pub atlas_allocation_id: u64,
    pub memory_budget_bytes: u64,
    pub raymarch_steps: u32,
    pub half_res: bool,
    pub volumes: Vec<TerrainVolumetricsVolumeReport>,
}

impl Default for TerrainVolumetricsReport {
    fn default() -> Self {
        Self {
            active_volume_count: 0,
            atlas_dimensions: [0, 0, 0],
            total_voxels: 0,
            texture_bytes: 0,
            atlas_allocation_id: 0,
            memory_budget_bytes: 16 * 1024 * 1024,
            raymarch_steps: 0,
            half_res: false,
            volumes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BundleRequest {
    pub pending: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl BundleRequest {
    pub fn none() -> Self {
        Self {
            pending: false,
            path: None,
            name: None,
        }
    }

    pub fn save(path: String, name: Option<String>) -> Self {
        Self {
            pending: true,
            path: Some(path),
            name,
        }
    }

    pub fn load(path: String) -> Self {
        Self {
            pending: true,
            path: Some(path),
            name: None,
        }
    }
}
