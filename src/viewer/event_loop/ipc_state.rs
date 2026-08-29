// src/viewer/event_loop/ipc_state.rs
// IPC state management for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

use std::collections::VecDeque;
use std::sync::{mpsc, Mutex, OnceLock};

use super::super::ipc::TerrainVolumetricsReport;
use super::super::ipc::ViewerStats;
use super::super::scene_review::SceneReviewSnapshot;
use super::super::viewer_enums::ViewerCmd;
use crate::picking::PickEvent;

pub struct QueuedIpcCommand {
    pub cmd: ViewerCmd,
    pub completion: mpsc::SyncSender<Result<(), String>>,
}

/// Global IPC command queue - static ensures visibility across threads. Every
/// item carries a one-shot execution completion channel.
static IPC_QUEUE: OnceLock<Mutex<VecDeque<QueuedIpcCommand>>> = OnceLock::new();

/// Global picking event queue for polling
static PICK_EVENTS: OnceLock<Mutex<Vec<PickEvent>>> = OnceLock::new();

/// Global lasso state string (simple shared state)
static LASSO_STATE: OnceLock<Mutex<String>> = OnceLock::new();

/// Get the global IPC command queue
pub fn get_ipc_queue() -> &'static Mutex<VecDeque<QueuedIpcCommand>> {
    IPC_QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Get the global picking event queue
pub fn get_pick_events() -> &'static Mutex<Vec<PickEvent>> {
    PICK_EVENTS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Get the global lasso state
pub fn get_lasso_state() -> &'static Mutex<String> {
    LASSO_STATE.get_or_init(|| Mutex::new("inactive".to_string()))
}

/// Global viewer stats for IPC queries
static IPC_STATS: OnceLock<Mutex<ViewerStats>> = OnceLock::new();

/// Get the global IPC stats
pub fn get_ipc_stats() -> &'static Mutex<ViewerStats> {
    IPC_STATS.get_or_init(|| Mutex::new(ViewerStats::default()))
}

/// Update IPC stats with current viewer state
pub fn update_ipc_stats(vb_ready: bool, vertex_count: u32, index_count: u32, scene_has_mesh: bool) {
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.vb_ready = vb_ready;
        stats.vertex_count = vertex_count;
        stats.index_count = index_count;
        stats.scene_has_mesh = scene_has_mesh;
    }
}

/// Update IPC transform stats
pub fn update_ipc_transform_stats(transform_version: u64, transform_is_identity: bool) {
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.transform_version = transform_version;
        stats.transform_is_identity = transform_is_identity;
    }
}

pub fn update_ipc_revision_stats(applied_command_revision: u64, rendered_frame_revision: u64) {
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.applied_command_revision = applied_command_revision;
        stats.rendered_frame_revision = rendered_frame_revision;
    }
}

pub fn update_ipc_media_diagnostics(
    diagnostics: Option<serde_json::Value>,
    render_error: Option<String>,
) {
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.media_diagnostics = diagnostics;
        stats.media_render_error = render_error;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_ipc_frame_stats(
    adapter_name: &str,
    adapter_vendor: u32,
    adapter_device: u32,
    adapter_backend: &str,
    adapter_device_type: &str,
    adapter_driver: &str,
    adapter_driver_info: &str,
    active_camera: &str,
    camera_anchor_origin: [f64; 3],
    camera_rebase_count: u64,
    history_invalidation_count: u64,
    last_vector_source_delta: [f64; 3],
    last_vector_packed_delta: [f32; 3],
    vector_source_bytes: u64,
    vector_render_cache_bytes: u64,
    vector_gpu_bytes: u64,
    vector_gpu_allocation_ids: Vec<u64>,
    vector_bvh_cpu_bytes: u64,
    frame_count: u64,
    taa_enabled: bool,
    taa_history_valid: bool,
    ssgi_enabled: bool,
    ssgi_temporal_enabled: bool,
    ssgi_history_valid: bool,
    ssr_enabled: bool,
    ssr_history_valid: bool,
    fog_enabled: bool,
    fog_history_valid: bool,
    temporal_history_allocation_ids: [u64; 6],
    terrain_revision: u64,
    terrain_heightmap_allocation_id: u64,
    terrain_heightmap_bytes: u64,
    terrain_shadow_binding_revision: u64,
    point_cloud_point_count: u64,
    point_cloud_visible_point_count: u64,
    point_cloud_source_bytes: u64,
    point_cloud_render_cache_bytes: u64,
    point_cloud_gpu_instance_bytes: u64,
    point_cloud_gpu_instance_id: u64,
) {
    let metrics = crate::core::memory_tracker::global_tracker().get_metrics();
    if let Ok(mut stats) = get_ipc_stats().lock() {
        stats.adapter_name.clear();
        stats.adapter_name.push_str(adapter_name);
        stats.adapter_vendor = adapter_vendor;
        stats.adapter_device = adapter_device;
        stats.adapter_backend.clear();
        stats.adapter_backend.push_str(adapter_backend);
        stats.adapter_device_type.clear();
        stats.adapter_device_type.push_str(adapter_device_type);
        stats.adapter_driver.clear();
        stats.adapter_driver.push_str(adapter_driver);
        stats.adapter_driver_info.clear();
        stats.adapter_driver_info.push_str(adapter_driver_info);
        stats.active_camera.clear();
        stats.active_camera.push_str(active_camera);
        stats.camera_anchor_origin = camera_anchor_origin;
        stats.camera_rebase_count = camera_rebase_count;
        stats.history_invalidation_count = history_invalidation_count;
        stats.last_vector_source_delta = last_vector_source_delta;
        stats.last_vector_packed_delta = last_vector_packed_delta;
        stats.vector_source_bytes = vector_source_bytes;
        stats.vector_render_cache_bytes = vector_render_cache_bytes;
        stats.vector_gpu_bytes = vector_gpu_bytes;
        stats.vector_gpu_allocation_ids = vector_gpu_allocation_ids;
        stats.vector_bvh_cpu_bytes = vector_bvh_cpu_bytes;
        stats.frame_count = frame_count;
        stats.taa_enabled = taa_enabled;
        stats.taa_history_valid = taa_history_valid;
        stats.ssgi_enabled = ssgi_enabled;
        stats.ssgi_temporal_enabled = ssgi_temporal_enabled;
        stats.ssgi_history_valid = ssgi_history_valid;
        stats.ssr_enabled = ssr_enabled;
        stats.ssr_history_valid = ssr_history_valid;
        stats.fog_enabled = fog_enabled;
        stats.fog_history_valid = fog_history_valid;
        stats.temporal_history_allocation_ids = temporal_history_allocation_ids;
        stats.terrain_revision = terrain_revision;
        stats.terrain_heightmap_allocation_id = terrain_heightmap_allocation_id;
        stats.terrain_heightmap_bytes = terrain_heightmap_bytes;
        stats.terrain_shadow_binding_revision = terrain_shadow_binding_revision;
        stats.point_cloud_point_count = point_cloud_point_count;
        stats.point_cloud_visible_point_count = point_cloud_visible_point_count;
        stats.point_cloud_source_bytes = point_cloud_source_bytes;
        stats.point_cloud_render_cache_bytes = point_cloud_render_cache_bytes;
        stats.point_cloud_gpu_instance_bytes = point_cloud_gpu_instance_bytes;
        stats.point_cloud_gpu_instance_id = point_cloud_gpu_instance_id;
        stats.tracked_buffer_count = metrics.buffer_count;
        stats.tracked_texture_count = metrics.texture_count;
        stats.tracked_total_bytes = metrics.total_bytes;
        stats.host_visible_bytes = metrics.host_visible_bytes;
        stats.peak_host_visible_bytes = metrics.peak_host_visible_bytes;
        stats.host_visible_limit_bytes = metrics.limit_bytes;
        stats.within_host_visible_budget = metrics.within_budget;
    }
}

/// Global terrain heterogeneous-volumetrics report for IPC queries.
static TERRAIN_VOLUMETRICS_REPORT: OnceLock<Mutex<TerrainVolumetricsReport>> = OnceLock::new();

pub fn get_terrain_volumetrics_report() -> &'static Mutex<TerrainVolumetricsReport> {
    TERRAIN_VOLUMETRICS_REPORT.get_or_init(|| Mutex::new(TerrainVolumetricsReport::default()))
}

pub fn update_terrain_volumetrics_report(report: TerrainVolumetricsReport) {
    if let Ok(mut current) = get_terrain_volumetrics_report().lock() {
        *current = report;
    }
}

/// Global TV16 scene-review snapshot for structured IPC queries.
static SCENE_REVIEW_STATE: OnceLock<Mutex<SceneReviewSnapshot>> = OnceLock::new();

pub fn get_scene_review_state() -> &'static Mutex<SceneReviewSnapshot> {
    SCENE_REVIEW_STATE.get_or_init(|| Mutex::new(SceneReviewSnapshot::default()))
}

pub fn update_scene_review_state(snapshot: SceneReviewSnapshot) {
    if let Ok(mut current) = get_scene_review_state().lock() {
        *current = snapshot;
    }
}

/// Bundle save request: (path, optional name)
static PENDING_BUNDLE_SAVE: OnceLock<Mutex<Option<(String, Option<String>)>>> = OnceLock::new();

/// Bundle load request: path
static PENDING_BUNDLE_LOAD: OnceLock<Mutex<Option<String>>> = OnceLock::new();

/// Get the pending bundle save request (path, optional name).
/// Calling this clears the pending request.
pub fn take_pending_bundle_save() -> Option<(String, Option<String>)> {
    let lock = PENDING_BUNDLE_SAVE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        guard.take()
    } else {
        None
    }
}

/// Set a pending bundle save request.
pub fn set_pending_bundle_save(path: String, name: Option<String>) {
    let lock = PENDING_BUNDLE_SAVE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        *guard = Some((path, name));
    }
}

/// Get the pending bundle load request (path).
/// Calling this clears the pending request.
pub fn take_pending_bundle_load() -> Option<String> {
    let lock = PENDING_BUNDLE_LOAD.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        guard.take()
    } else {
        None
    }
}

/// Set a pending bundle load request.
pub fn set_pending_bundle_load(path: String) {
    let lock = PENDING_BUNDLE_LOAD.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        *guard = Some(path);
    }
}
