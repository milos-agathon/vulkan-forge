//! Q3: GPU profiling markers & timestamp queries
//!
//! Provides GPU timing utilities for performance profiling and debugging.
//! Supports RenderDoc, Nsight Graphics, and RGP markers with configurable
//! timestamp collection for minimal overhead.
//!
//! CENSOR Task 4: timestamp queries are double-buffered. Two independent query
//! sets / resolve buffers / readback buffers are kept, selected by
//! `frame_parity`. `begin_scope`/`end_scope` write into the current slot,
//! `resolve_queries` resolves that slot, and `finish_frame` flips parity after
//! the caller submits. The async `get_results` reads the *other* slot (the
//! previously submitted frame) so it never stalls the current frame, while
//! `get_results_blocking` reads the slot just resolved+submitted for offline
//! single-shot renders.

use super::error::{RenderError, RenderResult};
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::*;

/// Number of frames in flight for the timestamp double buffer.
const SLOTS: usize = 2;

/// Handle for a GPU timing scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimingScopeId(usize);

/// GPU timing configuration
#[derive(Debug, Clone)]
pub struct GpuTimingConfig {
    /// Enable timestamp queries (requires TIMESTAMP_QUERY feature)
    pub enable_timestamps: bool,
    /// Enable pipeline statistics (requires PIPELINE_STATISTICS_QUERY feature)
    pub enable_pipeline_stats: bool,
    /// Enable debug markers for external profilers
    pub enable_debug_markers: bool,
    /// Label prefix for timing scopes
    pub label_prefix: String,
    /// Maximum number of timing queries per frame
    pub max_queries_per_frame: u32,
}

impl Default for GpuTimingConfig {
    fn default() -> Self {
        Self {
            enable_timestamps: true,
            enable_pipeline_stats: false, // Often not supported
            enable_debug_markers: true,
            label_prefix: "forge3d".to_string(),
            max_queries_per_frame: 64,
        }
    }
}

/// GPU timing measurement result
#[derive(Debug, Clone, Default)]
pub struct TimingResult {
    /// Scope name/label
    pub name: String,
    /// GPU time in milliseconds
    pub gpu_time_ms: f32,
    /// Whether timestamp was successfully measured
    pub timestamp_valid: bool,
    /// Number of draw calls recorded for the scope (0 when unknown)
    pub draw_calls: u32,
    /// Pipeline statistics (if available)
    pub pipeline_stats: Option<PipelineStatistics>,
}

impl TimingResult {
    /// Certificate-safe duration: invalid/absent timestamps are always zero.
    pub fn certificate_gpu_ms(&self) -> f64 {
        if self.timestamp_valid {
            self.gpu_time_ms as f64
        } else {
            0.0
        }
    }
}

/// Pipeline statistics from GPU
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    /// Number of vertex invocations
    pub vertex_invocations: u64,
    /// Number of clipping invocations
    pub clipper_invocations: u64,
    /// Number of fragment invocations
    pub fragment_invocations: u64,
    /// Number of compute invocations
    pub compute_invocations: u64,
}

/// Active timing scope for measuring GPU work
pub struct TimingScope<'a> {
    timing_manager: &'a mut GpuTimingManager,
    scope_id: TimingScopeId,
    encoder: &'a mut CommandEncoder,
}

impl<'a> TimingScope<'a> {
    /// Begin timing scope with debug marker
    pub fn begin(&mut self, label: &str) {
        self.timing_manager
            .begin_scope_internal(self.encoder, self.scope_id, label);
    }

    /// End timing scope
    pub fn end(self) {
        self.timing_manager
            .end_scope_internal(self.encoder, self.scope_id);
    }
}

/// GPU timing manager for performance profiling
pub struct GpuTimingManager {
    config: GpuTimingConfig,
    device: Arc<Device>,

    // Timestamp queries (double-buffered: one slot per frame in flight)
    timestamp_query_set: [Option<QuerySet>; SLOTS],
    timestamp_buffer: [Option<TrackedBuffer>; SLOTS],
    timestamp_readback_buffer: [Option<TrackedBuffer>; SLOTS],

    // Pipeline statistics queries (single-buffered; disabled by default)
    pipeline_stats_query_set: Option<QuerySet>,
    pipeline_stats_buffer: Option<TrackedBuffer>,
    pipeline_stats_readback_buffer: Option<TrackedBuffer>,

    // Per-slot timing state so a slot's labels always match its queries.
    active_scopes: HashMap<TimingScopeId, String>,
    query_index: [u32; SLOTS],
    scope_labels: [Vec<String>; SLOTS],
    scope_draw_calls: [Vec<u32>; SLOTS],

    /// Number of pipeline-statistics queries actually begun this frame.
    /// Single-buffered like the stats query set. Currently no scope ever begins
    /// a statistics query, so this stays 0 and `resolve_queries` skips the stats
    /// range (resolving a never-written range loses the device on wgpu 0.19 /
    /// Vulkan). A future caller that wires up statistics queries must increment
    /// this when it writes one.
    pipeline_stats_query_index: u32,

    /// Which slot the current frame writes/resolves into.
    frame_parity: usize,

    // Feature support
    supports_timestamps: bool,
    supports_pipeline_stats: bool,
    timestamp_period: f64, // Nanoseconds per timestamp unit
}

impl GpuTimingManager {
    /// Create new GPU timing manager
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: GpuTimingConfig,
    ) -> RenderResult<Self> {
        let features = device.features();
        let _limits = device.limits();

        let supports_timestamps =
            config.enable_timestamps && features.contains(Features::TIMESTAMP_QUERY);
        let supports_pipeline_stats =
            config.enable_pipeline_stats && features.contains(Features::PIPELINE_STATISTICS_QUERY);

        // wgpu 0.19 exposes the timestamp period (ns per tick) on the Queue.
        let timestamp_period = if supports_timestamps {
            queue.get_timestamp_period() as f64
        } else {
            1.0
        };

        let mut manager = Self {
            config: config.clone(),
            device: device.clone(),
            timestamp_query_set: [None, None],
            timestamp_buffer: [None, None],
            timestamp_readback_buffer: [None, None],
            pipeline_stats_query_set: None,
            pipeline_stats_buffer: None,
            pipeline_stats_readback_buffer: None,
            active_scopes: HashMap::new(),
            query_index: [0; SLOTS],
            scope_labels: [Vec::new(), Vec::new()],
            scope_draw_calls: [Vec::new(), Vec::new()],
            pipeline_stats_query_index: 0,
            frame_parity: 0,
            supports_timestamps,
            supports_pipeline_stats,
            timestamp_period,
        };

        // Initialize query sets and buffers
        manager.initialize_queries()?;

        Ok(manager)
    }

    fn initialize_queries(&mut self) -> RenderResult<()> {
        let query_count = self.config.max_queries_per_frame * 2; // Begin + End for each scope

        // Initialize timestamp queries (one independent slot per frame in flight)
        if self.supports_timestamps {
            let buffer_size = (query_count as u64) * std::mem::size_of::<u64>() as u64;
            for slot in 0..SLOTS {
                let query_set = self.device.create_query_set(&QuerySetDescriptor {
                    label: Some("gpu_timing_timestamps"),
                    ty: QueryType::Timestamp,
                    count: query_count,
                });

                let timestamp_buffer = tracked_create_buffer(
                    &self.device,
                    &BufferDescriptor {
                        label: Some(&format!("gpu_timing.resolve_slot{slot}")),
                        size: buffer_size,
                        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                )?;

                let timestamp_readback = tracked_create_buffer(
                    &self.device,
                    &BufferDescriptor {
                        label: Some(&format!("gpu_timing.readback_slot{slot}")),
                        size: buffer_size,
                        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    },
                )?;

                self.timestamp_query_set[slot] = Some(query_set);
                self.timestamp_buffer[slot] = Some(timestamp_buffer);
                self.timestamp_readback_buffer[slot] = Some(timestamp_readback);
            }
        }

        // Initialize pipeline statistics queries
        if self.supports_pipeline_stats {
            let stats_query_set = self.device.create_query_set(&QuerySetDescriptor {
                label: Some("gpu_timing_pipeline_stats"),
                ty: QueryType::PipelineStatistics(
                    PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
                        | PipelineStatisticsTypes::CLIPPER_INVOCATIONS
                        | PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS
                        | PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS,
                ),
                count: query_count,
            });

            let stats_buffer_size = (query_count as u64) * std::mem::size_of::<u64>() as u64 * 4; // 4 stats

            let pipeline_stats_buffer = tracked_create_buffer(
                &self.device,
                &BufferDescriptor {
                    label: Some("gpu_timing.pipeline_stats_resolve"),
                    size: stats_buffer_size,
                    usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            )?;

            let pipeline_stats_readback = tracked_create_buffer(
                &self.device,
                &BufferDescriptor {
                    label: Some("gpu_timing.pipeline_stats_readback"),
                    size: stats_buffer_size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )?;

            self.pipeline_stats_query_set = Some(stats_query_set);
            self.pipeline_stats_buffer = Some(pipeline_stats_buffer);
            self.pipeline_stats_readback_buffer = Some(pipeline_stats_readback);
        }

        Ok(())
    }

    /// Begin a new timing scope
    pub fn begin_scope<'a>(
        &'a mut self,
        encoder: &'a mut CommandEncoder,
        label: &str,
    ) -> TimingScopeId {
        let slot = self.frame_parity;
        let scope_id = TimingScopeId(self.scope_labels[slot].len());
        self.scope_labels[slot].push(label.to_string());
        self.scope_draw_calls[slot].push(0);
        self.active_scopes.insert(scope_id, label.to_string());

        self.begin_scope_internal(encoder, scope_id, label);
        scope_id
    }

    fn begin_scope_internal(
        &mut self,
        encoder: &mut CommandEncoder,
        _scope_id: TimingScopeId,
        label: &str,
    ) {
        let full_label = format!("{}.{}", self.config.label_prefix, label);

        // Insert debug marker for external profilers (RenderDoc, Nsight, RGP)
        if self.config.enable_debug_markers {
            encoder.push_debug_group(&full_label);
        }

        // Insert timestamp query into the current slot.
        let slot = self.frame_parity;
        if let Some(ref query_set) = self.timestamp_query_set[slot] {
            if self.query_index[slot] < self.config.max_queries_per_frame * 2 {
                encoder.write_timestamp(query_set, self.query_index[slot]);
                self.query_index[slot] += 1;
            }
        }
    }

    /// End timing scope
    pub fn end_scope(&mut self, encoder: &mut CommandEncoder, scope_id: TimingScopeId) {
        self.end_scope_internal(encoder, scope_id);
    }

    /// End timing scope, recording the number of draw calls issued inside it.
    pub fn end_scope_with_draws(
        &mut self,
        encoder: &mut CommandEncoder,
        scope_id: TimingScopeId,
        draw_calls: u32,
    ) {
        let slot = self.frame_parity;
        if let Some(entry) = self.scope_draw_calls[slot].get_mut(scope_id.0) {
            *entry = draw_calls;
        }
        self.end_scope_internal(encoder, scope_id);
    }

    fn end_scope_internal(&mut self, encoder: &mut CommandEncoder, scope_id: TimingScopeId) {
        // Insert end timestamp into the current slot.
        let slot = self.frame_parity;
        if let Some(ref query_set) = self.timestamp_query_set[slot] {
            if self.query_index[slot] < self.config.max_queries_per_frame * 2 {
                encoder.write_timestamp(query_set, self.query_index[slot]);
                self.query_index[slot] += 1;
            }
        }

        // Pop debug marker
        if self.config.enable_debug_markers {
            encoder.pop_debug_group();
        }

        self.active_scopes.remove(&scope_id);
    }

    /// Resolve timing queries and prepare for readback.
    ///
    /// Resolves the current slot (`frame_parity`) into its own readback buffer.
    /// Per-slot query counts and scope labels are retained so a later
    /// `get_results` / `get_results_blocking` can map them back to names.
    pub fn resolve_queries(&mut self, encoder: &mut CommandEncoder) {
        let slot = self.frame_parity;
        if self.query_index[slot] == 0 {
            return; // No queries to resolve
        }

        // Resolve timestamp queries for this slot.
        if let (Some(ref query_set), Some(ref buffer)) = (
            &self.timestamp_query_set[slot],
            &self.timestamp_buffer[slot],
        ) {
            encoder.resolve_query_set(query_set, 0..self.query_index[slot], buffer, 0);

            if let Some(ref readback_buffer) = &self.timestamp_readback_buffer[slot] {
                let size = (self.query_index[slot] as u64) * std::mem::size_of::<u64>() as u64;
                encoder.copy_buffer_to_buffer(buffer, 0, readback_buffer, 0, size);
            }
        }

        // Resolve pipeline statistics queries (single-buffered).
        //
        // DEFENSIVE: only resolve when at least one statistics query was
        // actually begun this frame. The scopes above write timestamps only and
        // never begin a pipeline-statistics query, so `pipeline_stats_query_index`
        // stays 0 and this range is skipped. Resolving a never-written
        // statistics range loses the device (→ panic at the next submit) on
        // wgpu 0.19 / Vulkan adapters that advertise PIPELINE_STATISTICS_QUERY,
        // even when `enable_pipeline_stats` was requested by the config.
        if self.pipeline_stats_query_index > 0 {
            if let (Some(ref query_set), Some(ref buffer)) =
                (&self.pipeline_stats_query_set, &self.pipeline_stats_buffer)
            {
                let stats_count = self.pipeline_stats_query_index;
                encoder.resolve_query_set(query_set, 0..stats_count, buffer, 0);

                if let Some(ref readback_buffer) = &self.pipeline_stats_readback_buffer {
                    let size = (stats_count as u64) * std::mem::size_of::<u64>() as u64 * 4;
                    encoder.copy_buffer_to_buffer(buffer, 0, readback_buffer, 0, size);
                }
            }
        }
    }

    /// Advance the double buffer. Call AFTER submitting the encoder that
    /// `resolve_queries` wrote into. Flips parity and clears the slot the next
    /// frame will write into (it was drained one frame earlier).
    pub fn finish_frame(&mut self) {
        self.frame_parity = (self.frame_parity + 1) % SLOTS;
        let slot = self.frame_parity;
        self.query_index[slot] = 0;
        self.scope_labels[slot].clear();
        self.scope_draw_calls[slot].clear();
        self.pipeline_stats_query_index = 0;
    }

    /// Get timing results for the previously submitted frame (async, viewer path).
    ///
    /// Reads the slot NOT currently being written (submitted last frame), so
    /// waiting on it does not stall the frame in flight. If that slot has no
    /// pending queries (e.g. the very first frame), falls back to the current
    /// slot — safe when the caller has already submitted and polled it.
    pub async fn get_results(&mut self) -> RenderResult<Vec<TimingResult>> {
        // Prefer the previously submitted slot; fall back to the current one.
        let other = (self.frame_parity + 1) % SLOTS;
        let read_slot = if self.query_index[other] >= 2 {
            other
        } else {
            self.frame_parity
        };

        self.read_results_from_slot(read_slot).await
    }

    /// Try to read the previously submitted frame without waiting for GPU
    /// progress. If its mapping is not ready after a non-blocking poll, discard
    /// that timing sample so the slot can be reused next frame.
    pub fn try_get_results(&mut self) -> RenderResult<Vec<TimingResult>> {
        let slot = (self.frame_parity + 1) % SLOTS;
        if self.query_index[slot] < 2 {
            return Ok(Vec::new());
        }

        let count = self.query_index[slot];
        let Some(readback_buffer) = self.timestamp_readback_buffer[slot].as_ref() else {
            return Ok(Vec::new());
        };
        let size = (count as u64) * std::mem::size_of::<u64>() as u64;
        let slice = readback_buffer.slice(0..size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(Maintain::Poll);

        match receiver.try_recv() {
            Ok(Ok(())) => {
                let data = slice.get_mapped_range();
                let timestamps = bytemuck::cast_slice::<u8, u64>(&data).to_vec();
                drop(data);
                readback_buffer.unmap();
                Ok(self.timing_results_from_timestamps(slot, &timestamps))
            }
            Ok(Err(error)) => Err(RenderError::Readback(format!(
                "Failed to map timestamp buffer: {error}"
            ))),
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                readback_buffer.unmap();
                Ok(Vec::new())
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err(RenderError::Readback(
                "Timestamp buffer mapping was cancelled".to_string(),
            )),
        }
    }

    /// Get timing results for the slot just resolved + submitted (offline single-shot).
    ///
    /// Reads the current slot (`frame_parity`) and blocks on device work with an
    /// explicit `Maintain::Wait`. Intended for single-submit offline renders.
    ///
    /// Consumes and resets the current slot on success: this is the offline
    /// counterpart to `finish_frame`'s per-slot reset, so a caller that repeats
    /// `resolve_queries -> submit -> get_results_blocking` without ever calling
    /// `finish_frame` does not accumulate unbounded per-slot state or
    /// re-report scopes from a prior call.
    pub fn get_results_blocking(&mut self) -> RenderResult<Vec<TimingResult>> {
        let slot = self.frame_parity;
        let results = pollster::block_on(self.read_results_from_slot(slot))?;
        self.query_index[slot] = 0;
        self.scope_labels[slot].clear();
        self.scope_draw_calls[slot].clear();
        self.pipeline_stats_query_index = 0;
        Ok(results)
    }

    async fn read_results_from_slot(&mut self, slot: usize) -> RenderResult<Vec<TimingResult>> {
        if self.query_index[slot] < 2 {
            return Ok(Vec::new()); // Need at least one begin/end pair
        }

        let count = self.query_index[slot];
        let timestamps = if let Some(ref readback_buffer) = &self.timestamp_readback_buffer[slot] {
            self.read_timestamp_buffer(readback_buffer, count).await?
        } else {
            Vec::new()
        };

        Ok(self.timing_results_from_timestamps(slot, &timestamps))
    }

    fn timing_results_from_timestamps(&self, slot: usize, timestamps: &[u64]) -> Vec<TimingResult> {
        let mut results = Vec::new();
        let scope_count = timestamps.len() / 2;
        for scope_index in 0..scope_count {
            let begin_idx = scope_index * 2;
            let end_idx = begin_idx + 1;
            let begin_ns = timestamps[begin_idx] as f64 * self.timestamp_period;
            let end_ns = timestamps[end_idx] as f64 * self.timestamp_period;
            let duration_ms = (end_ns - begin_ns) / 1_000_000.0; // Convert ns to ms

            let name = self.scope_labels[slot]
                .get(scope_index)
                .cloned()
                .unwrap_or_else(|| format!("scope_{}", scope_index));
            let draw_calls = self.scope_draw_calls[slot]
                .get(scope_index)
                .copied()
                .unwrap_or(0);

            // Honest validity (CENSOR audit F-04): a zero begin stamp means the
            // query was never written (driver skipped it), and end < begin is
            // wraparound/garbage. Consumers must report such scopes as 0.0
            // rather than presenting a garbage delta as a live timing.
            let timestamp_valid =
                timestamps[begin_idx] != 0 && timestamps[end_idx] >= timestamps[begin_idx];

            results.push(TimingResult {
                name,
                gpu_time_ms: duration_ms as f32,
                timestamp_valid,
                draw_calls,
                pipeline_stats: None, // Pipeline stats are not collected yet.
            });
        }

        results
    }

    async fn read_timestamp_buffer(&self, buffer: &Buffer, count: u32) -> RenderResult<Vec<u64>> {
        let size = (count as u64) * std::mem::size_of::<u64>() as u64;
        let slice = buffer.slice(0..size);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(Maintain::Wait);

        match receiver.receive().await {
            Some(Ok(())) => {
                let data = slice.get_mapped_range();
                let timestamps = bytemuck::cast_slice::<u8, u64>(&data).to_vec();
                drop(data);
                buffer.unmap();
                Ok(timestamps)
            }
            Some(Err(e)) => Err(RenderError::Readback(format!(
                "Failed to map timestamp buffer: {}",
                e
            ))),
            None => Err(RenderError::Readback(
                "Timestamp buffer mapping was cancelled".to_string(),
            )),
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &GpuTimingConfig {
        &self.config
    }

    /// Check if timing features are supported
    pub fn is_supported(&self) -> bool {
        self.supports_timestamps || self.supports_pipeline_stats
    }

    /// Get feature support information
    pub fn get_support_info(&self) -> HashMap<String, bool> {
        let mut info = HashMap::new();
        info.insert("timestamps".to_string(), self.supports_timestamps);
        info.insert("pipeline_stats".to_string(), self.supports_pipeline_stats);
        info.insert(
            "debug_markers".to_string(),
            self.config.enable_debug_markers,
        );
        info
    }
}

/// Convenience macro for timing GPU work
#[macro_export]
macro_rules! gpu_time {
    ($timing_manager:expr, $encoder:expr, $label:expr, $body:expr) => {{
        let scope_id = $timing_manager.begin_scope($encoder, $label);
        let result = $body;
        $timing_manager.end_scope($encoder, scope_id);
        result
    }};
}

/// Create default GPU timing configuration based on device features
pub fn create_default_config(device: &Device) -> GpuTimingConfig {
    let features = device.features();

    GpuTimingConfig {
        enable_timestamps: features.contains(Features::TIMESTAMP_QUERY),
        enable_pipeline_stats: features.contains(Features::PIPELINE_STATISTICS_QUERY),
        enable_debug_markers: true,
        label_prefix: "forge3d".to_string(),
        max_queries_per_frame: 32,
    }
}

/// One-shot per-render GPU timing for certified offscreen render paths
/// (CENSOR audit F-04).
///
/// Wraps an optional [`GpuTimingManager`]: present when the initialized global
/// device granted `TIMESTAMP_QUERY`, absent otherwise. Callers wrap each GPU
/// pass in `begin`/`end`, call `resolve` on the encoder BEFORE submitting it,
/// and after submission call [`OneShotTiming::record_into_certificate`], which
/// blocking-reads the timestamps and records one certificate pass per scope
/// with its live `gpu_ms`. When timing is unavailable the caller must fall
/// back to recording the same pass labels (same order) with `gpu_ms == 0.0`
/// so certificate pass lists stay identical across adapters — 0.0 is the
/// honest "no timestamps" value, never a fabricated timing.
pub struct OneShotTiming {
    manager: Option<GpuTimingManager>,
    failure: Option<String>,
    labels: Vec<String>,
}

impl OneShotTiming {
    /// Build a timing manager from the already-initialized global GPU context.
    /// Returns an inert instance when no context exists, `TIMESTAMP_QUERY` was
    /// not granted, or manager construction fails (each pass then records 0.0
    /// and the `capability_absent` degradation already explains why).
    pub fn for_current_device() -> Self {
        let Some(ctx) = crate::core::gpu::ctx_if_initialized() else {
            return Self::inert();
        };
        Self::for_device(ctx.device.clone(), ctx.queue.clone())
    }

    /// Build a timing manager for an explicit device/queue pair — for render
    /// paths that own their device instead of using the global context (e.g.
    /// `TerrainSpike`). Returns an inert instance when `TIMESTAMP_QUERY` was
    /// not granted or manager construction fails (each pass then records 0.0).
    pub fn for_device(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        if !device.features().contains(Features::TIMESTAMP_QUERY) {
            return Self::inert();
        }
        let config = GpuTimingConfig {
            enable_timestamps: true,
            // Timestamps only: enabling statistics here would resolve a
            // never-written statistics range (see Scene::take_render_timing).
            enable_pipeline_stats: false,
            enable_debug_markers: false,
            label_prefix: "forge3d".to_string(),
            max_queries_per_frame: 32,
        };
        match GpuTimingManager::new(device, queue, config) {
            Ok(manager) => Self {
                manager: Some(manager),
                failure: None,
                labels: Vec::new(),
            },
            Err(e) => Self {
                manager: None,
                failure: Some(format!("timing manager construction failed: {e}")),
                labels: Vec::new(),
            },
        }
    }

    fn inert() -> Self {
        Self {
            manager: None,
            failure: None,
            labels: Vec::new(),
        }
    }

    /// True when live timestamps will be recorded (used by callers to decide
    /// whether the 0.0 fallback pass records are needed).
    pub fn is_live(&self) -> bool {
        self.manager.is_some()
    }

    /// Open a timing scope around a GPU pass. Must be called with the encoder
    /// outside any active render/compute pass.
    pub fn begin(&mut self, encoder: &mut CommandEncoder, label: &str) -> Option<TimingScopeId> {
        self.labels.push(label.to_string());
        self.manager.as_mut().map(|t| t.begin_scope(encoder, label))
    }

    /// Close a scope opened by [`OneShotTiming::begin`], recording draw count.
    pub fn end(
        &mut self,
        encoder: &mut CommandEncoder,
        scope: Option<TimingScopeId>,
        draw_calls: u32,
    ) {
        if let (Some(t), Some(id)) = (self.manager.as_mut(), scope) {
            t.end_scope_with_draws(encoder, id, draw_calls);
        }
    }

    /// Resolve the query set into its readback buffer. Must be recorded on an
    /// encoder that is submitted before [`OneShotTiming::record_into_certificate`].
    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        if let Some(t) = self.manager.as_mut() {
            t.resolve_queries(encoder);
        }
    }

    /// Blocking-read the resolved timestamps and record every VALID scope into
    /// the active certificate capture with its live timing. Returns `true`
    /// when live passes were recorded; on `false` the caller records its 0.0
    /// fallback passes instead.
    pub fn record_into_certificate(mut self) -> bool {
        if let Some(consequence) = self.failure.as_deref() {
            self.record_failure(consequence);
            return false;
        }
        let Some(manager) = self.manager.as_mut() else {
            return false;
        };
        match manager.get_results_blocking() {
            Ok(results) if !results.is_empty() => {
                for result in &results {
                    if !result.timestamp_valid {
                        crate::core::degradation::record_degradation(
                            "timing_unavailable",
                            &result.name,
                            "timestamp query returned an unwritten or non-monotonic interval",
                        );
                    }
                    crate::core::certificate::record_pass(
                        &result.name,
                        result.certificate_gpu_ms(),
                        result.draw_calls,
                    );
                }
                true
            }
            Ok(_) => {
                self.record_failure("timestamp query resolved without any pass results");
                false
            }
            Err(e) => {
                self.record_failure(&format!("timestamp query readback failed: {e}"));
                false
            }
        }
    }

    fn record_failure(&self, consequence: &str) {
        for label in &self.labels {
            crate::core::degradation::record_degradation("timing_unavailable", label, consequence);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CENSOR Task 4: end-to-end timestamp measurement over a real GPU submit.
    ///
    /// Times a buffer-to-buffer copy inside a begin/end scope, resolves, submits,
    /// and reads back with `get_results_blocking`. Skips cleanly when no adapter
    /// is available or TIMESTAMP_QUERY was not granted.
    #[test]
    fn gpu_timing_double_buffered_measures_copy() {
        let Some((device, queue)) = crate::core::gpu::create_device_and_queue_for_test() else {
            eprintln!("[gpu_timing test] no GPU adapter available; skipping");
            return;
        };
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let config = create_default_config(&device);
        if !config.enable_timestamps {
            eprintln!("[gpu_timing test] TIMESTAMP_QUERY not granted; skipping");
            return;
        }

        let mut manager = GpuTimingManager::new(device.clone(), queue.clone(), config)
            .expect("create timing manager");
        assert!(
            manager.is_supported(),
            "timing must be supported once TIMESTAMP_QUERY is granted"
        );

        // Two small buffers to copy between; the copy is what we time.
        let src = tracked_create_buffer(
            &device,
            &BufferDescriptor {
                label: Some("gpu_timing_test_src"),
                size: 4096,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .expect("alloc");
        let dst = tracked_create_buffer(
            &device,
            &BufferDescriptor {
                label: Some("gpu_timing_test_dst"),
                size: 4096,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .expect("alloc");

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_timing"),
        });

        let scope = manager.begin_scope(&mut encoder, "test_copy");
        // Loop the copy so the timed interval is not sub-tick on fast adapters.
        for _ in 0..100 {
            encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 4096);
        }
        manager.end_scope_with_draws(&mut encoder, scope, 100);

        manager.resolve_queries(&mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(Maintain::Wait);

        let results = manager.get_results_blocking().expect("get results");
        assert_eq!(results.len(), 1, "expected exactly one timed scope");
        let r = &results[0];
        assert_eq!(r.name, "test_copy");
        assert!(r.timestamp_valid, "timestamp must be valid");
        assert_eq!(r.draw_calls, 100, "draw_calls must round-trip");
        // Timestamps are monotonic; a real interval is >= 0. The 100x copy makes
        // it typically > 0, but we assert >= 0.0 to stay robust across adapters
        // whose period rounds a tiny copy to zero.
        assert!(
            r.gpu_time_ms >= 0.0,
            "gpu_time_ms must be non-negative, got {}",
            r.gpu_time_ms
        );

        // CENSOR Task 4 regression: a second offline single-shot round
        // (resolve -> submit -> get_results_blocking) WITHOUT an intervening
        // `finish_frame()` must see exactly the new scope, not the old one
        // re-reported on top of it. This is the documented single-shot
        // pattern an offline-render certificate path will use once per
        // render, multiple renders per process.
        let mut encoder2 = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("gpu_timing_2"),
        });
        let scope2 = manager.begin_scope(&mut encoder2, "test_copy");
        encoder2.copy_buffer_to_buffer(&src, 0, &dst, 0, 4096);
        manager.end_scope_with_draws(&mut encoder2, scope2, 1);
        manager.resolve_queries(&mut encoder2);
        queue.submit(std::iter::once(encoder2.finish()));
        device.poll(Maintain::Wait);

        let results2 = manager
            .get_results_blocking()
            .expect("get results (second single-shot read)");
        assert_eq!(
            results2.len(),
            1,
            "get_results_blocking must not accumulate scopes across calls without finish_frame"
        );
        assert_eq!(results2[0].name, "test_copy");

        // Teardown: drop GPU resources explicitly and poll so the backend
        // processes destruction work, then deliberately leak the device/queue.
        // On this driver stack (wgpu 0.19 / Windows), dropping a device that
        // owned timestamp query sets hangs inside the backend even after a
        // final poll; the process is exiting anyway, so leaking is safe and
        // keeps the test harness from deadlocking.
        drop(results);
        drop(results2);
        drop(manager);
        drop(src);
        drop(dst);
        device.poll(Maintain::Wait);
        std::mem::forget(device);
        std::mem::forget(queue);
    }

    #[test]
    fn certificate_gpu_ms_zeros_invalid_timestamps() {
        assert_eq!(
            TimingResult {
                gpu_time_ms: 3.25,
                timestamp_valid: false,
                ..Default::default()
            }
            .certificate_gpu_ms(),
            0.0
        );
        assert_eq!(
            TimingResult {
                gpu_time_ms: 3.25,
                timestamp_valid: true,
                ..Default::default()
            }
            .certificate_gpu_ms(),
            3.25
        );
    }

    #[test]
    fn one_shot_failure_is_render_local_degradation() {
        crate::core::degradation::clear_degradations();
        crate::core::degradation::begin_degradation_capture();
        let timing = OneShotTiming {
            manager: None,
            failure: Some("forced readback failure".to_string()),
            labels: vec!["test.pass".to_string()],
        };
        assert!(!timing.record_into_certificate());
        let first = crate::core::degradation::finish_degradation_capture();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].kind, "timing_unavailable");
        assert_eq!(first[0].name, "test.pass");
        assert_eq!(first[0].consequence, "forced readback failure");

        crate::core::degradation::begin_degradation_capture();
        assert!(crate::core::degradation::finish_degradation_capture().is_empty());
        crate::core::degradation::clear_degradations();
    }
}
