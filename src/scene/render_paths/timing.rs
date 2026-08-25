// CENSOR Task 9: Scene render-path GPU-pass timing.
//
// A process-shared manager preserves one Metal query-set lifetime across
// distinct Scene instances. The process render lock covers both manager use
// and the global certificate capture, so one render still equals one capture.

fn shared_render_timing(
) -> std::sync::Arc<std::sync::Mutex<Option<crate::core::gpu_timing::GpuTimingManager>>> {
    static TIMING: std::sync::OnceLock<
        std::sync::Arc<std::sync::Mutex<Option<crate::core::gpu_timing::GpuTimingManager>>>,
    > = std::sync::OnceLock::new();
    TIMING
        .get_or_init(|| std::sync::Arc::new(std::sync::Mutex::new(None)))
        .clone()
}

fn shared_render_timing_owner() -> &'static crate::core::resource_tracker::AllocationOwner {
    static OWNER: std::sync::OnceLock<crate::core::resource_tracker::AllocationOwner> =
        std::sync::OnceLock::new();
    OWNER.get_or_init(crate::core::resource_tracker::AllocationOwner::new)
}

static SCENE_RENDER_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) struct SceneRenderScope {
    _allocation_scope: crate::core::resource_tracker::AllocationOwnerGuard,
    _render_guard: std::sync::MutexGuard<'static, ()>,
}

impl Scene {
    pub(super) fn shared_render_timing(
    ) -> std::sync::Arc<std::sync::Mutex<Option<crate::core::gpu_timing::GpuTimingManager>>> {
        shared_render_timing()
    }
    pub(super) fn record_terrain_shader_use() {
        let label = if crate::core::gpu::ctx_if_initialized()
            .is_some_and(|ctx| ctx.device.limits().max_bind_groups >= 6)
        {
            "vf.Terrain.shader.full"
        } else {
            "vf.Terrain.shader.minimal"
        };
        crate::core::shader_registry::record_shader_use(label);
    }

    pub(super) fn begin_certificate_capture(
        &self,
        entry_point: &str,
    ) -> (
        crate::core::certificate::RenderCaptureGuard,
        SceneRenderScope,
    ) {
        let (render_guard, lock_poisoned) = match SCENE_RENDER_SERIAL.lock() {
            Ok(guard) => (guard, false),
            Err(poisoned) => (poisoned.into_inner(), true),
        };
        let allocation_scope = self.allocation_owner.activate();
        let render_capture = crate::core::certificate::begin_render_capture_with_resources(
            entry_point,
            &[
                self.allocation_owner.id(),
                shared_render_timing_owner().id(),
            ],
        );
        if lock_poisoned {
            crate::core::degradation::record_degradation(
                "timing_unavailable",
                "scene_render_serial_poisoned",
                "the process render lock recovered from a prior panic; this render remains serialized",
            );
        }
        (
            render_capture,
            SceneRenderScope {
                _allocation_scope: allocation_scope,
                _render_guard: render_guard,
            },
        )
    }

    pub(super) fn finish_certificate_capture(
        &self,
        capture: crate::core::certificate::RenderCaptureGuard,
    ) {
        capture.finish();
    }

    /// Take the render-timing manager out of the scene, lazily constructing it
    /// on first use. Without `TIMESTAMP_QUERY`, the manager still preserves pass
    /// labels and draw counts while reporting `gpu_ms == 0`. Returned via
    /// [`Scene::store_render_timing`].
    pub(super) fn take_render_timing(&self) -> Option<crate::core::gpu_timing::GpuTimingManager> {
        let mut guard = match self.render_timing.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                crate::core::degradation::record_degradation(
                    "timing_unavailable",
                    "scene_timing_manager_poisoned",
                    "the shared timing manager lock recovered from a prior panic",
                );
                poisoned.into_inner()
            }
        };
        if guard.is_none() {
            let Some(g) = crate::core::gpu::ctx_if_initialized() else {
                crate::core::degradation::record_degradation(
                    "timing_unavailable",
                    "scene_gpu_context",
                    "GPU timing was unavailable because the process GPU context was not initialized",
                );
                return None;
            };
            let timestamps_available = g
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY);
            if !timestamps_available {
                crate::core::degradation::record_degradation(
                    "capability_absent",
                    "timestamp_query",
                    "scene passes are reported with gpu_ms equal to 0",
                );
            }
            // Timestamps only: this path never issues pipeline-statistics
            // queries, so enabling that query set would make `resolve_queries`
            // resolve a never-written statistics range and lose the device on
            // adapters that also advertise PIPELINE_STATISTICS_QUERY.
            let config = crate::core::gpu_timing::GpuTimingConfig {
                enable_timestamps: timestamps_available,
                enable_pipeline_stats: false,
                enable_debug_markers: false,
                label_prefix: "forge3d".to_string(),
                max_queries_per_frame: 32,
            };
            let _timing_allocation_scope = shared_render_timing_owner().activate();
            match crate::core::gpu_timing::GpuTimingManager::new(
                g.device.clone(),
                g.queue.clone(),
                config,
            ) {
                Ok(manager) => return Some(manager),
                Err(e) => {
                    log::warn!("failed to create Scene GPU timing manager: {e}");
                    crate::core::degradation::record_degradation(
                        "timing_unavailable",
                        "scene_timing_manager_create",
                        &format!("Scene GPU timing manager creation failed: {e}"),
                    );
                    return None;
                }
            }
        }
        guard.take()
    }

    /// Return a timing manager taken by [`Scene::take_render_timing`].
    pub(super) fn store_render_timing(
        &self,
        manager: Option<crate::core::gpu_timing::GpuTimingManager>,
    ) {
        if let Some(manager) = manager {
            let mut guard = match self.render_timing.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    crate::core::degradation::record_degradation(
                        "timing_unavailable",
                        "scene_timing_manager_poisoned",
                        "the shared timing manager lock recovered from a prior panic",
                    );
                    poisoned.into_inner()
                }
            };
            *guard = Some(manager);
        }
    }

    /// Finalize a render's timing: resolve+read the timestamps and record each
    /// timed pass into the global certificate capture. Must be called AFTER the
    /// encoder that recorded `resolve_queries` was submitted (the readback
    /// encoder does that here). Consumes the manager's current slot.
    pub(super) fn record_render_timings(
        &self,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    ) {
        if let Some(manager) = timing.as_mut() {
            match manager.get_results_blocking() {
                Ok(results) => {
                    for result in results {
                        // Invalid timestamps are recorded as 0.0, never as a
                        // garbage delta (CENSOR audit F-04).
                        crate::core::certificate::record_pass(
                            &result.name,
                            result.certificate_gpu_ms(),
                            result.draw_calls,
                        );
                    }
                }
                Err(e) => log::warn!("Scene GPU timing readback failed: {e}"),
            }
        }
    }
}

/// Open a timing scope around a Scene GPU pass when timing is active. The
/// returned id is threaded to [`scene_ts_end`]; `None` when timing is
/// unavailable. Must be called with the encoder outside any active render pass.
pub(super) fn scene_ts_begin(
    timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
) -> Option<crate::core::gpu_timing::TimingScopeId> {
    timing.as_mut().map(|t| t.begin_scope(encoder, label))
}

/// Close a timing scope opened by [`scene_ts_begin`], recording its draw count.
pub(super) fn scene_ts_end(
    timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    encoder: &mut wgpu::CommandEncoder,
    scope: Option<crate::core::gpu_timing::TimingScopeId>,
    draw_calls: u32,
) {
    if let (Some(t), Some(id)) = (timing.as_mut(), scope) {
        t.end_scope_with_draws(encoder, id, draw_calls);
    }
}
