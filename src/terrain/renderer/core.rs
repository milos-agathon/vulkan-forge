use super::*;
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};

/// Reusable GPU terrain scene (M2).
///
/// Owns the WGPU pipeline state for the PBR+POM terrain path and is free of
/// any PyO3 attributes so it can be reused by the interactive viewer and
/// other Rust callers.
pub struct TerrainScene {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) adapter: Arc<wgpu::Adapter>,
    pub(super) allocation_owner: crate::core::resource_tracker::AllocationOwner,
    pub(super) pipeline: Mutex<PipelineCache>,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) ibl_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) blit_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) blit_pipeline: wgpu::RenderPipeline,
    pub(super) aov_blit_pipeline: wgpu::RenderPipeline,
    pub(super) background_blit_pipeline: wgpu::RenderPipeline,
    pub(super) aether_background_blit_pipeline: wgpu::RenderPipeline,
    pub(super) normal_blit_pipeline: wgpu::RenderPipeline,
    pub(super) offline_compute: OfflineComputeResources,
    pub(super) sampler_linear: wgpu::Sampler,
    pub(super) sky_bind_group_layout0: wgpu::BindGroupLayout,
    pub(super) sky_bind_group_layout1: wgpu::BindGroupLayout,
    pub(super) sky_pipeline: wgpu::ComputePipeline,
    pub(super) aether_sky_bind_group_layout2: wgpu::BindGroupLayout,
    pub(super) aether_sky_pipeline: wgpu::ComputePipeline,
    pub(super) atmosphere_lut_cache: Mutex<Vec<super::atmosphere::luts::AtmosphereGpuLuts>>,
    pub(super) _sky_fallback_texture: TrackedTexture,
    pub(super) sky_fallback_view: wgpu::TextureView,
    pub(super) _atmosphere_scattering_fallback_texture: TrackedTexture,
    pub(super) atmosphere_scattering_fallback_view: wgpu::TextureView,
    pub(super) _height_curve_identity_texture: TrackedTexture,
    pub(super) height_curve_identity_view: wgpu::TextureView,
    pub(super) _water_mask_fallback_texture: TrackedTexture,
    pub(super) water_mask_fallback_view: wgpu::TextureView,
    pub(super) _ao_debug_fallback_texture: TrackedTexture,
    pub(super) ao_debug_fallback_view: wgpu::TextureView,
    pub(super) ao_debug_sampler: wgpu::Sampler,
    pub(super) ao_debug_view: Option<wgpu::TextureView>,
    pub(super) coarse_ao_texture: Option<TrackedTexture>,
    pub(super) coarse_ao_view: Option<wgpu::TextureView>,
    pub(super) detail_normal_fallback_view: wgpu::TextureView,
    pub(super) detail_normal_sampler: wgpu::Sampler,
    pub(super) height_ao_fallback_view: wgpu::TextureView,
    pub(super) height_ao_sampler: wgpu::Sampler,
    pub(super) sun_vis_fallback_view: wgpu::TextureView,
    pub(super) sun_vis_sampler: wgpu::Sampler,
    pub(super) height_ao_compute_pipeline: wgpu::ComputePipeline,
    pub(super) height_ao_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) height_ao_uniform_buffer: TrackedBuffer,
    pub(super) height_ao_texture: Mutex<Option<TrackedTexture>>,
    pub(super) height_ao_storage_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) height_ao_sample_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) height_ao_size: Mutex<(u32, u32)>,
    pub(super) sun_vis_compute_pipeline: wgpu::ComputePipeline,
    pub(super) sun_vis_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) sun_vis_uniform_buffer: TrackedBuffer,
    pub(super) sun_vis_texture: Mutex<Option<TrackedTexture>>,
    pub(super) sun_vis_storage_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) sun_vis_sample_view: Mutex<Option<wgpu::TextureView>>,
    pub(super) sun_vis_size: Mutex<(u32, u32)>,
    pub(super) height_curve_lut_sampler: wgpu::Sampler,
    pub(super) color_format: wgpu::TextureFormat,
    pub(super) light_buffer: Arc<Mutex<LightBuffer>>,
    pub(super) light_override: Mutex<Option<Vec<Light>>>,
    pub(super) noop_shadow: NoopShadow,
    pub(super) csm_renderer: crate::shadows::CsmRenderer,
    pub(super) shadow_depth_pipeline: wgpu::RenderPipeline,
    pub(super) shadow_depth_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) shadow_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) shadow_pcss_radius: f32,
    pub(super) shadow_technique: u32,
    pub(super) moment_pass: Option<crate::shadows::MomentGenerationPass>,
    pub(super) moment_blur_pass: Option<crate::shadows::ShadowBlurPass>,
    pub(super) fog_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) fog_uniform_buffer: TrackedBuffer,
    pub(super) water_reflection_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) water_reflection_uniform_buffer: TrackedBuffer,
    pub(super) water_reflection_texture: Mutex<TrackedTexture>,
    pub(super) water_reflection_view: Mutex<wgpu::TextureView>,
    pub(super) water_reflection_sampler: wgpu::Sampler,
    pub(super) water_reflection_depth_texture: Mutex<TrackedTexture>,
    pub(super) water_reflection_depth_view: Mutex<wgpu::TextureView>,
    pub(super) water_reflection_size: Mutex<(u32, u32)>,
    pub(super) water_reflection_fallback_view: wgpu::TextureView,
    pub(super) water_reflection_pipeline: wgpu::RenderPipeline,
    pub(super) material_layer_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) visibility_resolve_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) material_layer_uniform_buffer: TrackedBuffer,
    pub(super) vt_uniform_buffer: TrackedBuffer,
    pub(super) vt_fallback_uniform_buffer: TrackedBuffer,
    pub(super) _vt_atlas_fallback_textures: Vec<TrackedTexture>,
    pub(super) vt_atlas_fallback_views: Vec<wgpu::TextureView>,
    pub(super) _vt_page_table_fallback_texture: TrackedTexture,
    pub(super) vt_page_table_fallback_view: wgpu::TextureView,
    pub(super) vt_feedback_fallback_buffer: TrackedBuffer,
    /// Shader-written per-frame counters: material invocations, logical
    /// feedback records, fallback texels, and forward invocations.
    pub(super) vt_frame_counters_buffer: TrackedBuffer,
    pub(super) vt_atlas_sampler: wgpu::Sampler,
    pub(super) probe_grid_uniform_buffer: TrackedBuffer,
    pub(super) probe_ssbo: TrackedBuffer,
    pub(super) probe_grid_uniform_alloc_bytes: u64,
    pub(super) probe_ssbo_alloc_bytes: u64,
    pub(super) probe_grid_uniform_bytes: u64,
    pub(super) probe_ssbo_bytes: u64,
    pub(super) probe_cache_key: Option<u64>,
    pub(super) probe_cached_grid: Option<crate::terrain::probes::ProbeGridDesc>,
    pub(super) probe_cached_data: Vec<crate::terrain::probes::GpuProbeData>,
    pub(super) reflection_probe_grid_uniform_buffer: TrackedBuffer,
    pub(super) reflection_probe_sampler: wgpu::Sampler,
    pub(super) reflection_probe_fallback_texture: TrackedTexture,
    pub(super) _reflection_probe_fallback_view: wgpu::TextureView,
    pub(super) reflection_probe_texture: Option<TrackedTexture>,
    pub(super) reflection_probe_view: wgpu::TextureView,
    pub(super) reflection_probe_grid_uniform_alloc_bytes: u64,
    pub(super) reflection_probe_grid_uniform_bytes: u64,
    pub(super) reflection_probe_texture_alloc_bytes: u64,
    pub(super) reflection_probe_texture_bytes: u64,
    pub(super) reflection_probe_cache_key: Option<u64>,
    pub(super) reflection_probe_cached_grid: Option<crate::terrain::probes::ProbeGridDesc>,
    pub(super) reflection_probe_count: u32,
    pub(super) reflection_probe_resolution: u32,
    pub(super) reflection_probe_mip_levels: u32,
    pub(super) aov_pipeline: Mutex<Option<wgpu::RenderPipeline>>,
    pub(super) aov_pipeline_sample_count: Mutex<u32>,
    /// Bitmask of the albedo/normal/depth color targets carried by the cached
    /// one-shot AOV pipeline. Depth-only evidence must not pay for two unused
    /// RGBA16F attachments on every frame.
    pub(super) aov_pipeline_output_mask: Mutex<u8>,
    /// VERITAS: whether the cached AOV pipeline carries the 5th R32Uint
    /// source-id target (rebuilt when this toggles, like sample count).
    pub(super) aov_pipeline_source_id: Mutex<bool>,
    /// BOP-P2-02: whether the cached AOV pipeline consumes the indexed
    /// clipmap vertex stream (rebuilt when this toggles, like sample count).
    pub(super) aov_pipeline_clipmap: Mutex<bool>,
    pub(super) _dof_renderer: Mutex<Option<crate::core::dof::DofRenderer>>,
    pub(super) offline_state: Mutex<Option<OfflineAccumulationState>>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_renderer: crate::render::mesh_instanced::MeshInstancedRenderer,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_renderer_sample_count: u32,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_batches: Vec<crate::terrain::scatter::TerrainScatterBatch>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_last_frame_stats: crate::terrain::scatter::TerrainScatterFrameStats,
    #[cfg(feature = "enable-renderer-config")]
    pub(super) config: Arc<Mutex<crate::render::params::RendererConfig>>,
    pub(super) material_vt: Mutex<super::virtual_texture::TerrainMaterialVT>,
    pub(super) visibility_buffer: Mutex<Option<super::visibility_buffer::TerrainVisibilityBuffer>>,
    pub(super) cpu_visibility_oracle: Mutex<Option<super::visibility_buffer::CpuVisibilityOracle>>,
    pub(super) viewer_heightmap: Option<ViewerTerrainData>,
    pub(super) geometry_provider: Option<TerrainGeometryProvider>,
    pub(super) two_phase_culler: Option<crate::terrain::culling::two_phase::TwoPhaseTerrainCuller>,
    pub(super) terrain_minmax_pyramid:
        Option<crate::path_tracing::hybrid_compute::terrain_heightfield::TerrainMinMaxPyramid>,
    pub(super) culling_stats: crate::terrain::culling::two_phase::CullingStats,
    pub(super) height_streaming: Option<super::streaming::HeightVtFamilyRuntime>,
    /// CENSOR Task 9: owned per-render GPU timing manager, lazily constructed on
    /// the first render when the device granted `TIMESTAMP_QUERY`. Stored behind
    /// a `Mutex<Option<..>>` because the draw methods borrow `&self`; a render
    /// takes it out, scopes each pass, then puts it back.
    pub(super) gpu_timing: Mutex<Option<crate::core::gpu_timing::GpuTimingManager>>,
    pub(super) _tracked_scene_textures: Vec<crate::core::resource_tracker::ResourceHandle>,
}

pub struct ViewerTerrainData {
    pub heightmap: Vec<f32>,
    pub dimensions: (u32, u32),
    pub domain: (f32, f32),
    pub heightmap_texture: TrackedTexture,
    pub heightmap_view: wgpu::TextureView,
    pub vertex_buffer: TrackedBuffer,
    pub index_buffer: TrackedBuffer,
    pub index_count: u32,
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_fov_deg: f32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
}

#[pyclass(module = "forge3d._forge3d", name = "TerrainRenderer")]
pub struct TerrainRenderer {
    pub(super) scene: TerrainScene,
    pub(super) last_anamnesis_report: crate::core::anamnesis::CacheReport,
    pub(super) last_anamnesis_graph_submissions: usize,
}

pub(super) struct OfflineAccumulationState {
    pub(super) params: crate::terrain::render_params::TerrainRenderParams,
    pub(super) decoded: crate::terrain::render_params::DecodedTerrainSettings,
    pub(super) height_inputs: super::draw::UploadedHeightInputs,
    pub(super) materials: super::draw::PreparedMaterials,
    pub(super) ibl_bind_group: wgpu::BindGroup,
    pub(super) height_curve_lut_uploaded: Option<(TrackedTexture, wgpu::TextureView)>,
    pub(super) hdr_aov_pipeline: wgpu::RenderPipeline,
    pub(super) hdr_background_blit_pipeline: wgpu::RenderPipeline,
    pub(super) render_targets: super::draw::RenderTargets,
    pub(super) aov_targets: super::aov::TerrainAovTargets,
    pub(super) beauty_accumulation: crate::terrain::AccumulationBuffer,
    pub(super) albedo_accumulation: crate::terrain::AccumulationBuffer,
    pub(super) normal_accumulation: crate::terrain::AccumulationBuffer,
    pub(super) _depth_reference_texture: TrackedTexture,
    pub(super) depth_reference_view: wgpu::TextureView,
    pub(super) luminance_texture: TrackedTexture,
    pub(super) luminance_view: wgpu::TextureView,
    pub(super) luminance_width: u32,
    pub(super) luminance_height: u32,
    pub(super) jitter_sequence: crate::terrain::JitterSequence,
    pub(super) total_samples: u32,
    pub(super) out_width: u32,
    pub(super) out_height: u32,
    pub(super) internal_width: u32,
    pub(super) internal_height: u32,
    pub(super) needs_scaling: bool,
    pub(super) prev_tile_means: Vec<f32>,
    pub(super) prev_tile_mean_history: Vec<Vec<f32>>,
    pub(super) prev_tile_size: u32,
}

pub(super) struct OfflineComputeResources {
    pub(super) accumulate_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) accumulate_pipeline: wgpu::ComputePipeline,
    pub(super) resolve_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) resolve_pipeline: wgpu::ComputePipeline,
    pub(super) depth_extract_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) depth_extract_pipeline: wgpu::ComputePipeline,
    pub(super) depth_expand_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) depth_expand_pipeline: wgpu::ComputePipeline,
    pub(super) luminance_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) luminance_pipeline: wgpu::ComputePipeline,
    pub(super) tonemap_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) tonemap_pipeline: wgpu::ComputePipeline,
}

pub(super) struct NoopShadow {
    pub(super) _csm_uniform_buffer: TrackedBuffer,
    pub(super) _shadow_maps_texture: TrackedTexture,
    pub(super) _shadow_maps_view: wgpu::TextureView,
    pub(super) _shadow_sampler: wgpu::Sampler,
    pub(super) _moment_maps_texture: TrackedTexture,
    pub(super) moment_maps_view: wgpu::TextureView,
    pub(super) moment_sampler: wgpu::Sampler,
    pub(super) bind_group: wgpu::BindGroup,
}

pub(super) struct OverlayBinding {
    pub(super) uniform: OverlayUniforms,
    pub(super) lut: Option<Arc<crate::terrain::ColormapLUT>>,
}

pub(super) struct PipelineCache {
    pub(super) sample_count: u32,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) clipmap_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) visibility_write_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) visibility_resolve_pipeline: Option<wgpu::RenderPipeline>,
}

pub(super) const TERRAIN_DEFAULT_CASCADE_SPLITS: [f32; 4] = [50.0, 200.0, 800.0, 3000.0];
pub(super) const MATERIAL_LAYER_CAPACITY: usize = 4;
pub(super) const TERRAIN_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub(super) fn terrain_camera_mode_tag(camera_mode: &str) -> &str {
    camera_mode.split(':').next().unwrap_or(camera_mode).trim()
}

pub(super) fn is_mesh_camera_mode(camera_mode: &str) -> bool {
    terrain_camera_mode_tag(camera_mode).eq_ignore_ascii_case("mesh")
}

pub(super) fn is_clipmap_camera_mode(camera_mode: &str) -> bool {
    terrain_camera_mode_tag(camera_mode).eq_ignore_ascii_case("clipmap")
}

/// True when the camera mode carries the `zup` option (e.g. `"mesh:zup"`).
///
/// Mesh-mode terrain lives in the world XY plane with heights along +Z, but
/// the legacy orbit camera is parameterized around +Y with `up = Vec3::Y` —
/// a grid axis — so oblique views render rolled and north-south inverted.
/// `zup` opts into a Z-up orbit (theta = polar angle from +Z, 0 looks straight
/// down; phi = azimuth in the terrain plane) with `up = Vec3::Z`, matching the
/// interactive viewer's terrain semantics. Kept as an opt-in suffix so
/// existing `"mesh"` callers keep byte-identical output.
pub(super) fn is_zup_camera_mode(camera_mode: &str) -> bool {
    camera_mode
        .split(':')
        .skip(1)
        .any(|part| part.trim().eq_ignore_ascii_case("zup"))
}

pub(super) fn clipmap_camera_config(
    camera_mode: &str,
) -> Option<crate::terrain::clipmap::ClipmapConfig> {
    if !is_clipmap_camera_mode(camera_mode) {
        return None;
    }
    let mut parts = camera_mode.split(':').skip(1);
    let ring_count = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(4)
        .clamp(1, 8);
    let ring_resolution = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(64)
        .clamp(4, 256);
    let center_resolution = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(ring_resolution)
        .clamp(4, 256);
    let skirt_depth = parts
        .next()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(10.0)
        .max(0.0);
    let morph_range = parts
        .next()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.3)
        .clamp(0.0, 1.0);
    Some(crate::terrain::clipmap::ClipmapConfig {
        ring_count,
        ring_resolution,
        center_resolution,
        skirt_depth,
        morph_range,
    })
}

impl TerrainScene {
    pub(super) fn begin_certificate_capture(
        &self,
        entry_point: &str,
    ) -> (
        crate::core::certificate::RenderCaptureGuard,
        crate::core::resource_tracker::AllocationOwnerGuard,
    ) {
        let allocation_scope = self.allocation_owner.activate();
        let render_capture = crate::core::certificate::begin_render_capture_with_resources(
            entry_point,
            &[self.allocation_owner.id()],
        );
        if let Some(streaming) = &self.height_streaming {
            streaming.mosaic.record_certificate_usage();
        }
        (render_capture, allocation_scope)
    }

    pub(super) fn finish_certificate_capture(
        &self,
        capture: crate::core::certificate::RenderCaptureGuard,
    ) {
        capture.finish();
    }
}

impl TerrainScene {
    /// Vertex count for the procedural-grid (`vs_main`) draws.
    ///
    /// The beauty/AOV/offline passes draw through `TerrainGeometryProvider`,
    /// which uses the indexed clipmap mesh for clipmap camera modes; the
    /// clipmap branch here only serves the secondary passes (water
    /// reflection, shadow cascades) that keep a bounded procedural-grid
    /// proxy of the terrain.
    pub(super) fn terrain_vertex_count(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
    ) -> u32 {
        if is_clipmap_camera_mode(&params.camera_mode) {
            let grid_size = clipmap_camera_config(&params.camera_mode)
                .map(|config| config.ring_resolution.max(4))
                .unwrap_or(64);
            return 6 * (grid_size - 1) * (grid_size - 1);
        }
        if is_mesh_camera_mode(&params.camera_mode) {
            let grid_size: u32 = 512;
            return 6 * (grid_size - 1) * (grid_size - 1);
        }
        3
    }
}

// ---------------------------------------------------------------------------
// CENSOR Task 9: per-render GPU-pass timing helpers.
// ---------------------------------------------------------------------------

/// Open a timing scope around a GPU pass when timing is active. The returned
/// id is threaded to [`ts_end`]; `None` when timing is unavailable.
pub(super) fn ts_begin(
    timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
) -> Option<crate::core::gpu_timing::TimingScopeId> {
    timing.as_mut().map(|t| t.begin_scope(encoder, label))
}

/// Close a timing scope opened by [`ts_begin`], recording its draw-call count.
pub(super) fn ts_end(
    timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    encoder: &mut wgpu::CommandEncoder,
    scope: Option<crate::core::gpu_timing::TimingScopeId>,
    draw_calls: u32,
) {
    if let (Some(t), Some(id)) = (timing.as_mut(), scope) {
        t.end_scope_with_draws(encoder, id, draw_calls);
    }
}

impl TerrainScene {
    /// Take the render-timing manager out of the scene, lazily constructing it
    /// the first time when the device granted `TIMESTAMP_QUERY`. Returns `None`
    /// when timestamps are unavailable (the certificate then reports the passes
    /// with `gpu_ms == 0`). The caller returns it via [`store_render_timing`].
    pub(super) fn take_render_timing(&self) -> Option<crate::core::gpu_timing::GpuTimingManager> {
        let mut guard = self.gpu_timing.lock().ok()?;
        if guard.is_none() {
            if !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
            {
                return None;
            }
            // Timestamps only: this path never issues pipeline-statistics
            // queries, so enabling that query set would make `resolve_queries`
            // resolve a never-written statistics range and lose the device on
            // adapters that also advertise PIPELINE_STATISTICS_QUERY.
            let config = crate::core::gpu_timing::GpuTimingConfig {
                enable_timestamps: true,
                enable_pipeline_stats: false,
                enable_debug_markers: false,
                label_prefix: "forge3d".to_string(),
                max_queries_per_frame: 32,
            };
            match crate::core::gpu_timing::GpuTimingManager::new(
                self.device.clone(),
                self.queue.clone(),
                config,
            ) {
                Ok(manager) => return Some(manager),
                Err(e) => {
                    log::warn!("failed to create GPU timing manager: {e}");
                    return None;
                }
            }
        }
        guard.take()
    }

    /// Return a timing manager taken by [`take_render_timing`] to the scene.
    pub(super) fn store_render_timing(
        &self,
        manager: Option<crate::core::gpu_timing::GpuTimingManager>,
    ) {
        if let Some(manager) = manager {
            if let Ok(mut guard) = self.gpu_timing.lock() {
                *guard = Some(manager);
            }
        }
    }

    /// Finalize a render's timing: resolve+read the timestamps and record each
    /// timed pass into the global certificate capture. Consumes the manager's
    /// current slot (offline single-shot pattern). Must be called AFTER the
    /// render encoder was submitted (with `resolve_queries` recorded into it).
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
                Err(e) => log::warn!("GPU timing readback failed: {e}"),
            }
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct IblUniforms {
    pub(super) intensity: f32,
    pub(super) sin_theta: f32,
    pub(super) cos_theta: f32,
    pub(super) specular_mip_count: f32,
}

impl Drop for TerrainScene {
    fn drop(&mut self) {
        let tracker = crate::core::memory_tracker::global_tracker();
        if self.probe_grid_uniform_alloc_bytes > 0 {
            tracker.free_buffer_allocation(self.probe_grid_uniform_alloc_bytes, false);
        }
        if self.probe_ssbo_alloc_bytes > 0 {
            tracker.free_buffer_allocation(self.probe_ssbo_alloc_bytes, false);
        }
        if self.reflection_probe_grid_uniform_alloc_bytes > 0 {
            tracker.free_buffer_allocation(self.reflection_probe_grid_uniform_alloc_bytes, false);
        }
    }
}

#[cfg(test)]
mod camera_mode_tests {
    use super::{is_clipmap_camera_mode, is_mesh_camera_mode, is_zup_camera_mode};

    #[test]
    fn zup_suffix_is_detected_and_tag_helpers_still_match() {
        assert!(is_zup_camera_mode("mesh:zup"));
        assert!(is_zup_camera_mode("mesh:ZUP"));
        assert!(is_mesh_camera_mode("mesh:zup"));
        assert!(!is_zup_camera_mode("mesh"));
        assert!(!is_zup_camera_mode("screen"));
        assert!(!is_zup_camera_mode("zup"));
        assert!(is_clipmap_camera_mode("clipmap:4:64"));
        assert!(!is_zup_camera_mode("clipmap:4:64"));
        assert!(is_zup_camera_mode("clipmap:4:64:zup"));
    }
}
