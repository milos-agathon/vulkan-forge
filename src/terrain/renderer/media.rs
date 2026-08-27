//! Persistent GPU resources and deterministic state for NEPHELE's realtime path.
//!
//! Density/extinction physics belong to [`crate::media`]. This module owns only
//! the view-aligned froxel representation and its temporal/render-graph state.
#![allow(dead_code)] // Activated by the additive TerrainRenderParams.media adapter lane.

use crate::core::resource_tracker::{
    calculate_texture_descriptor_size, tracked_create_buffer, tracked_create_texture,
    tracked_host_allocation, ResourceHandle, TrackedBuffer, TrackedTexture,
};
use bytemuck::{Pod, Zeroable};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct MediaUniforms {
    pub(super) view_proj: [[f32; 4]; 4],
    pub(super) inv_view_proj: [[f32; 4]; 4],
    pub(super) previous_view_proj: [[f32; 4]; 4],
    pub(super) view: [[f32; 4]; 4],
    pub(super) camera: [f32; 4],
    pub(super) sun: [f32; 4],
    pub(super) sun_radiance: [f32; 4],
    pub(super) sigma_s: [f32; 4],
    pub(super) sigma_t: [f32; 4],
    pub(super) depth: [f32; 4],
    pub(super) grid: [u32; 4],
    pub(super) viewport: [u32; 4],
    /// x=terrain-occlusion enable, y=multiple-scatter order weight, z=phase g,
    /// w=0 isotropic / 1 Henyey-Greenstein.
    pub(super) scattering: [f32; 4],
    pub(super) terrain_albedo: [f32; 4],
    pub(super) diffuse_ibl: [f32; 4],
}

pub(super) const FROXEL_TILE_SIZE_PX: u32 = 8;
// The existing viewer froxel path establishes 64 as the realtime depth contract.
pub(super) const FROXEL_DEPTH_SLICES: u32 = 64;
pub(super) const FROXEL_OFF_AXIS_BORDER: u32 = 1;
const BLUE_NOISE_WIDTH: u32 = 8;
const BLUE_NOISE_HEIGHT: u32 = 8;
const BLUE_NOISE_ASSET: &str = include_str!("../../../assets/media/nephele_blue_noise_8x8.txt");

pub(super) struct RealtimeMediaCaptureTextures {
    pub(super) transmittance: Option<TrackedTexture>,
    pub(super) in_scatter: Option<TrackedTexture>,
    pub(super) cloud_shadow: Option<TrackedTexture>,
    pub(super) optical_depth: Option<TrackedTexture>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FroxelGrid {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth: u32,
}

impl FroxelGrid {
    pub(super) fn for_viewport(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1).div_ceil(FROXEL_TILE_SIZE_PX) + 2 * FROXEL_OFF_AXIS_BORDER,
            height: height.max(1).div_ceil(FROXEL_TILE_SIZE_PX) + 2 * FROXEL_OFF_AXIS_BORDER,
            depth: FROXEL_DEPTH_SLICES,
        }
    }
    pub(super) fn visible_width(self) -> u32 {
        self.width - 2 * FROXEL_OFF_AXIS_BORDER
    }
    pub(super) fn visible_height(self) -> u32 {
        self.height - 2 * FROXEL_OFF_AXIS_BORDER
    }
    fn extent(self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: self.depth,
        }
    }
}

/// Shared CPU/WGSL logarithmic view-depth transform:
/// `distance(u) = near * exp(log(far / near) * u)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FroxelDepthTransform {
    near: f32,
    far: f32,
    log_far_over_near: f32,
}

impl FroxelDepthTransform {
    pub(super) fn new(near: f32, far: f32) -> Result<Self, &'static str> {
        if !near.is_finite() || !far.is_finite() || near <= 0.0 || far <= near {
            return Err("froxel depth requires finite 0 < near < far");
        }
        Ok(Self {
            near,
            far,
            log_far_over_near: (far / near).ln(),
        })
    }
    pub(super) fn distance_at_unit(self, unit: f32) -> f32 {
        self.near * (self.log_far_over_near * unit.clamp(0.0, 1.0)).exp()
    }
    pub(super) fn unit_at_distance(self, distance: f32) -> f32 {
        ((distance.max(self.near) / self.near).ln() / self.log_far_over_near).clamp(0.0, 1.0)
    }
    pub(super) fn slice_center_distance(self, slice: u32, slices: u32) -> f32 {
        self.distance_at_unit((slice as f32 + 0.5) / slices.max(1) as f32)
    }
    pub(super) fn wgsl_params(self) -> [f32; 4] {
        [self.near, self.far, self.log_far_over_near, 0.0]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MediaHistoryKey {
    pub(super) camera: u64,
    pub(super) scene_depth: u64,
    /// Full canonical identity: density representation, coefficients and phase.
    pub(super) medium: crate::media::MediumIdentity,
    pub(super) lighting: u64,
    pub(super) viewport: (u32, u32),
    pub(super) resource_version: u64,
    pub(super) adapter: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HistoryDecision {
    Accept,
    RejectCamera,
    RejectDepth,
    RejectMedium,
    RejectLighting,
    RejectResize,
    RejectResourceVersion,
    RejectAdapter,
    RejectMissing,
}

impl HistoryDecision {
    fn diagnostic(self) -> (&'static str, &'static str) {
        match self {
            Self::Accept => ("accepted", "all temporal identities matched"),
            Self::RejectCamera => ("rejected", "camera changed"),
            Self::RejectDepth => ("rejected", "scene depth changed"),
            Self::RejectMedium => ("rejected", "canonical medium identity changed"),
            Self::RejectLighting => ("rejected", "lighting or radiance provider changed"),
            Self::RejectResize => ("rejected", "viewport changed"),
            Self::RejectResourceVersion => ("rejected", "media resource version changed"),
            Self::RejectAdapter => ("rejected", "GPU adapter changed"),
            Self::RejectMissing => ("rejected", "no prior media history exists"),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct MediaExecutionDiagnostics {
    pub majorant_proof: Option<crate::media::MajorantProof>,
    pub majorant_valid: bool,
    pub sample_count: u64,
    pub step_count: u64,
    pub temporal_history_decision: String,
    pub temporal_history_reason: String,
    pub host_visible_bytes: u64,
    pub froxel_device_local_bytes: u64,
    pub density_device_local_bytes: u64,
    pub majorant_device_local_bytes: u64,
    pub staging_readback_bytes: u64,
    pub adapter: String,
    pub backend: String,
    pub driver: String,
    pub source_revision: String,
    pub executed_multi_scatter: bool,
    pub single_scatter_dispatches: u64,
    pub multiple_scatter_dispatches: u64,
    pub terrain_trace_queries: u64,
    /// Executed surface-to-sun transport method.
    pub sun_transmittance_method: String,
    /// Named numerical bias, or `none_exact` for analytic transport.
    pub sun_transmittance_bias: String,
    /// Longest bounded surface-to-sun medium segment. `None` denotes an
    /// analytically integrated unbounded homogeneous segment.
    pub sun_transmittance_max_segment_length: Option<f64>,
    pub sun_transmittance_executed_steps: u64,
    /// Maximum absolute RGB difference between the executed fine estimate and
    /// its nested coarse estimate.
    pub sun_transmittance_max_abs_error: f64,
    pub single_scatter_luminance: f64,
    pub multiple_scatter_luminance: f64,
    /// Maximum dimensionless difference between the executed product of
    /// per-slice Beer factors and exp(-executed integrated extinction).
    pub energy_accounting_residual: Option<f64>,
}

pub(super) fn history_decision(
    previous: Option<MediaHistoryKey>,
    current: MediaHistoryKey,
) -> HistoryDecision {
    let Some(previous) = previous else {
        return HistoryDecision::RejectMissing;
    };
    if previous.camera != current.camera {
        HistoryDecision::RejectCamera
    } else if previous.scene_depth != current.scene_depth {
        HistoryDecision::RejectDepth
    } else if previous.medium != current.medium {
        HistoryDecision::RejectMedium
    } else if previous.lighting != current.lighting {
        HistoryDecision::RejectLighting
    } else if previous.viewport != current.viewport {
        HistoryDecision::RejectResize
    } else if previous.resource_version != current.resource_version {
        HistoryDecision::RejectResourceVersion
    } else if previous.adapter != current.adapter {
        HistoryDecision::RejectAdapter
    } else {
        HistoryDecision::Accept
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SunTransmittanceDiagnostic {
    pub(super) method: &'static str,
    pub(super) bias: &'static str,
    pub(super) segment_length: Option<f32>,
    /// Sum of the coarse and nested-fine integration loop iterations that
    /// actually executed. Rays missing the bounded density contribute zero.
    pub(super) executed_steps: u64,
    /// Maximum absolute RGB difference between the reported integration and
    /// the nested half-resolution estimate.
    pub(super) measured_abs_error: f32,
}

impl Default for SunTransmittanceDiagnostic {
    fn default() -> Self {
        Self {
            method: "not_executed",
            bias: "not_executed",
            segment_length: Some(0.0),
            executed_steps: 0,
            measured_abs_error: 0.0,
        }
    }
}

impl SunTransmittanceDiagnostic {
    fn record(&mut self, executed: Self) {
        self.method = executed.method;
        self.bias = executed.bias;
        self.segment_length = match (self.segment_length, executed.segment_length) {
            (Some(current), Some(next)) => Some(current.max(next)),
            _ => None,
        };
        self.executed_steps += executed.executed_steps;
        self.measured_abs_error = self.measured_abs_error.max(executed.measured_abs_error);
    }
}

#[cfg(test)]
fn depth32_history_matches(stored: f32, projected: f32) -> bool {
    stored.is_finite()
        && projected.is_finite()
        && (0.0..=1.0).contains(&stored)
        && (0.0..=1.0).contains(&projected)
        && stored.to_bits().abs_diff(projected.to_bits()) <= 1
}

#[cfg(test)]
fn beer_accounting_residual(extinctions: &[f32], step: f32) -> f32 {
    let mut product = 1.0f32;
    let mut integrated = 0.0f32;
    for &extinction in extinctions {
        product *= (-extinction * step).exp();
        integrated += extinction * step;
    }
    (product - (-integrated).exp()).abs()
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTerrainRay {
    origin_tmin: [f32; 4],
    direction_tmax: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTerrainHit {
    point_t: [f32; 4],
    normal_hit: [f32; 4],
}

struct RealtimeTerrainTrace {
    pipeline: wgpu::ComputePipeline,
    empty0: wgpu::BindGroup,
    empty1: wgpu::BindGroup,
    terrain_group: wgpu::BindGroup,
    query_group: wgpu::BindGroup,
    sun_hits: TrackedBuffer,
    phase_hits: TrackedBuffer,
    _terrain_uniform: TrackedBuffer,
    _curvature_uniform: TrackedBuffer,
    _pyramid: crate::path_tracing::hybrid_compute::terrain_heightfield::TerrainMinMaxPyramid,
    terrain_identity: u64,
    query_count: u64,
    allocation_bytes: u64,
}

struct TerminationReadback {
    values: Vec<f32>,
    allocation: ResourceHandle,
}

pub(super) struct TerrainMediaResources {
    pub(super) grid: FroxelGrid,
    pub(super) viewport: (u32, u32),
    pub(super) resource_version: u64,
    pub(super) device_local_bytes: u64,
    pub(super) density_device_local_bytes: u64,
    pub(super) staging_bytes: u64,
    pub(super) history_key: Option<MediaHistoryKey>,
    pub(super) last_history_decision: HistoryDecision,
    pub(super) medium_identity: Option<crate::media::MediumIdentity>,
    pub(super) majorant_proof: Option<crate::media::MajorantProof>,
    pub(super) sigma_s: [f32; 3],
    pub(super) sigma_t: [f32; 3],
    pub(super) phase: crate::media::Phase,
    pub(super) terrain_albedo: [f32; 3],
    pub(super) diffuse_ibl: [f32; 3],
    pub(super) depth_transform: FroxelDepthTransform,
    pub(super) previous_view_projection: glam::Mat4,
    pub(super) sun_transmittance_diagnostic: SunTransmittanceDiagnostic,
    pub(super) radiance_provider: Arc<TrackedTexture>,
    pub(super) radiance_provider_identity: u64,
    pub(super) blue_noise_identity: u64,
    pub(super) extinction: Arc<TrackedTexture>,
    pub(super) single_scatter: TrackedTexture,
    pub(super) in_scatter: Arc<TrackedTexture>,
    /// RGB transmittance over the complete canonical medium/sun segment.
    pub(super) light_transmittance: Arc<TrackedTexture>,
    pub(super) integrated: Arc<TrackedTexture>,
    pub(super) transmittance: Arc<TrackedTexture>,
    pub(super) cloud_shadow: Arc<TrackedTexture>,
    pub(super) optical_depth: Arc<TrackedTexture>,
    pub(super) composite_linear_hdr: Arc<TrackedTexture>,
    pub(super) history: Arc<TrackedTexture>,
    pub(super) history_depth: Arc<TrackedTexture>,
    pub(super) history_next: Arc<TrackedTexture>,
    pub(super) history_depth_next: Arc<TrackedTexture>,
    pub(super) blue_noise: TrackedTexture,
    pub(super) uniforms: TrackedBuffer,
    pub(super) pipelines: RealtimeMediaPipelines,
    terrain_trace: Option<RealtimeTerrainTrace>,
    single_scatter_dispatches: u64,
    multiple_scatter_dispatches: u64,
    density_froxel_count: u64,
    termination_readback: Option<TerminationReadback>,
    pending_history: Option<(MediaHistoryKey, HistoryDecision, glam::Mat4)>,
}

pub(super) struct RealtimeMediaPipelines {
    pub(super) inject_single: wgpu::ComputePipeline,
    pub(super) inject_multiple: wgpu::ComputePipeline,
    pub(super) integrate: wgpu::ComputePipeline,
    pub(super) composite_linear_hdr: wgpu::ComputePipeline,
}

/// Canonical NEPHELE execution state reused by the interactive terrain viewer.
pub(crate) struct ViewerMediaPass {
    resources: TerrainMediaResources,
    medium: crate::media::Medium,
    version: u64,
    viewer_terrain_key: Option<u64>,
    viewer_terrain_albedo: [f32; 3],
    prepared_medium_identity: Option<crate::media::MediumIdentity>,
    prepared_frame: Option<ViewerFramePreparation>,
}

#[derive(Clone, Copy)]
struct ViewerFramePreparation {
    camera_position: glam::Vec3,
    inverse_view_projection: glam::Mat4,
    depth: FroxelDepthTransform,
    sun_direction: glam::Vec3,
}

#[derive(Clone, Copy, Debug)]
struct ViewerTerrainTracePlacement {
    origin_xz: [f32; 2],
    spacing_xz: [f32; 2],
    height_offset: f32,
    height_scale: f32,
}

fn viewer_terrain_trace_placement(
    dimensions: (u32, u32),
    render_origin_xz: [f32; 2],
    render_span_xz: [f32; 2],
    height_min: f32,
    _height_range: f32,
    z_scale: f32,
) -> ViewerTerrainTracePlacement {
    ViewerTerrainTracePlacement {
        origin_xz: render_origin_xz,
        spacing_xz: [
            render_span_xz[0] / dimensions.0.saturating_sub(1).max(1) as f32,
            render_span_xz[1] / dimensions.1.saturating_sub(1).max(1) as f32,
        ],
        height_offset: height_min,
        // Matches forced PBR: world_y = (raw_height - height_min) * z_scale.
        height_scale: z_scale,
    }
}

impl ViewerMediaPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (u32, u32),
        medium: crate::media::Medium,
        version: u64,
    ) -> crate::core::error::RenderResult<Self> {
        medium
            .validate()
            .map_err(crate::core::error::RenderError::render)?;
        Ok(Self {
            resources: TerrainMediaResources::new(device, queue, viewport, version)?,
            medium,
            version,
            viewer_terrain_key: None,
            viewer_terrain_albedo: [0.0; 3],
            prepared_medium_identity: None,
            prepared_frame: None,
        })
    }

    pub(crate) fn light_transmittance_view(&self) -> wgpu::TextureView {
        self.resources
            .light_transmittance
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub(crate) fn prepare_viewer_frame(
        &mut self,
        queue: &wgpu::Queue,
        camera: glam::Vec3,
        view_projection: glam::Mat4,
        near: f32,
        far: f32,
        sun_direction: glam::Vec3,
    ) -> crate::core::error::RenderResult<()> {
        self.prepared_medium_identity = None;
        self.prepared_frame = None;
        let depth = FroxelDepthTransform::new(near, far)
            .map_err(crate::core::error::RenderError::render)?;
        let preparation = ViewerFramePreparation {
            camera_position: camera,
            inverse_view_projection: view_projection.inverse(),
            depth,
            sun_direction,
        };
        self.prepared_medium_identity = Some(self.resources.upload_canonical_extinction(
            queue,
            &self.medium,
            self.version,
            preparation.camera_position,
            preparation.inverse_view_projection,
            preparation.depth,
            preparation.sun_direction,
        )?);
        self.prepared_frame = Some(preparation);
        Ok(())
    }

    fn resize_resources_for_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (u32, u32),
    ) -> crate::core::error::RenderResult<()> {
        if self.resources.viewport == viewport {
            return Ok(());
        }
        let frame_was_prepared = self.prepared_medium_identity.is_some();
        let preparation = self.prepared_frame.take();
        self.prepared_medium_identity = None;
        self.resources = TerrainMediaResources::new(device, queue, viewport, self.version)?;
        let Some(preparation) = preparation.filter(|_| frame_was_prepared) else {
            return Ok(());
        };
        let identity = self.resources.upload_canonical_extinction(
            queue,
            &self.medium,
            self.version,
            preparation.camera_position,
            preparation.inverse_view_projection,
            preparation.depth,
            preparation.sun_direction,
        )?;
        self.prepared_frame = Some(preparation);
        self.prepared_medium_identity = Some(identity);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_viewer_terrain_trace(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
        viewport: (u32, u32),
        heights: &[f32],
        dimensions: (u32, u32),
        render_origin_xz: [f32; 2],
        render_span_xz: [f32; 2],
        height_min: f32,
        height_range: f32,
        z_scale: f32,
        terrain_revision: u64,
    ) -> crate::core::error::RenderResult<()> {
        self.resize_resources_for_viewport(device, queue, viewport)?;
        let viewer_key = stable_words_hash(
            [
                terrain_revision,
                u64::from(dimensions.0),
                u64::from(dimensions.1),
                u64::from(render_origin_xz[0].to_bits()),
                u64::from(render_origin_xz[1].to_bits()),
                u64::from(render_span_xz[0].to_bits()),
                u64::from(render_span_xz[1].to_bits()),
                u64::from(height_min.to_bits()),
                u64::from(height_range.to_bits()),
                u64::from(z_scale.to_bits()),
            ]
            .into_iter(),
        );
        if self.viewer_terrain_key != Some(viewer_key) {
            self.viewer_terrain_albedo =
                viewer_height_palette_mean(heights, height_min, height_range);
            self.viewer_terrain_key = Some(viewer_key);
        }
        let albedo = self.viewer_terrain_albedo;
        let terrain_identity = viewer_key
            ^ stable_words_hash(albedo.into_iter().map(|value| u64::from(value.to_bits())));
        let placement = viewer_terrain_trace_placement(
            dimensions,
            render_origin_xz,
            render_span_xz,
            height_min,
            height_range,
            z_scale,
        );
        self.resources.prepare_terrain_trace_at(
            device,
            queue,
            adapter.get_info().backend,
            heights,
            dimensions,
            placement.origin_xz,
            placement.spacing_xz,
            placement.height_scale,
            albedo,
            terrain_identity,
            placement.height_offset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter: &wgpu::Adapter,
        encoder: &mut wgpu::CommandEncoder,
        viewport: (u32, u32),
        camera: glam::Vec3,
        view: glam::Mat4,
        projection: glam::Mat4,
        near: f32,
        far: f32,
        sun_direction: glam::Vec3,
        sun_radiance: [f32; 3],
        terrain_revision: u64,
        scene_depth: &wgpu::Texture,
        scene_depth_view: &wgpu::TextureView,
        terrain_linear_hdr_view: &wgpu::TextureView,
        csm: &crate::shadows::CsmRenderer,
    ) -> crate::core::error::RenderResult<wgpu::TextureView> {
        self.resize_resources_for_viewport(device, queue, viewport)?;
        let view_projection = projection * view;
        let depth = FroxelDepthTransform::new(near, far)
            .map_err(crate::core::error::RenderError::render)?;
        let medium_identity = self.prepared_medium_identity.take().ok_or_else(|| {
            crate::core::error::RenderError::render(
                "viewer media frame was not prepared before terrain direct lighting",
            )
        })?;
        encoder.clear_texture(
            &self.resources.radiance_provider,
            &wgpu::ImageSubresourceRange::default(),
        );
        let phase = match self.resources.phase {
            crate::media::Phase::Isotropic => (0.0, 0.0),
            crate::media::Phase::HenyeyGreenstein { g } => (g, 1.0),
        };
        let history_key = MediaHistoryKey {
            camera: stable_words_hash(
                view_projection
                    .to_cols_array()
                    .into_iter()
                    .map(|value| u64::from(value.to_bits())),
            ),
            scene_depth: terrain_revision,
            medium: medium_identity,
            lighting: media_lighting_identity(sun_direction, sun_radiance, 1.0, true)
                ^ self.resources.blue_noise_identity
                ^ self
                    .resources
                    .terrain_trace
                    .as_ref()
                    .map_or(0, |trace| trace.terrain_identity),
            viewport,
            resource_version: self.version,
            adapter: stable_adapter_hash(&adapter.get_info()),
        };
        let mut depth_params = depth.wgsl_params();
        depth_params[3] = 0.2;
        let uniforms = MediaUniforms {
            view_proj: view_projection.to_cols_array_2d(),
            inv_view_proj: view_projection.inverse().to_cols_array_2d(),
            previous_view_proj: self.resources.previous_view_projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            camera: [camera.x, camera.y, camera.z, 0.0],
            sun: [sun_direction.x, sun_direction.y, sun_direction.z, 0.0],
            sun_radiance: [sun_radiance[0], sun_radiance[1], sun_radiance[2], 1.0],
            sigma_s: [
                self.resources.sigma_s[0],
                self.resources.sigma_s[1],
                self.resources.sigma_s[2],
                0.0,
            ],
            sigma_t: [
                self.resources.sigma_t[0],
                self.resources.sigma_t[1],
                self.resources.sigma_t[2],
                0.0,
            ],
            depth: depth_params,
            grid: [
                self.resources.grid.width,
                self.resources.grid.height,
                self.resources.grid.depth,
                0,
            ],
            viewport: [viewport.0, viewport.1, 0, 0],
            scattering: [1.0, 1.0, phase.0, phase.1],
            terrain_albedo: [
                self.resources.terrain_albedo[0],
                self.resources.terrain_albedo[1],
                self.resources.terrain_albedo[2],
                0.0,
            ],
            diffuse_ibl: [
                self.resources.diffuse_ibl[0],
                self.resources.diffuse_ibl[1],
                self.resources.diffuse_ibl[2],
                0.0,
            ],
        };
        self.resources.prepare_and_encode_inject(
            device,
            queue,
            encoder,
            csm,
            uniforms,
            history_key,
        )?;
        self.resources
            .encode_integrate(device, encoder, scene_depth_view)?;
        self.resources
            .encode_composite(device, encoder, terrain_linear_hdr_view)?;
        self.resources.encode_history_commit(encoder, scene_depth)?;
        Ok(self
            .resources
            .composite_linear_hdr
            .create_view(&wgpu::TextureViewDescriptor::default()))
    }

    pub(crate) fn diagnostics(&self, adapter: &wgpu::Adapter) -> MediaExecutionDiagnostics {
        self.resources.diagnostics(adapter)
    }
}

fn viewer_height_palette_mean(heights: &[f32], minimum: f32, range: f32) -> [f32; 3] {
    let mut sum = [0.0f64; 3];
    for &height in heights {
        let h = ((height - minimum) / range.max(1.0)).clamp(0.0, 1.0);
        let color = if h < 0.3 {
            mix_rgb([0.16, 0.42, 0.14], [0.32, 0.56, 0.21], h / 0.3)
        } else if h < 0.7 {
            mix_rgb([0.32, 0.56, 0.21], [0.50, 0.39, 0.24], (h - 0.3) / 0.4)
        } else {
            mix_rgb([0.50, 0.39, 0.24], [0.88, 0.88, 0.84], (h - 0.7) / 0.3)
        };
        for channel in 0..3 {
            sum[channel] += f64::from(color[channel]);
        }
    }
    let count = heights.len().max(1) as f64;
    sum.map(|value| (value / count) as f32)
}

fn mix_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|channel| a[channel] + (b[channel] - a[channel]) * t)
}

impl RealtimeMediaPipelines {
    fn new(device: &wgpu::Device) -> crate::core::error::RenderResult<Self> {
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "nephele.media.froxel.shader",
            include_str!("../../shaders/nephele_froxel.wgsl"),
        );
        let pipeline = |label: &'static str, entry_point: &'static str| {
            crate::core::shader_registry::try_create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module: &shader,
                    entry_point,
                },
            )
            .map_err(crate::core::error::RenderError::render)
        };
        Ok(Self {
            inject_single: pipeline(
                "nephele.media.inject.single.pipeline",
                "cs_nephele_inject_single",
            )?,
            inject_multiple: pipeline(
                "nephele.media.inject.multiple.pipeline",
                "cs_nephele_inject_multiple",
            )?,
            integrate: pipeline("nephele.media.integrate.pipeline", "cs_nephele_integrate")?,
            composite_linear_hdr: pipeline(
                "nephele.media.composite.linear_hdr.pipeline",
                "cs_nephele_composite_linear_hdr",
            )?,
        })
    }
}

impl TerrainMediaResources {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (u32, u32),
        resource_version: u64,
    ) -> crate::core::error::RenderResult<Self> {
        let viewport = (viewport.0.max(1), viewport.1.max(1));
        let grid = FroxelGrid::for_viewport(viewport.0, viewport.1);
        let froxel_usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let output_usage = froxel_usage;
        let (extinction, a) = tracked_texture(
            device,
            "nephele.media.froxel.extinction",
            grid.extent(),
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::Rgba16Float,
            froxel_usage,
        )?;
        let (in_scatter, b) = tracked_texture(
            device,
            "nephele.media.froxel.in_scatter",
            grid.extent(),
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::Rgba16Float,
            froxel_usage,
        )?;
        let (single_scatter, single_bytes) = tracked_texture(
            device,
            "nephele.media.froxel.single_scatter",
            grid.extent(),
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::Rgba16Float,
            froxel_usage,
        )?;
        let (light_transmittance, light_bytes) = tracked_texture(
            device,
            "nephele.media.froxel.light_transmittance",
            grid.extent(),
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::Rgba16Float,
            froxel_usage,
        )?;
        let out = wgpu::Extent3d {
            width: viewport.0,
            height: viewport.1,
            depth_or_array_layers: 1,
        };
        let (integrated, c) = tracked_texture(
            device,
            "nephele.media.aov.in_scatter",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            output_usage,
        )?;
        let (transmittance, d) = tracked_texture(
            device,
            "nephele.media.aov.transmittance",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            output_usage,
        )?;
        let (cloud_shadow, e) = tracked_texture(
            device,
            "nephele.media.aov.cloud_shadow",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            output_usage,
        )?;
        let (optical_depth, f) = tracked_texture(
            device,
            "nephele.media.aov.optical_depth",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            output_usage,
        )?;
        let (composite_linear_hdr, composite_bytes) = tracked_texture(
            device,
            "nephele.media.composite.linear_hdr",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            output_usage,
        )?;
        let (history, g) = tracked_texture(
            device,
            "nephele.media.history.previous.in_scatter",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            output_usage,
        )?;
        let (history_depth, h) = tracked_texture(
            device,
            "nephele.media.history.previous.depth",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
        )?;
        let (history_next, i) = tracked_texture(
            device,
            "nephele.media.history.current.in_scatter",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            output_usage,
        )?;
        let (history_depth_next, j) = tracked_texture(
            device,
            "nephele.media.history.current.depth",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
        )?;
        let (blue_noise, k) = tracked_texture(
            device,
            "nephele.media.blue_noise",
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::R8Uint,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )?;
        let (radiance_provider, provider_bytes) = tracked_texture(
            device,
            "nephele.media.radiance_provider",
            out,
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
        )?;
        let ranks = parse_blue_noise_asset().map_err(crate::core::error::RenderError::render)?;
        let blue_noise_identity = stable_words_hash(ranks.iter().copied().map(u64::from));
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &blue_noise,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &ranks,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(8),
            },
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        let pipelines = RealtimeMediaPipelines::new(device)?;
        let uniforms = tracked_create_buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("nephele.media.uniforms"),
                size: std::mem::size_of::<MediaUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        Ok(Self {
            grid,
            viewport,
            resource_version,
            device_local_bytes: a
                + b
                + single_bytes
                + light_bytes
                + c
                + d
                + e
                + f
                + composite_bytes
                + g
                + h
                + i
                + j
                + k
                + provider_bytes,
            density_device_local_bytes: a,
            staging_bytes: u64::from(grid.width)
                * u64::from(grid.height)
                * u64::from(grid.depth)
                * 16,
            history_key: None,
            last_history_decision: HistoryDecision::RejectMissing,
            medium_identity: None,
            majorant_proof: None,
            sigma_s: [0.0; 3],
            sigma_t: [0.0; 3],
            phase: crate::media::Phase::Isotropic,
            terrain_albedo: [0.0; 3],
            diffuse_ibl: [0.0; 3],
            depth_transform: FroxelDepthTransform::new(0.1, 1.0)
                .expect("constant depth transform is valid"),
            previous_view_projection: glam::Mat4::IDENTITY,
            sun_transmittance_diagnostic: SunTransmittanceDiagnostic::default(),
            radiance_provider: Arc::new(radiance_provider),
            radiance_provider_identity: 0,
            blue_noise_identity,
            extinction: Arc::new(extinction),
            single_scatter,
            in_scatter: Arc::new(in_scatter),
            light_transmittance: Arc::new(light_transmittance),
            integrated: Arc::new(integrated),
            transmittance: Arc::new(transmittance),
            cloud_shadow: Arc::new(cloud_shadow),
            optical_depth: Arc::new(optical_depth),
            composite_linear_hdr: Arc::new(composite_linear_hdr),
            history: Arc::new(history),
            history_depth: Arc::new(history_depth),
            history_next: Arc::new(history_next),
            history_depth_next: Arc::new(history_depth_next),
            blue_noise,
            uniforms,
            pipelines,
            terrain_trace: None,
            single_scatter_dispatches: 0,
            multiple_scatter_dispatches: 0,
            density_froxel_count: 0,
            termination_readback: None,
            pending_history: None,
        })
    }

    pub(super) fn upload_canonical_extinction(
        &mut self,
        queue: &wgpu::Queue,
        medium: &crate::media::Medium,
        density_version: u64,
        camera_position: glam::Vec3,
        inverse_view_projection: glam::Mat4,
        depth: FroxelDepthTransform,
        sun_direction: glam::Vec3,
    ) -> crate::core::error::RenderResult<crate::media::MediumIdentity> {
        let majorant = crate::media::MajorantGrid::construct(medium, density_version)
            .map_err(crate::core::error::RenderError::render)?;
        majorant
            .validate_for(medium)
            .map_err(crate::core::error::RenderError::render)?;
        let count =
            u64::from(self.grid.width) * u64::from(self.grid.height) * u64::from(self.grid.depth);
        let _staging =
            tracked_host_allocation(count * 16, "nephele.media.staging.extinction_and_light")?;
        let mut texels: Vec<u16> = Vec::with_capacity(count as usize * 4);
        let mut light_texels: Vec<u16> = Vec::with_capacity(count as usize * 4);
        let mut density_froxel_count = 0;
        let mut sun_transmittance_diagnostic = SunTransmittanceDiagnostic::default();
        for z in 0..self.grid.depth {
            for y in 0..self.grid.height {
                for x in 0..self.grid.width {
                    let world = froxel_world_position(
                        self.grid,
                        [x, y, z],
                        camera_position,
                        inverse_view_projection,
                        depth,
                    );
                    let physical_density = medium.density().physical_density(world.to_array());
                    density_froxel_count += u64::from(physical_density > 0.0);
                    let extinction = medium.extinction_at(world.to_array()).components();
                    for value in extinction {
                        texels.push(required_positive_f16(value, "extinction")?);
                    }
                    texels.push(required_positive_f16(physical_density, "density")?);
                    let (light_t, diagnostic) =
                        canonical_sun_transmittance(medium, world, sun_direction)?;
                    sun_transmittance_diagnostic.record(diagnostic);
                    light_texels.extend(light_t.map(|v| half::f16::from_f32(v).to_bits()));
                    light_texels.push(half::f16::ONE.to_bits());
                }
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.extinction,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.grid.width * 8),
                rows_per_image: Some(self.grid.height),
            },
            self.grid.extent(),
        );
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.light_transmittance,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&light_texels),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.grid.width * 8),
                rows_per_image: Some(self.grid.height),
            },
            self.grid.extent(),
        );
        self.density_froxel_count = density_froxel_count;
        self.sun_transmittance_diagnostic = sun_transmittance_diagnostic;
        let identity = medium.identity(density_version);
        self.medium_identity = Some(identity);
        self.majorant_proof = Some(majorant.proof().clone());
        self.sigma_s = medium.sigma_s().components();
        self.sigma_t = medium.sigma_t().components();
        self.phase = medium.phase();
        self.depth_transform = depth;
        Ok(identity)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_terrain_trace(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        backend: wgpu::Backend,
        heights: &[f32],
        dimensions: (u32, u32),
        terrain_span: f32,
        exaggeration: f32,
        albedo: [f32; 3],
        terrain_identity: u64,
        height_offset: f32,
    ) -> crate::core::error::RenderResult<()> {
        let spacing_xz = [
            terrain_span / dimensions.0.saturating_sub(1).max(1) as f32,
            terrain_span / dimensions.1.saturating_sub(1).max(1) as f32,
        ];
        let origin_xz = [
            -0.5 * dimensions.0.saturating_sub(1) as f32 * spacing_xz[0],
            -0.5 * dimensions.1.saturating_sub(1) as f32 * spacing_xz[1],
        ];
        self.prepare_terrain_trace_at(
            device,
            queue,
            backend,
            heights,
            dimensions,
            origin_xz,
            spacing_xz,
            exaggeration,
            albedo,
            terrain_identity,
            height_offset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_terrain_trace_at(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        backend: wgpu::Backend,
        heights: &[f32],
        dimensions: (u32, u32),
        origin_xz: [f32; 2],
        spacing_xz: [f32; 2],
        height_scale: f32,
        albedo: [f32; 3],
        terrain_identity: u64,
        height_offset: f32,
    ) -> crate::core::error::RenderResult<()> {
        if self
            .terrain_trace
            .as_ref()
            .is_some_and(|trace| trace.terrain_identity == terrain_identity)
        {
            return Ok(());
        }
        let normalized_heights;
        let heights = if height_offset == 0.0 {
            heights
        } else {
            normalized_heights = heights
                .iter()
                .map(|height| height - height_offset)
                .collect::<Vec<_>>();
            normalized_heights.as_slice()
        };
        let pyramid = crate::path_tracing::hybrid_compute::terrain_heightfield::TerrainMinMaxPyramid::from_heightfield(
            device,
            queue,
            heights,
            dimensions.0,
            dimensions.1,
        )?;
        let terrain_uniform = pyramid.uniforms_at_origin(
            origin_xz,
            spacing_xz,
            height_scale,
            albedo,
            1.0,
            self.viewport,
            1,
            2,
        );
        let curvature_uniform =
            crate::path_tracing::hybrid_compute::terrain_heightfield::EarthCurvatureUniforms {
                inv_two_r_prime: 0.0,
                _pad0: 0.0,
                ray_origin_geodetic: [0.0; 2],
                enabled: 0,
                _pad1: 0,
            };
        let terrain_buffer = crate::core::resource_tracker::tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("nephele.media.terrain_trace.uniforms"),
                contents: bytemuck::bytes_of(&terrain_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let curvature_buffer = crate::core::resource_tracker::tracked_create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("nephele.media.terrain_trace.curvature"),
                contents: bytemuck::bytes_of(&curvature_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let froxel_count =
            u64::from(self.grid.width) * u64::from(self.grid.height) * u64::from(self.grid.depth);
        let hit_bytes = froxel_count * std::mem::size_of::<GpuTerrainHit>() as u64;
        let make_hits = |label| {
            tracked_create_buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: hit_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                },
            )
        };
        let sun_hits = make_hits("nephele.media.terrain_trace.sun_hits")?;
        let phase_hits = make_hits("nephele.media.terrain_trace.phase_hits")?;
        let source = format!(
            "{}\n{}",
            crate::shader_sources::hybrid_kernel(),
            include_str!("../../shaders/nephele_realtime_terrain_trace.wgsl")
        );
        let shader = crate::core::shader_registry::create_labeled_shader_module(
            device,
            "nephele.media.terrain_trace.shader",
            &source,
        );
        let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("nephele.media.terrain_trace.pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main_nephele_realtime_terrain_trace",
            },
        )
        .map_err(crate::core::error::RenderError::render)?;
        let empty = |group, label| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(group),
                entries: &[],
            })
        };
        let empty0 = empty(0, "nephele.media.terrain_trace.group0.empty");
        let empty1 = empty(1, "nephele.media.terrain_trace.group1.empty");
        let height_view = pyramid
            .height_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let minmax_view = pyramid
            .minmax_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (height_binding, minmax_binding) = if backend == wgpu::Backend::Metal {
            (&minmax_view, &height_view)
        } else {
            (&height_view, &minmax_view)
        };
        let terrain_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.terrain_trace.group2"),
            layout: &pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(height_binding),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(minmax_binding),
                },
                buffer_entry(3, &terrain_buffer),
                buffer_entry(10, &curvature_buffer),
            ],
        });
        let blue_noise = self
            .blue_noise
            .create_view(&wgpu::TextureViewDescriptor::default());
        let extinction = self
            .extinction
            .create_view(&wgpu::TextureViewDescriptor::default());
        let query_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.terrain_trace.group3"),
            layout: &pipeline.get_bind_group_layout(3),
            entries: &[
                buffer_entry(8, &self.uniforms),
                texture_entry(9, &blue_noise),
                texture_entry(10, &extinction),
                buffer_entry(11, &sun_hits),
                buffer_entry(12, &phase_hits),
            ],
        });
        if let Some(previous) = self.terrain_trace.take() {
            self.device_local_bytes = self
                .device_local_bytes
                .saturating_sub(previous.allocation_bytes);
        }
        let allocation_bytes = pyramid.byte_size
            + hit_bytes * 2
            + std::mem::size_of_val(&terrain_uniform) as u64
            + std::mem::size_of_val(&curvature_uniform) as u64;
        self.device_local_bytes += allocation_bytes;
        self.terrain_albedo = albedo;
        self.terrain_trace = Some(RealtimeTerrainTrace {
            pipeline,
            empty0,
            empty1,
            terrain_group,
            query_group,
            sun_hits,
            phase_hits,
            _terrain_uniform: terrain_buffer,
            _curvature_uniform: curvature_buffer,
            _pyramid: pyramid,
            terrain_identity,
            query_count: self.density_froxel_count * 2,
            allocation_bytes,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_and_encode_inject(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        csm: &crate::shadows::CsmRenderer,
        mut uniforms: MediaUniforms,
        history_key: MediaHistoryKey,
    ) -> crate::core::error::RenderResult<HistoryDecision> {
        self.termination_readback = None;
        let decision = history_decision(self.history_key, history_key);
        uniforms.viewport[2] = u32::from(decision == HistoryDecision::Accept);
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));
        let extinction = self
            .extinction
            .create_view(&wgpu::TextureViewDescriptor::default());
        let light_transmittance = self
            .light_transmittance
            .create_view(&wgpu::TextureViewDescriptor::default());
        let blue_noise = self
            .blue_noise
            .create_view(&wgpu::TextureViewDescriptor::default());
        let single_scatter = self
            .single_scatter
            .create_view(&wgpu::TextureViewDescriptor::default());
        let in_scatter = self
            .in_scatter
            .create_view(&wgpu::TextureViewDescriptor::default());
        let radiance_provider = self
            .radiance_provider
            .create_view(&wgpu::TextureViewDescriptor::default());
        let terrain_shadow_maps = csm.shadow_maps.create_view(&wgpu::TextureViewDescriptor {
            label: Some("nephele.media.terrain-shadow-maps"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let trace = self.terrain_trace.as_ref().ok_or_else(|| {
            crate::core::error::RenderError::render(
                "realtime media execution requires a prepared production terrain_trace scene",
            )
        })?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nephele.media.terrain_trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&trace.pipeline);
            pass.set_bind_group(0, &trace.empty0, &[]);
            pass.set_bind_group(1, &trace.empty1, &[]);
            pass.set_bind_group(2, &trace.terrain_group, &[]);
            pass.set_bind_group(3, &trace.query_group, &[]);
            pass.dispatch_workgroups(
                self.grid.width.div_ceil(4),
                self.grid.height.div_ceil(4),
                self.grid.depth.div_ceil(4),
            );
        }

        let single_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.inject.single.group0"),
            layout: &self.pipelines.inject_single.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &self.uniforms),
                texture_entry(1, &extinction),
                texture_entry(2, &blue_noise),
                texture_entry(3, &terrain_shadow_maps),
                buffer_entry(4, &csm.uniform_buffer),
                texture_entry(5, &light_transmittance),
                texture_entry(6, &radiance_provider),
                buffer_entry(7, &trace.sun_hits),
                buffer_entry(8, &trace.phase_hits),
            ],
        });
        let single_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.inject.single.group1"),
            layout: &self.pipelines.inject_single.get_bind_group_layout(1),
            entries: &[texture_entry(0, &single_scatter)],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nephele.media.inject.single"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("nephele.media.inject.single.pipeline");
            pass.set_pipeline(&self.pipelines.inject_single);
            pass.set_bind_group(0, &single_group0, &[]);
            pass.set_bind_group(1, &single_group1, &[]);
            pass.dispatch_workgroups(
                self.grid.width.div_ceil(4),
                self.grid.height.div_ceil(4),
                self.grid.depth.div_ceil(4),
            );
        }
        self.single_scatter_dispatches = self.single_scatter_dispatches.wrapping_add(1);

        let multiple_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.inject.multiple.group0"),
            layout: &self.pipelines.inject_multiple.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &self.uniforms),
                texture_entry(1, &extinction),
                texture_entry(2, &blue_noise),
                buffer_entry(8, &trace.phase_hits),
            ],
        });
        let multiple_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.inject.multiple.group1"),
            layout: &self.pipelines.inject_multiple.get_bind_group_layout(1),
            entries: &[texture_entry(1, &in_scatter)],
        });
        let multiple_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.inject.multiple.group2"),
            layout: &self.pipelines.inject_multiple.get_bind_group_layout(2),
            entries: &[
                texture_entry(0, &extinction),
                texture_entry(9, &single_scatter),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nephele.media.inject.multiple"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use(
                "nephele.media.inject.multiple.pipeline",
            );
            pass.set_pipeline(&self.pipelines.inject_multiple);
            pass.set_bind_group(0, &multiple_group0, &[]);
            pass.set_bind_group(1, &multiple_group1, &[]);
            pass.set_bind_group(2, &multiple_group2, &[]);
            pass.dispatch_workgroups(
                self.grid.width.div_ceil(4),
                self.grid.height.div_ceil(4),
                self.grid.depth.div_ceil(4),
            );
        }
        self.multiple_scatter_dispatches = self.multiple_scatter_dispatches.wrapping_add(1);

        self.pending_history = Some((
            history_key,
            decision,
            glam::Mat4::from_cols_array_2d(&uniforms.view_proj),
        ));
        Ok(decision)
    }

    pub(super) fn encode_integrate(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene_depth_view: &wgpu::TextureView,
    ) -> crate::core::error::RenderResult<()> {
        if self.pending_history.is_none() {
            return Err(crate::core::error::RenderError::render(
                "nephele.media.integrate executed before nephele.media.inject",
            ));
        }
        let extinction = self
            .extinction
            .create_view(&wgpu::TextureViewDescriptor::default());
        let light_transmittance = self
            .light_transmittance
            .create_view(&wgpu::TextureViewDescriptor::default());
        let blue_noise = self
            .blue_noise
            .create_view(&wgpu::TextureViewDescriptor::default());
        let single_scatter = self
            .single_scatter
            .create_view(&wgpu::TextureViewDescriptor::default());
        let in_scatter = self
            .in_scatter
            .create_view(&wgpu::TextureViewDescriptor::default());

        let integrated = self
            .integrated
            .create_view(&wgpu::TextureViewDescriptor::default());
        let transmittance = self
            .transmittance
            .create_view(&wgpu::TextureViewDescriptor::default());
        let cloud_shadow = self
            .cloud_shadow
            .create_view(&wgpu::TextureViewDescriptor::default());
        let optical_depth = self
            .optical_depth
            .create_view(&wgpu::TextureViewDescriptor::default());
        let history = self
            .history
            .create_view(&wgpu::TextureViewDescriptor::default());
        let history_depth = self
            .history_depth
            .create_view(&wgpu::TextureViewDescriptor::default());
        let integrate_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.integrate.group0"),
            layout: &self.pipelines.integrate.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &self.uniforms),
                texture_entry(2, &blue_noise),
                texture_entry(5, &light_transmittance),
            ],
        });
        let integrate_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.integrate.group1.empty"),
            layout: &self.pipelines.integrate.get_bind_group_layout(1),
            entries: &[],
        });
        let integrate_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.integrate.group2"),
            layout: &self.pipelines.integrate.get_bind_group_layout(2),
            entries: &[
                texture_entry(0, &extinction),
                texture_entry(1, &in_scatter),
                texture_entry(2, scene_depth_view),
                texture_entry(3, &history),
                texture_entry(4, &history_depth),
                texture_entry(5, &transmittance),
                texture_entry(6, &integrated),
                texture_entry(7, &cloud_shadow),
                texture_entry(8, &optical_depth),
                texture_entry(9, &single_scatter),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nephele.media.integrate"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use("nephele.media.integrate.pipeline");
            pass.set_pipeline(&self.pipelines.integrate);
            pass.set_bind_group(0, &integrate_group0, &[]);
            pass.set_bind_group(1, &integrate_group1, &[]);
            pass.set_bind_group(2, &integrate_group2, &[]);
            pass.dispatch_workgroups(self.viewport.0.div_ceil(8), self.viewport.1.div_ceil(8), 1);
        }

        Ok(())
    }

    pub(super) fn encode_composite(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        terrain_linear_hdr_view: &wgpu::TextureView,
    ) -> crate::core::error::RenderResult<()> {
        if self.pending_history.is_none() {
            return Err(crate::core::error::RenderError::render(
                "nephele.media.composite executed before nephele.media.inject",
            ));
        }
        let integrated = self
            .integrated
            .create_view(&wgpu::TextureViewDescriptor::default());
        let transmittance = self
            .transmittance
            .create_view(&wgpu::TextureViewDescriptor::default());

        let composite = self
            .composite_linear_hdr
            .create_view(&wgpu::TextureViewDescriptor::default());
        let composite_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.composite.group0"),
            layout: &self.pipelines.composite_linear_hdr.get_bind_group_layout(0),
            entries: &[buffer_entry(0, &self.uniforms)],
        });
        let composite_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.composite.group1.empty"),
            layout: &self.pipelines.composite_linear_hdr.get_bind_group_layout(1),
            entries: &[],
        });
        let composite_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.composite.group2.empty"),
            layout: &self.pipelines.composite_linear_hdr.get_bind_group_layout(2),
            entries: &[],
        });
        let composite_group3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.composite.group3"),
            layout: &self.pipelines.composite_linear_hdr.get_bind_group_layout(3),
            entries: &[
                texture_entry(0, terrain_linear_hdr_view),
                texture_entry(1, &integrated),
                texture_entry(2, &transmittance),
                texture_entry(3, &composite),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nephele.media.composite.linear_hdr"),
                timestamp_writes: None,
            });
            crate::core::shader_registry::record_shader_use(
                "nephele.media.composite.linear_hdr.pipeline",
            );
            pass.set_pipeline(&self.pipelines.composite_linear_hdr);
            pass.set_bind_group(0, &composite_group0, &[]);
            pass.set_bind_group(1, &composite_group1, &[]);
            pass.set_bind_group(2, &composite_group2, &[]);
            pass.set_bind_group(3, &composite_group3, &[]);
            pass.dispatch_workgroups(self.viewport.0.div_ceil(8), self.viewport.1.div_ceil(8), 1);
        }

        Ok(())
    }

    pub(super) fn encode_history_commit(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene_depth: &wgpu::Texture,
    ) -> crate::core::error::RenderResult<HistoryDecision> {
        let (history_key, decision, view_projection) =
            self.pending_history.take().ok_or_else(|| {
                crate::core::error::RenderError::render(
                    "nephele.media.history.commit executed without a prepared media frame",
                )
            })?;

        encoder.copy_texture_to_texture(
            image_copy(&self.integrated, wgpu::TextureAspect::All),
            image_copy(&self.history_next, wgpu::TextureAspect::All),
            output_extent(self.viewport),
        );
        encoder.copy_texture_to_texture(
            image_copy(scene_depth, wgpu::TextureAspect::DepthOnly),
            image_copy(&self.history_depth_next, wgpu::TextureAspect::DepthOnly),
            output_extent(self.viewport),
        );
        std::mem::swap(&mut self.history, &mut self.history_next);
        std::mem::swap(&mut self.history_depth, &mut self.history_depth_next);
        self.history_key = Some(history_key);
        self.last_history_decision = decision;
        self.previous_view_projection = view_projection;
        Ok(decision)
    }

    fn termination_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> crate::core::error::RenderResult<&TerminationReadback> {
        if self.termination_readback.is_none() {
            let rgba = crate::core::hdr::read_hdr_texture(
                device,
                queue,
                &self.optical_depth,
                self.viewport.0,
                self.viewport.1,
                wgpu::TextureFormat::Rgba16Float,
            )
            .map_err(crate::core::error::RenderError::render)?;
            let values = rgba
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            let expected = u64::from(self.viewport.0) * u64::from(self.viewport.1);
            if u64::try_from(values.len()).ok() != Some(expected) {
                return Err(crate::core::error::RenderError::render(
                    "termination readback size does not match the internal viewport",
                ));
            }
            termination_integration_steps(&values, self.grid.depth)?;
            let bytes = u64::try_from(values.len()).map_err(|_| {
                crate::core::error::RenderError::render("termination readback length exceeds u64")
            })? * std::mem::size_of::<f32>() as u64;
            let allocation =
                tracked_host_allocation(bytes, "nephele.media.termination-readback-cache")?;
            self.termination_readback = Some(TerminationReadback { values, allocation });
        }
        self.termination_readback.as_ref().ok_or_else(|| {
            crate::core::error::RenderError::render("termination readback cache is unavailable")
        })
    }

    pub(super) fn allocation_breakdown(&self) -> crate::media::AllocationBreakdown {
        crate::media::AllocationBreakdown {
            // Persistent media resources are device-local. Acceptance peak
            // host-visible evidence comes from the scoped allocation ledger.
            host_visible_bytes: 0,
            froxel_device_local_bytes: self.device_local_bytes - self.density_device_local_bytes,
            density_device_local_bytes: self.density_device_local_bytes,
            majorant_device_local_bytes: 0,
            staging_readback_bytes: self.staging_bytes,
        }
    }

    fn diagnostics(&self, adapter: &wgpu::Adapter) -> MediaExecutionDiagnostics {
        let allocation = self.allocation_breakdown();
        let info = adapter.get_info();
        let (decision, reason) = self.last_history_decision.diagnostic();
        let froxel_count =
            u64::from(self.grid.width) * u64::from(self.grid.height) * u64::from(self.grid.depth);
        MediaExecutionDiagnostics {
            majorant_proof: self.majorant_proof.clone(),
            majorant_valid: self.majorant_proof.is_some(),
            sample_count: u64::from(self.viewport.0) * u64::from(self.viewport.1),
            step_count: froxel_count
                + self.sun_transmittance_diagnostic.executed_steps
                + self.density_froxel_count * u64::from(self.grid.depth) * 2
                + self
                    .terrain_trace
                    .as_ref()
                    .map_or(0, |trace| trace.query_count),
            temporal_history_decision: decision.into(),
            temporal_history_reason: reason.into(),
            host_visible_bytes: allocation.host_visible_bytes,
            froxel_device_local_bytes: allocation.froxel_device_local_bytes,
            density_device_local_bytes: allocation.density_device_local_bytes,
            majorant_device_local_bytes: allocation.majorant_device_local_bytes,
            staging_readback_bytes: allocation.staging_readback_bytes,
            adapter: info.name,
            backend: format!("{:?}", info.backend),
            driver: [info.driver, info.driver_info]
                .into_iter()
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
            source_revision: env!("FORGE3D_GIT_SHA_FULL").into(),
            executed_multi_scatter: self.multiple_scatter_dispatches > 0,
            single_scatter_dispatches: self.single_scatter_dispatches,
            multiple_scatter_dispatches: self.multiple_scatter_dispatches,
            terrain_trace_queries: self
                .terrain_trace
                .as_ref()
                .map_or(0, |trace| trace.query_count),
            sun_transmittance_method: self.sun_transmittance_diagnostic.method.into(),
            sun_transmittance_bias: self.sun_transmittance_diagnostic.bias.into(),
            sun_transmittance_max_segment_length: self
                .sun_transmittance_diagnostic
                .segment_length
                .map(f64::from),
            sun_transmittance_executed_steps: self.sun_transmittance_diagnostic.executed_steps,
            sun_transmittance_max_abs_error: f64::from(
                self.sun_transmittance_diagnostic.measured_abs_error,
            ),
            single_scatter_luminance: 0.0,
            multiple_scatter_luminance: 0.0,
            energy_accounting_residual: None,
        }
    }
}

fn buffer_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn texture_entry<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn image_copy(texture: &wgpu::Texture, aspect: wgpu::TextureAspect) -> wgpu::ImageCopyTexture<'_> {
    wgpu::ImageCopyTexture {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect,
    }
}

fn output_extent(viewport: (u32, u32)) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: viewport.0,
        height: viewport.1,
        depth_or_array_layers: 1,
    }
}

pub(crate) fn acceptance_readback_bytes(
    output: (u32, u32),
    internal: (u32, u32),
    include_no_medium: bool,
) -> u64 {
    let output_pixels = u64::from(output.0) * u64::from(output.1);
    let internal_pixels = u64::from(internal.0) * u64::from(internal.1);
    let beauty_rgba8 = 4 * output_pixels;
    let media_aovs_rgba16f = 4 * 8 * output_pixels;
    let no_medium_rgba8 = u64::from(include_no_medium) * 4 * output_pixels;
    let internal_transfers_rgba16f = 3 * 8 * internal_pixels;
    beauty_rgba8 + media_aovs_rgba16f + no_medium_rgba8 + internal_transfers_rgba16f
}

fn required_positive_f16(
    value: f32,
    quantity: &'static str,
) -> crate::core::error::RenderResult<u16> {
    let represented = half::f16::from_f32(value);
    if value > 0.0 && (!represented.is_finite() || represented == half::f16::ZERO) {
        return Err(crate::core::error::RenderError::render(format!(
            "positive canonical {quantity} {value} is not representable as finite nonzero f16"
        )));
    }
    Ok(represented.to_bits())
}

fn termination_integration_steps(
    termination: &[f32],
    froxel_depth: u32,
) -> crate::core::error::RenderResult<u64> {
    if froxel_depth == 0 {
        return Err(crate::core::error::RenderError::render(
            "termination integration requires a nonzero froxel depth",
        ));
    }
    let max_slice = (froxel_depth - 1) as f32;
    let mut steps = 0u64;
    for (index, &slice) in termination.iter().enumerate() {
        if !slice.is_finite() || slice < 0.0 || slice > max_slice || slice.fract() != 0.0 {
            return Err(crate::core::error::RenderError::render(format!(
                "termination slice {index} must be an integer in [0, {max_slice}], got {slice}"
            )));
        }
        steps = steps.checked_add(slice as u64 + 1).ok_or_else(|| {
            crate::core::error::RenderError::render("termination integration step count overflowed")
        })?;
    }
    Ok(steps)
}

fn add_termination_integration_steps(
    diagnostics: &mut MediaExecutionDiagnostics,
    termination: &[f32],
    froxel_depth: u32,
) -> crate::core::error::RenderResult<()> {
    diagnostics.step_count = diagnostics
        .step_count
        .checked_add(termination_integration_steps(termination, froxel_depth)?)
        .ok_or_else(|| {
            crate::core::error::RenderError::render("media execution step count overflowed")
        })?;
    Ok(())
}

fn fill_radiance_provider(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    viewport: (u32, u32),
    radiance: [f32; 3],
) -> crate::core::error::RenderResult<()> {
    let bytes = u64::from(viewport.0) * u64::from(viewport.1) * 8;
    let _allocation = tracked_host_allocation(bytes, "nephele.media.radiance-provider.fill")?;
    let pixel = radiance
        .map(half::f16::from_f32)
        .map(half::f16::to_bits)
        .into_iter()
        .chain([half::f16::ONE.to_bits()])
        .collect::<Vec<_>>();
    let texel_count =
        usize::try_from(u64::from(viewport.0) * u64::from(viewport.1)).map_err(|_| {
            crate::core::error::RenderError::render("radiance-provider fill size exceeds usize")
        })?;
    let texels = pixel.repeat(texel_count);
    queue.write_texture(
        image_copy(texture, wgpu::TextureAspect::All),
        bytemuck::cast_slice(&texels),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(viewport.0 * 8),
            rows_per_image: Some(viewport.1),
        },
        output_extent(viewport),
    );
    Ok(())
}

fn upload_directional_radiance_provider(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    viewport: (u32, u32),
    image: &crate::formats::hdr::HdrImage,
    environment_intensity: f32,
) -> crate::core::error::RenderResult<()> {
    if image.width == 0
        || image.height == 0
        || image.data.len() != image.pixel_count().saturating_mul(3)
    {
        return Err(crate::core::error::RenderError::render(
            "directional radiance provider requires a nonempty RGB equirectangular image",
        ));
    }
    let texel_count = usize::try_from(u64::from(viewport.0) * u64::from(viewport.1))
        .map_err(|_| crate::core::error::RenderError::render("radiance provider exceeds usize"))?;
    let _allocation = tracked_host_allocation(
        texel_count as u64 * 8,
        "nephele.media.radiance-provider.directional-upload",
    )?;
    let mut texels = Vec::with_capacity(texel_count * 4);
    for y in 0..viewport.1 {
        let source_y = ((u64::from(y) * u64::from(image.height)) / u64::from(viewport.1))
            .min(u64::from(image.height - 1)) as usize;
        for x in 0..viewport.0 {
            let source_x = ((u64::from(x) * u64::from(image.width)) / u64::from(viewport.0))
                .min(u64::from(image.width - 1)) as usize;
            let source = (source_y * image.width as usize + source_x) * 3;
            texels.extend(
                image.data[source..source + 3]
                    .iter()
                    .map(|value| scaled_ibl_radiance(*value, environment_intensity))
                    .map(half::f16::from_f32)
                    .map(half::f16::to_bits),
            );
            texels.push(half::f16::ONE.to_bits());
        }
    }
    queue.write_texture(
        image_copy(texture, wgpu::TextureAspect::All),
        bytemuck::cast_slice(&texels),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(viewport.0 * 8),
            rows_per_image: Some(viewport.1),
        },
        wgpu::Extent3d {
            width: viewport.0,
            height: viewport.1,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

fn scaled_ibl_radiance(radiance: f32, intensity: f32) -> f32 {
    radiance * intensity.max(0.0)
}

fn diffuse_ibl_irradiance(radiance: [f32; 3]) -> [f32; 3] {
    radiance.map(|value| value * std::f32::consts::PI)
}

#[cfg(feature = "extension-module")]
pub(crate) fn ibl_mean_radiance(env_maps: &crate::lighting::ibl_wrapper::IBL) -> [f32; 3] {
    let Some(image) = env_maps.hdr_image() else {
        return [0.0; 3];
    };
    let pixels = image.data.chunks_exact(3);
    let count = pixels.len();
    if count == 0 {
        return [0.0; 3];
    }
    let mut mean = [0.0f64; 3];
    for pixel in pixels {
        for channel in 0..3 {
            mean[channel] += f64::from(pixel[channel]);
        }
    }
    mean.map(|value| (value / count as f64) as f32 * env_maps.intensity.max(0.0))
}

#[cfg(feature = "extension-module")]
impl crate::terrain::renderer::TerrainScene {
    pub(super) fn replace_media_terrain_occlusion_enabled(
        &self,
        enabled: bool,
    ) -> crate::core::error::RenderResult<bool> {
        let mut value = self.media_terrain_occlusion_enabled.lock().map_err(|_| {
            crate::core::error::RenderError::render(
                "media terrain-occlusion control mutex poisoned",
            )
        })?;
        Ok(std::mem::replace(&mut *value, enabled))
    }

    pub(super) fn realtime_media_diagnostics(
        &self,
    ) -> crate::core::error::RenderResult<MediaExecutionDiagnostics> {
        let mut resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let resources = resources.as_mut().ok_or_else(|| {
            crate::core::error::RenderError::render("realtime media is not attached")
        })?;
        let mut diagnostics = resources.diagnostics(&self.adapter);
        let froxel_depth = resources.grid.depth;
        let termination = resources
            .termination_readback(self.device.as_ref(), self.queue.as_ref())?
            .values
            .as_slice();
        add_termination_integration_steps(&mut diagnostics, termination, froxel_depth)?;
        let rgba = crate::core::hdr::read_hdr_texture(
            self.device.as_ref(),
            self.queue.as_ref(),
            &resources.integrated,
            resources.viewport.0,
            resources.viewport.1,
            wgpu::TextureFormat::Rgba16Float,
        )
        .map_err(crate::core::error::RenderError::render)?;
        let mut total_luminance = 0.0f64;
        let mut multiple_luminance = 0.0f64;
        for pixel in rgba.chunks_exact(4) {
            total_luminance += f64::from(pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722);
            multiple_luminance += f64::from(pixel[3].max(0.0));
        }
        let pixel_count = u64::from(resources.viewport.0) * u64::from(resources.viewport.1);
        let normalization = 1.0 / pixel_count.max(1) as f64;
        diagnostics.multiple_scatter_luminance = multiple_luminance * normalization;
        diagnostics.single_scatter_luminance =
            (total_luminance - multiple_luminance).max(0.0) * normalization;
        let transmittance = crate::core::hdr::read_hdr_texture(
            self.device.as_ref(),
            self.queue.as_ref(),
            &resources.transmittance,
            resources.viewport.0,
            resources.viewport.1,
            wgpu::TextureFormat::Rgba16Float,
        )
        .map_err(crate::core::error::RenderError::render)?;
        diagnostics.energy_accounting_residual = Some(
            transmittance
                .chunks_exact(4)
                .map(|pixel| f64::from(pixel[3]))
                .fold(0.0, f64::max),
        );
        Ok(diagnostics)
    }

    pub(super) fn read_realtime_media_termination_slice(
        &self,
    ) -> crate::core::error::RenderResult<(Vec<f32>, (u32, u32), u32, ResourceHandle)> {
        let mut resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let resources = resources.as_mut().ok_or_else(|| {
            crate::core::error::RenderError::render("realtime media is not attached")
        })?;
        resources.termination_readback(self.device.as_ref(), self.queue.as_ref())?;
        let readback = resources.termination_readback.take().ok_or_else(|| {
            crate::core::error::RenderError::render("termination readback cache is unavailable")
        })?;
        Ok((
            readback.values,
            resources.viewport,
            resources.grid.depth,
            readback.allocation,
        ))
    }

    pub(super) fn realtime_media_staging_bytes(&self) -> crate::core::error::RenderResult<u64> {
        self.media_resources
            .lock()
            .map_err(|_| crate::core::error::RenderError::render("media_resources mutex poisoned"))?
            .as_ref()
            .map(|resources| resources.staging_bytes)
            .ok_or_else(|| {
                crate::core::error::RenderError::render("realtime media is not attached")
            })
    }

    pub(super) fn prepare_realtime_media_radiance_provider(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rendered: Option<&crate::terrain::renderer::atmosphere::RenderedSky>,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        environment: Option<&crate::formats::hdr::HdrImage>,
        environment_intensity: f32,
        fallback_radiance: [f32; 3],
    ) -> crate::core::error::RenderResult<()> {
        let mut resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let Some(resources) = resources.as_mut() else {
            return Ok(());
        };
        let sky = &decoded.sky;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        sky.enabled.hash(&mut hasher);
        sky.model.hash(&mut hasher);
        for value in [
            sky.turbidity,
            sky.ground_albedo,
            sky.ozone_du,
            sky.mie_g,
            sky.sun_intensity,
            sky.sun_size,
            sky.aerial_density,
            sky.sky_exposure,
        ] {
            value.to_bits().hash(&mut hasher);
        }
        sky.aerial_perspective.hash(&mut hasher);
        if let Some(lut) = sky.lut_handle.as_ref() {
            lut.deterministic_sha256().hash(&mut hasher);
        }
        if let Some(rendered) = rendered {
            encoder.copy_texture_to_texture(
                image_copy(&rendered.directional_texture, wgpu::TextureAspect::All),
                image_copy(&resources.radiance_provider, wgpu::TextureAspect::All),
                output_extent(resources.viewport),
            );
        } else if let Some(environment) = environment {
            upload_directional_radiance_provider(
                self.queue.as_ref(),
                &resources.radiance_provider,
                resources.viewport,
                environment,
                environment_intensity,
            )?;
            environment.width.hash(&mut hasher);
            environment.height.hash(&mut hasher);
            for value in &environment.data {
                value.to_bits().hash(&mut hasher);
            }
        } else {
            fill_radiance_provider(
                self.queue.as_ref(),
                &resources.radiance_provider,
                resources.viewport,
                fallback_radiance,
            )?;
        }
        for value in fallback_radiance {
            value.to_bits().hash(&mut hasher);
        }
        environment_intensity.to_bits().hash(&mut hasher);
        resources.radiance_provider_identity = hasher.finish();
        resources.diffuse_ibl = diffuse_ibl_irradiance(fallback_radiance);
        Ok(())
    }

    pub(super) fn realtime_media_graph_config(
        &self,
        viewport: (u32, u32),
    ) -> crate::core::error::RenderResult<
        crate::terrain::renderer::render_graph::TerrainMediaGraphConfig,
    > {
        let resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let Some(resources) = resources.as_ref() else {
            return Ok(
                crate::terrain::renderer::render_graph::TerrainMediaGraphConfig::disabled(
                    viewport.0, viewport.1,
                ),
            );
        };
        if resources.viewport != viewport {
            return Err(crate::core::error::RenderError::render(format!(
                "prepared realtime media viewport {:?} does not match render viewport {viewport:?}",
                resources.viewport
            )));
        }
        Ok(
            crate::terrain::renderer::render_graph::TerrainMediaGraphConfig {
                enabled: true,
                froxel_grid: resources.grid,
                resource_version: resources.resource_version,
            },
        )
    }

    pub(super) fn bind_realtime_media_graph_resources(
        &self,
        execution: &mut crate::core::framegraph_impl::RendererGraphExecution,
        handles: crate::terrain::renderer::render_graph::TerrainMediaGraphHandles,
        scene_depth: Arc<TrackedTexture>,
    ) -> crate::core::error::RenderResult<()> {
        let resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let resources = resources.as_ref().ok_or_else(|| {
            crate::core::error::RenderError::render(
                "media graph cannot bind resources without an attached canonical medium",
            )
        })?;
        execution.bind_texture(handles.scene_depth, scene_depth)?;
        execution.bind_texture(
            handles.radiance_provider,
            resources.radiance_provider.clone(),
        )?;
        execution.bind_texture(handles.extinction, resources.extinction.clone())?;
        execution.bind_texture(
            handles.light_transmittance,
            resources.light_transmittance.clone(),
        )?;
        execution.bind_texture(handles.in_scatter, resources.in_scatter.clone())?;
        execution.bind_texture(handles.integrated, resources.integrated.clone())?;
        execution.bind_texture(handles.transmittance, resources.transmittance.clone())?;
        execution.bind_texture(handles.cloud_shadow, resources.cloud_shadow.clone())?;
        execution.bind_texture(handles.optical_depth, resources.optical_depth.clone())?;
        execution.bind_texture(handles.history_previous, resources.history.clone())?;
        execution.bind_texture(
            handles.history_depth_previous,
            resources.history_depth.clone(),
        )?;
        execution.bind_texture(handles.history_current, resources.history_next.clone())?;
        execution.bind_texture(
            handles.history_depth_current,
            resources.history_depth_next.clone(),
        )?;
        execution.bind_texture(handles.composite, resources.composite_linear_hdr.clone())?;
        Ok(())
    }

    pub(super) fn prepare_realtime_media_for_terrain(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        medium: &crate::media::Medium,
        version: u64,
    ) -> crate::core::error::RenderResult<crate::media::MediumIdentity> {
        let width = ((params.size_px.0 as f32 * params.render_scale.clamp(0.25, 4.0))
            .round()
            .max(1.0)) as u32;
        let height = ((params.size_px.1 as f32 * params.render_scale.clamp(0.25, 4.0))
            .round()
            .max(1.0)) as u32;
        let (camera, view, projection) = Self::build_camera_matrices(params);
        let depth = FroxelDepthTransform::new(params.clip.0, params.clip.1)
            .map_err(crate::core::error::RenderError::render)?;
        self.prepare_realtime_media(
            (width, height),
            medium,
            version,
            camera,
            (projection * view).inverse(),
            depth,
            terrain_light_direction(&params.camera_mode, decoded.light.direction),
        )
        .map(|(identity, _)| identity)
    }

    pub(super) fn prepare_realtime_media_terrain_trace(
        &self,
        camera_mode: &str,
        heights: &[f32],
        dimensions: (u32, u32),
        terrain_data_hash: u64,
        terrain_span: f32,
        exaggeration: f32,
        albedo: [f32; 3],
    ) -> crate::core::error::RenderResult<()> {
        let mut resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let Some(resources) = resources.as_mut() else {
            return Ok(());
        };
        if !crate::terrain::is_yup_camera_mode(camera_mode) {
            return Err(crate::core::error::RenderError::render(
                "realtime terrain/media transport requires the authoritative mesh:yup world frame",
            ));
        }
        let terrain_identity = stable_words_hash(
            [
                terrain_data_hash,
                u64::from(dimensions.0),
                u64::from(dimensions.1),
                u64::from(terrain_span.to_bits()),
                u64::from(exaggeration.to_bits()),
            ]
            .into_iter()
            .chain(albedo.into_iter().map(|value| u64::from(value.to_bits()))),
        );
        resources.prepare_terrain_trace(
            self.device.as_ref(),
            self.queue.as_ref(),
            self.adapter.get_info().backend,
            heights,
            dimensions,
            terrain_span,
            exaggeration,
            albedo,
            terrain_identity,
            0.0,
        )
    }

    pub(super) fn prepare_realtime_media(
        &self,
        viewport: (u32, u32),
        medium: &crate::media::Medium,
        density_version: u64,
        camera: glam::Vec3,
        inverse_view_projection: glam::Mat4,
        depth: FroxelDepthTransform,
        sun_direction: glam::Vec3,
    ) -> crate::core::error::RenderResult<(crate::media::MediumIdentity, u64)> {
        let mut slot = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        if slot.as_ref().is_none_or(|r| r.viewport != viewport) {
            let version = slot
                .as_ref()
                .map_or(1, |r| r.resource_version.wrapping_add(1));
            *slot = Some(TerrainMediaResources::new(
                self.device.as_ref(),
                self.queue.as_ref(),
                viewport,
                version,
            )?);
        }
        let r = slot.as_mut().expect("created above");
        let identity = r.upload_canonical_extinction(
            self.queue.as_ref(),
            medium,
            density_version,
            camera,
            inverse_view_projection,
            depth,
            sun_direction,
        )?;
        Ok((identity, r.resource_version))
    }
    pub(super) fn clear_realtime_media(&self) -> crate::core::error::RenderResult<()> {
        *self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })? = None;
        Ok(())
    }

    pub(super) fn encode_attached_realtime_media_inject_for_terrain(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
    ) -> crate::core::error::RenderResult<Option<HistoryDecision>> {
        let (camera, view, projection) = Self::build_camera_matrices(params);
        self.encode_attached_realtime_media_inject_with_matrices(
            encoder, params, decoded, camera, view, projection,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_attached_realtime_media_inject_with_matrices(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        camera: glam::Vec3,
        view: glam::Mat4,
        projection: glam::Mat4,
    ) -> crate::core::error::RenderResult<Option<HistoryDecision>> {
        let view_projection = projection * view;
        let mut resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let Some(resources) = resources.as_mut() else {
            return Ok(None);
        };
        let expected_viewport = (
            ((params.size_px.0 as f32 * params.render_scale.clamp(0.25, 4.0))
                .round()
                .max(1.0)) as u32,
            ((params.size_px.1 as f32 * params.render_scale.clamp(0.25, 4.0))
                .round()
                .max(1.0)) as u32,
        );
        if resources.viewport != expected_viewport {
            return Err(crate::core::error::RenderError::render(
                "realtime media viewport does not match the linear-HDR terrain target",
            ));
        }
        let medium_identity = resources.medium_identity.ok_or_else(|| {
            crate::core::error::RenderError::render("canonical medium upload is incomplete")
        })?;
        let phase = match resources.phase {
            crate::media::Phase::Isotropic => (0.0, 0.0),
            crate::media::Phase::HenyeyGreenstein { g } => (g, 1.0),
        };
        let adapter_info = self.adapter.get_info();
        let sun_direction = terrain_light_direction(&params.camera_mode, decoded.light.direction);
        let terrain_occlusion_enabled =
            *self.media_terrain_occlusion_enabled.lock().map_err(|_| {
                crate::core::error::RenderError::render(
                    "media terrain-occlusion control mutex poisoned",
                )
            })?;
        let history_key = MediaHistoryKey {
            camera: stable_words_hash(
                view_projection
                    .to_cols_array()
                    .into_iter()
                    .map(|value| u64::from(value.to_bits())),
            ),
            scene_depth: stable_words_hash(
                [
                    u64::from(params.terrain_data_revision.is_some()),
                    params.terrain_data_revision.unwrap_or(0),
                    u64::from(params.z_scale.to_bits()),
                    u64::from(params.terrain_span.to_bits()),
                ]
                .into_iter(),
            ),
            medium: medium_identity,
            lighting: media_lighting_identity(
                sun_direction,
                decoded.light.color,
                decoded.light.intensity,
                terrain_occlusion_enabled,
            ) ^ resources.radiance_provider_identity
                ^ resources.blue_noise_identity
                ^ resources
                    .terrain_trace
                    .as_ref()
                    .map_or(0, |trace| trace.terrain_identity),
            viewport: resources.viewport,
            resource_version: resources.resource_version,
            adapter: stable_adapter_hash(&adapter_info),
        };
        let depth = FroxelDepthTransform::new(params.clip.0, params.clip.1)
            .map_err(crate::core::error::RenderError::render)?;
        let uniforms = MediaUniforms {
            view_proj: view_projection.to_cols_array_2d(),
            inv_view_proj: view_projection.inverse().to_cols_array_2d(),
            previous_view_proj: resources.previous_view_projection.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            camera: [camera.x, camera.y, camera.z, 0.0],
            sun: [sun_direction.x, sun_direction.y, sun_direction.z, 0.0],
            sun_radiance: [
                decoded.light.color[0],
                decoded.light.color[1],
                decoded.light.color[2],
                decoded.light.intensity,
            ],
            sigma_s: [
                resources.sigma_s[0],
                resources.sigma_s[1],
                resources.sigma_s[2],
                0.0,
            ],
            sigma_t: [
                resources.sigma_t[0],
                resources.sigma_t[1],
                resources.sigma_t[2],
                0.0,
            ],
            depth: {
                let mut values = depth.wgsl_params();
                // The existing live viewer fog path establishes 0.2 as its
                // deterministic temporal blend default.
                values[3] = 0.2;
                values
            },
            grid: [
                resources.grid.width,
                resources.grid.height,
                resources.grid.depth,
                0,
            ],
            viewport: [resources.viewport.0, resources.viewport.1, 0, 0],
            scattering: [f32::from(terrain_occlusion_enabled), 1.0, phase.0, phase.1],
            terrain_albedo: [
                resources.terrain_albedo[0],
                resources.terrain_albedo[1],
                resources.terrain_albedo[2],
                0.0,
            ],
            diffuse_ibl: [
                resources.diffuse_ibl[0],
                resources.diffuse_ibl[1],
                resources.diffuse_ibl[2],
                0.0,
            ],
        };
        resources
            .prepare_and_encode_inject(
                self.device.as_ref(),
                self.queue.as_ref(),
                encoder,
                &self.csm_renderer,
                uniforms,
                history_key,
            )
            .map(Some)
    }

    pub(super) fn encode_attached_realtime_media_integrate(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_depth_view: &wgpu::TextureView,
    ) -> crate::core::error::RenderResult<()> {
        self.media_resources
            .lock()
            .map_err(|_| crate::core::error::RenderError::render("media_resources mutex poisoned"))?
            .as_mut()
            .ok_or_else(|| {
                crate::core::error::RenderError::render("realtime media is not attached")
            })?
            .encode_integrate(self.device.as_ref(), encoder, scene_depth_view)
    }

    pub(super) fn encode_attached_realtime_media_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        terrain_linear_hdr_view: &wgpu::TextureView,
    ) -> crate::core::error::RenderResult<()> {
        self.media_resources
            .lock()
            .map_err(|_| crate::core::error::RenderError::render("media_resources mutex poisoned"))?
            .as_mut()
            .ok_or_else(|| {
                crate::core::error::RenderError::render("realtime media is not attached")
            })?
            .encode_composite(self.device.as_ref(), encoder, terrain_linear_hdr_view)
    }

    pub(super) fn commit_attached_realtime_media_history(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_depth: &wgpu::Texture,
    ) -> crate::core::error::RenderResult<HistoryDecision> {
        self.media_resources
            .lock()
            .map_err(|_| crate::core::error::RenderError::render("media_resources mutex poisoned"))?
            .as_mut()
            .ok_or_else(|| {
                crate::core::error::RenderError::render("realtime media is not attached")
            })?
            .encode_history_commit(encoder, scene_depth)
    }

    pub(super) fn realtime_media_composite_view(
        &self,
    ) -> crate::core::error::RenderResult<wgpu::TextureView> {
        let resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        Ok(resources
            .as_ref()
            .ok_or_else(|| {
                crate::core::error::RenderError::render("realtime media is not attached")
            })?
            .composite_linear_hdr
            .create_view(&wgpu::TextureViewDescriptor::default()))
    }

    pub(super) fn copy_realtime_media_capture_textures(
        &self,
        selected: [bool; 4],
    ) -> crate::core::error::RenderResult<RealtimeMediaCaptureTextures> {
        let resources = self.media_resources.lock().map_err(|_| {
            crate::core::error::RenderError::render("media_resources mutex poisoned")
        })?;
        let resources = resources.as_ref().ok_or_else(|| {
            crate::core::error::RenderError::render(
                "media AOV capture requires an attached canonical medium",
            )
        })?;
        let extent = output_extent(resources.viewport);
        let mut capture_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain.aov.media.capture.encoder"),
                });
        let mut copy = |enabled: bool,
                        label: &'static str,
                        source: &TrackedTexture|
         -> crate::core::error::RenderResult<Option<TrackedTexture>> {
            if !enabled {
                return Ok(None);
            }
            let (target, _) = tracked_texture(
                self.device.as_ref(),
                label,
                extent,
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            )?;
            capture_encoder.copy_texture_to_texture(
                image_copy(source, wgpu::TextureAspect::All),
                image_copy(&target, wgpu::TextureAspect::All),
                extent,
            );
            Ok(Some(target))
        };
        let captures = RealtimeMediaCaptureTextures {
            transmittance: copy(
                selected[0],
                "terrain.aov.transmittance.capture",
                &resources.transmittance,
            )?,
            in_scatter: copy(
                selected[1],
                "terrain.aov.in_scatter.capture",
                &resources.integrated,
            )?,
            cloud_shadow: copy(
                selected[2],
                "terrain.aov.cloud_shadow.capture",
                &resources.cloud_shadow,
            )?,
            optical_depth: copy(
                selected[3],
                "terrain.aov.optical_depth.capture",
                &resources.optical_depth,
            )?,
        };
        self.queue.submit(Some(capture_encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(captures)
    }
}

fn stable_words_hash(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for word in words {
        word.hash(&mut hasher);
    }
    hasher.finish()
}

fn media_lighting_identity(
    sun_direction: glam::Vec3,
    sun_color: [f32; 3],
    sun_intensity: f32,
    terrain_occlusion_enabled: bool,
) -> u64 {
    stable_words_hash(
        sun_direction
            .to_array()
            .into_iter()
            .chain(sun_color)
            .chain([sun_intensity])
            .map(|value| u64::from(value.to_bits()))
            .chain([u64::from(terrain_occlusion_enabled)]),
    )
}

fn stable_adapter_hash(info: &wgpu::AdapterInfo) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    info.name.hash(&mut hasher);
    info.vendor.hash(&mut hasher);
    info.device.hash(&mut hasher);
    format!("{:?}", info.backend).hash(&mut hasher);
    info.driver.hash(&mut hasher);
    info.driver_info.hash(&mut hasher);
    hasher.finish()
}

/// Convert the public terrain light's legacy Z-up azimuth/elevation axes into
/// the world axes selected by the terrain camera/geometry mode.
pub(crate) fn terrain_light_direction(camera_mode: &str, direction: [f32; 3]) -> glam::Vec3 {
    let direction = glam::Vec3::from_array(direction);
    if crate::terrain::is_yup_camera_mode(camera_mode) {
        glam::Vec3::new(direction.x, direction.z, direction.y)
    } else {
        direction
    }
}

fn froxel_world_position(
    grid: FroxelGrid,
    id: [u32; 3],
    camera: glam::Vec3,
    inv: glam::Mat4,
    depth: FroxelDepthTransform,
) -> glam::Vec3 {
    let visible = glam::Vec2::new(grid.visible_width() as f32, grid.visible_height() as f32);
    let uv = glam::Vec2::new(
        (id[0] as f32 + 0.5 - FROXEL_OFF_AXIS_BORDER as f32) / visible.x,
        (id[1] as f32 + 0.5 - FROXEL_OFF_AXIS_BORDER as f32) / visible.y,
    );
    let ndc = glam::Vec2::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far = inv * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
    let ray = (far.truncate() / far.w - camera).normalize();
    camera + ray * depth.distance_at_unit((id[2] as f32 + 0.5) / grid.depth as f32)
}

fn froxel_xy_for_uv(grid: FroxelGrid, uv: glam::Vec2, jitter: glam::Vec2) -> [u32; 2] {
    let visible = glam::Vec2::new(grid.visible_width() as f32, grid.visible_height() as f32);
    let position = uv * visible + jitter + glam::Vec2::splat(FROXEL_OFF_AXIS_BORDER as f32);
    [
        position.x.floor().clamp(0.0, (grid.width - 1) as f32) as u32,
        position.y.floor().clamp(0.0, (grid.height - 1) as f32) as u32,
    ]
}

fn canonical_sun_transmittance(
    medium: &crate::media::Medium,
    origin: glam::Vec3,
    direction: glam::Vec3,
) -> crate::core::error::RenderResult<([f32; 3], SunTransmittanceDiagnostic)> {
    let direction = direction.try_normalize().ok_or_else(|| {
        crate::core::error::RenderError::render("media sun direction must be finite and nonzero")
    })?;
    let interval = match medium.density() {
        crate::media::DensityField::Homogeneous(_) => {
            let extinction = medium.extinction_at(origin.to_array()).components();
            return Ok((
                extinction.map(|sigma_t| if sigma_t > 0.0 { 0.0 } else { 1.0 }),
                SunTransmittanceDiagnostic {
                    method: "analytic_unbounded_homogeneous",
                    bias: "none_exact",
                    segment_length: None,
                    executed_steps: 0,
                    measured_abs_error: 0.0,
                },
            ));
        }
        crate::media::DensityField::PerlinWorley(field) => {
            ray_box_interval(origin, direction, field.transform.bounds)
        }
        crate::media::DensityField::Grid3D(field) => {
            ray_box_interval(origin, direction, field.transform().bounds)
        }
    };
    let Some((start, end)) = interval else {
        return Ok((
            [1.0; 3],
            SunTransmittanceDiagnostic {
                method: "bounded_nested_midpoint",
                bias: "fine_midpoint_with_coarse_fine_abs_rgb_error",
                segment_length: Some(0.0),
                executed_steps: 0,
                measured_abs_error: 0.0,
            },
        ));
    };
    let segment_length = end - start;
    let coarse_steps = sun_integration_steps(medium.density(), direction, segment_length).max(1);
    let fine_steps = coarse_steps.saturating_mul(2);
    let integrate = |steps: u32| {
        let step = segment_length / steps as f32;
        let mut tau = [0.0f32; 3];
        for i in 0..steps {
            let point = origin + direction * (start + (i as f32 + 0.5) * step);
            let extinction = medium.extinction_at(point.to_array()).components();
            for channel in 0..3 {
                tau[channel] += extinction[channel] * step;
            }
        }
        tau.map(|value| (-value).exp())
    };
    let coarse = integrate(coarse_steps);
    let fine = integrate(fine_steps);
    Ok((
        fine,
        SunTransmittanceDiagnostic {
            method: "bounded_nested_midpoint",
            bias: "fine_midpoint_with_coarse_fine_abs_rgb_error",
            segment_length: Some(segment_length),
            executed_steps: u64::from(coarse_steps) + u64::from(fine_steps),
            measured_abs_error: (0..3)
                .map(|channel| (fine[channel] - coarse[channel]).abs())
                .fold(0.0, f32::max),
        },
    ))
}

fn sun_integration_steps(
    density: &crate::media::DensityField,
    direction: glam::Vec3,
    segment_length: f32,
) -> u32 {
    match density {
        crate::media::DensityField::Homogeneous(_) => 1,
        crate::media::DensityField::Grid3D(grid) => {
            let bounds = grid.transform().bounds;
            let dimensions = grid.dimensions();
            let crossings_per_world_unit = (0..3)
                .map(|axis| {
                    direction[axis].abs() * dimensions[axis].saturating_sub(1) as f32
                        / (bounds.max[axis] - bounds.min[axis])
                })
                .sum::<f32>();
            // Two midpoint samples per crossed source cell satisfy the
            // Nyquist rate of the trilinearly reconstructed R16 field.
            (segment_length * crossings_per_world_unit * 2.0).ceil() as u32
        }
        crate::media::DensityField::PerlinWorley(field) => {
            let highest_octave = field.octaves.saturating_sub(1);
            let highest_frequency = field.frequency * 2.0f32.powi(highest_octave as i32);
            (segment_length * highest_frequency * 2.0).ceil() as u32
        }
    }
}

fn ray_box_interval(
    origin: glam::Vec3,
    direction: glam::Vec3,
    bounds: crate::media::Bounds3,
) -> Option<(f32, f32)> {
    let mut enter = 0.0f32;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        let o = origin[axis];
        let d = direction[axis];
        if d == 0.0 {
            if o < bounds.min[axis] || o > bounds.max[axis] {
                return None;
            }
            continue;
        }
        let a = (bounds.min[axis] - o) / d;
        let b = (bounds.max[axis] - o) / d;
        enter = enter.max(a.min(b));
        exit = exit.min(a.max(b));
    }
    (exit > enter).then_some((enter, exit))
}

fn tracked_texture(
    device: &wgpu::Device,
    label: &'static str,
    size: wgpu::Extent3d,
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> crate::core::error::RenderResult<(TrackedTexture, u64)> {
    let desc = wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension,
        format,
        usage,
        view_formats: &[],
    };
    let bytes = calculate_texture_descriptor_size(&desc);
    Ok((tracked_create_texture(device, &desc)?, bytes))
}

fn parse_blue_noise_asset() -> Result<Vec<u8>, String> {
    let values: Vec<u8> = BLUE_NOISE_ASSET
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .flat_map(str::split_whitespace)
        .map(|v| v.parse::<u8>())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("invalid blue-noise rank: {e}"))?;
    if values.len() != (BLUE_NOISE_WIDTH * BLUE_NOISE_HEIGHT) as usize {
        return Err("blue-noise tile must contain 64 ranks".into());
    }
    let mut sorted = values.clone();
    sorted.sort_unstable();
    if sorted.iter().copied().ne(0..64) {
        return Err("blue-noise tile must contain every rank 0..63 exactly once".into());
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn radiance_provider_clear_is_valid_without_optional_clear_texture_feature() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("radiance-provider clear regression requires Apple Metal");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nephele.radiance-provider-clear.device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        ))
        .expect("feature-free Metal device must construct");
        assert!(!device.features().contains(wgpu::Features::CLEAR_TEXTURE));
        let texture = tracked_create_texture(
            &device,
            &wgpu::TextureDescriptor {
                label: Some("nephele.radiance-provider-clear.texture"),
                size: output_extent((1, 1)),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
        .expect("radiance-provider clear texture must be tracked");
        queue.write_texture(
            image_copy(&texture, wgpu::TextureAspect::All),
            bytemuck::cast_slice(&[half::f16::ONE.to_bits(); 4]),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            output_extent((1, 1)),
        );
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        fill_radiance_provider(&queue, &texture, (1, 1), [0.0; 3]).unwrap();
        queue.submit(std::iter::empty());
        let validation_error = pollster::block_on(device.pop_error_scope());
        assert!(
            validation_error.is_none(),
            "radiance-provider clear must not require an optional feature: {validation_error:?}"
        );
        let values = crate::core::hdr::read_hdr_texture(
            &device,
            &queue,
            &texture,
            1,
            1,
            wgpu::TextureFormat::Rgba16Float,
        )
        .unwrap();
        assert_eq!(values, [0.0, 0.0, 0.0, 1.0]);
    }

    #[cfg(all(not(feature = "extension-module"), target_os = "macos"))]
    #[test]
    fn apple_metal_viewer_media_submits_and_reads_main_and_aovs_deterministically() {
        struct Readback {
            composite: Vec<f32>,
            transmittance: Vec<f32>,
            in_scatter: Vec<f32>,
            cloud_shadow: Vec<f32>,
            optical_depth: Vec<f32>,
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("physical viewer-media proof requires an Apple Metal adapter");
        let adapter_info = adapter.get_info();
        assert_eq!(adapter_info.backend, wgpu::Backend::Metal);
        assert!(
            adapter_info.name.to_ascii_lowercase().contains("apple"),
            "physical viewer-media proof requires Apple silicon, got {}",
            adapter_info.name
        );
        assert!(adapter.features().contains(wgpu::Features::CLEAR_TEXTURE));
        assert!(adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE));
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("viewer-media-physical-device"),
                required_features: wgpu::Features::CLEAR_TEXTURE
                    | wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: adapter.limits(),
            },
            None,
        ))
        .expect("physical viewer-media proof requires a GPU device");
        let medium = crate::media::Medium::new(
            [0.12, 0.16, 0.20],
            [0.04, 0.03, 0.02],
            crate::media::Phase::Isotropic,
            crate::media::DensityField::Grid3D(
                crate::media::Grid3D::new(
                    crate::media::SpatialTransform {
                        bounds: crate::media::Bounds3 {
                            min: [-32.0; 3],
                            max: [32.0; 3],
                        },
                    },
                    [2, 2, 2],
                    vec![1.0; 8],
                    crate::media::DensityMapping {
                        physical_density_per_authored_unit: 0.1,
                    },
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let viewport = (8, 8);
        let depth = tracked_create_texture(
            &device,
            &wgpu::TextureDescriptor {
                label: Some("viewer-media-physical-depth"),
                size: output_extent(viewport),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
        .expect("physical viewer-media depth texture must be tracked");
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let color = tracked_create_texture(
            &device,
            &wgpu::TextureDescriptor {
                label: Some("viewer-media-physical-linear-hdr"),
                size: output_extent(viewport),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
        .expect("physical viewer-media HDR texture must be tracked");
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let mut csm = crate::shadows::CsmRenderer::new(
            &device,
            crate::shadows::CsmConfig {
                cascade_count: 1,
                shadow_map_size: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let camera = glam::Vec3::ZERO;
        let view = glam::Mat4::look_at_rh(camera, glam::Vec3::NEG_Z, glam::Vec3::Y);
        let projection = glam::Mat4::perspective_rh(60.0f32.to_radians(), 1.0, 0.1, 20.0);
        csm.update_cascades(view, projection, glam::Vec3::Y, 0.1, 20.0);
        csm.upload_uniforms(&queue);

        let run = |depth_clear: f32| {
            let mut pass = ViewerMediaPass::new(&device, &queue, viewport, medium.clone(), 7)
                .expect("live viewer media pass must construct");
            pass.prepare_viewer_terrain_trace(
                &device,
                &queue,
                &adapter,
                viewport,
                &[-2.0; 4],
                (2, 2),
                [-4.0, -4.0],
                [8.0, 8.0],
                -2.0,
                0.0,
                1.0,
                11,
            )
            .expect("live viewer media proof must prepare production terrain tracing");
            pass.prepare_viewer_frame(&queue, camera, projection * view, 0.1, 20.0, glam::Vec3::Y)
                .expect("live viewer media proof must prepare the canonical medium");
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewer-media-physical-encoder"),
            });
            {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer-media-physical-scene-clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 2.0,
                                g: 1.0,
                                b: 0.5,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(depth_clear),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            for shadow_view in &csm.shadow_map_views {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewer-media-physical-shadow-clear"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: shadow_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            pass.encode(
                &device,
                &queue,
                &adapter,
                &mut encoder,
                viewport,
                camera,
                view,
                projection,
                0.1,
                20.0,
                glam::Vec3::Y,
                [3.0, 2.0, 1.0],
                11,
                &depth,
                &depth_view,
                &color_view,
                &csm,
            )
            .expect("live viewer media encode must reach dispatch");
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
            let read = |texture: &wgpu::Texture| {
                crate::core::hdr::read_hdr_texture(
                    &device,
                    &queue,
                    texture,
                    viewport.0,
                    viewport.1,
                    wgpu::TextureFormat::Rgba16Float,
                )
                .expect("submitted viewer media texture must read back")
            };
            let mut diagnostics = pass.diagnostics(&adapter);
            assert_eq!(diagnostics.backend, "Metal");
            assert_eq!(diagnostics.sample_count, 64);
            assert!(diagnostics.majorant_valid);
            let base_steps = diagnostics.step_count;
            let froxel_depth = pass.resources.grid.depth;
            let termination = pass
                .resources
                .termination_readback(&device, &queue)
                .expect("submitted termination readback must validate")
                .values
                .clone();
            add_termination_integration_steps(&mut diagnostics, &termination, froxel_depth)
                .unwrap();
            assert_eq!(
                diagnostics.step_count - base_steps,
                termination
                    .iter()
                    .map(|slice| *slice as u64 + 1)
                    .sum::<u64>()
            );
            Readback {
                composite: read(&pass.resources.composite_linear_hdr),
                transmittance: read(&pass.resources.transmittance),
                in_scatter: read(&pass.resources.integrated),
                cloud_shadow: read(&pass.resources.cloud_shadow),
                optical_depth: read(&pass.resources.optical_depth),
            }
        };

        let first = run(1.0);
        let second = run(1.0);
        let invalid_depth = run(0.0);
        assert_eq!(first.composite, second.composite);
        assert_eq!(first.transmittance, second.transmittance);
        assert_eq!(first.in_scatter, second.in_scatter);
        assert_eq!(first.cloud_shadow, second.cloud_shadow);
        assert_eq!(first.optical_depth, second.optical_depth);
        assert!(first.composite.iter().all(|value| value.is_finite()));
        assert!(first.in_scatter.iter().all(|value| value.is_finite()));
        assert!(first
            .in_scatter
            .chunks_exact(4)
            .any(|pixel| pixel[..3].iter().any(|value| *value > 0.0)));
        assert!(first
            .transmittance
            .chunks_exact(4)
            .all(|pixel| pixel[..3].iter().all(|value| *value > 0.0 && *value < 1.0)));
        assert!(first
            .optical_depth
            .chunks_exact(4)
            .all(|pixel| pixel[..3].iter().all(|value| *value > 0.0)));
        assert!(invalid_depth
            .optical_depth
            .chunks_exact(4)
            .all(|pixel| pixel[..3].iter().all(|value| *value > 0.0)));
        assert!(invalid_depth
            .in_scatter
            .chunks_exact(4)
            .any(|pixel| pixel[..3].iter().any(|value| *value > 0.0)));
        assert!(first.cloud_shadow.chunks_exact(4).all(|pixel| pixel[..3]
            .iter()
            .all(|value| *value >= 0.0 && *value <= 1.0)));
        assert!(first
            .cloud_shadow
            .chunks_exact(4)
            .any(|pixel| pixel[..3].iter().any(|value| *value < 1.0)));
        assert!(first.composite[0] < 2.0 || first.composite[1] < 1.0);
    }
    use naga::{Expression, Handle, ShaderStage, Statement};
    use sha2::{Digest, Sha256};

    fn collect_calls(block: &naga::Block, calls: &mut Vec<Handle<naga::Function>>) {
        for statement in block {
            match statement {
                Statement::Block(block) => collect_calls(block, calls),
                Statement::If { accept, reject, .. } => {
                    collect_calls(accept, calls);
                    collect_calls(reject, calls);
                }
                Statement::Switch { cases, .. } => {
                    for case in cases {
                        collect_calls(&case.body, calls);
                    }
                }
                Statement::Loop {
                    body, continuing, ..
                } => {
                    collect_calls(body, calls);
                    collect_calls(continuing, calls);
                }
                Statement::Call { function, .. } => calls.push(*function),
                _ => {}
            }
        }
    }
    fn has_compare_sample(function: &naga::Function) -> bool {
        function.expressions.iter().any(|(_, e)| {
            matches!(
                e,
                Expression::ImageSample {
                    depth_ref: Some(_),
                    ..
                }
            )
        })
    }

    fn assert_compute_call_graph_safe(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("WGSL must validate");
        for entry in module
            .entry_points
            .iter()
            .filter(|e| e.stage == ShaderStage::Compute)
        {
            assert!(
                !has_compare_sample(&entry.function),
                "compute entry {} compares sampled depth",
                entry.name
            );
            let mut pending = Vec::new();
            collect_calls(&entry.function.body, &mut pending);
            let mut visited = std::collections::HashSet::new();
            while let Some(handle) = pending.pop() {
                if !visited.insert(handle) {
                    continue;
                }
                let function = &module.functions[handle];
                assert!(
                    !has_compare_sample(function),
                    "compute entry {} reaches comparison sampling",
                    entry.name
                );
                collect_calls(&function.body, &mut pending);
            }
        }
    }
    #[test]
    fn depth_transform_round_trips() {
        let t = FroxelDepthTransform::new(0.25, 4096.0).unwrap();
        let mut p = 0.0;
        for z in 0..64 {
            let d = t.slice_center_distance(z, 64);
            assert!(d > p);
            assert!((t.unit_at_distance(d) - (z as f32 + 0.5) / 64.0).abs() < 2e-6);
            p = d;
        }
    }
    #[test]
    fn history_rejects_all_dependencies() {
        let k = MediaHistoryKey {
            camera: 1,
            scene_depth: 2,
            medium: crate::media::MediumIdentity {
                version: 3,
                digest: [3; 32],
            },
            lighting: 4,
            viewport: (8, 8),
            resource_version: 5,
            adapter: 6,
        };
        assert_eq!(history_decision(Some(k), k), HistoryDecision::Accept);
        assert_eq!(history_decision(None, k), HistoryDecision::RejectMissing);
        assert_eq!(
            history_decision(Some(k), MediaHistoryKey { camera: 9, ..k }),
            HistoryDecision::RejectCamera
        );
        assert_eq!(
            history_decision(
                Some(k),
                MediaHistoryKey {
                    scene_depth: 9,
                    ..k
                }
            ),
            HistoryDecision::RejectDepth
        );
        assert_eq!(
            history_decision(
                Some(k),
                MediaHistoryKey {
                    medium: crate::media::MediumIdentity {
                        version: 9,
                        digest: [9; 32],
                    },
                    ..k
                }
            ),
            HistoryDecision::RejectMedium
        );
        assert_eq!(
            history_decision(Some(k), MediaHistoryKey { lighting: 9, ..k }),
            HistoryDecision::RejectLighting
        );
        let enabled = media_lighting_identity(glam::Vec3::Y, [1.0; 3], 2.0, true);
        let disabled = media_lighting_identity(glam::Vec3::Y, [1.0; 3], 2.0, false);
        assert_eq!(
            history_decision(
                Some(MediaHistoryKey {
                    lighting: enabled,
                    ..k
                }),
                MediaHistoryKey {
                    lighting: disabled,
                    ..k
                },
            ),
            HistoryDecision::RejectLighting
        );
        assert_eq!(
            history_decision(
                Some(k),
                MediaHistoryKey {
                    viewport: (9, 8),
                    ..k
                }
            ),
            HistoryDecision::RejectResize
        );
        assert_eq!(
            history_decision(
                Some(k),
                MediaHistoryKey {
                    resource_version: 9,
                    ..k
                }
            ),
            HistoryDecision::RejectResourceVersion
        );
        assert_eq!(
            history_decision(Some(k), MediaHistoryKey { adapter: 9, ..k }),
            HistoryDecision::RejectAdapter
        );
    }
    #[test]
    fn froxel_grid_covers_one_tile_beyond_every_viewport_edge() {
        let grid = FroxelGrid::for_viewport(640, 480);
        assert_eq!((grid.visible_width(), grid.visible_height()), (80, 60));
        assert_eq!((grid.width, grid.height), (82, 62));
    }
    #[test]
    fn edge_jitter_consumes_both_off_axis_border_tiles() {
        let grid = FroxelGrid::for_viewport(640, 480);
        assert_eq!(
            froxel_xy_for_uv(grid, glam::Vec2::ZERO, glam::Vec2::splat(-0.5)),
            [0, 0]
        );
        assert_eq!(
            froxel_xy_for_uv(grid, glam::Vec2::ONE, glam::Vec2::splat(0.5)),
            [grid.width - 1, grid.height - 1]
        );
    }
    #[test]
    fn sun_transmittance_integrates_complete_bounded_segment() {
        let density = crate::media::DensityField::Grid3D(
            crate::media::Grid3D::new(
                crate::media::SpatialTransform {
                    bounds: crate::media::Bounds3 {
                        min: [10.0, -1.0, -1.0],
                        max: [20.0, 1.0, 1.0],
                    },
                },
                [2, 2, 2],
                vec![1.0; 8],
                crate::media::DensityMapping {
                    physical_density_per_authored_unit: 1.0,
                },
            )
            .unwrap(),
        );
        let medium = crate::media::Medium::new(
            [0.1, 0.2, 0.3],
            [0.0; 3],
            crate::media::Phase::Isotropic,
            density,
        )
        .unwrap();
        let (transmittance, diagnostic) =
            canonical_sun_transmittance(&medium, glam::Vec3::ZERO, glam::Vec3::X).unwrap();
        for (actual, sigma_t) in transmittance.into_iter().zip([0.1, 0.2, 0.3]) {
            assert!((actual - (-sigma_t * 10.0f32).exp()).abs() < 1e-6);
        }
        assert_eq!(diagnostic.segment_length, Some(10.0));
        assert!(diagnostic.executed_steps >= 3);
        assert!(diagnostic.measured_abs_error <= 1e-6);
    }

    #[test]
    fn homogeneous_sun_transmittance_integrates_the_complete_unbounded_segment() {
        let medium = crate::media::Medium::new(
            [0.1, 0.2, 0.3],
            [0.0; 3],
            crate::media::Phase::Isotropic,
            crate::media::DensityField::Homogeneous(crate::media::Homogeneous {
                authored_density: 1.0,
                mapping: crate::media::DensityMapping {
                    physical_density_per_authored_unit: 1.0,
                },
            }),
        )
        .unwrap();
        let (transmittance, diagnostic) =
            canonical_sun_transmittance(&medium, glam::Vec3::ZERO, glam::Vec3::X).unwrap();
        assert_eq!(transmittance, [0.0; 3]);
        assert_eq!(diagnostic.method, "analytic_unbounded_homogeneous");
        assert_eq!(diagnostic.segment_length, None);
        assert_eq!(diagnostic.measured_abs_error, 0.0);
    }
    #[test]
    fn sun_step_diagnostic_sums_asymmetric_hits_and_zero_step_misses() {
        let density = crate::media::DensityField::Grid3D(
            crate::media::Grid3D::new(
                crate::media::SpatialTransform {
                    bounds: crate::media::Bounds3 {
                        min: [0.0, -1.0, -1.0],
                        max: [4.0, 1.0, 1.0],
                    },
                },
                [5, 2, 2],
                vec![1.0; 20],
                crate::media::DensityMapping {
                    physical_density_per_authored_unit: 1.0,
                },
            )
            .unwrap(),
        );
        let medium =
            crate::media::Medium::new([0.1; 3], [0.0; 3], crate::media::Phase::Isotropic, density)
                .unwrap();
        let diagnostics = [
            canonical_sun_transmittance(&medium, glam::Vec3::new(-1.0, 0.0, 0.0), glam::Vec3::X)
                .unwrap()
                .1,
            canonical_sun_transmittance(&medium, glam::Vec3::new(2.0, 0.0, 0.0), glam::Vec3::X)
                .unwrap()
                .1,
            canonical_sun_transmittance(&medium, glam::Vec3::new(-1.0, 3.0, 0.0), glam::Vec3::X)
                .unwrap()
                .1,
        ];
        let mut total = SunTransmittanceDiagnostic::default();
        for diagnostic in diagnostics {
            total.record(diagnostic);
        }
        assert_eq!(diagnostics[2].executed_steps, 0);
        assert_eq!(total.executed_steps, 36);
        assert_ne!(
            total.executed_steps,
            diagnostics
                .iter()
                .map(|diagnostic| diagnostic.executed_steps)
                .max()
                .unwrap()
                * diagnostics.len() as u64
        );
    }
    #[test]
    fn readback_accounting_tracks_dimensions_and_optional_no_medium_capture() {
        let output = (2, 3);
        let internal = (4, 5);
        assert_eq!(acceptance_readback_bytes(output, internal, false), 696);
        assert_eq!(acceptance_readback_bytes(output, internal, true), 720);
        assert_eq!(
            acceptance_readback_bytes(output, internal, false),
            4 * 6 + 4 * 8 * 6 + 3 * 8 * 20
        );
        assert_eq!(
            acceptance_readback_bytes(output, internal, true)
                - acceptance_readback_bytes(output, internal, false),
            4 * u64::from(output.0) * u64::from(output.1)
        );
    }
    #[test]
    fn canonical_f16_transport_rejects_positive_underflow_and_overflow() {
        assert_eq!(required_positive_f16(0.0, "density").unwrap(), 0);
        let minimum_subnormal = f32::from(half::f16::from_bits(1));
        assert_ne!(
            required_positive_f16(minimum_subnormal, "density").unwrap(),
            0
        );
        assert!(required_positive_f16(minimum_subnormal * 0.5, "density").is_err());
        assert!(required_positive_f16(f32::from(half::f16::MAX), "extinction").is_ok());
        let overflow_boundary = 65_520.0_f32;
        assert!(required_positive_f16(
            f32::from_bits(overflow_boundary.to_bits() - 1),
            "extinction"
        )
        .is_ok());
        assert!(required_positive_f16(overflow_boundary, "extinction").is_err());
    }
    #[test]
    fn blue_noise_integrity() {
        assert_eq!(
            format!("{:x}", Sha256::digest(BLUE_NOISE_ASSET.as_bytes())),
            "c4690ee9c66e9b2d1a608537cdc6c5384c93e69a42602093a6ce480606e780ea"
        );
        assert_eq!(parse_blue_noise_asset().unwrap().len(), 64);
    }
    #[test]
    fn execution_diagnostics_serialize_the_exact_public_schema() {
        let mut diagnostics = MediaExecutionDiagnostics {
            majorant_proof: Some(crate::media::MajorantProof::ExactConstant),
            majorant_valid: true,
            sample_count: 1,
            step_count: 2,
            temporal_history_decision: "accepted".into(),
            temporal_history_reason: "all temporal identities matched".into(),
            host_visible_bytes: 3,
            froxel_device_local_bytes: 4,
            density_device_local_bytes: 5,
            majorant_device_local_bytes: 6,
            staging_readback_bytes: 7,
            adapter: "adapter".into(),
            backend: "backend".into(),
            driver: "driver".into(),
            source_revision: "revision".into(),
            executed_multi_scatter: true,
            single_scatter_dispatches: 1,
            multiple_scatter_dispatches: 1,
            terrain_trace_queries: 2,
            sun_transmittance_method: "bounded_nested_midpoint".into(),
            sun_transmittance_bias: "fine_midpoint_with_coarse_fine_abs_rgb_error".into(),
            sun_transmittance_max_segment_length: Some(10.0),
            sun_transmittance_executed_steps: 12,
            sun_transmittance_max_abs_error: 1e-6,
            single_scatter_luminance: 0.25,
            multiple_scatter_luminance: 0.125,
            energy_accounting_residual: Some(5.960_464_477_539_063e-8),
        };
        add_termination_integration_steps(&mut diagnostics, &[0.0, 1.0, 3.0], 4).unwrap();
        assert_eq!(diagnostics.step_count, 9);
        assert!(add_termination_integration_steps(&mut diagnostics, &[f32::NAN], 4).is_err());
        assert!(add_termination_integration_steps(&mut diagnostics, &[4.0], 4).is_err());
        assert_eq!(diagnostics.step_count, 9);
        let value = serde_json::to_value(diagnostics).unwrap();
        let keys = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            keys,
            [
                "adapter",
                "backend",
                "density_device_local_bytes",
                "driver",
                "executed_multi_scatter",
                "froxel_device_local_bytes",
                "host_visible_bytes",
                "majorant_device_local_bytes",
                "majorant_proof",
                "majorant_valid",
                "sample_count",
                "single_scatter_dispatches",
                "single_scatter_luminance",
                "source_revision",
                "staging_readback_bytes",
                "step_count",
                "temporal_history_decision",
                "temporal_history_reason",
                "terrain_trace_queries",
                "sun_transmittance_method",
                "sun_transmittance_bias",
                "sun_transmittance_max_segment_length",
                "sun_transmittance_executed_steps",
                "sun_transmittance_max_abs_error",
                "multiple_scatter_dispatches",
                "multiple_scatter_luminance",
                "energy_accounting_residual",
            ]
            .into_iter()
            .collect()
        );
        assert_eq!(value["majorant_proof"], "ExactConstant");
        assert_eq!(value["majorant_valid"], true);
        assert_eq!(value["sun_transmittance_method"], "bounded_nested_midpoint");
        assert_eq!(
            value["sun_transmittance_bias"],
            "fine_midpoint_with_coarse_fine_abs_rgb_error"
        );
        assert_eq!(value["sun_transmittance_max_segment_length"], 10.0);
        assert_eq!(value["sun_transmittance_executed_steps"], 12);
        assert_eq!(value["sun_transmittance_max_abs_error"], 1e-6);
    }
    #[test]
    fn multiple_scatter_is_accounted() {
        let source = include_str!("../../shaders/nephele_froxel.wgsl");
        assert!(source.contains("cs_nephele_inject_single"));
        assert!(source.contains("cs_nephele_inject_multiple"));
        assert!(source.contains("gathered+=t*source"));
        assert!(!source.contains("let recycled="));
        assert!(!source.contains("incident-multiple"));
        let residual = beer_accounting_residual(&[0.2, 0.5, 1.0], 0.37);
        assert!(
            residual > 0.0,
            "the independently executed terms must not be replaced by zero"
        );
        assert!(residual <= f32::EPSILON);
        assert!(source.contains("abs(t.x-exp(-tau.x))"));
    }

    #[test]
    fn temporal_depth_accepts_one_depth32_code_and_rejects_real_mismatch() {
        let depth = 0.5_f32;
        let adjacent = f32::from_bits(depth.to_bits() + 1);
        assert!(depth32_history_matches(depth, adjacent));
        assert!(!depth32_history_matches(depth, 0.51));
        assert!(!depth32_history_matches(depth, f32::NAN));

        let shader = include_str!("../../shaders/nephele_froxel.wgsl");
        assert!(shader.contains("max(a,b)-min(a,b)<=1u"));
        assert!(shader.contains("depth32_history_matches(old_depth,projected_old_depth)"));
        assert!(!shader.contains("old_depth==projected_old_depth"));
    }

    #[test]
    fn live_viewer_paths_prepare_canonical_terrain_trace_before_encode() {
        let helper = include_str!("../../viewer/terrain/render/helpers.rs");
        let medium_prepare = helper
            .find("pass.prepare_viewer_frame(")
            .expect("viewer media helper must prepare the canonical medium");
        let terrain_prepare = helper
            .find("pass.prepare_viewer_terrain_trace(")
            .expect("viewer media helper must reconcile terrain resources");
        assert!(medium_prepare < terrain_prepare);
        for source in [
            include_str!("../../viewer/terrain/render/screen/setup.rs"),
            include_str!("../../viewer/terrain/render/offscreen/setup.rs"),
        ] {
            let prepare = source
                .find("self.prepare_canonical_media_frame(")
                .expect("viewer media path must prepare the canonical medium before drawing");
            let bind = source
                .find("self.prepare_pbr_bind_group_internal(&pbr_uniforms)")
                .expect("viewer media path must bind the canonical light transmittance");
            assert!(prepare < bind);
        }
        for source in [
            include_str!("../../viewer/terrain/render/screen/effects.rs"),
            include_str!("../../viewer/terrain/render/offscreen/effects.rs"),
        ] {
            assert!(source.contains("let output = pass.encode("));
        }
        let mean = viewer_height_palette_mean(&[0.0, 1.0], 0.0, 1.0);
        for (actual, expected) in mean.into_iter().zip([0.52, 0.65, 0.49]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn viewer_pbr_uses_rgb_same_medium_direct_light_and_keeps_media_hdr_linear() {
        let source = concat!(
            include_str!("../../shaders/includes/shadow_moments.wgsl"),
            "\n",
            include_str!("../../viewer/terrain/shader_pbr/terrain_pbr.wgsl")
        );
        naga::front::wgsl::parse_str(source).expect("viewer PBR WGSL must validate");
        assert!(source.contains("canonical_light_transmittance: texture_3d<f32>"));
        assert!(source.contains("diffuse * effective_shadow_tint * same_medium_direct_t"));
        assert!(source.contains("specular_color * same_medium_direct_t"));
        assert!(source.contains("water_spec * same_medium_direct_t"));
        assert!(source.contains("rim_light * same_medium_direct_t"));
        assert!(source.contains("fn terrain_view_depth(world_pos: vec3<f32>) -> f32"));
        assert_eq!(
            source
                .matches("let view_depth = terrain_view_depth(in.world_pos);")
                .count(),
            4
        );
        assert!(
            !source.contains("let view_depth = max(length(u.camera_pos.xyz - in.world_pos), 0.1);")
        );
        assert!(source.contains("fn preserve_overlay_lighting("));
        assert!(source.contains("return lit_linear * preserve_scale;"));
        assert!(!source.contains("preserve_overlay_scalar("));
        let linear_branch = source
            .find("if !linear_hdr_output {")
            .expect("media output must have an explicit display-transform boundary");
        let haze = source
            .find("let atmo_scale =")
            .expect("legacy display haze must remain available");
        assert!(linear_branch < haze);
    }

    #[test]
    fn terrain_media_surface_lookup_uses_froxel_xy_and_log_depth_coordinates() {
        let source = include_str!("../../shaders/terrain_pbr_pom.wgsl");
        assert!(source.contains("screen_position / vec2<f32>(8.0)"));
        assert!(source.contains("vec2<f32>(fog_uniforms.media_depth.w)"));
        assert!(source.contains("let unit_depth = clamp("));
        assert!(!source.contains("let xy = screen_position +"));
    }

    #[test]
    fn live_media_paths_fail_closed_when_pbr_initialization_fails() {
        let screen = include_str!("../../viewer/terrain/render/screen/resources.rs");
        let snapshot = include_str!("../../viewer/terrain/render/offscreen/setup.rs");
        for source in [screen, snapshot] {
            assert!(source.contains("if needs_canonical_media"));
            assert!(source.contains("return Err(e);"));
        }
        let screen_render = include_str!("../../viewer/terrain/render/screen/mod.rs");
        assert!(screen_render.contains("media_prepare_failed:"));
    }

    #[test]
    fn terrain_scene_binds_all_declared_media_graph_resources() {
        let draw = include_str!("draw/mod.rs");
        assert!(draw.contains("self.bind_realtime_media_graph_resources("));
        assert!(!draw.contains("nephele.media.integrate lost its authoritative depth transition"));
        let graph = include_str!("render_graph.rs");
        assert!(graph.contains(".write(m.radiance_provider)"));
        assert!(graph.contains(".read(m.radiance_provider)"));
        let aov = include_str!("aov.rs");
        let forward = aov
            .find("graph.execute_with_barriers(\"terrain.forward_aov\"")
            .expect("AOV forward graph pass must execute");
        let inject = aov[forward..]
            .find("graph.execute_with_barriers(\"nephele.media.inject\"")
            .map(|offset| forward + offset)
            .expect("AOV media inject graph pass must execute");
        assert!(aov[forward..inject].contains("self.prepare_realtime_media_radiance_provider("));
        let bindings = include_str!("media.rs");
        for handle in [
            "scene_depth",
            "radiance_provider",
            "extinction",
            "light_transmittance",
            "in_scatter",
            "integrated",
            "transmittance",
            "cloud_shadow",
            "optical_depth",
            "history_previous",
            "history_depth_previous",
            "history_current",
            "history_depth_current",
            "composite",
        ] {
            assert!(
                bindings.contains(&format!("handles.{handle}")),
                "production binding is missing declared media handle {handle}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn apple_metal_viewer_trace_matches_forced_pbr_geometry() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("viewer trace geometry regression requires Apple Metal");
        let adapter_info = adapter.get_info();
        assert_eq!(adapter_info.backend, wgpu::Backend::Metal);
        assert!(adapter_info.name.to_ascii_lowercase().contains("apple"));
        assert!(adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE));
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nephele.viewer-trace-geometry.device"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: adapter.limits(),
            },
            None,
        ))
        .expect("viewer trace geometry regression requires a Metal device");
        let dimensions = (5, 3);
        let origin = [137.0_f32, -211.0];
        let span = [-80.0_f32, 30.0];
        let height_min = 400.0_f32;
        let height_range = 250.0_f32;
        let z_scale = 1.75_f32;
        let heights = (0..dimensions.1)
            .flat_map(|z| {
                (0..dimensions.0).map(move |x| height_min + x as f32 * 40.0 + z as f32 * 45.0)
            })
            .collect::<Vec<_>>();
        let medium = crate::media::Medium::new(
            [0.1; 3],
            [0.1; 3],
            crate::media::Phase::Isotropic,
            crate::media::DensityField::Homogeneous(crate::media::Homogeneous {
                authored_density: 1.0,
                mapping: crate::media::DensityMapping {
                    physical_density_per_authored_unit: 1.0,
                },
            }),
        )
        .unwrap();
        let medium_identity = medium.identity(1);
        let mut viewer = ViewerMediaPass::new(&device, &queue, (1, 1), medium, 1).unwrap();
        viewer
            .prepare_viewer_frame(
                &queue,
                glam::Vec3::ZERO,
                glam::Mat4::IDENTITY,
                0.1,
                20.0,
                glam::Vec3::Y,
            )
            .unwrap();
        viewer
            .prepare_viewer_terrain_trace(
                &device,
                &queue,
                &adapter,
                dimensions,
                &heights,
                dimensions,
                origin,
                span,
                height_min,
                height_range,
                z_scale,
                7,
            )
            .unwrap();
        assert_eq!(viewer.resources.viewport, dimensions);
        assert_eq!(viewer.resources.medium_identity, Some(medium_identity));
        assert_eq!(viewer.prepared_medium_identity, Some(medium_identity));
        let trace = viewer.resources.terrain_trace.as_ref().unwrap();
        let samples = [(0_u32, 0_u32, 400.0_f32), (2, 1, 525.0), (4, 2, 650.0)];
        let placement = viewer_terrain_trace_placement(
            dimensions,
            origin,
            span,
            height_min,
            height_range,
            z_scale,
        );
        assert_eq!(placement.origin_xz, origin);
        assert_eq!(placement.spacing_xz, [-20.0, 15.0]);
        assert_eq!(placement.height_offset, height_min);
        assert_eq!(placement.height_scale, z_scale);
        let forced_pbr_y = |raw_height: f32| (raw_height - height_min) * z_scale;
        assert_eq!(525.0 - height_min, 125.0);
        assert_eq!(forced_pbr_y(525.0), 218.75);
        assert_ne!(forced_pbr_y(525.0), 0.07);
        let rays = samples.map(|(x, z, raw_height)| {
            let world = [
                placement.origin_xz[0] + x as f32 * placement.spacing_xz[0],
                forced_pbr_y(raw_height),
                placement.origin_xz[1] + z as f32 * placement.spacing_xz[1],
            ];
            let direction_x = match x {
                0 => 1.0 / 64.0,
                4 => -1.0 / 64.0,
                _ => 0.0,
            };
            let direction_z = match z {
                0 => -1.0 / 64.0,
                2 => 1.0 / 64.0,
                _ => 0.0,
            };
            GpuTerrainRay {
                origin_tmin: [
                    world[0] - direction_x * 1024.0,
                    world[1] + 1024.0,
                    world[2] - direction_z * 1024.0,
                    0.0,
                ],
                direction_tmax: [direction_x, -1.0, direction_z, 2048.0],
            }
        });
        let rays_buffer = crate::core::resource_tracker::tracked_create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("nephele.viewer-trace-geometry.rays"),
                contents: bytemuck::cast_slice(&rays),
                usage: wgpu::BufferUsages::STORAGE,
            },
        )
        .unwrap();
        let hit_bytes = std::mem::size_of_val(&rays) as u64;
        let hits_buffer = tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("nephele.viewer-trace-geometry.hits"),
                size: hit_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
        .unwrap();
        let readback = tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("nephele.viewer-trace-geometry.readback"),
                size: hit_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            },
        )
        .unwrap();
        let source = format!(
            "{}\n{}",
            crate::shader_sources::hybrid_kernel(),
            include_str!("../../shaders/nephele_terrain_trace_adapter.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nephele.viewer-trace-geometry.shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = crate::core::shader_registry::try_create_compute_pipeline_scoped(
            &device,
            &wgpu::ComputePipelineDescriptor {
                label: Some("nephele.viewer-trace-geometry.pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main_nephele_terrain_trace_adapter",
            },
        )
        .unwrap();
        let empty = |group| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nephele.viewer-trace-geometry.empty"),
                layout: &pipeline.get_bind_group_layout(group),
                entries: &[],
            })
        };
        let empty0 = empty(0);
        let empty1 = empty(1);
        let height_view = trace
            ._pyramid
            .height_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let minmax_view = trace
            ._pyramid
            .minmax_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let terrain_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.viewer-trace-geometry.terrain"),
            layout: &pipeline.get_bind_group_layout(2),
            entries: &[
                texture_entry(1, &minmax_view),
                texture_entry(2, &height_view),
                buffer_entry(3, &trace._terrain_uniform),
                buffer_entry(10, &trace._curvature_uniform),
            ],
        });
        let query_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.viewer-trace-geometry.query"),
            layout: &pipeline.get_bind_group_layout(3),
            entries: &[buffer_entry(8, &rays_buffer), buffer_entry(9, &hits_buffer)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nephele.viewer-trace-geometry.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nephele.viewer-trace-geometry.dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &empty0, &[]);
            pass.set_bind_group(1, &empty1, &[]);
            pass.set_bind_group(2, &terrain_group, &[]);
            pass.set_bind_group(3, &query_group, &[]);
            pass.dispatch_workgroups(rays.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&hits_buffer, 0, &readback, 0, hit_bytes);
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let hits = bytemuck::cast_slice::<u8, GpuTerrainHit>(&mapped);
        for (hit, (x, z, raw_height)) in hits.iter().zip(samples) {
            assert_eq!(
                hit.normal_hit[3], 1.0,
                "production terrain_trace missed texel ({x},{z}): {hit:?}"
            );
            assert_eq!(
                hit.point_t,
                [
                    origin[0] + x as f32 * -20.0,
                    forced_pbr_y(raw_height),
                    origin[1] + z as f32 * 15.0,
                    1024.0,
                ]
            );
        }
        drop(mapped);
        readback.unmap();
        assert!(viewer
            .prepare_viewer_frame(
                &queue,
                glam::Vec3::ZERO,
                glam::Mat4::IDENTITY,
                20.0,
                1.0,
                glam::Vec3::Y,
            )
            .is_err());
        assert!(viewer.prepared_medium_identity.is_none());
        assert!(viewer.prepared_frame.is_none());
    }

    #[test]
    fn ibl_intensity_and_lambert_fallback_preserve_uniform_environment_units() {
        assert_eq!(scaled_ibl_radiance(2.0, 0.25), 0.5);
        let radiance = [0.25, 0.5, 1.0];
        let irradiance = diffuse_ibl_irradiance(radiance);
        for channel in 0..3 {
            let lambert_exitant = 0.6 * irradiance[channel] / std::f32::consts::PI;
            assert!((lambert_exitant - 0.6 * radiance[channel]).abs() < 1e-6);
        }
    }

    #[test]
    fn colored_multiple_scatter_diagnostic_matches_analytic_beer_integral() {
        let extinction = [0.2_f32, 0.5, 1.0];
        let total_source = [2.0_f32, 3.0, 5.0];
        let single_source = [0.5_f32, 1.0, 2.0];
        let length = 2.0_f32;
        let mut integrated_multiple = [0.0_f32; 3];
        for channel in 0..3 {
            let multiple = total_source[channel] - single_source[channel];
            integrated_multiple[channel] =
                multiple * (1.0 - (-extinction[channel] * length).exp()) / extinction[channel];
        }
        let expected_luminance = integrated_multiple[0] * 0.2126
            + integrated_multiple[1] * 0.7152
            + integrated_multiple[2] * 0.0722;
        assert!((expected_luminance - 2.521_331_5).abs() < 1e-6);
    }

    #[test]
    fn reviewer_task_c_contracts_are_explicit() {
        let rust = include_str!("media.rs")
            .split("#[cfg(test)]\nmod tests")
            .next()
            .unwrap();
        let shader = include_str!("../../shaders/nephele_froxel.wgsl");
        let py_api = include_str!("py_api.rs");
        let aov = include_str!("aov.rs");

        assert!(rust.contains("environment_intensity: f32"));
        assert!(rust.contains("scaled_ibl_radiance(*value, environment_intensity)"));
        assert!(aov.contains("env_maps.intensity.max(0.0)"));
        assert!(rust.contains("resources.diffuse_ibl = diffuse_ibl_irradiance"));
        assert!(rust.contains("u64::from(terrain_occlusion_enabled)"));
        assert!(rust.contains("texture_entry(3, &terrain_shadow_maps)"));
        assert!(rust.contains("buffer_entry(4, &csm.uniform_buffer)"));
        assert!(rust.contains("dimension: Some(wgpu::TextureViewDimension::D2Array)"));
        assert!(rust.contains("texture_entry(9, &single_scatter)"));
        assert!(shader.contains("fn manual_csm_visibility"));
        assert!(shader.contains("*manual_csm_visibility(hit.point_t.xyz,n)"));
        assert!(shader.contains("let multiple_rgb=max(source-single_source"));
        assert!(shader.contains("multiple_scatter_rgb+=segment_weight*multiple_rgb"));
        assert!(py_api.contains("completed_ledger_report"));
    }
    #[test]
    fn shaders_validate_and_compute_call_graphs_use_manual_shadows() {
        assert_compute_call_graph_safe(include_str!("../../shaders/nephele_froxel.wgsl"));
        let trace_adapter = include_str!("../../shaders/nephele_realtime_terrain_trace.wgsl");
        assert_eq!(trace_adapter.matches(" = terrain_trace(").count(), 2);
        let assembled = format!(
            "{}\n{trace_adapter}",
            crate::shader_sources::hybrid_kernel()
        );
        assert_compute_call_graph_safe(&assembled);
        let froxel = include_str!("../../shaders/nephele_froxel.wgsl");
        assert!(froxel.contains("phase_value(dot(incident,normalize(media.sun.xyz)))"));
        assert!(!froxel.contains("phase_value(dot(normalize(media.sun.xyz),view_direction))"));
        let tonemap = format!(
            "{}\n{}\n{}",
            include_str!("../../shaders/includes/determinism.wgsl"),
            include_str!("../../shaders/includes/tonemap_common.wgsl"),
            include_str!("../../shaders/postprocess_tonemap.wgsl"),
        );
        assert_compute_call_graph_safe(&tonemap);
        let legacy = include_str!("../../shaders/volumetric.wgsl");
        assert!(!legacy.contains("For now, disable shadows"));
        assert_compute_call_graph_safe(legacy);
    }
    #[test]
    fn assembled_terrain_shader_consumes_same_medium_transmittance() {
        let source = crate::shader_sources::terrain();
        assert!(source.contains("nephele_same_medium_direct_transmittance"));
        let module = naga::front::wgsl::parse_str(&source).expect("terrain WGSL must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("terrain WGSL must validate");
    }

    #[test]
    fn yup_media_uses_the_reference_sun_axes() {
        let azimuth = 315.0_f32.to_radians();
        let elevation = 18.0_f32.to_radians();
        let legacy_z_up = [
            azimuth.cos() * elevation.cos(),
            azimuth.sin() * elevation.cos(),
            elevation.sin(),
        ];
        let expected_y_up = glam::Vec3::new(legacy_z_up[0], legacy_z_up[2], legacy_z_up[1]);
        assert!(terrain_light_direction("mesh:yup", legacy_z_up).abs_diff_eq(expected_y_up, 1e-6));
        assert!(terrain_light_direction("mesh", legacy_z_up)
            .abs_diff_eq(glam::Vec3::from_array(legacy_z_up), 1e-6));
    }
}
