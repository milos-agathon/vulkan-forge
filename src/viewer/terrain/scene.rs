// src/viewer/terrain/scene.rs
// Terrain scene management for the interactive viewer

use super::denoise::DenoisePass;
use super::render::TerrainUniforms;
use super::shader::TERRAIN_SHADER;
use super::vector_overlay::{VectorOverlayLayer, VectorOverlayStack};
use crate::core::resource_tracker::{TrackedBuffer, TrackedTexture};
use crate::shadows::{CsmConfig, CsmRenderer};
use anyhow::Result;
use glam::{DVec2, DVec3, Vec3};
use std::sync::Arc;

/// P0.1/M1: WBOIT compose shader for final compositing of accumulation buffers
const WBOIT_COMPOSE_SHADER: &str = r#"
// WBOIT Compose Shader - composites accumulation buffers to final output

@group(0) @binding(0)
var color_accumulation: texture_2d<f32>;

@group(0) @binding(1)
var reveal_accumulation: texture_2d<f32>;

@group(0) @binding(2)
var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex_index & 2u) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, -y) * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let accum_color = textureSample(color_accumulation, tex_sampler, in.uv);
    let reveal = textureSample(reveal_accumulation, tex_sampler, in.uv).r;
    
    let epsilon = 1e-5;
    var final_color: vec3<f32>;
    if (accum_color.a > epsilon) {
        final_color = accum_color.rgb / accum_color.a;
    } else {
        final_color = vec3<f32>(0.0);
    }
    
    let final_alpha = 1.0 - reveal;
    return vec4<f32>(final_color, final_alpha);
}
"#;

/// Stored terrain data for interactive viewer rendering
pub struct ViewerTerrainData {
    pub heightmap: Vec<f32>,
    pub dimensions: (u32, u32),
    pub domain: (f32, f32),
    pub revision: u64,
    pub raster_info: crate::gis::types::RasterInfo,
    /// Absolute viewer-world X/Z origin: `(map X, -map Y)`.
    pub world_origin_xz: DVec2,
    /// Positive physical viewer-world X/Z span.
    pub world_span_xz: DVec2,
    pub georeferencing_fallback: bool,
    pub _heightmap_texture: TrackedTexture,
    pub heightmap_view: wgpu::TextureView,
    pub vertex_buffer: TrackedBuffer,
    pub index_buffer: TrackedBuffer,
    pub index_count: u32,
    pub uniform_buffer: TrackedBuffer,
    pub bind_group: wgpu::BindGroup,
    // Camera
    pub cam_radius: f32,
    pub cam_phi_deg: f32,
    pub cam_theta_deg: f32,
    pub cam_fov_deg: f32,
    pub cam_target: DVec3,
    // Sun/lighting
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    pub sun_intensity: f32,
    pub sun_source: String,
    pub ambient: f32,
    // Terrain rendering
    pub z_scale: f32,
    pub shadow_intensity: f32,
    pub background_color: [f32; 3],
    pub water_level: f32,
    pub water_color: [f32; 3],
}

impl ViewerTerrainData {
    pub fn terrain_width(&self) -> f32 {
        self.dimensions.0 as f32
    }

    pub fn height_range(&self) -> f32 {
        self.domain.1 - self.domain.0
    }

    pub fn default_camera_target(&self) -> DVec3 {
        DVec3::new(
            self.world_origin_xz.x + self.world_span_xz.x * 0.5,
            f64::from(self.height_range() * self.z_scale * 0.5),
            self.world_origin_xz.y + self.world_span_xz.y * 0.5,
        )
    }

    pub fn camera_target(&self) -> DVec3 {
        self.cam_target
    }

    pub fn camera_eye(&self) -> DVec3 {
        Self::camera_eye_for(
            self.cam_target,
            self.cam_phi_deg,
            self.cam_theta_deg,
            self.cam_radius,
        )
    }

    pub fn camera_eye_for(target: DVec3, phi_deg: f32, theta_deg: f32, radius: f32) -> DVec3 {
        let phi = phi_deg.to_radians();
        let theta = theta_deg.to_radians();
        target
            + DVec3::new(
                f64::from(radius * theta.sin() * phi.cos()),
                f64::from(radius * theta.cos()),
                f64::from(radius * theta.sin() * phi.sin()),
            )
    }

    pub fn validate_camera_state(
        &self,
        anchor: &crate::camera::Anchor,
        phi_deg: f32,
        theta_deg: f32,
        radius: f32,
        fov_deg: f32,
        target: DVec3,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        if ![phi_deg, theta_deg, radius, fov_deg]
            .into_iter()
            .all(f32::is_finite)
        {
            return Err(
                crate::viewer::camera_controller::CameraFrameError::NonFinite {
                    role: crate::viewer::camera_controller::CoordRole::Target,
                },
            );
        }
        crate::viewer::camera_controller::validate_camera_pose(
            anchor,
            DVec3::new(target.x, 0.0, target.z),
            Self::camera_eye_for(target, phi_deg, theta_deg, radius),
            target,
        )?;
        Ok(())
    }

    pub fn camera_up(&self) -> Vec3 {
        let eye = self.camera_eye();
        let target = self.camera_target();
        let view_dir = (target - eye).normalize_or_zero();
        if view_dir.dot(DVec3::Y).abs() > 0.999 {
            -Vec3::Z
        } else {
            Vec3::Y
        }
    }

    /// Set camera state from animation keyframe values.
    pub fn set_camera_state(
        &mut self,
        phi_deg: f32,
        theta_deg: f32,
        radius: f32,
        fov_deg: f32,
        target: Option<[f64; 3]>,
    ) {
        self.cam_phi_deg = phi_deg;
        self.cam_theta_deg = theta_deg;
        self.cam_radius = radius;
        self.cam_fov_deg = fov_deg;
        if let Some(target) = target {
            self.cam_target = DVec3::from_array(target);
        }
    }
}

impl ViewerTerrainScene {
    /// Refresh anchor-derived CPU caches. Terrain vertices remain local; the
    /// current anchor is consumed when per-frame uniforms are prepared.
    pub(crate) fn refresh_anchor_caches(&mut self, _anchor: &crate::camera::Anchor) {
        self.repack_vector_overlays(_anchor);
    }

    /// Reset terrain temporal state without replacing GPU resources.
    pub(crate) fn invalidate_temporal_history(&mut self) {
        if let Some(taa) = self.taa_renderer.as_mut() {
            taa.reset_history();
        }
        self.taa_jitter = if self.taa_jitter.enabled {
            crate::core::jitter::JitterState::enabled()
        } else {
            crate::core::jitter::JitterState::new()
        };
    }

    pub(crate) fn is_taa_enabled(&self) -> bool {
        self.taa_renderer
            .as_ref()
            .is_some_and(crate::core::taa::TaaRenderer::is_enabled)
    }

    pub(crate) fn taa_history_valid(&self) -> bool {
        self.taa_renderer
            .as_ref()
            .is_some_and(crate::core::taa::TaaRenderer::history_valid)
    }

    pub(crate) fn taa_history_allocation_ids(&self) -> [u64; 2] {
        self.taa_renderer.as_ref().map_or(
            [0, 0],
            crate::core::taa::TaaRenderer::history_allocation_ids,
        )
    }

    pub(crate) fn screen_depth_view(&self) -> Option<&wgpu::TextureView> {
        self.depth_view.as_ref()
    }

    pub(crate) fn snapshot_depth_view(&self) -> Option<wgpu::TextureView> {
        self.snapshot_depth_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
    }
}

/// Simple terrain scene for interactive viewer
pub struct ViewerTerrainScene {
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<wgpu::Queue>,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) depth_texture: Option<TrackedTexture>,
    pub(super) depth_view: Option<wgpu::TextureView>,
    pub(super) depth_size: (u32, u32),
    /// Depth from the latest single-sample offscreen render, retained so the
    /// viewer can fill only uncovered pixels with the computed sky.
    pub(super) snapshot_depth_texture: Option<TrackedTexture>,
    pub terrain: Option<ViewerTerrainData>,
    /// PBR+POM rendering configuration (opt-in, default off)
    pub pbr_config: super::pbr_renderer::ViewerTerrainPbrConfig,
    /// PBR pipeline (created lazily when PBR mode enabled)
    pub pbr_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) pbr_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) pbr_uniform_buffer: Option<TrackedBuffer>,
    pub(super) pbr_bind_group: Option<wgpu::BindGroup>,
    pub(super) terrain_ibl_renderer: Option<crate::core::ibl::IBLRenderer>,
    pub(super) terrain_ibl_hdr_path: Option<std::path::PathBuf>,
    pub(super) terrain_ibl_specular_view: Option<wgpu::TextureView>,
    pub(super) terrain_ibl_irradiance_view: Option<wgpu::TextureView>,
    pub(super) terrain_ibl_brdf_view: Option<wgpu::TextureView>,
    pub(super) terrain_ibl_sampler: Option<wgpu::Sampler>,
    pub(super) terrain_ibl_specular_mip_count: u32,
    pub(super) terrain_ibl_fallback_cube: Option<TrackedTexture>,
    pub(super) terrain_ibl_fallback_cube_view: Option<wgpu::TextureView>,
    pub(super) terrain_ibl_fallback_brdf: Option<TrackedTexture>,
    pub(super) terrain_ibl_fallback_brdf_view: Option<wgpu::TextureView>,
    // Heightfield compute pipelines for AO and sun visibility
    pub(super) height_ao_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) height_ao_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) height_ao_texture: Option<TrackedTexture>,
    pub(super) height_ao_view: Option<wgpu::TextureView>,
    pub(super) height_ao_uniform_buffer: Option<TrackedBuffer>,
    pub(super) sun_vis_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) sun_vis_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) sun_vis_texture: Option<TrackedTexture>,
    pub(super) sun_vis_view: Option<wgpu::TextureView>,
    pub(super) sun_vis_uniform_buffer: Option<TrackedBuffer>,
    pub(super) sampler_nearest: Option<wgpu::Sampler>,
    // Fallback 1x1 white texture for when AO/sun_vis are disabled
    pub(super) fallback_texture: Option<TrackedTexture>,
    pub(super) fallback_texture_view: Option<wgpu::TextureView>,
    // Post-process pass for lens effects (distortion, CA, vignette)
    pub(super) post_process: Option<super::post_process::PostProcessPass>,
    // Depth of Field pass
    pub(super) dof_pass: Option<super::dof::DofPass>,
    // Motion blur accumulator
    pub(super) motion_blur_pass: Option<super::motion_blur::MotionBlurAccumulator>,
    pub(super) motion_blur_depth_pass: Option<super::motion_blur_depth::MotionBlurDepthAccumulator>,
    // P5: Volumetrics pass
    pub(super) volumetrics_pass: Option<super::volumetrics::VolumetricsPass>,
    // M5: Denoise pass
    pub(super) denoise_pass: Option<DenoisePass>,
    pub(super) surface_format: wgpu::TextureFormat,
    // Overlay layer stack for lit draped overlays
    pub overlay_stack: Option<super::overlay::OverlayStack>,
    // Option B: Vector overlay geometry stack
    pub vector_overlay_stack: Option<VectorOverlayStack>,
    // P0.1/M1: OIT mode for transparent overlay rendering
    pub oit_enabled: bool,
    pub oit_mode: String,
    // P0.1/M1: WBOIT accumulation resources
    pub(super) wboit_color_texture: Option<TrackedTexture>,
    pub(super) wboit_color_view: Option<wgpu::TextureView>,
    pub(super) wboit_reveal_texture: Option<TrackedTexture>,
    pub(super) wboit_reveal_view: Option<wgpu::TextureView>,
    pub(super) wboit_compose_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) wboit_compose_bind_group: Option<wgpu::BindGroup>,
    pub(super) wboit_compose_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) wboit_sampler: Option<wgpu::Sampler>,
    pub(super) wboit_size: (u32, u32),
    // P6.2: Shadow mapping support
    pub(super) csm_renderer: Option<crate::shadows::CsmRenderer>,
    pub(super) moment_pass: Option<crate::shadows::MomentGenerationPass>,
    pub(super) moment_blur_pass: Option<crate::shadows::ShadowBlurPass>,
    pub(super) csm_uniform_buffer: Option<TrackedBuffer>,

    // Shadow rendering resources
    pub(super) shadow_pipeline: Option<wgpu::RenderPipeline>,
    pub(super) shadow_uniform_buffers: Vec<TrackedBuffer>, // One per cascade
    pub(super) shadow_bind_groups: Vec<wgpu::BindGroup>,   // One per cascade
    /// Terrain revision captured by every live shadow bind group.
    pub(crate) shadow_bind_group_revision: u64,

    // P1.4: TAA support for terrain viewer
    pub(super) taa_renderer: Option<crate::core::taa::TaaRenderer>,
    pub(super) taa_jitter: crate::core::jitter::JitterState,
    pub(super) taa_velocity_texture: Option<TrackedTexture>,
    pub(super) taa_velocity_view: Option<wgpu::TextureView>,
    pub(super) taa_velocity_size: (u32, u32),
    pub(super) terrain_revision_counter: u64,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_renderer: crate::render::mesh_instanced::MeshInstancedRenderer,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_hlod_instance_buffer: TrackedBuffer,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_batches: Vec<crate::terrain::scatter::TerrainScatterBatch>,
    #[cfg(feature = "enable-gpu-instancing")]
    pub(super) scatter_last_frame_stats: crate::terrain::scatter::TerrainScatterFrameStats,
    /// Accumulated wall-clock time (seconds) for scatter wind animation.
    pub(super) scatter_elapsed_time: f32,
}

mod core;
mod overlays;
mod pbr_compute;
mod pipeline_init;
#[cfg(feature = "enable-gpu-instancing")]
pub(in crate::viewer::terrain) mod scatter;
mod terrain_load;
