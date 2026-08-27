// src/path_tracing/hybrid_compute.rs
// Hybrid path tracer combining SDF raymarching with mesh BVH traversal
// Extends the existing path tracing compute pipeline with hybrid scene support

use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::core::error::RenderError;
use crate::core::gpu::{align_copy_bpr, try_ctx};
use crate::core::resource_tracker::{
    tracked_create_buffer, tracked_create_buffer_init, tracked_create_texture,
};
use crate::path_tracing::aov::{AovFrames, AovKind};
use crate::path_tracing::compute::{Sphere, Uniforms};
use crate::sdf::HybridScene;

mod aether_post;
mod aether_reference;
mod layouts;
mod media_reference;
mod render;
mod render_terrain;
mod setup;
pub mod terrain_heightfield;

pub use aether_reference::{AetherSpectralReferenceDesc, AetherSpectralReferenceOutput};
pub use media_reference::TerrainMediaReferenceOutput;
pub use render_terrain::{TerrainReferenceDesc, TerrainReferenceOutput};
pub use terrain_heightfield::TerrainPtScene;

/// Additional uniforms for hybrid traversal
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct HybridUniforms {
    pub sdf_primitive_count: u32,
    pub sdf_node_count: u32,
    pub mesh_vertex_count: u32,
    pub mesh_index_count: u32,
    pub mesh_bvh_node_count: u32,
    pub traversal_mode: u32,
    pub _pad: [u32; 2],
}

/// Lighting uniforms for configurable lighting models
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct LightingUniforms {
    pub light_dir: [f32; 3],
    pub lighting_type: u32,
    pub light_color: [f32; 3],
    pub shadows_enabled: u32,
    pub ambient_color: [f32; 3],
    pub shadow_intensity: f32,
    pub hdri_intensity: f32,
    pub hdri_rotation: f32,
    pub specular_power: f32,
    pub _pad: [u32; 5],
}

/// Traversal mode for hybrid rendering
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TraversalMode {
    Hybrid = 0,
    SdfOnly = 1,
    MeshOnly = 2,
    /// Terrain heightfield as the primary intersectable (PROMETHEUS).
    TerrainOnly = 3,
}

impl Default for TraversalMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Hybrid path tracer parameters
#[derive(Clone, Debug)]
pub struct HybridTracerParams {
    pub base_uniforms: Uniforms,
    pub lighting_uniforms: LightingUniforms,
    pub traversal_mode: TraversalMode,
    pub early_exit_distance: f32,
    pub shadow_softness: f32,
}

impl Default for HybridTracerParams {
    fn default() -> Self {
        let azimuth = 315.0_f32.to_radians();
        let elevation = 45.0_f32.to_radians();
        let light_dir = [
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
            azimuth.sin() * elevation.cos(),
        ];

        Self {
            base_uniforms: Uniforms {
                width: 512,
                height: 512,
                frame_index: 0,
                aov_flags: 0,
                cam_origin: [0.0, 0.0, 0.0],
                cam_fov_y: std::f32::consts::PI / 4.0,
                cam_right: [1.0, 0.0, 0.0],
                cam_aspect: 1.0,
                cam_up: [0.0, 1.0, 0.0],
                cam_exposure: 1.0,
                cam_forward: [0.0, 0.0, -1.0],
                seed_hi: 12345,
                seed_lo: 67890,
                _pad_end: [0, 0, 0],
            },
            lighting_uniforms: LightingUniforms {
                light_dir,
                lighting_type: 1,
                light_color: [1.0, 0.95, 0.8],
                shadows_enabled: 1,
                ambient_color: [0.1, 0.12, 0.15],
                shadow_intensity: 0.6,
                hdri_intensity: 0.0,
                hdri_rotation: 0.0,
                specular_power: 32.0,
                _pad: [0, 0, 0, 0, 0],
            },
            traversal_mode: TraversalMode::Hybrid,
            early_exit_distance: 0.01,
            shadow_softness: 4.0,
        }
    }
}

/// Hybrid path tracer implementation
pub struct HybridPathTracer {
    layouts: HybridBindGroupLayouts,
    pipeline: wgpu::ComputePipeline,
    /// Accumulating terrain-reference entry (`main_terrain`).
    pipeline_terrain: wgpu::ComputePipeline,
    /// Acceptance-only stochastic spectral atmosphere reference.
    pipeline_aether_reference: wgpu::ComputePipeline,
    /// One-shot ReSTIR G-buffer entry (`main_terrain_gbuffer`).
    pipeline_terrain_gbuffer: wgpu::ComputePipeline,
    /// Canonical ReSTIR reuse passes (pt_restir_temporal/spatial.wgsl)
    /// dispatched between terrain accumulation frames.
    pipeline_restir_temporal: wgpu::ComputePipeline,
    pipeline_restir_spatial: wgpu::ComputePipeline,
}

struct HybridBindGroupLayouts {
    uniforms: wgpu::BindGroupLayout,
    scene: wgpu::BindGroupLayout,
    accum: wgpu::BindGroupLayout,
    output: wgpu::BindGroupLayout,
    terrain_gbuffer: wgpu::BindGroupLayout,
    restir_temporal: wgpu::BindGroupLayout,
    restir_spatial_scene: wgpu::BindGroupLayout,
    restir_spatial_reuse: wgpu::BindGroupLayout,
    empty: wgpu::BindGroupLayout,
}
