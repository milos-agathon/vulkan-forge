// src/terrain_renderer.rs
//! TerrainRenderer - GPU pipeline for PBR+POM terrain rendering
#![allow(deprecated)] // PyO3 GilRefs API deprecation - to be migrated to Bound API
//!
//! Implements a minimal-but-correct terrain rendering pipeline:
//! - Heightmap upload (numpy -> R32Float texture)
//! - Fullscreen triangle with triplanar PBR shading
//! - Parallax Occlusion Mapping (POM) support
//! - IBL (Image-Based Lighting) integration
//! - Colormap overlay
//!
//! Memory budget: <= 512 MiB host-visible allocations
//!
//! RELEVANT FILES: src/session.rs, src/material_set.rs, src/ibl_wrapper.rs,
//! src/terrain_render_params.rs, src/shaders/terrain_pbr_pom.wgsl

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use log::info;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use wgpu::TextureFormatFeatureFlags;

use super::render_params::{AddressModeNative, FilterModeNative};
use crate::lighting::types::{Light, LightType};
use crate::lighting::LightBuffer;

mod anamnesis;
mod aov;
mod atmosphere;
mod bind_groups;
mod constructor;
mod core;
mod draw;
mod geometry;
mod height_ao;
mod msaa;
mod offline;
mod pipeline_cache;
mod probes;
mod py_api;
mod render_graph;
mod resources;
mod runtime_contract;
#[cfg(feature = "enable-gpu-instancing")]
mod scatter;
mod shadows;
mod streaming;
mod uniforms;
mod upload;
mod viewer;
pub(crate) mod virtual_texture;
pub(crate) mod visibility_buffer;
mod water_reflection;

pub use self::core::{TerrainRenderer, TerrainScene, ViewerTerrainData};

use self::atmosphere::create_atmosphere_init_resources;
use self::bind_groups::create_base_bind_group_layouts;
use self::core::{
    clipmap_camera_config, is_clipmap_camera_mode, is_mesh_camera_mode, is_zup_camera_mode,
    ts_begin, ts_end, IblUniforms, NoopShadow, OverlayBinding, PipelineCache,
    MATERIAL_LAYER_CAPACITY, TERRAIN_DEFAULT_CASCADE_SPLITS,
};
use self::geometry::TerrainGeometryProvider;
use self::height_ao::create_heightfield_init_resources;
use self::msaa::{assert_msaa_invariants, log_msaa_debug, select_effective_msaa, MsaaInvariants};
use self::resources::create_base_init_resources;
use self::uniforms::{
    FogUniforms, HeightAoUniforms, MaterialLayerUniforms, OverlayUniforms, ShadowPassUniforms,
    SunVisUniforms,
};
use self::water_reflection::create_water_reflection_init_resources;
