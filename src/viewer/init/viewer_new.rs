// src/viewer/init/viewer_new.rs
// Orchestrates Viewer::new() using init/ factory functions
// Extracted from mod.rs as part of the viewer refactoring

use std::collections::VecDeque;
use std::sync::Arc;

use winit::window::Window;

use crate::core::error::RenderResult;
use crate::core::gpu_timing::{
    create_default_config as create_gpu_timing_config, GpuTimingManager,
};
use crate::core::resource_tracker::{tracked_create_buffer, TrackedBuffer};
use crate::core::shadows::{CsmConfig, CsmShadowMap};
use crate::render::params::SsrParams;

use super::super::camera_controller::CameraController;
use super::super::viewer_config::{FpsCounter, ViewerConfig};
use super::super::viewer_enums::{FogMode, VizMode};
use super::super::viewer_types::SkyUniforms;
use super::super::Viewer;
use super::{
    create_device_and_surface, create_fallback_pipeline, create_fog_resources,
    create_gbuffer_resources, create_gi_baseline_resources, create_lit_resources,
    create_sky_resources,
};
use crate::cli::args::GiVizMode;
use crate::picking::UnifiedPickingSystem;

impl Viewer {
    /// Create a new Viewer instance using the init/ factory functions
    pub async fn new(
        window: Arc<Window>,
        config: ViewerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Device and surface initialization
        let dev_res = create_device_and_surface(Arc::clone(&window), config.vsync).await?;
        let surface = dev_res.surface;
        let device = dev_res.device;
        let queue = dev_res.queue;
        #[cfg(feature = "extension-module")]
        let adapter = dev_res.adapter;
        #[cfg(not(feature = "extension-module"))]
        let _adapter = dev_res.adapter;
        let surface_config = dev_res.config;
        let adapter_info = dev_res.adapter_info;
        let adapter_name = adapter_info.name.clone();
        let adapter_vendor = adapter_info.vendor;
        let adapter_device = adapter_info.device;
        let adapter_backend = format!("{:?}", adapter_info.backend);
        let adapter_device_type = format!("{:?}", adapter_info.device_type);
        let adapter_driver = adapter_info.driver.clone();
        let adapter_driver_info = adapter_info.driver_info.clone();

        let width = surface_config.width;
        let height = surface_config.height;

        // Optional GPU timing manager for GI profiling
        let gi_timing = match GpuTimingManager::new(
            device.clone(),
            queue.clone(),
            create_gpu_timing_config(&device),
        ) {
            Ok(mgr) if mgr.is_supported() => Some(mgr),
            Ok(_) => None,
            Err(e) => {
                eprintln!("[viewer] GPU timing manager unavailable: {e}");
                None
            }
        };

        // Initialize P5 Screen-space effects manager (optional)
        let gi = match crate::core::screen_space_effects::ScreenSpaceEffectsManager::new(
            &device, width, height,
        ) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("Failed to create ScreenSpaceEffectsManager: {}", e);
                None
            }
        };

        // GBuffer resources (depends on GI manager)
        let gbuf =
            create_gbuffer_resources(&device, gi.as_ref(), width, height, surface_config.format)?;

        // CSM depth pipeline resources
        let (csm_depth_pipeline, csm_depth_camera) = if gi.is_some() {
            create_csm_depth_resources(&device)?
        } else {
            (None, None)
        };

        // Lit resources
        let lit = create_lit_resources(&device, width, height)?;

        // GI baseline resources
        let gi_base = create_gi_baseline_resources(&device, width, height)?;

        // Sky resources
        let sky = create_sky_resources(&device, &queue, width, height, surface_config.format)?;

        // Fog resources
        let fog = create_fog_resources(&device, width, height)?;

        // Fallback pipeline
        let fb_pipeline = create_fallback_pipeline(&device, surface_config.format);

        // HUD overlay renderer
        let mut hud =
            crate::core::text_overlay::TextOverlayRenderer::new(&device, surface_config.format)?;
        hud.set_enabled(true);
        hud.set_resolution(width, height);

        // Configure CSM
        let mut csm_config = CsmConfig::default();
        csm_config.camera_near = config.znear;
        csm_config.camera_far = config.zfar;
        let csm = Some(CsmShadowMap::new(device.as_ref(), csm_config.clone())?);
        let csm_depth_bind_group = match (&csm_depth_pipeline, &csm_depth_camera) {
            (Some(pipeline), Some(camera)) => {
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewer.csm.depth.bg"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera.as_entire_binding(),
                    }],
                }))
            }
            _ => None,
        };

        // Read sky params from environment
        let sky_params_init = read_sky_env_params();

        // Write initial sky params
        queue.write_buffer(&sky.sky_params, 0, bytemuck::bytes_of(&sky_params_init));

        let mut viewer = Self {
            window,
            surface,
            device: device.clone(),
            queue: queue.clone(),
            #[cfg(feature = "extension-module")]
            adapter,
            config: surface_config,
            camera: CameraController::new(),
            camera_anchor: crate::camera::Anchor::new(),
            frame_camera: None,
            camera_rebase_count: 0,
            history_invalidation_count: 0,
            last_vector_source_delta: [0.0; 3],
            last_vector_packed_delta: [0.0; 3],
            view_config: config,
            frame_count: 0,
            applied_command_revision: 0,
            rendered_frame_revision: 0,
            fps_counter: FpsCounter::new(),
            #[cfg(feature = "extension-module")]
            terrain_scene: None,
            terrain_viewer: None,
            keys_pressed: std::collections::HashSet::new(),
            shift_pressed: false,
            gi,
            gi_pass: None,
            ssr_params: SsrParams::default(),
            gi_seed: None,
            gi_timing,
            gi_gpu_hzb_ms: 0.0,
            gi_gpu_ssao_ms: 0.0,
            gi_gpu_ssgi_ms: 0.0,
            gi_gpu_ssr_ms: 0.0,
            gi_gpu_composite_ms: 0.0,
            snapshot_request: None,
            pending_snapshot_tex: None,
            pending_captures: VecDeque::new(),
            // GBuffer resources
            geom_bind_group_layout: gbuf.geom_bind_group_layout,
            geom_pipeline: gbuf.geom_pipeline,
            object_overlay_pipeline: gbuf.object_overlay_pipeline,
            geom_camera_buffer: gbuf.geom_camera_buffer,
            geom_bind_group: gbuf.geom_bind_group,
            geom_vb: gbuf.geom_vb,
            geom_ib: None,
            geom_index_count: 36,
            object_source_positions: Vec::new(),
            object_source_indices: Vec::new(),
            z_texture: gbuf.z_texture,
            z_view: gbuf.z_view,
            albedo_texture: gbuf.albedo_texture,
            albedo_view: gbuf.albedo_view,
            albedo_sampler: gbuf.albedo_sampler,
            ssr_env_texture: None,
            comp_bind_group_layout: gbuf.comp_bind_group_layout,
            comp_pipeline: gbuf.comp_pipeline,
            comp_uniform: None,
            comp_bind_group_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            // Lit resources
            lit_bind_group_layout: lit.lit_bind_group_layout,
            lit_pipeline: lit.lit_pipeline,
            lit_uniform: lit.lit_uniform,
            lit_output: lit.lit_output,
            lit_output_view: lit.lit_output_view,
            lit_bind_group_cache: std::cell::RefCell::new(None),
            // GI baseline resources
            gi_baseline_hdr: gi_base.gi_baseline_hdr,
            gi_baseline_hdr_view: gi_base.gi_baseline_hdr_view,
            gi_baseline_diffuse_hdr: gi_base.gi_baseline_diffuse_hdr,
            gi_baseline_diffuse_hdr_view: gi_base.gi_baseline_diffuse_hdr_view,
            gi_baseline_spec_hdr: gi_base.gi_baseline_spec_hdr,
            gi_baseline_spec_hdr_view: gi_base.gi_baseline_spec_hdr_view,
            gi_output_hdr: gi_base.gi_output_hdr,
            gi_output_hdr_view: gi_base.gi_output_hdr_view,
            gi_debug: gi_base.gi_debug,
            gi_debug_view: gi_base.gi_debug_view,
            gi_baseline_bgl: gi_base.gi_baseline_bgl,
            gi_baseline_pipeline: gi_base.gi_baseline_pipeline,
            gi_split_bgl: gi_base.gi_split_bgl,
            gi_split_pipeline: gi_base.gi_split_pipeline,
            gi_ao_weight: 1.0,
            gi_ssgi_weight: 1.0,
            gi_ssr_weight: 1.0,
            // Lit params
            lit_sun_direction_ws: [0.3, 0.6, -1.0],
            sky_sun_direction_ws: [0.3, 0.6, -1.0],
            lit_sun_intensity: 1.0,
            lit_directional_scale: 1.0,
            astro_observation_active: false,
            astro_observation_revision: 0,
            astro_terrain_revision: 0,
            lit_ibl_intensity: 0.6,
            lit_use_ibl: true,
            lit_ibl_rotation_deg: 0.0,
            lit_brdf: 4,
            lit_roughness: 0.5,
            lit_debug_mode: 0,
            fallback_pipeline: fb_pipeline,
            viz_mode: VizMode::Material,
            gi_viz_mode: GiVizMode::None,
            use_ssao_composite: true,
            ssao_composite_mul: 1.0,
            ssao_blur_enabled: true,
            ibl_renderer: None,
            ibl_env_view: Some(lit.dummy_env_view),
            ibl_sampler: Some(lit.dummy_env_sampler),
            ibl_hdr_path: None,
            ibl_cache_dir: None,
            ibl_base_resolution: None,
            viz_depth_max_override: None,
            auto_snapshot_path: std::env::var("FORGE3D_AUTO_SNAPSHOT_PATH").ok(),
            auto_snapshot_done: false,
            dump_p5_requested: false,
            adapter_name,
            adapter_vendor,
            adapter_device,
            adapter_backend,
            adapter_device_type,
            adapter_driver,
            adapter_driver_info,
            debug_logged_render_gate: false,
            // Sky resources
            sky_bind_group_layout0: sky.sky_bind_group_layout0,
            sky_bind_group_layout1: sky.sky_bind_group_layout1,
            sky_pipeline: sky.sky_pipeline,
            sky_params: sky.sky_params,
            sky_camera: sky.sky_camera,
            sky_output: sky.sky_output,
            sky_output_view: sky.sky_output_view,
            sky_present_bind_group_layout: sky.sky_present_bind_group_layout,
            sky_present_depth_pipeline: sky.sky_present_depth_pipeline,
            sky_present_flat_pipeline: sky.sky_present_flat_pipeline,
            sky_present_sampler: sky.sky_present_sampler,
            night_pipeline: sky.night_pipeline,
            night_instances: sky.night_instances,
            night_bind_group: sky.night_bind_group,
            _night_moon_texture: sky.night_moon_texture,
            night_instance_count: 0,
            astro_night_revision: 0,
            sky_bg0_cache: std::cell::RefCell::new(None),
            sky_bg1_cache: std::cell::RefCell::new(None),
            sky_present_bg_cache: std::cell::RefCell::new(None),
            sky_snapshot_cache: None,
            sky_enabled: true,
            // Fog resources
            fog_enabled: false,
            fog_params: fog.fog_params,
            fog_camera: fog.fog_camera,
            fog_output: fog.fog_output,
            fog_output_view: fog.fog_output_view,
            fog_history: fog.fog_history,
            fog_history_view: fog.fog_history_view,
            fog_bg0_cache: std::cell::RefCell::new(None),
            fog_bg1_cache: std::cell::RefCell::new(None),
            fog_bg2_cache: std::cell::RefCell::new(None),
            fog_bg2_half_cache: std::cell::RefCell::new(None),
            fog_upsample_bg_cache: std::cell::RefCell::new(None),
            fog_bg3_cache: std::cell::RefCell::new(None),
            fog_depth_sampler: fog.fog_depth_sampler,
            fog_history_sampler: fog.fog_history_sampler,
            fog_pipeline: fog.fog_pipeline,
            fog_frame_index: 0,
            fog_history_state: crate::core::temporal_history::TemporalHistoryState::invalid(),
            fog_bgl3: fog.fog_bgl3,
            _froxel_tex: fog._froxel_tex,
            froxel_view: fog.froxel_view,
            froxel_sampler: fog.froxel_sampler,
            froxel_build_pipeline: fog.froxel_build_pipeline,
            froxel_apply_pipeline: fog.froxel_apply_pipeline,
            fog_half_res_enabled: false,
            fog_output_half: fog.fog_output_half,
            fog_output_half_view: fog.fog_output_half_view,
            fog_history_half: fog.fog_history_half,
            fog_history_half_view: fog.fog_history_half_view,
            fog_upsample_bgl: fog.fog_upsample_bgl,
            fog_upsample_pipeline: fog.fog_upsample_pipeline,
            fog_upsample_params: fog.fog_upsample_params,
            fog_bilateral: true,
            fog_upsigma: 0.02,
            fog_bgl0: fog.fog_bgl0,
            fog_bgl1: fog.fog_bgl1,
            fog_bgl2: fog.fog_bgl2,
            _fog_shadow_map: fog._fog_shadow_map,
            fog_shadow_view: fog.fog_shadow_view,
            fog_shadow_sampler: fog.fog_shadow_sampler,
            fog_shadow_matrix: fog.fog_shadow_matrix,
            _fog_zero_tex: fog._fog_zero_tex,
            fog_zero_view: fog.fog_zero_view,
            fog_density: 0.02,
            fog_g: 0.0,
            fog_steps: 64,
            fog_temporal_alpha: 0.2,
            fog_use_shadows: false,
            fog_mode: FogMode::Raymarch,
            csm,
            _csm_config: csm_config,
            csm_depth_pipeline,
            csm_depth_camera,
            csm_depth_bind_group,
            // Sky controls
            sky_model_id: sky_params_init.model_pad[0],
            sky_turbidity: sky_params_init.sun_direction_turbidity[3],
            sky_ground_albedo: sky_params_init.ground_albedo_sun_size_sun_intensity_exposure[0],
            sky_exposure: sky_params_init.ground_albedo_sun_size_sun_intensity_exposure[3],
            sky_sun_intensity: sky_params_init.ground_albedo_sun_size_sun_intensity_exposure[2],
            // HUD
            hud_enabled: true,
            hud,
            // Label manager
            label_manager: crate::labels::LabelManager::new(width, height),
            // Unified picking system
            unified_picking: UnifiedPickingSystem::new(Arc::clone(&device), Arc::clone(&queue)),
            selected_feature_id: 0,
            selected_layer_name: String::new(),
            ssr_scene_loaded: false,
            ssr_scene_preset: None,
            // Object transform
            object_translation: glam::DVec3::ZERO,
            object_rotation: glam::Quat::IDENTITY,
            object_scale: glam::Vec3::ONE,
            transform_version: 0,
            // P0.1/M1: OIT (Order-Independent Transparency) - disabled by default
            oit_enabled: false,
            oit_mode: "auto".to_string(),
            // P1.1: Previous frame view-projection matrix for motion vectors
            prev_view_proj: glam::Mat4::IDENTITY,
            // P1.2: TAA jitter state (disabled by default, enable via IPC or config)
            taa_jitter: crate::core::jitter::JitterState::new(),
            // P1.3: TAA renderer (initialized lazily when enabled)
            taa_renderer: None,
            // P5: Point cloud state (initialized lazily when loaded)
            point_cloud: None,
            // Scene bundle save/load requests (handled by Python-side)
            pending_bundle_save: None,
            pending_bundle_load: None,
            scene_review_registry: crate::viewer::scene_review::ViewerSceneReviewRegistry::default(
            ),
            command_error: None,
        };

        viewer.sync_ssr_params_to_gi();

        Ok(viewer)
    }
}

/// Read sky parameters from environment variables
fn read_sky_env_params() -> SkyUniforms {
    let sun_direction = [0.3, 0.8, 0.5];
    let mut model = 1;
    let mut turbidity = 2.5;
    let mut ground_albedo = 0.2;
    let mut exposure = 1.0;
    let mut sun_intensity = 20.0;

    if let Ok(model_str) = std::env::var("FORGE3D_SKY_MODEL") {
        let key = model_str
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '_', ' '], "");
        model = match key.as_str() {
            "preetham" => 0,
            "hosekwilkie" => 1,
            _ => 1,
        };
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_TURBIDITY") {
        if let Ok(f) = v.parse::<f32>() {
            turbidity = f;
        }
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_GROUND") {
        if let Ok(f) = v.parse::<f32>() {
            ground_albedo = f;
        }
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_EXPOSURE") {
        if let Ok(f) = v.parse::<f32>() {
            exposure = f;
        }
    }
    if let Ok(v) = std::env::var("FORGE3D_SKY_INTENSITY") {
        if let Ok(f) = v.parse::<f32>() {
            sun_intensity = f;
        }
    }

    SkyUniforms::new(
        sun_direction,
        turbidity,
        ground_albedo,
        1.0,
        sun_intensity,
        exposure,
        model,
    )
}

/// Create CSM depth pipeline resources
fn create_csm_depth_resources(
    device: &Arc<wgpu::Device>,
) -> RenderResult<(Option<wgpu::RenderPipeline>, Option<TrackedBuffer>)> {
    let csm_depth_shader = crate::core::shader_registry::create_labeled_shader_module(
        device,
        "viewer.csm.depth.shader",
        CSM_DEPTH_SHADER,
    );

    let csm_depth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("viewer.csm.depth.bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(128),
            },
            count: None,
        }],
    });

    let csm_depth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewer.csm.depth.pl"),
        bind_group_layouts: &[&csm_depth_bgl],
        push_constant_ranges: &[],
    });

    let pipeline =
        crate::core::shader_registry::with_error_scope(device, "viewer.csm.depth.pipeline", || {
            crate::core::shader_registry::create_render_pipeline_scoped(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some("viewer.csm.depth.pipeline"),
                    layout: Some(&csm_depth_pl),
                    vertex: wgpu::VertexState {
                        module: &csm_depth_shader,
                        entry_point: "vs_main",
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 40,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 12,
                                    shader_location: 1,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x2,
                                    offset: 24,
                                    shader_location: 2,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x2,
                                    offset: 32,
                                    shader_location: 3,
                                },
                            ],
                        }],
                    },
                    fragment: None,
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                },
            )
        });

    let camera = tracked_create_buffer(
        device,
        &wgpu::BufferDescriptor {
            label: Some("viewer.csm.depth.camera"),
            size: std::mem::size_of::<crate::viewer::viewer_types::ViewerShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )?;

    Ok((Some(pipeline), Some(camera)))
}

const CSM_DEPTH_SHADER: &str = r#"
// Viewer-specific 128-byte ABI: two consecutive mat4x4<f32> values.
struct ViewerShadowUniforms {
    light_view_proj : mat4x4<f32>,
    object_model : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uShadow : ViewerShadowUniforms;

struct VSIn {
    @location(0) pos : vec3<f32>,
    @location(1) nrm : vec3<f32>,
    @location(2) uv  : vec2<f32>,
    @location(3) rough_metal : vec2<f32>,
};

@vertex
fn vs_main(inp: VSIn) -> @builtin(position) vec4<f32> {
    let pos_ws = uShadow.object_model * vec4<f32>(inp.pos, 1.0);
    return uShadow.light_view_proj * pos_ws;
}

@fragment
fn fs_main() { }
"#;
