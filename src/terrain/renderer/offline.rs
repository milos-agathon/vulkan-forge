use super::draw::RenderTargets;
use super::*;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedTexture,
};
use crate::terrain::accumulation::apply_jitter_to_projection;
use crate::PyValueError;
use half::f16;
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use std::time::Instant;

const OFFLINE_HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const OFFLINE_ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const OFFLINE_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const OFFLINE_LUMINANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const OFFLINE_LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEFAULT_METRIC_TILE_SIZE: u32 = 16;
const DEFAULT_METRIC_HISTORY_WINDOW: usize = 3;
const OFFLINE_LUMINANCE_EPSILON: f32 = 1e-4;
const OFFLINE_TERRAIN_FILMIC_OPERATOR: u32 = 5;

fn resolve_offline_jitter_sequence_samples(
    aa_samples: u32,
    jitter_sequence_samples: Option<u32>,
) -> Result<u32, &'static str> {
    let sample_count = jitter_sequence_samples.unwrap_or(aa_samples);
    if sample_count == 0 {
        return Err("jitter_sequence_samples must be >= 1");
    }
    Ok(sample_count)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OfflineAccumulateUniforms {
    sample_index: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OfflineResolveUniforms {
    width: u32,
    height: u32,
    sample_count: u32,
    renormalize_normals: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OfflineLuminanceUniforms {
    width: u32,
    height: u32,
    sample_count: u32,
    _pad: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OfflineTonemapUniforms {
    width: u32,
    height: u32,
    operator_index: u32,
    _pad0: u32,
    white_point: f32,
    gamma: f32,
    _pad1: [f32; 2],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OfflineDepthCopyUniforms {
    width: u32,
    height: u32,
    _pad: [u32; 2],
}

impl TerrainScene {
    pub(super) fn create_offline_compute_resources(
        device: &wgpu::Device,
    ) -> super::core::OfflineComputeResources {
        fn create_pipeline(
            device: &wgpu::Device,
            label: &str,
            bind_group_layout: &wgpu::BindGroupLayout,
            shader_source: &str,
            entry_point: &str,
        ) -> wgpu::ComputePipeline {
            let shader = crate::core::shader_registry::create_labeled_shader_module(
                device,
                label,
                shader_source,
            );
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label}.layout")),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[],
            });
            crate::core::shader_registry::create_compute_pipeline_scoped(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point,
                },
            )
        }

        let accumulate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain.offline.accumulate.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: OFFLINE_ACCUM_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let accumulate_pipeline = create_pipeline(
            device,
            "terrain.offline.accumulate.pipeline",
            &accumulate_bind_group_layout,
            include_str!("../../shaders/offline_accumulate.wgsl"),
            "main",
        );

        let resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain.offline.resolve.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: OFFLINE_HDR_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let resolve_pipeline = create_pipeline(
            device,
            "terrain.offline.resolve.pipeline",
            &resolve_bind_group_layout,
            include_str!("../../shaders/offline_resolve.wgsl"),
            "main",
        );

        let depth_extract_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain.offline.depth_extract.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: OFFLINE_DEPTH_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let depth_extract_pipeline = create_pipeline(
            device,
            "terrain.offline.depth_extract.pipeline",
            &depth_extract_bind_group_layout,
            include_str!("../../shaders/offline_depth_extract.wgsl"),
            "main",
        );

        let depth_expand_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain.offline.depth_expand.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: OFFLINE_HDR_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let depth_expand_pipeline = create_pipeline(
            device,
            "terrain.offline.depth_expand.pipeline",
            &depth_expand_bind_group_layout,
            include_str!("../../shaders/offline_depth_expand.wgsl"),
            "main",
        );

        let luminance_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain.offline.luminance.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: OFFLINE_LUMINANCE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let luminance_pipeline = create_pipeline(
            device,
            "terrain.offline.luminance.pipeline",
            &luminance_bind_group_layout,
            include_str!("../../shaders/offline_luminance.wgsl"),
            "main",
        );

        let tonemap_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain.offline.tonemap.bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: OFFLINE_LDR_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let tonemap_shader_source = format!(
            "{}\n{}\n{}",
            include_str!("../../shaders/includes/determinism.wgsl"),
            include_str!("../../shaders/includes/tonemap_common.wgsl"),
            include_str!("../../shaders/tonemap_terrain_offline.wgsl")
        );
        let tonemap_pipeline = create_pipeline(
            device,
            "terrain.offline.tonemap.pipeline",
            &tonemap_bind_group_layout,
            &tonemap_shader_source,
            "main",
        );

        super::core::OfflineComputeResources {
            accumulate_bind_group_layout,
            accumulate_pipeline,
            resolve_bind_group_layout,
            resolve_pipeline,
            depth_extract_bind_group_layout,
            depth_extract_pipeline,
            depth_expand_bind_group_layout,
            depth_expand_pipeline,
            luminance_bind_group_layout,
            luminance_pipeline,
            tonemap_bind_group_layout,
            tonemap_pipeline,
        }
    }

    pub(super) fn offline_session_active(&self) -> Result<bool> {
        Ok(self
            .offline_state
            .lock()
            .map_err(|_| anyhow!("offline_state mutex poisoned"))?
            .is_some())
    }

    fn create_offline_output_texture(
        &self,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Result<(TrackedTexture, wgpu::TextureView)> {
        let texture = tracked_create_texture(
            self.device.as_ref(),
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            },
        )?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((texture, view))
    }

    fn build_offline_state(
        &mut self,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'_, f32>,
        water_mask: Option<PyReadonlyArray2<'_, f32>>,
        jitter_sequence_samples: u32,
    ) -> Result<super::core::OfflineAccumulationState> {
        let decoded = params.decoded().clone();
        self.prepare_frame_lighting(&decoded)?;

        let mut offline_params = params.clone();
        offline_params.msaa_samples = 1;

        let height_inputs =
            self.upload_height_inputs(heightmap, water_mask, offline_params.terrain_data_revision)?;
        self.prepare_geometry(
            &offline_params,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            height_inputs.terrain_data_hash,
        )?;
        let probe_world_span = if is_mesh_camera_mode(&offline_params.camera_mode) {
            offline_params.terrain_span.max(1e-3)
        } else {
            1.0
        };
        super::probes::prepare_probes(
            self,
            &decoded.probes,
            probe_world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            offline_params.z_scale,
            height_inputs.terrain_data_hash,
        )?;
        super::probes::prepare_reflection_probes(
            self,
            &decoded.reflection_probes,
            material_set,
            env_maps,
            &offline_params,
            &decoded,
            probe_world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            offline_params.z_scale,
            height_inputs.terrain_data_hash,
        )?;

        let materials =
            self.prepare_material_context_with_mode(material_set, &offline_params, &decoded, true)?;
        let ibl_bind_group = self.prepare_ibl_bind_group(env_maps)?;
        let height_curve_lut_uploaded = if offline_params.height_curve_mode.as_str() == "lut" {
            offline_params
                .height_curve_lut
                .as_ref()
                .map(|lut| self.upload_height_curve_lut(lut.as_ref().as_slice()))
                .transpose()?
        } else {
            None
        };

        let render_targets =
            self.create_render_targets_for_format(&offline_params, 1, 1, OFFLINE_HDR_FORMAT)?;
        let aov_targets = self.create_aov_render_targets(
            render_targets.internal_width,
            render_targets.internal_height,
            1,
            super::aov::AOV_ALL_BITS,
            false,
        )?;
        let beauty_accumulation = crate::terrain::AccumulationBuffer::new(
            self.device.as_ref(),
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        let albedo_accumulation = crate::terrain::AccumulationBuffer::new(
            self.device.as_ref(),
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        let normal_accumulation = crate::terrain::AccumulationBuffer::new(
            self.device.as_ref(),
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        let (depth_reference_texture, depth_reference_view) = self.create_offline_output_texture(
            "terrain.offline.depth_reference",
            render_targets.internal_width,
            render_targets.internal_height,
            OFFLINE_DEPTH_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        )?;
        let luminance_width = (render_targets.internal_width + 3) / 4;
        let luminance_height = (render_targets.internal_height + 3) / 4;
        let (luminance_texture, luminance_view) = self.create_offline_output_texture(
            "terrain.offline.luminance",
            luminance_width,
            luminance_height,
            OFFLINE_LUMINANCE_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        )?;
        let out_width = render_targets.out_width;
        let out_height = render_targets.out_height;
        let internal_width = render_targets.internal_width;
        let internal_height = render_targets.internal_height;
        let needs_scaling = render_targets.needs_scaling;
        let light_buffer = self
            .light_buffer
            .lock()
            .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
        let hdr_aov_pipeline = Self::create_aov_render_pipeline(
            self.device.as_ref(),
            &self.bind_group_layout,
            light_buffer.bind_group_layout(),
            &self.ibl_bind_group_layout,
            &self.shadow_bind_group_layout,
            &self.fog_bind_group_layout,
            &self.water_reflection_bind_group_layout,
            &self.material_layer_bind_group_layout,
            OFFLINE_HDR_FORMAT,
            1,
            super::aov::AOV_ALL_BITS,
            // VERITAS source-id capture is one-shot only; accumulation would
            // average ids, which is meaningless.
            false,
            is_clipmap_camera_mode(&offline_params.camera_mode),
        );
        let hdr_background_blit_pipeline = Self::create_depth_blit_pipeline(
            self.device.as_ref(),
            &self.blit_bind_group_layout,
            OFFLINE_HDR_FORMAT,
            1,
        );
        drop(light_buffer);

        Ok(super::core::OfflineAccumulationState {
            params: offline_params.clone(),
            decoded,
            height_inputs,
            materials,
            ibl_bind_group,
            height_curve_lut_uploaded,
            hdr_aov_pipeline,
            hdr_background_blit_pipeline,
            render_targets,
            aov_targets,
            beauty_accumulation,
            albedo_accumulation,
            normal_accumulation,
            _depth_reference_texture: depth_reference_texture,
            depth_reference_view,
            luminance_texture,
            luminance_view,
            luminance_width,
            luminance_height,
            jitter_sequence: crate::terrain::JitterSequence::new(
                jitter_sequence_samples,
                offline_params.aa_seed,
            ),
            total_samples: 0,
            out_width,
            out_height,
            internal_width,
            internal_height,
            needs_scaling,
            prev_tile_means: Vec::new(),
            prev_tile_mean_history: Vec::new(),
            prev_tile_size: 0,
        })
    }

    /// Render one offline accumulation sample.
    ///
    /// `batch_scope_draws` (CENSOR F-04): when `Some(draws)`, this sample's
    /// encoder is bracketed in a "terrain.offline.accumulate_batch" timing
    /// scope (recorded with `draws` = the batch's sample count) and the
    /// queries are resolved before this sample's submit. `accumulate_batch`
    /// passes it for the FIRST sample of a batch only, so the certificate
    /// keeps exactly one pass per batch; the timed region is that one
    /// representative sample.
    fn render_offline_sample(
        &mut self,
        state: &mut super::core::OfflineAccumulationState,
        jitter: (f32, f32),
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
        batch_scope_draws: Option<u32>,
    ) -> Result<()> {
        // Re-prepare per sample: interleaved non-offline renders may have
        // switched the geometry provider; the clipmap mesh cache makes this
        // free when nothing changed.
        self.prepare_geometry(
            &state.params,
            &state.height_inputs.heightmap_data,
            (state.height_inputs.width, state.height_inputs.height),
            state.height_inputs.terrain_data_hash,
        )?;
        let (eye, view, proj) = Self::build_camera_matrices(&state.params);
        let jittered_proj = apply_jitter_to_projection(
            proj,
            jitter.0,
            jitter.1,
            state.internal_width,
            state.internal_height,
        );
        let uniforms =
            Self::build_uniforms_with_matrices(&state.params, &state.decoded, view, jittered_proj);
        super::runtime_contract::record_observation(
            "terrain.accumulate_batch",
            &uniforms,
            &state.materials.shading_uniforms,
            &state.materials.overlay_binding.uniform,
            &state.height_inputs.heightmap_data,
            state.height_inputs.width,
            state.height_inputs.height,
        )
        .map_err(anyhow::Error::msg)?;
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.offline.uniform_buffer"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain.offline.encoder"),
            });

        let batch_scope = if batch_scope_draws.is_some() {
            ts_begin(timing, &mut encoder, "terrain.offline.accumulate_batch")
        } else {
            None
        };

        let material_vt_ready = self.prepare_material_vt_frame(
            &mut encoder,
            &state.params,
            &state.decoded,
            state.materials.gpu_materials.layer_count,
            state.internal_width,
            state.internal_height,
        )?;
        {
            let render_targets = &state.render_targets;
            let aov_targets = &state.aov_targets;

            let height_ao_computed = self.compute_height_ao_pass(
                &mut encoder,
                &state.height_inputs.heightmap_view,
                render_targets.internal_width,
                render_targets.internal_height,
                state.height_inputs.width,
                state.height_inputs.height,
                &state.params,
                &state.decoded,
            )?;
            let sun_vis_computed = self.compute_sun_visibility_pass(
                &mut encoder,
                &state.height_inputs.heightmap_view,
                render_targets.internal_width,
                render_targets.internal_height,
                state.height_inputs.width,
                state.height_inputs.height,
                &state.params,
                &state.decoded,
            )?;
            let height_curve_view = state
                .height_curve_lut_uploaded
                .as_ref()
                .map(|(texture, _)| texture.create_view(&wgpu::TextureViewDescriptor::default()))
                .unwrap_or_else(|| {
                    self._height_curve_identity_texture
                        .create_view(&wgpu::TextureViewDescriptor::default())
                });

            let shadow_setup = self.prepare_shadow_setup(
                &mut encoder,
                &state.params,
                &state.decoded,
                &state.height_inputs.heightmap_view,
                &height_curve_view,
                state.height_inputs.width,
                state.height_inputs.height,
            )?;
            let shadow_bind_group = shadow_setup
                .shadow_bind_group
                .as_ref()
                .unwrap_or(&self.noop_shadow.bind_group);
            let sky_texture = self.render_sky_texture(
                &mut encoder,
                &state.decoded,
                view,
                jittered_proj,
                eye,
                render_targets.internal_width,
                render_targets.internal_height,
            )?;
            let sky_view = sky_texture
                .as_ref()
                .map(|sky| &sky.view)
                .unwrap_or(&self.sky_fallback_view);
            let atmosphere_scattering_view = sky_texture
                .as_ref()
                .and_then(|sky| sky.scattering_view.as_ref())
                .unwrap_or(&self.atmosphere_scattering_fallback_view);

            let main_height_view = self.main_pass_height_view(&state.height_inputs.heightmap_view);
            let pass_bind_groups = self.create_terrain_pass_bind_groups(
                &uniform_buffer,
                main_height_view,
                state.materials.material_view(),
                state.materials.material_sampler(),
                state.materials.material_normal_view(),
                state.materials.material_roughness_view(),
                state.materials.material_mask_view(),
                state.materials.material_map_sampler(),
                &state.materials.shading_buffer,
                state.materials.colormap_view(),
                state.materials.colormap_sampler(),
                &state.materials.overlay_buffer,
                &height_curve_view,
                state.height_inputs.water_mask_view_uploaded.as_ref(),
                sky_view,
                atmosphere_scattering_view,
                height_ao_computed,
                sun_vis_computed,
                &state.decoded,
                shadow_setup.height_min,
                shadow_setup.height_exag,
                if is_zup_camera_mode(&state.params.camera_mode) {
                    eye.z
                } else {
                    eye.y
                },
                material_vt_ready,
            )?;

            let water_reflection_bind_group = self.prepare_water_reflection_bind_group(
                &mut encoder,
                &state.params,
                &state.decoded,
                render_targets.internal_width,
                render_targets.internal_height,
                eye,
                view,
                jittered_proj,
                main_height_view,
                state.materials.material_view(),
                state.materials.material_sampler(),
                &state.materials.shading_buffer,
                state.materials.colormap_view(),
                state.materials.colormap_sampler(),
                &state.materials.overlay_buffer,
                &height_curve_view,
                state.height_inputs.water_mask_view_uploaded.as_ref(),
                height_ao_computed,
                sun_vis_computed,
                &state.ibl_bind_group,
                shadow_bind_group,
                &pass_bind_groups.fog,
                &pass_bind_groups.material_layer,
            )?;

            if let Some(sky) = sky_texture.as_ref() {
                if sky.linear_hdr {
                    // Both textures are exact-resolution RGBA16F. A direct
                    // copy preserves linear HDR radiance and avoids a filtered
                    // fullscreen presentation pass before accumulation.
                    encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture {
                            texture: &sky.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyTexture {
                            texture: &render_targets.internal_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: render_targets.internal_width,
                            height: render_targets.internal_height,
                            depth_or_array_layers: 1,
                        },
                    );
                } else {
                    self.blit_background_texture_with_pipeline(
                        &mut encoder,
                        render_targets,
                        &sky.view,
                        &state.hdr_background_blit_pipeline,
                    )?;
                }
            }

            self.run_main_pass_with_aov_pipeline(
                &mut encoder,
                &state.params,
                render_targets,
                aov_targets,
                &state.hdr_aov_pipeline,
                &pass_bind_groups.main,
                &state.ibl_bind_group,
                shadow_bind_group,
                &pass_bind_groups.fog,
                &water_reflection_bind_group,
                &pass_bind_groups.material_layer,
                sky_texture.is_some(),
            )?;

            if state.total_samples == 0 {
                self.dispatch_offline_depth_extract_pass(
                    &mut encoder,
                    &aov_targets.required_depth()?.internal_view,
                    &state.depth_reference_view,
                    state.internal_width,
                    state.internal_height,
                )?;
            }
        }

        self.dispatch_offline_accumulation_pass(
            &mut encoder,
            &state.render_targets.internal_view,
            &mut state.beauty_accumulation,
            state.total_samples,
        )?;
        self.dispatch_offline_accumulation_pass(
            &mut encoder,
            &state.aov_targets.required_albedo()?.internal_view,
            &mut state.albedo_accumulation,
            state.total_samples,
        )?;
        self.dispatch_offline_accumulation_pass(
            &mut encoder,
            &state.aov_targets.required_normal()?.internal_view,
            &mut state.normal_accumulation,
            state.total_samples,
        )?;

        self.stage_material_vt_feedback_readback(&mut encoder)?;
        if let Some(draws) = batch_scope_draws {
            ts_end(timing, &mut encoder, batch_scope, draws);
            if let Some(manager) = timing.as_mut() {
                manager.resolve_queries(&mut encoder);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.finish_material_vt_frame()?;
        state.total_samples += 1;
        Ok(())
    }
    fn dispatch_offline_accumulation_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sample_view: &wgpu::TextureView,
        accumulation: &mut crate::terrain::AccumulationBuffer,
        sample_index: u32,
    ) -> Result<()> {
        let uniforms = OfflineAccumulateUniforms {
            sample_index,
            width: accumulation.width,
            height: accumulation.height,
            _pad: 0,
        };
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.offline.accumulate.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.offline.accumulate.bind_group"),
            layout: &self.offline_compute.accumulate_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(accumulation.current_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(accumulation.write_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain.offline.accumulate.pass"),
            timestamp_writes: None,
        });
        crate::core::shader_registry::record_shader_use("terrain.offline.accumulate.pipeline");
        pass.set_pipeline(&self.offline_compute.accumulate_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            (accumulation.width + 7) / 8,
            (accumulation.height + 7) / 8,
            1,
        );
        drop(pass);

        accumulation.swap();
        accumulation.increment_sample();
        Ok(())
    }

    fn blit_background_texture_with_pipeline(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_targets: &RenderTargets,
        source_view: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
    ) -> Result<()> {
        let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.offline.background.blit.bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
            ],
        });

        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = if render_targets.msaa_view.is_some() {
            Some(&render_targets.internal_view)
        } else {
            None
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.offline.background.blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.15,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        crate::core::shader_registry::record_shader_use("terrain.blit.depth.shader");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &blit_bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_main_pass_with_aov_pipeline(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_targets: &RenderTargets,
        aov_targets: &super::aov::TerrainAovTargets,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        water_reflection_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
        preserve_background: bool,
    ) -> Result<()> {
        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = if render_targets.msaa_view.is_some() {
            Some(&render_targets.internal_view)
        } else {
            None
        };

        let light_buffer_guard = self
            .light_buffer
            .lock()
            .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
        let light_bind_group = light_buffer_guard
            .bind_group()
            .expect("LightBuffer should always provide a bind group");

        let albedo = aov_targets.required_albedo()?;
        let normal = aov_targets.required_normal()?;
        let depth = aov_targets.required_depth()?;
        let albedo_view = albedo.msaa_view.as_ref().unwrap_or(&albedo.internal_view);
        let albedo_resolve = if albedo.msaa_view.is_some() {
            Some(&albedo.internal_view)
        } else {
            None
        };
        let normal_view = normal.msaa_view.as_ref().unwrap_or(&normal.internal_view);
        let normal_resolve = if normal.msaa_view.is_some() {
            Some(&normal.internal_view)
        } else {
            None
        };
        let depth_view = depth.msaa_view.as_ref().unwrap_or(&depth.internal_view);
        let depth_resolve = if depth.msaa_view.is_some() {
            Some(&depth.internal_view)
        } else {
            None
        };

        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: if preserve_background {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        })
                    },
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: albedo_view,
                resolve_target: albedo_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: normal_view,
                resolve_target: normal_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: depth_view,
                resolve_target: depth_resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.offline.render_pass.aov"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    // Background may arrive through an AETHER texture copy,
                    // which never initializes depth. Clear explicitly for
                    // every offline terrain pass; color preservation is an
                    // independent concern.
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        crate::core::shader_registry::record_shader_use("terrain_pbr_pom.aov.shader");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_bind_group(1, light_bind_group, &[]);
        pass.set_bind_group(2, ibl_bind_group, &[]);
        pass.set_bind_group(3, shadow_bind_group, &[]);
        pass.set_bind_group(4, fog_bind_group, &[]);
        pass.set_bind_group(5, water_reflection_bind_group, &[]);
        pass.set_bind_group(6, material_layer_bind_group, &[]);

        if params.culling == "hzb_two_phase" {
            crate::core::degradation::record_degradation(
                "rendering_fallback",
                "terrain_hzb_two_phase_offline_aov",
                "offline AOV rendering uses the direct clipmap path; two-phase HZB applies to the beauty pass",
            );
        }
        self.geometry_provider()?.draw(&mut pass);
        Ok(())
    }

    fn upload_rgba16_texture(
        &self,
        width: u32,
        height: u32,
        data: &[f32],
        label: &str,
    ) -> Result<TrackedTexture> {
        let texture = tracked_create_texture(
            self.device.as_ref(),
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: OFFLINE_HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;

        let mut bytes = Vec::with_capacity(data.len() * 2);
        for value in data {
            bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(texture)
    }

    fn dispatch_offline_resolve_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        accumulation: &crate::terrain::AccumulationBuffer,
        output_view: &wgpu::TextureView,
        sample_count: u32,
        renormalize_normals: bool,
    ) -> Result<()> {
        let uniforms = OfflineResolveUniforms {
            width: accumulation.width,
            height: accumulation.height,
            sample_count: sample_count.max(1),
            renormalize_normals: renormalize_normals as u32,
        };
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.offline.resolve.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.offline.resolve.bind_group"),
            layout: &self.offline_compute.resolve_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(accumulation.current_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain.offline.resolve.pass"),
            timestamp_writes: None,
        });
        crate::core::shader_registry::record_shader_use("terrain.offline.resolve.pipeline");
        pass.set_pipeline(&self.offline_compute.resolve_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            (accumulation.width + 7) / 8,
            (accumulation.height + 7) / 8,
            1,
        );
        Ok(())
    }

    fn dispatch_offline_depth_copy_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        bind_group_layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
        label_prefix: &str,
    ) -> Result<()> {
        let uniforms = OfflineDepthCopyUniforms {
            width,
            height,
            _pad: [0; 2],
        };
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label_prefix}.uniforms")),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label_prefix}.bind_group")),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{label_prefix}.pass")),
            timestamp_writes: None,
        });
        crate::core::shader_registry::record_shader_use(&format!("{label_prefix}.pipeline"));
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        Ok(())
    }

    fn dispatch_offline_depth_extract_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.dispatch_offline_depth_copy_pass(
            encoder,
            input_view,
            output_view,
            width,
            height,
            &self.offline_compute.depth_extract_bind_group_layout,
            &self.offline_compute.depth_extract_pipeline,
            "terrain.offline.depth_extract",
        )
    }

    fn dispatch_offline_depth_expand_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.dispatch_offline_depth_copy_pass(
            encoder,
            input_view,
            output_view,
            width,
            height,
            &self.offline_compute.depth_expand_bind_group_layout,
            &self.offline_compute.depth_expand_pipeline,
            "terrain.offline.depth_expand",
        )
    }

    fn dispatch_offline_luminance_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        accumulation: &crate::terrain::AccumulationBuffer,
        output_view: &wgpu::TextureView,
        sample_count: u32,
    ) -> Result<()> {
        let uniforms = OfflineLuminanceUniforms {
            width: accumulation.width,
            height: accumulation.height,
            sample_count: sample_count.max(1),
            _pad: 0,
        };
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.offline.luminance.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.offline.luminance.bind_group"),
            layout: &self.offline_compute.luminance_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(accumulation.current_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain.offline.luminance.pass"),
            timestamp_writes: None,
        });
        crate::core::shader_registry::record_shader_use("terrain.offline.luminance.pipeline");
        pass.set_pipeline(&self.offline_compute.luminance_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            ((accumulation.width + 3) / 4 + 7) / 8,
            ((accumulation.height + 3) / 4 + 7) / 8,
            1,
        );
        Ok(())
    }

    fn dispatch_offline_tonemap_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        operator_index: u32,
        white_point: f32,
        gamma: f32,
    ) -> Result<()> {
        let uniforms = OfflineTonemapUniforms {
            width,
            height,
            operator_index,
            _pad0: 0,
            white_point,
            gamma,
            _pad1: [0.0; 2],
        };
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.offline.tonemap.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.offline.tonemap.bind_group"),
            layout: &self.offline_compute.tonemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain.offline.tonemap.pass"),
            timestamp_writes: None,
        });
        crate::core::shader_registry::record_shader_use("terrain.offline.tonemap.pipeline");
        pass.set_pipeline(&self.offline_compute.tonemap_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        Ok(())
    }

    fn resolved_offline_tonemap_operator(
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
    ) -> u32 {
        let tonemap = &decoded.tonemap;
        let has_override = tonemap.lut_enabled
            || tonemap.white_balance_enabled
            || tonemap.operator_index != 2
            || (tonemap.white_point - 4.0).abs() > f32::EPSILON;
        if has_override {
            tonemap.operator_index
        } else {
            OFFLINE_TERRAIN_FILMIC_OPERATOR
        }
    }

    fn tile_luminance_means(
        luminance: &[f32],
        width: u32,
        height: u32,
        tile_size: u32,
    ) -> Vec<f32> {
        let tile = tile_size.max(1) as usize;
        let width_usize = width as usize;
        let height_usize = height as usize;
        let mut means = Vec::new();

        for y0 in (0..height_usize).step_by(tile) {
            for x0 in (0..width_usize).step_by(tile) {
                let y1 = (y0 + tile).min(height_usize);
                let x1 = (x0 + tile).min(width_usize);
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for y in y0..y1 {
                    let row_offset = y * width_usize;
                    for x in x0..x1 {
                        let idx = row_offset + x;
                        sum += luminance[idx];
                        count += 1;
                    }
                }
                means.push(sum / count.max(1) as f32);
            }
        }

        means
    }
}

#[pymethods]
impl TerrainRenderer {
    #[pyo3(signature = (
        material_set,
        env_maps,
        params,
        heightmap,
        water_mask=None,
        jitter_sequence_samples=None
    ))]
    pub fn begin_offline_accumulation<'py>(
        &mut self,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<'py, f32>,
        water_mask: Option<PyReadonlyArray2<'py, f32>>,
        jitter_sequence_samples: Option<u32>,
    ) -> PyResult<()> {
        if self
            .scene
            .offline_session_active()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to query offline state: {e:#}")))?
        {
            return Err(PyRuntimeError::new_err(
                "An offline accumulation session is already active.",
            ));
        }

        // Keep missing or malformed VT family requests fail-before-mutation:
        // build_offline_state allocates the complete offline graph and the
        // returned state is installed as an active session below.
        self.scene
            .validate_material_vt_request(params, material_set.materials().len() as u32)
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "Failed to begin offline accumulation during material VT preflight: {error:#}"
                ))
            })?;

        let jitter_sequence_samples =
            resolve_offline_jitter_sequence_samples(params.aa_samples, jitter_sequence_samples)
                .map_err(PyValueError::new_err)?;

        let state = self
            .scene
            .build_offline_state(
                material_set,
                env_maps,
                params,
                heightmap,
                water_mask,
                jitter_sequence_samples,
            )
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to begin offline accumulation: {e:#}"))
            })?;

        let mut guard = self
            .scene
            .offline_state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
        *guard = Some(state);
        Ok(())
    }

    #[pyo3(signature = (sample_count))]
    pub fn accumulate_batch(
        &mut self,
        py: Python<'_>,
        sample_count: u32,
    ) -> PyResult<Py<crate::OfflineBatchResult>> {
        if sample_count == 0 {
            return Err(PyValueError::new_err("sample_count must be >= 1"));
        }

        let (certificate_capture, _allocation_scope) = self
            .scene
            .begin_certificate_capture("terrain.accumulate_batch");
        // CENSOR F-04: live GPU timing through the scene's per-render timing
        // manager. The first sample of the batch carries the one
        // "terrain.offline.accumulate_batch" scope (see render_offline_sample).
        let mut timing = self.scene.take_render_timing();

        let start = Instant::now();
        let mut state = {
            let mut guard = self
                .scene
                .offline_state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
            guard.take().ok_or_else(|| {
                PyRuntimeError::new_err("No offline accumulation session is active")
            })?
        };

        let work_result = (|| -> PyResult<()> {
            for sample_index in 0..sample_count {
                let jitter = state.jitter_sequence.next();
                let batch_scope_draws = if sample_index == 0 {
                    Some(sample_count)
                } else {
                    None
                };
                self.scene
                    .render_offline_sample(&mut state, jitter, &mut timing, batch_scope_draws)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!("Offline sample render failed: {e:#}"))
                    })?;
            }
            Ok(())
        })();

        let total_samples = state.total_samples;
        let mut guard = self
            .scene
            .offline_state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
        *guard = Some(state);
        if let Err(err) = work_result {
            // Drop the manager rather than storing it: the batch scope may
            // hold unresolved/undrained queries that would otherwise leak
            // into the NEXT render's certificate. take_render_timing lazily
            // rebuilds one.
            drop(timing);
            return Err(err);
        }
        // Record the batch pass: live timestamps when they read back, else the
        // honest 0.0 record with the same label so certificate pass lists stay
        // identical across adapters.
        let mut recorded_live = false;
        let mut drained = true;
        if let Some(manager) = timing.as_mut() {
            match manager.get_results_blocking() {
                Ok(results) => {
                    for result in &results {
                        // Invalid timestamps are recorded as 0.0, never as a
                        // garbage delta (CENSOR audit F-04).
                        crate::core::certificate::record_pass(
                            &result.name,
                            result.certificate_gpu_ms(),
                            result.draw_calls,
                        );
                        recorded_live = true;
                    }
                }
                Err(e) => {
                    drained = false;
                    log::warn!("GPU timing readback failed: {e}");
                }
            }
        }
        if !recorded_live {
            crate::core::certificate::record_pass(
                "terrain.offline.accumulate_batch",
                0.0,
                sample_count,
            );
        }
        if drained {
            self.scene.store_render_timing(timing);
        }
        self.scene.finish_certificate_capture(certificate_capture);
        Py::new(
            py,
            crate::OfflineBatchResult::new(total_samples, start.elapsed().as_secs_f64() * 1000.0),
        )
    }
    #[pyo3(signature = (target_variance, tile_size=DEFAULT_METRIC_TILE_SIZE))]
    pub fn read_accumulation_metrics(
        &mut self,
        py: Python<'_>,
        target_variance: f32,
        tile_size: u32,
    ) -> PyResult<Py<crate::OfflineMetrics>> {
        let mut state = {
            let mut guard = self
                .scene
                .offline_state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
            guard.take().ok_or_else(|| {
                PyRuntimeError::new_err("No offline accumulation session is active")
            })?
        };
        if state.total_samples == 0 {
            let mut guard = self
                .scene
                .offline_state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
            *guard = Some(state);
            return Err(PyRuntimeError::new_err(
                "Cannot read accumulation metrics before rendering any samples",
            ));
        }

        let metrics_result = (|| -> PyResult<crate::OfflineMetrics> {
            let mut encoder =
                self.scene
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("terrain.offline.metrics.encoder"),
                    });
            self.scene
                .dispatch_offline_luminance_pass(
                    &mut encoder,
                    &state.beauty_accumulation,
                    &state.luminance_view,
                    state.total_samples,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("Luminance pass failed: {e:#}")))?;
            self.scene.queue.submit(Some(encoder.finish()));

            let luminance = py
                .allow_threads(|| {
                    crate::core::hdr::read_r32_texture(
                        &self.scene.device,
                        &self.scene.queue,
                        &state.luminance_texture,
                        state.luminance_width,
                        state.luminance_height,
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(format!("Luminance readback failed: {e}")))?;
            let means = TerrainScene::tile_luminance_means(
                &luminance,
                state.luminance_width,
                state.luminance_height,
                tile_size,
            );
            let prev_compatible = state.prev_tile_size == tile_size
                && !state.prev_tile_mean_history.is_empty()
                && state
                    .prev_tile_mean_history
                    .iter()
                    .all(|history| history.len() == means.len());
            let deltas: Vec<f32> = if prev_compatible {
                means
                    .iter()
                    .enumerate()
                    .map(|(idx, current)| {
                        let baseline = state
                            .prev_tile_mean_history
                            .iter()
                            .map(|history| history[idx])
                            .sum::<f32>()
                            / state.prev_tile_mean_history.len() as f32;
                        let denom = current
                            .abs()
                            .max(baseline.abs())
                            .max(OFFLINE_LUMINANCE_EPSILON);
                        (*current - baseline).abs() / denom
                    })
                    .collect()
            } else {
                vec![1.0; means.len()]
            };
            state.prev_tile_means = means.clone();
            state.prev_tile_mean_history.push(means);
            if state.prev_tile_mean_history.len() > DEFAULT_METRIC_HISTORY_WINDOW {
                state.prev_tile_mean_history.remove(0);
            }
            state.prev_tile_size = tile_size;

            let mut sorted = deltas.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mean_delta = if deltas.is_empty() {
                0.0
            } else {
                deltas.iter().sum::<f32>() / deltas.len() as f32
            };
            let p95_index = if sorted.is_empty() {
                0
            } else {
                ((sorted.len() - 1) as f32 * 0.95).round() as usize
            };
            let p95_delta = sorted.get(p95_index).copied().unwrap_or(0.0);
            let max_tile_delta = sorted.last().copied().unwrap_or(0.0);
            let converged_tiles = deltas
                .iter()
                .filter(|delta| **delta < target_variance.max(0.0))
                .count();
            let converged_tile_ratio = if deltas.is_empty() {
                1.0
            } else {
                converged_tiles as f32 / deltas.len() as f32
            };
            Ok(crate::OfflineMetrics::new(
                state.total_samples,
                mean_delta,
                p95_delta,
                max_tile_delta,
                converged_tile_ratio,
            ))
        })();

        let mut guard = self
            .scene
            .offline_state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
        *guard = Some(state);

        Py::new(py, metrics_result?)
    }

    #[pyo3(signature = ())]
    pub fn resolve_offline_hdr<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Py<crate::HdrFrame>, Py<crate::AovFrame>)> {
        let guard = self
            .scene
            .offline_state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
        let state = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("No offline accumulation session is active"))?;
        if state.total_samples == 0 {
            return Err(PyRuntimeError::new_err(
                "Cannot resolve offline HDR before rendering any samples",
            ));
        }

        let internal_width = state.internal_width;
        let internal_height = state.internal_height;
        let out_width = state.out_width;
        let out_height = state.out_height;
        let needs_scaling = state.needs_scaling;
        let decoded = state.decoded.clone();

        let mut encoder =
            self.scene
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain.offline.resolve.encoder"),
                });

        let (beauty_internal, beauty_internal_view) = self
            .scene
            .create_offline_output_texture(
                "terrain.offline.beauty.internal",
                internal_width,
                internal_height,
                OFFLINE_HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Beauty target alloc failed: {e:#}")))?;
        self.scene
            .dispatch_offline_resolve_pass(
                &mut encoder,
                &state.beauty_accumulation,
                &beauty_internal_view,
                state.total_samples,
                false,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Beauty resolve pass failed: {e:#}")))?;
        let beauty_texture = self
            .scene
            .resolve_aux_output(
                &mut encoder,
                &decoded,
                beauty_internal,
                beauty_internal_view,
                out_width,
                out_height,
                needs_scaling,
                false,
                "terrain.offline.beauty.resolved",
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Beauty resolve failed: {e:#}")))?;

        let (albedo_internal, albedo_internal_view) = self
            .scene
            .create_offline_output_texture(
                "terrain.offline.albedo.internal",
                internal_width,
                internal_height,
                OFFLINE_HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Albedo target alloc failed: {e:#}")))?;
        self.scene
            .dispatch_offline_resolve_pass(
                &mut encoder,
                &state.albedo_accumulation,
                &albedo_internal_view,
                state.total_samples,
                false,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Albedo resolve pass failed: {e:#}")))?;
        let albedo_texture = self
            .scene
            .resolve_aux_output(
                &mut encoder,
                &decoded,
                albedo_internal,
                albedo_internal_view,
                out_width,
                out_height,
                needs_scaling,
                false,
                "terrain.offline.albedo.resolved",
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Albedo resolve failed: {e:#}")))?;

        let (normal_internal, normal_internal_view) = self
            .scene
            .create_offline_output_texture(
                "terrain.offline.normal.internal",
                internal_width,
                internal_height,
                OFFLINE_HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Normal target alloc failed: {e:#}")))?;
        self.scene
            .dispatch_offline_resolve_pass(
                &mut encoder,
                &state.normal_accumulation,
                &normal_internal_view,
                state.total_samples,
                true,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Normal resolve pass failed: {e:#}")))?;
        let normal_texture = self
            .scene
            .resolve_aux_output(
                &mut encoder,
                &decoded,
                normal_internal,
                normal_internal_view,
                out_width,
                out_height,
                needs_scaling,
                true,
                "terrain.offline.normal.resolved",
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Normal resolve failed: {e:#}")))?;

        let (depth_internal, depth_internal_view) = self
            .scene
            .create_offline_output_texture(
                "terrain.offline.depth.internal",
                internal_width,
                internal_height,
                OFFLINE_HDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Depth target alloc failed: {e:#}")))?;
        self.scene
            .dispatch_offline_depth_expand_pass(
                &mut encoder,
                &state.depth_reference_view,
                &depth_internal_view,
                internal_width,
                internal_height,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Depth expand pass failed: {e:#}")))?;
        let depth_texture = self
            .scene
            .resolve_aux_output(
                &mut encoder,
                &decoded,
                depth_internal,
                depth_internal_view,
                out_width,
                out_height,
                needs_scaling,
                false,
                "terrain.offline.depth.resolved",
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Depth resolve failed: {e:#}")))?;

        self.scene.queue.submit(Some(encoder.finish()));
        drop(guard);

        let hdr_frame = crate::HdrFrame::new(
            self.scene.device.clone(),
            self.scene.queue.clone(),
            beauty_texture,
            out_width,
            out_height,
        );
        let aov_frame = crate::AovFrame::new(
            self.scene.device.clone(),
            self.scene.queue.clone(),
            Some(albedo_texture),
            Some(normal_texture),
            Some(depth_texture),
            None,
            out_width,
            out_height,
        );
        Ok((Py::new(py, hdr_frame)?, Py::new(py, aov_frame)?))
    }

    #[pyo3(signature = (data, size))]
    pub fn upload_hdr_frame<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray3<'py, f32>,
        size: (u32, u32),
    ) -> PyResult<Py<crate::HdrFrame>> {
        if data.ndim() != 3 {
            return Err(PyValueError::new_err("data must be a 3D float32 array"));
        }
        let shape = data.shape();
        let (height, width, channels) = (shape[0] as u32, shape[1] as u32, shape[2]);
        if (width, height) != size {
            return Err(PyValueError::new_err(format!(
                "size {:?} does not match data shape ({}, {})",
                size, width, height
            )));
        }
        if !matches!(channels, 3 | 4) {
            return Err(PyValueError::new_err("data must have 3 or 4 channels"));
        }

        let array = data.as_array();
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for row in array.outer_iter() {
            for pixel in row.outer_iter() {
                rgba.push(pixel[0]);
                rgba.push(pixel[1]);
                rgba.push(pixel[2]);
                rgba.push(if channels == 4 { pixel[3] } else { 1.0 });
            }
        }

        let texture = self
            .scene
            .upload_rgba16_texture(width, height, &rgba, "terrain.offline.hdr_upload")
            .map_err(|e| PyRuntimeError::new_err(format!("HDR upload failed: {e:#}")))?;
        Py::new(
            py,
            crate::HdrFrame::new(
                self.scene.device.clone(),
                self.scene.queue.clone(),
                texture,
                width,
                height,
            ),
        )
    }

    #[pyo3(signature = (hdr_frame))]
    pub fn tonemap_offline_hdr<'py>(
        &mut self,
        py: Python<'py>,
        hdr_frame: &crate::HdrFrame,
    ) -> PyResult<Py<crate::Frame>> {
        let (width, height) = hdr_frame.dimensions();
        let (operator_index, white_point, gamma) = {
            let guard = self
                .scene
                .offline_state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
            if let Some(state) = guard.as_ref() {
                (
                    TerrainScene::resolved_offline_tonemap_operator(&state.decoded),
                    state.decoded.tonemap.white_point,
                    state.params.gamma.max(0.1),
                )
            } else {
                (OFFLINE_TERRAIN_FILMIC_OPERATOR, 4.0, 2.2)
            }
        };

        let (texture, output_view) = self
            .scene
            .create_offline_output_texture(
                "terrain.offline.tonemapped",
                width,
                height,
                OFFLINE_LDR_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Tonemap target alloc failed: {e:#}")))?;
        let hdr_view = hdr_frame
            .texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.scene
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("terrain.offline.tonemap.encoder"),
                });
        self.scene
            .dispatch_offline_tonemap_pass(
                &mut encoder,
                &hdr_view,
                &output_view,
                width,
                height,
                operator_index,
                white_point,
                gamma,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Tonemap pass failed: {e:#}")))?;
        self.scene.queue.submit(Some(encoder.finish()));
        let frame = crate::Frame::new(
            self.scene.device.clone(),
            self.scene.queue.clone(),
            texture,
            width,
            height,
            OFFLINE_LDR_FORMAT,
        );
        self.end_offline_accumulation()?;
        Py::new(py, frame)
    }

    #[pyo3(signature = ())]
    pub fn end_offline_accumulation(&mut self) -> PyResult<()> {
        let mut guard = self
            .scene
            .offline_state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("offline_state mutex poisoned"))?;
        *guard = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_offline_jitter_sequence_samples, OFFLINE_DEPTH_FORMAT, OFFLINE_LDR_FORMAT,
    };

    #[test]
    fn test_resolve_offline_jitter_sequence_samples_defaults_to_aa_samples() {
        assert_eq!(resolve_offline_jitter_sequence_samples(8, None).unwrap(), 8);
    }

    #[test]
    fn test_resolve_offline_jitter_sequence_samples_uses_explicit_override() {
        assert_eq!(
            resolve_offline_jitter_sequence_samples(8, Some(24)).unwrap(),
            24
        );
    }

    #[test]
    fn test_resolve_offline_jitter_sequence_samples_rejects_zero() {
        assert_eq!(
            resolve_offline_jitter_sequence_samples(8, Some(0)).unwrap_err(),
            "jitter_sequence_samples must be >= 1"
        );
    }

    #[test]
    fn test_offline_depth_reference_uses_scalar_r32() {
        assert_eq!(OFFLINE_DEPTH_FORMAT, wgpu::TextureFormat::R32Float);
    }

    #[test]
    fn test_offline_tonemap_output_stays_linear_rgba8() {
        assert_eq!(OFFLINE_LDR_FORMAT, wgpu::TextureFormat::Rgba8Unorm);
    }
}
