use super::*;
use crate::viewer::event_loop::update_terrain_volumetrics_report;
use crate::viewer::ipc::TerrainVolumetricsReport;

impl ViewerTerrainScene {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain_viewer.bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(160),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let shader = crate::core::shader_registry::create_labeled_shader_module(
            &device,
            "terrain_viewer.shader",
            TERRAIN_SHADER,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain_viewer.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = crate::core::shader_registry::create_render_pipeline_scoped(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("terrain_viewer.pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        );

        #[cfg(feature = "enable-gpu-instancing")]
        // Match the offscreen terrain scatter composition mode.
        let scatter_renderer =
            crate::render::mesh_instanced::MeshInstancedRenderer::new_with_depth_state(
                &device,
                target_format,
                Some(wgpu::TextureFormat::Depth32Float),
                1,
                wgpu::CompareFunction::LessEqual,
                false,
            )?;
        #[cfg(feature = "enable-gpu-instancing")]
        let scatter_hlod_instance_buffer = crate::core::resource_tracker::tracked_create_buffer(
            &device,
            &wgpu::BufferDescriptor {
                label: Some("terrain.viewer.scatter.hlod.instance_buffer"),
                size: (std::mem::size_of::<f32>() * 16) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            depth_texture: None,
            depth_view: None,
            depth_size: (0, 0),
            snapshot_depth_texture: None,
            terrain: None,
            pbr_config: crate::viewer::terrain::pbr_renderer::ViewerTerrainPbrConfig::default(),
            pbr_pipeline: None,
            pbr_bind_group_layout: None,
            pbr_uniform_buffer: None,
            pbr_bind_group: None,
            terrain_ibl_renderer: None,
            terrain_ibl_hdr_path: None,
            terrain_ibl_specular_view: None,
            terrain_ibl_irradiance_view: None,
            terrain_ibl_brdf_view: None,
            terrain_ibl_sampler: None,
            terrain_ibl_specular_mip_count: 1,
            terrain_ibl_fallback_cube: None,
            terrain_ibl_fallback_cube_view: None,
            terrain_ibl_fallback_brdf: None,
            terrain_ibl_fallback_brdf_view: None,
            height_ao_pipeline: None,
            height_ao_bind_group_layout: None,
            height_ao_texture: None,
            height_ao_view: None,
            height_ao_uniform_buffer: None,
            sun_vis_pipeline: None,
            sun_vis_bind_group_layout: None,
            sun_vis_texture: None,
            sun_vis_view: None,
            sun_vis_uniform_buffer: None,
            sampler_nearest: None,
            fallback_texture: None,
            fallback_texture_view: None,
            post_process: None,
            dof_pass: None,
            motion_blur_pass: None,
            motion_blur_depth_pass: None,
            volumetrics_pass: None,
            denoise_pass: None,
            surface_format: target_format,
            overlay_stack: None,
            vector_overlay_stack: None,
            oit_enabled: false,
            oit_mode: "off".to_string(),
            wboit_color_texture: None,
            wboit_color_view: None,
            wboit_reveal_texture: None,
            wboit_reveal_view: None,
            wboit_compose_pipeline: None,
            wboit_compose_bind_group: None,
            wboit_compose_bind_group_layout: None,
            wboit_sampler: None,
            wboit_size: (0, 0),
            csm_renderer: None,
            moment_pass: None,
            moment_blur_pass: None,
            csm_uniform_buffer: None,
            shadow_pipeline: None,
            shadow_uniform_buffers: Vec::new(),
            shadow_bind_groups: Vec::new(),
            shadow_bind_group_revision: 0,
            // P1.4: TAA support
            taa_renderer: None,
            taa_jitter: crate::core::jitter::JitterState::new(),
            taa_velocity_texture: None,
            taa_velocity_view: None,
            taa_velocity_size: (0, 0),
            terrain_revision_counter: 0,
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_renderer,
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_hlod_instance_buffer,
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_batches: Vec::new(),
            #[cfg(feature = "enable-gpu-instancing")]
            scatter_last_frame_stats: crate::terrain::scatter::TerrainScatterFrameStats::default(),
            scatter_elapsed_time: 0.0,
        })
    }

    /// P0.1/M1: Set OIT mode for transparent overlay rendering
    pub fn set_oit_mode(&mut self, enabled: bool, mode: &str) {
        self.oit_enabled = enabled;
        self.oit_mode = mode.to_string();
        println!("[terrain_scene] OIT set: enabled={} mode={}", enabled, mode);
    }

    /// P1.4: Enable or disable TAA for terrain viewer
    pub fn set_taa_enabled(&mut self, enabled: bool) {
        if enabled && self.taa_renderer.is_none() {
            // Get current dimensions from depth texture or use defaults
            let (width, height) = if self.depth_size.0 > 0 {
                self.depth_size
            } else {
                (1920, 1080) // Default, will resize on first render
            };

            match crate::core::taa::TaaRenderer::new(&self.device, width, height) {
                Ok(renderer) => {
                    self.taa_renderer = Some(renderer);
                    println!(
                        "[terrain_taa] TAA renderer initialized ({}x{})",
                        width, height
                    );
                }
                Err(e) => {
                    eprintln!("[terrain_taa] Failed to create TAA renderer: {}", e);
                    return;
                }
            }
        }

        if let Some(ref mut taa) = self.taa_renderer {
            taa.set_enabled(enabled);
        }

        // Enable/disable jitter
        if enabled {
            self.taa_jitter = crate::core::jitter::JitterState::enabled();
            println!("[terrain_taa] TAA enabled with jitter");
        } else {
            self.taa_jitter = crate::core::jitter::JitterState::new();
            println!("[terrain_taa] TAA disabled");
        }
    }

    /// P1.4: Set TAA parameters (history weight, jitter scale)
    pub fn set_taa_params(&mut self, history_weight: Option<f32>, jitter_scale: Option<f32>) {
        if let Some(w) = history_weight {
            if let Some(ref mut taa) = self.taa_renderer {
                taa.set_history_weight(w);
            }
        }

        if let Some(scale) = jitter_scale {
            self.taa_jitter.set_scale(scale);
            // Note: we don't automatically disable jitter if scale is 0 here,
            // as the user might want to temporarily zero the scale but keep state enabled.
        }

        let current_weight = self
            .taa_renderer
            .as_ref()
            .map(|t| t.history_weight())
            .unwrap_or(0.0);
        println!(
            "[terrain_taa] params updated: weight={:.2} jitter_scale={:.2}",
            current_weight, self.taa_jitter.scale
        );
    }

    /// Configure PBR terrain rendering
    pub fn set_terrain_pbr(
        &mut self,
        enabled: Option<bool>,
        hdr_path: Option<String>,
        ibl_intensity: Option<f32>,
        hdr_rotate_deg: Option<f32>,
        shadow_technique: Option<String>,
        shadow_map_res: Option<u32>,
        exposure: Option<f32>,
        msaa: Option<u32>,
        normal_strength: Option<f32>,
        height_ao: Option<crate::viewer::viewer_enums::ViewerHeightAoConfig>,
        sun_visibility: Option<crate::viewer::viewer_enums::ViewerSunVisConfig>,
        materials: Option<crate::viewer::viewer_enums::ViewerMaterialLayerConfig>,
        vector_overlay: Option<crate::viewer::viewer_enums::ViewerVectorOverlayConfig>,
        tonemap: Option<crate::viewer::viewer_enums::ViewerTonemapConfig>,
        lens_effects: Option<crate::viewer::viewer_enums::ViewerLensEffectsConfig>,
        dof: Option<crate::viewer::viewer_enums::ViewerDofConfig>,
        motion_blur: Option<crate::viewer::viewer_enums::ViewerMotionBlurConfig>,
        volumetrics: Option<crate::viewer::viewer_enums::ViewerVolumetricsConfig>,
        denoise: Option<crate::viewer::viewer_enums::ViewerDenoiseConfig>,
        debug_mode: Option<u32>,
    ) -> Result<()> {
        // Stage every configuration change first. Shadow resources are also
        // staged before this candidate is published, so a rejected allocation
        // cannot leave the scene configured for resources it does not own.
        let mut candidate = self.pbr_config.clone();
        candidate.apply_updates(
            enabled,
            hdr_path,
            ibl_intensity,
            hdr_rotate_deg,
            shadow_technique,
            shadow_map_res,
            exposure,
            msaa,
            normal_strength,
            height_ao,
            sun_visibility,
            materials,
            vector_overlay,
            tonemap,
            denoise.clone(),
            debug_mode,
        );

        // Handle specialized config updates
        if let Some(lens) = lens_effects {
            candidate.apply_lens_effects(
                lens.enabled,
                lens.vignette_strength,
                lens.vignette_radius,
                lens.vignette_softness,
                lens.distortion,
                lens.chromatic_aberration,
            );
        }

        if let Some(d) = dof {
            candidate.apply_dof(
                d.enabled,
                d.f_stop,
                d.focus_distance,
                d.focal_length,
                &d.quality,
                d.tilt_pitch,
                d.tilt_yaw,
            );
        }

        if let Some(mb) = motion_blur {
            candidate.apply_motion_blur(
                mb.enabled,
                mb.samples,
                mb.shutter_open,
                mb.shutter_close,
                mb.cam_phi_delta,
                mb.cam_theta_delta,
                mb.cam_radius_delta,
            );
        }

        if let Some(v) = volumetrics {
            let density_volumes = v
                .density_volumes
                .iter()
                .map(
                    |volume| crate::viewer::terrain::pbr_renderer::DensityVolumeConfig {
                        preset: volume.preset.clone(),
                        center: volume.center,
                        size: volume.size,
                        resolution: volume.resolution,
                        density_scale: volume.density_scale,
                        edge_softness: volume.edge_softness,
                        noise_strength: volume.noise_strength,
                        floor_offset: volume.floor_offset,
                        ceiling: volume.ceiling,
                        plume_spread: volume.plume_spread,
                        wind: volume.wind,
                        seed: volume.seed,
                    },
                )
                .collect::<Vec<_>>();
            candidate.apply_volumetrics(
                v.enabled,
                &v.mode,
                v.density,
                v.height_falloff,
                v.scattering,
                v.absorption,
                v.light_shafts,
                v.shaft_intensity,
                v.steps,
                v.half_res,
                &density_volumes,
            );
        }

        let requires_moments =
            crate::viewer::terrain::pbr_renderer::shadow_technique_requires_moments(
                &candidate.shadow_technique,
            );
        crate::shadows::validate_shadow_allocation_budget(
            candidate.shadow_map_res.clamp(512, 8192),
            4,
            requires_moments,
        )?;
        if candidate.enabled {
            self.init_shadows_for_config(
                candidate.shadow_map_res.clamp(512, 8192),
                requires_moments,
            )?;
        }
        self.pbr_config = candidate;

        // Re-init specialized passes if enabled
        if self.pbr_config.lens_effects.enabled {
            self.init_post_process();
        }
        if self.pbr_config.dof.enabled {
            self.init_dof_pass();
        }
        if self.pbr_config.denoise.enabled {
            self.init_denoise_pass();
        }
        if !self.pbr_config.volumetrics.is_effectively_enabled()
            || self.pbr_config.volumetrics.density_volumes.is_empty()
        {
            update_terrain_volumetrics_report(TerrainVolumetricsReport::default());
        }

        println!(
            "[terrain_pbr] updated: {}",
            self.pbr_config.to_display_string()
        );
        Ok(())
    }
}
