use super::setup::RenderTargets;
use super::*;
use crate::core::resource_tracker::TrackedTexture;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MediaTonemapUniforms {
    exposure: f32,
    white_point: f32,
    gamma: f32,
    operator_index: u32,
    lut_enabled: u32,
    lut_strength: f32,
    lut_size: f32,
    white_balance_enabled: u32,
    temperature: f32,
    tint: f32,
    output_gamma_enabled: f32,
    _pad1: f32,
}

fn parse_cube_lut(path: &str) -> Result<(String, u32, Vec<u8>)> {
    use std::hash::{Hash, Hasher};
    let source = std::fs::read_to_string(path)
        .map_err(|error| anyhow!("failed to read terrain tonemap LUT {path:?}: {error}"))?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    source.hash(&mut hasher);
    let key = format!("{path}:{:016x}", hasher.finish());
    let mut size = None;
    let mut values = Vec::new();
    for (line_number, raw) in source.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("TITLE") {
            continue;
        }
        if let Some(value) = line.strip_prefix("LUT_3D_SIZE") {
            size = Some(
                value
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| anyhow!("invalid LUT_3D_SIZE at {path}:{}", line_number + 1))?,
            );
            continue;
        }
        if line.starts_with("DOMAIN_MIN 0 0 0") || line.starts_with("DOMAIN_MAX 1 1 1") {
            continue;
        }
        if line.starts_with("DOMAIN_") {
            return Err(anyhow!(
                "terrain tonemap LUT requires the canonical [0,1] domain: {path}:{}",
                line_number + 1
            ));
        }
        let components = line
            .split_whitespace()
            .map(str::parse::<f32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!("invalid LUT sample at {path}:{}", line_number + 1))?;
        if components.len() != 3
            || components
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(anyhow!(
                "LUT sample must contain three finite [0,1] values at {path}:{}",
                line_number + 1
            ));
        }
        values.extend(
            components
                .into_iter()
                .map(|value| (value * 255.0).round() as u8),
        );
        values.push(255);
    }
    let size = size.ok_or_else(|| anyhow!("terrain tonemap LUT {path:?} has no LUT_3D_SIZE"))?;
    let expected = usize::try_from(u64::from(size).pow(3) * 4)
        .map_err(|_| anyhow!("terrain tonemap LUT size exceeds usize"))?;
    if values.len() != expected {
        return Err(anyhow!(
            "terrain tonemap LUT {path:?} has {} bytes; expected {expected}",
            values.len()
        ));
    }
    Ok((key, size, values))
}

#[derive(Clone, Copy)]
enum TerrainDrawSource<'a> {
    Direct,
    Lod(&'a crate::terrain::clipmap::gpu_lod::GpuLodDrawResources),
    Culled(&'a crate::terrain::culling::two_phase::CullDrawResources),
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn resolve_realtime_media_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        render_targets: &RenderTargets,
    ) -> Result<(Arc<TrackedTexture>, u32, u32)> {
        if !render_targets.linear_hdr {
            return Err(anyhow!(
                "realtime media presentation requires a floating-point linear-HDR target"
            ));
        }
        let composite = self.realtime_media_composite_view()?;
        self.resolve_realtime_media_output_from_view(
            encoder,
            params,
            decoded,
            render_targets,
            &composite,
        )
    }

    pub(in crate::terrain::renderer) fn resolve_realtime_media_output_from_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        render_targets: &RenderTargets,
        composite: &wgpu::TextureView,
    ) -> Result<(Arc<TrackedTexture>, u32, u32)> {
        if !render_targets.linear_hdr {
            return Err(anyhow!(
                "realtime media presentation requires a floating-point linear-HDR target"
            ));
        }
        let mut lut_guard = self
            .media_tonemap_lut
            .lock()
            .map_err(|_| anyhow!("terrain media tonemap LUT mutex poisoned"))?;
        if decoded.tonemap.lut_enabled {
            let path = decoded.tonemap.lut_path.as_deref().ok_or_else(|| {
                anyhow!("terrain media tonemap LUT is enabled without a LUT path")
            })?;
            let (key, size, data) = parse_cube_lut(path)?;
            if lut_guard.as_ref().is_none_or(|lut| lut.key != key) {
                let texture = crate::core::resource_tracker::tracked_create_texture(
                    self.device.as_ref(),
                    &wgpu::TextureDescriptor {
                        label: Some("nephele.media.tonemap.lut"),
                        size: wgpu::Extent3d {
                            width: size,
                            height: size,
                            depth_or_array_layers: size,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D3,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                )?;
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(size * 4),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                *lut_guard = Some(super::super::core::MediaTonemapLut {
                    key,
                    _texture: texture,
                    view,
                    size,
                });
            }
        }
        let (lut_view, lut_size) = lut_guard
            .as_ref()
            .map_or((&self.media_tonemap_identity_lut_view, 2), |lut| {
                (&lut.view, lut.size)
            });
        let uniforms = MediaTonemapUniforms {
            exposure: params.exposure,
            white_point: decoded.tonemap.white_point,
            gamma: params.gamma(),
            operator_index: decoded.tonemap.operator_index,
            lut_enabled: u32::from(decoded.tonemap.lut_enabled),
            lut_strength: decoded.tonemap.lut_strength,
            lut_size: lut_size as f32,
            white_balance_enabled: u32::from(decoded.tonemap.white_balance_enabled),
            temperature: decoded.tonemap.temperature,
            tint: decoded.tonemap.tint,
            output_gamma_enabled: 1.0,
            _pad1: 0.0,
        };
        let uniform_buffer = crate::core::resource_tracker::tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("nephele.media.tonemap.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nephele.media.tonemap.bind_group"),
            layout: &self.media_tonemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(composite),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.media_tonemap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.media_tonemap_lut_sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nephele.media.tonemap.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_targets.resolved_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        crate::core::shader_registry::record_shader_use("nephele.media.tonemap.pipeline");
        pass.set_pipeline(&self.media_tonemap_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        Ok((
            render_targets.resolved_texture.clone(),
            render_targets.out_width,
            render_targets.out_height,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn encode_forward_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        height_inputs: &UploadedHeightInputs,
        materials: &PreparedMaterials,
        uniform_buffer: &Arc<crate::core::resource_tracker::TrackedBuffer>,
        ibl_bind_group: &wgpu::BindGroup,
        media_environment_radiance: [f32; 3],
        media_environment: Option<&crate::formats::hdr::HdrImage>,
        media_environment_intensity: f32,
        height_curve_view: &wgpu::TextureView,
        render_targets: &RenderTargets,
        shadow_setup: &crate::terrain::renderer::shadows::ShadowSetup,
        material_vt_ready: bool,
        height_ao_computed: bool,
        sun_vis_computed: bool,
        time_seconds: f32,
        timing: &mut Option<crate::core::gpu_timing::GpuTimingManager>,
    ) -> Result<()> {
        let shadow_bind_group = shadow_setup
            .shadow_bind_group
            .as_ref()
            .unwrap_or(&self.noop_shadow.bind_group);
        // Sky, atmosphere, and water must reconstruct the same rays as the
        // main terrain uniforms. ShadowSetup retains its own legacy camera for
        // cascade work and is not an authoritative view-camera source.
        let (camera_eye, camera_view, camera_proj) = Self::build_camera_matrices(params);
        let camera_height = if is_zup_camera_mode(&params.camera_mode) {
            camera_eye.z
        } else {
            camera_eye.y
        };
        let sky_scope = ts_begin(timing, encoder, "terrain.sky");
        let sky_texture = self.render_sky_texture(
            encoder,
            decoded,
            camera_view,
            camera_proj,
            camera_eye,
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        ts_end(timing, encoder, sky_scope, 0);
        let sky_view = sky_texture
            .as_ref()
            .map(|sky| &sky.view)
            .unwrap_or(&self.sky_fallback_view);
        let atmosphere_scattering_view = sky_texture
            .as_ref()
            .and_then(|sky| sky.scattering_view.as_ref())
            .unwrap_or(&self.atmosphere_scattering_fallback_view);
        self.prepare_realtime_media_radiance_provider(
            encoder,
            sky_texture.as_ref(),
            decoded,
            media_environment,
            media_environment_intensity,
            media_environment_radiance,
        )?;
        let main_height_view = self.main_pass_height_view(&height_inputs.heightmap_view);
        let pass_bind_groups = self.create_terrain_pass_bind_groups(
            uniform_buffer,
            main_height_view,
            materials.material_view(),
            materials.material_sampler(),
            materials.material_normal_view(),
            materials.material_roughness_view(),
            materials.material_mask_view(),
            materials.material_map_sampler(),
            &materials.shading_buffer,
            materials.colormap_view(),
            materials.colormap_sampler(),
            &materials.overlay_buffer,
            height_curve_view,
            height_inputs.water_mask_view_uploaded.as_ref(),
            sky_view,
            atmosphere_scattering_view,
            height_ao_computed,
            sun_vis_computed,
            decoded,
            shadow_setup.height_min,
            shadow_setup.height_exag,
            camera_height,
            material_vt_ready,
        )?;
        let water_reflection_bind_group = self.prepare_water_reflection_bind_group(
            encoder,
            params,
            decoded,
            render_targets.internal_width,
            render_targets.internal_height,
            camera_eye,
            camera_view,
            camera_proj,
            main_height_view,
            materials.material_view(),
            materials.material_sampler(),
            &materials.shading_buffer,
            materials.colormap_view(),
            materials.colormap_sampler(),
            &materials.overlay_buffer,
            height_curve_view,
            height_inputs.water_mask_view_uploaded.as_ref(),
            height_ao_computed,
            sun_vis_computed,
            ibl_bind_group,
            shadow_bind_group,
            &pass_bind_groups.fog,
            &pass_bind_groups.material_layer,
        )?;
        if let Some(sky) = sky_texture.as_ref() {
            let scope = ts_begin(timing, encoder, "terrain.background");
            self.blit_background_texture(encoder, render_targets, &sky.view, sky.linear_hdr)?;
            ts_end(timing, encoder, scope, 1);
        }
        let main_scope = ts_begin(timing, encoder, "terrain.main");
        let terrain_draw_calls = self.run_main_pass(
            encoder,
            params,
            decoded,
            render_targets,
            &pass_bind_groups.main,
            ibl_bind_group,
            shadow_bind_group,
            &pass_bind_groups.fog,
            &water_reflection_bind_group,
            &pass_bind_groups.material_layer,
            sky_texture.is_some(),
        )?;
        ts_end(timing, encoder, main_scope, terrain_draw_calls);

        #[cfg(feature = "enable-gpu-instancing")]
        {
            let scatter_state = self.build_scatter_render_state(
                params,
                decoded,
                height_inputs.width,
                height_inputs.height,
                shadow_setup.view_matrix,
                shadow_setup.proj_matrix,
                shadow_setup.eye,
                time_seconds,
            );
            self.render_scatter_pass(
                encoder,
                render_targets,
                &height_inputs.heightmap_view,
                shadow_setup.shadow_bind_group.as_ref(),
                &scatter_state,
            )?;
        }
        #[cfg(not(feature = "enable-gpu-instancing"))]
        let _ = time_seconds;
        Ok(())
    }

    pub(in crate::terrain::renderer) fn blit_background_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_targets: &RenderTargets,
        source_view: &wgpu::TextureView,
        linear_hdr: bool,
    ) -> Result<()> {
        let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.background.blit.bind_group"),
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

        let msaa_pipeline = if render_targets.sample_count > 1 {
            Some(if linear_hdr {
                Self::create_aether_depth_blit_pipeline(
                    self.device.as_ref(),
                    &self.blit_bind_group_layout,
                    self.color_format,
                    render_targets.sample_count,
                )
            } else {
                Self::create_depth_blit_pipeline(
                    self.device.as_ref(),
                    &self.blit_bind_group_layout,
                    self.color_format,
                    render_targets.sample_count,
                )
            })
        } else {
            None
        };
        let blit_pipeline = msaa_pipeline.as_ref().unwrap_or(if linear_hdr {
            &self.aether_background_blit_pipeline
        } else {
            &self.background_blit_pipeline
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.background.blit_pass"),
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

            crate::core::shader_registry::record_shader_use(if linear_hdr {
                "terrain.aether.blit.shader"
            } else if render_targets.sample_count > 1 {
                "terrain.blit.depth.shader"
            } else {
                "terrain.blit.shader"
            });
            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &blit_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_main_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        render_targets: &RenderTargets,
        bind_group: &wgpu::BindGroup,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        water_reflection_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
        preserve_background: bool,
    ) -> Result<u32> {
        let geometry = self.geometry_provider()?;
        // TESSELLA: the indirect-draw capability decision lives in
        // `core::capabilities` so the fallback it implies is named in one place
        // and recorded from inside the render capture (see below).
        let granted = self.device.features();
        let draw_mode = crate::core::capabilities::IndirectDrawMode::negotiate(granted);
        let first_instance = draw_mode.first_instance;
        let multi_draw_count = draw_mode.multi_draw_count;
        let half_height = (decoded.clamp.height_range.1 - decoded.clamp.height_range.0).abs()
            * params.z_scale.abs()
            * 0.5;
        let skirt = clipmap_camera_config(&params.camera_mode)
            .map(|config| config.ring_resolution as f32 * 0.001 * params.z_scale.abs())
            .unwrap_or(0.0);
        // `culling="none"` is the pixel-correctness oracle: submit the exact
        // combined clipmap mesh once. Feeding every off-screen region through
        // the compacted indirect path changes overlap/order at ring corners
        // and previously produced a uniformly magenta baseline on Metal.
        let indirect = if params.culling == "none" {
            None
        } else {
            geometry.encode_indirect(
                self.queue.as_ref(),
                encoder,
                params,
                (-half_height - skirt, half_height),
                first_instance,
            )
        };
        // Only a frame that actually issues indirect draws may claim the CPU
        // draw-loop fallback: `culling="none"` and non-clipmap Grid geometry both
        // leave `indirect` as None and draw once, directly. Recording here (and
        // not in `TerrainGeometry::draw_indirect_buffers`, which runs up to four
        // times per frame) keeps the claim exactly as strong as the truth.
        if indirect.is_some() {
            draw_mode.record_fallbacks(granted);
        }
        let hzb_requested = params.culling == "hzb_two_phase";
        let hzb_enabled = hzb_requested
            && render_targets.sample_count == 1
            && indirect.is_some()
            && self.two_phase_culler.is_some();
        if hzb_requested && !hzb_enabled {
            crate::core::degradation::record_degradation(
                "rendering_fallback",
                "terrain_hzb_two_phase",
                "requires single-sample clipmap geometry; using frustum-only indirect draws",
            );
        }
        let visibility_requested = params.shading == "visibility";
        let visibility_enabled =
            visibility_requested && geometry.is_clipmap() && render_targets.sample_count == 1;
        if visibility_requested && !visibility_enabled {
            crate::core::degradation::record_degradation(
                "rendering_fallback",
                "terrain_visibility_buffer",
                "requires single-sample clipmap geometry; using forward material shading",
            );
        }
        if visibility_enabled {
            self.ensure_visibility_buffer(
                render_targets.internal_width,
                render_targets.internal_height,
            )?;
        } else if geometry.is_clipmap() && render_targets.sample_count == 1 {
            // Reuse the counters readback to capture the actual forward
            // material/feedback invocation baseline.
            self.ensure_visibility_buffer(
                render_targets.internal_width,
                render_targets.internal_height,
            )?;
        }
        encoder.clear_buffer(&self.vt_frame_counters_buffer, 0, None);
        if hzb_enabled {
            let lod = indirect.expect("HZB requires clipmap LOD resources");
            let (_, view, proj) = Self::build_camera_matrices(params);
            let view_proj = proj * view;
            let culler = self
                .two_phase_culler
                .as_ref()
                .expect("HZB culler prepared with clipmap geometry");
            culler.phase1(
                self.queue.as_ref(),
                encoder,
                view_proj,
                (-half_height - skirt, half_height),
                first_instance,
            );
            let phase1_source = TerrainDrawSource::Culled(culler.phase1_resources());
            let phase1_visibility_calls = if visibility_enabled {
                self.encode_visibility_draw_pass(
                    encoder,
                    render_targets,
                    bind_group,
                    preserve_background,
                    false,
                    phase1_source,
                    multi_draw_count,
                    first_instance,
                )?
            } else {
                0
            };
            let phase1_calls = if visibility_enabled {
                phase1_visibility_calls
            } else {
                self.encode_terrain_draw_pass(
                    encoder,
                    render_targets,
                    bind_group,
                    ibl_bind_group,
                    shadow_bind_group,
                    fog_bind_group,
                    water_reflection_bind_group,
                    material_layer_bind_group,
                    preserve_background,
                    false,
                    false,
                    phase1_source,
                    multi_draw_count,
                    first_instance,
                )?
            };
            culler.build_phase2_hzb(self.device.as_ref(), encoder, &render_targets.depth_view)?;
            culler.phase2(
                self.queue.as_ref(),
                encoder,
                view_proj,
                (-half_height - skirt, half_height),
                first_instance,
            );
            let phase2_source = TerrainDrawSource::Culled(culler.phase2_resources());
            let phase2_visibility_calls = if visibility_enabled {
                self.encode_visibility_draw_pass(
                    encoder,
                    render_targets,
                    bind_group,
                    true,
                    true,
                    phase2_source,
                    multi_draw_count,
                    first_instance,
                )?
            } else {
                0
            };
            let phase2_calls = if visibility_enabled {
                phase2_visibility_calls
            } else {
                self.encode_terrain_draw_pass(
                    encoder,
                    render_targets,
                    bind_group,
                    ibl_bind_group,
                    shadow_bind_group,
                    fog_bind_group,
                    water_reflection_bind_group,
                    material_layer_bind_group,
                    true,
                    true,
                    false,
                    phase2_source,
                    multi_draw_count,
                    first_instance,
                )?
            };
            if visibility_enabled {
                self.encode_visibility_resolve_pass(
                    encoder,
                    render_targets,
                    bind_group,
                    ibl_bind_group,
                    shadow_bind_group,
                    fog_bind_group,
                    water_reflection_bind_group,
                    material_layer_bind_group,
                    preserve_background,
                    phase1_source,
                    multi_draw_count,
                    first_instance,
                )?;
                self.encode_visibility_resolve_pass(
                    encoder,
                    render_targets,
                    bind_group,
                    ibl_bind_group,
                    shadow_bind_group,
                    fog_bind_group,
                    water_reflection_bind_group,
                    material_layer_bind_group,
                    true,
                    phase2_source,
                    multi_draw_count,
                    first_instance,
                )?;
            }
            self.stage_visibility_stats(encoder)?;
            // No second pyramid build here: `build_phase2_hzb` above already
            // produced the max-reduced pyramid, and `finish_frame`'s index flip
            // hands that same pyramid to the next frame's phase 1. See
            // `TwoPhaseCuller::build_phase2_hzb`.
            culler.stage_stats(encoder, lod);
            return Ok(phase1_calls + phase2_calls);
        }
        let draw_source = indirect
            .map(TerrainDrawSource::Lod)
            .unwrap_or(TerrainDrawSource::Direct);
        let visibility_calls = if visibility_enabled {
            self.encode_visibility_draw_pass(
                encoder,
                render_targets,
                bind_group,
                preserve_background,
                false,
                draw_source,
                multi_draw_count,
                first_instance,
            )?
        } else {
            0
        };
        let draw_calls = if visibility_enabled {
            visibility_calls
        } else {
            self.encode_terrain_draw_pass(
                encoder,
                render_targets,
                bind_group,
                ibl_bind_group,
                shadow_bind_group,
                fog_bind_group,
                water_reflection_bind_group,
                material_layer_bind_group,
                preserve_background,
                false,
                false,
                draw_source,
                multi_draw_count,
                first_instance,
            )?
        };
        if visibility_enabled {
            self.encode_visibility_resolve_pass(
                encoder,
                render_targets,
                bind_group,
                ibl_bind_group,
                shadow_bind_group,
                fog_bind_group,
                water_reflection_bind_group,
                material_layer_bind_group,
                preserve_background,
                draw_source,
                multi_draw_count,
                first_instance,
            )?;
            self.stage_visibility_stats(encoder)?;
        } else if geometry.is_clipmap() && render_targets.sample_count == 1 {
            self.stage_visibility_stats(encoder)?;
        }
        Ok(draw_calls)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_visibility_draw_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_targets: &RenderTargets,
        bind_group: &wgpu::BindGroup,
        load_depth: bool,
        load_visibility: bool,
        draw_source: TerrainDrawSource<'_>,
        multi_draw_count: bool,
        first_instance: bool,
    ) -> Result<u32> {
        let geometry = self.geometry_provider()?;
        let pipeline_cache = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer pipeline mutex poisoned"))?;
        let pipeline = pipeline_cache
            .visibility_write_pipeline
            .as_ref()
            .ok_or_else(|| anyhow!("visibility write pipeline not initialized"))?;
        let visibility = self
            .visibility_buffer
            .lock()
            .map_err(|_| anyhow!("terrain visibility buffer mutex poisoned"))?;
        let visibility = visibility
            .as_ref()
            .ok_or_else(|| anyhow!("terrain visibility buffer not initialized"))?;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.visibility.write_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: visibility.view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if load_visibility {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if load_depth {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        crate::core::shader_registry::record_shader_use("terrain_visbuffer_write.shader");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let draw_calls = match draw_source {
            TerrainDrawSource::Direct => {
                geometry.draw(&mut pass);
                1
            }
            TerrainDrawSource::Lod(resources) => {
                geometry.draw_indirect(&mut pass, resources, multi_draw_count, first_instance)
            }
            TerrainDrawSource::Culled(resources) => {
                geometry.draw_culled(&mut pass, resources, multi_draw_count, first_instance)
            }
        };
        Ok(draw_calls)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_visibility_resolve_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_targets: &RenderTargets,
        bind_group: &wgpu::BindGroup,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        water_reflection_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
        preserve_color: bool,
        draw_source: TerrainDrawSource<'_>,
        multi_draw_count: bool,
        first_instance: bool,
    ) -> Result<()> {
        let resolve_bind_group = self.visibility_resolve_bind_group()?;
        let pipeline_cache = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer pipeline mutex poisoned"))?;
        let pipeline = pipeline_cache
            .visibility_resolve_pipeline
            .as_ref()
            .ok_or_else(|| anyhow!("visibility resolve pipeline not initialized"))?;
        let light_buffer_guard = self
            .light_buffer
            .lock()
            .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
        let light_bind_group = light_buffer_guard
            .bind_group()
            .expect("LightBuffer should always provide a bind group");
        let color_view = render_targets
            .msaa_view
            .as_ref()
            .unwrap_or(&render_targets.internal_view);
        let resolve_target = render_targets
            .msaa_view
            .as_ref()
            .map(|_| &render_targets.internal_view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.visibility.geometry_resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: if preserve_color {
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
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        crate::core::shader_registry::record_shader_use("terrain_visbuffer_resolve.shader");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_bind_group(1, light_bind_group, &[]);
        pass.set_bind_group(2, ibl_bind_group, &[]);
        pass.set_bind_group(3, shadow_bind_group, &[]);
        pass.set_bind_group(4, fog_bind_group, &[]);
        pass.set_bind_group(5, water_reflection_bind_group, &[]);
        pass.set_bind_group(6, material_layer_bind_group, &[]);
        pass.set_bind_group(7, &resolve_bind_group, &[]);
        let geometry = self.geometry_provider()?;
        match draw_source {
            TerrainDrawSource::Direct => geometry.draw(&mut pass),
            TerrainDrawSource::Lod(resources) => {
                geometry.draw_indirect(&mut pass, resources, multi_draw_count, first_instance);
            }
            TerrainDrawSource::Culled(resources) => {
                geometry.draw_culled(&mut pass, resources, multi_draw_count, first_instance);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_terrain_draw_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_targets: &RenderTargets,
        bind_group: &wgpu::BindGroup,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        water_reflection_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
        preserve_color: bool,
        load_depth: bool,
        visibility_resolve: bool,
        draw_source: TerrainDrawSource<'_>,
        multi_draw_count: bool,
        first_instance: bool,
    ) -> Result<u32> {
        let geometry = self.geometry_provider()?;
        let pipeline_cache = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer pipeline mutex poisoned"))?;

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

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: if preserve_color {
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
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &render_targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: if load_depth {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(1.0)
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            crate::core::shader_registry::record_shader_use(if visibility_resolve {
                "terrain_visbuffer_resolve.shader"
            } else if geometry.is_clipmap() {
                "terrain_pbr_pom.clipmap.shader"
            } else {
                "terrain_pbr_pom.shader"
            });
            pass.set_pipeline(if geometry.is_clipmap() {
                if visibility_resolve {
                    pipeline_cache
                        .visibility_resolve_pipeline
                        .as_ref()
                        .ok_or_else(|| anyhow!("visibility resolve pipeline not initialized"))?
                } else {
                    pipeline_cache.clipmap_pipeline.as_ref().ok_or_else(|| {
                        anyhow!("clipmap pipeline not initialized for clipmap geometry")
                    })?
                }
            } else {
                &pipeline_cache.pipeline
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_bind_group(1, light_bind_group, &[]);
            pass.set_bind_group(2, ibl_bind_group, &[]);
            pass.set_bind_group(3, shadow_bind_group, &[]);
            pass.set_bind_group(4, fog_bind_group, &[]);
            pass.set_bind_group(5, water_reflection_bind_group, &[]);
            pass.set_bind_group(6, material_layer_bind_group, &[]);
            let draw_calls = match draw_source {
                TerrainDrawSource::Direct => {
                    geometry.draw(&mut pass);
                    1
                }
                TerrainDrawSource::Lod(resources) => {
                    geometry.draw_indirect(&mut pass, resources, multi_draw_count, first_instance)
                }
                TerrainDrawSource::Culled(resources) => {
                    geometry.draw_culled(&mut pass, resources, multi_draw_count, first_instance)
                }
            };
            return Ok(draw_calls);
        }
    }

    pub(in crate::terrain::renderer) fn resolve_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        render_targets: &RenderTargets,
    ) -> Result<(Arc<TrackedTexture>, u32, u32)> {
        if !render_targets.needs_scaling {
            return Ok((
                render_targets.resolved_texture.clone(),
                render_targets.out_width,
                render_targets.out_height,
            ));
        }
        let sampling = &decoded.sampling;
        let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.blit.sampler"),
            address_mode_u: Self::map_address_mode(sampling.address_u),
            address_mode_v: Self::map_address_mode(sampling.address_v),
            address_mode_w: Self::map_address_mode(sampling.address_w),
            mag_filter: Self::map_filter_mode(sampling.mag_filter),
            min_filter: Self::map_filter_mode(sampling.min_filter),
            mipmap_filter: Self::map_filter_mode(sampling.mip_filter),
            anisotropy_clamp: sampling.anisotropy as u16,
            ..Default::default()
        });
        let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.blit.bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_targets.internal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });

        {
            let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_targets.resolved_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            crate::core::shader_registry::record_shader_use("terrain.blit.shader");
            blit_pass.set_pipeline(&self.blit_pipeline);
            blit_pass.set_bind_group(0, &blit_bind_group, &[]);
            blit_pass.draw(0..3, 0..1);
        }

        let _ = params;
        Ok((
            render_targets.resolved_texture.clone(),
            render_targets.out_width,
            render_targets.out_height,
        ))
    }
}

#[cfg(test)]
mod media_tonemap_tests {
    #[test]
    fn realtime_media_honors_configured_aces_instead_of_legacy_filmic() {
        let source = include_str!("execute.rs");
        assert!(source.contains("operator_index: decoded.tonemap.operator_index"));
        assert!(!source.contains("operator_index: 5,"));
    }

    #[test]
    fn realtime_media_uses_authoritative_tonemap_contract() {
        let shader = include_str!("../../../shaders/postprocess_tonemap.wgsl");
        let common = include_str!("../../../shaders/includes/tonemap_common.wgsl");
        assert!(shader.contains("apply_white_balance"));
        assert!(shader.contains("sample_lut"));
        assert!(shader.contains("linear_to_srgb(tonemapped_color)"));
        assert!(!shader.contains("pow(tonemapped_color"));
        assert!(common.contains("0.0031308"));
        assert!(common.contains("12.92"));
        assert!(common.contains("let a = vec3<f32>(0.055)"));
        assert!(common.contains("(vec3<f32>(1.0) + a) * powed"));
        let constructor = include_str!("../pipeline_cache.rs");
        assert!(constructor.contains("postprocess_tonemap.wgsl"));
        assert!(constructor.contains("tonemap_common.wgsl"));
        assert!(!constructor.contains("struct MediaTonemapUniforms"));
    }

    #[test]
    fn enabled_and_acceptance_no_medium_share_one_presentation_resolver() {
        let aov = include_str!("../aov.rs");
        let py_api = include_str!("../py_api.rs");
        assert!(aov.contains("render_nephele_acceptance_no_medium"));
        assert!(aov.contains("if presentation_enabled"));
        assert!(aov.contains("resolve_realtime_media_output_from_view"));
        assert!(py_api.contains("include_no_medium=false"));
        assert!(py_api.contains("no_medium_beauty"));
        assert!(py_api.contains("camera_contract"));
        let acceptance = py_api
            .split("fn _capture_nephele_acceptance")
            .nth(1)
            .expect("private acceptance capture must exist");
        assert!(
            acceptance
                .find("read_realtime_media_termination_slice")
                .unwrap()
                < acceptance.find("clear_realtime_media").unwrap(),
            "enabled termination evidence must be read before the no-medium baseline detaches media"
        );
    }
}
