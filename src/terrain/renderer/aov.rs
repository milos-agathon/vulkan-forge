use super::draw::RenderTargets;
use super::*;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedTexture,
};
use crate::terrain::render_params;

const TERRAIN_AOV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(super) const AOV_ALBEDO_BIT: u8 = 1 << 0;
pub(super) const AOV_NORMAL_BIT: u8 = 1 << 1;
pub(super) const AOV_DEPTH_BIT: u8 = 1 << 2;
pub(super) const AOV_ALL_BITS: u8 = AOV_ALBEDO_BIT | AOV_NORMAL_BIT | AOV_DEPTH_BIT;

fn requested_aov_output_mask(settings: &render_params::AovSettingsNative) -> u8 {
    // `render_with_aov` is itself an explicit capture request. Preserve its
    // established contract: the per-channel flags select outputs even when
    // `enabled` is false (the master flag controls ambient AOV behavior on
    // other render surfaces).
    (u8::from(settings.albedo) * AOV_ALBEDO_BIT)
        | (u8::from(settings.normal) * AOV_NORMAL_BIT)
        | (u8::from(settings.depth) * AOV_DEPTH_BIT)
}

#[cfg(test)]
mod output_mask_tests {
    use super::*;

    #[test]
    fn explicit_capture_uses_channel_flags_when_master_switch_is_off() {
        let settings = render_params::AovSettingsNative::default();
        assert!(!settings.enabled);
        assert_eq!(requested_aov_output_mask(&settings), AOV_ALL_BITS);
    }

    #[test]
    fn depth_only_capture_selects_only_the_depth_attachment() {
        let mut settings = render_params::AovSettingsNative::default();
        settings.albedo = false;
        settings.normal = false;
        assert_eq!(requested_aov_output_mask(&settings), AOV_DEPTH_BIT);
    }
}

fn ensure_aov_shading_supported(shading: &str) -> Result<()> {
    if shading == "visibility" {
        return Err(anyhow!(
            "terrain AOV capture does not support shading='visibility'; use \
             render_terrain_pbr_pom for visibility rendering"
        ));
    }
    Ok(())
}

pub(super) struct AovAttachmentTarget {
    pub(super) internal_texture: TrackedTexture,
    pub(super) internal_view: wgpu::TextureView,
    pub(super) _msaa_texture: Option<TrackedTexture>,
    pub(super) msaa_view: Option<wgpu::TextureView>,
}

pub(super) struct TerrainAovTargets {
    pub(super) albedo: Option<AovAttachmentTarget>,
    pub(super) normal: Option<AovAttachmentTarget>,
    pub(super) depth: Option<AovAttachmentTarget>,
    /// VERITAS: optional single-sample R32Uint per-pixel source-id target.
    pub(super) source_id: Option<AovAttachmentTarget>,
}

impl TerrainAovTargets {
    pub(super) fn required_albedo(&self) -> Result<&AovAttachmentTarget> {
        self.albedo
            .as_ref()
            .ok_or_else(|| anyhow!("offline AOV graph requires the albedo attachment"))
    }

    pub(super) fn required_normal(&self) -> Result<&AovAttachmentTarget> {
        self.normal
            .as_ref()
            .ok_or_else(|| anyhow!("offline AOV graph requires the normal attachment"))
    }

    pub(super) fn required_depth(&self) -> Result<&AovAttachmentTarget> {
        self.depth
            .as_ref()
            .ok_or_else(|| anyhow!("offline AOV graph requires the depth attachment"))
    }
}

fn aov_color_attachment(target: &AovAttachmentTarget) -> wgpu::RenderPassColorAttachment<'_> {
    let view = target.msaa_view.as_ref().unwrap_or(&target.internal_view);
    let resolve_target = target.msaa_view.as_ref().map(|_| &target.internal_view);
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        },
    }
}

impl TerrainScene {
    fn ensure_aov_pipeline_sample_count(
        &self,
        effective_msaa: u32,
        output_mask: u8,
        include_source_id: bool,
        clipmap_geometry: bool,
    ) -> Result<()> {
        let mut aov_pipeline = self
            .aov_pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV pipeline mutex poisoned"))?;
        let mut sample_count = self
            .aov_pipeline_sample_count
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV sample count mutex poisoned"))?;
        let mut cached_output_mask = self
            .aov_pipeline_output_mask
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV output-mask mutex poisoned"))?;
        let mut source_id_flag = self
            .aov_pipeline_source_id
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV source-id flag mutex poisoned"))?;
        let mut clipmap_flag = self
            .aov_pipeline_clipmap
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV clipmap flag mutex poisoned"))?;

        if aov_pipeline.is_none()
            || *sample_count != effective_msaa
            || *cached_output_mask != output_mask
            || *source_id_flag != include_source_id
            || *clipmap_flag != clipmap_geometry
        {
            let light_buffer = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            *aov_pipeline = Some(Self::create_aov_render_pipeline(
                self.device.as_ref(),
                &self.bind_group_layout,
                light_buffer.bind_group_layout(),
                &self.ibl_bind_group_layout,
                &self.shadow_bind_group_layout,
                &self.fog_bind_group_layout,
                &self.water_reflection_bind_group_layout,
                &self.material_layer_bind_group_layout,
                self.color_format,
                effective_msaa,
                output_mask,
                include_source_id,
                clipmap_geometry,
            ));
            *sample_count = effective_msaa;
            *cached_output_mask = output_mask;
            *source_id_flag = include_source_id;
            *clipmap_flag = clipmap_geometry;
        }

        Ok(())
    }

    /// VERITAS: single-sample R32Uint source-id attachment. Registry/ledger
    /// accounting is owned by the `TrackedTexture` RAII wrapper (freed when
    /// the owning `AovFrame` drops the texture).
    fn create_source_id_attachment_target(
        &self,
        width: u32,
        height: u32,
    ) -> Result<AovAttachmentTarget> {
        let internal_texture = tracked_create_texture(
            self.device.as_ref(),
            &wgpu::TextureDescriptor {
                label: Some("terrain.aov.source_id"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )?;
        let internal_view = internal_texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(AovAttachmentTarget {
            internal_texture,
            internal_view,
            _msaa_texture: None,
            msaa_view: None,
        })
    }

    fn create_aov_attachment_target(
        &self,
        label: &str,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Result<AovAttachmentTarget> {
        let internal_texture = tracked_create_texture(
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
                format: TERRAIN_AOV_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let internal_view = internal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let msaa_texture = if sample_count > 1 {
            Some(tracked_create_texture(
                self.device.as_ref(),
                &wgpu::TextureDescriptor {
                    label: Some(&format!("{label}.msaa")),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format: TERRAIN_AOV_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                },
            )?)
        } else {
            None
        };
        let msaa_view = msaa_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        Ok(AovAttachmentTarget {
            internal_texture,
            internal_view,
            _msaa_texture: msaa_texture,
            msaa_view,
        })
    }

    pub(super) fn create_aov_render_targets(
        &self,
        width: u32,
        height: u32,
        sample_count: u32,
        output_mask: u8,
        include_source_id: bool,
    ) -> Result<TerrainAovTargets> {
        Ok(TerrainAovTargets {
            albedo: (output_mask & AOV_ALBEDO_BIT != 0)
                .then(|| {
                    self.create_aov_attachment_target(
                        "terrain.aov.albedo",
                        width,
                        height,
                        sample_count,
                    )
                })
                .transpose()?,
            normal: (output_mask & AOV_NORMAL_BIT != 0)
                .then(|| {
                    self.create_aov_attachment_target(
                        "terrain.aov.normal",
                        width,
                        height,
                        sample_count,
                    )
                })
                .transpose()?,
            depth: (output_mask & AOV_DEPTH_BIT != 0)
                .then(|| {
                    self.create_aov_attachment_target(
                        "terrain.aov.depth",
                        width,
                        height,
                        sample_count,
                    )
                })
                .transpose()?,
            source_id: include_source_id
                .then(|| self.create_source_id_attachment_target(width, height))
                .transpose()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_main_pass_with_aov(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &crate::terrain::render_params::TerrainRenderParams,
        render_targets: &RenderTargets,
        aov_targets: &TerrainAovTargets,
        bind_group: &wgpu::BindGroup,
        ibl_bind_group: &wgpu::BindGroup,
        shadow_bind_group: &wgpu::BindGroup,
        fog_bind_group: &wgpu::BindGroup,
        water_reflection_bind_group: &wgpu::BindGroup,
        material_layer_bind_group: &wgpu::BindGroup,
        preserve_background: bool,
    ) -> Result<()> {
        let aov_pipeline_guard = self
            .aov_pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer AOV pipeline mutex poisoned"))?;
        let pipeline = aov_pipeline_guard
            .as_ref()
            .ok_or_else(|| anyhow!("terrain AOV pipeline not initialized"))?;

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

        let mut color_attachments = vec![
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
            aov_targets.albedo.as_ref().map(aov_color_attachment),
            aov_targets.normal.as_ref().map(aov_color_attachment),
            aov_targets.depth.as_ref().map(aov_color_attachment),
        ];
        // VERITAS: 5th target — cleared to zeros (SOURCE_ID_NONE) so pixels
        // no terrain fragment touches carry no attribution.
        if let Some(source_id) = aov_targets.source_id.as_ref() {
            color_attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: &source_id.internal_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }));
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.render_pass.aov"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &render_targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: if preserve_background {
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
                    "terrain_hzb_two_phase_aov",
                    "AOV rendering uses the direct clipmap path; two-phase HZB applies to the beauty pass",
                );
            }
            self.geometry_provider()?.draw(&mut pass);
        }

        Ok(())
    }

    pub(super) fn resolve_aux_output(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        internal_texture: TrackedTexture,
        internal_view: wgpu::TextureView,
        out_width: u32,
        out_height: u32,
        needs_scaling: bool,
        renormalize_normals: bool,
        label: &str,
    ) -> Result<TrackedTexture> {
        if !needs_scaling {
            return Ok(internal_texture);
        }

        let output_texture = tracked_create_texture(
            self.device.as_ref(),
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: out_width,
                    height: out_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TERRAIN_AOV_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?;
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampling = &decoded.sampling;
        let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.aov.blit.sampler"),
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
            label: Some("terrain.aov.blit.bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&internal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });
        let blit_pipeline = if renormalize_normals {
            &self.normal_blit_pipeline
        } else {
            &self.aov_blit_pipeline
        };

        {
            let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.aov.blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
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
            crate::core::shader_registry::record_shader_use(if renormalize_normals {
                "terrain.blit.normal.shader"
            } else {
                "terrain.blit.shader"
            });
            blit_pass.set_pipeline(blit_pipeline);
            blit_pass.set_bind_group(0, &blit_bind_group, &[]);
            blit_pass.draw(0..3, 0..1);
        }

        Ok(output_texture)
    }

    /// Internal render method with populated terrain AOV capture.
    /// Returns (beauty_frame, aov_frame) tuple.
    pub(crate) fn render_internal_with_aov(
        &mut self,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &render_params::TerrainRenderParams,
        heightmap: numpy::PyReadonlyArray2<'_, f32>,
        water_mask: Option<numpy::PyReadonlyArray2<'_, f32>>,
        time_seconds: f32,
    ) -> Result<(crate::Frame, crate::AovFrame)> {
        ensure_aov_shading_supported(&params.shading)?;
        let (certificate_capture, _allocation_scope) =
            self.begin_certificate_capture("terrain.render_internal_with_aov");
        let decoded = params.decoded();
        self.ensure_shadow_atlas(&decoded.shadow)?;
        let (height_height, height_width) = heightmap.as_array().dim();
        let mut declaration_uniforms = Vec::new();
        declaration_uniforms.extend_from_slice(&params.size_px.0.to_le_bytes());
        declaration_uniforms.extend_from_slice(&params.size_px.1.to_le_bytes());
        declaration_uniforms.extend_from_slice(&(height_width as u64).to_le_bytes());
        declaration_uniforms.extend_from_slice(&(height_height as u64).to_le_bytes());
        declaration_uniforms.extend_from_slice(&time_seconds.to_bits().to_le_bytes());
        declaration_uniforms.extend_from_slice(
            &params
                .terrain_data_revision
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        let mut graph = super::render_graph::build_terrain_render_graph(
            params.size_px.0,
            params.size_px.1,
            params.size_px.0,
            params.size_px.1,
            height_width as u32,
            height_height as u32,
            self.csm_renderer.allocation_size,
            self.csm_renderer.allocation_layers,
            self.color_format,
            true,
            super::render_graph::TerrainPassDeclarations {
                prepare: declaration_uniforms.clone(),
                shadow: declaration_uniforms.clone(),
                forward: declaration_uniforms.clone(),
                resolve: declaration_uniforms,
                prepared_output_size: std::mem::size_of::<crate::terrain::TerrainUniforms>() as u64,
            },
            false,
        )?
        .plan;
        debug_assert_eq!(graph.labels.len(), 4);
        let mut timing = self.take_render_timing();
        let height_inputs = graph.execute_with_barriers("terrain.prepare", |_barriers| {
            self.prepare_frame_lighting(decoded)?;
            let inputs =
                self.upload_height_inputs(heightmap, water_mask, params.terrain_data_revision)?;
            self.prepare_geometry(
                params,
                &inputs.heightmap_data,
                (inputs.width, inputs.height),
                inputs.terrain_data_hash,
            )?;
            Ok::<_, anyhow::Error>(inputs)
        })?;
        let probe_world_span = if is_mesh_camera_mode(&params.camera_mode) {
            params.terrain_span.max(1e-3)
        } else {
            1.0
        };
        super::probes::prepare_probes(
            self,
            &decoded.probes,
            probe_world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            params.z_scale,
            height_inputs.terrain_data_hash,
        )?;
        super::probes::prepare_reflection_probes(
            self,
            &decoded.reflection_probes,
            material_set,
            env_maps,
            params,
            decoded,
            probe_world_span,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            params.z_scale,
            height_inputs.terrain_data_hash,
        )?;
        let materials = self.prepare_material_context(material_set, params, decoded)?;

        let uniforms = self.build_uniforms(
            params,
            decoded,
            height_inputs.width as f32,
            height_inputs.height as f32,
        )?;
        super::runtime_contract::record_observation(
            "terrain.render_internal_with_aov",
            &uniforms,
            &materials.shading_uniforms,
            &materials.overlay_binding.uniform,
            &height_inputs.heightmap_data,
            height_inputs.width,
            height_inputs.height,
        )
        .map_err(anyhow::Error::msg)?;
        let uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.uniform_buffer"),
                contents: bytemuck::cast_slice(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let ibl_bind_group = self.prepare_ibl_bind_group(env_maps)?;
        let lut_texture_uploaded = if params.height_curve_mode.as_str() == "lut" {
            params
                .height_curve_lut
                .as_ref()
                .map(|lut| self.upload_height_curve_lut(lut.as_ref().as_slice()))
                .transpose()?
        } else {
            None
        };
        let height_curve_view = lut_texture_uploaded
            .as_ref()
            .map(|(texture, _)| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .unwrap_or_else(|| {
                self._height_curve_identity_texture
                    .create_view(&wgpu::TextureViewDescriptor::default())
            });

        let requested_msaa = params.msaa_samples.max(1);
        let effective_msaa =
            select_effective_msaa(requested_msaa, self.color_format, &self.adapter);
        if effective_msaa != requested_msaa {
            log::warn!(
                "MSAA: requested {} not supported for {:?}; using {}",
                requested_msaa,
                self.color_format,
                effective_msaa
            );
        }

        // VERITAS: the source-id map must describe exactly the emitted image —
        // R32Uint cannot be multisample-resolved and must not be rescaled, so
        // unsupported configurations are explicit errors, never silent skips.
        let aov_output_mask = requested_aov_output_mask(&decoded.aov);
        let want_source_id = decoded.aov.enabled && decoded.aov.source_id;
        if want_source_id && effective_msaa > 1 {
            return Err(anyhow!(
                "AOV source_id capture requires msaa_samples=1 (R32Uint targets cannot be multisample-resolved); got {effective_msaa}"
            ));
        }

        let needs_clipmap = is_clipmap_camera_mode(&params.camera_mode);
        self.ensure_pipeline_sample_count(effective_msaa, needs_clipmap)?;
        self.ensure_aov_pipeline_sample_count(
            effective_msaa,
            aov_output_mask,
            want_source_id,
            needs_clipmap,
        )?;
        let render_targets = self.create_render_targets(params, requested_msaa, effective_msaa)?;
        if want_source_id && render_targets.needs_scaling {
            return Err(anyhow!(
                "AOV source_id capture requires render_scale=1.0 (per-pixel attribution cannot be resampled)"
            ));
        }
        let aov_targets = self.create_aov_render_targets(
            render_targets.internal_width,
            render_targets.internal_height,
            effective_msaa,
            aov_output_mask,
            want_source_id,
        )?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("terrain.encoder.aov"),
            });
        let vt_scope = ts_begin(&mut timing, &mut encoder, "terrain.material_vt");
        let material_vt_ready = self.prepare_material_vt_frame(
            &mut encoder,
            params,
            decoded,
            materials.gpu_materials.layer_count,
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        ts_end(&mut timing, &mut encoder, vt_scope, 0);

        let ao_scope = ts_begin(&mut timing, &mut encoder, "terrain.height_ao");
        let height_ao_computed = self.compute_height_ao_pass(
            &mut encoder,
            &height_inputs.heightmap_view,
            render_targets.internal_width,
            render_targets.internal_height,
            height_inputs.width,
            height_inputs.height,
            params,
            decoded,
        )?;
        ts_end(&mut timing, &mut encoder, ao_scope, 0);

        let sun_vis_scope = ts_begin(&mut timing, &mut encoder, "terrain.sun_visibility");
        let sun_vis_computed = self.compute_sun_visibility_pass(
            &mut encoder,
            &height_inputs.heightmap_view,
            render_targets.internal_width,
            render_targets.internal_height,
            height_inputs.width,
            height_inputs.height,
            params,
            decoded,
        )?;
        ts_end(&mut timing, &mut encoder, sun_vis_scope, 0);

        let shadow_setup = graph.execute_with_barriers("terrain.shadow", |_barriers| {
            let shadow_scope = ts_begin(&mut timing, &mut encoder, "terrain.shadow");
            let setup = self.prepare_shadow_setup(
                &mut encoder,
                params,
                decoded,
                &height_inputs.heightmap_view,
                &height_curve_view,
                height_inputs.width,
                height_inputs.height,
            )?;
            ts_end(&mut timing, &mut encoder, shadow_scope, 0);
            Ok::<_, anyhow::Error>(setup)
        })?;
        let shadow_bind_group = shadow_setup
            .shadow_bind_group
            .as_ref()
            .unwrap_or(&self.noop_shadow.bind_group);
        // Keep sky/atmosphere/water aligned with the camera matrices already
        // uploaded to the main terrain pass; ShadowSetup owns cascade state.
        let (camera_eye, camera_view, camera_proj) = Self::build_camera_matrices(params);
        let camera_height = if is_zup_camera_mode(&params.camera_mode) {
            camera_eye.z
        } else {
            camera_eye.y
        };
        let sky_scope = ts_begin(&mut timing, &mut encoder, "terrain.sky");
        let sky_texture = self.render_sky_texture(
            &mut encoder,
            decoded,
            camera_view,
            camera_proj,
            camera_eye,
            render_targets.internal_width,
            render_targets.internal_height,
        )?;
        ts_end(&mut timing, &mut encoder, sky_scope, 0);
        let sky_view = sky_texture
            .as_ref()
            .map(|sky| &sky.view)
            .unwrap_or(&self.sky_fallback_view);
        let atmosphere_scattering_view = sky_texture
            .as_ref()
            .and_then(|sky| sky.scattering_view.as_ref())
            .unwrap_or(&self.atmosphere_scattering_fallback_view);

        let main_height_view = self.main_pass_height_view(&height_inputs.heightmap_view);
        let pass_bind_groups = self.create_terrain_pass_bind_groups(
            &uniform_buffer,
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
            &height_curve_view,
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
            &mut encoder,
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
            &height_curve_view,
            height_inputs.water_mask_view_uploaded.as_ref(),
            height_ao_computed,
            sun_vis_computed,
            &ibl_bind_group,
            shadow_bind_group,
            &pass_bind_groups.fog,
            &pass_bind_groups.material_layer,
        )?;

        if let Some(sky) = sky_texture.as_ref() {
            let bg_scope = ts_begin(&mut timing, &mut encoder, "terrain.background");
            self.blit_background_texture(&mut encoder, &render_targets, &sky.view, sky.linear_hdr)?;
            ts_end(&mut timing, &mut encoder, bg_scope, 1);
        }

        graph.execute_with_barriers("terrain.forward_aov", |barriers| {
            if barriers.is_empty() {
                return Err(anyhow::anyhow!(
                    "terrain.forward_aov lost its compiled shadow transition"
                ));
            }
            let main_scope = ts_begin(&mut timing, &mut encoder, "terrain.main");
            self.run_main_pass_with_aov(
                &mut encoder,
                params,
                &render_targets,
                &aov_targets,
                &pass_bind_groups.main,
                &ibl_bind_group,
                shadow_bind_group,
                &pass_bind_groups.fog,
                &water_reflection_bind_group,
                &pass_bind_groups.material_layer,
                sky_texture.is_some(),
            )?;
            ts_end(&mut timing, &mut encoder, main_scope, 1);
            Ok::<_, anyhow::Error>(())
        })?;

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
                &mut encoder,
                &render_targets,
                &height_inputs.heightmap_view,
                shadow_setup.shadow_bind_group.as_ref(),
                &scatter_state,
            )?;
        }

        let needs_scaling = render_targets.needs_scaling;
        let (
            final_texture,
            final_width,
            final_height,
            albedo_texture,
            normal_texture,
            depth_texture,
        ) = graph.execute_with_barriers("terrain.resolve_aov", |barriers| {
            if barriers.is_empty() {
                return Err(anyhow::anyhow!(
                    "terrain.resolve_aov lost its compiled beauty transition"
                ));
            }
            let resolve_scope = ts_begin(&mut timing, &mut encoder, "terrain.resolve");
            let (final_texture, final_width, final_height) =
                self.resolve_output(&mut encoder, params, decoded, &render_targets)?;

            let albedo_texture = aov_targets
                .albedo
                .map(|target| {
                    self.resolve_aux_output(
                        &mut encoder,
                        decoded,
                        target.internal_texture,
                        target.internal_view,
                        final_width,
                        final_height,
                        needs_scaling,
                        false,
                        "terrain.aov.albedo.resolved",
                    )
                })
                .transpose()?;
            let normal_texture = aov_targets
                .normal
                .map(|target| {
                    self.resolve_aux_output(
                        &mut encoder,
                        decoded,
                        target.internal_texture,
                        target.internal_view,
                        final_width,
                        final_height,
                        needs_scaling,
                        true,
                        "terrain.aov.normal.resolved",
                    )
                })
                .transpose()?;
            let depth_texture = aov_targets
                .depth
                .map(|target| {
                    self.resolve_aux_output(
                        &mut encoder,
                        decoded,
                        target.internal_texture,
                        target.internal_view,
                        final_width,
                        final_height,
                        needs_scaling,
                        false,
                        "terrain.aov.depth.resolved",
                    )
                })
                .transpose()?;
            ts_end(
                &mut timing,
                &mut encoder,
                resolve_scope,
                1 + aov_output_mask.count_ones(),
            );
            self.stage_material_vt_feedback_readback(&mut encoder)?;
            Ok::<_, anyhow::Error>((
                final_texture,
                final_width,
                final_height,
                albedo_texture,
                normal_texture,
                depth_texture,
            ))
        })?;
        if let Some(t) = timing.as_mut() {
            t.resolve_queries(&mut encoder);
        }
        self.queue.submit(Some(encoder.finish()));
        graph.finish()?;
        self.finish_material_vt_frame()?;

        self.record_render_timings(&mut timing);
        self.store_render_timing(timing);
        self.finish_certificate_capture(certificate_capture);

        let aov_frame = crate::AovFrame::new(
            self.device.clone(),
            self.queue.clone(),
            albedo_texture,
            normal_texture,
            depth_texture,
            // VERITAS: needs_scaling is rejected above, so the internal
            // texture is already at the final output dimensions.
            aov_targets.source_id.map(|target| target.internal_texture),
            final_width,
            final_height,
        );

        let beauty_frame = crate::Frame::new(
            self.device.clone(),
            self.queue.clone(),
            final_texture,
            final_width,
            final_height,
            self.color_format,
        );

        Ok((beauty_frame, aov_frame))
    }

    pub(super) fn log_color_debug(
        &self,
        _params: &render_params::TerrainRenderParams,
        binding: &OverlayBinding,
    ) {
        let debug_mode = binding.uniform.params1[1] as i32;
        let albedo_mode = match binding.uniform.params1[2] as i32 {
            0 => "material",
            1 => "colormap",
            2 => "mix",
            _ => "unknown",
        };
        let blend_mode = match binding.uniform.params1[0] as i32 {
            0 => "Replace",
            1 => "Alpha",
            2 => "Multiply",
            3 => "Additive",
            _ => "unknown",
        };

        log::info!(target: "color.debug", "╔══════════════════════════════════════════════════");
        log::info!(target: "color.debug", "║ Color Configuration Debug");
        log::info!(target: "color.debug", "╠══════════════════════════════════════════════════");
        log::info!(target: "color.debug", "║ Domain: [{}, {}]", binding.uniform.params0[0],
            binding.uniform.params0[0] + 1.0 / binding.uniform.params0[1].max(1e-6));
        log::info!(target: "color.debug", "║ Overlay Strength: {}", binding.uniform.params0[2]);
        log::info!(target: "color.debug", "║ Colormap Strength: {}", binding.uniform.params1[3]);
        log::info!(target: "color.debug", "║ Albedo Mode: {}", albedo_mode);
        log::info!(target: "color.debug", "║ Blend Mode: {}", blend_mode);
        log::info!(target: "color.debug", "║ Debug Mode: {}", debug_mode);
        log::info!(target: "color.debug", "║ Gamma: {}", binding.uniform.params2[0]);
        log::info!(target: "color.debug", "║ Roughness Mult: {}", binding.uniform.params2[1]);
        log::info!(target: "color.debug", "║ Spec AA Enabled: {}", binding.uniform.params2[2]);

        if binding.lut.is_some() {
            log::info!(target: "color.debug", "╠══════════════════════════════════════════════════");
            log::info!(target: "color.debug", "║ LUT Samples:");
            log::info!(target: "color.debug", "║   t=0.0 probe ready");
            log::info!(target: "color.debug", "║   t=0.5 probe ready");
            log::info!(target: "color.debug", "║   t=1.0 probe ready");
            log::info!(target: "color.debug", "║ LUT texture bound: yes");
        } else {
            log::info!(target: "color.debug", "║ LUT texture bound: no");
        }
        log::info!(target: "color.debug", "╚══════════════════════════════════════════════════");
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_aov_shading_supported;

    #[test]
    fn visibility_aov_is_rejected_instead_of_silently_rendering_forward() {
        ensure_aov_shading_supported("forward").expect("forward AOV must remain supported");
        let error = ensure_aov_shading_supported("visibility")
            .expect_err("visibility AOV must fail closed until it has a real resolve path");
        assert!(error
            .to_string()
            .contains("does not support shading='visibility'"));
    }
}
