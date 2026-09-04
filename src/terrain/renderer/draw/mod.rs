use super::*;
use crate::core::anamnesis::{
    leaf_key, EngineFingerprint, GraphPassAction, GraphPassOutcome, GraphScheduler,
};
use crate::core::framegraph_impl::{RendererPassContext, ResourceHandle};
use crate::core::resource_tracker::tracked_create_buffer_init;
use std::collections::BTreeMap;
use std::io::Error as IoError;

mod execute;
mod setup;

pub(in crate::terrain::renderer) use setup::{
    PreparedMaterials, RenderTargets, UploadedHeightInputs,
};

fn io_error(error: impl std::fmt::Display) -> IoError {
    IoError::other(error.to_string())
}

fn restore_texture(
    context: &RendererPassContext<'_>,
    handle: ResourceHandle,
    payload: &[u8],
    width: u32,
    height: u32,
    layers: u32,
    aspect: wgpu::TextureAspect,
) -> Result<()> {
    context.queue().write_texture(
        wgpu::ImageCopyTexture {
            texture: context.texture(handle)?,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        payload,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
    );
    Ok(())
}

impl TerrainScene {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn render_internal(
        &mut self,
        material_set: &crate::render::material_set::MaterialSet,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
        params: &crate::terrain::render_params::TerrainRenderParams,
        heightmap: PyReadonlyArray2<f32>,
        water_mask: Option<PyReadonlyArray2<f32>>,
        time_seconds: f32,
        cache: Option<&super::anamnesis::TerrainCacheOptions>,
    ) -> Result<(crate::Frame, crate::core::anamnesis::CacheReport, usize)> {
        let (certificate_capture, _allocation_scope) =
            self.begin_certificate_capture("terrain.render_internal");
        let mut timing = self.take_render_timing();
        let decoded = params.decoded();
        self.ensure_shadow_atlas(&decoded.shadow)?;

        self.prepare_frame_lighting(decoded)?;
        let height_inputs =
            self.upload_height_inputs(heightmap, water_mask, params.terrain_data_revision)?;
        self.prepare_geometry(
            params,
            &height_inputs.heightmap_data,
            (height_inputs.width, height_inputs.height),
            height_inputs.terrain_data_hash,
        )?;
        let probe_world_span = if is_mesh_camera_mode(&params.camera_mode)
            || is_clipmap_camera_mode(&params.camera_mode)
        {
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
            "terrain.render_internal",
            &uniforms,
            &materials.shading_uniforms,
            &materials.overlay_binding.uniform,
            &height_inputs.heightmap_data,
            height_inputs.width,
            height_inputs.height,
        )
        .map_err(anyhow::Error::msg)?;
        let prepared_bytes = bytemuck::cast_slice(&uniforms).to_vec();
        let uniform_buffer = Arc::new(tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.uniform_buffer"),
                contents: &prepared_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?);
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
        self.ensure_pipeline_sample_count(
            effective_msaa,
            is_clipmap_camera_mode(&params.camera_mode),
        )?;
        let render_targets = self.create_render_targets(params, requested_msaa, effective_msaa)?;

        let declaration_uniforms = cache
            .map(|options| options.state_bytes.clone())
            .unwrap_or_else(|| {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&params.size_px.0.to_le_bytes());
                bytes.extend_from_slice(&params.size_px.1.to_le_bytes());
                bytes.extend_from_slice(&height_inputs.width.to_le_bytes());
                bytes.extend_from_slice(&height_inputs.height.to_le_bytes());
                bytes.extend_from_slice(&time_seconds.to_bits().to_le_bytes());
                bytes
            });
        let mut prepare_declaration = declaration_uniforms.clone();
        prepare_declaration.extend_from_slice(&(prepared_bytes.len() as u64).to_le_bytes());
        prepare_declaration.extend_from_slice(&prepared_bytes);
        let mut resolve_declaration = Vec::new();
        resolve_declaration.extend_from_slice(&params.size_px.0.to_le_bytes());
        resolve_declaration.extend_from_slice(&params.size_px.1.to_le_bytes());
        resolve_declaration.extend_from_slice(&render_targets.internal_width.to_le_bytes());
        resolve_declaration.extend_from_slice(&render_targets.internal_height.to_le_bytes());
        resolve_declaration.extend_from_slice(&params.render_scale.to_bits().to_le_bytes());
        let shadow_width = self.csm_renderer.allocation_size;
        let shadow_layers = decoded
            .shadow
            .cascades
            .max(1)
            .min(self.csm_renderer.allocation_layers);
        let mut shadow_declaration = cache
            .map(|options| options.shadow_bytes.clone())
            .unwrap_or_else(|| declaration_uniforms.clone());
        shadow_declaration.extend_from_slice(&shadow_width.to_le_bytes());
        shadow_declaration.extend_from_slice(&shadow_layers.to_le_bytes());
        shadow_declaration.push(u8::from(decoded.shadow.enabled));
        shadow_declaration
            .extend_from_slice(&(decoded.shadow.technique.len() as u32).to_le_bytes());
        shadow_declaration.extend_from_slice(decoded.shadow.technique.as_bytes());
        let graph_bundle = super::render_graph::build_terrain_render_graph(
            params.size_px.0,
            params.size_px.1,
            render_targets.internal_width,
            render_targets.internal_height,
            height_inputs.width,
            height_inputs.height,
            shadow_width,
            shadow_layers,
            self.color_format,
            false,
            super::render_graph::TerrainPassDeclarations {
                prepare: prepare_declaration,
                shadow: shadow_declaration,
                forward: declaration_uniforms,
                resolve: resolve_declaration,
                prepared_output_size: prepared_bytes.len() as u64,
            },
            cache.is_some(),
        )?;
        debug_assert_eq!(graph_bundle.plan.labels.len(), 4);
        let handles = graph_bundle.handles;
        let scheduler_plan = graph_bundle.plan.clone();
        let mut execution = graph_bundle
            .plan
            .begin_execution(self.device.clone(), self.queue.clone());
        execution.bind_texture(handles.height, height_inputs.heightmap_texture.clone())?;
        execution.bind_buffer(handles.prepared, uniform_buffer.clone())?;
        execution.bind_texture(handles.shadow, self.csm_renderer.shadow_maps.clone())?;
        execution.bind_texture(handles.beauty, render_targets.internal_texture.clone())?;
        execution.bind_texture(handles.resolved, render_targets.resolved_texture.clone())?;

        let mut scheduler = match cache {
            Some(options) => GraphScheduler::new(
                options.store().map_err(io_error)?,
                options.capability_bytes.clone(),
                options.engine_bytes.clone(),
            ),
            None => {
                GraphScheduler::disabled(Vec::new(), EngineFingerprint::current().canonical_bytes())
            }
        };
        let source_key = cache
            .map(|options| leaf_key(&options.height_bytes))
            .unwrap_or_else(|| leaf_key(bytemuck::cast_slice(&height_inputs.heightmap_data)));
        let leaf_keys = BTreeMap::from([(handles.height, source_key)]);
        let mut shadow_setup = None;
        let mut material_vt_started = false;
        let mut timing_needs_resolve = false;
        let mut hzb_frame_staged = false;
        let mut visibility_frame_staged = false;

        scheduler
            .execute_graph_with(&scheduler_plan, &leaf_keys, |pass, action| {
                let result: Result<GraphPassOutcome> = (|| match (pass.name.as_str(), action) {
                    ("terrain.prepare", GraphPassAction::Execute { capture_output, .. }) => {
                        execution
                            .run_pass("terrain.prepare", |_context| Ok::<_, anyhow::Error>(()))?;
                        Ok(GraphPassOutcome::Executed(if capture_output {
                            super::anamnesis::encode_blob(
                                super::anamnesis::BlobKind::Prepared,
                                prepared_bytes.len() as u32,
                                1,
                                1,
                                &prepared_bytes,
                            )
                        } else {
                            Vec::new()
                        }))
                    }
                    ("terrain.prepare", GraphPassAction::Restore { blob, .. }) => {
                        let payload = super::anamnesis::require_blob(
                            blob,
                            super::anamnesis::BlobKind::Prepared,
                            prepared_bytes.len() as u32,
                            1,
                            1,
                        )?;
                        execution.run_pass("terrain.prepare", |context| {
                            context.queue().write_buffer(
                                context.buffer(handles.prepared)?,
                                0,
                                payload,
                            );
                            Ok::<_, anyhow::Error>(())
                        })?;
                        Ok(GraphPassOutcome::Restored)
                    }
                    ("terrain.shadow", GraphPassAction::Execute { capture_output, .. }) => {
                        let setup = execution.run_pass("terrain.shadow", |context| {
                            let scope = ts_begin(&mut timing, context.encoder(), "terrain.shadow");
                            let setup = self.prepare_shadow_setup(
                                context.encoder(),
                                params,
                                decoded,
                                &height_inputs.heightmap_view,
                                &height_curve_view,
                                height_inputs.width,
                                height_inputs.height,
                            )?;
                            ts_end(&mut timing, context.encoder(), scope, 0);
                            Ok::<_, anyhow::Error>(setup)
                        })?;
                        timing_needs_resolve = true;
                        shadow_setup = Some(setup);
                        let payload = if capture_output && decoded.shadow.enabled {
                            let depth = execution.read_texture_tight(
                                handles.shadow,
                                shadow_width,
                                shadow_width,
                                shadow_layers,
                                4,
                                wgpu::TextureAspect::DepthOnly,
                            )?;
                            super::anamnesis::encode_depth_rle(&depth)?
                        } else {
                            Vec::new()
                        };
                        Ok(GraphPassOutcome::Executed(if capture_output {
                            super::anamnesis::encode_blob(
                                super::anamnesis::BlobKind::ShadowDepth32,
                                shadow_width,
                                shadow_width,
                                if decoded.shadow.enabled {
                                    shadow_layers
                                } else {
                                    0
                                },
                                &payload,
                            )
                        } else {
                            Vec::new()
                        }))
                    }
                    ("terrain.shadow", GraphPassAction::Restore { blob, .. }) => {
                        let layers = if decoded.shadow.enabled {
                            shadow_layers
                        } else {
                            0
                        };
                        let payload = super::anamnesis::require_blob(
                            blob,
                            super::anamnesis::BlobKind::ShadowDepth32,
                            shadow_width,
                            shadow_width,
                            layers,
                        )?;
                        let restored_depth = if decoded.shadow.enabled {
                            Some(super::anamnesis::decode_depth_rle(
                                payload,
                                shadow_width as usize
                                    * shadow_width as usize
                                    * shadow_layers as usize
                                    * 4,
                            )?)
                        } else {
                            None
                        };
                        let setup = execution.run_pass("terrain.shadow", |context| {
                            let setup = self.prepare_shadow_setup(
                                context.encoder(),
                                params,
                                decoded,
                                &height_inputs.heightmap_view,
                                &height_curve_view,
                                height_inputs.width,
                                height_inputs.height,
                            )?;
                            context.discard_encoder();
                            if decoded.shadow.enabled {
                                let restored_depth =
                                    restored_depth.as_deref().ok_or_else(|| {
                                        anyhow!(
                                            "enabled shadow restoration has no decoded depth bytes"
                                        )
                                    })?;
                                restore_texture(
                                    context,
                                    handles.shadow,
                                    restored_depth,
                                    shadow_width,
                                    shadow_width,
                                    shadow_layers,
                                    wgpu::TextureAspect::DepthOnly,
                                )?;
                            }
                            Ok::<_, anyhow::Error>(setup)
                        })?;
                        shadow_setup = Some(setup);
                        Ok(GraphPassOutcome::Restored)
                    }
                    (
                        "terrain.forward",
                        GraphPassAction::Execute {
                            barriers,
                            capture_output,
                        },
                    ) => {
                        if barriers.is_empty() {
                            return Err(anyhow!(
                                "terrain.forward lost its compiled shadow transition"
                            ));
                        }
                        let setup = shadow_setup.as_ref().ok_or_else(|| {
                            anyhow!("terrain.forward has no materialized shadow state")
                        })?;
                        execution.run_pass("terrain.forward", |context| {
                            let encoder = context.encoder();
                            let vt_scope = ts_begin(&mut timing, encoder, "terrain.material_vt");
                            let material_vt_ready = self.prepare_material_vt_frame(
                                encoder,
                                params,
                                decoded,
                                materials.gpu_materials.layer_count,
                                render_targets.internal_width,
                                render_targets.internal_height,
                            )?;
                            material_vt_started = true;
                            ts_end(&mut timing, encoder, vt_scope, 0);
                            let ao_scope = ts_begin(&mut timing, encoder, "terrain.height_ao");
                            let height_ao = self.compute_height_ao_pass(
                                encoder,
                                &height_inputs.heightmap_view,
                                render_targets.internal_width,
                                render_targets.internal_height,
                                height_inputs.width,
                                height_inputs.height,
                                params,
                                decoded,
                            )?;
                            ts_end(&mut timing, encoder, ao_scope, 0);
                            let sun_scope =
                                ts_begin(&mut timing, encoder, "terrain.sun_visibility");
                            let sun_vis = self.compute_sun_visibility_pass(
                                encoder,
                                &height_inputs.heightmap_view,
                                render_targets.internal_width,
                                render_targets.internal_height,
                                height_inputs.width,
                                height_inputs.height,
                                params,
                                decoded,
                            )?;
                            ts_end(&mut timing, encoder, sun_scope, 0);
                            self.encode_forward_pass(
                                encoder,
                                params,
                                decoded,
                                &height_inputs,
                                &materials,
                                &uniform_buffer,
                                &ibl_bind_group,
                                &height_curve_view,
                                &render_targets,
                                setup,
                                material_vt_ready,
                                height_ao,
                                sun_vis,
                                time_seconds,
                                &mut timing,
                            )?;
                            hzb_frame_staged = params.culling == "hzb_two_phase"
                                && render_targets.sample_count == 1
                                && self.two_phase_culler.is_some();
                            visibility_frame_staged = is_clipmap_camera_mode(&params.camera_mode)
                                && render_targets.sample_count == 1;
                            Ok::<_, anyhow::Error>(())
                        })?;
                        timing_needs_resolve = true;
                        Ok(GraphPassOutcome::Executed(if capture_output {
                            let payload = execution.read_texture_tight(
                                handles.beauty,
                                render_targets.internal_width,
                                render_targets.internal_height,
                                1,
                                4,
                                wgpu::TextureAspect::All,
                            )?;
                            super::anamnesis::encode_blob(
                                super::anamnesis::BlobKind::Rgba8,
                                render_targets.internal_width,
                                render_targets.internal_height,
                                1,
                                &payload,
                            )
                        } else {
                            Vec::new()
                        }))
                    }
                    ("terrain.forward", GraphPassAction::Restore { blob, barriers }) => {
                        if barriers.is_empty() {
                            return Err(anyhow!(
                                "terrain.forward restore lost its compiled shadow transition"
                            ));
                        }
                        let payload = super::anamnesis::require_blob(
                            blob,
                            super::anamnesis::BlobKind::Rgba8,
                            render_targets.internal_width,
                            render_targets.internal_height,
                            1,
                        )?;
                        execution.run_pass("terrain.forward", |context| {
                            restore_texture(
                                context,
                                handles.beauty,
                                payload,
                                render_targets.internal_width,
                                render_targets.internal_height,
                                1,
                                wgpu::TextureAspect::All,
                            )
                        })?;
                        Ok(GraphPassOutcome::Restored)
                    }
                    (
                        "terrain.resolve",
                        GraphPassAction::Execute {
                            barriers,
                            capture_output,
                        },
                    ) => {
                        if barriers.is_empty() {
                            return Err(anyhow!(
                                "terrain.resolve lost its compiled beauty transition"
                            ));
                        }
                        execution.run_pass("terrain.resolve", |context| {
                            let encoder = context.encoder();
                            let scope = ts_begin(&mut timing, encoder, "terrain.resolve");
                            self.resolve_output(encoder, params, decoded, &render_targets)?;
                            ts_end(&mut timing, encoder, scope, 1);
                            if material_vt_started {
                                self.stage_material_vt_feedback_readback(encoder)?;
                            }
                            if let Some(timing) = timing.as_mut() {
                                timing.resolve_queries(encoder);
                            }
                            Ok::<_, anyhow::Error>(())
                        })?;
                        timing_needs_resolve = false;
                        Ok(GraphPassOutcome::Executed(if capture_output {
                            let payload = execution.read_texture_tight(
                                handles.resolved,
                                render_targets.out_width,
                                render_targets.out_height,
                                1,
                                4,
                                wgpu::TextureAspect::All,
                            )?;
                            super::anamnesis::encode_blob(
                                super::anamnesis::BlobKind::Rgba8,
                                render_targets.out_width,
                                render_targets.out_height,
                                1,
                                &payload,
                            )
                        } else {
                            Vec::new()
                        }))
                    }
                    ("terrain.resolve", GraphPassAction::Restore { blob, barriers }) => {
                        if barriers.is_empty() {
                            return Err(anyhow!(
                                "terrain.resolve restore lost its compiled beauty transition"
                            ));
                        }
                        let payload = super::anamnesis::require_blob(
                            blob,
                            super::anamnesis::BlobKind::Rgba8,
                            render_targets.out_width,
                            render_targets.out_height,
                            1,
                        )?;
                        execution.run_pass("terrain.resolve", |context| {
                            restore_texture(
                                context,
                                handles.resolved,
                                payload,
                                render_targets.out_width,
                                render_targets.out_height,
                                1,
                                wgpu::TextureAspect::All,
                            )?;
                            if timing_needs_resolve {
                                if let Some(timing) = timing.as_mut() {
                                    timing.resolve_queries(context.encoder());
                                }
                            }
                            Ok::<_, anyhow::Error>(())
                        })?;
                        timing_needs_resolve = false;
                        Ok(GraphPassOutcome::Restored)
                    }
                    (label, _) => Err(anyhow!("unexpected native terrain graph pass {label:?}")),
                })();
                result.map_err(io_error)
            })
            .map_err(io_error)?;
        let submitted = execution.finish()?;
        debug_assert!(
            cache.is_none() || !scheduler.report().hits.is_empty() || submitted > 0,
            "cold native terrain graph must submit production work"
        );
        if visibility_frame_staged && params.shading == "visibility" && params.culling == "frustum"
        {
            self.refresh_cpu_visibility_oracle_from_gpu_selection(
                params,
                &height_inputs.heightmap_data,
                (height_inputs.width, height_inputs.height),
            )?;
        }
        if material_vt_started {
            self.finish_material_vt_frame()?;
        }
        if hzb_frame_staged {
            let (_, view, proj) = Self::build_camera_matrices(params);
            if let Some(culler) = self.two_phase_culler.as_mut() {
                self.culling_stats = culler.finish_frame(self.device.as_ref(), proj * view)?;
                crate::terrain::culling::two_phase::publish_stats(self.culling_stats);
            }
        } else {
            self.culling_stats = crate::terrain::culling::two_phase::CullingStats::default();
            crate::terrain::culling::two_phase::publish_stats(self.culling_stats);
        }
        if visibility_frame_staged {
            self.finish_visibility_frame()?;
        } else {
            super::visibility_buffer::publish_stats(Default::default());
        }
        self.record_render_timings(&mut timing);
        self.store_render_timing(timing);
        self.finish_certificate_capture(certificate_capture);
        let report = scheduler.into_report();
        let frame = crate::Frame::new(
            self.device.clone(),
            self.queue.clone(),
            render_targets.resolved_texture.clone(),
            render_targets.out_width,
            render_targets.out_height,
            self.color_format,
        );
        Ok((frame, report, submitted))
    }
}
