use super::*;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::terrain::renderer::core::TERRAIN_DEPTH_FORMAT;

pub(in crate::terrain::renderer) struct RenderTargets {
    pub(in crate::terrain::renderer) internal_texture: Arc<TrackedTexture>,
    pub(in crate::terrain::renderer) internal_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) _msaa_texture: Option<Arc<TrackedTexture>>,
    pub(in crate::terrain::renderer) msaa_view: Option<wgpu::TextureView>,
    pub(in crate::terrain::renderer) _depth_texture: Arc<TrackedTexture>,
    pub(in crate::terrain::renderer) depth_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) resolved_texture: Arc<TrackedTexture>,
    pub(in crate::terrain::renderer) resolved_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) out_width: u32,
    pub(in crate::terrain::renderer) out_height: u32,
    pub(in crate::terrain::renderer) internal_width: u32,
    pub(in crate::terrain::renderer) internal_height: u32,
    pub(in crate::terrain::renderer) needs_scaling: bool,
    pub(in crate::terrain::renderer) sample_count: u32,
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn prepare_frame_lighting(
        &self,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
    ) -> Result<()> {
        let override_lights = self
            .light_override
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer light override mutex poisoned"))?
            .clone();
        let mut light_buffer_guard = self
            .light_buffer
            .lock()
            .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
        light_buffer_guard.next_frame();

        let lights = if let Some(lights) = override_lights {
            lights
        } else if decoded.light.intensity > 0.0 {
            vec![Light {
                kind: LightType::Directional.as_u32(),
                intensity: decoded.light.intensity,
                range: 0.0,
                env_texture_index: 0,
                color: decoded.light.color,
                _pad1: 0.0,
                pos_ws: [0.0; 3],
                _pad2: 0.0,
                dir_ws: decoded.light.direction,
                _pad3: 0.0,
                cone_cos: [1.0, 1.0],
                area_half: [0.0, 0.0],
            }]
        } else {
            vec![Light {
                kind: LightType::Directional.as_u32(),
                intensity: 0.0,
                range: 0.0,
                env_texture_index: 0,
                color: [1.0, 1.0, 1.0],
                _pad1: 0.0,
                pos_ws: [0.0; 3],
                _pad2: 0.0,
                dir_ws: [0.0, 1.0, 0.0],
                _pad3: 0.0,
                cone_cos: [1.0, 1.0],
                area_half: [0.0, 0.0],
            }]
        };

        light_buffer_guard
            .update(self.device.as_ref(), self.queue.as_ref(), &lights)
            .map_err(|e| anyhow!("Failed to update light buffer: {}", e))?;
        Ok(())
    }

    pub(in crate::terrain::renderer) fn ensure_pipeline_sample_count(
        &self,
        effective_msaa: u32,
        needs_clipmap: bool,
    ) -> Result<()> {
        let mut pipeline_cache = self
            .pipeline
            .lock()
            .map_err(|_| anyhow!("TerrainRenderer pipeline mutex poisoned"))?;
        if pipeline_cache.sample_count != effective_msaa {
            let light_buffer = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            pipeline_cache.pipeline = Self::create_render_pipeline(
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
            );
            // Invalidate so the clipmap variant is rebuilt at the new sample
            // count on next use.
            pipeline_cache.clipmap_pipeline = None;
            pipeline_cache.visibility_write_pipeline = None;
            pipeline_cache.visibility_resolve_pipeline = None;
            pipeline_cache.sample_count = effective_msaa;
        }
        if needs_clipmap && pipeline_cache.clipmap_pipeline.is_none() {
            let light_buffer = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            pipeline_cache.clipmap_pipeline = Some(Self::create_clipmap_render_pipeline(
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
            ));
        }
        if needs_clipmap
            && effective_msaa == 1
            && pipeline_cache.visibility_write_pipeline.is_none()
        {
            let light_buffer = self
                .light_buffer
                .lock()
                .map_err(|_| anyhow!("Light buffer mutex poisoned"))?;
            pipeline_cache.visibility_write_pipeline =
                Some(Self::create_clipmap_visibility_write_pipeline(
                    self.device.as_ref(),
                    &self.bind_group_layout,
                ));
            pipeline_cache.visibility_resolve_pipeline =
                Some(Self::create_clipmap_visibility_resolve_pipeline(
                    self.device.as_ref(),
                    &self.bind_group_layout,
                    light_buffer.bind_group_layout(),
                    &self.ibl_bind_group_layout,
                    &self.shadow_bind_group_layout,
                    &self.fog_bind_group_layout,
                    &self.water_reflection_bind_group_layout,
                    &self.material_layer_bind_group_layout,
                    &self.visibility_resolve_bind_group_layout,
                    self.color_format,
                ));
        }
        Ok(())
    }

    pub(in crate::terrain::renderer) fn create_render_targets(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        requested_msaa: u32,
        effective_msaa: u32,
    ) -> Result<RenderTargets> {
        self.create_render_targets_for_format(
            params,
            requested_msaa,
            effective_msaa,
            self.color_format,
        )
    }

    pub(in crate::terrain::renderer) fn create_render_targets_for_format(
        &self,
        params: &crate::terrain::render_params::TerrainRenderParams,
        requested_msaa: u32,
        effective_msaa: u32,
        color_format: wgpu::TextureFormat,
    ) -> Result<RenderTargets> {
        let (out_width, out_height) = params.size_px;
        let render_scale = params.render_scale.clamp(0.25, 4.0);
        let internal_width = ((out_width as f32 * render_scale).round().max(1.0)) as u32;
        let internal_height = ((out_height as f32 * render_scale).round().max(1.0)) as u32;
        let needs_scaling = internal_width != out_width || internal_height != out_height;

        let internal_texture = Arc::new(tracked_create_texture(
            self.device.as_ref(),
            &wgpu::TextureDescriptor {
                label: Some("terrain.internal.render_target"),
                size: wgpu::Extent3d {
                    width: internal_width,
                    height: internal_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )?);
        let internal_view = internal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let msaa_texture = if effective_msaa > 1 {
            Some(Arc::new(tracked_create_texture(
                self.device.as_ref(),
                &wgpu::TextureDescriptor {
                    label: Some("terrain.msaa.render_target"),
                    size: wgpu::Extent3d {
                        width: internal_width,
                        height: internal_height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: effective_msaa,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                },
            )?))
        } else {
            None
        };
        let msaa_view = msaa_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let depth_texture = Arc::new(tracked_create_texture(
            self.device.as_ref(),
            &wgpu::TextureDescriptor {
                label: Some("terrain.depth.render_target"),
                size: wgpu::Extent3d {
                    width: internal_width,
                    height: internal_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: effective_msaa,
                dimension: wgpu::TextureDimension::D2,
                format: TERRAIN_DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | if effective_msaa == 1 {
                        wgpu::TextureUsages::TEXTURE_BINDING
                    } else {
                        wgpu::TextureUsages::empty()
                    },
                view_formats: &[],
            },
        )?);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let resolved_texture = if needs_scaling {
            Arc::new(tracked_create_texture(
                self.device.as_ref(),
                &wgpu::TextureDescriptor {
                    label: Some("terrain.output.resolved"),
                    size: wgpu::Extent3d {
                        width: out_width,
                        height: out_height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )?)
        } else {
            internal_texture.clone()
        };
        let resolved_view = resolved_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let color_attachment_sample_count = if effective_msaa > 1 {
            effective_msaa
        } else {
            1
        };
        let resolve_sample_count = if effective_msaa > 1 { Some(1) } else { None };

        log_msaa_debug(
            &self.adapter,
            color_format,
            Some(TERRAIN_DEPTH_FORMAT),
            requested_msaa,
            effective_msaa,
            color_attachment_sample_count,
            resolve_sample_count,
            Some(effective_msaa),
            effective_msaa,
        );

        let invariants = MsaaInvariants {
            effective_msaa,
            pipeline_sample_count: effective_msaa,
            color_attachment_sample_count,
            has_resolve_target: effective_msaa > 1,
            resolve_sample_count,
            depth_sample_count: Some(effective_msaa),
            readback_sample_count: 1,
        };
        assert_msaa_invariants(&invariants, color_format)?;

        Ok(RenderTargets {
            internal_texture,
            internal_view,
            _msaa_texture: msaa_texture,
            msaa_view,
            _depth_texture: depth_texture,
            depth_view,
            resolved_texture,
            resolved_view,
            out_width,
            out_height,
            internal_width,
            internal_height,
            needs_scaling,
            sample_count: effective_msaa,
        })
    }
}
