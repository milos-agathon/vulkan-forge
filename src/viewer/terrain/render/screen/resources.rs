use super::ScreenRenderFlags;
use crate::core::resource_tracker::tracked_create_texture;
use crate::viewer::terrain::ViewerTerrainScene;

impl ViewerTerrainScene {
    pub(super) fn prepare_screen_resources(
        &mut self,
        width: u32,
        height: u32,
    ) -> anyhow::Result<ScreenRenderFlags> {
        self.ensure_depth(width, height)?;

        let needs_canonical_media = self.canonical_media.is_some();
        let needs_pbr = self.pbr_config.enabled || needs_canonical_media;
        let scene_format = self.scene_color_format();
        if needs_pbr && self.pbr_pipeline.is_none() {
            if let Err(e) = self.init_pbr_pipeline(scene_format) {
                if needs_canonical_media {
                    return Err(e);
                }
                eprintln!("[render] Failed to initialize PBR pipeline: {}", e);
            }
        }

        if self.pbr_config.enabled
            && (self.pbr_config.height_ao.enabled || self.pbr_config.sun_visibility.enabled)
        {
            if let Err(e) = self.init_heightfield_compute_pipelines() {
                eprintln!(
                    "[render] Failed to initialize heightfield compute pipelines: {}",
                    e
                );
            }
        }

        let use_pbr = needs_pbr && self.pbr_pipeline.is_some();
        let needs_taa = self
            .taa_renderer
            .as_ref()
            .is_some_and(crate::core::taa::TaaRenderer::is_enabled);
        let needs_dof = self.pbr_config.dof.enabled;
        let needs_post_process = self.pbr_config.lens_effects.enabled
            && (self.pbr_config.lens_effects.distortion.abs() > 0.001
                || self.pbr_config.lens_effects.chromatic_aberration > 0.001
                || self.pbr_config.lens_effects.vignette_strength > 0.001);
        let needs_volumetrics =
            self.canonical_media.is_none() && self.pbr_config.volumetrics.is_effectively_enabled();
        let denoise_requested = self.pbr_config.denoise.enabled;
        let needs_denoise = false;
        let needs_dof_scratch = needs_volumetrics && needs_post_process && !needs_dof;

        if denoise_requested {
            static WARN_SCREEN_DENOISE_DISABLED: std::sync::Once = std::sync::Once::new();
            WARN_SCREEN_DENOISE_DISABLED.call_once(|| {
                eprintln!(
                    "[terrain] Screen-space denoise is currently disabled on the viewer path; skipping interactive denoise"
                );
            });
        }

        if let Some(taa) = self.taa_renderer.as_mut() {
            taa.resize(&self.device, width, height)?;
        }
        if needs_taa {
            self.ensure_taa_velocity_texture(width, height)?;
        }

        if (needs_taa || needs_post_process || needs_volumetrics || needs_canonical_media)
            && self.post_process.is_none()
        {
            self.init_post_process();
        }
        if (needs_dof || needs_dof_scratch) && self.dof_pass.is_none() {
            self.init_dof_pass();
        }
        if needs_volumetrics && self.volumetrics_pass.is_none() {
            self.init_volumetrics_pass();
        }
        if needs_denoise && self.denoise_pass.is_none() {
            self.init_denoise_pass();
        }

        if needs_denoise {
            if let Some(ref mut denoise) = self.denoise_pass {
                let _ = denoise.get_input_view(width, height);
            }
        }
        if needs_dof || needs_dof_scratch {
            if let Some(ref mut dof) = self.dof_pass {
                let _ = dof.get_input_view(width, height, scene_format);
            }
        }

        let has_vector_overlays_early = self
            .vector_overlay_stack
            .as_ref()
            .map(|s| s.visible_layer_count() > 0)
            .unwrap_or(false);
        if has_vector_overlays_early && self.oit_enabled {
            if self.wboit_compose_bind_group.is_none() || self.wboit_size != (width, height) {
                self.init_wboit(width, height);
            }
        }

        if needs_taa || needs_post_process || needs_volumetrics || needs_canonical_media {
            if let Some(ref mut pp) = self.post_process {
                let _ = pp.get_intermediate_view(width, height, scene_format);
            }
        }

        Ok(ScreenRenderFlags {
            use_pbr,
            needs_taa,
            needs_dof,
            needs_post_process,
            needs_volumetrics,
            needs_canonical_media,
            needs_denoise,
        })
    }

    fn ensure_taa_velocity_texture(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if self.taa_velocity_view.is_some() && self.taa_velocity_size == (width, height) {
            return Ok(());
        }

        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain_taa.zero_velocity"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;

        let bytes_per_pixel = 4u32;
        let row_bytes = width.saturating_mul(bytes_per_pixel);
        let padded_row_bytes = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let data = vec![0u8; padded_row_bytes as usize * height as usize];
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
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.taa_velocity_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.taa_velocity_texture = Some(texture);
        self.taa_velocity_size = (width, height);
        Ok(())
    }
}
