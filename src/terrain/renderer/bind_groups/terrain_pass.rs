use super::*;
use crate::core::atmosphere::AETHER_RADIOMETRIC_SCALE_MAX;

pub(in crate::terrain::renderer) struct TerrainPassBindGroups {
    pub(in crate::terrain::renderer) main: wgpu::BindGroup,
    pub(in crate::terrain::renderer) fog: wgpu::BindGroup,
    pub(in crate::terrain::renderer) material_layer: wgpu::BindGroup,
}

impl TerrainScene {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn create_terrain_main_bind_group(
        &self,
        label: &'static str,
        uniform_buffer: &wgpu::Buffer,
        heightmap_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        material_sampler: &wgpu::Sampler,
        shading_buffer: &wgpu::Buffer,
        colormap_view: &wgpu::TextureView,
        colormap_sampler: &wgpu::Sampler,
        overlay_buffer: &wgpu::Buffer,
        height_curve_view: &wgpu::TextureView,
        water_mask_view_uploaded: Option<&wgpu::TextureView>,
        height_ao_computed: bool,
        sun_vis_computed: bool,
    ) -> Result<wgpu::BindGroup> {
        let height_ao_sample_guard = self
            .height_ao_sample_view
            .lock()
            .map_err(|_| anyhow!("height_ao_sample_view mutex poisoned"))?;
        let sun_vis_sample_guard = self
            .sun_vis_sample_view
            .lock()
            .map_err(|_| anyhow!("sun_vis_sample_view mutex poisoned"))?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: shading_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(colormap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(colormap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: overlay_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(height_curve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.height_curve_lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(
                        water_mask_view_uploaded.unwrap_or(&self.water_mask_fallback_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(
                        self.ao_debug_view
                            .as_ref()
                            .or(self.coarse_ao_view.as_ref())
                            .unwrap_or(&self.ao_debug_fallback_view),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(&self.detail_normal_fallback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::Sampler(&self.detail_normal_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(if height_ao_computed {
                        height_ao_sample_guard
                            .as_ref()
                            .unwrap_or(&self.height_ao_fallback_view)
                    } else {
                        &self.height_ao_fallback_view
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::Sampler(&self.height_ao_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(if sun_vis_computed {
                        sun_vis_sample_guard
                            .as_ref()
                            .unwrap_or(&self.sun_vis_fallback_view)
                    } else {
                        &self.sun_vis_fallback_view
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::Sampler(&self.sun_vis_sampler),
                },
            ],
        });
        drop(sun_vis_sample_guard);
        drop(height_ao_sample_guard);

        Ok(bind_group)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::terrain::renderer) fn create_terrain_pass_bind_groups(
        &self,
        uniform_buffer: &wgpu::Buffer,
        heightmap_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        material_sampler: &wgpu::Sampler,
        material_normal_view: &wgpu::TextureView,
        material_roughness_view: &wgpu::TextureView,
        material_mask_view: &wgpu::TextureView,
        material_map_sampler: &wgpu::Sampler,
        shading_buffer: &wgpu::Buffer,
        colormap_view: &wgpu::TextureView,
        colormap_sampler: &wgpu::Sampler,
        overlay_buffer: &wgpu::Buffer,
        height_curve_view: &wgpu::TextureView,
        water_mask_view_uploaded: Option<&wgpu::TextureView>,
        sky_view: &wgpu::TextureView,
        atmosphere_scattering_view: &wgpu::TextureView,
        height_ao_computed: bool,
        sun_vis_computed: bool,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        height_min: f32,
        height_exag: f32,
        eye_y: f32,
        material_vt_ready: bool,
    ) -> Result<TerrainPassBindGroups> {
        let main = self.create_terrain_main_bind_group(
            "terrain_pbr_pom.bind_group",
            uniform_buffer,
            heightmap_view,
            material_view,
            material_sampler,
            shading_buffer,
            colormap_view,
            colormap_sampler,
            overlay_buffer,
            height_curve_view,
            water_mask_view_uploaded,
            height_ao_computed,
            sun_vis_computed,
        )?;

        let fog_base_height = if decoded.fog.base_height <= 0.0 {
            height_min * height_exag
        } else {
            decoded.fog.base_height
        };
        let sky_enabled = decoded.sky.enabled;
        let sky_aerial_enabled = sky_enabled && decoded.sky.aerial_perspective;
        let aether_planet_lut = decoded
            .sky
            .lut_handle
            .as_ref()
            .map(|handle| {
                let config = handle.config();
                let dimensions = handle.luts().metadata.dimensions;
                [
                    config.bottom_radius_m,
                    config.top_radius_m,
                    dimensions.scattering_height as f32,
                    dimensions.scattering_nu as f32,
                ]
            })
            .unwrap_or([6_360_000.0, 6_460_000.0, 2.0, 2.0]);
        let fog_uniforms = FogUniforms {
            params0: [
                decoded.fog.density,
                decoded.fog.height_falloff,
                fog_base_height,
                eye_y,
            ],
            fog_inscatter: [
                decoded.fog.inscatter[0],
                decoded.fog.inscatter[1],
                decoded.fog.inscatter[2],
                if decoded.sky.model == 3 { 1.0 } else { 0.0 },
            ],
            sky_params0: [
                if sky_enabled { 1.0 } else { 0.0 },
                if sky_aerial_enabled {
                    decoded.sky.aerial_density.max(0.0)
                } else {
                    0.0
                },
                if sky_aerial_enabled { 1.0 } else { 0.0 },
                decoded
                    .sky
                    .sun_intensity
                    .clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX),
            ],
            sky_params1: if decoded.sky.model == 3 {
                [
                    decoded.sky.ozone_du,
                    decoded.sky.mie_g,
                    decoded.sky.turbidity,
                    decoded
                        .sky
                        .sky_exposure
                        .clamp(0.0, AETHER_RADIOMETRIC_SCALE_MAX),
                ]
            } else {
                [
                    decoded.sky.sun_size.max(0.0),
                    decoded.light.direction[2].max(0.0),
                    decoded.sky.turbidity.clamp(1.0, 10.0),
                    decoded.sky.sky_exposure.max(0.0),
                ]
            },
            aether_sun_direction: [
                decoded.light.direction[0],
                decoded.light.direction[1],
                decoded.light.direction[2],
                0.0,
            ],
            aether_planet_lut,
        };
        self.queue.write_buffer(
            &self.fog_uniform_buffer,
            0,
            bytemuck::bytes_of(&fog_uniforms),
        );
        let fog = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.atmosphere.bind_group"),
            layout: &self.fog_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.fog_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(atmosphere_scattering_view),
                },
            ],
        });

        let materials = &decoded.materials;
        let variation = &materials.variation;
        let variation_enabled = variation.snow_macro_amplitude > 0.0
            || variation.snow_detail_amplitude > 0.0
            || variation.rock_macro_amplitude > 0.0
            || variation.rock_detail_amplitude > 0.0
            || variation.wetness_macro_amplitude > 0.0
            || variation.wetness_detail_amplitude > 0.0;
        let deg_to_rad = std::f32::consts::PI / 180.0;
        let material_layer_uniforms = MaterialLayerUniforms {
            snow_params0: [
                materials.snow_altitude_min,
                materials.snow_altitude_blend,
                materials.snow_slope_max * deg_to_rad,
                materials.snow_slope_blend * deg_to_rad,
            ],
            snow_params1: [
                materials.snow_aspect_influence,
                materials.snow_roughness,
                if materials.snow_enabled { 1.0 } else { 0.0 },
                materials.snow_subsurface_strength,
            ],
            snow_color: [
                materials.snow_color[0],
                materials.snow_color[1],
                materials.snow_color[2],
                0.0,
            ],
            snow_sss_tint: [
                materials.snow_subsurface_tint[0],
                materials.snow_subsurface_tint[1],
                materials.snow_subsurface_tint[2],
                0.0,
            ],
            rock_params: [
                materials.rock_slope_min * deg_to_rad,
                materials.rock_slope_blend * deg_to_rad,
                materials.rock_roughness,
                if materials.rock_enabled { 1.0 } else { 0.0 },
            ],
            rock_color: [
                materials.rock_color[0],
                materials.rock_color[1],
                materials.rock_color[2],
                materials.rock_subsurface_strength,
            ],
            rock_sss_tint: [
                materials.rock_subsurface_tint[0],
                materials.rock_subsurface_tint[1],
                materials.rock_subsurface_tint[2],
                0.0,
            ],
            wetness_params: [
                materials.wetness_strength,
                materials.wetness_slope_influence,
                if materials.wetness_enabled { 1.0 } else { 0.0 },
                materials.wetness_subsurface_strength,
            ],
            wetness_sss_tint: [
                materials.wetness_subsurface_tint[0],
                materials.wetness_subsurface_tint[1],
                materials.wetness_subsurface_tint[2],
                0.0,
            ],
            variation_params0: [
                variation.macro_scale,
                variation.detail_scale,
                variation.octaves.clamp(1, 8) as f32,
                if variation_enabled { 1.0 } else { 0.0 },
            ],
            snow_variation: [
                variation.snow_macro_amplitude,
                variation.snow_detail_amplitude,
                0.0,
                0.0,
            ],
            rock_variation: [
                variation.rock_macro_amplitude,
                variation.rock_detail_amplitude,
                0.0,
                0.0,
            ],
            wetness_variation: [
                variation.wetness_macro_amplitude,
                variation.wetness_detail_amplitude,
                0.0,
                0.0,
            ],
            map_flags: [
                if materials.normal_path.is_some() {
                    1.0
                } else {
                    0.0
                },
                if materials.roughness_path.is_some() {
                    1.0
                } else {
                    0.0
                },
                if materials.mask_path.is_some() {
                    1.0
                } else {
                    0.0
                },
                0.0,
            ],
        };
        self.queue.write_buffer(
            &self.material_layer_uniform_buffer,
            0,
            bytemuck::bytes_of(&material_layer_uniforms),
        );
        let build_material_layer_bind_group =
            |atlas_views: &[wgpu::TextureView],
             page_table_view: &wgpu::TextureView,
             feedback_buffer: &wgpu::Buffer| {
                let atlas_view_refs = atlas_views.iter().collect::<Vec<_>>();
                let atlas_resource =
                    if super::super::virtual_texture::bindless_bc_supported(self.device.as_ref()) {
                        wgpu::BindingResource::TextureViewArray(&atlas_view_refs)
                    } else {
                        wgpu::BindingResource::TextureView(
                            atlas_views
                                .first()
                                .expect("VT fallback always has an atlas view"),
                        )
                    };
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain.material_layer.bind_group"),
                    layout: &self.material_layer_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.material_layer_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.probe_grid_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.probe_ssbo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self
                                .reflection_probe_grid_uniform_buffer
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(
                                &self.reflection_probe_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(
                                &self.reflection_probe_sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.vt_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.vt_fallback_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: atlas_resource,
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::Sampler(&self.vt_atlas_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::TextureView(page_table_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: feedback_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: wgpu::BindingResource::TextureView(material_normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: wgpu::BindingResource::TextureView(material_roughness_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: wgpu::BindingResource::TextureView(material_mask_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 15,
                            resource: wgpu::BindingResource::Sampler(material_map_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: self.vt_frame_counters_buffer.as_entire_binding(),
                        },
                    ],
                })
            };

        let material_layer = if material_vt_ready {
            let material_vt = self
                .material_vt
                .lock()
                .map_err(|_| anyhow!("material_vt mutex poisoned"))?;
            if let Some(bindings) = material_vt.binding_resources() {
                build_material_layer_bind_group(
                    bindings.atlas_views,
                    bindings.page_table_view,
                    bindings
                        .feedback_buffer
                        .unwrap_or(&self.vt_feedback_fallback_buffer),
                )
            } else {
                build_material_layer_bind_group(
                    &self.vt_atlas_fallback_views,
                    &self.vt_page_table_fallback_view,
                    &self.vt_feedback_fallback_buffer,
                )
            }
        } else {
            build_material_layer_bind_group(
                &self.vt_atlas_fallback_views,
                &self.vt_page_table_fallback_view,
                &self.vt_feedback_fallback_buffer,
            )
        };

        Ok(TerrainPassBindGroups {
            main,
            fog,
            material_layer,
        })
    }
}
