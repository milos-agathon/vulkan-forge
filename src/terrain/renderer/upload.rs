use super::*;
use crate::core::resource_tracker::{tracked_create_texture, TrackedTexture};
use crate::terrain::render_params;

impl TerrainScene {
    pub(super) fn extract_overlay_binding(
        &self,
        params: &render_params::TerrainRenderParams,
        offline_hdr_output: bool,
    ) -> OverlayBinding {
        let overlays = params.overlays();
        let env_debug_mode = std::env::var("VF_COLOR_DEBUG_MODE")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        let debug_mode = if params.debug_mode > 0 {
            params.debug_mode as i32
        } else {
            env_debug_mode
        };
        let debug_mode_f = if (100..=113).contains(&debug_mode)
            || (40..=42).contains(&debug_mode)
            || (50..=53).contains(&debug_mode)
        {
            debug_mode as f32
        } else {
            debug_mode.clamp(0, 33) as f32
        };
        if debug_mode != 0 {
            info!(
                "debug_mode: params={}, env={}, resolved={}",
                params.debug_mode, env_debug_mode, debug_mode_f
            );
        }

        let roughness_mult = std::env::var("VF_ROUGHNESS_MULT")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0)
            .max(0.001);

        let spec_aa_enabled = std::env::var("VF_SPEC_AA_ENABLED")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);

        let specaa_sigma_scale = std::env::var("VF_SPECAA_SIGMA_SCALE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0)
            .max(0.0);

        let albedo_mode_f = match params.albedo_mode() {
            "material" => 0.0,
            "colormap" => 1.0,
            "mix" => 2.0,
            _ => 2.0,
        };

        let colormap_strength = params.colormap_strength().clamp(0.0, 1.0);
        let gamma = params.gamma().max(0.1);
        let ao_weight = params.ao_weight.clamp(0.0, 1.0);
        let ao_fallback_enabled = if self.coarse_ao_view.is_some() {
            1.0
        } else {
            0.0
        };

        let decoded = params.decoded();
        let detail = &decoded.detail;
        let detail_enabled = if detail.enabled { 1.0 } else { 0.0 };
        let detail_scale = detail.detail_scale.max(0.1);
        let detail_normal_strength = detail.normal_strength.clamp(0.0, 1.0);
        let detail_albedo_noise = detail.albedo_noise.clamp(0.0, 0.5);
        let detail_fade_start = detail.fade_start.max(0.0);
        let detail_fade_end = detail.fade_end.max(detail_fade_start + 1.0);
        // AETHER's sky blit uses the exact IEC sRGB transfer. Force the same
        // presentation contract for terrain whenever the spectral sky is
        // active so a single frame cannot mix exact sRGB and legacy gamma 2.2.
        let output_srgb_eotf =
            if params.output_srgb_eotf || (decoded.sky.enabled && decoded.sky.model == 3) {
                1.0
            } else {
                0.0
            };
        let offline_hdr_flag = if offline_hdr_output { 1.0 } else { 0.0 };

        let mut binding = OverlayBinding {
            uniform: OverlayUniforms {
                params0: [0.0; 4],
                params1: [0.0, debug_mode_f, albedo_mode_f, colormap_strength],
                params2: [gamma, roughness_mult, spec_aa_enabled, specaa_sigma_scale],
                params3: [
                    ao_weight,
                    ao_fallback_enabled,
                    params.hue_variation_strength.clamp(0.0, 0.2),
                    0.0,
                ],
                params4: [
                    detail_enabled,
                    detail_scale,
                    detail_normal_strength,
                    detail_albedo_noise,
                ],
                params5: [
                    detail_fade_start,
                    detail_fade_end,
                    output_srgb_eotf,
                    offline_hdr_flag,
                ],
            },
            lut: None,
        };

        pyo3::Python::with_gil(|py| {
            for overlay_py in overlays {
                let overlay_ref = overlay_py.borrow(py);
                if let Some(colormap) = overlay_ref.colormap_clone() {
                    let domain = overlay_ref.domain_tuple();
                    let range = domain.1 - domain.0;
                    let inv_range = if range.abs() > f32::EPSILON {
                        1.0 / range
                    } else {
                        0.0
                    };
                    let strength = overlay_ref.strength_value().max(0.0);
                    let offset = overlay_ref.offset();
                    let mode_value = match overlay_ref.blend_mode().to_ascii_lowercase().as_str() {
                        "replace" => 0.0,
                        "alpha" => 1.0,
                        "multiply" | "mul" => 2.0,
                        "add" | "additive" => 3.0,
                        _ => 1.0,
                    };

                    binding.uniform.params0 = [domain.0, inv_range, strength, offset];
                    binding.uniform.params1[0] = mode_value;
                    binding.lut = Some(colormap.lut.clone());
                    break;
                }
            }
        });

        binding
    }

    pub(super) fn upload_heightmap_texture(
        &self,
        width: u32,
        height: u32,
        data: &[f32],
    ) -> Result<TrackedTexture> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.heightmap"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
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
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size,
        );

        Ok(texture)
    }

    pub(super) fn upload_water_mask_texture(
        &self,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<TrackedTexture> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.water_mask"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
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
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            size,
        );

        Ok(texture)
    }

    pub(super) fn upload_height_curve_lut_internal(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[f32],
    ) -> Result<(TrackedTexture, wgpu::TextureView)> {
        let width = 256u32;
        let height = 1u32;
        if data.len() != width as usize {
            return Err(anyhow!("height_curve_lut must have length 256"));
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = tracked_create_texture(
            device,
            &wgpu::TextureDescriptor {
                label: Some("terrain.height_curve_lut"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok((texture, view))
    }

    pub(super) fn upload_height_curve_lut(
        &self,
        data: &[f32],
    ) -> Result<(TrackedTexture, wgpu::TextureView)> {
        Self::upload_height_curve_lut_internal(&self.device, &self.queue, data)
    }

    pub(super) fn build_uniforms(
        &self,
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
        _terrain_width: f32,
        _terrain_height: f32,
    ) -> Result<Vec<f32>> {
        let (_eye, view, proj) = Self::build_camera_matrices(params);
        Ok(Self::build_uniforms_with_matrices(
            params, decoded, view, proj,
        ))
    }

    pub(super) fn build_uniforms_with_matrices(
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
        view: glam::Mat4,
        proj: glam::Mat4,
    ) -> Vec<f32> {
        let mut uniforms = Vec::with_capacity(48);
        uniforms.extend_from_slice(&view.to_cols_array());
        uniforms.extend_from_slice(&proj.to_cols_array());
        uniforms.extend_from_slice(&[
            decoded.light.direction[0],
            decoded.light.direction[1],
            decoded.light.direction[2],
            decoded.light.intensity,
        ]);

        let is_mesh_mode = is_mesh_camera_mode(&params.camera_mode);
        let is_clipmap_mode = is_clipmap_camera_mode(&params.camera_mode);
        let spacing = if is_mesh_mode || is_clipmap_mode {
            params.terrain_span.max(1e-3)
        } else {
            1.0
        };
        uniforms.extend_from_slice(&[spacing, spacing, params.z_scale, params.render_scale]);

        let camera_mode = if is_mesh_mode || is_clipmap_mode {
            1.0
        } else {
            0.0
        };
        let grid_size = clipmap_camera_config(&params.camera_mode)
            .map(|config| config.ring_resolution as f32)
            .unwrap_or(512.0);
        uniforms.extend_from_slice(&[camera_mode, grid_size, params.clip.0, params.clip.1]);

        uniforms
    }

    pub(super) fn build_camera_matrices(
        params: &render_params::TerrainRenderParams,
    ) -> (glam::Vec3, glam::Mat4, glam::Mat4) {
        let phi_rad = params.cam_phi_deg.to_radians();
        let theta_rad = params.cam_theta_deg.to_radians();

        let (eye_offset, up) = if is_zup_camera_mode(&params.camera_mode) {
            // Z-up orbit for terrain that lives in the XY plane with heights
            // along +Z (mesh mode): theta stays "0 looks straight down", phi
            // is the azimuth within the terrain plane. Near-vertical views
            // fall back to +Y for `up` because look_at is degenerate when the
            // view direction is parallel to the up vector.
            let offset = glam::Vec3::new(
                params.cam_radius * theta_rad.sin() * phi_rad.cos(),
                params.cam_radius * theta_rad.sin() * phi_rad.sin(),
                params.cam_radius * theta_rad.cos(),
            );
            let up = if theta_rad.sin().abs() < 1e-4 {
                glam::Vec3::Y
            } else {
                glam::Vec3::Z
            };
            (offset, up)
        } else {
            (
                glam::Vec3::new(
                    params.cam_radius * theta_rad.sin() * phi_rad.cos(),
                    params.cam_radius * theta_rad.cos(),
                    params.cam_radius * theta_rad.sin() * phi_rad.sin(),
                ),
                glam::Vec3::Y,
            )
        };

        let target = glam::Vec3::from_array(params.cam_target);
        let eye = target + eye_offset;
        let view = glam::Mat4::look_at_rh(eye, target, up);
        let aspect = params.size_px.0 as f32 / params.size_px.1 as f32;
        let proj = glam::Mat4::perspective_rh(
            params.fov_y_deg.to_radians(),
            aspect,
            params.clip.0,
            params.clip.1,
        );
        (eye, view, proj)
    }

    pub(super) fn build_shading_uniforms(
        &self,
        material_set: &crate::render::material_set::MaterialSet,
        gpu_materials: &crate::render::material_set::GpuMaterialSet,
        params: &render_params::TerrainRenderParams,
        decoded: &render_params::DecodedTerrainSettings,
    ) -> Result<Vec<f32>> {
        let pom_flags = {
            let mut flags = 0u32;
            if decoded.pom.enabled {
                flags |= 1;
                if decoded.pom.occlusion {
                    flags |= 1 << 1;
                }
                if decoded.pom.shadow {
                    flags |= 1 << 2;
                }
            }
            flags
        };

        let (pom_min_steps, pom_max_steps, pom_refine_steps) = if decoded.pom.enabled {
            (
                decoded.pom.min_steps as f32,
                decoded.pom.max_steps as f32,
                decoded.pom.refine_steps as f32,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let mut uniforms = Vec::with_capacity(44);
        uniforms.extend_from_slice(&[
            decoded.triplanar.scale,
            decoded.triplanar.blend_sharpness,
            decoded.triplanar.normal_strength,
            if decoded.pom.enabled {
                decoded.pom.scale
            } else {
                0.0
            },
        ]);
        uniforms.extend_from_slice(&[
            pom_min_steps,
            pom_max_steps,
            pom_refine_steps,
            pom_flags as f32,
        ]);

        let layer_centers = gpu_materials.layer_centers();
        uniforms.extend_from_slice(&layer_centers);

        let mut layer_roughness = [1.0f32; MATERIAL_LAYER_CAPACITY];
        let mut layer_metallic = [0.0f32; MATERIAL_LAYER_CAPACITY];
        let active_layers = gpu_materials.layer_count as usize;
        let clamp_layers = active_layers.clamp(1, MATERIAL_LAYER_CAPACITY);
        for (idx, material) in material_set
            .materials()
            .iter()
            .enumerate()
            .take(clamp_layers)
        {
            layer_roughness[idx] = material.roughness;
            layer_metallic[idx] = material.metallic;
        }
        uniforms.extend_from_slice(&layer_roughness);
        uniforms.extend_from_slice(&layer_metallic);

        let layer_count_f = gpu_materials.layer_count.max(1) as f32;
        let blend_half = if layer_count_f <= 1.0 {
            1.0
        } else {
            f32::max(0.5 / layer_count_f, 0.05)
        };
        uniforms.extend_from_slice(&[
            layer_count_f,
            blend_half,
            decoded.lod.bias,
            decoded.lod.lod0_bias,
        ]);

        let light_intensity = decoded.light.intensity;
        uniforms.extend_from_slice(&[
            decoded.light.color[0] * light_intensity,
            decoded.light.color[1] * light_intensity,
            decoded.light.color[2] * light_intensity,
            params.exposure,
        ]);
        uniforms.extend_from_slice(&[
            decoded.clamp.height_range.0,
            decoded.clamp.height_range.1,
            decoded.clamp.slope_range.0,
            decoded.clamp.slope_range.1,
        ]);
        uniforms.extend_from_slice(&[
            decoded.clamp.ambient_range.0,
            decoded.clamp.ambient_range.1,
            decoded.clamp.shadow_range.0,
            decoded.clamp.shadow_range.1,
        ]);
        uniforms.extend_from_slice(&[
            decoded.clamp.occlusion_range.0,
            decoded.clamp.occlusion_range.1,
            decoded.lod.level as f32,
            decoded.sampling.anisotropy as f32,
        ]);

        let mode_f = match params.height_curve_mode.as_str() {
            "linear" => 0.0,
            "pow" => 1.0,
            "smoothstep" => 2.0,
            "lut" => 3.0,
            _ => 0.0,
        };
        let strength = params.height_curve_strength.clamp(0.0, 1.0);
        let power = params.height_curve_power.max(0.01);
        let lambert_k = params.lambert_contrast.clamp(0.0, 1.0);
        uniforms.extend_from_slice(&[mode_f, strength, power, lambert_k]);

        Ok(uniforms)
    }
}
