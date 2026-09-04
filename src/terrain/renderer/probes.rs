use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use half::f16;
use pyo3::Python;

use super::*;
use crate::core::resource_tracker::{tracked_create_buffer_init, tracked_create_texture};
use crate::render::material_set::MaterialSet;
use crate::terrain::probes::{
    pack_probes_for_upload, GpuProbeData, HeightfieldAnalyticalBaker, HeightfieldReflectionBaker,
    ProbeBaker, ProbeGridDesc, ProbeGridUniformsGpu, ProbePlacement, ReflectionAlbedoMode,
    ReflectionCaptureLighting, ReflectionOverlay, ReflectionOverlayBlend,
    ReflectionProbeGridUniformsGpu, ReflectionProbeSet, ReflectionTerrainMaterial,
};
use crate::terrain::render_params::{
    DecodedTerrainSettings, ProbeSettingsNative, ReflectionProbeSettingsNative, TerrainRenderParams,
};

fn hash_probe_bake_inputs(
    settings: &ProbeSettingsNative,
    terrain_span: f32,
    z_scale: f32,
    terrain_data_hash: u64,
    height_dims: (u32, u32),
) -> u64 {
    let mut hasher = DefaultHasher::new();
    terrain_span.to_bits().hash(&mut hasher);
    z_scale.to_bits().hash(&mut hasher);
    settings.grid_dims.hash(&mut hasher);
    settings
        .origin
        .map(|(x, y)| (x.to_bits(), y.to_bits()))
        .hash(&mut hasher);
    settings
        .spacing
        .map(|(x, y)| (x.to_bits(), y.to_bits()))
        .hash(&mut hasher);
    settings.height_offset.to_bits().hash(&mut hasher);
    settings.ray_count.hash(&mut hasher);
    settings
        .fallback_blend_distance
        .map(f32::to_bits)
        .hash(&mut hasher);
    settings
        .sky_color
        .map(f32::to_bits)
        .into_iter()
        .for_each(|bits| bits.hash(&mut hasher));
    settings.sky_intensity.to_bits().hash(&mut hasher);
    height_dims.hash(&mut hasher);
    terrain_data_hash.hash(&mut hasher);
    hasher.finish()
}

fn hash_reflection_probe_inputs(
    settings: &ReflectionProbeSettingsNative,
    terrain_span: f32,
    z_scale: f32,
    terrain_data_hash: u64,
    height_dims: (u32, u32),
    env_maps: &crate::lighting::ibl_wrapper::IBL,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    terrain_span.to_bits().hash(&mut hasher);
    z_scale.to_bits().hash(&mut hasher);
    settings.grid_dims.hash(&mut hasher);
    settings
        .origin
        .map(|(x, y)| (x.to_bits(), y.to_bits()))
        .hash(&mut hasher);
    settings
        .spacing
        .map(|(x, y)| (x.to_bits(), y.to_bits()))
        .hash(&mut hasher);
    settings.height_offset.to_bits().hash(&mut hasher);
    settings.resolution.hash(&mut hasher);
    settings.ray_count.hash(&mut hasher);
    settings.trace_steps.hash(&mut hasher);
    settings.trace_refine_steps.hash(&mut hasher);
    height_dims.hash(&mut hasher);
    terrain_data_hash.hash(&mut hasher);
    env_maps.environment_path.hash(&mut hasher);
    env_maps.intensity.to_bits().hash(&mut hasher);
    env_maps.rotation_deg.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn sample_height_for_placement(
    heightfield: &[f32],
    height_dims: (u32, u32),
    terrain_span: f32,
    world_x: f32,
    world_y: f32,
) -> f32 {
    let (width, height) = height_dims;
    if width == 0 || height == 0 || heightfield.is_empty() {
        return 0.0;
    }
    if width == 1 || height == 1 {
        return if heightfield[0].is_finite() {
            heightfield[0]
        } else {
            0.0
        };
    }

    let u = ((world_x / terrain_span) + 0.5).clamp(0.0, 1.0);
    let v = ((world_y / terrain_span) + 0.5).clamp(0.0, 1.0);
    let fx = u * (width - 1) as f32;
    let fy = v * (height - 1) as f32;
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let sample = |x: u32, y: u32| {
        let value = heightfield[(y * width + x) as usize];
        value.is_finite().then_some(value)
    };
    let samples = [
        ((1.0 - tx) * (1.0 - ty), sample(x0, y0)),
        (tx * (1.0 - ty), sample(x1, y0)),
        ((1.0 - tx) * ty, sample(x0, y1)),
        (tx * ty, sample(x1, y1)),
    ];

    let mut sum = 0.0;
    let mut weight = 0.0;
    for (wgt, value) in samples {
        if let Some(value) = value {
            sum += value * wgt;
            weight += wgt;
        }
    }
    if weight > 0.0 {
        sum / weight
    } else {
        0.0
    }
}

pub(super) fn resolve_placement_from_grid(
    grid_dims: (u32, u32),
    origin: Option<(f32, f32)>,
    spacing: Option<(f32, f32)>,
    height_offset: f32,
    terrain_span: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
) -> ProbePlacement {
    let cols = grid_dims.0.max(1);
    let rows = grid_dims.1.max(1);
    let half_span = terrain_span * 0.5;

    let auto_spacing_x = if cols > 1 {
        terrain_span / (cols - 1) as f32
    } else {
        terrain_span
    };
    let auto_spacing_y = if rows > 1 {
        terrain_span / (rows - 1) as f32
    } else {
        terrain_span
    };
    let auto_origin_x = if cols > 1 { -half_span } else { 0.0 };
    let auto_origin_y = if rows > 1 { -half_span } else { 0.0 };

    let origin = origin.unwrap_or((auto_origin_x, auto_origin_y));
    let spacing = spacing.unwrap_or((auto_spacing_x, auto_spacing_y));

    let grid = ProbeGridDesc {
        origin: [origin.0, origin.1],
        spacing: [spacing.0, spacing.1],
        dims: [cols, rows],
        height_offset,
        influence_radius: 0.0,
    };

    let positions_ws = (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                let wx = grid.origin[0] + grid.spacing[0] * col as f32;
                let wy = grid.origin[1] + grid.spacing[1] * row as f32;
                let wz =
                    sample_height_for_placement(heightfield, height_dims, terrain_span, wx, wy)
                        * z_scale
                        + height_offset;
                [wx, wy, wz]
            })
        })
        .collect();

    ProbePlacement::new(grid, positions_ws)
}

fn axis_blend_distances(spacing: [f32; 2], blend_distance: Option<f32>) -> (f32, f32) {
    if let Some(distance) = blend_distance {
        let clamped = distance.max(1e-6);
        (clamped, clamped)
    } else {
        ((spacing[0] * 2.0).max(1e-6), (spacing[1] * 2.0).max(1e-6))
    }
}

fn reflection_axis_blend_distances(
    spacing: [f32; 2],
    blend_distance: Option<(f32, f32)>,
) -> (f32, f32) {
    if let Some((x, y)) = blend_distance {
        (x.max(1e-6), y.max(1e-6))
    } else {
        ((spacing[0] * 2.0).max(1e-6), (spacing[1] * 2.0).max(1e-6))
    }
}

fn parse_hex_rgb(hex: &str) -> Option<[f32; 3]> {
    let trimmed = hex.trim().trim_start_matches('#');
    if trimmed.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&trimmed[0..2], 16).ok()? as f32 / 255.0;
    let g = u8::from_str_radix(&trimmed[2..4], 16).ok()? as f32 / 255.0;
    let b = u8::from_str_radix(&trimmed[4..6], 16).ok()? as f32 / 255.0;
    Some([r, g, b])
}

fn extract_reflection_overlay(params: &TerrainRenderParams) -> Option<ReflectionOverlay> {
    Python::with_gil(|py| {
        if let Some(overlay_py) = params.overlays().into_iter().next() {
            let overlay_ref = overlay_py.borrow(py);
            let colormap = overlay_ref.colormap_clone()?;
            let mut stops = Vec::with_capacity(colormap.stops.len());
            for (value, color_hex) in &colormap.stops {
                let color = parse_hex_rgb(color_hex)?;
                stops.push((*value, color));
            }
            let blend_mode = match overlay_ref.blend_mode().to_ascii_lowercase().as_str() {
                "replace" => ReflectionOverlayBlend::Replace,
                "add" | "additive" => ReflectionOverlayBlend::Add,
                "multiply" | "mul" => ReflectionOverlayBlend::Multiply,
                _ => ReflectionOverlayBlend::Alpha,
            };
            return Some(ReflectionOverlay {
                domain: overlay_ref.domain_tuple(),
                strength: overlay_ref.strength_value().max(0.0),
                offset: overlay_ref.offset(),
                blend_mode,
                stops,
            });
        }
        None
    })
}

fn resolve_material_color(
    material_set: &MaterialSet,
    index: usize,
    fallback: [f32; 3],
) -> [f32; 3] {
    material_set
        .get_material(index)
        .map(|material| {
            [
                material.base_color[0],
                material.base_color[1],
                material.base_color[2],
            ]
        })
        .unwrap_or(fallback)
}

fn build_reflection_material(
    material_set: &MaterialSet,
    params: &TerrainRenderParams,
    decoded: &DecodedTerrainSettings,
) -> ReflectionTerrainMaterial {
    let albedo_mode = match params.albedo_mode.as_str() {
        "material" => ReflectionAlbedoMode::Material,
        "colormap" => ReflectionAlbedoMode::Colormap,
        _ => ReflectionAlbedoMode::Mix,
    };
    let materials = &decoded.materials;
    ReflectionTerrainMaterial {
        albedo_mode,
        colormap_strength: params.colormap_strength.clamp(0.0, 1.0),
        raw_height_range: decoded.clamp.height_range,
        overlay: extract_reflection_overlay(params),
        grass_color: resolve_material_color(material_set, 1, [0.18, 0.38, 0.10]),
        dirt_color: resolve_material_color(material_set, 2, [0.35, 0.25, 0.15]),
        rock_color: resolve_material_color(material_set, 0, materials.rock_color),
        snow_color: resolve_material_color(material_set, 3, materials.snow_color),
        snow_enabled: materials.snow_enabled,
        snow_altitude_min: materials.snow_altitude_min,
        snow_altitude_blend: materials.snow_altitude_blend,
        snow_slope_max_deg: materials.snow_slope_max,
        snow_slope_blend_deg: materials.snow_slope_blend,
        snow_aspect_influence: materials.snow_aspect_influence,
        rock_enabled: materials.rock_enabled,
        rock_slope_min_deg: materials.rock_slope_min,
        rock_slope_blend_deg: materials.rock_slope_blend,
        wetness_enabled: materials.wetness_enabled,
        wetness_strength: materials.wetness_strength,
        wetness_slope_influence: materials.wetness_slope_influence,
    }
}

fn compute_height_bounds(heightfield: &[f32], z_scale: f32) -> (f32, f32) {
    let mut min_height = f32::INFINITY;
    let mut max_height = f32::NEG_INFINITY;
    for value in heightfield {
        if value.is_finite() {
            let scaled = *value * z_scale;
            min_height = min_height.min(scaled);
            max_height = max_height.max(scaled);
        }
    }
    if !min_height.is_finite() || !max_height.is_finite() {
        (0.0, 0.0)
    } else {
        (min_height, max_height)
    }
}

fn reflection_texture_bytes(probe_count: u32, resolution: u32, mip_levels: u32) -> u64 {
    let mut texel_count = 0u64;
    for mip in 0..mip_levels {
        let size = (resolution >> mip).max(1) as u64;
        texel_count += size * size;
    }
    texel_count * probe_count as u64 * 6 * 8
}

impl TerrainScene {
    pub(super) fn upload_probe_data(
        &mut self,
        grid_uniforms: &ProbeGridUniformsGpu,
        probe_data: &[GpuProbeData],
        active_probe_count: usize,
    ) -> Result<()> {
        let required_bytes = (probe_data.len() * std::mem::size_of::<GpuProbeData>()) as u64;
        if self.probe_ssbo_alloc_bytes != required_bytes {
            let tracker = crate::core::memory_tracker::global_tracker();
            if self.probe_ssbo_alloc_bytes > 0 {
                tracker.free_buffer_allocation(self.probe_ssbo_alloc_bytes, false);
            }
            self.probe_ssbo = tracked_create_buffer_init(
                &self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("terrain.probes.ssbo"),
                    contents: bytemuck::cast_slice(probe_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            )?;
            tracker.track_buffer_allocation(required_bytes, false)?;
            self.probe_ssbo_alloc_bytes = required_bytes;
        } else {
            self.queue
                .write_buffer(&self.probe_ssbo, 0, bytemuck::cast_slice(probe_data));
        }

        self.queue.write_buffer(
            &self.probe_grid_uniform_buffer,
            0,
            bytemuck::bytes_of(grid_uniforms),
        );

        self.probe_grid_uniform_bytes = if active_probe_count > 0 {
            std::mem::size_of::<ProbeGridUniformsGpu>() as u64
        } else {
            0
        };
        self.probe_ssbo_bytes = (active_probe_count * std::mem::size_of::<GpuProbeData>()) as u64;
        Ok(())
    }

    pub(super) fn clear_reflection_probe_texture(&mut self) {
        self.reflection_probe_texture = None;
        self.reflection_probe_view =
            self.reflection_probe_fallback_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("terrain.reflection_probes.active_view"),
                    format: Some(wgpu::TextureFormat::Rgba16Float),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                });
        self.reflection_probe_texture_alloc_bytes = 0;
        self.reflection_probe_texture_bytes = 0;
        self.reflection_probe_count = 0;
        self.reflection_probe_resolution = 0;
        self.reflection_probe_mip_levels = 0;
    }

    fn upload_reflection_probe_texture(&mut self, baked: &ReflectionProbeSet) -> Result<()> {
        if baked.probes.is_empty() {
            self.clear_reflection_probe_texture();
            return Ok(());
        }

        let probe_count = baked.probes.len() as u32;
        let resolution = baked.resolution.max(1);
        let mip_levels = baked.mip_level_count.max(1);
        let required_bytes = reflection_texture_bytes(probe_count, resolution, mip_levels);
        let needs_recreate = self.reflection_probe_texture.is_none()
            || self.reflection_probe_count != probe_count
            || self.reflection_probe_resolution != resolution
            || self.reflection_probe_mip_levels != mip_levels;

        if needs_recreate {
            let texture = tracked_create_texture(
                &self.device,
                &wgpu::TextureDescriptor {
                    label: Some("terrain.reflection_probes.texture"),
                    size: wgpu::Extent3d {
                        width: resolution,
                        height: resolution,
                        depth_or_array_layers: probe_count * 6,
                    },
                    mip_level_count: mip_levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )?;
            self.reflection_probe_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain.reflection_probes.view"),
                format: Some(wgpu::TextureFormat::Rgba16Float),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(mip_levels),
                base_array_layer: 0,
                array_layer_count: Some(probe_count * 6),
            });
            self.reflection_probe_texture = Some(texture);
        }

        let texture = self
            .reflection_probe_texture
            .as_ref()
            .expect("reflection probe texture should exist");
        for (probe_index, probe) in baked.probes.iter().enumerate() {
            for (mip_index, mip) in probe.mips.iter().enumerate() {
                let bytes_per_row = mip.size * 8;
                let rows_per_image = mip.size;
                for face_index in 0..6usize {
                    let face_texels = mip.face_texels(face_index);
                    let mut packed = Vec::with_capacity(face_texels.len() * 4);
                    for texel in face_texels {
                        packed.push(f16::from_f32(texel[0]).to_bits());
                        packed.push(f16::from_f32(texel[1]).to_bits());
                        packed.push(f16::from_f32(texel[2]).to_bits());
                        packed.push(f16::from_f32(texel[3]).to_bits());
                    }
                    self.queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture,
                            mip_level: mip_index as u32,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: probe_index as u32 * 6 + face_index as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytemuck::cast_slice(&packed),
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(bytes_per_row),
                            rows_per_image: Some(rows_per_image),
                        },
                        wgpu::Extent3d {
                            width: mip.size,
                            height: mip.size,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }

        self.reflection_probe_texture_alloc_bytes = required_bytes;
        self.reflection_probe_texture_bytes = required_bytes;
        self.reflection_probe_count = probe_count;
        self.reflection_probe_resolution = resolution;
        self.reflection_probe_mip_levels = mip_levels;
        Ok(())
    }
}

pub(super) fn prepare_probes(
    scene: &mut TerrainScene,
    settings: &ProbeSettingsNative,
    terrain_span: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
    terrain_data_hash: u64,
) -> Result<()> {
    if !settings.enabled {
        scene.upload_probe_data(
            &ProbeGridUniformsGpu::disabled(),
            &[GpuProbeData::zeroed()],
            0,
        )?;
        return Ok(());
    }

    let bake_key = hash_probe_bake_inputs(
        settings,
        terrain_span,
        z_scale,
        terrain_data_hash,
        height_dims,
    );
    if scene.probe_cache_key != Some(bake_key)
        || scene.probe_cached_grid.is_none()
        || scene.probe_cached_data.is_empty()
    {
        let placement = resolve_placement_from_grid(
            settings.grid_dims,
            settings.origin,
            settings.spacing,
            settings.height_offset,
            terrain_span,
            heightfield,
            height_dims,
            z_scale,
        );
        let scaled_heightfield = heightfield
            .iter()
            .map(|value| {
                if value.is_finite() {
                    *value * z_scale
                } else {
                    *value
                }
            })
            .collect();
        let baker = HeightfieldAnalyticalBaker {
            heightfield: scaled_heightfield,
            height_dims,
            terrain_span: [terrain_span, terrain_span],
            sky_color: settings.sky_color,
            sky_intensity: settings.sky_intensity,
            ray_count: settings.ray_count.max(1),
            max_trace_distance: terrain_span,
        };
        match baker.bake(&placement) {
            Ok(irradiance) => {
                scene.probe_cache_key = Some(bake_key);
                scene.probe_cached_grid = Some(placement.grid.clone());
                scene.probe_cached_data = pack_probes_for_upload(&irradiance);
            }
            Err(error) => {
                log::warn!(target: "terrain.probes", "Failed to bake irradiance probes: {error}");
                scene.probe_cache_key = None;
                scene.probe_cached_grid = None;
                scene.probe_cached_data.clear();
                scene.upload_probe_data(
                    &ProbeGridUniformsGpu::disabled(),
                    &[GpuProbeData::zeroed()],
                    0,
                )?;
                return Ok(());
            }
        }
    }

    let grid = match scene.probe_cached_grid.clone() {
        Some(grid) if !scene.probe_cached_data.is_empty() => grid,
        _ => {
            log::warn!(
                target: "terrain.probes",
                "Probe cache missing grid data after bake; disabling probes for this frame"
            );
            scene.probe_cache_key = None;
            scene.probe_cached_grid = None;
            scene.probe_cached_data.clear();
            scene.upload_probe_data(
                &ProbeGridUniformsGpu::disabled(),
                &[GpuProbeData::zeroed()],
                0,
            )?;
            return Ok(());
        }
    };
    let probe_count = scene.probe_cached_data.len();
    let (blend_x, blend_y) = axis_blend_distances(grid.spacing, settings.fallback_blend_distance);
    let uniforms = ProbeGridUniformsGpu {
        grid_origin: [grid.origin[0], grid.origin[1], grid.height_offset, 1.0],
        grid_params: [
            grid.spacing[0],
            grid.spacing[1],
            grid.dims[0] as f32,
            grid.dims[1] as f32,
        ],
        blend_params: [blend_x, blend_y, 1.0, probe_count as f32],
    };
    let gpu_data = scene.probe_cached_data.clone();
    scene.upload_probe_data(&uniforms, &gpu_data, probe_count)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_reflection_probes(
    scene: &mut TerrainScene,
    settings: &ReflectionProbeSettingsNative,
    material_set: &MaterialSet,
    env_maps: &crate::lighting::ibl_wrapper::IBL,
    params: &TerrainRenderParams,
    decoded: &DecodedTerrainSettings,
    terrain_span: f32,
    heightfield: &[f32],
    height_dims: (u32, u32),
    z_scale: f32,
    terrain_data_hash: u64,
) -> Result<()> {
    if !settings.enabled {
        scene.clear_reflection_probe_texture();
        scene.queue.write_buffer(
            &scene.reflection_probe_grid_uniform_buffer,
            0,
            bytemuck::bytes_of(&ReflectionProbeGridUniformsGpu::disabled()),
        );
        scene.reflection_probe_grid_uniform_bytes = 0;
        return Ok(());
    }

    let bake_key = hash_reflection_probe_inputs(
        settings,
        terrain_span,
        z_scale,
        terrain_data_hash,
        height_dims,
        env_maps,
    );
    if scene.reflection_probe_cache_key != Some(bake_key)
        || scene.reflection_probe_cached_grid.is_none()
        || scene.reflection_probe_texture.is_none()
    {
        let placement = resolve_placement_from_grid(
            settings.grid_dims,
            settings.origin,
            settings.spacing,
            settings.height_offset,
            terrain_span,
            heightfield,
            height_dims,
            z_scale,
        );
        let material = build_reflection_material(material_set, params, decoded);
        let baker = HeightfieldReflectionBaker {
            heightfield,
            height_dims,
            terrain_span: [terrain_span, terrain_span],
            z_scale,
            resolution: settings.resolution.max(4),
            prefilter_sample_count: settings.ray_count.max(1),
            trace_steps: settings.trace_steps.max(8),
            trace_refine_steps: settings.trace_refine_steps,
            max_trace_distance: terrain_span * 1.5,
            material,
            lighting: ReflectionCaptureLighting {
                env_image: env_maps.hdr_image.as_deref(),
                env_intensity: env_maps.intensity.max(0.0),
                env_rotation_rad: env_maps.rotation_rad(),
                light_dir: decoded.light.direction,
                light_color: decoded.light.color,
                light_intensity: decoded.light.intensity,
            },
        };
        match baker.bake(&placement) {
            Ok(baked) => {
                scene.upload_reflection_probe_texture(&baked)?;
                scene.reflection_probe_cache_key = Some(bake_key);
                scene.reflection_probe_cached_grid = Some(placement.grid.clone());
            }
            Err(error) => {
                log::warn!(
                    target: "terrain.reflection_probes",
                    "Failed to bake reflection probes: {error}"
                );
                scene.reflection_probe_cache_key = None;
                scene.reflection_probe_cached_grid = None;
                scene.clear_reflection_probe_texture();
                scene.queue.write_buffer(
                    &scene.reflection_probe_grid_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&ReflectionProbeGridUniformsGpu::disabled()),
                );
                scene.reflection_probe_grid_uniform_bytes = 0;
                return Ok(());
            }
        }
    }

    let grid = match scene.reflection_probe_cached_grid.clone() {
        Some(grid) if scene.reflection_probe_count > 0 => grid,
        _ => {
            scene.queue.write_buffer(
                &scene.reflection_probe_grid_uniform_buffer,
                0,
                bytemuck::bytes_of(&ReflectionProbeGridUniformsGpu::disabled()),
            );
            scene.reflection_probe_grid_uniform_bytes = 0;
            return Ok(());
        }
    };
    let (height_min, height_max) = compute_height_bounds(heightfield, z_scale);
    let (blend_x, blend_y) =
        reflection_axis_blend_distances(grid.spacing, settings.fallback_blend_distance);
    let half_spacing_x = if grid.dims[0] > 1 {
        grid.spacing[0] * 0.5
    } else {
        terrain_span * 0.5
    };
    let half_spacing_y = if grid.dims[1] > 1 {
        grid.spacing[1] * 0.5
    } else {
        terrain_span * 0.5
    };
    let bounds_min_x = grid.origin[0] - half_spacing_x;
    let bounds_min_y = grid.origin[1] - half_spacing_y;
    let bounds_max_x =
        grid.origin[0] + grid.spacing[0] * (grid.dims[0].saturating_sub(1)) as f32 + half_spacing_x;
    let bounds_max_y =
        grid.origin[1] + grid.spacing[1] * (grid.dims[1].saturating_sub(1)) as f32 + half_spacing_y;
    let uniforms = ReflectionProbeGridUniformsGpu {
        grid_origin: [grid.origin[0], grid.origin[1], grid.height_offset, 1.0],
        grid_params: [
            grid.spacing[0],
            grid.spacing[1],
            grid.dims[0] as f32,
            grid.dims[1] as f32,
        ],
        blend_params: [
            blend_x,
            blend_y,
            settings.strength.clamp(0.0, 1.0),
            scene.reflection_probe_count as f32,
        ],
        scene_bounds_min: [
            bounds_min_x,
            bounds_min_y,
            height_min,
            scene.reflection_probe_resolution as f32,
        ],
        scene_bounds_max: [
            bounds_max_x,
            bounds_max_y,
            height_max.max(height_min + 1e-3),
            scene.reflection_probe_mip_levels.max(1) as f32,
        ],
    };
    scene.queue.write_buffer(
        &scene.reflection_probe_grid_uniform_buffer,
        0,
        bytemuck::bytes_of(&uniforms),
    );
    scene.reflection_probe_grid_uniform_bytes =
        std::mem::size_of::<ReflectionProbeGridUniformsGpu>() as u64;
    Ok(())
}
