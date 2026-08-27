use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::*;
use crate::core::resource_tracker::{
    tracked_create_buffer_init, tracked_create_texture, TrackedBuffer, TrackedTexture,
};

pub(in crate::terrain::renderer) struct UploadedHeightInputs {
    pub(in crate::terrain::renderer) width: u32,
    pub(in crate::terrain::renderer) height: u32,
    pub(in crate::terrain::renderer) heightmap_data: Vec<f32>,
    pub(in crate::terrain::renderer) terrain_data_hash: u64,
    pub(in crate::terrain::renderer) heightmap_texture: Arc<TrackedTexture>,
    pub(in crate::terrain::renderer) heightmap_view: wgpu::TextureView,
    pub(in crate::terrain::renderer) _water_mask_texture: Option<Arc<TrackedTexture>>,
    pub(in crate::terrain::renderer) water_mask_view_uploaded: Option<wgpu::TextureView>,
}

pub(in crate::terrain::renderer) struct MaterialMapResources {
    _normal_texture: TrackedTexture,
    normal_view: wgpu::TextureView,
    _roughness_texture: TrackedTexture,
    roughness_view: wgpu::TextureView,
    _mask_texture: TrackedTexture,
    mask_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

pub(in crate::terrain::renderer) struct PreparedMaterials {
    pub(in crate::terrain::renderer) gpu_materials:
        Arc<crate::render::material_set::GpuMaterialSet>,
    pub(in crate::terrain::renderer) shading_buffer: TrackedBuffer,
    pub(in crate::terrain::renderer) shading_uniforms: Vec<f32>,
    pub(in crate::terrain::renderer) overlay_buffer: TrackedBuffer,
    pub(in crate::terrain::renderer) overlay_binding: OverlayBinding,
    pub(in crate::terrain::renderer) fallback_colormap_view: Option<wgpu::TextureView>,
    pub(in crate::terrain::renderer) material_maps: MaterialMapResources,
    pub(in crate::terrain::renderer) terrain_trace_albedo: [f32; 3],
}

impl PreparedMaterials {
    pub(in crate::terrain::renderer) fn material_view(&self) -> &wgpu::TextureView {
        &self.gpu_materials.view
    }

    pub(in crate::terrain::renderer) fn material_sampler(&self) -> &wgpu::Sampler {
        &self.gpu_materials.sampler
    }

    pub(in crate::terrain::renderer) fn colormap_view(&self) -> &wgpu::TextureView {
        if let Some(lut) = self.overlay_binding.lut.as_ref() {
            &lut.view
        } else {
            self.fallback_colormap_view.as_ref().unwrap()
        }
    }

    pub(in crate::terrain::renderer) fn colormap_sampler(&self) -> &wgpu::Sampler {
        if let Some(lut) = self.overlay_binding.lut.as_ref() {
            &lut.sampler
        } else {
            &self.gpu_materials.sampler
        }
    }

    pub(in crate::terrain::renderer) fn material_normal_view(&self) -> &wgpu::TextureView {
        &self.material_maps.normal_view
    }

    pub(in crate::terrain::renderer) fn material_roughness_view(&self) -> &wgpu::TextureView {
        &self.material_maps.roughness_view
    }

    pub(in crate::terrain::renderer) fn material_mask_view(&self) -> &wgpu::TextureView {
        &self.material_maps.mask_view
    }

    pub(in crate::terrain::renderer) fn material_map_sampler(&self) -> &wgpu::Sampler {
        &self.material_maps.sampler
    }
}

impl TerrainScene {
    pub(in crate::terrain::renderer) fn upload_height_inputs(
        &mut self,
        heightmap: PyReadonlyArray2<f32>,
        water_mask: Option<PyReadonlyArray2<f32>>,
        terrain_data_revision: Option<u64>,
    ) -> Result<UploadedHeightInputs> {
        let heightmap_array = heightmap.as_array();
        let (height, width) = (heightmap_array.shape()[0], heightmap_array.shape()[1]);
        if width == 0 || height == 0 {
            return Err(anyhow!("Heightmap dimensions must be > 0"));
        }

        let mut heightmap_data = Vec::with_capacity(width * height);
        let terrain_data_hash = if let Some(revision) = terrain_data_revision {
            // Caller-managed revisions let probe preparation skip per-sample hashing.
            for value in heightmap_array.iter() {
                heightmap_data.push(*value);
            }
            let mut hasher = DefaultHasher::new();
            (width as u32, height as u32).hash(&mut hasher);
            revision.hash(&mut hasher);
            hasher.finish()
        } else {
            let mut hasher = DefaultHasher::new();
            (width as u32, height as u32).hash(&mut hasher);
            for value in heightmap_array.iter() {
                let height = *value;
                heightmap_data.push(height);
                height.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        };
        let heightmap_texture = Arc::new(self.upload_heightmap_texture(
            width as u32,
            height as u32,
            &heightmap_data,
        )?);
        let heightmap_view = heightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.compute_coarse_ao_from_heightmap(width as u32, height as u32, &heightmap_data)?;

        let (water_mask_texture, water_mask_view_uploaded) = if let Some(mask) = water_mask {
            let mask_array = mask.as_array();
            if mask_array.shape() == heightmap_array.shape() {
                let mut mask_bytes = Vec::with_capacity(width * height);
                let mut water_count = 0usize;
                let mut has_gradient = false;
                for value in mask_array.iter() {
                    let v = value.clamp(0.0, 1.0);
                    if v > 0.0 {
                        water_count += 1;
                        if v > 0.01 && v < 0.99 {
                            has_gradient = true;
                        }
                    }
                    mask_bytes.push((v * 255.0) as u8);
                }
                log::info!(
                    target: "terrain.water",
                    "Uploading water mask: {}x{}, {} water pixels ({:.2}%), distance_encoded={}",
                    width, height, water_count,
                    100.0 * water_count as f64 / (width * height) as f64,
                    has_gradient
                );
                let tex =
                    self.upload_water_mask_texture(width as u32, height as u32, &mask_bytes)?;
                let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                (Some(Arc::new(tex)), Some(view))
            } else {
                log::warn!(
                    target: "terrain.water",
                    "Water mask shape {:?} does not match heightmap shape {:?}; using fallback",
                    mask_array.shape(),
                    heightmap_array.shape()
                );
                (None, None)
            }
        } else {
            (None, None)
        };

        Ok(UploadedHeightInputs {
            width: width as u32,
            height: height as u32,
            heightmap_data,
            terrain_data_hash,
            heightmap_texture,
            heightmap_view,
            _water_mask_texture: water_mask_texture,
            water_mask_view_uploaded,
        })
    }

    pub(in crate::terrain::renderer) fn prepare_material_context_with_mode(
        &self,
        material_set: &crate::render::material_set::MaterialSet,
        params: &crate::terrain::render_params::TerrainRenderParams,
        decoded: &crate::terrain::render_params::DecodedTerrainSettings,
        offline_hdr_output: bool,
    ) -> Result<PreparedMaterials> {
        let gpu_materials = material_set
            .gpu(self.device.as_ref(), self.queue.as_ref())
            .map_err(|err| {
                PyRuntimeError::new_err(format!("Failed to prepare material textures: {err:#}"))
            })?;
        let material_maps = self.prepare_material_map_resources(&decoded.materials)?;

        let shading_uniforms =
            self.build_shading_uniforms(material_set, gpu_materials.as_ref(), params, decoded)?;
        let shading_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.shading_buffer"),
                contents: bytemuck::cast_slice(&shading_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let overlay_binding = self.extract_overlay_binding(params, offline_hdr_output);
        self.log_color_debug(params, &overlay_binding);

        let overlay_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.overlay_buffer"),
                contents: bytemuck::bytes_of(&overlay_binding.uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        let fallback_colormap_view = if overlay_binding.lut.is_none() {
            Some(
                gpu_materials
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("terrain.fallback.colormap.view"),
                        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    }),
            )
        } else {
            None
        };

        Ok(PreparedMaterials {
            gpu_materials,
            shading_buffer,
            shading_uniforms,
            overlay_buffer,
            overlay_binding,
            fallback_colormap_view,
            material_maps,
            terrain_trace_albedo: material_set
                .materials()
                .first()
                .map_or([1.0; 3], |material| {
                    [
                        material.base_color[0],
                        material.base_color[1],
                        material.base_color[2],
                    ]
                }),
        })
    }

    fn prepare_material_map_resources(
        &self,
        materials: &crate::terrain::render_params::MaterialLayerSettingsNative,
    ) -> Result<MaterialMapResources> {
        let (normal_texture, normal_view) = self.upload_material_map_texture(
            materials.normal_path.as_deref(),
            [128, 128, 255, 255],
            "terrain.material_maps.normal",
        )?;
        let (roughness_texture, roughness_view) = self.upload_material_map_texture(
            materials.roughness_path.as_deref(),
            [255, 255, 255, 255],
            "terrain.material_maps.roughness",
        )?;
        let (mask_texture, mask_view) = self.upload_material_map_texture(
            materials.mask_path.as_deref(),
            [255, 255, 255, 255],
            "terrain.material_maps.mask",
        )?;
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.material_maps.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Ok(MaterialMapResources {
            _normal_texture: normal_texture,
            normal_view,
            _roughness_texture: roughness_texture,
            roughness_view,
            _mask_texture: mask_texture,
            mask_view,
            sampler,
        })
    }

    fn upload_material_map_texture(
        &self,
        path: Option<&str>,
        fallback_rgba: [u8; 4],
        label: &'static str,
    ) -> Result<(TrackedTexture, wgpu::TextureView)> {
        let (width, height, pixels) = if let Some(path) = path {
            let image = image::open(path)
                .map_err(|err| anyhow!("Failed to load terrain material map '{}': {err}", path))?;
            let rgba = image.to_rgba8();
            let (width, height) = rgba.dimensions();
            if width == 0 || height == 0 {
                return Err(anyhow!(
                    "Terrain material map '{}' has zero dimensions",
                    path
                ));
            }
            (width, height, rgba.into_raw())
        } else {
            (1, 1, fallback_rgba.to_vec())
        };

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
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
        )?;
        let (padded, padded_bpr) = pad_rgba8_rows(width, height, &pixels);
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        Ok((texture, view))
    }

    pub(in crate::terrain::renderer) fn prepare_ibl_bind_group(
        &self,
        env_maps: &crate::lighting::ibl_wrapper::IBL,
    ) -> Result<wgpu::BindGroup> {
        let ibl_resources = env_maps.ensure_gpu_resources(&self.device, &self.queue)?;
        let (sin_theta, cos_theta) = env_maps.rotation_rad().sin_cos();
        let ibl_uniforms = IblUniforms {
            intensity: env_maps.intensity.max(0.0),
            sin_theta,
            cos_theta,
            specular_mip_count: ibl_resources.specular_mip_count.max(1) as f32,
        };
        let ibl_uniform_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.ibl_uniform_buffer"),
                contents: bytemuck::bytes_of(&ibl_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        )?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_pbr_pom.ibl_bind_group"),
            layout: &self.ibl_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        ibl_resources.specular_view.as_ref(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        ibl_resources.irradiance_view.as_ref(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(ibl_resources.sampler.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(ibl_resources.brdf_view.as_ref()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ibl_uniform_buffer.as_entire_binding(),
                },
            ],
        }))
    }
}

fn pad_rgba8_rows(width: u32, height: u32, pixels: &[u8]) -> (Vec<u8>, u32) {
    let row_bytes = width as usize * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_row_bytes = ((row_bytes + align - 1) / align) * align;
    if padded_row_bytes == row_bytes {
        return (pixels.to_vec(), row_bytes as u32);
    }
    let mut padded = vec![0u8; padded_row_bytes * height as usize];
    for row in 0..height as usize {
        let src = row * row_bytes;
        let dst = row * padded_row_bytes;
        padded[dst..dst + row_bytes].copy_from_slice(&pixels[src..src + row_bytes]);
    }
    (padded, padded_row_bytes as u32)
}
