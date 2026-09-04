use super::*;
use crate::core::resource_tracker::{tracked_create_buffer_init, TrackedBuffer};

impl TerrainScene {
    pub fn load_terrain_for_viewer(&mut self, dem_path: &str) -> Result<()> {
        use std::path::Path;

        let path = Path::new(dem_path);
        if !path.exists() {
            return Err(anyhow!("DEM file not found: {}", dem_path));
        }

        let (heightmap, width, height, domain) = self.load_geotiff_heightmap(dem_path)?;
        let heightmap_texture = self.upload_heightmap_texture(width, height, &heightmap)?;
        let heightmap_view = heightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let grid_resolution = 4096u32.min(width.min(height));
        let (vertex_buffer, index_buffer, index_count) =
            self.create_terrain_mesh(grid_resolution)?;

        let terrain_span = (width.max(height) as f32) * 1.0;
        let cam_radius = terrain_span * 0.8;

        self.viewer_heightmap = Some(ViewerTerrainData {
            heightmap,
            dimensions: (width, height),
            domain,
            heightmap_texture,
            heightmap_view,
            vertex_buffer,
            index_buffer,
            index_count,
            cam_radius,
            cam_phi_deg: 135.0,
            cam_theta_deg: 45.0,
            cam_fov_deg: 55.0,
            sun_azimuth_deg: 135.0,
            sun_elevation_deg: 35.0,
            sun_intensity: 3.0,
        });

        println!(
            "[terrain] Loaded {}x{} DEM, domain: {:.1}..{:.1}",
            width, height, domain.0, domain.1
        );

        Ok(())
    }

    fn load_geotiff_heightmap(&self, path: &str) -> Result<(Vec<f32>, u32, u32, (f32, f32))> {
        use std::fs::File;

        let file = File::open(path).map_err(|e| anyhow!("Failed to open DEM file: {}", e))?;
        let mut decoder = tiff::decoder::Decoder::new(file)
            .map_err(|e| anyhow!("Failed to decode TIFF: {}", e))?;

        let (width, height) = decoder
            .dimensions()
            .map_err(|e| anyhow!("Failed to get TIFF dimensions: {}", e))?;

        let image = decoder
            .read_image()
            .map_err(|e| anyhow!("Failed to read TIFF image: {}", e))?;

        let heightmap: Vec<f32> = match image {
            tiff::decoder::DecodingResult::F32(data) => data,
            tiff::decoder::DecodingResult::F64(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I16(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U16(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I32(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U32(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U8(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I8(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::U64(data) => data.iter().map(|&v| v as f32).collect(),
            tiff::decoder::DecodingResult::I64(data) => data.iter().map(|&v| v as f32).collect(),
        };

        let mut min_h = f32::MAX;
        let mut max_h = f32::MIN;
        for &h in &heightmap {
            if h.is_finite() {
                min_h = min_h.min(h);
                max_h = max_h.max(h);
            }
        }

        let heightmap: Vec<f32> = heightmap
            .iter()
            .map(|&h| if h.is_finite() { h } else { min_h })
            .collect();

        Ok((heightmap, width, height, (min_h, max_h)))
    }

    fn create_terrain_mesh(
        &self,
        grid_resolution: u32,
    ) -> Result<(TrackedBuffer, TrackedBuffer, u32)> {
        let mut vertices: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let grid = grid_resolution as usize;
        let inv_grid = 1.0 / (grid - 1) as f32;

        for y in 0..grid {
            for x in 0..grid {
                let u = x as f32 * inv_grid;
                let v = y as f32 * inv_grid;
                vertices.push(u - 0.5);
                vertices.push(v - 0.5);
                vertices.push(u);
                vertices.push(v);
            }
        }

        for y in 0..(grid - 1) {
            for x in 0..(grid - 1) {
                let i = (y * grid + x) as u32;
                indices.push(i);
                indices.push(i + grid as u32);
                indices.push(i + 1);
                indices.push(i + 1);
                indices.push(i + grid as u32);
                indices.push(i + grid as u32 + 1);
            }
        }

        let vertex_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.viewer.vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        )?;

        let index_buffer = tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.viewer.index_buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        )?;

        Ok((vertex_buffer, index_buffer, indices.len() as u32))
    }

    pub fn has_viewer_terrain(&self) -> bool {
        self.viewer_heightmap.is_some()
    }

    pub fn viewer_terrain(&self) -> Option<&ViewerTerrainData> {
        self.viewer_heightmap.as_ref()
    }

    pub fn viewer_terrain_mut(&mut self) -> Option<&mut ViewerTerrainData> {
        self.viewer_heightmap.as_mut()
    }

    pub fn set_viewer_camera(&mut self, phi_deg: f32, theta_deg: f32, radius: f32, fov_deg: f32) {
        if let Some(ref mut terrain) = self.viewer_heightmap {
            terrain.cam_phi_deg = phi_deg;
            terrain.cam_theta_deg = theta_deg;
            terrain.cam_radius = radius;
            terrain.cam_fov_deg = fov_deg;
        }
    }

    pub fn set_viewer_sun(&mut self, azimuth_deg: f32, elevation_deg: f32, intensity: f32) {
        if let Some(ref mut terrain) = self.viewer_heightmap {
            terrain.sun_azimuth_deg = azimuth_deg;
            terrain.sun_elevation_deg = elevation_deg;
            terrain.sun_intensity = intensity;
        }
    }

    pub fn render_viewer_terrain(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        _target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        background_initialized: bool,
    ) -> bool {
        let terrain = match &self.viewer_heightmap {
            Some(t) => t,
            None => return false,
        };

        let phi_rad = terrain.cam_phi_deg.to_radians();
        let theta_rad = terrain.cam_theta_deg.to_radians();
        let radius = terrain.cam_radius;
        let (tw, th) = terrain.dimensions;
        let terrain_center = glam::Vec3::new(0.0, (terrain.domain.0 + terrain.domain.1) * 0.5, 0.0);

        let eye_x = terrain_center.x + radius * theta_rad.sin() * phi_rad.cos();
        let eye_y = terrain_center.y + radius * theta_rad.cos();
        let eye_z = terrain_center.z + radius * theta_rad.sin() * phi_rad.sin();
        let eye = glam::Vec3::new(eye_x, eye_y, eye_z);

        let view_dir = (terrain_center - eye).normalize_or_zero();
        let up = if view_dir.dot(glam::Vec3::Y).abs() > 0.999 {
            -glam::Vec3::Z
        } else {
            glam::Vec3::Y
        };
        let view = glam::Mat4::look_at_rh(eye, terrain_center, up);
        let aspect = width as f32 / height as f32;
        let proj = glam::Mat4::perspective_rh(
            terrain.cam_fov_deg.to_radians(),
            aspect,
            1.0,
            radius * 10.0,
        );

        let sun_az_rad = terrain.sun_azimuth_deg.to_radians();
        let sun_el_rad = terrain.sun_elevation_deg.to_radians();
        let sun_dir = glam::Vec3::new(
            sun_el_rad.cos() * sun_az_rad.sin(),
            sun_el_rad.sin(),
            sun_el_rad.cos() * sun_az_rad.cos(),
        )
        .normalize();

        let h_range = terrain.domain.1 - terrain.domain.0;
        let spacing = (tw.max(th) as f32) / 256.0;
        let uniforms =
            crate::terrain::TerrainUniforms::new(view, proj, sun_dir, 1.0, spacing, h_range, 1.0);

        let uniform_buffer = match tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.viewer.uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        ) {
            Ok(buffer) => buffer,
            Err(err) => {
                log::error!(target: "terrain.viewer", "failed to allocate viewer uniform buffer: {err}");
                return false;
            }
        };

        // The viewer has no TerrainRenderParams object. Preserve its prior binding
        // bytes except for the new hue lane, which keeps the historical 0.08 default.
        let mut viewer_overlay_lanes = uniforms.to_debug_lanes_44();
        viewer_overlay_lanes[14] = 0.08;
        let viewer_overlay_buffer = match tracked_create_buffer_init(
            self.device.as_ref(),
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain.viewer.overlay_uniforms"),
                contents: bytemuck::cast_slice(&viewer_overlay_lanes),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        ) {
            Ok(buffer) => buffer,
            Err(err) => {
                log::error!(target: "terrain.viewer", "failed to allocate viewer overlay uniform buffer: {err}");
                return false;
            }
        };

        let pipeline_guard = self.pipeline.lock().unwrap();
        let pipeline = &pipeline_guard.pipeline;
        let lut_view = &self.height_curve_identity_view;
        let lut_sampler = &self.sampler_linear;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.viewer.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&terrain.heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: viewer_overlay_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&self.height_curve_identity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.height_curve_lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&self.height_ao_fallback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&self.ao_debug_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&self.sun_vis_fallback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Sampler(&self.csm_renderer.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&self.detail_normal_fallback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Sampler(&self.detail_normal_sampler),
                },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain.viewer.render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if background_initialized {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: terrain.sun_intensity.min(1.0) as f64 * 0.5,
                                g: terrain.sun_intensity.min(1.0) as f64 * 0.7,
                                b: terrain.sun_intensity.min(1.0) as f64 * 0.9,
                                a: 1.0,
                            })
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_bind_group(1, &self.noop_shadow.bind_group, &[]);
            pass.set_vertex_buffer(0, terrain.vertex_buffer.slice(..));
            pass.set_index_buffer(terrain.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..terrain.index_count, 0, 0..1);
        }

        true
    }
}
