use super::*;
use crate::core::resource_tracker::{tracked_create_buffer_init, tracked_create_texture};
use crate::viewer::event_loop::update_terrain_volumetrics_report;
use crate::viewer::ipc::TerrainVolumetricsReport;
use glam::{DVec2, DVec3};

const MAX_VIEWER_TERRAIN_GRID_RESOLUTION: u32 = 2048;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TerrainFootprint {
    pub world_origin_xz: DVec2,
    pub world_span_xz: DVec2,
    pub fallback: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct TerrainPathPreflight {
    raster_info: crate::gis::types::RasterInfo,
    footprint: TerrainFootprint,
}

impl TerrainPathPreflight {
    /// Default absolute terrain focus derived without decoding pixels or
    /// allocating renderer resources.
    pub(crate) fn default_focus(&self) -> DVec3 {
        DVec3::new(
            self.footprint.world_origin_xz.x + self.footprint.world_span_xz.x * 0.5,
            0.0,
            self.footprint.world_origin_xz.y + self.footprint.world_span_xz.y * 0.5,
        )
    }

    pub(crate) fn max_horizontal_span(&self) -> f64 {
        self.footprint.world_span_xz.max_element()
    }
}

/// Translate a north-up GeoTIFF affine into the viewer basis
/// `(map X, display height, -map Y)`. Invalid present metadata fails closed;
/// only genuinely absent transform metadata selects the legacy local fallback.
pub(super) fn terrain_footprint_from_info(
    info: &crate::gis::types::RasterInfo,
) -> crate::gis::error::GisResult<TerrainFootprint> {
    let Some((a, b, c, d, e, f)) = info.transform else {
        return Ok(TerrainFootprint {
            world_origin_xz: DVec2::ZERO,
            world_span_xz: DVec2::new(f64::from(info.width), f64::from(info.height)),
            fallback: true,
        });
    };
    let coefficients = [a, b, c, d, e, f];
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(crate::gis::error::GisError::InvalidTransform(format!(
            "unsupported_non_finite_transform: path={} coefficients={coefficients:?}",
            info.path
        )));
    }
    if b != 0.0 || d != 0.0 {
        return Err(crate::gis::error::GisError::InvalidTransform(format!(
            "unsupported_rotated_or_sheared_transform: path={} coefficients={coefficients:?}",
            info.path
        )));
    }
    if a <= 0.0 || e >= 0.0 {
        return Err(crate::gis::error::GisError::InvalidTransform(format!(
            "unsupported_axis_orientation: path={} coefficients={coefficients:?}",
            info.path
        )));
    }
    let span = DVec2::new(a * f64::from(info.width), -e * f64::from(info.height));
    if !span.is_finite() || span.x <= 0.0 || span.y <= 0.0 {
        return Err(crate::gis::error::GisError::InvalidTransform(format!(
            "invalid_zero_or_non_finite_span: path={} coefficients={coefficients:?}",
            info.path
        )));
    }
    Ok(TerrainFootprint {
        world_origin_xz: DVec2::new(c, -f),
        world_span_xz: span,
        fallback: false,
    })
}

impl ViewerTerrainScene {
    /// Validate raster metadata before an IPC load is enqueued. Malformed or
    /// unsupported affine metadata is therefore rejected before terrain-scene
    /// or GPU-resource allocation can begin.
    pub(crate) fn preflight_terrain_path(path: &str) -> Result<TerrainPathPreflight> {
        let raster_info = crate::gis::raster_info::read_raster_info(path)?;
        let footprint = terrain_footprint_from_info(&raster_info)?;
        Ok(TerrainPathPreflight {
            raster_info,
            footprint,
        })
    }

    pub fn load_terrain(&mut self, path: &str) -> Result<()> {
        use std::fs::File;

        // Metadata is a trust boundary. A read/decode failure is not equivalent
        // to an absent transform and must not silently become local placement.
        let preflight = Self::preflight_terrain_path(path)?;
        let raster_info = preflight.raster_info;
        let footprint = preflight.footprint;

        let file = File::open(path)?;
        let mut decoder = tiff::decoder::Decoder::new(file)?;
        let (width, height) = decoder.dimensions()?;
        let image = decoder.read_image()?;

        let mut heightmap: Vec<f32> = match image {
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

        // Filter out nodata values (common nodata: -9999, -32768, etc.)
        let (min_h, max_h) = heightmap
            .iter()
            .filter(|h| h.is_finite() && **h > -1000.0 && **h < 10000.0)
            .fold((f32::MAX, f32::MIN), |(min, max), &h| {
                (min.min(h), max.max(h))
            });

        // Debug: print height range to diagnose flat terrain issue
        let h_range = max_h - min_h;
        println!(
            "[terrain] Height range: {:.1} to {:.1} (range: {:.1})",
            min_h, max_h, h_range
        );

        // Replace NoData values with min_h to prevent edge artifacts
        // NoData values are typically: NaN, Inf, < -1000, > 10000
        for h in heightmap.iter_mut() {
            if !h.is_finite() || *h < -1000.0 || *h > 10000.0 {
                *h = min_h;
            }
        }

        let heightmap_texture = tracked_create_texture(
            &self.device,
            &wgpu::TextureDescriptor {
                label: Some("terrain_viewer.heightmap"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
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
                texture: &heightmap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&heightmap),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let heightmap_view = heightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let grid_res = terrain_grid_resolution(width, height);
        let (vertices, indices) = create_grid_mesh(grid_res);

        let vertex_buffer = tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain_viewer.vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            },
        )?;

        let index_buffer = tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain_viewer.index_buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            },
        )?;

        let terrain_width = width as f32;
        let terrain_span = crate::camera::Anchor::direction_to_render(DVec3::new(
            footprint.world_span_xz.x,
            0.0,
            footprint.world_span_xz.y,
        ))
        .abs()
        .max_element();
        let cam_radius = terrain_span * 1.5;

        let uniforms = TerrainUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            sun_dir: [0.5, 0.8, 0.3, 0.0],
            terrain_params: [min_h, max_h - min_h, terrain_width, 1.0],
            lighting: [1.0, 0.3, 0.5, -999999.0],
            background: [0.5, 0.7, 0.9, 0.0],
            water_color: [0.2, 0.4, 0.6, 0.0],
            render_origin_xz: [0.0, 0.0],
            render_span_xz: [width as f32, height as f32],
        };

        let uniform_buffer = tracked_create_buffer_init(
            &self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("terrain_viewer.uniform_buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        )?;

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain_viewer.sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_viewer.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&heightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let next_revision = self.terrain_revision_counter.wrapping_add(1);
        let mut terrain = ViewerTerrainData {
            heightmap,
            dimensions: (width, height),
            domain: (min_h, max_h),
            revision: next_revision,
            raster_info,
            world_origin_xz: footprint.world_origin_xz,
            world_span_xz: footprint.world_span_xz,
            georeferencing_fallback: footprint.fallback,
            _heightmap_texture: heightmap_texture,
            heightmap_view,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            uniform_buffer,
            bind_group,
            cam_radius,
            cam_phi_deg: 135.0,
            cam_theta_deg: 45.0,
            cam_fov_deg: 55.0,
            cam_target: DVec3::ZERO,
            sun_azimuth_deg: 135.0,
            sun_elevation_deg: 35.0,
            sun_intensity: 1.0,
            sun_source: "manual_angles".to_string(),
            ambient: 0.3,
            z_scale: 1.0,
            shadow_intensity: 0.5,
            background_color: [0.5, 0.7, 0.9],
            water_level: -999999.0,
            water_color: [0.2, 0.4, 0.6],
        };
        terrain.cam_target = terrain.default_camera_target();
        // Stage every dependent shadow binding against the candidate texture
        // before publication. During the swap, drop old bind groups while the
        // old tracked texture owner is still accounted, then replace terrain;
        // no live bind group can outlast the owner recorded in the ledger.
        let staged_shadow_bind_groups = self.build_shadow_bind_groups_for(&terrain);
        let staged_shadow_revision = if staged_shadow_bind_groups.is_empty() {
            0
        } else {
            next_revision
        };
        let old_shadow_bind_groups =
            std::mem::replace(&mut self.shadow_bind_groups, staged_shadow_bind_groups);
        drop(old_shadow_bind_groups);
        let old_terrain = self.terrain.replace(terrain);
        drop(old_terrain);
        self.terrain_revision_counter = next_revision;
        self.shadow_bind_group_revision = staged_shadow_revision;
        update_terrain_volumetrics_report(TerrainVolumetricsReport::default());

        println!(
            "[terrain] Loaded {}x{} DEM, domain: {:.1}..{:.1}, grid={}x{}",
            width, height, min_h, max_h, grid_res, grid_res
        );
        Ok(())
    }

    pub fn has_terrain(&self) -> bool {
        self.terrain.is_some()
    }

    pub fn set_camera(
        &mut self,
        phi: f32,
        theta: f32,
        radius: f32,
        fov: f32,
        target: Option<[f64; 3]>,
        anchor: &crate::camera::Anchor,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        if let Some(ref mut t) = self.terrain {
            let target = target.map(DVec3::from).unwrap_or(t.cam_target);
            t.validate_camera_state(anchor, phi, theta, radius, fov, target)?;
            t.set_camera_state(phi, theta, radius, fov, Some(target.to_array()));
        }
        Ok(())
    }

    pub fn set_sun(&mut self, azimuth: f32, elevation: f32, intensity: f32, source: &str) {
        if let Some(ref mut t) = self.terrain {
            t.sun_azimuth_deg = azimuth;
            t.sun_elevation_deg = elevation;
            t.sun_intensity = intensity;
            t.sun_source = source.to_string();
        }
    }

    pub fn get_params(&self) -> Option<String> {
        self.terrain.as_ref().map(|t| format!(
            "phi={:.1} theta={:.1} radius={:.0} fov={:.1} target=({:.1}, {:.1}, {:.1}) | sun_az={:.1} sun_el={:.1} intensity={:.2} ambient={:.2} source={} | zscale={:.2} shadow={:.2} | origin=({:.6},{:.6}) span=({:.6},{:.6}) fallback={} crs={:?} transform={:?}",
            t.cam_phi_deg, t.cam_theta_deg, t.cam_radius, t.cam_fov_deg,
            t.cam_target[0], t.cam_target[1], t.cam_target[2],
            t.sun_azimuth_deg, t.sun_elevation_deg, t.sun_intensity, t.ambient, t.sun_source,
            t.z_scale, t.shadow_intensity,
            t.world_origin_xz.x, t.world_origin_xz.y,
            t.world_span_xz.x, t.world_span_xz.y,
            t.georeferencing_fallback, t.raster_info.crs_authority, t.raster_info.transform
        ))
    }

    pub fn handle_mouse_drag(
        &mut self,
        anchor: &crate::camera::Anchor,
        dx: f32,
        dy: f32,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        if let Some(ref mut t) = self.terrain {
            let phi = t.cam_phi_deg + dx * 0.3;
            let theta = (t.cam_theta_deg - dy * 0.3).clamp(5.0, 85.0);
            t.validate_camera_state(
                anchor,
                phi,
                theta,
                t.cam_radius,
                t.cam_fov_deg,
                t.cam_target,
            )?;
            t.cam_phi_deg = phi;
            t.cam_theta_deg = theta;
        }
        Ok(())
    }

    pub fn handle_scroll(
        &mut self,
        anchor: &crate::camera::Anchor,
        delta: f32,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        if let Some(ref mut t) = self.terrain {
            let factor = (-delta * 0.05).exp();
            let radius = (t.cam_radius * factor).clamp(100.0, 50000.0);
            t.validate_camera_state(
                anchor,
                t.cam_phi_deg,
                t.cam_theta_deg,
                radius,
                t.cam_fov_deg,
                t.cam_target,
            )?;
            t.cam_radius = radius;
        }
        Ok(())
    }

    pub fn handle_keys(
        &mut self,
        anchor: &crate::camera::Anchor,
        forward: f32,
        right: f32,
        up: f32,
    ) -> Result<(), crate::viewer::camera_controller::CameraFrameError> {
        if let Some(ref mut t) = self.terrain {
            let phi = t.cam_phi_deg + right * 2.0;
            let theta = (t.cam_theta_deg - forward * 2.0).clamp(5.0, 85.0);
            let radius = (t.cam_radius * (1.0 - up * 0.02)).clamp(100.0, 50000.0);
            t.validate_camera_state(anchor, phi, theta, radius, t.cam_fov_deg, t.cam_target)?;
            t.cam_phi_deg = phi;
            t.cam_theta_deg = theta;
            t.cam_radius = radius;
        }
        Ok(())
    }
}

fn terrain_grid_resolution(width: u32, height: u32) -> u32 {
    width
        .min(height)
        .clamp(2, MAX_VIEWER_TERRAIN_GRID_RESOLUTION)
}

fn create_grid_mesh(resolution: u32) -> (Vec<f32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let inv = 1.0 / (resolution - 1) as f32;

    for y in 0..resolution {
        for x in 0..resolution {
            let u = x as f32 * inv;
            let v = y as f32 * inv;
            vertices.extend_from_slice(&[u, v, u, v]);
        }
    }

    for y in 0..(resolution - 1) {
        for x in 0..(resolution - 1) {
            let i = y * resolution + x;
            indices.extend_from_slice(&[
                i,
                i + resolution,
                i + 1,
                i + 1,
                i + resolution,
                i + resolution + 1,
            ]);
        }
    }

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::{terrain_footprint_from_info, terrain_grid_resolution};
    use crate::gis::types::RasterInfo;
    use std::path::PathBuf;

    #[test]
    fn viewer_terrain_grid_resolution_keeps_small_heightmaps_native() {
        assert_eq!(terrain_grid_resolution(512, 384), 384);
    }

    #[test]
    fn viewer_terrain_grid_resolution_caps_large_heightmaps() {
        assert_eq!(terrain_grid_resolution(11589, 10518), 2048);
        assert_eq!(terrain_grid_resolution(5794, 5259), 2048);
    }

    #[test]
    fn missing_transform_uses_exact_legacy_pixel_footprint() {
        let info = RasterInfo::new(PathBuf::from("local.tif"), 640, 480, 1);
        let footprint = terrain_footprint_from_info(&info).unwrap();
        assert_eq!(footprint.world_origin_xz, glam::DVec2::ZERO);
        assert_eq!(footprint.world_span_xz, glam::DVec2::new(640.0, 480.0));
        assert!(footprint.fallback);
    }

    #[test]
    fn north_up_transform_uses_exact_viewer_basis_and_keeps_unknown_crs() {
        let mut info = RasterInfo::new(PathBuf::from("projected.tif"), 20, 10, 1);
        info.transform = Some((30.0, 0.0, 500_000.25, 0.0, -20.0, 5_500_000.75));
        info.crs_authority = Some(std::collections::HashMap::from([
            ("authority".to_string(), "EXAMPLE".to_string()),
            ("code".to_string(), "987654".to_string()),
        ]));
        let footprint = terrain_footprint_from_info(&info).unwrap();
        assert_eq!(
            footprint.world_origin_xz,
            glam::DVec2::new(500_000.25, -5_500_000.75)
        );
        assert_eq!(footprint.world_span_xz, glam::DVec2::new(600.0, 200.0));
        assert!(!footprint.fallback);
        assert_eq!(info.crs_wkt, None);
    }

    #[test]
    fn present_invalid_transform_fails_closed() {
        for transform in [
            (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            (1.0, 0.25, 0.0, 0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (f64::NAN, 0.0, 0.0, 0.0, -1.0, 0.0),
        ] {
            let mut info = RasterInfo::new(PathBuf::from("invalid.tif"), 16, 16, 1);
            info.transform = Some(transform);
            assert!(terrain_footprint_from_info(&info).is_err());
        }
    }
}
